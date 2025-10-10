"""
Search Database Interface
Query extracted judgments using keywords

Usage:
    python search_database.py
"""

import sqlite3
from typing import List, Dict
import sys

class JudgmentSearch:
    """Search interface for judgment database"""
    
    def __init__(self, db_path: str = "extracted_data/database.db"):
        try:
            self.conn = sqlite3.connect(db_path)
            self.conn.row_factory = sqlite3.Row
        except sqlite3.Error as e:
            print(f"❌ Error: Could not connect to database at '{db_path}'")
            print(f"   Make sure you've run batch_processor.py first!")
            sys.exit(1)
    
    def search(self, **filters) -> List[Dict]:
        """
        Search judgments with multiple filters
        
        Available filters:
        - case_number: str
        - case_type: str
        - petitioner: str
        - respondent: str
        - advocate: str
        - section: str
        - rule_outcome: str
        - year: str
        """
        query = "SELECT DISTINCT j.* FROM judgments j"
        joins = []
        conditions = []
        params = []
        
        # Handle advocate search
        if filters.get('advocate'):
            joins.append("JOIN advocates a ON j.id = a.judgment_id")
            conditions.append("a.advocate_name LIKE ?")
            params.append(f"%{filters['advocate']}%")
        
        # Handle law/section search
        if filters.get('section'):
            joins.append("JOIN laws l ON j.id = l.judgment_id")
            conditions.append("l.law_text LIKE ?")
            params.append(f"%{filters['section']}%")
        
        # Add joins
        if joins:
            query += " " + " ".join(joins)
        
        # Add conditions
        if filters.get('case_number'):
            conditions.append("j.case_number = ?")
            params.append(filters['case_number'])
        
        if filters.get('case_type'):
            conditions.append("j.case_type LIKE ?")
            params.append(f"%{filters['case_type']}%")
        
        if filters.get('petitioner'):
            conditions.append("j.petitioner_name LIKE ?")
            params.append(f"%{filters['petitioner']}%")
        
        if filters.get('respondent'):
            conditions.append("j.respondent_name LIKE ?")
            params.append(f"%{filters['respondent']}%")
        
        if filters.get('rule_outcome'):
            conditions.append("j.rule_outcome LIKE ?")
            params.append(f"%{filters['rule_outcome']}%")
        
        if filters.get('year'):
            conditions.append("(j.case_year = ? OR j.judgment_date LIKE ?)")
            params.extend([filters['year'], f"%{filters['year']}%"])
        
        # Build WHERE clause
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        # Order by date
        query += " ORDER BY j.judgment_date DESC"
        
        # Execute query
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        
        # Convert to dict list
        results = []
        for row in cursor.fetchall():
            result = dict(row)
            
            # Fetch related data
            result['advocates'] = self._get_advocates(result['id'])
            result['laws'] = self._get_laws(result['id'])
            result['judges'] = self._get_judges(result['id'])
            
            results.append(result)
        
        return results
    
    def _get_advocates(self, judgment_id: int) -> Dict[str, List[str]]:
        """Get advocates for a judgment"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT advocate_name, side FROM advocates WHERE judgment_id = ?",
            (judgment_id,)
        )
        
        advocates = {'petitioner': [], 'respondent': []}
        for row in cursor.fetchall():
            advocates[row['side']].append(row['advocate_name'])
        
        return advocates
    
    def _get_laws(self, judgment_id: int) -> List[str]:
        """Get laws cited in a judgment"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT law_text FROM laws WHERE judgment_id = ?",
            (judgment_id,)
        )
        
        return [row['law_text'] for row in cursor.fetchall()]
    
    def _get_judges(self, judgment_id: int) -> List[str]:
        """Get judges for a judgment"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT judge_name FROM judges WHERE judgment_id = ?",
            (judgment_id,)
        )
        
        return [row['judge_name'] for row in cursor.fetchall()]
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        cursor = self.conn.cursor()
        
        stats = {}
        
        cursor.execute("SELECT COUNT(*) as count FROM judgments")
        stats['total_judgments'] = cursor.fetchone()['count']
        
        cursor.execute("SELECT COUNT(DISTINCT case_type) as count FROM judgments WHERE case_type IS NOT NULL")
        stats['case_types'] = cursor.fetchone()['count']
        
        cursor.execute("SELECT COUNT(DISTINCT advocate_name) as count FROM advocates")
        stats['total_advocates'] = cursor.fetchone()['count']
        
        cursor.execute("SELECT COUNT(*) as count FROM laws")
        stats['total_laws'] = cursor.fetchone()['count']
        
        cursor.execute("SELECT case_type, COUNT(*) as count FROM judgments WHERE case_type IS NOT NULL GROUP BY case_type ORDER BY count DESC")
        stats['case_type_breakdown'] = [(row['case_type'], row['count']) for row in cursor.fetchall()]
        
        return stats
    
    def display_result(self, result: Dict, index: int = 1):
        """Display a single search result"""
        print(f"\n{'─'*70}")
        print(f"[{index}] {result['full_case_id'] or result['file_name']}")
        print(f"{'─'*70}")
        
        # Basic info
        if result['judgment_date']:
            print(f"📅 Judgment Date: {result['judgment_date']}")
        
        if result['court_name']:
            print(f"🏛️  Court: {result['court_name']}")
        
        if result['judgment_outcome']:
            print(f"⚖️  Outcome: {result['judgment_outcome'].upper()}")
        
        if result['rule_outcome']:
            print(f"📜 Rule: {result['rule_outcome']}")
        
        # Parties
        if result['petitioner_name'] or result['respondent_name']:
            print(f"\n👥 Parties:")
            if result['petitioner_name']:
                print(f"   • Petitioner: {result['petitioner_name']}")
            if result['respondent_name']:
                print(f"   • Respondent: {result['respondent_name']}")
        
        # Advocates
        advocates = result['advocates']
        if advocates['petitioner'] or advocates['respondent']:
            print(f"\n👔 Advocates:")
            if advocates['petitioner']:
                print(f"   • For Petitioner: {', '.join(advocates['petitioner'])}")
            if advocates['respondent']:
                print(f"   • For Respondent: {', '.join(advocates['respondent'])}")
        
        # Laws
        if result['laws']:
            print(f"\n⚖️  Laws Cited:")
            for law in result['laws'][:5]:  # Show first 5
                print(f"   • {law}")
            if len(result['laws']) > 5:
                print(f"   ... and {len(result['laws']) - 5} more")
        
        # Judges
        if result['judges']:
            print(f"\n👨‍⚖️ Judges: {', '.join(result['judges'])}")
        
        # Summary
        if result['judgment_summary']:
            print(f"\n📝 Summary:")
            print(f"   {result['judgment_summary']}")
    
    def close(self):
        """Close database connection"""
        self.conn.close()


def interactive_search():
    """Interactive search interface"""
    print("\n" + "█"*70)
    print("  JUDGMENT SEARCH INTERFACE")
    print("█"*70 + "\n")
    
    searcher = JudgmentSearch()
    
    # Show statistics
    print("📊 Database Statistics:")
    stats = searcher.get_stats()
    print(f"  • Total Judgments: {stats['total_judgments']}")
    print(f"  • Total Advocates: {stats['total_advocates']}")
    print(f"  • Laws Cited: {stats['total_laws']}")
    
    if stats['case_type_breakdown']:
        print(f"\n  Case Types:")
        for case_type, count in stats['case_type_breakdown']:
            print(f"    • {case_type}: {count}")
    
    print("\n" + "="*70)
    print("SEARCH OPTIONS")
    print("="*70)
    print("\nEnter your search criteria (press Enter to skip):\n")
    
    # Get search filters from user
    filters = {}
    
    case_number = input("🔢 Case Number: ").strip()
    if case_number:
        filters['case_number'] = case_number
    
    case_type = input("📁 Case Type (e.g., Writ Petition): ").strip()
    if case_type:
        filters['case_type'] = case_type
    
    petitioner = input("👤 Petitioner Name: ").strip()
    if petitioner:
        filters['petitioner'] = petitioner
    
    respondent = input("👤 Respondent Name: ").strip()
    if respondent:
        filters['respondent'] = respondent
    
    advocate = input("👔 Advocate Name: ").strip()
    if advocate:
        filters['advocate'] = advocate
    
    section = input("⚖️  Law/Section (e.g., Article 102): ").strip()
    if section:
        filters['section'] = section
    
    rule_outcome = input("📜 Rule Outcome (e.g., discharged): ").strip()
    if rule_outcome:
        filters['rule_outcome'] = rule_outcome
    
    year = input("📅 Year: ").strip()
    if year:
        filters['year'] = year
    
    # Perform search
    if not filters:
        print("\n⚠️  No filters provided. Showing all judgments...\n")
    
    print("\n🔍 Searching...")
    results = searcher.search(**filters)
    
    # Display results
    print(f"\n{'='*70}")
    print(f"SEARCH RESULTS: {len(results)} judgment(s) found")
    print(f"{'='*70}")
    
    if not results:
        print("\n❌ No judgments match your criteria.")
        print("\nTips:")
        print("  • Try partial names (e.g., 'Kamal' instead of full name)")
        print("  • Check spelling")
        print("  • Use fewer filters")
    else:
        for i, result in enumerate(results, 1):
            searcher.display_result(result, i)
            
            # Pagination for many results
            if i % 5 == 0 and i < len(results):
                response = input(f"\n--- Showing {i}/{len(results)} results. Continue? (y/n): ").strip().lower()
                if response != 'y':
                    break
    
    searcher.close()
    
    print("\n" + "─"*70)
    print("Search complete!")
    print("─"*70 + "\n")


def quick_search_examples():
    """Show some quick search examples"""
    print("\n" + "█"*70)
    print("  QUICK SEARCH EXAMPLES")
    print("█"*70 + "\n")
    
    searcher = JudgmentSearch()
    
    examples = [
        ("Search by Advocate 'Kamal'", {'advocate': 'Kamal'}),
        ("Search by Article 102", {'section': 'Article 102'}),
        ("Search by Rule Outcome 'discharged'", {'rule_outcome': 'discharged'}),
        ("Writ Petitions from 2024", {'case_type': 'Writ', 'year': '2024'}),
    ]
    
    for title, filters in examples:
        print(f"\n{'='*70}")
        print(f"EXAMPLE: {title}")
        print(f"Filters: {filters}")
        print(f"{'='*70}")
        
        results = searcher.search(**filters)
        print(f"\nFound: {len(results)} result(s)")
        
        if results:
            # Show first result
            searcher.display_result(results[0], 1)
            
            if len(results) > 1:
                print(f"\n... and {len(results) - 1} more result(s)")
    
    searcher.close()


def main():
    """Main function"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--examples':
        quick_search_examples()
    else:
        interactive_search()


if __name__ == "__main__":
    main()
    