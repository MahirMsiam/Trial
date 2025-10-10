"""
Extraction Quality Analyzer
Analyzes the 8,233 judgments to show extraction quality
"""

import sqlite3
from typing import Dict, List

class QualityAnalyzer:
    """Analyze extraction quality across all judgments"""
    
    def __init__(self, db_path: str = "extracted_data/database.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
    
    def analyze_all(self):
        """Run complete quality analysis"""
        print("\n" + "█"*70)
        print("  EXTRACTION QUALITY ANALYSIS - 8,233 JUDGMENTS")
        print("█"*70 + "\n")
        
        # Overall statistics
        self.print_overall_stats()
        
        # Field completeness
        self.print_field_completeness()
        
        # Case type breakdown
        self.print_case_type_breakdown()
        
        # Top advocates
        self.print_top_advocates()
        
        # Most cited laws
        self.print_most_cited_laws()
        
        # Temporal distribution
        self.print_temporal_distribution()
        
        # Sample judgments
        self.print_sample_judgments()
    
    def print_overall_stats(self):
        """Print overall database statistics"""
        print("="*70)
        print("OVERALL STATISTICS")
        print("="*70 + "\n")
        
        cursor = self.conn.cursor()
        
        # Total judgments
        cursor.execute("SELECT COUNT(*) as count FROM judgments")
        total = cursor.fetchone()['count']
        print(f"📊 Total Judgments: {total:,}")
        
        # Total advocates
        cursor.execute("SELECT COUNT(DISTINCT advocate_name) as count FROM advocates")
        advocates = cursor.fetchone()['count']
        print(f"👔 Unique Advocates: {advocates:,}")
        
        # Total laws
        cursor.execute("SELECT COUNT(*) as count FROM laws")
        laws = cursor.fetchone()['count']
        print(f"⚖️  Total Law Citations: {laws:,}")
        
        # Unique laws
        cursor.execute("SELECT COUNT(DISTINCT law_text) as count FROM laws")
        unique_laws = cursor.fetchone()['count']
        print(f"⚖️  Unique Laws/Sections: {unique_laws:,}")
        
        # Case types
        cursor.execute("SELECT COUNT(DISTINCT case_type) as count FROM judgments WHERE case_type IS NOT NULL")
        case_types = cursor.fetchone()['count']
        print(f"📁 Different Case Types: {case_types}")
        
        # Date range
        cursor.execute("""
            SELECT MIN(judgment_date) as earliest, MAX(judgment_date) as latest 
            FROM judgments WHERE judgment_date IS NOT NULL
        """)
        dates = cursor.fetchone()
        if dates['earliest']:
            print(f"📅 Date Range: {dates['earliest']} to {dates['latest']}")
    
    def print_field_completeness(self):
        """Analyze how many judgments have each field"""
        print("\n\n" + "="*70)
        print("FIELD EXTRACTION COMPLETENESS")
        print("="*70 + "\n")
        
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT COUNT(*) as total FROM judgments")
        total = cursor.fetchone()['total']
        
        fields = [
            ('case_number', 'Case Number'),
            ('case_type', 'Case Type'),
            ('petitioner_name', 'Petitioner Name'),
            ('respondent_name', 'Respondent Name'),
            ('judgment_date', 'Judgment Date'),
            ('court_name', 'Court Name'),
            ('rule_type', 'Rule Type'),
            ('rule_outcome', 'Rule Outcome'),
            ('judgment_outcome', 'Judgment Outcome'),
        ]
        
        print(f"{'Field':<25} {'Count':<12} {'Percentage':<12} {'Status'}")
        print("─"*70)
        
        for field, name in fields:
            cursor.execute(f"SELECT COUNT(*) as count FROM judgments WHERE {field} IS NOT NULL AND {field} != ''")
            count = cursor.fetchone()['count']
            percentage = (count / total) * 100
            
            if percentage >= 80:
                status = "✅ Excellent"
            elif percentage >= 60:
                status = "⚠️  Good"
            elif percentage >= 40:
                status = "⚠️  Fair"
            else:
                status = "❌ Needs Work"
            
            print(f"{name:<25} {count:>6,}/{total:<6,} {percentage:>6.1f}%     {status}")
        
        # Check related tables
        print("\n" + "─"*70)
        
        # Advocates
        cursor.execute("SELECT COUNT(DISTINCT judgment_id) as count FROM advocates")
        count = cursor.fetchone()['count']
        percentage = (count / total) * 100
        status = "✅ Excellent" if percentage >= 80 else "⚠️  Good" if percentage >= 60 else "❌ Needs Work"
        print(f"{'Has Advocate Names':<25} {count:>6,}/{total:<6,} {percentage:>6.1f}%     {status}")
        
        # Laws
        cursor.execute("SELECT COUNT(DISTINCT judgment_id) as count FROM laws")
        count = cursor.fetchone()['count']
        percentage = (count / total) * 100
        status = "✅ Excellent" if percentage >= 80 else "⚠️  Good" if percentage >= 60 else "❌ Needs Work"
        print(f"{'Has Law Citations':<25} {count:>6,}/{total:<6,} {percentage:>6.1f}%     {status}")
        
        # Judges
        cursor.execute("SELECT COUNT(DISTINCT judgment_id) as count FROM judges")
        count = cursor.fetchone()['count']
        percentage = (count / total) * 100
        status = "✅ Excellent" if percentage >= 80 else "⚠️  Good" if percentage >= 60 else "❌ Needs Work"
        print(f"{'Has Judge Names':<25} {count:>6,}/{total:<6,} {percentage:>6.1f}%     {status}")
    
    def print_case_type_breakdown(self):
        """Show case type distribution"""
        print("\n\n" + "="*70)
        print("CASE TYPE BREAKDOWN")
        print("="*70 + "\n")
        
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT case_type, COUNT(*) as count 
            FROM judgments 
            WHERE case_type IS NOT NULL 
            GROUP BY case_type 
            ORDER BY count DESC 
            LIMIT 15
        """)
        
        results = cursor.fetchall()
        
        if results:
            print(f"{'Case Type':<35} {'Count':<10} {'Percentage'}")
            print("─"*70)
            
            cursor.execute("SELECT COUNT(*) as total FROM judgments WHERE case_type IS NOT NULL")
            total = cursor.fetchone()['total']
            
            for row in results:
                percentage = (row['count'] / total) * 100
                bar = "█" * int(percentage / 2)
                print(f"{row['case_type']:<35} {row['count']:>6,}    {percentage:>5.1f}% {bar}")
        else:
            print("⚠️  No case types found")
    
    def print_top_advocates(self):
        """Show most frequent advocates"""
        print("\n\n" + "="*70)
        print("TOP 20 ADVOCATES (Most Frequent)")
        print("="*70 + "\n")
        
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT advocate_name, COUNT(*) as count 
            FROM advocates 
            GROUP BY advocate_name 
            ORDER BY count DESC 
            LIMIT 20
        """)
        
        results = cursor.fetchall()
        
        if results:
            print(f"{'Rank':<6} {'Advocate Name':<40} {'Cases'}")
            print("─"*70)
            
            for i, row in enumerate(results, 1):
                print(f"{i:<6} {row['advocate_name']:<40} {row['count']:>6,}")
        else:
            print("⚠️  No advocates found")
    
    def print_most_cited_laws(self):
        """Show most cited laws"""
        print("\n\n" + "="*70)
        print("TOP 20 MOST CITED LAWS/SECTIONS")
        print("="*70 + "\n")
        
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT law_text, COUNT(*) as count 
            FROM laws 
            GROUP BY law_text 
            ORDER BY count DESC 
            LIMIT 20
        """)
        
        results = cursor.fetchall()
        
        if results:
            print(f"{'Rank':<6} {'Law/Section':<50} {'Citations'}")
            print("─"*70)
            
            for i, row in enumerate(results, 1):
                law_text = row['law_text'][:48] + "..." if len(row['law_text']) > 48 else row['law_text']
                print(f"{i:<6} {law_text:<50} {row['count']:>6,}")
        else:
            print("⚠️  No laws found")
    
    def print_temporal_distribution(self):
        """Show judgments by year"""
        print("\n\n" + "="*70)
        print("TEMPORAL DISTRIBUTION (Judgments by Year)")
        print("="*70 + "\n")
        
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT case_year, COUNT(*) as count 
            FROM judgments 
            WHERE case_year IS NOT NULL 
            GROUP BY case_year 
            ORDER BY case_year DESC 
            LIMIT 15
        """)
        
        results = cursor.fetchall()
        
        if results:
            print(f"{'Year':<10} {'Count':<10} {'Distribution'}")
            print("─"*70)
            
            max_count = max(row['count'] for row in results)
            
            for row in results:
                bar_length = int((row['count'] / max_count) * 40)
                bar = "█" * bar_length
                print(f"{row['case_year']:<10} {row['count']:>6,}    {bar}")
        else:
            print("⚠️  No year data found")
    
    def print_sample_judgments(self):
        """Show sample judgments"""
        print("\n\n" + "="*70)
        print("SAMPLE JUDGMENTS (Random 3)")
        print("="*70)
        
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM judgments 
            WHERE case_number IS NOT NULL 
            ORDER BY RANDOM() 
            LIMIT 3
        """)
        
        for i, row in enumerate(cursor.fetchall(), 1):
            print(f"\n{'─'*70}")
            print(f"[{i}] {row['full_case_id'] or row['file_name']}")
            print(f"{'─'*70}")
            
            if row['judgment_date']:
                print(f"📅 Date: {row['judgment_date']}")
            if row['court_name']:
                print(f"🏛️  Court: {row['court_name']}")
            if row['petitioner_name']:
                print(f"👤 Petitioner: {row['petitioner_name']}")
            if row['respondent_name']:
                print(f"👤 Respondent: {row['respondent_name']}")
            if row['judgment_outcome']:
                print(f"⚖️  Outcome: {row['judgment_outcome']}")
    
    def close(self):
        """Close connection"""
        self.conn.close()


def main():
    """Run quality analysis"""
    analyzer = QualityAnalyzer()
    analyzer.analyze_all()
    analyzer.close()
    
    print("\n\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)
    
    print("\n💡 KEY INSIGHTS:")
    print("  • Check field completeness percentages")
    print("  • Review case type distribution")
    print("  • See most active advocates")
    print("  • Identify commonly cited laws")
    
    print("\n🚀 NEXT STEPS:")
    print("  1. Test search: python search_database.py")
    print("  2. If quality is good (>80%), move to Step 2")
    print("  3. If quality needs work, share results for improvements")
    print("\n")


if __name__ == "__main__":
    main()
