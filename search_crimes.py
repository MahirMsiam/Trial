import sqlite3
import re
import json
from typing import List, Dict
from crime_keywords import CRIME_KEYWORDS


class CrimeSearcher:
    def __init__(self, db_path: str = "extracted_data/database.db"):
        self.db_path = db_path

    def search_crimes_in_database(self, crime_types: List[str] = None) -> Dict[str, List[Dict]]:
        """
        Search for crime keywords in the existing database
        Returns cases containing the specified crime keywords
        """
        if crime_types is None:
            crime_types = list(CRIME_KEYWORDS.keys())

        # Connect to existing database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get all judgments with full_text
        cursor.execute("""
            SELECT id, case_number, case_year, case_type, full_case_id,
                   petitioner_name, respondent_name, judgment_date, 
                   judgment_outcome, judgment_summary, court_name, full_text
            FROM judgments 
            WHERE full_text IS NOT NULL
        """)

        all_cases = cursor.fetchall()
        conn.close()

        results = {crime: [] for crime in crime_types}

        # Search each case for crime keywords
        for case in all_cases:
            case_dict = {
                'id': case[0],
                'case_number': case[1],
                'case_year': case[2],
                'case_type': case[3],
                'full_case_id': case[4],
                'petitioner_name': case[5],
                'respondent_name': case[6],
                'judgment_date': case[7],
                'judgment_outcome': case[8],
                'judgment_summary': case[9],
                'court_name': case[10],
                'full_text_preview': case[11][:500] + "..." if case[11] else ""  # Preview only
            }

            case_text_lower = case[11].lower() if case[11] else ""

            # Search for each crime type
            for crime_type in crime_types:
                keywords = CRIME_KEYWORDS.get(crime_type, [])
                found = False

                for keyword in keywords:
                    # Use word boundaries for better matching
                    pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                    if re.search(pattern, case_text_lower):
                        results[crime_type].append(case_dict)
                        found = True
                        break

        return results

    def get_crime_statistics(self) -> Dict[str, int]:
        """Get statistics of crime cases found"""
        results = self.search_crimes_in_database()
        return {crime: len(cases) for crime, cases in results.items() if len(cases) > 0}

    def export_results(self, results: Dict[str, List[Dict]], output_file: str = "crime_search_results.json"):
        """Export results to JSON file"""
        # Remove full_text_preview to keep file size manageable
        clean_results = {}
        for crime, cases in results.items():
            if cases:  # Only include crimes with results
                clean_results[crime] = []
                for case in cases:
                    clean_case = case.copy()
                    clean_case.pop('full_text_preview', None)
                    clean_results[crime].append(clean_case)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, ensure_ascii=False, indent=2)

        print(f"✅ Results exported to: {output_file}")
        print(f"📊 Total crime categories found: {len(clean_results)}")
        for crime, cases in clean_results.items():
            print(f"   • {crime}: {len(cases)} cases")


def main():
    print("🔍 Legal Crime Keyword Search")
    print("=" * 50)
    print("Searching your existing database of 8,233+ judgments...")

    searcher = CrimeSearcher()

    # Get crime statistics
    print("\n📊 Analyzing crime patterns in your database...")
    stats = searcher.get_crime_statistics()

    if stats:
        print("\n📈 Crime Case Counts (found in your database):")
        print("-" * 50)
        for crime, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
            print(f"{crime:<25}: {count:>4} cases")
    else:
        print("\n❌ No crime-related cases found with current keywords.")
        print("💡 Try expanding keywords or checking different terms.")
        return

    # Ask user if they want detailed results
    print(f"\n{len(stats)} crime categories found with matching cases.")
    export = input("Export detailed results to JSON? (y/n): ").strip().lower()

    if export in ['y', 'yes']:
        filename = input("Enter filename (or press Enter for default): ").strip()
        if not filename:
            filename = "crime_search_results.json"
        if not filename.endswith('.json'):
            filename += '.json'

        # Get full results for export
        all_crimes = list(stats.keys())
        full_results = searcher.search_crimes_in_database(all_crimes)
        searcher.export_results(full_results, filename)
    else:
        print("✅ Search completed. Results available in memory.")


if __name__ == "__main__":
    main()