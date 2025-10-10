"""
Database Setup Script
Uses your existing JudgmentDatabase class to create the SQLite database
"""
import os
from batch_processor import JudgmentDatabase

def main():
    print("🔧 Setting up SQLite database for legal judgments...")
    print("="*60)
    
    # Create the database
    db = JudgmentDatabase("extracted_data/database.db")
    
    # Get and display stats
    stats = db.get_stats()
    
    print("✅ Database created successfully!")
    print(f"📁 Location: extracted_data/database.db")
    print("\n📋 Database structure:")
    print("   • judgments (main case details)")
    print("   • advocates (petitioner/respondent lawyers)")
    print("   • laws (sections, articles, acts cited)")
    print("   • judges (bench members)")
    print("   • citations (referenced cases)")
    print("\n⚡ Search indexes created for fast queries")
    print("\n📊 Initial statistics:")
    print(f"   Total judgments: {stats['total_judgments']}")
    print(f"   Case types: {stats['case_types']}")
    print(f"   Total advocates: {stats['total_advocates']}")
    print(f"   Laws cited: {stats['total_laws_cited']}")
    
    db.close()
    print("\n✅ Setup complete! Ready to populate with your JSON data.")

if __name__ == "__main__":
    main()