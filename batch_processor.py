"""
Batch PDF Processor
Processes all PDFs from multiple folders and stores in database

Usage:
    python batch_processor.py
"""

import os
import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from bengali_pdf_extractor import ImprovedJudgmentExtractor as JudgmentExtractor, SearchableJudgment
import logging

class JudgmentDatabase:
    """SQLite database for searchable judgments"""
    
    def __init__(self, db_path: str = "extracted_data/database.db"):
        self.db_path = db_path
        
        # Create directory if needed
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Connect and create tables
        self.conn = sqlite3.connect(db_path)
        self.create_tables()
    
    def create_tables(self):
        """Create database schema"""
        cursor = self.conn.cursor()
        
        # Main judgments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS judgments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT UNIQUE,
                case_number TEXT,
                case_year TEXT,
                case_type TEXT,
                full_case_id TEXT,
                
                petitioner_name TEXT,
                respondent_name TEXT,
                
                judgment_date TEXT,
                judgment_outcome TEXT,
                judgment_summary TEXT,
                
                court_name TEXT,
                rule_type TEXT,
                rule_outcome TEXT,
                
                language TEXT,
                page_count INTEGER,
                
                full_text TEXT,
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Advocates table (many-to-many)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS advocates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                judgment_id INTEGER,
                advocate_name TEXT,
                side TEXT,  -- 'petitioner' or 'respondent'
                FOREIGN KEY (judgment_id) REFERENCES judgments(id)
            )
        """)
        
        # Laws/Sections table (many-to-many)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS laws (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                judgment_id INTEGER,
                law_text TEXT,
                law_type TEXT,  -- 'section', 'article', 'act'
                FOREIGN KEY (judgment_id) REFERENCES judgments(id)
            )
        """)
        
        # Judges table (many-to-many)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS judges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                judgment_id INTEGER,
                judge_name TEXT,
                FOREIGN KEY (judgment_id) REFERENCES judgments(id)
            )
        """)
        
        # Case citations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS citations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                judgment_id INTEGER,
                cited_case TEXT,
                FOREIGN KEY (judgment_id) REFERENCES judgments(id)
            )
        """)
        
        # Create indexes for fast searching
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_case_number ON judgments(case_number)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_case_type ON judgments(case_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_petitioner ON judgments(petitioner_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_respondent ON judgments(respondent_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_judgment_date ON judgments(judgment_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_advocate_name ON advocates(advocate_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_law_text ON laws(law_text)")
        
        self.conn.commit()
    
    def insert_judgment(self, judgment: SearchableJudgment) -> int:
        """Insert judgment and return its ID"""
        cursor = self.conn.cursor()
        
        # Check if already exists
        cursor.execute("SELECT id FROM judgments WHERE file_name = ?", (judgment.file_name,))
        existing = cursor.fetchone()
        
        if existing:
            print(f"  ⚠️  Already in database, updating: {judgment.file_name}")
            judgment_id = existing[0]
            self.update_judgment(judgment_id, judgment)
            return judgment_id
        
        # Insert main judgment
        cursor.execute("""
            INSERT INTO judgments (
                file_name, case_number, case_year, case_type, full_case_id,
                petitioner_name, respondent_name,
                judgment_date, judgment_outcome, judgment_summary,
                court_name, rule_type, rule_outcome,
                language, page_count, full_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            judgment.file_name,
            judgment.case_number,
            judgment.case_year,
            judgment.case_type,
            judgment.full_case_id,
            judgment.petitioner_name,
            judgment.respondent_name,
            judgment.judgment_date,
            judgment.judgment_outcome,
            judgment.judgment_summary,
            judgment.court_name,
            judgment.rule_type,
            judgment.rule_outcome,
            judgment.language,
            judgment.page_count,
            judgment.full_text
        ))
        
        judgment_id = cursor.lastrowid
        
        # Insert advocates
        for advocate in judgment.petitioner_advocates:
            cursor.execute(
                "INSERT INTO advocates (judgment_id, advocate_name, side) VALUES (?, ?, ?)",
                (judgment_id, advocate, 'petitioner')
            )
        
        for advocate in judgment.respondent_advocates:
            cursor.execute(
                "INSERT INTO advocates (judgment_id, advocate_name, side) VALUES (?, ?, ?)",
                (judgment_id, advocate, 'respondent')
            )
        
        # Insert laws
        for section in judgment.sections_cited:
            cursor.execute(
                "INSERT INTO laws (judgment_id, law_text, law_type) VALUES (?, ?, ?)",
                (judgment_id, section, 'section')
            )
        
        for article in judgment.articles_cited:
            cursor.execute(
                "INSERT INTO laws (judgment_id, law_text, law_type) VALUES (?, ?, ?)",
                (judgment_id, article, 'article')
            )
        
        for act in judgment.acts_cited:
            cursor.execute(
                "INSERT INTO laws (judgment_id, law_text, law_type) VALUES (?, ?, ?)",
                (judgment_id, act, 'act')
            )
        
        # Insert judges
        for judge in judgment.judges:
            cursor.execute(
                "INSERT INTO judges (judgment_id, judge_name) VALUES (?, ?)",
                (judgment_id, judge)
            )
        
        # Insert citations
        for citation in judgment.cases_cited:
            cursor.execute(
                "INSERT INTO citations (judgment_id, cited_case) VALUES (?, ?)",
                (judgment_id, citation)
            )
        
        self.conn.commit()
        return judgment_id
    
    def update_judgment(self, judgment_id: int, judgment: SearchableJudgment):
        """Update existing judgment"""
        cursor = self.conn.cursor()
        
        # Delete old related data
        cursor.execute("DELETE FROM advocates WHERE judgment_id = ?", (judgment_id,))
        cursor.execute("DELETE FROM laws WHERE judgment_id = ?", (judgment_id,))
        cursor.execute("DELETE FROM judges WHERE judgment_id = ?", (judgment_id,))
        cursor.execute("DELETE FROM citations WHERE judgment_id = ?", (judgment_id,))
        
        # Update main record
        cursor.execute("""
            UPDATE judgments SET
                case_number=?, case_year=?, case_type=?, full_case_id=?,
                petitioner_name=?, respondent_name=?,
                judgment_date=?, judgment_outcome=?, judgment_summary=?,
                court_name=?, rule_type=?, rule_outcome=?,
                language=?, page_count=?, full_text=?,
                updated_at=CURRENT_TIMESTAMP
            WHERE id=?
        """, (
            judgment.case_number, judgment.case_year, judgment.case_type, judgment.full_case_id,
            judgment.petitioner_name, judgment.respondent_name,
            judgment.judgment_date, judgment.judgment_outcome, judgment.judgment_summary,
            judgment.court_name, judgment.rule_type, judgment.rule_outcome,
            judgment.language, judgment.page_count, judgment.full_text,
            judgment_id
        ))
        
        # Re-insert related data (same as insert_judgment)
        for advocate in judgment.petitioner_advocates:
            cursor.execute(
                "INSERT INTO advocates (judgment_id, advocate_name, side) VALUES (?, ?, ?)",
                (judgment_id, advocate, 'petitioner')
            )
        
        for advocate in judgment.respondent_advocates:
            cursor.execute(
                "INSERT INTO advocates (judgment_id, advocate_name, side) VALUES (?, ?, ?)",
                (judgment_id, advocate, 'respondent')
            )
        
        for section in judgment.sections_cited:
            cursor.execute(
                "INSERT INTO laws (judgment_id, law_text, law_type) VALUES (?, ?, ?)",
                (judgment_id, section, 'section')
            )
        
        for article in judgment.articles_cited:
            cursor.execute(
                "INSERT INTO laws (judgment_id, law_text, law_type) VALUES (?, ?, ?)",
                (judgment_id, article, 'article')
            )
        
        for judge in judgment.judges:
            cursor.execute(
                "INSERT INTO judges (judgment_id, judge_name) VALUES (?, ?)",
                (judgment_id, judge)
            )
        
        self.conn.commit()
    
    def search(self, **filters) -> List[Dict]:
        """Search judgments with filters"""
        query = "SELECT DISTINCT j.* FROM judgments j"
        conditions = []
        params = []
        
        # Join tables as needed
        if filters.get('advocate'):
            query += " JOIN advocates a ON j.id = a.judgment_id"
            conditions.append("a.advocate_name LIKE ?")
            params.append(f"%{filters['advocate']}%")
        
        if filters.get('section') or filters.get('law'):
            query += " JOIN laws l ON j.id = l.judgment_id"
            law_term = filters.get('section') or filters.get('law')
            conditions.append("l.law_text LIKE ?")
            params.append(f"%{law_term}%")
        
        # Add WHERE conditions
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
        
        if filters.get('year') or filters.get('judgment_year'):
            year = filters.get('year') or filters.get('judgment_year')
            conditions.append("(j.case_year = ? OR j.judgment_date LIKE ?)")
            params.extend([year, f"%{year}%"])
        
        # Build final query
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        
        columns = [desc[0] for desc in cursor.description]
        results = []
        
        for row in cursor.fetchall():
            results.append(dict(zip(columns, row)))
        
        return results
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        cursor = self.conn.cursor()
        
        stats = {}
        
        cursor.execute("SELECT COUNT(*) FROM judgments")
        stats['total_judgments'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT case_type) FROM judgments WHERE case_type IS NOT NULL")
        stats['case_types'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT advocate_name) FROM advocates")
        stats['total_advocates'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM laws")
        stats['total_laws_cited'] = cursor.fetchone()[0]
        
        return stats
    
    def close(self):
        """Close database connection"""
        self.conn.close()


class BatchProcessor:
    """Process multiple PDF folders"""
    
    def __init__(self, output_dir: str = "extracted_data"):
        self.output_dir = output_dir
        self.json_dir = os.path.join(output_dir, "json")
        self.log_dir = "logs"
        
        # Create directories
        os.makedirs(self.json_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Setup logging
        log_file = os.path.join(self.log_dir, f"extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize extractor and database
        self.extractor = JudgmentExtractor()
        self.database = JudgmentDatabase()
    
    def find_all_pdfs(self, folders: List[str]) -> List[str]:
        """Find all PDF files in given folders"""
        pdf_files = []
        
        for folder in folders:
            if not os.path.exists(folder):
                self.logger.warning(f"Folder not found: {folder}")
                continue
            
            for root, dirs, files in os.walk(folder):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        pdf_path = os.path.join(root, file)
                        pdf_files.append(pdf_path)
        
        return pdf_files
    
    def process_pdf(self, pdf_path: str) -> bool:
        """Process single PDF"""
        try:
            self.logger.info(f"Processing: {pdf_path}")
            
            # Extract judgment
            judgment = self.extractor.extract(pdf_path)
            
            # Save to JSON
            json_filename = Path(pdf_path).stem + ".json"
            json_path = os.path.join(self.json_dir, json_filename)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                f.write(self.extractor.to_json(judgment))
            
            # Save to database
            judgment_id = self.database.insert_judgment(judgment)
            
            self.logger.info(f"✓ Saved (ID: {judgment_id}): {judgment.full_case_id or pdf_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ Failed {pdf_path}: {str(e)}")
            return False
    
    def process_all(self, folders: List[str]):
        """Process all PDFs from folders"""
        self.logger.info("="*70)
        self.logger.info("BATCH PROCESSING STARTED")
        self.logger.info("="*70)
        
        # Find all PDFs
        pdf_files = self.find_all_pdfs(folders)
        total = len(pdf_files)
        
        self.logger.info(f"\nFound {total} PDF files")
        self.logger.info(f"Folders: {folders}")
        self.logger.info(f"Output: {self.output_dir}")
        self.logger.info("")
        
        if total == 0:
            self.logger.warning("No PDF files found!")
            return
        
        # Process each PDF
        success_count = 0
        failed_files = []
        
        for i, pdf_path in enumerate(pdf_files, 1):
            print(f"\n[{i}/{total}] ", end="")
            
            if self.process_pdf(pdf_path):
                success_count += 1
            else:
                failed_files.append(pdf_path)
        
        # Final report
        self.logger.info("\n" + "="*70)
        self.logger.info("BATCH PROCESSING COMPLETE")
        self.logger.info("="*70)
        self.logger.info(f"\nTotal PDFs: {total}")
        self.logger.info(f"Successful: {success_count}")
        self.logger.info(f"Failed: {len(failed_files)}")
        
        if failed_files:
            self.logger.info("\nFailed files:")
            for file in failed_files:
                self.logger.info(f"  • {file}")
        
        # Database statistics
        stats = self.database.get_stats()
        self.logger.info("\n" + "─"*70)
        self.logger.info("DATABASE STATISTICS:")
        self.logger.info(f"  • Total Judgments: {stats['total_judgments']}")
        self.logger.info(f"  • Case Types: {stats['case_types']}")
        self.logger.info(f"  • Total Advocates: {stats['total_advocates']}")
        self.logger.info(f"  • Laws Cited: {stats['total_laws_cited']}")
        
        self.logger.info("\n✅ Data saved to:")
        self.logger.info(f"  • JSON files: {self.json_dir}")
        self.logger.info(f"  • Database: {self.database.db_path}")
        self.logger.info(f"  • Logs: {self.log_dir}")
        
        self.database.close()


def main():
    """Main execution"""
    print("\n" + "█"*70)
    print("  BATCH PDF PROCESSOR - Legal Judgment Extractor")
    print("█"*70 + "\n")
    
    # === CONFIGURATION ===
    # Add your folder names here
    PDF_FOLDERS = [
        "ad_judgments",  # Replace with your actual folder names
        "sc_judgments"
    ]
    
    OUTPUT_DIR = "extracted_data"
    
    # === PROCESS ===
    processor = BatchProcessor(output_dir=OUTPUT_DIR)
    processor.process_all(PDF_FOLDERS)
    
    print("\n" + "─"*70)
    print("Next steps:")
    print("  1. Check extracted_data/json/ for JSON files")
    print("  2. Check extracted_data/database.db for searchable database")
    print("  3. Check logs/ for processing logs")
    print("  4. Use the search interface to query judgments")
    print("─"*70 + "\n")


if __name__ == "__main__":
    main()
