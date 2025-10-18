"""
Database Optimization Utilities

Provides tools for analyzing query performance, creating indexes,
and optimizing the SQLite database.
"""

import sqlite3
import logging
import os
from typing import Dict, List, Optional
from contextlib import contextmanager
import threading

from config import DATABASE_PATH

logger = logging.getLogger(__name__)


def analyze_query_performance(db_path: str = DATABASE_PATH) -> Dict:
    """
    Analyze query performance using EXPLAIN QUERY PLAN
    
    Args:
        db_path: Path to SQLite database
        
    Returns:
        Dict with query plans and recommendations
    """
    logger.info(f"Analyzing query performance for database: {db_path}")
    
    if not os.path.exists(db_path):
        logger.error(f"Database not found: {db_path}")
        return {'error': 'Database not found'}
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    results = {
        'queries': [],
        'recommendations': []
    }
    
    # Define common queries to analyze
    test_queries = [
        ("Keyword search on full_text", "SELECT * FROM judgments WHERE full_text LIKE '%murder%' LIMIT 10"),
        ("Filter by case_type", "SELECT * FROM judgments WHERE case_type = 'Criminal Appeal' LIMIT 10"),
        ("Filter by year", "SELECT * FROM judgments WHERE case_year = 2023 LIMIT 10"),
        ("Filter by petitioner", "SELECT * FROM judgments WHERE petitioner_name LIKE '%State%' LIMIT 10"),
        ("Composite filter", "SELECT * FROM judgments WHERE case_type = 'Criminal Appeal' AND case_year = 2023 LIMIT 10"),
        ("JOIN with advocates", "SELECT j.*, a.advocate_name FROM judgments j JOIN advocates a ON j.id = a.judgment_id LIMIT 10"),
        ("JOIN with laws", "SELECT j.*, l.law_text FROM judgments j JOIN laws l ON j.id = l.judgment_id LIMIT 10"),
        ("Date range filter", "SELECT * FROM judgments WHERE judgment_date BETWEEN '2023-01-01' AND '2023-12-31' LIMIT 10")
    ]
    
    for query_name, query in test_queries:
        try:
            # Get query plan
            cursor.execute(f"EXPLAIN QUERY PLAN {query}")
            plan = cursor.fetchall()
            
            query_info = {
                'name': query_name,
                'query': query,
                'plan': [row[3] for row in plan],  # Detail column
                'uses_index': any('USING INDEX' in str(row[3]) for row in plan),
                'full_scan': any('SCAN' in str(row[3]) and 'USING INDEX' not in str(row[3]) for row in plan)
            }
            results['queries'].append(query_info)
            
            # Generate recommendations
            if query_info['full_scan']:
                results['recommendations'].append(
                    f"Consider adding index for: {query_name}"
                )
        except Exception as e:
            logger.error(f"Error analyzing query '{query_name}': {e}")
    
    conn.close()
    
    logger.info(f"Analyzed {len(results['queries'])} queries, {len(results['recommendations'])} recommendations")
    return results


def create_additional_indexes(db_path: str = DATABASE_PATH):
    """
    Create additional indexes for common query patterns
    
    Args:
        db_path: Path to SQLite database
    """
    logger.info(f"Creating additional indexes for database: {db_path}")
    
    if not os.path.exists(db_path):
        logger.error(f"Database not found: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    indexes = [
        # Composite indexes for common filter combinations
        ("idx_case_type_year", "CREATE INDEX IF NOT EXISTS idx_case_type_year ON judgments(case_type, case_year)"),
        ("idx_judgment_date_type", "CREATE INDEX IF NOT EXISTS idx_judgment_date_type ON judgments(judgment_date, case_type)"),
        ("idx_case_year_type", "CREATE INDEX IF NOT EXISTS idx_case_year_type ON judgments(case_year, case_type)"),
        
        # Indexes on foreign keys for faster JOINs
        ("idx_advocates_judgment", "CREATE INDEX IF NOT EXISTS idx_advocates_judgment ON advocates(judgment_id)"),
        ("idx_laws_judgment", "CREATE INDEX IF NOT EXISTS idx_laws_judgment ON laws(judgment_id)"),
        ("idx_judges_judgment", "CREATE INDEX IF NOT EXISTS idx_judges_judgment ON judges(judgment_id)"),
        
        # Additional text search indexes (using actual column names)
        ("idx_petitioner_respondent", "CREATE INDEX IF NOT EXISTS idx_petitioner_respondent ON judgments(petitioner_name, respondent_name)"),
    ]
    
    created_count = 0
    for index_name, create_sql in indexes:
        try:
            cursor.execute(create_sql)
            logger.info(f"Created index: {index_name}")
            created_count += 1
        except Exception as e:
            logger.error(f"Error creating index '{index_name}': {e}")
    
    # Consider FTS (Full-Text Search) virtual table
    try:
        # Check if FTS table already exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='judgments_fts'")
        if not cursor.fetchone():
            logger.info("Creating FTS5 virtual table for full_text search...")
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS judgments_fts 
                USING fts5(full_text, content=judgments, content_rowid=id)
            """)
            # Populate FTS table
            cursor.execute("INSERT INTO judgments_fts(judgments_fts) VALUES('rebuild')")
            logger.info("FTS5 virtual table created and populated")
            created_count += 1
        else:
            logger.info("FTS5 virtual table already exists")
    except Exception as e:
        logger.warning(f"Could not create FTS virtual table (may not be critical): {e}")
    
    conn.commit()
    conn.close()
    
    logger.info(f"Index creation completed: {created_count} indexes created/verified")


def optimize_database(db_path: str = DATABASE_PATH):
    """
    Optimize database by running VACUUM and ANALYZE
    
    Args:
        db_path: Path to SQLite database
    """
    logger.info(f"Optimizing database: {db_path}")
    
    if not os.path.exists(db_path):
        logger.error(f"Database not found: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Get database size before optimization
        size_before = os.path.getsize(db_path)
        
        # Run VACUUM to reclaim space and defragment
        logger.info("Running VACUUM...")
        cursor.execute("VACUUM")
        
        # Run ANALYZE to update query planner statistics
        logger.info("Running ANALYZE...")
        cursor.execute("ANALYZE")
        
        conn.commit()
        
        # Get database size after optimization
        size_after = os.path.getsize(db_path)
        space_saved = size_before - size_after
        
        logger.info(f"Optimization completed: {space_saved / 1024 / 1024:.2f} MB space saved")
        logger.info(f"Database size: {size_after / 1024 / 1024:.2f} MB")
    except Exception as e:
        logger.error(f"Error during optimization: {e}")
    finally:
        conn.close()


def get_database_stats(db_path: str = DATABASE_PATH) -> Dict:
    """
    Get database statistics
    
    Args:
        db_path: Path to SQLite database
        
    Returns:
        Dict with database statistics
    """
    logger.info(f"Getting database statistics for: {db_path}")
    
    if not os.path.exists(db_path):
        logger.error(f"Database not found: {db_path}")
        return {'error': 'Database not found'}
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    stats = {}
    
    try:
        # Database file size
        stats['database_size_mb'] = os.path.getsize(db_path) / 1024 / 1024
        
        # Detect if dbstat is available
        has_dbstat = True
        try:
            cursor.execute("SELECT 1 FROM dbstat LIMIT 1")
            cursor.fetchone()
        except Exception:
            has_dbstat = False
            logger.warning("dbstat virtual table not available; table/index sizes will be estimated or omitted")
        
        # Get table sizes and row counts
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        tables = cursor.fetchall()
        
        stats['tables'] = {}
        for (table_name,) in tables:
            # Row count
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]
            except Exception as e:
                logger.warning(f"Failed to get row count for {table_name}: {e}")
                row_count = 0
            
            # Estimate table size
            table_size_mb = None
            if has_dbstat:
                try:
                    cursor.execute(f"SELECT SUM(pgsize) FROM dbstat WHERE name='{table_name}'")
                    result = cursor.fetchone()
                    table_size = result[0] if result[0] else 0
                    table_size_mb = table_size / 1024 / 1024
                except Exception as e:
                    logger.warning(f"Failed to get size for {table_name} via dbstat: {e}")
            
            stats['tables'][table_name] = {
                'row_count': row_count,
                'size_mb': table_size_mb
            }
        
        # Get index information
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'")
        indexes = cursor.fetchall()
        
        stats['indexes'] = {}
        for (index_name,) in indexes:
            index_size_mb = None
            if has_dbstat:
                try:
                    cursor.execute(f"SELECT SUM(pgsize) FROM dbstat WHERE name='{index_name}'")
                    result = cursor.fetchone()
                    index_size = result[0] if result[0] else 0
                    index_size_mb = index_size / 1024 / 1024
                except Exception as e:
                    logger.warning(f"Failed to get size for index {index_name}: {e}")
            
            stats['indexes'][index_name] = {
                'size_mb': index_size_mb
            }
        
        stats['total_indexes'] = len(indexes)
        stats['total_tables'] = len(tables)
        
        # Identify largest tables by row count
        largest_tables = sorted(
            stats['tables'].items(),
            key=lambda x: x[1]['row_count'],
            reverse=True
        )[:5]
        stats['largest_tables'] = [name for name, _ in largest_tables]
        
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        stats['error'] = str(e)
    finally:
        conn.close()
    
    return stats


class ConnectionPool:
    """
    Simple connection pool for SQLite
    
    Note: SQLite has limited concurrency support, so this is mainly
    useful for avoiding repeated connection overhead.
    """
    
    def __init__(self, db_path: str = DATABASE_PATH, pool_size: int = 5):
        """
        Initialize connection pool
        
        Args:
            db_path: Path to SQLite database
            pool_size: Number of connections to maintain
        """
        self.db_path = db_path
        self.pool_size = pool_size
        self._pool: List[sqlite3.Connection] = []
        self._lock = threading.Lock()
        self._in_use = set()
        
        # Create initial connections
        for _ in range(pool_size):
            conn = sqlite3.connect(db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._pool.append(conn)
        
        logger.info(f"ConnectionPool initialized with {pool_size} connections to {db_path}")
    
    def get_connection(self) -> sqlite3.Connection:
        """
        Get a connection from the pool
        
        Returns:
            Database connection
        """
        with self._lock:
            while not self._pool:
                # Pool exhausted, wait a bit (in real implementation, use condition variable)
                import time
                time.sleep(0.1)
            
            conn = self._pool.pop()
            self._in_use.add(id(conn))
            return conn
    
    def return_connection(self, conn: sqlite3.Connection):
        """
        Return a connection to the pool
        
        Args:
            conn: Connection to return
        """
        with self._lock:
            conn_id = id(conn)
            if conn_id in self._in_use:
                self._in_use.remove(conn_id)
                self._pool.append(conn)
    
    @contextmanager
    def get_cursor(self):
        """
        Context manager for getting a cursor with automatic connection return
        
        Yields:
            Database cursor
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self.return_connection(conn)
    
    def close_all(self):
        """Close all connections in the pool"""
        with self._lock:
            for conn in self._pool:
                conn.close()
            self._pool.clear()
            self._in_use.clear()
        logger.info("All connections closed")


if __name__ == '__main__':
    # CLI tool for database optimization
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if len(sys.argv) < 2:
        print("Database Optimization Tool")
        print("\nUsage:")
        print("  python database_optimizer.py analyze   - Analyze query performance")
        print("  python database_optimizer.py optimize  - Create indexes and optimize")
        print("  python database_optimizer.py stats     - Show database statistics")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == 'analyze':
        print("\n=== Analyzing Query Performance ===\n")
        results = analyze_query_performance()
        
        if 'error' in results:
            print(f"Error: {results['error']}")
        else:
            print(f"Analyzed {len(results['queries'])} queries:\n")
            for query in results['queries']:
                print(f"Query: {query['name']}")
                print(f"  Uses Index: {query['uses_index']}")
                print(f"  Full Scan: {query['full_scan']}")
                print(f"  Plan: {', '.join(query['plan'])}\n")
            
            if results['recommendations']:
                print("\nRecommendations:")
                for rec in results['recommendations']:
                    print(f"  - {rec}")
            else:
                print("\nNo recommendations - queries are well-optimized!")
    
    elif command == 'optimize':
        print("\n=== Optimizing Database ===\n")
        create_additional_indexes()
        print()
        optimize_database()
        print("\nOptimization completed!")
    
    elif command == 'stats':
        print("\n=== Database Statistics ===\n")
        stats = get_database_stats()
        
        if 'error' in stats:
            print(f"Error: {stats['error']}")
        else:
            print(f"Database Size: {stats['database_size_mb']:.2f} MB")
            print(f"Total Tables: {stats['total_tables']}")
            print(f"Total Indexes: {stats['total_indexes']}\n")
            
            print("Tables:")
            for table_name, table_info in stats['tables'].items():
                print(f"  {table_name}:")
                print(f"    Rows: {table_info['row_count']:,}")
                print(f"    Size: {table_info['size_mb']:.2f} MB")
            
            print("\nLargest Tables:")
            for table_name in stats['largest_tables']:
                print(f"  - {table_name} ({stats['tables'][table_name]['row_count']:,} rows)")
    
    else:
        print(f"Unknown command: {command}")
        print("Use 'analyze', 'optimize', or 'stats'")
        sys.exit(1)
