import sqlite3
import json
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
from config import (
    DATABASE_PATH, FAISS_INDEX_PATH, CHUNKS_MAP_PATH,
    EMBEDDING_MODEL, TOP_K_CHUNKS, TOP_K_SQL_RESULTS,
    SIMILARITY_THRESHOLD, CRIME_KEYWORDS
)
import logging_config  # noqa: F401
import logging

# Get logger
logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retrieval combining semantic search (FAISS) and keyword search (SQL)."""
    
    def __init__(self):
        """Initialize the hybrid retriever with FAISS index, embeddings model, and database."""
        logger.info("Initializing HybridRetriever...")
        
        # Load sentence transformer model
        try:
            self.model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info(f"✅ Loaded embedding model: {EMBEDDING_MODEL}")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            raise
        
        # Load FAISS index - allow degraded mode if missing
        try:
            self.index = faiss.read_index(FAISS_INDEX_PATH)
            logger.info(f"✅ Loaded FAISS index with {self.index.ntotal} vectors")
        except Exception as e:
            logger.warning(f"⚠️  FAISS index not found or failed to load: {e}")
            logger.warning("⚠️  Semantic search disabled. Run 'python create_index.py' to build the index.")
            self.index = None
        
        # Load chunks map - allow degraded mode if missing
        try:
            with open(CHUNKS_MAP_PATH, 'r', encoding='utf-8') as f:
                self.chunks_map = json.load(f)
            logger.info(f"✅ Loaded chunks map with {len(self.chunks_map)} chunks")
        except Exception as e:
            logger.warning(f"⚠️  Chunks map not found or failed to load: {e}")
            logger.warning("⚠️  Semantic search disabled. Run 'python create_index.py' to build the index.")
            self.chunks_map = {}
        
        # Establish database connection
        try:
            self.db_path = DATABASE_PATH
            logger.info(f"✅ Database path set to: {self.db_path}")
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            raise
    
    def _get_db_connection(self):
        """Get a new database connection."""
        return sqlite3.connect(self.db_path)
    
    def retrieve_semantic(self, query: str, top_k: int = TOP_K_CHUNKS) -> List[Dict]:
        """
        Retrieve relevant chunks using semantic search with FAISS.
        Collects top_k * 3 hits to allow for multiple chunks per case.
        
        Args:
            query: Search query string
            top_k: Number of top results to return (will get more hits for grouping)
            
        Returns:
            List of dicts with chunk text, similarity scores, and metadata
        """
        # Return empty if semantic search is disabled (degraded mode)
        if self.index is None or not self.chunks_map:
            logger.warning("Semantic search disabled - index or chunks_map not available")
            return []
        
        logger.info(f"Performing semantic search for: '{query}'")
        
        try:
            # Generate query embedding (normalized for cosine similarity)
            query_embedding = self.model.encode([query], normalize_embeddings=True)
            
            # Search FAISS index - get top_k * 3 hits to allow grouping by case
            similarities, indices = self.index.search(
                np.array(query_embedding, dtype='float32'),
                top_k * 3
            )
            
            results = []
            for idx, similarity in zip(indices[0], similarities[0]):
                # Similarity is already cosine similarity (inner product of normalized vectors)
                
                # Filter by threshold
                if similarity < SIMILARITY_THRESHOLD:
                    continue
                
                # Get chunk data from map
                chunk_idx = str(idx)
                if chunk_idx not in self.chunks_map:
                    logger.warning(f"Chunk index {chunk_idx} not found in chunks_map")
                    continue
                
                chunk_data = self.chunks_map[chunk_idx]
                case_id = chunk_data.get('case_id')
                
                result = {
                    "chunk_text": chunk_data.get('text', ''),
                    "similarity": float(similarity),
                    "case_id": case_id,
                    "case_number": chunk_data.get('case_number', 'Unknown'),
                    "case_type": chunk_data.get('case_type', 'Unknown'),
                    "judgment_date": chunk_data.get('judgment_date', 'Unknown'),
                    "petitioner": chunk_data.get('petitioner', 'Unknown'),
                    "respondent": chunk_data.get('respondent', 'Unknown'),
                    "full_case_id": chunk_data.get('full_case_id', 'Unknown'),
                    "source": "semantic"
                }
                results.append(result)
            
            logger.info(f"✅ Semantic search returned {len(results)} chunk hits")
            return results
            
        except Exception as e:
            logger.error(f"❌ Semantic search failed: {e}")
            return []
    
    def retrieve_keyword(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        """
        Retrieve relevant judgments using SQL keyword search.
        
        Args:
            query: Search query string
            filters: Optional filters (case_type, year, etc.)
            
        Returns:
            List of matching judgments with metadata
        """
        logger.info(f"Performing keyword search for: '{query}'")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build SQL query
                sql = "SELECT id, case_number, case_type, judgment_date, petitioner_name, respondent_name, full_text FROM judgments WHERE 1=1"
                params = []
                
                # Add text search
                if query:
                    sql += " AND full_text LIKE ?"
                    params.append(f"%{query}%")
                
                # Apply filters
                if filters:
                    if 'case_type' in filters:
                        sql += " AND case_type LIKE ?"
                        params.append(f"%{filters['case_type']}%")
                    
                    if 'year' in filters:
                        sql += " AND judgment_date LIKE ?"
                        params.append(f"%{filters['year']}%")
                    
                    if 'petitioner' in filters:
                        sql += " AND petitioner_name LIKE ?"
                        params.append(f"%{filters['petitioner']}%")
                    
                    if 'respondent' in filters:
                        sql += " AND respondent_name LIKE ?"
                        params.append(f"%{filters['respondent']}%")
                
                sql += f" LIMIT {TOP_K_SQL_RESULTS}"
                
                cursor.execute(sql, params)
                rows = cursor.fetchall()
                
                results = []
                for row in rows:
                    result = {
                        "case_id": row[0],
                        "case_number": row[1] or 'Unknown',
                        "case_type": row[2] or 'Unknown',
                        "judgment_date": row[3] or 'Unknown',
                        "petitioner": row[4] or 'Unknown',
                        "respondent": row[5] or 'Unknown',
                        "chunk_text": row[6][:500] if row[6] else '',  # First 500 chars as preview
                        "full_case_id": f"{row[2] or ''} {row[1] or ''}".strip(),
                        "source": "keyword"
                    }
                    results.append(result)
            
            logger.info(f"✅ Keyword search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"❌ Keyword search failed: {e}")
            return []
    
    def retrieve_by_crime_category(self, query: str) -> List[Dict]:
        """
        Retrieve cases by crime category using predefined keywords.
        
        Args:
            query: Search query potentially containing crime keywords
            
        Returns:
            List of cases matching detected crime categories
        """
        logger.info(f"Checking for crime categories in: '{query}'")
        
        detected_crimes = []
        query_lower = query.lower()
        
        # Detect crime keywords
        for crime_type, keywords in CRIME_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    detected_crimes.append((crime_type, keyword))
                    break
        
        if not detected_crimes:
            logger.info("No crime categories detected")
            return []
        
        logger.info(f"Detected crime categories: {[c[0] for c in detected_crimes]}")
        
        # Search for all detected crime types
        all_results = []
        for crime_type, keyword in detected_crimes:
            results = self.retrieve_keyword(keyword, filters=None)
            for result in results:
                result['crime_category'] = crime_type
                result['matched_keyword'] = keyword
            all_results.extend(results)
        
        # Deduplicate by case_id
        seen_ids = set()
        unique_results = []
        for result in all_results:
            if result['case_id'] not in seen_ids:
                seen_ids.add(result['case_id'])
                unique_results.append(result)
        
        logger.info(f"✅ Crime category search returned {len(unique_results)} unique results")
        return unique_results
    
    def hybrid_retrieve(self, query: str, top_k: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
        """
        Perform hybrid retrieval combining semantic and keyword search.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            filters: Optional search filters
            
        Returns:
            Ranked list of case-level results with multiple chunks stored in 'chunks' field
        """
        logger.info(f"Performing hybrid retrieval for: '{query}'")
        
        from config import HYBRID_WEIGHT_SEMANTIC, HYBRID_WEIGHT_KEYWORD
        
        # Check for crime categories first
        crime_results = self.retrieve_by_crime_category(query)
        
        # Perform both searches
        semantic_results = self.retrieve_semantic(query, top_k=top_k)
        keyword_results = self.retrieve_keyword(query, filters=filters)
        
        # Group semantic results by case_id and keep top N=3 chunks per case
        case_chunks = {}  # case_id -> list of chunk dicts
        
        for result in semantic_results:
            case_id = result['case_id']
            if case_id not in case_chunks:
                case_chunks[case_id] = []
            
            # Keep up to 3 best chunks per case (highest similarity)
            if len(case_chunks[case_id]) < 3:
                result['semantic_score'] = result['similarity']
                result['keyword_score'] = 0.0
                case_chunks[case_id].append(result)
            else:
                # Replace lowest scoring chunk if this one is better
                min_chunk = min(case_chunks[case_id], key=lambda x: x['similarity'])
                if result['similarity'] > min_chunk['similarity']:
                    case_chunks[case_id].remove(min_chunk)
                    result['semantic_score'] = result['similarity']
                    result['keyword_score'] = 0.0
                    case_chunks[case_id].append(result)
        
        # Merge keyword results
        for result in keyword_results:
            case_id = result['case_id']
            if case_id not in case_chunks:
                # Create preview chunk for keyword-only case
                case_chunks[case_id] = []
            
            # Check if we already have this chunk (by text similarity)
            is_duplicate = False
            for existing in case_chunks[case_id]:
                if existing['chunk_text'][:100] == result['chunk_text'][:100]:
                    # Merge scores
                    existing['keyword_score'] = 1.0
                    is_duplicate = True
                    break
            
            if not is_duplicate and len(case_chunks[case_id]) < 3:
                result['semantic_score'] = 0.0
                result['keyword_score'] = 1.0
                case_chunks[case_id].append(result)
        
        # Add crime results
        for result in crime_results:
            case_id = result['case_id']
            if case_id not in case_chunks:
                case_chunks[case_id] = []
            
            # Check for duplicates
            is_duplicate = any(
                existing['chunk_text'][:100] == result['chunk_text'][:100]
                for existing in case_chunks[case_id]
            )
            
            if not is_duplicate and len(case_chunks[case_id]) < 3:
                result['semantic_score'] = 0.0
                result['keyword_score'] = 1.0
                case_chunks[case_id].append(result)
        
        # Build case-level results with chunks stored in 'chunks' field
        case_results = []
        for case_id, chunks in case_chunks.items():
            if not chunks:
                continue
            
            # Sort chunks by score (best first)
            chunks.sort(key=lambda x: x.get('similarity', x.get('semantic_score', 0)), reverse=True)
            
            # Calculate case-level hybrid score (weighted mean of chunk scores)
            case_semantic_score = sum(c.get('semantic_score', 0) for c in chunks) / len(chunks)
            case_keyword_score = sum(c.get('keyword_score', 0) for c in chunks) / len(chunks)
            case_hybrid_score = (
                HYBRID_WEIGHT_SEMANTIC * case_semantic_score +
                HYBRID_WEIGHT_KEYWORD * case_keyword_score
            )
            
            # Use first/best chunk for case-level metadata
            best_chunk = chunks[0]
            
            case_result = {
                "case_id": case_id,
                "case_number": best_chunk.get('case_number', 'Unknown'),
                "case_type": best_chunk.get('case_type', 'Unknown'),
                "judgment_date": best_chunk.get('judgment_date', 'Unknown'),
                "petitioner": best_chunk.get('petitioner', 'Unknown'),
                "respondent": best_chunk.get('respondent', 'Unknown'),
                "full_case_id": best_chunk.get('full_case_id', 'Unknown'),
                "chunks": chunks,  # Store all chunks for this case
                "hybrid_score": case_hybrid_score,
                "chunk_count": len(chunks)
            }
            case_results.append(case_result)
        
        # Sort by case-level hybrid score
        case_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        # Return top-k cases
        final_results = case_results[:top_k]
        logger.info(f"✅ Hybrid retrieval returned {len(final_results)} cases with {sum(r['chunk_count'] for r in final_results)} total chunks")
        
        return final_results
    
    def get_full_judgment(self, case_id: int) -> Dict:
        """
        Retrieve complete judgment details including related data.
        
        Args:
            case_id: Database ID of the judgment
            
        Returns:
            Dict with complete judgment information
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Determine available columns using PRAGMA
                cols = set(r[1] for r in cursor.execute("PRAGMA table_info(judgments)").fetchall())
                base_cols = ["id", "case_number", "case_type", "judgment_date", "petitioner_name", "respondent_name", "full_text"]
                optional_cols = [c for c in ("judges", "outcome") if c in cols]
                select_cols = base_cols + optional_cols
                
                sql = f"SELECT {', '.join(select_cols)} FROM judgments WHERE id = ?"
                cursor.execute(sql, (case_id,))
                row = cursor.fetchone()
                
                if not row:
                    return {}
                
                # Build judgment dict using indexes from select_cols
                idx = {c: i for i, c in enumerate(select_cols)}
                judgment = {
                    "id": row[idx["id"]],
                    "case_number": row[idx["case_number"]],
                    "case_type": row[idx["case_type"]],
                    "judgment_date": row[idx["judgment_date"]],
                    "petitioner": row[idx["petitioner_name"]],
                    "respondent": row[idx["respondent_name"]],
                    "full_text": row[idx["full_text"]],
                    "judges": row[idx["judges"]] if "judges" in idx else None,
                    "outcome": row[idx["outcome"]] if "outcome" in idx else None,
                }
                
                # Get advocates (if table exists)
                try:
                    cursor.execute("SELECT name, role FROM advocates WHERE case_id = ?", (case_id,))
                    judgment['advocates'] = [{"name": row[0], "role": row[1]} for row in cursor.fetchall()]
                except sqlite3.OperationalError:
                    judgment['advocates'] = []
                
                # Get cited laws (if table exists)
                try:
                    cursor.execute("SELECT law_name, section FROM laws WHERE case_id = ?", (case_id,))
                    judgment['laws'] = [{"law": row[0], "section": row[1]} for row in cursor.fetchall()]
                except sqlite3.OperationalError:
                    judgment['laws'] = []
            
            return judgment
            
        except Exception as e:
            logger.error(f"❌ Failed to get full judgment for case_id {case_id}: {e}")
            return {}


if __name__ == '__main__':
    # Test the retriever
    print("Testing HybridRetriever...")
    retriever = HybridRetriever()
    
    test_query = "murder"
    print(f"\nTest Query: '{test_query}'")
    
    results = retriever.hybrid_retrieve(test_query, top_k=3)
    print(f"\nFound {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Case: {result['full_case_id']}")
        print(f"   Score: {result['hybrid_score']:.3f}")
        print(f"   Chunks: {result.get('chunk_count', 1)}")
        # Print preview from first chunk
        if 'chunks' in result and result['chunks']:
            print(f"   Text: {result['chunks'][0]['chunk_text'][:100]}...")
        elif 'chunk_text' in result:
            print(f"   Text: {result['chunk_text'][:100]}...")
