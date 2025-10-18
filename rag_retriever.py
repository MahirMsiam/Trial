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
    SIMILARITY_THRESHOLD, CRIME_KEYWORDS,
    CACHE_ENABLED, USE_BM25, USE_RRF, BM25_K1, BM25_B, RRF_K,
    HYBRID_WEIGHT_SEMANTIC, HYBRID_WEIGHT_KEYWORD, USE_CONNECTION_POOL,
    CACHE_VERSION
)
import logging_config  # noqa: F401
import logging

# Initialize logger first before any usage
logger = logging.getLogger(__name__)

# Import caching and ranking modules
try:
    from cache_manager import QueryCache, EmbeddingCache
    from ranking_algorithms import BM25Ranker, ReciprocalRankFusion, normalize_scores
    CACHING_AVAILABLE = True
    RANKING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Optional modules not available: {e}")
    CACHING_AVAILABLE = False
    RANKING_AVAILABLE = False
    normalize_scores = None  # Define fallback

# Module-level connection pool
_connection_pool = None

def get_connection_pool():
    """Get or create module-level connection pool"""
    global _connection_pool
    if _connection_pool is None and USE_CONNECTION_POOL:
        try:
            from database_optimizer import ConnectionPool
            _connection_pool = ConnectionPool(DATABASE_PATH, pool_size=5)
            logger.info("✅ Connection pool initialized (size=5)")
        except Exception as e:
            logger.warning(f"⚠️  Failed to initialize connection pool: {e}")
    return _connection_pool


class HybridRetriever:
    """Hybrid retrieval combining semantic search (FAISS) and keyword search (SQL)."""
    
    def __init__(self):
        """Initialize the hybrid retriever with FAISS index, embeddings model, and database."""
        logger.info("Initializing HybridRetriever...")
        
        # Initialize connection pool if enabled
        self.connection_pool = get_connection_pool() if USE_CONNECTION_POOL else None
        
        # Load sentence transformer model
        try:
            self.model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info(f"✅ Loaded embedding model: {EMBEDDING_MODEL}")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            raise
        
        # Initialize caching if enabled and available
        self.query_cache = None
        self.embedding_cache = None
        if CACHE_ENABLED and CACHING_AVAILABLE:
            try:
                self.query_cache = QueryCache()
                # Pass EMBEDDING_MODEL as model_id to EmbeddingCache
                self.embedding_cache = EmbeddingCache(model_id=EMBEDDING_MODEL)
                logger.info("✅ Caching enabled (query and embedding caches initialized)")
            except Exception as e:
                logger.warning(f"⚠️  Failed to initialize caching: {e}")
        else:
            logger.info("ℹ️  Caching disabled")
        
        # Initialize BM25 ranker if enabled
        self.bm25_ranker = None
        if USE_BM25 and RANKING_AVAILABLE:
            self._initialize_bm25()
        
        # Initialize RRF if enabled
        self.rrf = None
        if USE_RRF and RANKING_AVAILABLE:
            try:
                self.rrf = ReciprocalRankFusion(k=RRF_K)
                logger.info(f"✅ RRF initialized (k={RRF_K})")
            except Exception as e:
                logger.warning(f"⚠️  Failed to initialize RRF: {e}")
        
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
        """
        Get a database connection or use pool if available.
        
        Returns:
            Context manager that yields (cursor, connection) tuple.
            When using pool, connection is None as it's managed by pool.
        """
        from contextlib import contextmanager
        
        @contextmanager
        def _connection_context():
            if self.connection_pool:
                # Use connection pool - it handles commit/rollback/return
                with self.connection_pool.get_cursor() as cursor:
                    yield cursor, None
            else:
                # Direct connection - we manage lifecycle
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                try:
                    cursor = conn.cursor()
                    yield cursor, conn
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
                finally:
                    conn.close()
        
        return _connection_context()

    
    def close(self):
        """Close connection pool if used."""
        if self.connection_pool:
            try:
                self.connection_pool.close_all()
                logger.info("✅ Connection pool closed")
            except Exception as e:
                logger.warning(f"Failed to close connection pool: {e}")
    
    def _initialize_bm25(self):
        """Initialize BM25 ranker with corpus from database."""
        try:
            from config import BM25_CORPUS_LIMIT
            logger.info(f"Initializing BM25 ranker (corpus limit: {BM25_CORPUS_LIMIT})...")
            
            with self._get_db_connection() as (cursor, conn):
                cursor.execute(f"SELECT full_text FROM judgments LIMIT {BM25_CORPUS_LIMIT}")
                corpus = [row[0] for row in cursor.fetchall() if row[0]]
            
            if corpus:
                self.bm25_ranker = BM25Ranker(corpus, k1=BM25_K1, b=BM25_B)
                logger.info(f"✅ BM25 ranker initialized with {len(corpus)} documents")
            else:
                logger.warning("⚠️  No documents found for BM25 initialization")
        except Exception as e:
            logger.warning(f"⚠️  Failed to initialize BM25 ranker: {e}")
    
    def rebuild_bm25(self):
        """
        Rebuild BM25 corpus from database.
        Useful for refreshing after database updates.
        """
        if not USE_BM25 or not RANKING_AVAILABLE:
            logger.warning("BM25 is not enabled or not available")
            return False
        
        try:
            logger.info("Rebuilding BM25 corpus...")
            self._initialize_bm25()
            return self.bm25_ranker is not None
        except Exception as e:
            logger.error(f"❌ Failed to rebuild BM25 corpus: {e}")
            return False
    
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
        
        # Check query cache (include config version, model, and threshold in key)
        if self.query_cache:
            try:
                # Normalize query for consistent caching
                normalized_query = query.lower().strip()
                cache_key = f"semantic:{CACHE_VERSION}:{EMBEDDING_MODEL}:{SIMILARITY_THRESHOLD}:{normalized_query}:{top_k}"
                cached_results = self.query_cache.get(cache_key)
                if cached_results is not None:
                    logger.info(f"✅ Cache hit for semantic search: '{query}'")
                    return cached_results
            except Exception as e:
                logger.warning(f"Cache get failed: {e}")
        
        logger.info(f"Performing semantic search for: '{query}'")
        
        try:
            # Check embedding cache (include model in key)
            query_embedding = None
            if self.embedding_cache:
                try:
                    query_embedding = self.embedding_cache.get_cached_embedding(query)
                    if query_embedding is not None:
                        logger.debug("Using cached embedding")
                        query_embedding = np.array([query_embedding], dtype='float32')
                except Exception as e:
                    logger.warning(f"Embedding cache get failed: {e}")
            
            # Generate query embedding if not cached
            if query_embedding is None:
                query_embedding = self.model.encode([query], normalize_embeddings=True)
                # Cache the embedding
                if self.embedding_cache:
                    try:
                        self.embedding_cache.cache_embedding(query, query_embedding[0].tolist())
                    except Exception as e:
                        logger.warning(f"Embedding cache set failed: {e}")
            
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
            
            # Cache results (with config version and model in key)
            if self.query_cache:
                try:
                    # Normalize query for consistent caching
                    normalized_query = query.lower().strip()
                    cache_key = f"semantic:{CACHE_VERSION}:{EMBEDDING_MODEL}:{SIMILARITY_THRESHOLD}:{normalized_query}:{top_k}"
                    self.query_cache.set(cache_key, results)
                except Exception as e:
                    logger.warning(f"Cache set failed: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Semantic search failed: {e}")
            return []
    
    def retrieve_keyword(self, query: str, filters: Optional[Dict] = None, limit: int = TOP_K_SQL_RESULTS) -> List[Dict]:
        """
        Retrieve relevant judgments using SQL keyword search.
        
        Args:
            query: Search query string
            filters: Optional filters (case_type, year, petitioner, respondent, advocate, section, rule_outcome, etc.)
            limit: Maximum number of results to return
            
        Returns:
            List of matching judgments with metadata
        """
        # Coerce and bound limit safely
        try:
            limit = int(limit or TOP_K_SQL_RESULTS)
        except Exception:
            limit = TOP_K_SQL_RESULTS
        limit = max(1, min(limit, 1000))
        
        # Check query cache
        if self.query_cache:
            try:
                import hashlib
                # Normalize query for consistent caching
                normalized_query = query.lower().strip()
                filters_hash = hashlib.md5(json.dumps(filters or {}, sort_keys=True).encode()).hexdigest()
                cache_key = f"keyword:{CACHE_VERSION}:{normalized_query}:{filters_hash}:{limit}"
                cached_results = self.query_cache.get(cache_key)
                if cached_results is not None:
                    logger.info(f"✅ Cache hit for keyword search: '{query}'")
                    return cached_results
            except Exception as e:
                logger.warning(f"Cache get failed: {e}")
        
        logger.info(f"Performing keyword search for: '{query}'")
        
        try:
            with self._get_db_connection() as (cursor, conn):
                # Check which optional tables exist
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                existing_tables = {row[0] for row in cursor.fetchall()}
                has_advocates = 'advocates' in existing_tables
                has_laws = 'laws' in existing_tables
                
                # Build SQL query with JOINs for advocate and section filtering
                sql = "SELECT DISTINCT j.id, j.case_number, j.case_type, j.judgment_date, j.petitioner_name, j.respondent_name, j.full_text FROM judgments j"
                joins = []
                conditions = []
                params = []
                
                # Handle advocate search (JOIN advocates table) - only if table exists
                if filters and filters.get('advocate') and has_advocates:
                    joins.append("JOIN advocates a ON j.id = a.judgment_id")
                    conditions.append("a.advocate_name LIKE ?")
                    params.append(f"%{filters['advocate']}%")
                elif filters and filters.get('advocate') and not has_advocates:
                    logger.warning("Advocate filter requested but 'advocates' table does not exist")
                
                # Handle law/section search (JOIN laws table) - only if table exists
                if filters and filters.get('section') and has_laws:
                    joins.append("JOIN laws l ON j.id = l.judgment_id")
                    conditions.append("l.law_text LIKE ?")
                    params.append(f"%{filters['section']}%")
                elif filters and filters.get('section') and not has_laws:
                    logger.warning("Section filter requested but 'laws' table does not exist")
                
                # Add joins to query
                if joins:
                    sql += " " + " ".join(joins)
                
                # Add WHERE clause start
                sql += " WHERE 1=1"
                
                # Add text search on full_text
                if query:
                    conditions.append("j.full_text LIKE ?")
                    params.append(f"%{query}%")
                
                # Apply additional filters
                if filters:
                    if 'case_type' in filters:
                        conditions.append("j.case_type LIKE ?")
                        params.append(f"%{filters['case_type']}%")
                    
                    if 'case_number' in filters:
                        conditions.append("j.case_number = ?")
                        params.append(filters['case_number'])
                    
                    if 'year' in filters:
                        conditions.append("(j.case_year = ? OR j.judgment_date LIKE ?)")
                        params.extend([filters['year'], f"%{filters['year']}%"])
                    
                    if 'petitioner' in filters:
                        conditions.append("j.petitioner_name LIKE ?")
                        params.append(f"%{filters['petitioner']}%")
                    
                    if 'respondent' in filters:
                        conditions.append("j.respondent_name LIKE ?")
                        params.append(f"%{filters['respondent']}%")
                    
                    if 'rule_outcome' in filters:
                        conditions.append("j.rule_outcome LIKE ?")
                        params.append(f"%{filters['rule_outcome']}%")
                
                # Add all conditions
                if conditions:
                    sql += " AND " + " AND ".join(conditions)
                
                # Parameterize LIMIT for safety
                sql += " LIMIT ?"
                params.append(limit)
                
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
                        "full_text": row[6] or '',  # Full text for BM25
                        "full_case_id": f"{row[2] or ''} {row[1] or ''}".strip(),
                        "source": "keyword"
                    }
                    results.append(result)
            
            # Apply BM25 ranking if enabled
            if self.bm25_ranker and results:
                try:
                    logger.debug("Applying BM25 ranking to keyword results")
                    results = self.bm25_ranker.rank_documents(query, results)
                except Exception as e:
                    logger.warning(f"BM25 ranking failed: {e}")
            
            logger.info(f"✅ Keyword search returned {len(results)} results")
            
            # Cache results
            if self.query_cache:
                try:
                    import hashlib
                    # Normalize query for consistent caching
                    normalized_query = query.lower().strip()
                    filters_hash = hashlib.md5(json.dumps(filters or {}, sort_keys=True).encode()).hexdigest()
                    cache_key = f"keyword:{CACHE_VERSION}:{normalized_query}:{filters_hash}:{limit}"
                    self.query_cache.set(cache_key, results)
                except Exception as e:
                    logger.warning(f"Cache set failed: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Keyword search failed: {e}")
            return []
    
    def retrieve_by_crime_category(self, query: str, limit_per_crime: int = 50) -> List[Dict]:
        """
        Retrieve cases by crime category using predefined keywords.
        
        Args:
            query: Search query potentially containing crime keywords
            limit_per_crime: Maximum results per crime type
            
        Returns:
            List of cases matching detected crime categories
        """
        import hashlib
        
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
        
        def norm_hash(text):
            """Normalize text and compute hash for deduplication"""
            normalized = (text or '')[:300].lower()
            return hashlib.md5(normalized.encode()).hexdigest()
        
        # Search for all detected crime types with per-crime deduplication
        all_results = []
        for crime_type, keyword in detected_crimes:
            # Fetch extra results for headroom
            results = self.retrieve_keyword(keyword, filters=None, limit=limit_per_crime * 2)
            
            # Deduplicate within this crime type
            seen_ids = set()
            seen_hashes = set()
            crime_unique = []
            
            for result in results:
                case_id = result['case_id']
                text_hash = norm_hash(result.get('chunk_text', ''))
                
                if case_id not in seen_ids and text_hash not in seen_hashes:
                    seen_ids.add(case_id)
                    seen_hashes.add(text_hash)
                    result['crime_category'] = crime_type
                    result['matched_keyword'] = keyword
                    crime_unique.append(result)
                    
                    if len(crime_unique) >= limit_per_crime:
                        break
            
            # If BM25 scores available, keep top results by score
            if crime_unique and 'bm25_score' in crime_unique[0]:
                crime_unique.sort(key=lambda x: x.get('bm25_score', 0), reverse=True)
                crime_unique = crime_unique[:limit_per_crime]
            
            all_results.extend(crime_unique)
        
        # Final cross-crime deduplication by case_id
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
        # Check query cache
        if self.query_cache:
            try:
                import hashlib
                # Normalize query for consistent caching
                normalized_query = query.lower().strip()
                filters_hash = hashlib.md5(json.dumps(filters or {}, sort_keys=True).encode()).hexdigest()
                cache_key = f"hybrid:{CACHE_VERSION}:{normalized_query}:{top_k}:{filters_hash}"
                cached_results = self.query_cache.get(cache_key)
                if cached_results is not None:
                    logger.info(f"✅ Cache hit for hybrid search: '{query}'")
                    return cached_results
            except Exception as e:
                logger.warning(f"Cache get failed: {e}")
        
        logger.info(f"Performing hybrid retrieval for: '{query}'")
        
        from config import HYBRID_WEIGHT_SEMANTIC, HYBRID_WEIGHT_KEYWORD, CRIME_WEIGHT_HYBRID
        
        # Check for crime categories first
        crime_results = self.retrieve_by_crime_category(query)
        
        # Perform both searches
        semantic_results = self.retrieve_semantic(query, top_k=top_k)
        keyword_results = self.retrieve_keyword(query, filters=filters)
        
        # Use RRF if enabled, otherwise use weighted average
        if self.rrf and (semantic_results or keyword_results):
            try:
                logger.debug("Using RRF for result fusion")
                # Prepare ranked lists for RRF
                ranked_lists = []
                weights = []
                
                if semantic_results:
                    # Sort by similarity for RRF
                    semantic_sorted = sorted(semantic_results, key=lambda x: x.get('similarity', 0), reverse=True)
                    ranked_lists.append(semantic_sorted)
                    weights.append(HYBRID_WEIGHT_SEMANTIC)
                
                if keyword_results:
                    # Sort by BM25 score if available, otherwise keep order
                    if keyword_results and 'bm25_score' in keyword_results[0]:
                        keyword_sorted = sorted(keyword_results, key=lambda x: x.get('bm25_score', 0), reverse=True)
                    else:
                        keyword_sorted = keyword_results
                    ranked_lists.append(keyword_sorted)
                    weights.append(HYBRID_WEIGHT_KEYWORD)
                
                if crime_results:
                    ranked_lists.append(crime_results)
                    weights.append(CRIME_WEIGHT_HYBRID)  # Configurable weight for crime results
                
                # Fuse with RRF
                fused_results = self.rrf.fuse_with_weights(ranked_lists, weights, id_field='case_id')
                
                # Group by case and organize chunks
                case_results = self._organize_fused_results(fused_results, top_k)
                
            except Exception as e:
                logger.warning(f"RRF fusion failed, falling back to weighted average: {e}")
                # Fall through to weighted average method
                case_results = self._hybrid_retrieve_weighted(query, top_k, semantic_results, keyword_results, crime_results)
        else:
            # Use traditional weighted average
            case_results = self._hybrid_retrieve_weighted(query, top_k, semantic_results, keyword_results, crime_results)
        
        logger.info(f"✅ Hybrid retrieval returned {len(case_results)} cases with {sum(r['chunk_count'] for r in case_results)} total chunks")
        
        # Cache results
        if self.query_cache:
            try:
                import hashlib
                # Normalize query for consistent caching
                normalized_query = query.lower().strip()
                filters_hash = hashlib.md5(json.dumps(filters or {}, sort_keys=True).encode()).hexdigest()
                cache_key = f"hybrid:{CACHE_VERSION}:{normalized_query}:{top_k}:{filters_hash}"
                self.query_cache.set(cache_key, case_results)
            except Exception as e:
                logger.warning(f"Cache set failed: {e}")
        
        return case_results
    
    def _hybrid_retrieve_weighted(self, query: str, top_k: int, semantic_results: List[Dict], 
                                   keyword_results: List[Dict], crime_results: List[Dict]) -> List[Dict]:
        """Helper method for weighted average hybrid retrieval (original implementation)"""
        from config import HYBRID_WEIGHT_SEMANTIC, HYBRID_WEIGHT_KEYWORD
        
        # Normalize BM25 scores in keyword results using helper function
        if keyword_results and 'bm25_score' in keyword_results[0]:
            if normalize_scores:
                # Use helper function from ranking_algorithms
                keyword_results = normalize_scores(keyword_results, 'bm25_score')
                # Copy normalized score to _bm25_norm for backward compatibility
                for r in keyword_results:
                    r['_bm25_norm'] = r.get('bm25_score_normalized', 0.0)
            else:
                # Fallback: manual normalization
                scores = [r.get('bm25_score', 0.0) for r in keyword_results]
                lo, hi = min(scores), max(scores)
                for r in keyword_results:
                    if hi == lo:
                        r['_bm25_norm'] = 1.0 if hi > 0 else 0.0
                    else:
                        r['_bm25_norm'] = (r.get('bm25_score', 0) - lo) / (hi - lo)
        
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
            
            # Use normalized BM25 score if available, otherwise default to 1.0
            keyword_score = result.get('_bm25_norm', 1.0)
            
            # Check if we already have this chunk (by text similarity)
            is_duplicate = False
            for existing in case_chunks[case_id]:
                if existing['chunk_text'][:100] == result['chunk_text'][:100]:
                    # Merge scores - use max BM25 score for duplicates
                    existing['keyword_score'] = max(existing['keyword_score'], keyword_score)
                    is_duplicate = True
                    break
            
            if not is_duplicate and len(case_chunks[case_id]) < 3:
                result['semantic_score'] = 0.0
                result['keyword_score'] = keyword_score
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
            with self._get_db_connection() as (cursor, conn):
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
                    cursor.execute("SELECT name, role FROM advocates WHERE judgment_id = ?", (case_id,))
                    judgment['advocates'] = [{"name": row[0], "role": row[1]} for row in cursor.fetchall()]
                except sqlite3.OperationalError:
                    judgment['advocates'] = []
                
                # Get cited laws (if table exists)
                try:
                    cursor.execute("SELECT law_name, section FROM laws WHERE judgment_id = ?", (case_id,))
                    judgment['laws'] = [{"law": row[0], "section": row[1]} for row in cursor.fetchall()]
                except sqlite3.OperationalError:
                    judgment['laws'] = []
            
            return judgment
            
        except Exception as e:
            logger.error(f"❌ Failed to get full judgment for case_id {case_id}: {e}")
            return {}
    
    def _organize_fused_results(self, fused_results: List[Dict], top_k: int) -> List[Dict]:
        """
        Organize RRF-fused results into case-level format with chunks
        
        Args:
            fused_results: RRF-fused results with rrf_score
            top_k: Number of top cases to return
            
        Returns:
            List of case-level results
        """
        case_chunks = {}
        
        for result in fused_results:
            case_id = result.get('case_id')
            if case_id is None:
                continue
            
            if case_id not in case_chunks:
                case_chunks[case_id] = []
            
            # Keep up to 3 chunks per case
            if len(case_chunks[case_id]) < 3:
                case_chunks[case_id].append(result)
        
        # Build case-level results
        case_results = []
        for case_id, chunks in case_chunks.items():
            if not chunks:
                continue
            
            # Use best chunk for case metadata
            best_chunk = chunks[0]
            
            # Average RRF score for case-level score
            avg_rrf_score = sum(c.get('rrf_score', 0) for c in chunks) / len(chunks)
            
            case_result = {
                "case_id": case_id,
                "case_number": best_chunk.get('case_number', 'Unknown'),
                "case_type": best_chunk.get('case_type', 'Unknown'),
                "judgment_date": best_chunk.get('judgment_date', 'Unknown'),
                "petitioner": best_chunk.get('petitioner', 'Unknown'),
                "respondent": best_chunk.get('respondent', 'Unknown'),
                "full_case_id": best_chunk.get('full_case_id', 'Unknown'),
                "chunks": chunks,
                "hybrid_score": avg_rrf_score,  # Use RRF score as hybrid score
                "chunk_count": len(chunks)
            }
            case_results.append(case_result)
        
        # Sort by hybrid score (RRF score)
        case_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        return case_results[:top_k]
    
    def invalidate_cache(self):
        """Clear all caches (useful after reindexing)"""
        if self.query_cache:
            try:
                self.query_cache.invalidate('*')
                logger.info("✅ Query cache cleared")
            except Exception as e:
                logger.error(f"Failed to clear query cache: {e}")
        
        if self.embedding_cache:
            try:
                self.embedding_cache.invalidate('*')
                logger.info("✅ Embedding cache cleared")
            except Exception as e:
                logger.error(f"Failed to clear embedding cache: {e}")


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
