"""
Cache Manager Module

Provides multi-level caching for query results, LLM responses, and embeddings.
Supports both in-memory and Redis backends.
"""

import json
import hashlib
import logging
import threading
from typing import Any, Dict, Optional
from datetime import datetime, timedelta
from collections import OrderedDict

# Import config
from config import (
    CACHE_ENABLED,
    CACHE_BACKEND,
    CACHE_TTL_QUERY,
    CACHE_TTL_LLM,
    CACHE_TTL_EMBEDDING,
    REDIS_URL
)

logger = logging.getLogger(__name__)


class CacheManager:
    """Base cache manager with support for in-memory and Redis backends"""
    
    def __init__(self, default_ttl: int = 3600, max_items: Optional[int] = None):
        """
        Initialize cache manager
        
        Args:
            default_ttl: Default time-to-live in seconds
            max_items: Maximum number of items in memory cache (LRU eviction when exceeded)
        """
        self.default_ttl = default_ttl
        self.max_items = max_items
        self.backend = CACHE_BACKEND
        self._lock = threading.Lock()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        
        if self.backend == 'memory':
            # In-memory cache: OrderedDict for LRU support with (value, expiry) tuples
            self._cache: OrderedDict[str, tuple] = OrderedDict()
            logger.info(f"Initialized in-memory cache (max_items={max_items})")
        elif self.backend == 'redis':
            try:
                import redis
                self._redis = redis.from_url(REDIS_URL, decode_responses=True)
                # Test connection
                self._redis.ping()
                logger.info(f"Initialized Redis cache: {REDIS_URL}")
            except Exception as e:
                logger.error(f"Failed to initialize Redis cache: {e}")
                logger.warning("Falling back to in-memory cache")
                self.backend = 'memory'
                self._cache = OrderedDict()
        else:
            logger.warning(f"Unknown cache backend '{self.backend}', using memory")
            self.backend = 'memory'
            self._cache = OrderedDict()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve cached value by key
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        try:
            if self.backend == 'memory':
                with self._lock:
                    if key in self._cache:
                        value, expiry = self._cache[key]
                        if expiry is None or datetime.now() < expiry:
                            # Move to end for LRU
                            self._cache.move_to_end(key)
                            self._hits += 1
                            logger.debug(f"Cache hit: {key}")
                            return value
                        else:
                            # Expired
                            del self._cache[key]
                    self._misses += 1
                    logger.debug(f"Cache miss: {key}")
                    return None
            else:  # redis
                value = self._redis.get(key)
                if value is not None:
                    self._hits += 1
                    logger.debug(f"Cache hit: {key}")
                    return json.loads(value)
                self._misses += 1
                logger.debug(f"Cache miss: {key}")
                return None
        except Exception as e:
            logger.error(f"Cache get error for key '{key}': {e}")
            return None
    
    def _evict_lru_if_needed(self):
        """
        Evict least recently used item if cache exceeds max_items.
        Must be called with lock held.
        """
        if self.max_items and len(self._cache) >= self.max_items:
            # Remove oldest item (first item in OrderedDict)
            oldest_key, _ = self._cache.popitem(last=False)
            logger.debug(f"LRU eviction: {oldest_key}")
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Store value with optional TTL
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None for default)
        """
        try:
            ttl = ttl or self.default_ttl
            
            if self.backend == 'memory':
                with self._lock:
                    # Evict LRU item if needed before adding new one
                    self._evict_lru_if_needed()
                    expiry = datetime.now() + timedelta(seconds=ttl) if ttl else None
                    self._cache[key] = (value, expiry)
                logger.debug(f"Cached: {key} (TTL: {ttl}s)")
            else:  # redis
                serialized = json.dumps(value)
                self._redis.setex(key, ttl, serialized)
                logger.debug(f"Cached in Redis: {key} (TTL: {ttl}s)")
        except Exception as e:
            logger.error(f"Cache set error for key '{key}': {e}")
    
    def invalidate(self, pattern: str):
        """
        Clear cache entries matching pattern
        
        Args:
            pattern: Pattern to match (supports '*' wildcard)
            
        Note:
            WARNING: Using pattern='*' will clear ALL cache entries.
            Use sparingly in production as it affects all cached data.
        """
        try:
            if self.backend == 'memory':
                with self._lock:
                    if pattern == '*':
                        count = len(self._cache)
                        self._cache.clear()
                        logger.info(f"Cleared all cache entries ({count} items)")
                    else:
                        # Simple pattern matching for memory cache
                        keys_to_delete = [k for k in self._cache.keys() if self._match_pattern(k, pattern)]
                        for key in keys_to_delete:
                            del self._cache[key]
                        logger.info(f"Invalidated {len(keys_to_delete)} cache entries matching '{pattern}'")
            else:  # redis
                if pattern == '*':
                    logger.warning("Clearing ALL Redis cache entries - this is destructive!")
                    self._redis.flushdb()
                    logger.info("Cleared all Redis cache entries")
                else:
                    # Use SCAN instead of KEYS for better performance at scale
                    deleted_count = 0
                    cursor = 0  # SCAN cursor must be integer, not string
                    max_iterations = 10000  # Safety guard to prevent infinite loops
                    iterations = 0
                    
                    while iterations < max_iterations:
                        cursor, keys = self._redis.scan(cursor=cursor, match=pattern, count=1000)
                        if keys:
                            self._redis.delete(*keys)
                            deleted_count += len(keys)
                        iterations += 1
                        if cursor == 0:  # SCAN returns int 0 when complete
                            break
                    
                    if iterations >= max_iterations:
                        logger.warning(f"Redis SCAN hit max iterations ({max_iterations}), may not have completed")
                    
                    logger.info(f"Invalidated {deleted_count} Redis cache entries matching '{pattern}'")
        except Exception as e:
            logger.error(f"Cache invalidation error for pattern '{pattern}': {e}")
    
    def get_stats(self) -> Dict:
        """
        Return cache statistics
        
        Returns:
            Dict with hits, misses, hit_rate, size
        """
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0.0
        
        if self.backend == 'memory':
            size = len(self._cache)
        else:
            try:
                size = self._redis.dbsize()
            except:
                size = 0
        
        return {
            'backend': self.backend,
            'hits': self._hits,
            'misses': self._misses,
            'total_requests': total,
            'hit_rate': round(hit_rate, 2),
            'size': size
        }
    
    @staticmethod
    def _match_pattern(text: str, pattern: str) -> bool:
        """Simple wildcard pattern matching"""
        if pattern == '*':
            return True
        if '*' in pattern:
            parts = pattern.split('*')
            if not text.startswith(parts[0]):
                return False
            if not text.endswith(parts[-1]):
                return False
            return True
        return text == pattern


class QueryCache(CacheManager):
    """Specialized cache for search results"""
    
    def __init__(self):
        super().__init__(default_ttl=CACHE_TTL_QUERY)
    
    def cache_search_results(self, query: str, filters: Dict, results: list, ttl: Optional[int] = None):
        """
        Cache search results
        
        Args:
            query: Search query
            filters: Search filters
            results: Search results
            ttl: Optional TTL override
        """
        cache_key = self._generate_search_key(query, filters)
        self.set(cache_key, results, ttl)
    
    def get_cached_search(self, query: str, filters: Dict) -> Optional[list]:
        """
        Get cached search results
        
        Args:
            query: Search query
            filters: Search filters
            
        Returns:
            Cached results or None
        """
        cache_key = self._generate_search_key(query, filters)
        return self.get(cache_key)
    
    @staticmethod
    def _generate_search_key(query: str, filters: Dict) -> str:
        """Generate cache key from query and filters"""
        # Normalize query
        normalized_query = query.lower().strip()
        # Sort filters for consistent hashing
        filters_str = json.dumps(filters, sort_keys=True)
        combined = f"{normalized_query}:{filters_str}"
        key_hash = hashlib.md5(combined.encode()).hexdigest()
        return f"search:{key_hash}"


class LLMResponseCache(CacheManager):
    """Specialized cache for LLM responses"""
    
    def __init__(self):
        super().__init__(default_ttl=CACHE_TTL_LLM)
    
    def cache_llm_response(self, query: str, context_hash: str, response: str, ttl: Optional[int] = None):
        """
        Cache LLM response
        
        Args:
            query: User query
            context_hash: Hash of context used
            response: LLM response
            ttl: Optional TTL override
        """
        cache_key = self._generate_llm_key(query, context_hash)
        self.set(cache_key, response, ttl)
    
    def get_cached_llm_response(self, query: str, context_hash: str) -> Optional[str]:
        """
        Get cached LLM response
        
        Args:
            query: User query
            context_hash: Hash of context used
            
        Returns:
            Cached response or None
        """
        cache_key = self._generate_llm_key(query, context_hash)
        return self.get(cache_key)
    
    @staticmethod
    def _generate_llm_key(query: str, context_hash: str) -> str:
        """Generate cache key from query and context hash"""
        # Normalize query (lowercase, strip whitespace)
        normalized_query = query.lower().strip()
        combined = f"{normalized_query}:{context_hash}"
        key_hash = hashlib.md5(combined.encode()).hexdigest()
        return f"llm:{key_hash}"


class EmbeddingCache(CacheManager):
    """Specialized cache for query embeddings"""
    
    def __init__(self):
        super().__init__(default_ttl=CACHE_TTL_EMBEDDING)
    
    def cache_embedding(self, text: str, embedding: list, ttl: Optional[int] = None):
        """
        Cache embedding
        
        Args:
            text: Text that was embedded
            embedding: Embedding vector
            ttl: Optional TTL override
        """
        cache_key = self._generate_embedding_key(text)
        self.set(cache_key, embedding, ttl)
    
    def get_cached_embedding(self, text: str) -> Optional[list]:
        """
        Get cached embedding
        
        Args:
            text: Text to get embedding for
            
        Returns:
            Cached embedding or None
        """
        cache_key = self._generate_embedding_key(text)
        return self.get(cache_key)
    
    @staticmethod
    def _generate_embedding_key(text: str) -> str:
        """Generate cache key from text"""
        # Normalize text
        normalized_text = text.lower().strip()
        text_hash = hashlib.md5(normalized_text.encode()).hexdigest()
        return f"embedding:{text_hash}"


# Module-level function for easy access
def get_cache_enabled() -> bool:
    """Check if caching is enabled"""
    return CACHE_ENABLED


if __name__ == '__main__':
    # Test cache functionality
    logging.basicConfig(level=logging.DEBUG)
    
    print("Testing CacheManager...")
    cache = CacheManager(default_ttl=5)
    
    # Test set and get
    cache.set('test_key', {'data': 'test_value'})
    result = cache.get('test_key')
    print(f"Get result: {result}")
    
    # Test stats
    stats = cache.get_stats()
    print(f"Cache stats: {stats}")
    
    # Test invalidation
    cache.invalidate('*')
    result = cache.get('test_key')
    print(f"After invalidation: {result}")
    
    print("\nTesting QueryCache...")
    query_cache = QueryCache()
    query_cache.cache_search_results("test query", {"limit": 10}, [{"id": 1}])
    cached = query_cache.get_cached_search("test query", {"limit": 10})
    print(f"Cached search result: {cached}")
    
    print("\nCache tests completed!")
