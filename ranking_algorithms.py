"""
Advanced Ranking Algorithms Module

Implements BM25 (Best Matching 25) and Reciprocal Rank Fusion (RRF)
for improved search result ranking.

NOTE: Uses custom BM25 implementation (not rank-bm25 library).
- Custom implementation provides full control and no external dependency
- rank-bm25 library commented out in requirements_rag.txt
- To use library version, replace BM25Ranker with:
  from rank_bm25 import BM25Okapi
"""

import math
import logging
from typing import List, Dict
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)


class BM25Ranker:
    """
    BM25 (Best Matching 25) ranking algorithm for keyword search
    
    BM25 is a probabilistic ranking function that considers term frequency,
    inverse document frequency, and document length normalization.
    """
    
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 ranker
        
        Args:
            corpus: List of documents (strings) for building statistics
            k1: Term frequency saturation parameter (typical: 1.2-2.0)
            b: Length normalization parameter (0.0-1.0, typical: 0.75)
        """
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.corpus_size = len(corpus)
        
        # Calculate document frequencies and average document length
        self.doc_freqs = defaultdict(int)
        self.idf = {}
        self.doc_lengths = []
        self.avgdl = 0
        
        if self.corpus_size > 0:
            self._build_statistics()
        
        logger.info(f"BM25Ranker initialized with {self.corpus_size} documents (k1={k1}, b={b})")
    
    def _build_statistics(self):
        """Build document frequency statistics from corpus"""
        total_length = 0
        
        for doc in self.corpus:
            tokens = self._tokenize(doc)
            self.doc_lengths.append(len(tokens))
            total_length += len(tokens)
            
            # Count unique terms in document
            unique_terms = set(tokens)
            for term in unique_terms:
                self.doc_freqs[term] += 1
        
        # Calculate average document length
        self.avgdl = total_length / self.corpus_size if self.corpus_size > 0 else 0
        
        # Calculate IDF for each term
        for term, freq in self.doc_freqs.items():
            # IDF = log((N - df + 0.5) / (df + 0.5) + 1)
            idf = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)
            self.idf[term] = idf
        
        logger.debug(f"Built BM25 statistics: avgdl={self.avgdl:.2f}, unique_terms={len(self.idf)}")
    
    def score(self, query: str, document: str) -> float:
        """
        Calculate BM25 score for a document given a query
        
        Args:
            query: Query string
            document: Document string
            
        Returns:
            BM25 score (higher is better)
        """
        # Early return when avgdl is 0 (empty corpus or no tokens)
        if self.avgdl == 0:
            return 0.0
        
        query_tokens = self._tokenize(query)
        doc_tokens = self._tokenize(document)
        doc_length = len(doc_tokens)
        
        # Early return when document length is 0
        if doc_length == 0:
            return 0.0
        
        # Count term frequencies in document
        doc_term_freqs = Counter(doc_tokens)
        
        score = 0.0
        for term in query_tokens:
            if term not in self.idf:
                # Term not in corpus, skip
                continue
            
            # Term frequency in document
            tf = doc_term_freqs.get(term, 0)
            
            # IDF score
            idf = self.idf[term]
            
            # BM25 formula:
            # score += IDF(qi) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * |D| / avgdl))
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avgdl)
            
            # Skip if denominator would be zero (shouldn't happen with proper avgdl check, but defensive)
            if denominator == 0:
                continue
            
            score += idf * (numerator / denominator)
        
        return score
    
    def rank_documents(self, query: str, documents: List[Dict]) -> List[Dict]:
        """
        Rank documents by BM25 score
        
        Args:
            query: Search query
            documents: List of document dicts (must have 'full_text' or 'chunk_text' field)
            
        Returns:
            Documents sorted by BM25 score (descending) with 'bm25_score' field added
        """
        if not documents:
            return []
        
        # Score each document
        for doc in documents:
            # Try to get text from common fields
            text = doc.get('full_text') or doc.get('chunk_text') or doc.get('text') or ''
            doc['bm25_score'] = self.score(query, str(text))
        
        # Sort by BM25 score descending
        ranked = sorted(documents, key=lambda x: x.get('bm25_score', 0), reverse=True)
        
        logger.debug(f"Ranked {len(ranked)} documents by BM25")
        return ranked
    
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Simple tokenization (lowercase and split on whitespace/punctuation)
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Lowercase and simple split
        # Remove common punctuation
        import re
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        return tokens


class ReciprocalRankFusion:
    """
    Reciprocal Rank Fusion (RRF) for combining multiple ranked lists
    
    RRF assigns scores based on reciprocal ranks: score = 1 / (k + rank)
    where k is a constant (typically 60) and rank is 1-indexed position.
    """
    
    def __init__(self, k: int = 60):
        """
        Initialize RRF
        
        Args:
            k: RRF constant (default: 60, typical range: 30-100)
        """
        self.k = k
        logger.info(f"ReciprocalRankFusion initialized (k={k})")
    
    def fuse(self, ranked_lists: List[List[Dict]], id_field: str = 'id') -> List[Dict]:
        """
        Fuse multiple ranked lists using RRF
        
        Args:
            ranked_lists: List of ranked document lists
            id_field: Field name for document ID
            
        Returns:
            Fused ranking sorted by RRF score
        """
        # Calculate RRF scores for each document
        rrf_scores = defaultdict(float)
        doc_map = {}  # Store document objects
        
        for ranked_list in ranked_lists:
            for rank, doc in enumerate(ranked_list, start=1):
                doc_id = doc.get(id_field)
                if doc_id is None:
                    continue
                
                # RRF score: 1 / (k + rank)
                rrf_score = 1.0 / (self.k + rank)
                rrf_scores[doc_id] += rrf_score
                
                # Store document (use first occurrence)
                if doc_id not in doc_map:
                    doc_map[doc_id] = doc
        
        # Create result list with RRF scores
        fused_results = []
        for doc_id, score in rrf_scores.items():
            doc = doc_map[doc_id].copy()
            doc['rrf_score'] = score
            fused_results.append(doc)
        
        # Sort by RRF score descending
        fused_results.sort(key=lambda x: x['rrf_score'], reverse=True)
        
        logger.debug(f"Fused {len(ranked_lists)} lists into {len(fused_results)} unique documents")
        return fused_results
    
    def fuse_with_weights(self, ranked_lists: List[List[Dict]], weights: List[float],
                          id_field: str = 'id') -> List[Dict]:
        """
        Fuse multiple ranked lists with weights
        
        Args:
            ranked_lists: List of ranked document lists
            weights: Weight for each list (must match length of ranked_lists)
            id_field: Field name for document ID
            
        Returns:
            Fused ranking sorted by weighted RRF score
        """
        if len(ranked_lists) != len(weights):
            logger.error(f"Mismatch: {len(ranked_lists)} lists but {len(weights)} weights")
            # Fallback to equal weights
            weights = [1.0] * len(ranked_lists)
        
        # Normalize weights to sum to 1.0
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(weights)] * len(weights)
        
        # Calculate weighted RRF scores
        rrf_scores = defaultdict(float)
        doc_map = {}
        
        for ranked_list, weight in zip(ranked_lists, weights):
            for rank, doc in enumerate(ranked_list, start=1):
                doc_id = doc.get(id_field)
                if doc_id is None:
                    continue
                
                # Weighted RRF score: weight * (1 / (k + rank))
                rrf_score = weight * (1.0 / (self.k + rank))
                rrf_scores[doc_id] += rrf_score
                
                if doc_id not in doc_map:
                    doc_map[doc_id] = doc
        
        # Create result list
        fused_results = []
        for doc_id, score in rrf_scores.items():
            doc = doc_map[doc_id].copy()
            doc['rrf_score'] = score
            fused_results.append(doc)
        
        # Sort by weighted RRF score descending
        fused_results.sort(key=lambda x: x['rrf_score'], reverse=True)
        
        logger.debug(f"Fused {len(ranked_lists)} lists with weights {weights} into {len(fused_results)} documents")
        return fused_results


def normalize_scores(results: List[Dict], score_field: str) -> List[Dict]:
    """
    Normalize scores to [0, 1] range using min-max normalization
    
    Args:
        results: List of result dicts
        score_field: Name of score field to normalize
        
    Returns:
        Results with normalized scores
    """
    if not results:
        return results
    
    scores = [r.get(score_field, 0) for r in results]
    min_score = min(scores)
    max_score = max(scores)
    
    # Handle edge case: all scores are the same
    if max_score == min_score:
        for r in results:
            r[f'{score_field}_normalized'] = 1.0 if max_score > 0 else 0.0
    else:
        for r in results:
            score = r.get(score_field, 0)
            normalized = (score - min_score) / (max_score - min_score)
            r[f'{score_field}_normalized'] = normalized
    
    return results


def combine_scores(results: List[Dict], score_fields: List[str], weights: List[float]) -> List[Dict]:
    """
    Combine multiple score fields with weights
    
    Args:
        results: List of result dicts
        score_fields: Names of score fields to combine
        weights: Weight for each score field
        
    Returns:
        Results with 'combined_score' field, sorted by combined score
    """
    if len(score_fields) != len(weights):
        logger.error(f"Mismatch: {len(score_fields)} fields but {len(weights)} weights")
        weights = [1.0 / len(score_fields)] * len(score_fields)
    
    # Normalize weights
    total_weight = sum(weights)
    if total_weight > 0:
        weights = [w / total_weight for w in weights]
    else:
        weights = [1.0 / len(weights)] * len(weights)
    
    # Calculate combined scores
    for result in results:
        combined = 0.0
        for field, weight in zip(score_fields, weights):
            score = result.get(field, 0)
            combined += weight * score
        result['combined_score'] = combined
    
    # Sort by combined score
    results.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
    
    return results


if __name__ == '__main__':
    # Test BM25
    logging.basicConfig(level=logging.DEBUG)
    
    print("Testing BM25Ranker...")
    corpus = [
        "The quick brown fox jumps over the lazy dog",
        "A quick brown dog outpaces a quick fox",
        "The dog is lazy but the fox is quick"
    ]
    
    bm25 = BM25Ranker(corpus, k1=1.5, b=0.75)
    
    query = "quick fox"
    documents = [
        {'id': 1, 'full_text': corpus[0]},
        {'id': 2, 'full_text': corpus[1]},
        {'id': 3, 'full_text': corpus[2]}
    ]
    
    ranked = bm25.rank_documents(query, documents)
    print(f"\nQuery: '{query}'")
    for doc in ranked:
        print(f"  ID {doc['id']}: score={doc['bm25_score']:.4f}")
    
    print("\n\nTesting ReciprocalRankFusion...")
    rrf = ReciprocalRankFusion(k=60)
    
    # Simulate two different ranked lists
    list1 = [{'id': 1, 'score': 10}, {'id': 2, 'score': 8}, {'id': 3, 'score': 6}]
    list2 = [{'id': 2, 'score': 10}, {'id': 3, 'score': 9}, {'id': 1, 'score': 5}]
    
    fused = rrf.fuse([list1, list2])
    print("\nFused results:")
    for doc in fused:
        print(f"  ID {doc['id']}: rrf_score={doc['rrf_score']:.4f}")
    
    # Test weighted fusion
    fused_weighted = rrf.fuse_with_weights([list1, list2], weights=[0.7, 0.3])
    print("\nWeighted fused results (0.7, 0.3):")
    for doc in fused_weighted:
        print(f"  ID {doc['id']}: rrf_score={doc['rrf_score']:.4f}")
    
    print("\nRanking tests completed!")
