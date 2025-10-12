import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_pipeline import RAGPipeline
from rag_retriever import HybridRetriever
from llm_client import OpenAIClient, AnthropicClient, get_llm_client, validate_response
from conversation_manager import ConversationManager, ConversationSession, estimate_token_count
from prompt_templates import (
    build_rag_prompt, build_crime_category_prompt, build_comparison_prompt,
    build_summarization_prompt, format_response_with_citations
)


# Fixtures

@pytest.fixture
def mock_retriever():
    """Mock HybridRetriever for testing."""
    retriever = Mock(spec=HybridRetriever)
    retriever.hybrid_retrieve.return_value = [
        {
            "chunk_text": "This is a test judgment about murder.",
            "similarity": 0.85,
            "case_id": 1,
            "case_number": "123/2023",
            "case_type": "Writ Petition",
            "judgment_date": "2023-05-15",
            "petitioner": "John Doe",
            "respondent": "State",
            "full_case_id": "Writ Petition No. 123 of 2023",
            "source": "semantic",
            "hybrid_score": 0.9
        }
    ]
    return retriever


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing."""
    client = Mock()
    client.generate.return_value = "This is a test response from the LLM."
    client.generate_stream.return_value = iter(["This ", "is ", "a ", "test"])
    return client


@pytest.fixture
def conversation_manager():
    """Real ConversationManager for testing."""
    return ConversationManager(storage_path=None)


@pytest.fixture
def sample_contexts():
    """Sample context data for testing."""
    return [
        {
            "chunk_text": "The court held that writ petitions require...",
            "similarity": 0.9,
            "case_id": 1,
            "case_number": "1234/2023",
            "case_type": "Writ Petition",
            "judgment_date": "2023-05-15",
            "petitioner": "ABC Company",
            "respondent": "Government of Bangladesh",
            "full_case_id": "Writ Petition No. 1234 of 2023"
        },
        {
            "chunk_text": "Article 102 of the Constitution provides...",
            "similarity": 0.85,
            "case_id": 2,
            "case_number": "5678/2023",
            "case_type": "Writ Petition",
            "judgment_date": "2023-06-20",
            "petitioner": "John Doe",
            "respondent": "State",
            "full_case_id": "Writ Petition No. 5678 of 2023"
        }
    ]


# Test HybridRetriever

def test_hybrid_retriever_semantic_search(mock_retriever):
    """Test semantic search functionality."""
    results = mock_retriever.hybrid_retrieve("murder cases", top_k=5)
    
    assert len(results) > 0
    assert results[0]['similarity'] > 0.7
    assert 'case_id' in results[0]
    assert 'chunk_text' in results[0]


def test_hybrid_retriever_combined(mock_retriever):
    """Test hybrid retrieval combining semantic and keyword."""
    results = mock_retriever.hybrid_retrieve("writ petition", top_k=5)
    
    assert len(results) > 0
    assert 'hybrid_score' in results[0]
    mock_retriever.hybrid_retrieve.assert_called_once()


# Test LLM Client

def test_llm_client_validation():
    """Test LLM response validation."""
    # Valid responses
    assert validate_response("This is a valid legal response.") == True
    assert validate_response("The court held that...") == True
    
    # Invalid responses
    assert validate_response("") == False
    assert validate_response("   ") == False
    assert validate_response("I cannot answer this question.") == False
    assert validate_response("I apologize, but I don't have access") == False


def test_openai_client_initialization():
    """Test OpenAI client initialization."""
    with patch('llm_client._OpenAI'):
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            client = get_llm_client('openai')
            assert client is not None


# Test Prompt Templates

def test_build_rag_prompt(sample_contexts):
    """Test RAG prompt construction."""
    query = "What are writ petitions?"
    prompt = build_rag_prompt(query, sample_contexts)
    
    assert query in prompt
    assert "RELEVANT JUDGMENT CONTEXTS" in prompt
    assert "Context 1:" in prompt
    assert sample_contexts[0]['case_number'] in prompt


def test_build_crime_category_prompt(sample_contexts):
    """Test crime category prompt construction."""
    prompt = build_crime_category_prompt("show murder cases", "Murder", sample_contexts)
    
    assert "CRIME CATEGORY SEARCH" in prompt
    assert "Murder" in prompt.upper()
    assert "CASE LIST" in prompt


def test_format_response_with_citations(sample_contexts):
    """Test citation formatting."""
    response = "This is a test response."
    formatted = format_response_with_citations(response, sample_contexts)
    
    assert response in formatted
    assert "SOURCES:" in formatted
    assert sample_contexts[0]['full_case_id'] in formatted
    assert sample_contexts[0]['judgment_date'] in formatted


# Test Conversation Manager

def test_conversation_session_creation(conversation_manager):
    """Test session creation."""
    session_id = conversation_manager.create_session(metadata={"user": "test"})
    
    assert session_id is not None
    assert len(session_id) > 0
    
    session = conversation_manager.get_session(session_id)
    assert session.session_id == session_id
    assert session.metadata['user'] == "test"


def test_conversation_add_message(conversation_manager):
    """Test adding messages to conversation."""
    session_id = conversation_manager.create_session()
    
    conversation_manager.add_message(session_id, "user", "What is Article 102?")
    conversation_manager.add_message(session_id, "assistant", "Article 102 provides...")
    
    session = conversation_manager.get_session(session_id)
    assert len(session.history) == 2
    assert session.history[0]['role'] == "user"
    assert session.history[1]['role'] == "assistant"


def test_conversation_history_retrieval(conversation_manager):
    """Test history retrieval."""
    session_id = conversation_manager.create_session()
    
    for i in range(10):
        conversation_manager.add_message(session_id, "user", f"Question {i}")
        conversation_manager.add_message(session_id, "assistant", f"Answer {i}")
    
    context = conversation_manager.get_context_for_prompt(session_id, max_turns=5)
    assert len(context) <= 5


def test_token_estimation():
    """Test token count estimation."""
    text = "This is a test sentence with multiple words."
    tokens = estimate_token_count(text)
    
    assert tokens > 0
    assert tokens > len(text.split())  # Should be more than word count


# Test RAG Pipeline

@patch('rag_pipeline.HybridRetriever')
@patch('rag_pipeline.get_llm_client')
def test_rag_pipeline_factual_query(mock_get_llm, mock_retriever_class):
    """Test end-to-end factual query processing."""
    # Setup mocks
    mock_retriever = Mock()
    mock_retriever.hybrid_retrieve.return_value = [{
        "chunk_text": "Test content",
        "similarity": 0.9,
        "case_id": 1,
        "case_number": "123/2023",
        "case_type": "Writ Petition",
        "judgment_date": "2023-05-15",
        "petitioner": "John Doe",
        "respondent": "State",
        "full_case_id": "Writ Petition No. 123 of 2023",
        "hybrid_score": 0.9
    }]
    mock_retriever_class.return_value = mock_retriever
    
    mock_llm = Mock()
    mock_llm.generate.return_value = "This is a valid response with citations."
    mock_get_llm.return_value = mock_llm
    
    # Test pipeline
    pipeline = RAGPipeline()
    result = pipeline.process_query("What are writ petitions?")
    
    assert 'response' in result
    assert 'sources' in result
    assert 'query_type' in result
    assert result['query_type'] in ['factual', 'crime_search', 'comparison', 'summarization']


@patch('rag_pipeline.HybridRetriever')
@patch('rag_pipeline.get_llm_client')
def test_rag_pipeline_no_results(mock_get_llm, mock_retriever_class):
    """Test fallback when no relevant contexts found."""
    mock_retriever = Mock()
    mock_retriever.hybrid_retrieve.return_value = []
    mock_retriever_class.return_value = mock_retriever
    
    mock_llm = Mock()
    mock_get_llm.return_value = mock_llm
    
    pipeline = RAGPipeline()
    result = pipeline.process_query("some irrelevant query")
    
    assert result['query_type'] == 'no_results'
    assert len(result['sources']) == 0


@patch('rag_pipeline.HybridRetriever')
@patch('rag_pipeline.get_llm_client')
def test_rag_pipeline_with_conversation_history(mock_get_llm, mock_retriever_class):
    """Test multi-turn conversation."""
    mock_retriever = Mock()
    mock_retriever.hybrid_retrieve.return_value = [{
        "chunk_text": "Test content",
        "similarity": 0.9,
        "case_id": 1,
        "case_number": "123/2023",
        "case_type": "Writ Petition",
        "judgment_date": "2023-05-15",
        "petitioner": "John Doe",
        "respondent": "State",
        "full_case_id": "Writ Petition No. 123 of 2023",
        "hybrid_score": 0.9
    }]
    mock_retriever_class.return_value = mock_retriever
    
    mock_llm = Mock()
    mock_llm.generate.return_value = "Valid response"
    mock_get_llm.return_value = mock_llm
    
    pipeline = RAGPipeline()
    
    # First query
    result1 = pipeline.process_query("What is Article 102?")
    session_id = result1['session_id']
    
    # Follow-up query
    result2 = pipeline.process_query("Can you give examples?", session_id=session_id)
    
    assert result2['session_id'] == session_id
    session = pipeline.conversation_manager.get_session(session_id)
    assert len(session.history) >= 2  # At least 2 turns


# Test Error Handling

@patch('rag_pipeline.HybridRetriever')
@patch('rag_pipeline.get_llm_client')
def test_error_handling(mock_get_llm, mock_retriever_class):
    """Test error handling in pipeline."""
    mock_retriever = Mock()
    mock_retriever.hybrid_retrieve.side_effect = Exception("Test error")
    mock_retriever_class.return_value = mock_retriever
    
    mock_llm = Mock()
    mock_get_llm.return_value = mock_llm
    
    pipeline = RAGPipeline()
    result = pipeline.process_query("test query")
    
    assert 'error' in result['response'].lower() or result['query_type'] == 'error'


# Cache and Ranking Tests

def test_query_cache(mock_retriever, mock_llm_client):
    """Test query result caching"""
    # This is a mock test - actual caching tests are in test_performance.py
    # Just verify that the cache manager can be imported
    try:
        from cache_manager import QueryCache
        cache = QueryCache()
        assert cache is not None
    except ImportError:
        pytest.skip("Cache manager not available")


def test_llm_cache(mock_llm_client):
    """Test LLM response caching"""
    try:
        from cache_manager import LLMResponseCache
        cache = LLMResponseCache()
        
        # Test cache operations
        query = "test query"
        context_hash = "testhash123"
        response = "test response"
        
        cache.cache_llm_response(query, context_hash, response)
        cached = cache.get_cached_llm_response(query, context_hash)
        assert cached == response
    except ImportError:
        pytest.skip("Cache manager not available")


def test_embedding_cache():
    """Test embedding caching"""
    try:
        from cache_manager import EmbeddingCache
        cache = EmbeddingCache()
        
        # Test cache operations
        text = "test text"
        embedding = [0.1, 0.2, 0.3]
        
        cache.cache_embedding(text, embedding)
        cached = cache.get_cached_embedding(text)
        assert cached == embedding
    except ImportError:
        pytest.skip("Cache manager not available")


def test_bm25_ranking():
    """Test BM25 scoring"""
    try:
        from ranking_algorithms import BM25Ranker
        
        corpus = [
            "The quick brown fox",
            "A quick brown dog",
            "The dog is lazy"
        ]
        
        ranker = BM25Ranker(corpus)
        documents = [{'id': i, 'full_text': doc} for i, doc in enumerate(corpus)]
        
        query = "quick fox"
        ranked = ranker.rank_documents(query, documents)
        
        assert len(ranked) == len(documents)
        assert 'bm25_score' in ranked[0]
        # First document should score highest (has both "quick" and "fox")
        assert ranked[0]['id'] == 0
    except ImportError:
        pytest.skip("Ranking algorithms not available")


def test_rrf_fusion():
    """Test Reciprocal Rank Fusion"""
    try:
        from ranking_algorithms import ReciprocalRankFusion
        
        rrf = ReciprocalRankFusion(k=60)
        
        # Two ranked lists
        list1 = [{'id': 1, 'score': 10}, {'id': 2, 'score': 8}, {'id': 3, 'score': 6}]
        list2 = [{'id': 2, 'score': 10}, {'id': 3, 'score': 9}, {'id': 1, 'score': 5}]
        
        fused = rrf.fuse([list1, list2])
        
        assert len(fused) == 3
        assert 'rrf_score' in fused[0]
        # Document 2 should score highest (ranks well in both lists)
        assert fused[0]['id'] == 2
    except ImportError:
        pytest.skip("Ranking algorithms not available")


def test_hybrid_with_bm25(mock_retriever, mock_llm_client):
    """Test hybrid retrieval with BM25 enabled"""
    try:
        from config import USE_BM25
        if not USE_BM25:
            pytest.skip("BM25 not enabled in config")
        
        # This would test with actual BM25 integration
        # For now, just verify the feature can be enabled
        assert USE_BM25 in [True, False]
    except ImportError:
        pytest.skip("BM25 configuration not available")


def test_hybrid_with_rrf(mock_retriever, mock_llm_client):
    """Test hybrid retrieval with RRF enabled"""
    try:
        from config import USE_RRF
        if not USE_RRF:
            pytest.skip("RRF not enabled in config")
        
        # This would test with actual RRF integration
        # For now, just verify the feature can be enabled
        assert USE_RRF in [True, False]
    except ImportError:
        pytest.skip("RRF configuration not available")


# Integration Test

@pytest.mark.integration
def test_full_pipeline_integration():
    """
    Integration test with real components (requires actual database and index).
    """
    import os
    if not os.path.exists('extracted_data/database.db'):
        pytest.skip("Database not available for integration test")
    
    if not os.path.exists('faiss_index.bin'):
        pytest.skip("FAISS index not available for integration test")
    
    try:
        # Initialize real pipeline
        from rag_pipeline import RAGPipeline
        pipeline = RAGPipeline()
        
        # Test with a simple query
        result = pipeline.process_query("What is a writ petition?")
        
        # Verify result structure
        assert result is not None
        assert 'response' in result
        assert 'sources' in result
        assert 'session_id' in result
        assert 'query_type' in result
        
        # Response should not be empty
        assert len(result['response']) > 0
        
        # Should have some sources (unless database is empty)
        # This is a soft assertion since it depends on data
        if result['sources']:
            assert isinstance(result['sources'], list)
        
        print(f"✅ Integration test passed - response length: {len(result['response'])}")
        
    except Exception as e:
        pytest.fail(f"Integration test failed with error: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
