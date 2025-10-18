"""
Unit tests for GeminiClient.

Tests the Gemini API client implementation including:
- Initialization and configuration
- Prompt combination (system + user)
- Response generation
- Streaming behavior
- Safety filter handling
- Error handling and retries
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestGeminiClientInitialization:
    """Test GeminiClient initialization."""
    
    @patch('llm_client.genai')
    def test_init_with_api_key(self, mock_genai):
        """Test initialization with API key."""
        from llm_client import GeminiClient
        
        client = GeminiClient(api_key="test-key", model="gemini-pro")
        
        assert client.api_key == "test-key"
        assert client.model == "gemini-pro"
        mock_genai.configure.assert_called_once_with(api_key="test-key")
    
    @patch('llm_client.genai', None)
    def test_init_without_sdk(self):
        """Test initialization fails without SDK installed."""
        from llm_client import GeminiClient
        
        with pytest.raises(ImportError) as exc_info:
            GeminiClient(api_key="test-key")
        
        assert "google-generativeai package is required" in str(exc_info.value)
    
    @patch('llm_client.genai')
    @patch('llm_client.GEMINI_API_KEY', None)
    def test_init_without_api_key(self, mock_genai):
        """Test initialization fails without API key."""
        from llm_client import GeminiClient
        
        with pytest.raises(ValueError) as exc_info:
            GeminiClient()
        
        assert "Gemini API key is required" in str(exc_info.value)


class TestGeminiClientGenerate:
    """Test GeminiClient.generate() method."""
    
    @patch('llm_client.genai')
    def test_generate_success(self, mock_genai):
        """Test successful response generation."""
        from llm_client import GeminiClient
        
        # Setup mock
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "This is a test response."
        mock_response.prompt_feedback = Mock()
        mock_response.prompt_feedback.block_reason = None
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        # Test
        client = GeminiClient(api_key="test-key", model="gemini-pro")
        result = client.generate("What is a writ?", "You are a legal assistant.")
        
        # Assertions
        assert result == "This is a test response."
        mock_genai.GenerativeModel.assert_called_with("gemini-pro")
        mock_model.generate_content.assert_called_once()
        
        # Verify prompt combination
        call_args = mock_model.generate_content.call_args
        prompt_arg = call_args[0][0]
        assert "You are a legal assistant." in prompt_arg
        assert "What is a writ?" in prompt_arg
        assert "User Query:" in prompt_arg
    
    @patch('llm_client.genai')
    def test_generate_with_custom_params(self, mock_genai):
        """Test generation with custom temperature and max_tokens."""
        from llm_client import GeminiClient
        
        # Setup mock
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Response"
        mock_response.prompt_feedback = Mock()
        mock_response.prompt_feedback.block_reason = None
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        # Test
        client = GeminiClient(api_key="test-key")
        result = client.generate("Test", "System", temperature=0.7, max_tokens=1000)
        
        # Verify generation_config is a dict
        call_args = mock_model.generate_content.call_args
        config = call_args[1]['generation_config']
        assert isinstance(config, dict)
        assert config['temperature'] == 0.7
        assert config['max_output_tokens'] == 1000
    
    @patch('llm_client.genai')
    def test_generate_safety_block(self, mock_genai):
        """Test handling of safety filter blocks."""
        from llm_client import GeminiClient
        
        # Setup mock with safety block
        mock_model = Mock()
        mock_response = Mock()
        mock_response.prompt_feedback = Mock()
        mock_response.prompt_feedback.block_reason = "SAFETY"
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        # Test
        client = GeminiClient(api_key="test-key")
        result = client.generate("Harmful query", "System")
        
        # Should return safety message
        assert "cannot provide a response" in result
        assert "content safety restrictions" in result


class TestGeminiClientGenerateStream:
    """Test GeminiClient.generate_stream() method."""
    
    @patch('llm_client.genai')
    def test_generate_stream_success(self, mock_genai):
        """Test successful streaming generation."""
        from llm_client import GeminiClient
        
        # Setup mock with streaming chunks
        mock_model = Mock()
        
        # Create mock chunks
        chunk1 = Mock()
        chunk1.text = "First "
        chunk2 = Mock()
        chunk2.text = "second "
        chunk3 = Mock()
        chunk3.text = "third."
        
        # Mock response with context manager support
        mock_response = Mock()
        mock_response.__iter__ = Mock(return_value=iter([chunk1, chunk2, chunk3]))
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_response.resolve = Mock()
        mock_response.prompt_feedback = Mock()
        mock_response.prompt_feedback.block_reason = None
        
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        # Test
        client = GeminiClient(api_key="test-key")
        chunks = list(client.generate_stream("Test query", "System prompt"))
        
        # Assertions
        assert chunks == ["First ", "second ", "third."]
        mock_model.generate_content.assert_called_once()
        mock_response.resolve.assert_called_once()
        
        # Verify streaming config
        call_args = mock_model.generate_content.call_args
        assert call_args[1]['stream'] is True
        assert isinstance(call_args[1]['generation_config'], dict)
    
    @patch('llm_client.genai')
    def test_generate_stream_with_empty_chunks(self, mock_genai):
        """Test streaming handles chunks without text attribute."""
        from llm_client import GeminiClient
        
        # Setup mock with some empty chunks
        mock_model = Mock()
        
        chunk1 = Mock()
        chunk1.text = "Valid"
        chunk2 = Mock(spec=[])  # No text attribute
        chunk3 = Mock()
        chunk3.text = None  # text is None
        chunk4 = Mock()
        chunk4.text = "Text"
        
        # Mock response with context manager support
        mock_response = Mock()
        mock_response.__iter__ = Mock(return_value=iter([chunk1, chunk2, chunk3, chunk4]))
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_response.resolve = Mock()
        mock_response.prompt_feedback = Mock()
        mock_response.prompt_feedback.block_reason = None
        
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        # Test
        client = GeminiClient(api_key="test-key")
        chunks = list(client.generate_stream("Test", "System"))
        
        # Should only yield valid chunks
        assert chunks == ["Valid", "Text"]
        mock_response.resolve.assert_called_once()
    
    @patch('llm_client.genai')
    def test_generate_stream_safety_block(self, mock_genai):
        """Test streaming handles safety blocks with no content."""
        from llm_client import GeminiClient
        
        # Setup mock with blocked response (no chunks)
        mock_model = Mock()
        
        # Mock response with context manager support
        mock_response = Mock()
        mock_response.__iter__ = Mock(return_value=iter([]))  # No chunks
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_response.resolve = Mock()
        mock_response.prompt_feedback = Mock()
        mock_response.prompt_feedback.block_reason = "SAFETY"
        
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        # Test
        client = GeminiClient(api_key="test-key")
        chunks = list(client.generate_stream("Harmful query", "System"))
        
        # Should yield safety message
        assert len(chunks) == 1
        assert "cannot provide a response" in chunks[0]
        assert "content safety restrictions" in chunks[0]
        mock_response.resolve.assert_called_once()


class TestGeminiClientErrorHandling:
    """Test GeminiClient error handling and retries."""
    
    @patch('llm_client.genai')
    @patch('llm_client.google_exceptions')
    def test_retry_on_resource_exhausted(self, mock_google_exc, mock_genai):
        """Test retry on quota exceeded error."""
        from llm_client import GeminiClient
        
        # Setup mock that fails twice then succeeds
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Success"
        mock_response.prompt_feedback = Mock()
        mock_response.prompt_feedback.block_reason = None
        
        # Create retriable exception
        mock_quota_error = Mock(spec=Exception)
        mock_google_exc.ResourceExhausted = type(mock_quota_error)
        
        mock_model.generate_content.side_effect = [
            mock_google_exc.ResourceExhausted("Quota exceeded"),
            mock_response
        ]
        mock_genai.GenerativeModel.return_value = mock_model
        
        # Test
        client = GeminiClient(api_key="test-key")
        
        with patch('time.sleep'):  # Skip actual sleep
            result = client.generate("Test", "System")
        
        # Should succeed after retry
        assert result == "Success"
        assert mock_model.generate_content.call_count == 2
    
    @patch('llm_client.genai')
    def test_fail_fast_on_invalid_request(self, mock_genai):
        """Test immediate failure on invalid request errors."""
        from llm_client import GeminiClient
        
        # Setup mock that raises invalid error
        mock_model = Mock()
        mock_model.generate_content.side_effect = ValueError("Invalid API key")
        mock_genai.GenerativeModel.return_value = mock_model
        
        # Test
        client = GeminiClient(api_key="test-key")
        
        with pytest.raises(ValueError) as exc_info:
            client.generate("Test", "System")
        
        assert "Invalid API key" in str(exc_info.value)
        # Should fail fast, not retry
        assert mock_model.generate_content.call_count == 1
    
    @patch('llm_client.genai')
    @patch('llm_client.google_exceptions')
    def test_fail_fast_on_invalid_argument(self, mock_google_exc, mock_genai):
        """Test immediate failure on InvalidArgument (non-retriable Gemini error)."""
        from llm_client import GeminiClient
        
        # Setup mock that raises InvalidArgument
        mock_model = Mock()
        
        # Create InvalidArgument exception
        mock_invalid_arg_error = Mock(spec=Exception)
        mock_google_exc.InvalidArgument = type(mock_invalid_arg_error)
        
        mock_model.generate_content.side_effect = mock_google_exc.InvalidArgument("Invalid request format")
        mock_genai.GenerativeModel.return_value = mock_model
        
        # Test
        client = GeminiClient(api_key="test-key")
        
        with pytest.raises(Exception) as exc_info:
            client.generate("Test", "System")
        
        assert "Invalid request format" in str(exc_info.value)
        # Should fail immediately without retries
        assert mock_model.generate_content.call_count == 1


class TestGeminiClientIntegration:
    """Integration tests for GeminiClient."""
    
    @patch('llm_client.genai')
    def test_full_workflow(self, mock_genai):
        """Test complete workflow from initialization to response."""
        from llm_client import GeminiClient
        
        # Setup
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Article 102 deals with writ petitions in Bangladesh."
        mock_response.prompt_feedback = Mock()
        mock_response.prompt_feedback.block_reason = None
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        # Test full workflow
        client = GeminiClient(api_key="test-gemini-key", model="gemini-1.5-pro")
        
        system_prompt = "You are a legal expert on Bangladesh law."
        user_query = "What is Article 102?"
        
        response = client.generate(user_query, system_prompt)
        
        # Verify
        assert "Article 102" in response
        assert client.model == "gemini-1.5-pro"
        
        # Check prompt was properly combined
        call_args = mock_model.generate_content.call_args[0][0]
        assert system_prompt in call_args
        assert user_query in call_args


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
