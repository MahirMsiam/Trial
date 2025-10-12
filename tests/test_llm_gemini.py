"""
Unit tests for Gemini LLM client implementation.

Tests GeminiClient generate() and generate_stream() methods with mocked SDK.
"""
import os
import sys
import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Iterator

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestGeminiClientInitialization:
    """Test GeminiClient initialization and configuration."""
    
    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test-api-key', 'GEMINI_MODEL': 'gemini-1.5-pro'})
    @patch('llm_client.genai')
    def test_initialization_success(self, mock_genai):
        """Test successful GeminiClient initialization."""
        from llm_client import GeminiClient
        
        client = GeminiClient()
        
        assert client.api_key == 'test-api-key'
        assert client.model == 'gemini-1.5-pro'
        mock_genai.configure.assert_called_once_with(api_key='test-api-key')
    
    @patch('llm_client.GEMINI_API_KEY', None)
    @patch('llm_client.genai')
    def test_initialization_missing_api_key(self, mock_genai):
        """Test GeminiClient initialization fails without API key."""
        from llm_client import GeminiClient
        
        with pytest.raises(ValueError, match="Gemini API key is required"):
            GeminiClient()
    
    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test-key'})
    @patch('llm_client.genai', None)
    def test_initialization_missing_sdk(self):
        """Test GeminiClient initialization fails without SDK installed."""
        from llm_client import GeminiClient
        
        with pytest.raises(ImportError, match="google-generativeai package is required"):
            GeminiClient()


class TestGeminiClientGenerate:
    """Test GeminiClient.generate() method."""
    
    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test-api-key'})
    @patch('llm_client.genai')
    def test_generate_success(self, mock_genai):
        """Test successful response generation."""
        from llm_client import GeminiClient
        
        # Mock the response
        mock_response = Mock()
        mock_response.text = "This is a test response"
        mock_response.prompt_feedback = None
        
        # Mock the model
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        client = GeminiClient()
        result = client.generate("Test prompt", "Test system prompt")
        
        assert result == "This is a test response"
        mock_model.generate_content.assert_called_once()
    
    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test-api-key'})
    @patch('llm_client.genai')
    def test_generate_with_custom_params(self, mock_genai):
        """Test generation with custom temperature and max_tokens."""
        from llm_client import GeminiClient
        
        mock_response = Mock()
        mock_response.text = "Response with custom params"
        mock_response.prompt_feedback = None
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        client = GeminiClient()
        result = client.generate(
            "Test prompt", 
            "Test system prompt", 
            temperature=0.5, 
            max_tokens=500
        )
        
        assert result == "Response with custom params"
        # Verify custom params were passed in generation_config
        call_args = mock_model.generate_content.call_args
        assert call_args[1]['generation_config']['temperature'] == 0.5
        assert call_args[1]['generation_config']['max_output_tokens'] == 500
    
    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test-api-key'})
    @patch('llm_client.genai')
    def test_generate_safety_block(self, mock_genai):
        """Test handling of safety-blocked response."""
        from llm_client import GeminiClient
        
        # Mock a safety-blocked response
        mock_response = Mock()
        mock_response.text = ""
        mock_feedback = Mock()
        mock_feedback.block_reason = "SAFETY"
        mock_response.prompt_feedback = mock_feedback
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        client = GeminiClient()
        result = client.generate("Unsafe prompt", "Test system prompt")
        
        expected = "I apologize, but I cannot provide a response to this query due to content safety restrictions."
        assert result == expected
    
    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test-api-key'})
    @patch('llm_client.genai')
    def test_generate_text_extraction_fallback(self, mock_genai):
        """Test text extraction fallback when response.text is empty."""
        from llm_client import GeminiClient
        
        # Mock response with empty text but content in candidates
        mock_part = Mock()
        mock_part.text = "Fallback text from parts"
        
        mock_content = Mock()
        mock_content.parts = [mock_part]
        
        mock_candidate = Mock()
        mock_candidate.content = mock_content
        
        mock_response = Mock()
        mock_response.text = ""
        mock_response.prompt_feedback = None
        mock_response.candidates = [mock_candidate]
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        client = GeminiClient()
        result = client.generate("Test prompt", "Test system prompt")
        
        assert result == "Fallback text from parts"
    
    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test-api-key'})
    @patch('llm_client.genai')
    @patch('llm_client.google_exceptions')
    def test_generate_retry_on_resource_exhausted(self, mock_exceptions, mock_genai):
        """Test retry behavior on ResourceExhausted error."""
        from llm_client import GeminiClient
        
        # Create actual exception classes (not Mocks)
        class MockResourceExhausted(Exception):
            pass
        
        mock_exceptions.ResourceExhausted = MockResourceExhausted
        mock_exceptions.ServiceUnavailable = type('ServiceUnavailable', (Exception,), {})
        mock_exceptions.DeadlineExceeded = type('DeadlineExceeded', (Exception,), {})
        mock_exceptions.InternalServerError = type('InternalServerError', (Exception,), {})
        
        # Also mock non-retriable exceptions as types (getattr returns these)
        mock_exceptions.InvalidArgument = type('InvalidArgument', (Exception,), {})
        mock_exceptions.FailedPrecondition = type('FailedPrecondition', (Exception,), {})
        mock_exceptions.PermissionDenied = type('PermissionDenied', (Exception,), {})
        mock_exceptions.Unauthenticated = type('Unauthenticated', (Exception,), {})
        
        # First call raises ResourceExhausted, second call succeeds
        mock_response = Mock()
        mock_response.text = "Success after retry"
        mock_response.prompt_feedback = None
        
        mock_model = Mock()
        mock_model.generate_content.side_effect = [
            MockResourceExhausted("Rate limit exceeded"),
            mock_response
        ]
        mock_genai.GenerativeModel.return_value = mock_model
        
        client = GeminiClient()
        
        # Patch sleep to avoid delays in tests
        with patch('time.sleep'):
            result = client.generate("Test prompt", "Test system prompt")
        
        assert result == "Success after retry"
        assert mock_model.generate_content.call_count == 2
    
    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test-api-key'})
    @patch('llm_client.genai')
    @patch('llm_client.google_exceptions', None)
    def test_generate_fallback_retry_without_google_exceptions(self, mock_genai):
        """Test fallback retry pattern when google_exceptions is unavailable."""
        from llm_client import GeminiClient
        
        # First call raises TimeoutError, second call succeeds
        mock_response = Mock()
        mock_response.text = "Success after timeout retry"
        mock_response.prompt_feedback = None
        
        mock_model = Mock()
        mock_model.generate_content.side_effect = [
            TimeoutError("Connection timeout"),
            mock_response
        ]
        mock_genai.GenerativeModel.return_value = mock_model
        
        client = GeminiClient()
        
        with patch('time.sleep'):
            result = client.generate("Test prompt", "Test system prompt")
        
        assert result == "Success after timeout retry"
        assert mock_model.generate_content.call_count == 2
    
    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test-api-key'})
    @patch('llm_client.genai')
    def test_generate_retriable_pattern_without_exceptions(self, mock_genai):
        """Test retriable pattern matching when exception type matching fails."""
        from llm_client import GeminiClient
        
        # First call raises exception with retriable message, second succeeds
        mock_response = Mock()
        mock_response.text = "Success after quota retry"
        mock_response.prompt_feedback = None
        
        mock_model = Mock()
        mock_model.generate_content.side_effect = [
            Exception("Quota exceeded for this resource"),
            mock_response
        ]
        mock_genai.GenerativeModel.return_value = mock_model
        
        client = GeminiClient()
        
        with patch('time.sleep'):
            result = client.generate("Test prompt", "Test system prompt")
        
        assert result == "Success after quota retry"
        assert mock_model.generate_content.call_count == 2


class TestGeminiClientGenerateStream:
    """Test GeminiClient.generate_stream() method."""
    
    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test-api-key'})
    @patch('llm_client.genai')
    def test_generate_stream_success(self, mock_genai):
        """Test successful streaming response generation."""
        from llm_client import GeminiClient
        
        # Mock streaming chunks
        mock_chunk1 = Mock()
        mock_chunk1.text = "Hello "
        mock_chunk2 = Mock()
        mock_chunk2.text = "world!"
        
        # Mock the stream response with context manager
        mock_stream = MagicMock()
        # __enter__ should return self (the iterable), not a list
        mock_stream.__enter__.return_value = mock_stream
        mock_stream.__iter__.return_value = iter([mock_chunk1, mock_chunk2])
        mock_stream.__exit__.return_value = None
        mock_stream.resolve.return_value = None
        mock_stream.prompt_feedback = None
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_stream
        mock_genai.GenerativeModel.return_value = mock_model
        
        client = GeminiClient()
        stream = client.generate_stream("Test prompt", "Test system prompt")
        
        # Collect all yielded text
        result = list(stream)
        
        assert result == ["Hello ", "world!"]
        mock_stream.resolve.assert_called_once()
    
    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test-api-key'})
    @patch('llm_client.genai')
    def test_generate_stream_empty_chunks(self, mock_genai):
        """Test streaming with empty chunks (only chunks with text should be yielded)."""
        from llm_client import GeminiClient
        
        # Mock chunks with some having no text
        mock_chunk1 = Mock()
        mock_chunk1.text = "Start"
        mock_chunk2 = Mock()
        mock_chunk2.text = None  # Empty chunk
        mock_chunk3 = Mock()
        mock_chunk3.text = "End"
        
        mock_stream = MagicMock()
        mock_stream.__enter__.return_value = mock_stream
        mock_stream.__iter__.return_value = iter([mock_chunk1, mock_chunk2, mock_chunk3])
        mock_stream.__exit__.return_value = None
        mock_stream.resolve.return_value = None
        mock_stream.prompt_feedback = None
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_stream
        mock_genai.GenerativeModel.return_value = mock_model
        
        client = GeminiClient()
        stream = client.generate_stream("Test prompt", "Test system prompt")
        
        result = list(stream)
        
        # Only chunks with text should be yielded
        assert result == ["Start", "End"]
    
    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test-api-key'})
    @patch('llm_client.genai')
    def test_generate_stream_safety_block(self, mock_genai):
        """Test streaming with safety-blocked response."""
        from llm_client import GeminiClient
        
        # Mock empty stream (no chunks) with safety block
        mock_feedback = Mock()
        mock_feedback.block_reason = "SAFETY"
        
        mock_stream = MagicMock()
        mock_stream.__enter__.return_value = mock_stream
        mock_stream.__iter__.return_value = iter([])  # No chunks
        mock_stream.__exit__.return_value = None
        mock_stream.resolve.return_value = None
        mock_stream.prompt_feedback = mock_feedback
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_stream
        mock_genai.GenerativeModel.return_value = mock_model
        
        client = GeminiClient()
        stream = client.generate_stream("Unsafe prompt", "Test system prompt")
        
        result = list(stream)
        
        expected = "I apologize, but I cannot provide a response to this query due to content safety restrictions."
        assert result == [expected]
    
    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test-api-key'})
    @patch('llm_client.genai')
    def test_generate_stream_retry_on_initial_request(self, mock_genai):
        """Test retry behavior when initial stream creation fails."""
        from llm_client import GeminiClient
        
        # Mock successful chunks
        mock_chunk = Mock()
        mock_chunk.text = "Retry success"
        
        mock_stream = MagicMock()
        mock_stream.__enter__.return_value = mock_stream
        mock_stream.__iter__.return_value = iter([mock_chunk])
        mock_stream.__exit__.return_value = None
        mock_stream.resolve.return_value = None
        mock_stream.prompt_feedback = None
        
        # First call raises TimeoutError, second call succeeds
        mock_model = Mock()
        mock_model.generate_content.side_effect = [
            TimeoutError("Stream timeout"),
            mock_stream
        ]
        mock_genai.GenerativeModel.return_value = mock_model
        
        client = GeminiClient()
        
        with patch('time.sleep'):
            stream = client.generate_stream("Test prompt", "Test system prompt")
            result = list(stream)
        
        assert result == ["Retry success"]
        assert mock_model.generate_content.call_count == 2


class TestGeminiClientPromptComposition:
    """Test centralized prompt composition."""
    
    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test-api-key'})
    @patch('llm_client.genai')
    def test_prompt_composition_gemini_style(self, mock_genai):
        """Test that Gemini client uses correct prompt format."""
        from llm_client import GeminiClient
        
        mock_response = Mock()
        mock_response.text = "Response"
        mock_response.prompt_feedback = None
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        client = GeminiClient()
        client.generate("User query here", "System context here")
        
        # Verify the composed prompt format
        call_args = mock_model.generate_content.call_args
        prompt = call_args[0][0]
        
        assert "System context here" in prompt
        assert "User Query: User query here" in prompt
        assert "Response:" in prompt


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
