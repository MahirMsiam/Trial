import os
import time
from abc import ABC, abstractmethod
from typing import Iterator, Optional
from config import (
    LLM_PROVIDER, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, GEMINI_MODEL,
    LOCAL_LLM_ENDPOINT, LOCAL_LLM_MODEL
)
import logging_config  # noqa: F401
import logging

# Get logger
logger = logging.getLogger(__name__)

# Import OpenAI at module scope for test mocking
try:
    from openai import OpenAI as _OpenAI
except ImportError:
    _OpenAI = None

# Import Gemini SDK
try:
    import google.generativeai as genai
except ImportError:
    genai = None

# Import Google API Core exceptions for Gemini retry handling
try:
    from google.api_core import exceptions as google_exceptions
except ImportError:
    google_exceptions = None


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: str, temperature: float = None, max_tokens: int = None) -> str:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    def generate_stream(self, prompt: str, system_prompt: str) -> Iterator[str]:
        """Generate a streaming response from the LLM."""
        pass
    
    def _compose_prompt(self, system_prompt: str, user_prompt: str, style: str = "generic") -> str:
        """
        Centralized prompt composition to ensure consistent formatting across providers.
        
        Args:
            system_prompt: System instructions/context
            user_prompt: User query
            style: Prompt style format ('generic', 'gemini', 'local')
            
        Returns:
            Formatted prompt string
        """
        if style == "gemini":
            # Gemini doesn't have separate system role
            return f"{system_prompt}\n\nUser Query: {user_prompt}\n\nResponse:"
        elif style == "local":
            # Local LLM uses assistant-style formatting
            return f"{system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
        else:
            # Generic format
            return f"{system_prompt}\n\n{user_prompt}"
    
    def _retry_with_backoff(self, func, max_retries: int = 3, retriable_exceptions: tuple = (), non_retriable_exceptions: tuple = (), retriable_patterns: list = None):
        """
        Retry a function with exponential backoff.
        
        Handles transient errors gracefully and fails fast on non-retriable errors.
        
        Args:
            func: Function to retry
            max_retries: Maximum number of retry attempts
            retriable_exceptions: Tuple of exception types that should be retried
            non_retriable_exceptions: Tuple of exception types that should fail immediately
            retriable_patterns: List of strings to check in error messages for retriable errors
        """
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                # Check for non-retriable exceptions first (fail immediately)
                if non_retriable_exceptions and isinstance(e, non_retriable_exceptions):
                    logger.error(f"Non-retriable error detected: {e}")
                    raise
                
                # Check if error is retriable by exception type
                is_retriable = False
                if retriable_exceptions:
                    is_retriable = isinstance(e, retriable_exceptions)
                
                # Check error message for retriable patterns
                error_msg = str(e).lower()
                
                # Check for non-retriable patterns first
                if "invalid" in error_msg or "unauthorized" in error_msg or "permission" in error_msg:
                    logger.error(f"Non-retriable error detected: {e}")
                    raise
                
                # Check for retriable patterns
                if not is_retriable and retriable_patterns:
                    for pattern in retriable_patterns:
                        if pattern in error_msg:
                            is_retriable = True
                            break
                
                # If last attempt, always raise
                if attempt == max_retries - 1:
                    raise
                
                wait_time = 2 ** attempt
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)


class OpenAIClient(LLMClient):
    """OpenAI API client."""
    
    def __init__(self, api_key: str = None, model: str = None):
        """Initialize OpenAI client."""
        if _OpenAI is None:
            raise ImportError("openai package is required. Install with: pip install openai")
        
        self.api_key = api_key or OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.model = model or LLM_MODEL
        self.client = _OpenAI(api_key=self.api_key)
        logger.info(f"✅ Initialized OpenAI client with model: {self.model}")
    
    def generate(self, prompt: str, system_prompt: str, temperature: float = None, max_tokens: int = None) -> str:
        """Generate a response using OpenAI API."""
        try:
            def _call_api():
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature or LLM_TEMPERATURE,
                    max_tokens=max_tokens or LLM_MAX_TOKENS
                )
                return response.choices[0].message.content
            
            result = self._retry_with_backoff(_call_api)
            logger.info("✅ OpenAI generation successful")
            return result
            
        except Exception as e:
            logger.error(f"❌ OpenAI generation failed: {e}")
            raise
    
    def generate_stream(self, prompt: str, system_prompt: str) -> Iterator[str]:
        """Generate a streaming response using OpenAI API."""
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"❌ OpenAI streaming failed: {e}")
            raise


class AnthropicClient(LLMClient):
    """Anthropic Claude API client."""
    
    def __init__(self, api_key: str = None, model: str = None):
        """Initialize Anthropic client."""
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("anthropic package is required. Install with: pip install anthropic")
        
        self.api_key = api_key or ANTHROPIC_API_KEY
        if not self.api_key:
            raise ValueError("Anthropic API key is required")
        
        self.model = model or LLM_MODEL
        self.client = Anthropic(api_key=self.api_key)
        logger.info(f"✅ Initialized Anthropic client with model: {self.model}")
    
    def generate(self, prompt: str, system_prompt: str, temperature: float = None, max_tokens: int = None) -> str:
        """Generate a response using Anthropic API."""
        try:
            def _call_api():
                response = self.client.messages.create(
                    model=self.model,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature or LLM_TEMPERATURE,
                    max_tokens=max_tokens or LLM_MAX_TOKENS
                )
                return response.content[0].text
            
            result = self._retry_with_backoff(_call_api)
            logger.info("✅ Anthropic generation successful")
            return result
            
        except Exception as e:
            logger.error(f"❌ Anthropic generation failed: {e}")
            raise
    
    def generate_stream(self, prompt: str, system_prompt: str) -> Iterator[str]:
        """Generate a streaming response using Anthropic API."""
        try:
            with self.client.messages.stream(
                model=self.model,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS
            ) as stream:
                for text in stream.text_stream:
                    yield text
                    
        except Exception as e:
            logger.error(f"❌ Anthropic streaming failed: {e}")
            raise


class LocalLLMClient(LLMClient):
    """Local LLM client for self-hosted models."""
    
    def __init__(self, endpoint: str = None, model: str = None):
        """Initialize local LLM client."""
        try:
            import requests
        except ImportError:
            raise ImportError("requests package is required. Install with: pip install requests")
        
        self.endpoint = endpoint or LOCAL_LLM_ENDPOINT
        self.model = model or LOCAL_LLM_MODEL
        self.requests = requests
        logger.info(f"✅ Initialized Local LLM client: {self.endpoint}")
    
    def generate(self, prompt: str, system_prompt: str, temperature: float = None, max_tokens: int = None) -> str:
        """Generate a response using local LLM API."""
        try:
            full_prompt = self._compose_prompt(system_prompt, prompt, style="local")
            
            def _call_api():
                response = self.requests.post(
                    f"{self.endpoint}/v1/completions",
                    json={
                        "model": self.model,
                        "prompt": full_prompt,
                        "temperature": temperature or LLM_TEMPERATURE,
                        "max_tokens": max_tokens or LLM_MAX_TOKENS
                    },
                    timeout=60
                )
                response.raise_for_status()
                return response.json()['choices'][0]['text']
            
            result = self._retry_with_backoff(_call_api)
            logger.info("✅ Local LLM generation successful")
            return result
            
        except Exception as e:
            logger.error(f"❌ Local LLM generation failed: {e}")
            raise
    
    def generate_stream(self, prompt: str, system_prompt: str) -> Iterator[str]:
        """Generate a streaming response using local LLM API."""
        try:
            full_prompt = self._compose_prompt(system_prompt, prompt, style="local")
            
            response = self.requests.post(
                f"{self.endpoint}/v1/completions",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "temperature": LLM_TEMPERATURE,
                    "max_tokens": LLM_MAX_TOKENS,
                    "stream": True
                },
                stream=True,
                timeout=60
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    yield line.decode('utf-8')
                    
        except Exception as e:
            logger.error(f"❌ Local LLM streaming failed: {e}")
            raise


class GeminiClient(LLMClient):
    """Google Gemini API client implementation."""
    
    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Gemini API key (optional, uses config if not provided)
            model: Model name (optional, uses config if not provided)
        """
        if genai is None:
            raise ImportError(
                "google-generativeai package is required for Gemini support. "
                "Install it with: pip install google-generativeai>=0.3.0"
            )
        
        self.api_key = api_key or GEMINI_API_KEY
        if not self.api_key:
            raise ValueError(
                "Gemini API key is required. Set GEMINI_API_KEY in .env file or pass api_key parameter."
            )
        
        # Configure the SDK with API key
        genai.configure(api_key=self.api_key)
        
        self.model = model or GEMINI_MODEL
        logger.info(f"✅ Initialized Gemini client with model: {self.model}")
    
    def generate(self, prompt: str, system_prompt: str, temperature: float = None, max_tokens: int = None) -> str:
        """Generate a response using Gemini API."""
        try:
            # Create GenerativeModel instance
            model = genai.GenerativeModel(self.model)
            
            # Gemini doesn't have separate system role, so combine prompts
            full_prompt = self._compose_prompt(system_prompt, prompt, style="gemini")
            
            # Define Gemini-specific retriable and non-retriable exceptions
            retriable = ()
            non_retriable = ()
            retriable_patterns = ["quota", "exceeded", "temporar", "timeout", "rate limit", "unavailable"]
            
            if google_exceptions is not None:
                retriable = (
                    google_exceptions.ResourceExhausted,
                    google_exceptions.ServiceUnavailable,
                    google_exceptions.DeadlineExceeded,
                    google_exceptions.InternalServerError,
                )
                non_retriable = (
                    getattr(google_exceptions, 'InvalidArgument', type(None)),
                    getattr(google_exceptions, 'FailedPrecondition', type(None)),
                    getattr(google_exceptions, 'PermissionDenied', type(None)),
                    getattr(google_exceptions, 'Unauthenticated', type(None)),
                )
            else:
                # Fallback: use built-in exception types when google_exceptions unavailable
                retriable = (TimeoutError, ConnectionError)
            
            def _call_api():
                # Configure generation parameters as dict to avoid SDK version mismatches
                generation_config = {
                    "temperature": temperature or LLM_TEMPERATURE,
                    "max_output_tokens": max_tokens or LLM_MAX_TOKENS,
                }
                
                response = model.generate_content(
                    full_prompt,
                    generation_config=generation_config
                )
                
                # Check for safety blocks
                if hasattr(response, 'prompt_feedback') and hasattr(response.prompt_feedback, 'block_reason'):
                    if response.prompt_feedback.block_reason:
                        logger.warning(f"⚠️ Gemini response blocked by safety filters: {response.prompt_feedback.block_reason}")
                        return "I apologize, but I cannot provide a response to this query due to content safety restrictions."
                
                # Extract text with fallback
                if response.text:
                    return response.text
                
                # Fallback: extract from candidates if text is empty
                try:
                    if hasattr(response, 'candidates') and response.candidates:
                        candidate = response.candidates[0]
                        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                            parts = candidate.content.parts
                            text_parts = []
                            for part in parts:
                                # Support both dict and object shapes
                                if isinstance(part, dict):
                                    text_parts.append(part.get('text', ''))
                                elif hasattr(part, 'text'):
                                    text_parts.append(part.text)
                            return ''.join(text_parts) or ""
                except Exception as extract_error:
                    logger.warning(f"Failed to extract text from candidates: {extract_error}")
                
                return ""
            
            result = self._retry_with_backoff(
                _call_api, 
                retriable_exceptions=retriable, 
                non_retriable_exceptions=non_retriable,
                retriable_patterns=retriable_patterns
            )
            logger.info("✅ Gemini generation successful")
            return result
            
        except Exception as e:
            logger.error(f"❌ Gemini generation failed: {e}")
            raise
    
    def generate_stream(self, prompt: str, system_prompt: str) -> Iterator[str]:
        """Generate a streaming response using Gemini API."""
        try:
            # Create GenerativeModel instance
            model = genai.GenerativeModel(self.model)
            
            # Gemini doesn't have separate system role, so combine prompts
            full_prompt = self._compose_prompt(system_prompt, prompt, style="gemini")
            
            # Configure generation parameters as dict to avoid SDK version mismatches
            generation_config = {
                "temperature": LLM_TEMPERATURE,
                "max_output_tokens": LLM_MAX_TOKENS,
            }
            
            # Define retry configuration (same as generate())
            retriable = ()
            non_retriable = ()
            retriable_patterns = ["quota", "exceeded", "temporar", "timeout", "rate limit", "unavailable"]
            
            if google_exceptions is not None:
                retriable = (
                    google_exceptions.ResourceExhausted,
                    google_exceptions.ServiceUnavailable,
                    google_exceptions.DeadlineExceeded,
                    google_exceptions.InternalServerError,
                )
                non_retriable = (
                    getattr(google_exceptions, 'InvalidArgument', type(None)),
                    getattr(google_exceptions, 'FailedPrecondition', type(None)),
                    getattr(google_exceptions, 'PermissionDenied', type(None)),
                    getattr(google_exceptions, 'Unauthenticated', type(None)),
                )
            else:
                # Fallback: use built-in exception types when google_exceptions unavailable
                retriable = (TimeoutError, ConnectionError)
            
            # Wrap initial stream creation in retry logic
            def _start_stream():
                return model.generate_content(
                    full_prompt,
                    generation_config=generation_config,
                    stream=True
                )
            
            response = self._retry_with_backoff(
                _start_stream,
                retriable_exceptions=retriable,
                non_retriable_exceptions=non_retriable,
                retriable_patterns=retriable_patterns
            )
            
            # Track if any content was yielded
            content_yielded = False
            block_reason = None
            
            # Use context manager for proper streaming finalization
            with response as stream:
                for chunk in stream:
                    text = getattr(chunk, "text", None)
                    if text:
                        content_yielded = True
                        yield text
                # Finalize stream and populate metadata
                stream.resolve()
                # Capture safety block info before exiting context
                block_reason = getattr(getattr(stream, "prompt_feedback", None), "block_reason", None)
            
            # After context exits, check captured safety block info
            if block_reason and not content_yielded:
                logger.warning(f"⚠️ Gemini streaming blocked by safety filters: {block_reason}")
                yield "I apologize, but I cannot provide a response to this query due to content safety restrictions."
                    
        except Exception as e:
            logger.error(f"❌ Gemini streaming failed: {e}")
            raise


class FakeLLMClient(LLMClient):
    """Fake LLM client for testing that returns mock responses."""
    
    def generate(self, prompt: str, system_prompt: str, temperature: float = None, max_tokens: int = None) -> str:
        """Return a fake response for testing."""
        logger.debug("Using FakeLLMClient for testing")
        return "This is a mock response from FakeLLMClient. Based on the provided context, the answer is: [mock legal analysis]."
    
    def generate_stream(self, prompt: str, system_prompt: str) -> Iterator[str]:
        """Return a fake streaming response for testing."""
        logger.debug("Using FakeLLMClient streaming for testing")
        response = "This is a mock streaming response from FakeLLMClient. Based on the provided context, the answer is: [mock legal analysis]."
        # Split into chunks to simulate streaming
        words = response.split()
        for word in words:
            yield word + " "


def get_llm_client(provider: str = None) -> LLMClient:
    """
    Factory function to get appropriate LLM client.
    
    Args:
        provider: LLM provider ('openai', 'anthropic', 'gemini', 'local', 'fake')
        
    Returns:
        Configured LLM client instance
    """
    from config import USE_FAKE_LLM_FOR_TESTS
    
    # Override with fake LLM if testing flag is set
    if USE_FAKE_LLM_FOR_TESTS:
        logger.info("Using FakeLLMClient for testing (USE_FAKE_LLM_FOR_TESTS=true)")
        return FakeLLMClient()
    
    provider = provider or LLM_PROVIDER
    
    if provider == 'fake':
        return FakeLLMClient()
    
    elif provider == 'openai':
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required. Set it in .env file.")
        return OpenAIClient()
    
    elif provider == 'anthropic':
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is required. Set it in .env file.")
        return AnthropicClient()
    
    elif provider == 'gemini':
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is required. Set it in .env file.")
        return GeminiClient()
    
    elif provider == 'local':
        return LocalLLMClient()
    
    else:
        raise ValueError(f"Unknown LLM provider: {provider}. Choose from: openai, anthropic, gemini, local, fake")


def validate_response(response: str) -> bool:
    """
    Validate LLM response quality.
    
    Args:
        response: LLM generated response
        
    Returns:
        True if valid, False otherwise
    """
    if not response or len(response.strip()) == 0:
        return False
    
    # Whitelist Gemini safety message as valid (it's an intentional response)
    gemini_safety_message = "I apologize, but I cannot provide a response to this query due to content safety restrictions."
    if response.strip() == gemini_safety_message:
        return True
    
    # Check for refusal patterns
    refusal_patterns = [
        "i cannot", "i can't", "i am not able",
        "i don't have access", "i apologize", "sorry"
    ]
    
    response_lower = response.lower()[:200]  # Check first 200 chars
    if any(pattern in response_lower for pattern in refusal_patterns):
        logger.warning("Response contains refusal pattern")
        return False
    
    return True


if __name__ == '__main__':
    # Test the LLM client
    print("Testing LLM Client...")
    
    try:
        client = get_llm_client()
        test_prompt = "What is a writ petition?"
        test_system = "You are a legal assistant."
        
        print("\nGenerating response...")
        response = client.generate(test_prompt, test_system)
        print(f"\nResponse: {response[:200]}...")
        
        is_valid = validate_response(response)
        print(f"\nValidation: {'✅ Valid' if is_valid else '❌ Invalid'}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
