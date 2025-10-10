import os
import time
from abc import ABC, abstractmethod
from typing import Iterator, Optional
from config import (
    LLM_PROVIDER, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    OPENAI_API_KEY, ANTHROPIC_API_KEY, LOCAL_LLM_ENDPOINT, LOCAL_LLM_MODEL
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
    
    def _retry_with_backoff(self, func, max_retries: int = 3):
        """Retry a function with exponential backoff."""
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
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
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
            
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
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
            
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


def get_llm_client(provider: str = None) -> LLMClient:
    """
    Factory function to get appropriate LLM client.
    
    Args:
        provider: LLM provider ('openai', 'anthropic', 'local')
        
    Returns:
        Configured LLM client instance
    """
    provider = provider or LLM_PROVIDER
    
    if provider == 'openai':
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required. Set it in .env file.")
        return OpenAIClient()
    
    elif provider == 'anthropic':
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is required. Set it in .env file.")
        return AnthropicClient()
    
    elif provider == 'local':
        return LocalLLMClient()
    
    else:
        raise ValueError(f"Unknown LLM provider: {provider}. Choose from: openai, anthropic, local")


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
