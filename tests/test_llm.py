"""Tests for LLM integration modules."""
import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock

from openweather.llm.llm_manager import LLMManager, ProviderType
from openweather.llm.ollama_mlx_client import OllamaClient, MLX_AVAILABLE

@pytest.mark.asyncio
async def test_llm_manager_initialization():
    """Test LLM manager initialization."""
    with patch('openweather.llm.llm_manager.OllamaClient') as mock_ollama, \
         patch('openweather.llm.llm_manager.MLXClient', create=True) as mock_mlx, \
         patch('openweather.llm.llm_manager.HuggingFaceInferenceAPIClient') as mock_hf, \
         patch('openweather.core.config.settings') as mock_settings:
        
        # Configure settings
        mock_settings.USE_OLLAMA = True
        mock_settings.USE_MLX = True
        mock_settings.HF_API_KEY = MagicMock()
        mock_settings.HF_API_KEY.get_secret_value.return_value = "dummy_key"
        mock_settings.OPENAI_API_KEY = None
        
        # Initialize manager
        manager = LLMManager()
        
        # Verify clients were initialized correctly
        mock_ollama.assert_called_once()
        if MLX_AVAILABLE:
            mock_mlx.assert_called_once()
        mock_hf.assert_called_once()

@pytest.mark.asyncio
async def test_list_available_providers():
    """Test listing available providers."""
    manager = LLMManager()
    
    # Override clients for testing
    manager.ollama_client = MagicMock()
    manager.mlx_client = None
    manager.hf_client = MagicMock()
    manager.openai_client = None
    
    # Check available providers
    providers = manager.list_available_providers()
    
    assert providers["local_ollama"] is True
    assert providers["local_mlx"] is False
    assert providers["huggingface"] is True
    assert providers["openai"] is False

@pytest.mark.asyncio
async def test_generate_text_ollama():
    """Test generating text with Ollama."""
    manager = LLMManager()
    
    # Mock Ollama client
    mock_ollama = AsyncMock()
    mock_ollama.generate.return_value = ("Test response", {"provider_used": "local_ollama"})
    manager.ollama_client = mock_ollama
    
    # Call generate_text
    response, metadata = await manager.generate_text(
        prompt="Test prompt",
        provider="local_ollama",
        system_prompt="You are a helpful assistant"
    )
    
    # Verify the response
    assert response == "Test response"
    assert metadata["provider_used"] == "local_ollama"
    
    # Verify the client was called with correct parameters
    mock_ollama.generate.assert_called_once_with(
        prompt="Test prompt",
        system_prompt="You are a helpful assistant",
        temperature=None,
        max_tokens=None
    )

@pytest.mark.asyncio
async def test_ollama_client():
    """Test Ollama client functionality."""
    client = OllamaClient(model="llama3")
    
    # Mock HTTP response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "response": "Test response",
        "total_duration": 1234,
        "prompt_eval_count": 10,
        "eval_count": 20
    }
    
    with patch('httpx.AsyncClient.post', return_value=mock_response) as mock_post:
        response, metadata = await client.generate(
            prompt="Test prompt",
            system_prompt="You are a helpful assistant"
        )
        
        # Verify the response
        assert response == "Test response"
        assert metadata["provider_used"] == "local_ollama"
        assert metadata["model_used"] == "llama3"
        assert metadata["tokens_used"]["prompt_tokens"] == 10
        assert metadata["tokens_used"]["completion_tokens"] == 20
        
        # Verify the request was made correctly
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert kwargs["json"]["model"] == "llama3"
        assert kwargs["json"]["prompt"] == "Test prompt"
        assert kwargs["json"]["system"] == "You are a helpful assistant" 