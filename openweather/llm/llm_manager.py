"""Manager for coordinating different LLM providers."""
import logging
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple, Literal
from typing_extensions import Annotated
from datetime import datetime, timezone
import asyncio
import hashlib
import json
import re

from pydantic import Field

from openweather.core.config import settings
from openweather.llm.ollama_mlx_client import (
    OllamaClient,
    MLXClient,
    MLX_AVAILABLE
)
from openweather.llm.huggingface_client import HuggingFaceInferenceAPIClient
from openweather.llm.openai_client import OpenAIClient
from openweather.llm.hf_client import HuggingFaceClient
from llm import get_embedding
from openweather.data.cache import WeatherCache

logger = logging.getLogger(__name__)

# Type definition for LLM provider choices
ProviderType = Literal["local_ollama", "local_mlx", "huggingface", "openai"]

class LLMManager:
    """Enhanced LLM manager with caching and Datasette integration."""
    
    def __init__(self, cache: Optional[WeatherCache] = None):
        self.cache = cache or WeatherCache()
        self.clients = {
            ProviderType.OLLAMA: OllamaClient(),
            ProviderType.OPENAI: OpenAIClient(),
            ProviderType.HUGGINGFACE: HuggingFaceClient()
        }
        self.active_providers = self._detect_available_providers()
        logger.info(f"Initialized LLM manager with providers: {self.active_providers}")

    def _detect_available_providers(self) -> List[ProviderType]:
        """Detect which LLM providers are available."""
        available = []
        for provider, client in self.clients.items():
            if client.is_available():
                available.append(provider)
        return available

    async def generate_text(
        self,
        prompt: str,
        provider: Optional[ProviderType] = None,
        use_cache: bool = True,
        **kwargs
    ) -> Tuple[str, Dict]:
        """Generate text with caching and embedding support."""
        query_hash = self._generate_query_hash(prompt, provider, kwargs)
        
        # Check cache first
        if use_cache:
            cached = self.cache.get_llm_response(query_hash)
            if cached:
                return cached, {"from_cache": True, "provider": "cache"}
        
        # Generate embedding for similarity search
        prompt_embedding = get_embedding(prompt)
        
        # Check for similar cached queries
        similar = self.cache.search_similar_queries(prompt_embedding)
        if similar and use_cache:
            return similar[0]["response"], {"from_cache": True, "similar": True}
        
        # Generate new response
        response, metadata = await self._call_llm_provider(
            prompt, provider, **kwargs
        )
        
        # Store in cache
        self.cache.store_llm_response(
            query_hash=query_hash,
            response=response,
            model=metadata.get("model", "unknown")
        )
        
        return response, metadata

    def _generate_query_hash(self, prompt: str, provider: str, kwargs: Dict) -> str:
        """Generate consistent hash for query caching."""
        key = json.dumps({
            "prompt": prompt,
            "provider": provider,
            "kwargs": kwargs
        }, sort_keys=True)
        return hashlib.sha256(key.encode()).hexdigest()

    def _call_llm_provider(self, prompt: str, provider: Optional[ProviderType], **kwargs) -> Tuple[str, Dict]:
        """Call the specified LLM provider."""
        try:
            # Determine provider to use
            provider = self._resolve_provider(provider)
            if not provider:
                raise ValueError("No available LLM providers")
            
            logger.debug(f"Using LLM provider: {provider}")
            client = self.clients[provider]
            
            # Generate text with the selected provider
            response, metadata = await client.generate(
                prompt=prompt,
                **kwargs
            )
            
            return response, {
                "provider_used": provider,
                "model_used": metadata.get("model", "unknown"),
                "tokens_used": metadata.get("tokens", {})
            }
            
        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            raise

    def _resolve_provider(self, preferred: Optional[ProviderType]) -> Optional[ProviderType]:
        """Resolve which provider to use based on preference and availability."""
        if preferred and preferred in self.active_providers:
            return preferred
        return self.active_providers[0] if self.active_providers else None

    async def list_available_providers(self) -> Dict[str, Dict]:
        """List all available LLM providers and their models."""
        providers_info = {}
        for provider in self.active_providers:
            try:
                models = await self.clients[provider].list_models()
                providers_info[provider] = {
                    "status": "available",
                    "models": models
                }
            except Exception as e:
                providers_info[provider] = {
                    "status": f"error: {str(e)}",
                    "models": []
                }
        return providers_info
        
    async def list_available_providers_models(self) -> Dict[str, Dict[str, Any]]:
        """List all available LLM providers and their status."""
        providers_info = {}
        
        # Check Ollama status
        if self.clients[ProviderType.OLLAMA] and settings.USE_OLLAMA:
            try:
                ollama_available = await self.clients[ProviderType.OLLAMA].check_model_availability()
                providers_info["local_ollama"] = {
                    "status": "configured" if ollama_available else "error",
                    "default_model": self.clients[ProviderType.OLLAMA].model,
                    "available_models": [self.clients[ProviderType.OLLAMA].model] if ollama_available else [],
                    "notes": "Model available" if ollama_available else "Model not found"
                }
            except Exception as e:
                providers_info["local_ollama"] = {
                    "status": "error",
                    "error": str(e),
                    "notes": "Ollama client error"
                }
        else:
            providers_info["local_ollama"] = {
                "status": "disabled" if not settings.USE_OLLAMA else "not_available"
            }
            
        # Check MLX status
        if self.clients[ProviderType.MLX] and MLX_AVAILABLE and settings.USE_MLX:
            providers_info["local_mlx"] = {
                "status": "configured" if getattr(self.clients[ProviderType.MLX], "is_initialized", False) else "error",
                "default_model": self.clients[ProviderType.MLX].model_path,
                "notes": "Model loaded" if getattr(self.clients[ProviderType.MLX], "is_initialized", False) else "Model loading failed"
            }
        else:
            providers_info["local_mlx"] = {
                "status": "disabled" if not settings.USE_MLX else "not_available",
                "notes": "MLX not available" if not MLX_AVAILABLE else "MLX disabled in settings"
            }
            
        # Check Hugging Face status
        if self.clients[ProviderType.HUGGINGFACE] and settings.HF_API_KEY:
            providers_info["huggingface"] = {
                "status": "configured",
                "default_model": self.clients[ProviderType.HUGGINGFACE].model_id
            }
        else:
            providers_info["huggingface"] = {
                "status": "disabled" if not settings.HF_API_KEY else "needs_api_key"
            }
            
        # Check OpenAI status
        if self.clients[ProviderType.OPENAI] and settings.OPENAI_API_KEY:
            providers_info["openai"] = {
                "status": "configured",
                "default_model": self.clients[ProviderType.OPENAI].model if self.clients[ProviderType.OPENAI] else settings.OPENAI_MODEL_NAME
            }
        else:
            providers_info["openai"] = {
                "status": "disabled" if not settings.OPENAI_API_KEY else "not_available",
                "notes": "OpenAI client not initialized"
            }
            
        return providers_info

    def get_datasette_connection(self):
        """Get Datasette instance for analytics."""
        from datasette.app import Datasette
        return Datasette(
            [str(self.cache.db_path)],
            metadata={
                "title": "OpenWeather LLM Analytics",
                "plugins": {
                    "datasette-llm-embed": {
                        "enable_embeddings": True
                    }
                }
            }
        )

    async def generate_json_response(
        self,
        prompt: str,
        llm_provider: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """
        Generates a response from the LLM and attempts to parse it as JSON.
        The prompt should explicitly ask the LLM to provide a JSON formatted response.
        """
        chosen_provider = llm_provider or self.default_provider_name
        client = self._get_client(chosen_provider)
        
        provider_model_name = model_name or self.get_default_model_for_provider(chosen_provider)

        raw_response_text, metadata = await client.generate_text(
            prompt,
            model_name=provider_model_name,
            temperature=temperature or settings.DEFAULT_LLM_TEMPERATURE,
            max_tokens=max_tokens or settings.DEFAULT_LLM_MAX_TOKENS,
            # Some clients might support a specific "json_mode" or similar parameter.
            # Pass it here if available: e.g., json_mode=True
        )

        metadata["prompt_used"] = prompt # Add prompt to metadata for debugging

        if raw_response_text:
            try:
                # Attempt to find and parse a JSON block (e.g., ```json ... ```)
                # Regex tries to find ```json ... ``` first, then a standalone JSON object/array.
                json_match = re.search(r"```json\s*([\s\S]+?)\s*```|({[\s\S]*}|\[[\s\S]*\])", raw_response_text, re.DOTALL)
                
                json_str_to_parse = None
                if json_match:
                    if json_match.group(1): # Content within ```json ... ```
                        json_str_to_parse = json_match.group(1)
                    elif json_match.group(2): # Standalone JSON object/array
                        json_str_to_parse = json_match.group(2)
                
                if json_str_to_parse:
                    parsed_json = json.loads(json_str_to_parse)
                    logger.info(f"Successfully parsed JSON from LLM response using regex match.")
                    return parsed_json, metadata
                else:
                    # If no clear JSON block is found, try parsing the whole response.
                    # This is a fallback and might be less reliable.
                    logger.warning("No distinct JSON block found in LLM response. Attempting to parse entire response.")
                    try:
                        parsed_json = json.loads(raw_response_text)
                        logger.info("Successfully parsed entire LLM response as JSON.")
                        return parsed_json, metadata
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse entire LLM response as JSON. Response: {raw_response_text[:500]}...")
                        return None, metadata

            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON from LLM response: {e}. Response excerpt: {raw_response_text[:500]}...")
                return None, metadata
            except Exception as e:
                logger.error(f"An unexpected error occurred during JSON response processing: {e}. Response excerpt: {raw_response_text[:500]}...")
                return None, metadata
        
        logger.warning("LLM returned an empty response.")
        return None, metadata 