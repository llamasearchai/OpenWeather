"""Ollama and MLX client implementations for LLM integration."""
import asyncio
import logging
import os
import json
import random
import time
from typing import Optional, Dict, Any, Tuple, List
from typing_extensions import Literal
from contextlib import AsyncExitStack

import httpx
from pydantic import HttpUrl

from openweather.core.config import settings
from openweather.core.utils import is_apple_silicon

logger = logging.getLogger(__name__)

class OllamaClient:
    """Client for interfacing with Ollama local LLM server."""
    
    def __init__(
        self, 
        base_url: Optional[HttpUrl] = None, 
        model: Optional[str] = None,
        timeout: int = 120
    ):
        """Initialize Ollama client with base URL and model name."""
        self.base_url = base_url or settings.OLLAMA_HOST
        self.model = model or settings.OLLAMA_MODEL
        self.timeout = timeout
        self._client = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        async with AsyncExitStack() as stack:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout
            )
            return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None
            
    async def check_model_availability(self) -> bool:
        """Check if the configured model is available in Ollama."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                models = response.json().get("models", [])
                available = any(model.get("name") == self.model for model in models)
                if available:
                    logger.debug(f"Ollama model '{self.model}' is available")
                else:
                    logger.warning(f"Ollama model '{self.model}' not found")
                return available
        except Exception as e:
            logger.error(f"Error checking Ollama model availability: {str(e)}")
            return False
            
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """Generate response from Ollama model."""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": stream,
                "options": {
                    "temperature": temperature or settings.LLM_TEMPERATURE,
                    "num_predict": max_tokens or settings.LLM_MAX_TOKENS
                }
            }
            
            # Add system prompt if provided
            if system_prompt:
                payload["system"] = system_prompt
                
            # M3 Mac optimization - use native Metal acceleration
            payload["options"]["mirostat"] = 0
            
            # Set optimal thread count for M3 Max (adjust based on CPU cores)
            if is_apple_silicon():
                payload["options"]["num_gpu"] = -1  # Use all available GPUs
                payload["options"]["num_thread"] = 8  # Optimized for M3 Max cores
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                if stream:
                    # Handle streaming response
                    generated_text = ""
                    async for line in response.aiter_lines():
                        if line.strip():
                            data = json.loads(line)
                            if "response" in data:
                                generated_text += data["response"]
                                if data.get("done", False):
                                    metadata = {
                                        "provider_used": "local_ollama",
                                        "model_used": self.model,
                                        "total_duration": data.get("total_duration"),
                                        "load_duration": data.get("load_duration"),
                                        "prompt_eval_count": data.get("prompt_eval_count"),
                                        "eval_count": data.get("eval_count"),
                                        "tokens_used": {
                                            "prompt_tokens": data.get("prompt_eval_count", 0),
                                            "completion_tokens": data.get("eval_count", 0),
                                            "total_tokens": (data.get("prompt_eval_count", 0) + 
                                                           data.get("eval_count", 0))
                                        }
                                    }
                                    return generated_text, metadata
                    return generated_text, {}
                else:
                    # Handle non-streaming response
                    data = response.json()
                    if "response" in data:
                        metadata = {
                            "provider_used": "local_ollama",
                            "model_used": self.model,
                            "total_duration": data.get("total_duration"),
                            "load_duration": data.get("load_duration"),
                            "prompt_eval_count": data.get("prompt_eval_count"),
                            "eval_count": data.get("eval_count"),
                            "tokens_used": {
                                "prompt_tokens": data.get("prompt_eval_count", 0),
                                "completion_tokens": data.get("eval_count", 0),
                                "total_tokens": (data.get("prompt_eval_count", 0) + 
                                               data.get("eval_count", 0))
                            }
                        }
                        return data["response"], metadata
                    return None, {"error": "No response field in Ollama response"}
                    
        except httpx.HTTPStatusError as e:
            error_msg = f"Ollama HTTP error {e.response.status_code}: {str(e)}"
            logger.error(error_msg)
            return None, {"error": error_msg, "status_code": e.response.status_code}
        except Exception as e:
            logger.exception("Ollama generation error: %s", str(e))
            return None, {"error": str(e)}

# MLX support is conditional for Apple Silicon
MLX_AVAILABLE = False
try:
    if is_apple_silicon():
        import mlx.core as mx
        import mlx.nn as mnn
        from mlx_lm import load, generate
        
        MLX_AVAILABLE = True
        logger.debug("MLX libraries successfully imported")
    else:
        logger.info("System is not Apple Silicon, MLX support disabled")
except ImportError as e:
    logger.warning(f"MLX libraries not available: {str(e)}. MLX functionality will be disabled.")
    class MLXClient:
        def __init__(self, *args, **kwargs):
            self.is_initialized = False
            logger.warning("MLXClient: MLX not available")
except Exception as e:
    logger.warning("MLX initialization error: %s", str(e))
    MLX_AVAILABLE = False
    class MLXClient:
        def __init__(self, *args, **kwargs):
            self.is_initialized = False
            logger.warning("MLXClient: MLX not available")
            
# Only define MLXClient if MLX is available
if MLX_AVAILABLE:
    class MLXClient:
        """Client for running local LLMs using Apple's MLX framework."""
        
        def __init__(
            self, 
            model_path: Optional[str] = None
        ):
            """Initialize MLX model client."""
            self.model_path = model_path or settings.MLX_MODEL_PATH
            self.is_initialized = False
            self.model = None
            self.tokenizer = None
            
            try:
                logger.info(f"Loading MLX model from {self.model_path}")
                
                # Check if path is local file or Hugging Face model ID
                if os.path.exists(self.model_path):
                    # Load from local file
                    self.model, self.tokenizer = load(self.model_path)
                else:
                    # Load from Hugging Face or cached model
                    self.model, self.tokenizer = load(self.model_path, use_cache=True)
                
                # Optimize model for M3 Max
                if hasattr(self.model, "process_batch_size"):
                    self.model.process_batch_size = 8  # Optimized for M3 Max
                
                self.is_initialized = True
                logger.info(f"MLX model {self.model_path} loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load MLX model: {str(e)}")
                self.is_initialized = False
                
        def generate(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None
        ) -> Tuple[Optional[str], Dict[str, Any]]:
            """Generate response using MLX model."""
            if not self.is_initialized:
                error_msg = "MLX model not initialized"
                logger.error(error_msg)
                return None, {"error": error_msg}
                
            try:
                # Combine system prompt with main prompt if provided
                if system_prompt:
                    prompt = f"{system_prompt}\n\n{prompt}"
                    
                # Record start time for performance metrics
                start_time = time.time()
                
                generated_text = generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt,
                    temp=temperature or settings.LLM_TEMPERATURE,
                    max_tokens=max_tokens or settings.LLM_MAX_TOKENS,
                    verbose=False
                )
                
                # Calculate duration
                duration = time.time() - start_time
                
                # Estimate token counts
                input_tokens = len(self.tokenizer.encode(prompt))
                output_tokens = len(self.tokenizer.encode(generated_text))
                
                return generated_text, {
                    "provider_used": "local_mlx",
                    "model_used": self.model_path,
                    "total_duration": duration,
                    "tokens_used": {
                        "prompt_tokens": input_tokens,
                        "completion_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens
                    }
                }
                
            except Exception as e:
                logger.exception("MLX generation error: %s", str(e))
                return None, {"error": str(e)} 