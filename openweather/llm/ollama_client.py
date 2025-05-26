import asyncio
import httpx
import logging
from typing import Optional, Dict, Any, List
from openweather.core.config import settings

logger = logging.getLogger(__name__)

class OllamaClient:
    """Client for interacting with local Ollama LLM server."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(base_url=base_url)
        self.default_model = settings.OLLAMA_DEFAULT_MODEL

    def is_available(self) -> bool:
        """Check if Ollama server is available."""
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Tuple[str, Dict]:
        """Generate text using Ollama API."""
        model = model or self.default_model
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9)
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
            
        try:
            response = await self.client.post("/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
            
            return data["response"], {
                "model": model,
                "tokens": {
                    "prompt": data.get("prompt_eval_count"),
                    "completion": data.get("eval_count")
                }
            }
        except Exception as e:
            logger.error(f"Ollama generation failed: {str(e)}")
            raise

    async def list_models(self) -> List[str]:
        """List available Ollama models."""
        try:
            response = await self.client.get("/api/tags")
            response.raise_for_status()
            return [model["name"] for model in response.json().get("models", [])]
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {str(e)}")
            return [] 