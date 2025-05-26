"""OpenAI API client for LLM integration."""
import asyncio
import logging
import time
from typing import Optional, Dict, Any, Tuple, List
from typing_extensions import Literal
import openai

from openweather.core.config import settings

logger = logging.getLogger(__name__)

class OpenAIClient:
    """Client for OpenAI API."""
    
    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY
        self.default_model = settings.OPENAI_DEFAULT_MODEL
        openai.api_key = self.api_key

    def is_available(self) -> bool:
        """Check if OpenAI client is properly configured."""
        return bool(self.api_key)

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Tuple[str, Dict]:
        """Generate text using OpenAI API."""
        model = model or self.default_model
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = await openai.ChatCompletion.acreate(
                model=model,
                messages=messages,
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 0.9),
                max_tokens=kwargs.get("max_tokens", 1000)
            )
            
            return response.choices[0].message.content, {
                "model": model,
                "tokens": {
                    "prompt": response.usage.prompt_tokens,
                    "completion": response.usage.completion_tokens
                }
            }
        except Exception as e:
            logger.error(f"OpenAI generation failed: {str(e)}")
            raise

    async def list_models(self) -> List[str]:
        """List available OpenAI models."""
        try:
            models = await openai.Model.alist()
            return [model.id for model in models.data if "gpt" in model.id]
        except Exception as e:
            logger.error(f"Failed to list OpenAI models: {str(e)}")
            return [] 