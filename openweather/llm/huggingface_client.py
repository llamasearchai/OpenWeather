"""Client for Hugging Face's Inference API."""
import asyncio
import logging
import time
from typing import Optional, Dict, Any, Tuple
from typing_extensions import Literal
from http import HTTPStatus

import httpx
from pydantic import HttpUrl

from openweather.core.config import settings

logger = logging.getLogger(__name__)

class HuggingFaceInferenceAPIClient:
    """Client for accessing Hugging Face's Inference API."""
    
    def __init__(
        self,
        model_id: str = None,
        api_key: Optional[str] = None,
        client: Optional[httpx.AsyncClient] = None
    ):
        """Initialize Hugging Face Inference API client."""
        self.model_id = model_id or settings.HF_MODEL_NAME
        self.api_key = api_key or (settings.HF_API_KEY.get_secret_value() if settings.HF_API_KEY else None)
        self.client = client or httpx.AsyncClient(
            base_url="https://api-inference.huggingface.co/models/",
            timeout=30.0
        )
        
    async def generate(
        self,
        prompt: str,
        model_id_override: Optional[str] = None,
        temperature: float = 0.7,
        max_new_tokens: int = 256,
        system_prompt: Optional[str] = None
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """Generate response using Hugging Face Inference API."""
        model_id = model_id_override or self.model_id
        headers = {"Content-Type": "application/json"}
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Add system prompt if provided
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"<|system|>\n{system_prompt}\n\n<|user|>\n{prompt}\n\n<|assistant|>"
            
        payload = {
            "inputs": full_prompt,
            "parameters": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "return_full_text": False,
                "do_sample": True if temperature > 0.01 else False,
                "wait_for_model": True
            }
        }
        
        start_time = time.time()
        
        try:
            async with self.client as client:
                response = await client.post(f"/{model_id}", json=payload, headers=headers)
                response.raise_for_status()
                
                # Calculate duration
                duration = time.time() - start_time
                
                # Handle different response formats
                response_json = response.json()
                
                if isinstance(response_json, list) and "generated_text" in response_json[0]:
                    generated_text = response_json[0]["generated_text"]
                elif isinstance(response_json, dict) and "generated_text" in response_json:
                    generated_text = response_json["generated_text"]
                else:
                    logger.warning("Unexpected Hugging Face response format: %s", response_json)
                    generated_text = None
                    
                return generated_text, {
                    "provider_used": "huggingface",
                    "model_used": model_id,
                    "total_duration": duration,
                    "tokens_used": {
                        "prompt_tokens": len(prompt.split()),  # Approximate
                        "completion_tokens": len(generated_text.split()) if generated_text else 0,  # Approximate
                        "total_tokens": None  # Hugging Face API doesn't provide exact token counts
                    }
                }
                
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            error_msg = f"Hugging Face API error {status_code}: {str(e)}"
            logger.error(error_msg)
            return None, {
                "error": error_msg,
                "status_code": status_code
            }
        except httpx.RequestError as e:
            error_msg = f"Network error with Hugging Face API: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return None, {"error": error_msg}
        except Exception as e:
            error_msg = f"Unexpected error in Hugging Face API: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return None, {"error": error_msg} 