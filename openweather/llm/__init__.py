"""LLM integration module for OpenWeather application."""
import logging
from openweather.llm.llm_manager import LLMManager, ProviderType
from openweather.llm.ollama_mlx_client import OllamaClient, MLXClient, MLX_AVAILABLE
from openweather.llm.huggingface_client import HuggingFaceInferenceAPIClient
from openweather.llm.langchain_utils import WeatherAnalysisLangChain, WEATHER_SUMMARY_PROMPT_LANGCHAIN

logger = logging.getLogger(__name__)
logger.info("`openweather.llm` package initialized.")

# Import OpenAI client if available
try:
    from openweather.llm.openai_client import OpenAIClient
    OPENAI_CLIENT_AVAILABLE = True
except ImportError:
    OPENAI_CLIENT_AVAILABLE = False
    logger.warning("OpenAI client not available - package not installed")

__all__ = [
    "LLMManager",
    "ProviderType",
    "OllamaClient",
    "MLXClient",
    "MLX_AVAILABLE",
    "HuggingFaceInferenceAPIClient",
    "WeatherAnalysisLangChain",
    "WEATHER_SUMMARY_PROMPT_LANGCHAIN"
]

if OPENAI_CLIENT_AVAILABLE:
    __all__.append("OpenAIClient") 