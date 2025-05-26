"""FastAPI dependencies for the OpenWeather API."""
from typing import Any, Dict, List, Optional
from functools import lru_cache
import logging

from fastapi import Depends
from http import HTTPStatus

from openweather.llm.llm_manager import LLMManager, ProviderType
from openweather.data.data_loader import WeatherDataOrchestrator
from openweather.models.weather_predictor import AbstractWeatherPredictor
from openweather.models.physics_model_stub import StubPhysicsEnhancedModel
from openweather.models.mlx_model_runner import MLXWeatherModelRunner, MLX_MODEL_RUNNER_AVAILABLE
from openweather.services.forecast_service import ForecastService
from openweather.core.config import settings

logger = logging.getLogger(__name__)

@lru_cache()
def get_llm_manager_dependency() -> LLMManager:
    """Get or create LLM manager instance."""
    logger.debug("Creating LLMManager instance")
    return LLMManager()

@lru_cache()
def get_data_orchestrator_dependency() -> WeatherDataOrchestrator:
    """Get or create data orchestrator instance."""
    logger.debug("Creating WeatherDataOrchestrator instance")
    return WeatherDataOrchestrator()

@lru_cache()
def get_primary_custom_weather_model_dependency() -> Optional[AbstractWeatherPredictor]:
    """Get or create the primary custom weather model."""
    logger.debug("Initializing primary custom weather model")
    
    primary_model: Optional[AbstractWeatherPredictor] = None
    
    try:
        # Attempt to load MLX model first if enabled and available
        if settings.USE_MLX and MLX_MODEL_RUNNER_AVAILABLE:
            logger.debug("MLX is enabled and available, attempting to load MLX model as primary.")
            try:
                mlx_model = MLXWeatherModelRunner(model_path=settings.MLX_MODEL_PATH)
                if mlx_model.is_loaded:
                    logger.info("MLX model loaded successfully as primary custom model.")
                    primary_model = mlx_model
                else:
                    logger.warning("MLX model failed to load, will try StubPhysicsEnhancedModel.")
            except Exception as e:
                logger.warning(f"Failed to initialize MLX model: {str(e)}, falling back.")
        
        # If MLX model wasn't loaded or isn't preferred/available, try StubPhysicsEnhancedModel
        if not primary_model:
            logger.debug("Attempting to load StubPhysicsEnhancedModel as primary or fallback.")
            try:
                stub_model = StubPhysicsEnhancedModel(config=settings.STUB_PHYSICS_MODEL_CONFIG)
                if stub_model.is_loaded:
                    logger.info("StubPhysicsEnhancedModel loaded successfully as custom model.")
                    primary_model = stub_model
                else:
                    logger.warning("StubPhysicsEnhancedModel failed to load.")
            except Exception as e:
                logger.warning(f"Failed to initialize StubPhysicsEnhancedModel: {str(e)}")

        if not primary_model:
             logger.warning("No custom weather model could be loaded.")

        return primary_model
            
    except Exception as e:
        logger.error(f"Critical error initializing custom weather model: {str(e)}")
        return None

def get_forecast_service(
    llm_manager: LLMManager = Depends(get_llm_manager_dependency),
    data_orchestrator: WeatherDataOrchestrator = Depends(get_data_orchestrator_dependency),
    custom_model: Optional[AbstractWeatherPredictor] = Depends(get_primary_custom_weather_model_dependency)
) -> ForecastService:
    """Get forecast service with dependencies."""
    logger.debug("Creating ForecastService instance")
    return ForecastService(data_orchestrator, llm_manager, custom_model) 