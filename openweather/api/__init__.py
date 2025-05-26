"""API module for OpenWeather application."""
import logging
from openweather.api.main import app
from openweather.api.routes import api_router_v1
from openweather.api.dependencies import (
    get_llm_manager_dependency,
    get_data_orchestrator_dependency,
    get_forecast_service,
    get_primary_custom_weather_model_dependency
)

logger = logging.getLogger(__name__)
logger.info("`openweather.api` package initialized, exposing FastAPI app and core dependencies.")

__all__ = [
    "app", 
    "api_router_v1",
    "get_llm_manager_dependency",
    "get_data_orchestrator_dependency",
    "get_forecast_service",
    "get_primary_custom_weather_model_dependency"
] 