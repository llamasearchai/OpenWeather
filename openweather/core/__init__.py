"""Core module for OpenWeather application."""
from logging import getLogger
from openweather.core.config import settings
from openweather.core.utils import (
    get_platform_info,
    is_apple_silicon,
    parse_location_string,
    format_weather_data_for_llm,
    setup_logging,
    PROJECT_ROOT,
    _degrees_to_cardinal,
    _format_temperature_rich
)
from openweather.core.models_shared import (
    Coordinate,
    LocationInfo,
    DailyForecast,
    CurrentWeather,
    WeatherForecastResponse,
    LLMAnalysisRequest,
    LLMAnalysisResponse,
    Alert,
    WeatherAlertResponse,
    ForecastAndAnalysisApiResponse
)

logger = getLogger(__name__)
logger.info("`openweather.core` module initialized, exposing settings, utils, and shared models.")

__all__ = [
    "settings",
    "get_platform_info",
    "is_apple_silicon",
    "parse_location_string",
    "format_weather_data_for_llm",
    "setup_logging",
    "PROJECT_ROOT",
    "_degrees_to_cardinal",
    "_format_temperature_rich",
    # Coordinate models
    "Coordinate",
    "LocationInfo",
    # Weather models
    "DailyForecast",
    "CurrentWeather",
    "WeatherForecastResponse",
    # LLM models
    "LLMAnalysisRequest",
    "LLMAnalysisResponse",
    # Alert models
    "Alert",
    "WeatherAlertResponse",
    "ForecastAndAnalysisApiResponse"
] 