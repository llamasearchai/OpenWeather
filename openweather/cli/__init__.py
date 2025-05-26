"""CLI module for OpenWeather application."""
import logging
from openweather.cli.main import app_cli
from openweather.cli.utils_cli import (
    display_forecast_rich,
    display_llm_providers_rich,
    display_analyst_response_rich
)

logger = logging.getLogger(__name__)
logger.info("`openweather.cli` package initialized, exposing Typer app and CLI utilities.")

__all__ = [
    "app_cli", 
    "display_forecast_rich", 
    "display_llm_providers_rich",
    "display_analyst_response_rich"
] 