"""Services module for OpenWeather application."""
import logging
from openweather.services.forecast_service import ForecastService

logger = logging.getLogger(__name__)
logger.info("`openweather.services` package initialized.")

__all__ = ["ForecastService"] 