"""Web interface package for OpenWeather application."""
import logging
from openweather.web.app import app as web_app

logger = logging.getLogger(__name__)
logger.debug("`openweather.web` package initialized.")

__all__ = ["web_app"]