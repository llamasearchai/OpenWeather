"""Data module for OpenWeather application."""
import logging
from openweather.data.data_loader import WeatherDataOrchestrator, AbstractWeatherDataSource, OpenMeteoDataSource
from openweather.data.data_simulator import (
    get_simulated_weather_forecast,
    generate_simulated_daily_forecast,
    generate_simulated_current_weather
)

logger = logging.getLogger(__name__)
logger.info("`openweather.data` package initialized.")

__all__ = [
    "WeatherDataOrchestrator",
    "AbstractWeatherDataSource",
    "OpenMeteoDataSource",
    "get_simulated_weather_forecast",
    "generate_simulated_daily_forecast",
    "generate_simulated_current_weather"
] 