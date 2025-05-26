"""Models module for OpenWeather application."""
import logging
from openweather.models.weather_predictor import AbstractWeatherPredictor
from openweather.models.physics_model_stub import StubPhysicsEnhancedModel
from openweather.models.mlx_model_runner import MLXWeatherModelRunner, MLX_MODEL_RUNNER_AVAILABLE

logger = logging.getLogger(__name__)
logger.info("`openweather.models` package initialized.")
logger.info("MLX Model Runner available: %s", MLX_MODEL_RUNNER_AVAILABLE)

__all__ = [
    "AbstractWeatherPredictor",
    "StubPhysicsEnhancedModel",
    "MLXWeatherModelRunner",
    "MLX_MODEL_RUNNER_AVAILABLE"
] 