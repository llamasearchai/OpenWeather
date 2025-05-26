"""Stub implementation of a physics-enhanced weather model."""
import asyncio
import logging
import random
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple
from typing_extensions import Annotated

from openweather.models.weather_predictor import AbstractWeatherPredictor
from openweather.core.models_shared import WeatherForecastResponse, DailyForecast

logger = logging.getLogger(__name__)

class StubPhysicsEnhancedModel(AbstractWeatherPredictor):
    """Stub implementation of a physics-enhanced model for demonstration."""
    
    model_name = "StubPhysicsEnhancedPredictor"
    model_version = "1.0-stub"
    
    def __init__(self, model_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """Initialize the physics-enhanced model with default configuration."""
        # Merge provided config with defaults
        default_config = {
            "adjustment_factor_temp_max": 0.5,
            "adjustment_factor_temp_min": -0.2,
            "random_noise_stddev": 0.1
        }
        
        if config:
            default_config.update(config)
            
        super().__init__(model_path, default_config)
        
    async def _load_model(self) -> None:
        """Simulate model loading."""
        logger.info("Simulating loading of physics-enhanced model")
        # Simulate loading time
        await asyncio.sleep(0.5)
        self.is_loaded = True # is_loaded is set in base class after this successfully returns
        
    async def predict(self, input_data: WeatherForecastResponse) -> WeatherForecastResponse:
        """Apply physics-based adjustments to the input forecast."""
        await super().predict(input_data) # Call base class to check if loaded
        
        # Create a deep copy to avoid modifying original
        output_forecast = input_data.model_copy(deep=True)
        
        # Apply adjustments to temperature forecasts
        adjustment_factor_max = self.config["adjustment_factor_temp_max"]
        adjustment_factor_min = self.config["adjustment_factor_temp_min"]
        noise_stddev = self.config["random_noise_stddev"]
        
        for i, forecast in enumerate(output_forecast.daily_forecasts):
            # Apply physics-based adjustments (simplified)
            adjusted_temp_max = forecast.temp_max_celsius + adjustment_factor_max + random.gauss(0, noise_stddev)
            adjusted_temp_min = forecast.temp_min_celsius + adjustment_factor_min + random.gauss(0, noise_stddev)
            
            # Update forecast values
            output_forecast.daily_forecasts[i] = forecast.model_copy(update={
                "temp_max_celsius": round(adjusted_temp_max, 1),
                "temp_min_celsius": round(adjusted_temp_min, 1),
                "detailed_summary": f"{forecast.detailed_summary} (adjusted by StubPhysicsModel)"
            })
            
        # Update model info in the response
        # Ensure model_info is a dict before updating
        if not isinstance(output_forecast.model_info, dict):
            output_forecast.model_info = {}
        output_forecast.model_info.update({
            "name": self.model_name,
            "version": self.model_version,
            "path": self.model_path,
            "config": self.config
        })
        
        # Update data source information
        output_forecast.data_source = f"[Enhanced by StubPhysicsModel] {output_forecast.data_source}"
        
        return output_forecast 