"""Weather prediction model interfaces and base classes."""
import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple, Union, Type, TypeVar

# This import is problematic, WeatherForecastResponse is in core.models_shared
# from openweather.core.models import WeatherForecastResponse 
from openweather.core.models_shared import WeatherForecastResponse


logger = logging.getLogger(__name__)

class AbstractWeatherPredictor(ABC):
    """Abstract base class for weather prediction models."""
    
    model_name: str = "base_predictor"
    model_version: str = "1.0"
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the weather predictor with model path and configuration."""
        self.model_path = model_path
        self.config = config or {}
        self.is_loaded = False
        
        try:
            # Changed to await _load_model as it is async now
            asyncio.run(self._load_model()) 
            self.is_loaded = True
            logger.info(f"Model {self.model_name} version {self.model_version} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {str(e)}")
            self.is_loaded = False
            
    @abstractmethod
    async def _load_model(self) -> None:
        """Load the model asynchronously."""
        pass
        
    @abstractmethod
    async def predict(self, input_data: WeatherForecastResponse) -> WeatherForecastResponse:
        """Make a prediction using the model."""
        if not self.is_loaded:
            raise RuntimeError(f"Model {self.model_name} is not loaded.")
        # Added return type to satisfy ABC (though it will be overridden)
        return input_data
            
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            "name": self.model_name,
            "version": self.model_version,
            "path": self.model_path,
            "loaded_status": self.is_loaded,
            "config": self.config
        }
        
    def __repr__(self):
        """String representation of the model."""
        return f"{self.__class__.__name__}(name={self.model_name}, version={self.model_version}, loaded={self.is_loaded})" 