"""Data loaders for weather forecast data from external APIs."""
import asyncio
import logging
from datetime import date, datetime, timezone, timedelta
from typing import Optional, List, Dict, Any, Tuple, Type, TypeVar, Union
from typing_extensions import Annotated
from enum import Enum

import httpx
from pydantic import BaseModel, Field, validator, HttpUrl
from openweather.core.utils import _degrees_to_cardinal, parse_location_string
from openweather.core.models_shared import (
    Coordinate, LocationInfo, DailyForecast, CurrentWeather, WeatherForecastResponse
)
from openweather.core.config import settings
from openweather.data.data_simulator import get_simulated_weather_forecast
from openweather.data.openmeteo_client import OpenMeteoClient
from openweather.data.cache import WeatherCache

logger = logging.getLogger(__name__)

# WMO Weather codes mapping
WMO_WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Drizzle: Light",
    53: "Drizzle: Moderate",
    55: "Drizzle: Dense intensity",
    56: "Freezing Drizzle: Light",
    57: "Freezing Drizzle: Dense",
    61: "Rain: Slight",
    63: "Rain: Moderate",
    65: "Rain: Heavy intensity",
    66: "Freezing Rain: Light",
    67: "Freezing Rain: Heavy",
    71: "Snow fall: Slight",
    73: "Snow fall: Moderate",
    75: "Snow fall: Heavy",
    77: "Snow grains",
    80: "Rain showers: Slight",
    81: "Rain showers: Moderate",
    82: "Rain showers: Violent",
    85: "Snow showers: Slight",
    86: "Snow showers: Heavy",
    95: "Thunderstorm: Slight or moderate",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail"
}

def _map_wmo_code_to_text(wmo_code: Optional[int]) -> Optional[str]:
    """Map WMO weather code to descriptive text."""
    if wmo_code is None:
        return None
    return WMO_WEATHER_CODES.get(wmo_code, f"Unknown code {wmo_code}")

class DataSourceType(Enum):
    """Types of weather data sources."""
    OPEN_METEO = "open-meteo"
    SIMULATION = "simulation"
    WEATHER_API = "weather-api"
    DATASETTE = "datasette"

class AbstractWeatherDataSource:
    """Abstract base class for weather data sources."""
    
    def __init__(self, source_name: str):
        """Initialize the data source with a name."""
        self.source_name = source_name
        
    async def get_forecast(
        self, 
        latitude: float, 
        longitude: float, 
        num_days: int,
        location_name: Optional[str] = None
    ) -> Optional[WeatherForecastResponse]:
        """Get weather forecast from this data source."""
        raise NotImplementedError("Subclasses must implement get_forecast()")

class OpenMeteoDataSource(AbstractWeatherDataSource):
    """Weather data source for the Open-Meteo API."""
    
    def __init__(self):
        """Initialize Open-Meteo data source."""
        super().__init__("open-meteo")
        self.client = httpx.AsyncClient(timeout=30.0)
        
    async def get_forecast(
        self, 
        latitude: float, 
        longitude: float, 
        num_days: int,
        location_name: Optional[str] = None
    ) -> Optional[WeatherForecastResponse]:
        """Get weather forecast from Open-Meteo API."""
        base_url = "https://api.open-meteo.com/v1/forecast"
        
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "daily": (
                "weather_code,temperature_2m_max,temperature_2m_min,"
                "precipitation_sum,precipitation_probability_mean,"
                "wind_speed_10m_max,wind_direction_10m_dominant,"
                "uv_index_max,sunrise,sunset"
            ),
            "current": (
                "temperature_2m,relative_humidity_2m,apparent_temperature,"
                "is_day,precipitation,rain,showers,snowfall,weather_code,"
                "cloud_cover,pressure_msl,wind_speed_10m,wind_direction_10m,"
                "wind_gusts_10m"
            ),
            "timezone": "UTC",
            "forecast_days": min(max(num_days, 1), 16)
        }
        
        try:
            response = await self.client.get(base_url, params=params)
            response.raise_for_status()
            return self._parse_openmeteo_response(response.json(), latitude, longitude, location_name)
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Open-Meteo API error {e.response.status_code}: {str(e)}")
            return None
        except Exception as e:
            logger.exception("Error fetching Open-Meteo data: %s", str(e))
            return None
            
    def _parse_openmeteo_response(
        self,
        json_data: Dict[str, Any],
        latitude: float,
        longitude: float,
        location_name: Optional[str] = None
    ) -> WeatherForecastResponse:
        """Parse Open-Meteo API response into WeatherForecastResponse."""
        # Extract current weather data
        current_data = json_data.get("current", {})
        daily_data = json_data.get("daily", {})
        
        # Create location info
        location_info = LocationInfo(
            name=location_name or "Unknown",
            coordinates=Coordinate(latitude=latitude, longitude=longitude)
        )
        
        # Parse current weather
        current_weather = None
        if "temperature_2m" in current_data:
            current_temp = current_data["temperature_2m"]
            wind_speed = current_data.get("wind_speed_10m")
            wind_dir_deg = current_data.get("wind_direction_10m")
            
            current_weather = CurrentWeather(
                observed_at_utc=datetime.fromtimestamp(current_data.get("time", 0), tz=timezone.utc),
                temp_celsius=current_temp,
                feels_like_celsius=current_data.get("apparent_temperature"),
                condition_code=current_data.get("weather_code"),
                condition_text=_map_wmo_code_to_text(current_data.get("weather_code")),
                wind_speed_kph=wind_speed,
                wind_direction_degrees=wind_dir_deg,
                wind_direction_cardinal=_degrees_to_cardinal(wind_dir_deg) if wind_dir_deg else None,
                pressure_mb=current_data.get("pressure_msl"),
                precipitation_mm_last_hour=current_data.get("precipitation"),
                humidity_percent=current_data.get("relative_humidity_2m"),
                cloud_cover_percent=current_data.get("cloud_cover"),
                visibility_km=current_data.get("visibility"), # This field might not be in Open-Meteo, ensure it's handled if None
                uv_index=current_data.get("uv_index") # This field might not be in Open-Meteo, ensure it's handled if None
            )
            
        # Parse daily forecasts
        daily_forecasts = []
        if "temperature_2m_max" in daily_data and "time" in daily_data:
            dates = daily_data["time"]
            for i in range(len(dates)):
                date_str = dates[i]
                daily_forecast = DailyForecast(
                    date=datetime.strptime(date_str, "%Y-%m-%d").date(),
                    temp_max_celsius=daily_data["temperature_2m_max"][i],
                    temp_min_celsius=daily_data["temperature_2m_min"][i],
                    condition_code=daily_data.get("weather_code", [None])[i] if "weather_code" in daily_data else None,
                    condition_text=_map_wmo_code_to_text(daily_data.get("weather_code", [None])[i] if "weather_code" in daily_data else None),
                    precipitation_mm=daily_data.get("precipitation_sum", [0])[i] if "precipitation_sum" in daily_data else 0,
                    precipitation_chance_percent=daily_data.get("precipitation_probability_mean", [0])[i] if "precipitation_probability_mean" in daily_data else 0,
                    wind_speed_kph=daily_data.get("wind_speed_10m_max", [None])[i] if "wind_speed_10m_max" in daily_data else None,
                    wind_direction_degrees=daily_data.get("wind_direction_10m_dominant", [None])[i] if "wind_direction_10m_dominant" in daily_data else None,
                    wind_direction_cardinal=_degrees_to_cardinal(
                        daily_data.get("wind_direction_10m_dominant", [None])[i] if "wind_direction_10m_dominant" in daily_data else None
                    ) if daily_data.get("wind_direction_10m_dominant") and i < len(daily_data.get("wind_direction_10m_dominant", [])) else None,
                    uv_index=daily_data.get("uv_index_max", [None])[i] if "uv_index_max" in daily_data and i < len(daily_data.get("uv_index_max", [])) else None,
                    sunrise_utc=datetime.fromisoformat(daily_data.get("sunrise", [None])[i]).replace(tzinfo=timezone.utc) if "sunrise" in daily_data and i < len(daily_data.get("sunrise", [])) and daily_data.get("sunrise", [None])[i] is not None else None,
                    sunset_utc=datetime.fromisoformat(daily_data.get("sunset", [None])[i]).replace(tzinfo=timezone.utc) if "sunset" in daily_data and i < len(daily_data.get("sunset", [])) and daily_data.get("sunset", [None])[i] is not None else None,
                    detailed_summary=f"Open-Meteo forecast for {date_str}"
                )
                daily_forecasts.append(daily_forecast)
                
        return WeatherForecastResponse(
            location=location_info,
            current_weather=current_weather,
            daily_forecasts=daily_forecasts,
            generated_at_utc=datetime.now(timezone.utc),
            data_source="Open-Meteo API",
            model_info={
                "name": "Open-Meteo",
                "version": "2.0",
                "api_url": "https://api.open-meteo.com/v1/forecast"
            }
        )

class WeatherDataOrchestrator:
    """Enhanced with data source scoring and aggregation"""
    
    def __init__(self, cache: Optional[WeatherCache] = None):
        self.openmeteo_client = OpenMeteoClient()
        self.cache = cache or WeatherCache()
        self.data_sources: List[AbstractWeatherDataSource] = [
            OpenMeteoDataSource(),
            # Add other data sources here. For example:
            # SimulatedDataSource() # if you create one
        ]
        self.source_weights = {
            "open-meteo": 0.8, # Higher weight for a real API
            "simulation": 0.3
            # Add weights for other sources
        }
        # self.consensus_threshold = 0.7 # This can be used in more advanced consensus logic

    async def get_weather_data(
        self,
        location_str: str,
        num_days: int = 5,
        # preferred_source: Optional[str] = None # This parameter is not currently used in the consensus logic
    ) -> Optional[WeatherForecastResponse]:
        """Get consensus forecast from multiple sources, returning a WeatherForecastResponse."""
        cache_key = f"weather:{location_str}:{num_days}:consensus"
        
        cached_data = await self._get_fresh_cached_data(cache_key)
        if cached_data:
            try:
                return WeatherForecastResponse.parse_obj(cached_data)
            except Exception as e:
                logger.warning(f"Failed to parse cached data: {e}. Fetching fresh data.")

        # Gather forecasts from all sources
        # parse_location_string might raise ValueError if format is incorrect
        try:
            lat, lon = parse_location_string(location_str)
        except ValueError as e:
            logger.error(f"Invalid location string format: {location_str} - {e}")
            return None # Or raise an error to be caught by the caller

        forecast_results = await self._gather_forecasts(lat, lon, num_days, location_str)
        
        consensus_dict = self._calculate_consensus(forecast_results, location_str, num_days, lat, lon)
        
        if consensus_dict:
            try:
                response_obj = WeatherForecastResponse.parse_obj(consensus_dict)
                # Cache the dictionary representation of the object
                await self.cache.set(cache_key, response_obj.dict(), ttl=3600) 
                return response_obj
            except Exception as e:
                logger.error(f"Error parsing consensus data into WeatherForecastResponse: {e}")
                # Fallback to a basic simulation if parsing final consensus fails
                sim_forecast_dict = get_simulated_weather_forecast(location_str, num_days, lat, lon)
                if sim_forecast_dict:
                    return WeatherForecastResponse.parse_obj(sim_forecast_dict)
        return None

    async def _gather_forecasts(self, latitude: float, longitude: float, num_days: int, location_name: str) -> List[Union[WeatherForecastResponse, Exception]]:
        """Parallel fetch from all sources."""
        tasks = [
            source.get_forecast(
                latitude=latitude,
                longitude=longitude,
                num_days=num_days,
                location_name=location_name
            )
            for source in self.data_sources
        ]
        # return_exceptions=True allows us to see which sources failed
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

    def _calculate_consensus(
        self, 
        forecasts: List[Union[WeatherForecastResponse, Exception]], 
        location_str: str, 
        num_days: int,
        latitude: float, # Added for simulation fallback
        longitude: float # Added for simulation fallback
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate weighted consensus forecast.
        Returns a dictionary compatible with WeatherForecastResponse.
        """
        valid_forecasts: List[WeatherForecastResponse] = [
            f for f in forecasts 
            if isinstance(f, WeatherForecastResponse) and f is not None
        ]
        
        if not valid_forecasts:
            logger.warning("No valid forecasts from any source. Falling back to simulation for consensus.")
            return get_simulated_weather_forecast(location_str, num_days, latitude, longitude)

        # Simplified consensus: For now, we'll prioritize and use the first valid forecast from a weighted list.
        # A more robust implementation would merge data fields.
        
        # Sort forecasts by a predefined preference or weight if available
        # For instance, if OpenMeteoDataSource is preferred:
        sorted_valid_forecasts = sorted(
            valid_forecasts,
            key=lambda f: self.source_weights.get(f.data_source_key or "unknown", 0.1), # data_source_key from AbstractWeatherDataSource
            reverse=True
        )
        
        primary_forecast = sorted_valid_forecasts[0]
        
        consensus_data = primary_forecast.dict() # Get a dictionary representation

        # Update key fields to indicate consensus
        consensus_data["data_source"] = f"Consensus (Primary: {primary_forecast.data_source})"
        consensus_data["model_info"] = {
            "name": "OpenWeather Consensus Model",
            "version": "1.0",
            "primary_source_model": primary_forecast.model_info,
            "contributing_sources": list(set(f.data_source for f in valid_forecasts))
        }
        
        # Placeholder for more complex field averaging/merging:
        # Example: Averaging temperatures across all daily forecasts
        # if len(valid_forecasts) > 1:
        #     for i in range(len(consensus_data["daily_forecasts"])):
        #         avg_temp_max = sum(f.daily_forecasts[i].temp_max_celsius for f in valid_forecasts if len(f.daily_forecasts) > i) / len(valid_forecasts)
        #         consensus_data["daily_forecasts"][i]["temp_max_celsius"] = round(avg_temp_max, 1)
        #         # Similar for temp_min_celsius, precipitation_mm etc.

        return consensus_data

    async def _get_fresh_cached_data(self, cache_key: str) -> Optional[dict]:
        """Get cached data if it's fresh enough (e.g., within 30 minutes)."""
        cached_item = await self.cache.get(cache_key)
        if cached_item:
            # Assuming cached_item is already a dict from WeatherForecastResponse.dict()
            # We need to check 'generated_at_utc' from the dict.
            generated_at_str = cached_item.get("generated_at_utc")
            if generated_at_str:
                try:
                    generated_at = datetime.fromisoformat(generated_at_str)
                    if datetime.now(timezone.utc) - generated_at < timedelta(minutes=settings.WEATHER_CACHE_TTL_MINUTES): # Using configured TTL
                        logger.info(f"Using fresh cached data for {cache_key}")
                        return cached_item
                    else:
                        logger.info(f"Cached data for {cache_key} is stale.")
                except ValueError:
                    logger.warning(f"Could not parse 'generated_at_utc' from cache for {cache_key}")
            else:
                logger.warning(f"'generated_at_utc' not found in cached item for {cache_key}")
        return None 