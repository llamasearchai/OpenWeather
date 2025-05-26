"""Weather service for comprehensive weather data management."""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib

import httpx
from pydantic import BaseModel, Field

from openweather.core.config import settings
from openweather.core.monitoring import MetricsCollector, performance_monitor
from openweather.models.weather import (
    WeatherData, 
    ForecastData, 
    WeatherAlert,
    WeatherConditions,
    Location
)

logger = logging.getLogger(__name__)

class DataProvider(str, Enum):
    """Weather data providers."""
    OPENWEATHER = "openweather"
    WEATHER_API = "weather_api"
    NATIONAL_WEATHER_SERVICE = "nws"
    VISUAL_CROSSING = "visual_crossing"

class WeatherServiceError(Exception):
    """Weather service specific errors."""
    pass

class DataProviderError(WeatherServiceError):
    """Data provider specific errors."""
    pass

@dataclass
class CacheEntry:
    """Cache entry for weather data."""
    data: Union[WeatherData, ForecastData, List[WeatherAlert]]
    timestamp: datetime
    ttl: int
    provider: DataProvider
    location_key: str

    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return datetime.utcnow() > self.timestamp + timedelta(seconds=self.ttl)

class WeatherCache:
    """In-memory cache for weather data with TTL support."""
    
    def __init__(self, max_size: int = 1000):
        self._cache: Dict[str, CacheEntry] = {}
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
    
    def _generate_key(self, provider: DataProvider, location: Location, 
                     data_type: str, **kwargs) -> str:
        """Generate cache key."""
        key_data = {
            "provider": provider.value,
            "lat": location.latitude,
            "lon": location.longitude,
            "type": data_type,
            **kwargs
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, provider: DataProvider, location: Location, 
            data_type: str, **kwargs) -> Optional[Union[WeatherData, ForecastData, List[WeatherAlert]]]:
        """Get cached data."""
        key = self._generate_key(provider, location, data_type, **kwargs)
        
        if key in self._cache:
            entry = self._cache[key]
            if not entry.is_expired():
                self._hits += 1
                return entry.data
            else:
                # Remove expired entry
                del self._cache[key]
        
        self._misses += 1
        return None
    
    def set(self, provider: DataProvider, location: Location, data_type: str,
            data: Union[WeatherData, ForecastData, List[WeatherAlert]], 
            ttl: int = 300, **kwargs) -> None:
        """Set cached data."""
        if len(self._cache) >= self._max_size:
            # Remove oldest entry
            oldest_key = min(self._cache.keys(), 
                           key=lambda k: self._cache[k].timestamp)
            del self._cache[oldest_key]
        
        key = self._generate_key(provider, location, data_type, **kwargs)
        self._cache[key] = CacheEntry(
            data=data,
            timestamp=datetime.utcnow(),
            ttl=ttl,
            provider=provider,
            location_key=f"{location.latitude},{location.longitude}"
        )
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
    
    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0
        
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "entries": len(self._cache),
            "max_size": self._max_size
        }

class WeatherDataProvider:
    """Base class for weather data providers."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=30.0)
        self.metrics = MetricsCollector()
    
    async def get_current_weather(self, location: Location) -> WeatherData:
        """Get current weather data."""
        raise NotImplementedError
    
    async def get_forecast(self, location: Location, days: int = 7) -> ForecastData:
        """Get weather forecast."""
        raise NotImplementedError
    
    async def get_alerts(self, location: Location) -> List[WeatherAlert]:
        """Get weather alerts."""
        raise NotImplementedError
    
    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()

class OpenWeatherMapProvider(WeatherDataProvider):
    """OpenWeatherMap API provider."""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://api.openweathermap.org/data/2.5"
    
    @performance_monitor
    async def get_current_weather(self, location: Location) -> WeatherData:
        """Get current weather from OpenWeatherMap."""
        try:
            url = f"{self.base_url}/weather"
            params = {
                "lat": location.latitude,
                "lon": location.longitude,
                "appid": self.api_key,
                "units": "metric"
            }
            
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            return WeatherData(
                location=location,
                timestamp=datetime.utcnow(),
                temperature=data["main"]["temp"],
                humidity=data["main"]["humidity"],
                pressure=data["main"]["pressure"],
                wind_speed=data["wind"].get("speed", 0),
                wind_direction=data["wind"].get("deg", 0),
                visibility=data.get("visibility", 10000) / 1000,  # Convert to km
                conditions=WeatherConditions(
                    main=data["weather"][0]["main"],
                    description=data["weather"][0]["description"],
                    cloud_cover=data["clouds"]["all"]
                ),
                provider=DataProvider.OPENWEATHER.value
            )
            
        except httpx.HTTPError as e:
            self.metrics.increment_counter("openweather_api_errors")
            raise DataProviderError(f"OpenWeatherMap API error: {e}")
        except KeyError as e:
            raise DataProviderError(f"Invalid OpenWeatherMap response format: {e}")
    
    @performance_monitor
    async def get_forecast(self, location: Location, days: int = 7) -> ForecastData:
        """Get forecast from OpenWeatherMap."""
        try:
            url = f"{self.base_url}/forecast"
            params = {
                "lat": location.latitude,
                "lon": location.longitude,
                "appid": self.api_key,
                "units": "metric"
            }
            
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            forecasts = []
            for item in data["list"][:days * 8]:  # 8 forecasts per day (3-hour intervals)
                forecast_time = datetime.fromtimestamp(item["dt"])
                
                forecasts.append(WeatherData(
                    location=location,
                    timestamp=forecast_time,
                    temperature=item["main"]["temp"],
                    humidity=item["main"]["humidity"],
                    pressure=item["main"]["pressure"],
                    wind_speed=item["wind"].get("speed", 0),
                    wind_direction=item["wind"].get("deg", 0),
                    visibility=10.0,  # Default visibility
                    conditions=WeatherConditions(
                        main=item["weather"][0]["main"],
                        description=item["weather"][0]["description"],
                        cloud_cover=item["clouds"]["all"]
                    ),
                    provider=DataProvider.OPENWEATHER.value
                ))
            
            return ForecastData(
                location=location,
                forecast_time=datetime.utcnow(),
                forecasts=forecasts,
                provider=DataProvider.OPENWEATHER.value
            )
            
        except httpx.HTTPError as e:
            self.metrics.increment_counter("openweather_forecast_errors")
            raise DataProviderError(f"OpenWeatherMap forecast API error: {e}")

class WeatherService:
    """Comprehensive weather service with multiple providers and caching."""
    
    def __init__(self):
        self.cache = WeatherCache(max_size=2000)
        self.providers: Dict[DataProvider, WeatherDataProvider] = {}
        self.metrics = MetricsCollector()
        self.default_provider = DataProvider.OPENWEATHER
        self._setup_providers()
    
    def _setup_providers(self) -> None:
        """Setup weather data providers."""
        # OpenWeatherMap provider
        if hasattr(settings, 'OPENWEATHER_API_KEY') and settings.OPENWEATHER_API_KEY:
            self.providers[DataProvider.OPENWEATHER] = OpenWeatherMapProvider(
                api_key=str(settings.OPENWEATHER_API_KEY)
            )
    
    def add_provider(self, provider_type: DataProvider, provider: WeatherDataProvider) -> None:
        """Add a weather data provider."""
        self.providers[provider_type] = provider
    
    @performance_monitor
    async def get_current_weather(self, location: Location, 
                                provider: Optional[DataProvider] = None) -> WeatherData:
        """Get current weather with caching and failover."""
        provider = provider or self.default_provider
        
        # Try cache first
        cached_data = self.cache.get(provider, location, "current")
        if cached_data:
            self.metrics.increment_counter("weather_cache_hits")
            return cached_data
        
        # Get from provider
        if provider not in self.providers:
            raise WeatherServiceError(f"Provider {provider} not available")
        
        try:
            weather_data = await self.providers[provider].get_current_weather(location)
            
            # Cache the result
            self.cache.set(provider, location, "current", weather_data, ttl=300)
            self.metrics.increment_counter("weather_api_calls")
            
            return weather_data
            
        except DataProviderError as e:
            self.metrics.increment_counter("weather_provider_errors")
            # Try fallback providers
            for fallback_provider in self.providers:
                if fallback_provider != provider:
                    try:
                        logger.warning(f"Trying fallback provider {fallback_provider}")
                        weather_data = await self.providers[fallback_provider].get_current_weather(location)
                        self.cache.set(fallback_provider, location, "current", weather_data, ttl=300)
                        return weather_data
                    except DataProviderError:
                        continue
            
            raise WeatherServiceError(f"All weather providers failed: {e}")
    
    @performance_monitor
    async def get_forecast(self, location: Location, days: int = 7,
                          provider: Optional[DataProvider] = None) -> ForecastData:
        """Get weather forecast with caching."""
        provider = provider or self.default_provider
        
        # Try cache first
        cached_data = self.cache.get(provider, location, "forecast", days=days)
        if cached_data:
            self.metrics.increment_counter("forecast_cache_hits")
            return cached_data
        
        # Get from provider
        if provider not in self.providers:
            raise WeatherServiceError(f"Provider {provider} not available")
        
        try:
            forecast_data = await self.providers[provider].get_forecast(location, days)
            
            # Cache the result (longer TTL for forecasts)
            self.cache.set(provider, location, "forecast", forecast_data, ttl=1800, days=days)
            self.metrics.increment_counter("forecast_api_calls")
            
            return forecast_data
            
        except DataProviderError as e:
            self.metrics.increment_counter("forecast_provider_errors")
            raise WeatherServiceError(f"Forecast provider failed: {e}")
    
    async def get_weather_alerts(self, location: Location,
                               provider: Optional[DataProvider] = None) -> List[WeatherAlert]:
        """Get weather alerts for location."""
        provider = provider or self.default_provider
        
        # Try cache first
        cached_data = self.cache.get(provider, location, "alerts")
        if cached_data:
            self.metrics.increment_counter("alerts_cache_hits")
            return cached_data
        
        # Get from provider
        if provider not in self.providers:
            return []  # Return empty list if no provider available
        
        try:
            alerts = await self.providers[provider].get_alerts(location)
            
            # Cache the result
            self.cache.set(provider, location, "alerts", alerts, ttl=600)
            self.metrics.increment_counter("alerts_api_calls")
            
            return alerts
            
        except DataProviderError as e:
            logger.warning(f"Failed to get weather alerts: {e}")
            return []
    
    async def get_historical_weather(self, location: Location, 
                                   start_date: datetime, end_date: datetime) -> List[WeatherData]:
        """Get historical weather data."""
        # This would typically require a different API endpoint or service
        # For now, return empty list as placeholder
        logger.info(f"Historical weather requested for {location} from {start_date} to {end_date}")
        return []
    
    def get_cache_stats(self) -> Dict[str, Union[int, float]]:
        """Get cache statistics."""
        return self.cache.get_stats()
    
    def clear_cache(self) -> None:
        """Clear weather cache."""
        self.cache.clear()
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all weather providers."""
        health_status = {}
        
        for provider_type, provider in self.providers.items():
            try:
                # Try a simple location for health check
                test_location = Location(latitude=40.7128, longitude=-74.0060)  # NYC
                await provider.get_current_weather(test_location)
                health_status[provider_type.value] = True
            except Exception:
                health_status[provider_type.value] = False
        
        return health_status
    
    async def close(self) -> None:
        """Close all providers."""
        for provider in self.providers.values():
            await provider.close()

# Global weather service instance
weather_service = WeatherService() 