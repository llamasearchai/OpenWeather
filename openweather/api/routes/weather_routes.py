"""API routes for weather service operations."""
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field, validator

from openweather.core.monitoring import MetricsCollector, performance_monitor
from openweather.models.weather import Location, WeatherData, ForecastData, WeatherAlert
from openweather.services.weather_service import weather_service, DataProvider, WeatherServiceError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/weather", tags=["Weather"])
metrics = MetricsCollector()

# Request/Response Models
class LocationRequest(BaseModel):
    """Location request model."""
    latitude: float = Field(..., ge=-90, le=90, description="Latitude in degrees")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude in degrees")
    name: Optional[str] = Field(None, description="Location name")

class WeatherResponse(BaseModel):
    """Weather data response model."""
    location: LocationRequest
    timestamp: datetime
    temperature: float
    humidity: int
    pressure: float
    wind_speed: float
    wind_direction: float
    visibility: float
    conditions: Optional[Dict[str, Any]]
    provider: str

class ForecastResponse(BaseModel):
    """Forecast response model."""
    location: LocationRequest
    forecast_time: datetime
    forecasts: List[WeatherResponse]
    provider: str
    total_forecasts: int

class WeatherAlertResponse(BaseModel):
    """Weather alert response model."""
    alert_id: str
    severity: str
    title: str
    description: str
    start_time: datetime
    end_time: Optional[datetime]
    areas: List[str]

class CacheStatsResponse(BaseModel):
    """Cache statistics response model."""
    hits: int
    misses: int
    hit_rate: float
    entries: int
    max_size: int

class ProviderHealthResponse(BaseModel):
    """Provider health status response."""
    provider: str
    status: str
    last_check: datetime
    response_time_ms: Optional[float]

# API Endpoints
@router.get("/current", response_model=WeatherResponse, 
           summary="Get current weather conditions")
@performance_monitor
async def get_current_weather(
    latitude: float = Query(..., ge=-90, le=90, description="Latitude in degrees"),
    longitude: float = Query(..., ge=-180, le=180, description="Longitude in degrees"),
    provider: Optional[DataProvider] = Query(None, description="Weather data provider"),
    name: Optional[str] = Query(None, description="Location name")
):
    """Get current weather conditions for a location."""
    try:
        location = Location(latitude=latitude, longitude=longitude, name=name)
        
        weather_data = await weather_service.get_current_weather(location, provider)
        
        metrics.increment_counter("weather_current_requests")
        
        return WeatherResponse(
            location=LocationRequest(
                latitude=weather_data.location.latitude,
                longitude=weather_data.location.longitude,
                name=weather_data.location.name
            ),
            timestamp=weather_data.timestamp,
            temperature=weather_data.temperature,
            humidity=weather_data.humidity,
            pressure=weather_data.pressure,
            wind_speed=weather_data.wind_speed,
            wind_direction=weather_data.wind_direction,
            visibility=weather_data.visibility,
            conditions=weather_data.conditions.__dict__ if weather_data.conditions else None,
            provider=weather_data.provider
        )
        
    except WeatherServiceError as e:
        logger.error(f"Weather service error: {e}")
        metrics.increment_counter("weather_service_errors")
        raise HTTPException(status_code=503, detail=f"Weather service error: {str(e)}")
    except ValueError as e:
        logger.warning(f"Invalid weather request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error getting current weather: {e}")
        metrics.increment_counter("weather_unexpected_errors")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/forecast", response_model=ForecastResponse,
           summary="Get weather forecast")
@performance_monitor
async def get_weather_forecast(
    latitude: float = Query(..., ge=-90, le=90, description="Latitude in degrees"),
    longitude: float = Query(..., ge=-180, le=180, description="Longitude in degrees"),
    days: int = Query(7, ge=1, le=14, description="Number of forecast days"),
    provider: Optional[DataProvider] = Query(None, description="Weather data provider"),
    name: Optional[str] = Query(None, description="Location name")
):
    """Get weather forecast for a location."""
    try:
        location = Location(latitude=latitude, longitude=longitude, name=name)
        
        forecast_data = await weather_service.get_forecast(location, days, provider)
        
        metrics.increment_counter("weather_forecast_requests")
        
        # Convert forecast data to response format
        forecast_responses = []
        for forecast in forecast_data.forecasts:
            forecast_responses.append(WeatherResponse(
                location=LocationRequest(
                    latitude=forecast.location.latitude,
                    longitude=forecast.location.longitude,
                    name=forecast.location.name
                ),
                timestamp=forecast.timestamp,
                temperature=forecast.temperature,
                humidity=forecast.humidity,
                pressure=forecast.pressure,
                wind_speed=forecast.wind_speed,
                wind_direction=forecast.wind_direction,
                visibility=forecast.visibility,
                conditions=forecast.conditions.__dict__ if forecast.conditions else None,
                provider=forecast.provider
            ))
        
        return ForecastResponse(
            location=LocationRequest(
                latitude=forecast_data.location.latitude,
                longitude=forecast_data.location.longitude,
                name=forecast_data.location.name
            ),
            forecast_time=forecast_data.forecast_time,
            forecasts=forecast_responses,
            provider=forecast_data.provider,
            total_forecasts=len(forecast_responses)
        )
        
    except WeatherServiceError as e:
        logger.error(f"Weather service error: {e}")
        metrics.increment_counter("weather_service_errors")
        raise HTTPException(status_code=503, detail=f"Weather service error: {str(e)}")
    except ValueError as e:
        logger.warning(f"Invalid forecast request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error getting forecast: {e}")
        metrics.increment_counter("weather_unexpected_errors")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/alerts", response_model=List[WeatherAlertResponse],
           summary="Get weather alerts")
@performance_monitor
async def get_weather_alerts(
    latitude: float = Query(..., ge=-90, le=90, description="Latitude in degrees"),
    longitude: float = Query(..., ge=-180, le=180, description="Longitude in degrees"),
    provider: Optional[DataProvider] = Query(None, description="Weather data provider"),
    name: Optional[str] = Query(None, description="Location name")
):
    """Get weather alerts for a location."""
    try:
        location = Location(latitude=latitude, longitude=longitude, name=name)
        
        alerts = await weather_service.get_weather_alerts(location, provider)
        
        metrics.increment_counter("weather_alerts_requests")
        metrics.set_gauge("weather_active_alerts", len(alerts))
        
        # Convert alerts to response format
        alert_responses = []
        for alert in alerts:
            alert_responses.append(WeatherAlertResponse(
                alert_id=alert.id,
                severity=alert.severity,
                title=alert.title,
                description=alert.description,
                start_time=alert.start_time,
                end_time=alert.end_time,
                areas=alert.areas
            ))
        
        return alert_responses
        
    except WeatherServiceError as e:
        logger.error(f"Weather service error: {e}")
        metrics.increment_counter("weather_service_errors")
        raise HTTPException(status_code=503, detail=f"Weather service error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error getting alerts: {e}")
        metrics.increment_counter("weather_unexpected_errors")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/providers", summary="Get available weather providers")
async def get_weather_providers():
    """Get list of available weather data providers."""
    providers = []
    
    for provider in DataProvider:
        provider_info = {
            "name": provider.value,
            "display_name": provider.value.replace("_", " ").title(),
            "available": provider in weather_service.providers
        }
        providers.append(provider_info)
    
    return {
        "providers": providers,
        "default_provider": weather_service.default_provider.value,
        "total_providers": len(providers),
        "active_providers": len(weather_service.providers)
    }

@router.get("/providers/health", response_model=List[ProviderHealthResponse],
           summary="Check weather provider health")
@performance_monitor
async def check_provider_health():
    """Check health status of all weather providers."""
    try:
        health_status = await weather_service.health_check()
        
        health_responses = []
        for provider_name, is_healthy in health_status.items():
            health_responses.append(ProviderHealthResponse(
                provider=provider_name,
                status="healthy" if is_healthy else "unhealthy",
                last_check=datetime.utcnow(),
                response_time_ms=None  # Would need to measure this
            ))
        
        metrics.increment_counter("weather_health_checks")
        
        return health_responses
        
    except Exception as e:
        logger.error(f"Error checking provider health: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@router.get("/cache/stats", response_model=CacheStatsResponse,
           summary="Get weather cache statistics")
@performance_monitor
async def get_cache_stats():
    """Get weather service cache statistics."""
    try:
        cache_stats = weather_service.get_cache_stats()
        
        return CacheStatsResponse(**cache_stats)
        
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get cache statistics")

@router.delete("/cache", summary="Clear weather cache")
@performance_monitor
async def clear_weather_cache():
    """Clear all weather service caches."""
    try:
        weather_service.clear_cache()
        
        metrics.increment_counter("weather_cache_clears")
        
        return {
            "status": "success",
            "message": "Weather cache cleared successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear cache")

@router.get("/historical", summary="Get historical weather data")
@performance_monitor
async def get_historical_weather(
    latitude: float = Query(..., ge=-90, le=90, description="Latitude in degrees"),
    longitude: float = Query(..., ge=-180, le=180, description="Longitude in degrees"),
    start_date: datetime = Query(..., description="Start date for historical data"),
    end_date: datetime = Query(..., description="End date for historical data"),
    name: Optional[str] = Query(None, description="Location name")
):
    """Get historical weather data for a location."""
    try:
        location = Location(latitude=latitude, longitude=longitude, name=name)
        
        # Validate date range
        if start_date >= end_date:
            raise ValueError("Start date must be before end date")
        
        if end_date > datetime.utcnow():
            raise ValueError("End date cannot be in the future")
        
        historical_data = await weather_service.get_historical_weather(
            location, start_date, end_date
        )
        
        metrics.increment_counter("weather_historical_requests")
        
        # Convert to response format
        weather_responses = []
        for data in historical_data:
            weather_responses.append(WeatherResponse(
                location=LocationRequest(
                    latitude=data.location.latitude,
                    longitude=data.location.longitude,
                    name=data.location.name
                ),
                timestamp=data.timestamp,
                temperature=data.temperature,
                humidity=data.humidity,
                pressure=data.pressure,
                wind_speed=data.wind_speed,
                wind_direction=data.wind_direction,
                visibility=data.visibility,
                conditions=data.conditions.__dict__ if data.conditions else None,
                provider=data.provider
            ))
        
        return {
            "location": LocationRequest(
                latitude=latitude,
                longitude=longitude,
                name=name
            ),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "data_points": len(weather_responses),
            "historical_data": weather_responses
        }
        
    except ValueError as e:
        logger.warning(f"Invalid historical request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error getting historical data: {e}")
        metrics.increment_counter("weather_unexpected_errors")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/stats", summary="Get weather service statistics")
@performance_monitor
async def get_weather_stats():
    """Get comprehensive weather service statistics."""
    try:
        cache_stats = weather_service.get_cache_stats()
        
        stats = {
            "service_info": {
                "active_providers": len(weather_service.providers),
                "default_provider": weather_service.default_provider.value,
                "total_providers_available": len(DataProvider)
            },
            "cache_statistics": cache_stats,
            "request_metrics": {
                "current_weather_requests": metrics.counters.get("weather_current_requests", 0),
                "forecast_requests": metrics.counters.get("weather_forecast_requests", 0),
                "alerts_requests": metrics.counters.get("weather_alerts_requests", 0),
                "historical_requests": metrics.counters.get("weather_historical_requests", 0),
                "service_errors": metrics.counters.get("weather_service_errors", 0),
                "cache_clears": metrics.counters.get("weather_cache_clears", 0)
            },
            "provider_status": await weather_service.health_check()
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting weather stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get weather statistics") 