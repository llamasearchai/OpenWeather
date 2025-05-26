"""Shared data models for the OpenWeather application."""
from enum import Enum
from typing import Optional, List, Dict, Any, Literal
from datetime import date, datetime, timezone
from pydantic import BaseModel, Field, field_validator, AwareDatetime
from pydantic_core import MultiHostUrl
from typing_extensions import Annotated

# Coordinate and location models
class Coordinate(BaseModel):
    """Geographic coordinate."""
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)

class LocationInfo(BaseModel):
    """Location information."""
    name: str
    coordinates: Coordinate
    country_code: Optional[str] = None
    timezone: Optional[str] = None

# Weather condition codes from WMO Weather Interpretation
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

class WeatherCondition(Enum):
    """Weather condition types."""
    CLEAR = "clear"
    PARTLY_CLOUDY = "partly_cloudy"
    CLOUDY = "cloudy"
    RAIN = "rain"
    SNOW = "snow"
    STORM = "storm"
    FOG = "fog"
    EXTREME = "extreme"

class DailyForecast(BaseModel):
    """Daily weather forecast data."""
    date: date
    temp_max_celsius: Optional[float] = None
    temp_min_celsius: Optional[float] = None
    condition_code: Optional[int] = None
    condition_text: Optional[str] = None
    precipitation_mm: Optional[float] = Field(None, ge=0)
    precipitation_chance_percent: Optional[int] = Field(None, ge=0, le=100)
    wind_speed_kph: Optional[float] = None
    wind_gust_kph: Optional[float] = None
    wind_direction_degrees: Optional[int] = None
    wind_direction_cardinal: Optional[str] = None
    humidity_percent: Optional[int] = Field(None, ge=0, le=100)
    uv_index: Optional[float] = Field(None, ge=0)
    sunrise_utc: Optional[AwareDatetime] = None
    sunset_utc: Optional[AwareDatetime] = None
    moonrise_utc: Optional[AwareDatetime] = None
    moonset_utc: Optional[AwareDatetime] = None
    moon_phase: Optional[str] = None
    detailed_summary: Optional[str] = None

class CurrentWeather(BaseModel):
    """Current weather conditions."""
    observed_at_utc: AwareDatetime
    temp_celsius: float
    feels_like_celsius: Optional[float] = None
    condition_code: Optional[int] = None
    condition_text: Optional[str] = None
    wind_speed_kph: Optional[float] = None
    wind_direction_degrees: Optional[int] = None
    wind_direction_cardinal: Optional[str] = None
    pressure_mb: Optional[float] = None
    precipitation_mm_last_hour: Optional[float] = Field(None, ge=0)
    humidity_percent: Optional[int] = Field(None, ge=0, le=100)
    cloud_cover_percent: Optional[int] = Field(None, ge=0, le=100)
    visibility_km: Optional[float] = Field(None, ge=0)
    uv_index: Optional[float] = Field(None, ge=0)

class WeatherForecastResponse(BaseModel):
    """Complete weather forecast response."""
    location: LocationInfo
    current_weather: Optional[CurrentWeather] = None
    daily_forecasts: List[DailyForecast]
    generated_at_utc: AwareDatetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    data_source: str
    model_info: Optional[Dict[str, Any]] = None

class LLMAnalysisRequest(BaseModel):
    """Request for LLM analysis of weather data."""
    forecast_data: WeatherForecastResponse
    query: Optional[str] = None
    llm_provider: Optional[str] = None
    llm_model_name: Optional[str] = None
    output_format: Literal["markdown", "json", "text"] = "markdown"
    custom_prompt_template: Optional[str] = None

class LLMAnalysisResponse(BaseModel):
    """Response from LLM analysis of weather data."""
    request_details: LLMAnalysisRequest
    analysis_text: str
    provider_used: str
    model_used: str
    generated_at_utc: AwareDatetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    tokens_used: Optional[Dict[str, int]] = None
    error_message: Optional[str] = None

class AlertType(Enum):
    """Types of weather alerts."""
    WEATHER = "weather"
    CLIMATE = "climate"
    ENVIRONMENTAL = "environmental"

class AlertSeverity(Enum):
    """Severity levels for weather alerts."""
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    EXTREME = "extreme"

class Alert(BaseModel):
    """Weather alert information."""
    event_name: str
    start_time_utc: AwareDatetime
    end_time_utc: AwareDatetime
    severity: Literal["minor", "moderate", "severe", "extreme"]
    description: str
    issuing_authority: Optional[str] = None
    affected_areas_summary: Optional[str] = None

class WeatherAlertResponse(BaseModel):
    """Weather alerts response."""
    location: LocationInfo
    alerts: List[Alert]
    generated_at_utc: AwareDatetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    data_source: Optional[str] = None

class ForecastAndAnalysisApiResponse(WeatherForecastResponse):
    """Combined forecast and analysis response for API."""
    llm_analysis: Optional[LLMAnalysisResponse] = None 