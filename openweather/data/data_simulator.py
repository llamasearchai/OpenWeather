"""Weather data simulator for offline or testing use."""
import logging
import math
from datetime import date, timedelta, datetime, timezone
from random import choice, uniform, randint, gauss
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

from openweather.core.utils import parse_location_string, _degrees_to_cardinal
from openweather.core.models_shared import (
    Coordinate, LocationInfo, DailyForecast, CurrentWeather, WeatherForecastResponse
)

logger = logging.getLogger(__name__)

# Simulated weather conditions
WEATHER_CONDITIONS_SIM = [
    (100, "Clear"),
    (200, "Partly Cloudy"),
    (300, "Cloudy"),
    (400, "Light Rain"),
    (401, "Rain"),
    (402, "Heavy Rain"),
    (500, "Light Snow"),
    (501, "Snow"),
    (502, "Heavy Snow"),
    (600, "Thunderstorm"),
    (700, "Fog"),
    (800, "Windy"),
    (900, "Extreme Weather")
]

def _map_condition_code_to_text_sim(code: int) -> str:
    """Map simulated condition code to text description."""
    return next((desc for c, desc in WEATHER_CONDITIONS_SIM if c == code), "Unknown")

def generate_simulated_daily_forecast(
    current_date_obj: date,
    day_offset: int,
    base_temp_celsius: float
) -> DailyForecast:
    """Generate a simulated daily forecast with realistic variations."""
    forecast_date = current_date_obj + timedelta(days=day_offset)
    
    # Base temperature with seasonal variation
    day_of_year = forecast_date.timetuple().tm_yday
    seasonal_effect = 10 * math.sin(2 * math.pi * (day_of_year - 172) / 365)
    
    # Daily variations
    day_temp_base = base_temp_celsius + seasonal_effect
    temp_range = 10 + abs(gauss(0, 2))
    
    temp_min = round(day_temp_base - temp_range/2 + gauss(0, 1), 1)
    temp_max = round(day_temp_base + temp_range/2 + gauss(0, 1), 1)
    
    # Weather condition
    condition_code = choice([c for c, _ in WEATHER_CONDITIONS_SIM])
    condition_text = _map_condition_code_to_text_sim(condition_code)
    
    # Precipitation
    precipitation_chance = randint(0, 100)
    precipitation_amount = 0
    if precipitation_chance > 70:
        precipitation_amount = round(uniform(0.5, 5.0), 1)
    elif precipitation_chance > 30:
        precipitation_amount = round(uniform(0.1, 0.5), 1)
        
    # Wind
    wind_speed = round(uniform(5, 30), 1)
    wind_direction_degrees = randint(0, 359)
    wind_direction_cardinal = _degrees_to_cardinal(wind_direction_degrees)
    
    # UV index
    uv_index = round(uniform(0, 8) if forecast_date.month in range(3, 10) else uniform(0, 3), 1)
    
    # Sunrise and sunset (simplified)
    sunrise_hour = 6 + randint(-1, 1)
    sunset_hour = 18 + randint(-1, 1)
    sunrise_utc = datetime.combine(forecast_date, datetime.min.time()).replace(tzinfo=timezone.utc).replace(hour=sunrise_hour)
    sunset_utc = datetime.combine(forecast_date, datetime.min.time()).replace(tzinfo=timezone.utc).replace(hour=sunset_hour)
    
    return DailyForecast(
        date=forecast_date,
        temp_max_celsius=temp_max,
        temp_min_celsius=temp_min,
        condition_code=condition_code,
        condition_text=condition_text,
        precipitation_mm=precipitation_amount,
        precipitation_chance_percent=precipitation_chance,
        wind_speed_kph=wind_speed,
        wind_direction_degrees=wind_direction_degrees,
        wind_direction_cardinal=wind_direction_cardinal,
        uv_index=uv_index,
        sunrise_utc=sunrise_utc,
        sunset_utc=sunset_utc,
        detailed_summary=f"Simulated {condition_text} with temperatures between {temp_min}°C and {temp_max}°C"
    )

def generate_simulated_current_weather(
    location_info: LocationInfo,
    base_temp_celsius: float
) -> CurrentWeather:
    """Generate simulated current weather data."""
    current_date = datetime.now(timezone.utc)
    
    # Current temperature with random variation
    current_temp = round(base_temp_celsius + gauss(0, 1), 1)
    feels_like = round(current_temp + uniform(-2, 2), 1)
    
    # Weather condition
    condition_code = choice([c for c, _ in WEATHER_CONDITIONS_SIM])
    condition_text = _map_condition_code_to_text_sim(condition_code)
    
    # Wind
    wind_speed = round(uniform(5, 30), 1)
    wind_gust = round(wind_speed * uniform(1.2, 2.0), 1) if wind_speed > 10 else None
    wind_direction_degrees = randint(0, 359)
    wind_direction_cardinal = _degrees_to_cardinal(wind_direction_degrees)
    
    # Other weather data
    pressure_mb = round(uniform(980, 1040), 1)
    precipitation_last_hour = 0.0
    
    # Determine precipitation based on condition
    if "Rain" in condition_text:
        precipitation_last_hour = round(uniform(0.1, 5.0), 1)
    elif "Snow" in condition_text:
        precipitation_last_hour = round(uniform(0.1, 3.0), 1)
        
    # Cloud cover and visibility
    cloud_cover = randint(0, 100)
    visibility = round(uniform(0.1, 10.0), 1) if cloud_cover > 80 else round(uniform(2.0, 10.0), 1)
    
    # Humidity
    humidity = randint(40, 95)
    
    return CurrentWeather(
        observed_at_utc=current_date,
        temp_celsius=current_temp,
        feels_like_celsius=feels_like,
        condition_code=condition_code,
        condition_text=condition_text,
        wind_speed_kph=wind_speed,
        wind_direction_degrees=wind_direction_degrees,
        wind_direction_cardinal=wind_direction_cardinal,
        pressure_mb=pressure_mb,
        precipitation_mm_last_hour=precipitation_last_hour,
        humidity_percent=humidity,
        cloud_cover_percent=cloud_cover,
        visibility_km=visibility,
        uv_index=round(uniform(0, 8), 1) if current_date.month in range(3, 10) else round(uniform(0, 3), 1)
    )

def get_simulated_weather_forecast(
    location_str: str,
    num_days: int
) -> WeatherForecastResponse:
    """Generate a simulated weather forecast response."""
    # Parse location string
    latitude, longitude, resolved_name = parse_location_string(location_str)
    
    # Generate base temperature from latitude
    base_temp_celsius = round(25 - abs(latitude) / 4 + gauss(0, 2), 1)
    
    # Create location info
    location_info = LocationInfo(
        name=resolved_name,
        coordinates=Coordinate(latitude=latitude, longitude=longitude)
    )
    
    # Generate current weather
    current_weather = generate_simulated_current_weather(location_info, base_temp_celsius)
    
    # Generate forecasts for each day
    current_date = datetime.now(timezone.utc).date()
    daily_forecasts = [
        generate_simulated_daily_forecast(current_date, day_offset, base_temp_celsius)
        for day_offset in range(num_days)
    ]
    
    return WeatherForecastResponse(
        location=location_info,
        current_weather=current_weather,
        daily_forecasts=daily_forecasts,
        data_source="Local Simulation",
        model_info={"name": "BasicSimulator", "version": "1.0"}
    )

if __name__ == "__main__":
    import asyncio
    import json
    
    async def main():
        # Example usage
        forecast = get_simulated_weather_forecast("London", 7)
        print(forecast.model_dump_json(indent=2))
        
    asyncio.run(main()) 