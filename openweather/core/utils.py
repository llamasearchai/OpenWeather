"""Utility functions for the OpenWeather application."""
import asyncio
import logging
import math
import platform
import sys
from enum import StrEnum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import date, datetime, timezone
from random import gauss

import httpx
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
from rich import print as rprint

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def get_platform_info() -> Dict[str, str]:
    """Get detailed system platform information."""
    return {
        "python_version": sys.version.split(" ")[0],
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor()
    }

def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon (M1/M2/M3)."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"

def parse_location_string(location_str: str) -> Tuple[float, float, str]:
    """Parse location string into latitude, longitude, and location name."""
    location_str = location_str.strip()
    
    # Predefined city coordinates
    CITY_COORDINATES = {
        "london": (51.5074, -0.1278, "London"),
        "new york": (40.7128, -74.0060, "New York"),
        "tokyo": (35.6895, 139.6917, "Tokyo"),
        "paris": (48.8566, 2.3522, "Paris"),
        "berlin": (52.5200, 13.4050, "Berlin"),
        "bangalore": (12.9716, 77.5946, "Bangalore"),
        "san francisco": (37.7749, -122.4194, "San Francisco"),
        "dubai": (25.2048, 55.2708, "Dubai"),
        "sydney": (-33.8688, 151.2093, "Sydney"),
        "moscow": (55.7558, 37.6173, "Moscow")
    }
    
    # Try parsing as lat,lon
    try:
        lat, lon = map(float, location_str.split(","))
        return lat, lon, f"Lat: {lat}, Lon: {lon}"
    except ValueError:
        pass
    
    # Try matching city name
    location_str_lower = location_str.lower()
    if location_str_lower in CITY_COORDINATES:
        return CITY_COORDINATES[location_str_lower]
    
    # Default to London if no match found
    logger.warning(f"Location '{location_str}' not found, defaulting to London")
    return CITY_COORDINATES["london"]

def _degrees_to_cardinal(degrees: int) -> str:
    """Convert wind direction degrees to cardinal direction."""
    if degrees is None:
        return ""
    
    directions = [
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"
    ]
    
    # Calculate index in the directions list
    index = round(degrees / 22.5) % 16
    return directions[index]

def format_weather_data_for_llm(weather_data: Dict, location_name: Optional[str] = None) -> str:
    """Convert weather data to structured string for LLM prompts."""
    location = location_name or weather_data["location"]["name"]
    
    # Format header with location information
    header = f"Weather Forecast for {location}\n"
    header += f"Coordinates: {weather_data['location']['coordinates']['latitude']:.2f}, {weather_data['location']['coordinates']['longitude']:.2f}\n"
    header += f"Data Source: {weather_data['data_source']}\n"
    
    # Format current weather
    current_section = ""
    if "current_weather" in weather_data and weather_data["current_weather"]:
        cw = weather_data["current_weather"]
        current_section = "\nCurrent Weather:\n"
        current_section += f"- Temperature: {cw['temp_celsius']:.1f}°C"
        if "feels_like_celsius" in cw and cw["feels_like_celsius"]:
            current_section += f" (feels like {cw['feels_like_celsius']:.1f}°C)"
        current_section += "\n"
        
        if "condition_text" in cw and cw["condition_text"]:
            current_section += f"- Condition: {cw['condition_text']}\n"
        if "wind_speed_kph" in cw and cw["wind_speed_kph"]:
            wind_info = f"- Wind: {cw['wind_speed_kph']:.1f} km/h"
            if "wind_direction_cardinal" in cw and cw["wind_direction_cardinal"]:
                wind_info += f" from {cw['wind_direction_cardinal']}"
            current_section += wind_info + "\n"
        if "humidity_percent" in cw and cw["humidity_percent"]:
            current_section += f"- Humidity: {cw['humidity_percent']}%\n"
        if "pressure_mb" in cw and cw["pressure_mb"]:
            current_section += f"- Pressure: {cw['pressure_mb']} mb\n"
        if "visibility_km" in cw and cw["visibility_km"]:
            current_section += f"- Visibility: {cw['visibility_km']} km\n"
    
    # Format daily forecasts
    forecast_section = "\nDaily Forecasts:\n"
    for i, df in enumerate(weather_data["daily_forecasts"]):
        day_str = "Today" if i == 0 else "Tomorrow" if i == 1 else datetime.strptime(df["date"], "%Y-%m-%d").strftime("%A, %B %d") if isinstance(df["date"], str) else df["date"].strftime("%A, %B %d")
        forecast_section += f"\n{day_str}:\n"
        forecast_section += f"- Temperature: {df['temp_min_celsius']:.1f}°C to {df['temp_max_celsius']:.1f}°C\n"
        forecast_section += f"- Condition: {df['condition_text']}\n"
        
        if "precipitation_mm" in df and df["precipitation_mm"] is not None and df["precipitation_mm"] > 0:
            forecast_section += f"- Precipitation: {df['precipitation_mm']:.1f}mm"
            if "precipitation_chance_percent" in df and df["precipitation_chance_percent"] is not None:
                forecast_section += f" ({df['precipitation_chance_percent']}% chance)"
            forecast_section += "\n"
        
        if "wind_speed_kph" in df and df["wind_speed_kph"]:
            forecast_section += f"- Wind: {df['wind_speed_kph']:.1f} km/h"
            if "wind_direction_cardinal" in df and df["wind_direction_cardinal"]:
                forecast_section += f" from {df['wind_direction_cardinal']}"
            forecast_section += "\n"
            
        if "uv_index" in df and df["uv_index"]:
            forecast_section += f"- UV Index: {df['uv_index']:.1f}\n"
            
        if "detailed_summary" in df and df["detailed_summary"]:
            forecast_section += f"- Summary: {df['detailed_summary']}\n"
    
    # Combine all sections
    formatted_data = f"{header}\n{current_section}\n{forecast_section}"
    
    return formatted_data

def _format_temperature_rich(temp: Optional[float]) -> Text:
    """Format temperature with color based on value."""
    if temp is None:
        return Text("-", style="dim")
    
    text = Text(f"{temp:.1f}" if isinstance(temp, float) else f"{temp}")
    
    if temp < 0:
        text.stylize("bold bright_blue")
    elif temp < 10:
        text.stylize("blue")
    elif temp < 20:
        text.stylize("green")
    elif temp < 30:
        text.stylize("yellow")
    else:
        text.stylize("bold red")
        
    return text

def setup_logging(log_level_str: str) -> None:
    """Configure application logging with specified log level."""
    # Convert log_level_str to logging module constant
    level = getattr(logging, log_level_str.upper(), logging.INFO)
    
    # Remove any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler with format
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger.addHandler(handler)
    root_logger.setLevel(level)
    
    # Set higher log level for noisy libraries unless in DEBUG
    if level > logging.DEBUG:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("uvicorn").setLevel(logging.WARNING)
    
    logger.info("Logging configured with level: %s", log_level_str) 