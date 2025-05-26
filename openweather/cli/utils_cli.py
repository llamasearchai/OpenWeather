"""Utility functions for the OpenWeather CLI."""
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
from rich import print as rprint

from openweather.core.models_shared import WeatherForecastResponse, DailyForecast, CurrentWeather, LLMAnalysisResponse
from openweather.core.utils import _format_temperature_rich, _degrees_to_cardinal

logger = logging.getLogger(__name__)
console = Console()

def display_forecast_rich(
    forecast_data: WeatherForecastResponse,
    llm_analysis: Optional[LLMAnalysisResponse] = None,
    raw_json: bool = False
):
    """Display weather forecast using Rich library for beautiful CLI output."""
    if raw_json:
        rprint(forecast_data.model_dump_json(indent=2))
        if llm_analysis:
            rprint(llm_analysis.model_dump_json(indent=2))
        return

    # Location panel
    location_info = forecast_data.location
    loc_panel_content = Text(f"{location_info.name}\n", style="bold white on blue")
    loc_panel_content.append(f"Lat: {location_info.coordinates.latitude:.2f}, Lon: {location_info.coordinates.longitude:.2f}\n")
    loc_panel_content.append(f"Source: {forecast_data.data_source}")
    if forecast_data.model_info:
        model_name = forecast_data.model_info.get("name", "Unknown Model")
        model_version = forecast_data.model_info.get("version", "N/A")
        loc_panel_content.append(f" | Model: {model_name} v{model_version}")
    rprint(Panel(loc_panel_content, title="Location", expand=False))

    # Current weather (if available)
    if forecast_data.current_weather:
        cw = forecast_data.current_weather
        current_table = Table(title="Current Weather", show_header=False, box=None)
        current_table.add_column(style="cyan")
        current_table.add_column()
        
        current_table.add_row("Temperature:", _format_temperature_rich(cw.temp_celsius).append(" °C"))
        if cw.feels_like_celsius:
            current_table.add_row("Feels Like:", _format_temperature_rich(cw.feels_like_celsius).append(" °C"))
        if cw.condition_text:
            current_table.add_row("Condition:", Text(cw.condition_text))
        if cw.wind_speed_kph:
            wind_str = f"{cw.wind_speed_kph:.1f} km/h"
            if cw.wind_direction_cardinal:
                wind_str += f" {cw.wind_direction_cardinal}"
            current_table.add_row("Wind:", Text(wind_str))
        if cw.humidity_percent is not None:
            current_table.add_row("Humidity:", Text(f"{cw.humidity_percent}%"))
        if cw.pressure_mb:
            current_table.add_row("Pressure:", Text(f"{cw.pressure_mb:.0f} mb"))
        if cw.visibility_km:
            current_table.add_row("Visibility:", Text(f"{cw.visibility_km} km"))
        rprint(current_table)

    # Daily forecasts table
    daily_table = Table(title="Daily Forecasts", expand=True)
    daily_table.add_column("Date", style="magenta")
    daily_table.add_column("Condition", style="green")
    daily_table.add_column("Temp (Min/Max)", style="cyan", justify="right")
    daily_table.add_column("Precipitation (mm / %)", style="blue", justify="right")
    daily_table.add_column("Wind (km/h)", style="yellow", justify="right")
    daily_table.add_column("UV", style="red", justify="right")

    for i, day in enumerate(forecast_data.daily_forecasts):
        date_str = "Today" if i == 0 else "Tomorrow" if i == 1 else day.date.strftime("%a, %b %d")
        
        temp_str = Text()
        temp_str.append(_format_temperature_rich(day.temp_min_celsius)).append(" / ")
        temp_str.append(_format_temperature_rich(day.temp_max_celsius)).append(" °C")
        
        precip_str = Text()
        if day.precipitation_mm is not None and day.precipitation_mm > 0:
            precip_str.append(f"{day.precipitation_mm:.1f}mm")
            if day.precipitation_chance_percent is not None:
                precip_str.append(f" ({day.precipitation_chance_percent}%) ")
        else:
            precip_str.append("-")
            
        wind_str = Text()
        if day.wind_speed_kph:
            wind_str.append(f"{day.wind_speed_kph:.1f}")
            if day.wind_direction_cardinal:
                wind_str.append(f" {day.wind_direction_cardinal}")
        else:
            wind_str.append("-")
            
        uv_str = f"{day.uv_index:.1f}" if day.uv_index is not None else "-"
        
        daily_table.add_row(
            date_str,
            day.condition_text or "-",
            temp_str,
            precip_str,
            wind_str,
            uv_str
        )
    rprint(daily_table)

    # LLM Analysis (if available)
    if llm_analysis and llm_analysis.analysis_text:
        analysis_panel = Panel(
            Text(llm_analysis.analysis_text, overflow="fold"),
            title=f"LLM Analysis ({llm_analysis.provider_used} / {llm_analysis.model_used})",
            border_style="yellow",
            expand=False
        )
        rprint(analysis_panel)
        if llm_analysis.error_message:
            rprint(Panel(Text(llm_analysis.error_message, style="bold red"), title="LLM Error"))

def display_llm_providers_rich(providers_status: Dict[str, Dict[str, Any]]):
    """Display available LLM providers and their status using Rich table."""
    table = Table(title="Available LLM Providers", expand=True)
    table.add_column("Provider", style="cyan", overflow="fold")
    table.add_column("Status", style="green", overflow="fold")
    table.add_column("Default Model", style="magenta", overflow="fold")
    table.add_column("Notes/Error", style="yellow", overflow="fold")

    for provider_name, status_info in providers_status.items():
        status_text = Text(status_info.get("status", "unknown"))
        if status_info.get("status") == "configured":
            status_text.stylize("bold green")
        elif status_info.get("status") == "error":
            status_text.stylize("bold red")
        else:
            status_text.stylize("dim yellow")
            
        table.add_row(
            provider_name,
            status_text,
            status_info.get("default_model", "N/A"),
            status_info.get("notes") or status_info.get("error", "-")
        )
    rprint(table)

def display_analyst_response_rich(response_data: Dict[str, Any]):
    """Display the response from the analyst agent using Rich panel."""
    status = response_data.get("status", "unknown")
    response_text = response_data.get("response_text", "No response text.")
    location = response_data.get("location_used", "N/A")
    llm_provider = response_data.get("llm_provider_used", "N/A")
    llm_model = response_data.get("llm_model_used", "N/A")
    
    panel_title = f"Analyst Response ({llm_provider}/{llm_model})"
    panel_content = Text()
    panel_content.append(f"Location: {location}\n", style="bold")
    panel_content.append(f"Status: {status}\n\n", style="bold green" if status == "success" else "bold red")
    panel_content.append(response_text)
    
    rprint(Panel(panel_content, title=panel_title, border_style="cyan", expand=False)) 