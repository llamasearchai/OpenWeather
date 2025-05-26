import typer
from typing import Optional
from pathlib import Path
from openweather.data.cache import WeatherCache
from openweather.llm.llm_manager import LLMManager
import subprocess
import asyncio

app = typer.Typer(help="Administrative commands for OpenWeather")

@app.command()
def clear_cache(
    cache_type: str = typer.Argument("all", help="Type of cache to clear [all|weather|llm]")
):
    """Clear cached data."""
    cache = WeatherCache()
    if cache_type in ["all", "weather"]:
        cache.db["weather_data"].delete_where()
    if cache_type in ["all", "llm"]:
        cache.db["llm_responses"].delete_where()
    typer.echo(f"Cleared {cache_type} cache")

@app.command()
def datasette(
    port: int = typer.Option(8001, help="Port to run Datasette on"),
    open_browser: bool = typer.Option(False, "--open", help="Open in browser")
):
    """Launch Datasette analytics dashboard."""
    llm_manager = LLMManager()
    ds = llm_manager.get_datasette_connection()
    cmd = ["datasette", str(WeatherCache().db_path), "--port", str(port)]
    if open_browser:
        cmd.append("--open-browser")
    subprocess.run(cmd)

@app.command()
def setup_llm(
    provider: str = typer.Argument(..., help="LLM provider to configure"),
    api_key: Optional[str] = typer.Option(None, help="API key for the provider")
):
    """Configure LLM provider credentials."""
    if provider == "openai":
        if not api_key:
            api_key = typer.prompt("Enter your OpenAI API key", hide_input=True)
        subprocess.run(["llm", "keys", "set", "openai"], input=api_key.encode())
    elif provider == "huggingface":
        if not api_key:
            api_key = typer.prompt("Enter your HuggingFace API key", hide_input=True)
        subprocess.run(["llm", "keys", "set", "huggingface"], input=api_key.encode())
    else:
        typer.echo(f"Unsupported provider: {provider}", err=True)
        raise typer.Exit(1)
    typer.echo(f"Configured {provider} credentials")

@app.command()
def check_services():
    """Check status of all services."""
    from openweather.services.forecast_service import ForecastService
    from openweather.data.data_loader import WeatherDataOrchestrator
    
    typer.echo("Service Status:")
    typer.echo("-" * 40)
    
    # Check data services
    orchestrator = WeatherDataOrchestrator()
    try:
        test_forecast = asyncio.run(orchestrator.get_weather_data("London", 1))
        status = "OK" if test_forecast else "Warning (no data)"
        typer.echo(f"Weather Service: {status}")
    except Exception as e:
        typer.echo(f"Data Service: Error ({str(e)})")
    
    # Check LLM services
    llm_manager = LLMManager()
    providers = asyncio.run(llm_manager.list_available_providers())
    for provider, info in providers.items():
        status = "OK" if info["status"] == "available" else "Error"
        typer.echo(f"LLM {provider}: {status} ({info.get('models', 'no models')})") 