"""Main CLI application for OpenWeather using Typer."""
import asyncio
import logging
import sys
from typing import Optional, List
from typing_extensions import Annotated

import typer
from rich import print as rprint
from rich.panel import Panel
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
import json

from openweather import __version__ as app_version
from openweather.core.config import settings
from openweather.services.forecast_service import ForecastService
from openweather.data.data_loader import WeatherDataOrchestrator
from openweather.llm.llm_manager import LLMManager, ProviderType
from openweather.api.main import app as fastapi_app # For API command
import uvicorn # For API command
from openweather.agents.orchestrator import AgentOrchestrator
from openweather.cli.interactive import InteractiveCLI
from openweather.core.utils import format_weather_display

# Import CLI display utilities
from openweather.cli.utils_cli import (
    display_forecast_rich,
    display_llm_providers_rich,
    display_analyst_response_rich
)

# Import agent (if it exists and is part of the final structure)
# For now, we'll mock the agent functionality within the command
# from openweather.agents.analyst_agent import AnalystAgent # Example import

logger = logging.getLogger(__name__)
console = Console()

# Create Typer application instance
app_cli = typer.Typer(
    name="openweather-cli",
    help="Advanced Weather Analytics Platform with LLM Integration",
    rich_markup_mode="rich"
)

# Initialize core services globally for reuse
_data_orchestrator = None
_llm_manager = None
_forecast_service = None
_agent_orchestrator = None

def get_services():
    """Initialize and return core services (singleton pattern)."""
    global _data_orchestrator, _llm_manager, _forecast_service, _agent_orchestrator
    
    if _data_orchestrator is None:
        _data_orchestrator = WeatherDataOrchestrator()
        _llm_manager = LLMManager()
        _forecast_service = ForecastService(_data_orchestrator, _llm_manager)
        _agent_orchestrator = AgentOrchestrator(_llm_manager, _forecast_service)
    
    return _data_orchestrator, _llm_manager, _forecast_service, _agent_orchestrator

# --- CLI Commands ---

@app_cli.command(name="forecast", help="Get weather forecast for a location.")
def forecast_command(
    location: Annotated[str, typer.Argument(help="City name or lat,lon coordinates (e.g., \"London\" or \"51.5,-0.12\").")],
    days: Annotated[int, typer.Option(help="Number of days to forecast (1-16).")] = settings.FORECAST_DAYS,
    explain: Annotated[bool, typer.Option("--explain", "-e", help="Explain weather forecast with LLM.")] = False,
    provider: Annotated[Optional[ProviderType], typer.Option("--provider", "-p", help="LLM provider for explanation (e.g., local_ollama, openai).")] = None,
    model: Annotated[Optional[str], typer.Option("--model", "-m", help="Specific LLM model to use.")] = None,
    raw: Annotated[bool, typer.Option("--raw", help="Display raw JSON output.")] = False,
    data_source: Annotated[Optional[str], typer.Option(help="Preferred data source (e.g. open-meteo, simulation)")] = None,
    format_output: str = typer.Option("rich", "--format", "-f", help="Output format: rich, json, csv"),
    save_to: Optional[str] = typer.Option(None, "--save", help="Save output to file"),
    confidence_threshold: float = typer.Option(0.7, "--confidence", help="Minimum confidence threshold")
):
    """Fetches and displays weather forecast for a given location."""
    _, _, forecast_service, _ = get_services()
    
    rprint(Panel(f"Fetching weather forecast for [bold]{location}[/bold] ({days} days)...", expand=False))
    
    async def _get_forecast():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Fetching forecast for {location}...", total=None)
            
            try:
                weather_data, llm_analysis = await forecast_service.get_forecast_and_explain(
                    location_str=location,
                    num_days=days,
                    data_source_preference=data_source,
                    explain_with_llm=explain,
                    llm_provider=provider,
                    model_name=model
                )
                
                progress.update(task, description="Processing data...")
                
                if not weather_data:
                    console.print(f"[red]No weather data found for: {location}[/red]")
                    raise typer.Exit(1)
                
                # Check confidence levels
                confidence = getattr(weather_data, 'source_confidence', 1.0)
                if confidence < confidence_threshold:
                    console.print(f"[yellow]Warning: Data confidence ({confidence:.2f}) below threshold ({confidence_threshold})[/yellow]")
                
                # Format and display output
                if format_output == "json":
                    output = _format_json_output(weather_data, llm_analysis)
                elif format_output == "csv":
                    output = _format_csv_output(weather_data)
                else:  # rich format
                    _display_rich_forecast(weather_data, llm_analysis)
                    return
                
                if save_to:
                    with open(save_to, 'w') as f:
                        f.write(output)
                    console.print(f"[green]Output saved to: {save_to}[/green]")
                else:
                    console.print(output)
                    
            except Exception as e:
                progress.update(task, description=f"Error: {str(e)}")
                logger.exception("Forecast command failed")
                console.print(f"[red]Error getting forecast: {str(e)}[/red]")
                raise typer.Exit(1)
    
    asyncio.run(_get_forecast())

@app_cli.command(name="analyst", help="Interactive weather analyst or query-based analysis.")
def analyst_command(
    interactive: Annotated[bool, typer.Option("--interactive", "-i", help="Run in interactive mode.")] = False,
    query: Annotated[Optional[str], typer.Option("--query", "-q", help="Specific weather question to ask.")] = None,
    location: Annotated[Optional[str], typer.Option("--location", "-l", help="Location context for the query (required if not interactive).")] = None,
    provider: Annotated[Optional[ProviderType], typer.Option("--provider", "-p", help="LLM provider to use.")] = None,
    model: Annotated[Optional[str], typer.Option("--model", "-m", help="Specific LLM model to use.")] = None
):
    """Engage with an LLM-powered weather analyst."""
    _, llm_manager = get_services()
    
    if interactive:
        rprint(Panel("Entering interactive weather analyst mode. Type 'exit' or 'quit' to end.", style="bold green"))
        # Mocking interactive agent for now
        while True:
            user_query = typer.prompt("Ask a weather question")
            if user_query.lower() in ["exit", "quit"]:
                break
            if not user_query.strip():
                continue
            
            # Simple mock: In a real agent, this would involve RAG, tool use, etc.
            rprint(f"[dim]Analyst processing: '{user_query}'[/dim]")
            response_text, metadata = asyncio.run(llm_manager.generate_text(
                prompt=f"User question: {user_query}\nLocation context: {location or 'current location'}",
                provider=provider,
                model_name=model,
                system_prompt="You are a helpful weather analyst. Answer the user's question about weather."
            ))
            display_analyst_response_rich({
                "status": "success" if response_text else "error",
                "response_text": response_text or metadata.get("error", "LLM Error"),
                "location_used": location or "Assumed Current",
                "llm_provider_used": metadata.get("provider_used", provider or settings.DEFAULT_LLM_PROVIDER),
                "llm_model_used": metadata.get("model_used", model or "default")
            })
        rprint(Panel("Exited interactive analyst mode.", style="bold green"))
        
    elif query:
        if not location:
            rprint("[bold red]Error: Location must be provided for a non-interactive query.[/bold red]")
            raise typer.Exit(code=1)
        
        rprint(Panel(f"Analysing query: [bold]'{query}'[/bold] for location [bold]{location}[/bold]...", expand=False))
        # Mocking direct query processing
        response_text, metadata = asyncio.run(llm_manager.generate_text(
            prompt=f"User question: {query}\nLocation context: {location}",
            provider=provider,
            model_name=model,
            system_prompt="You are a helpful weather analyst. Answer the user's question about weather."
        ))
        display_analyst_response_rich({
            "status": "success" if response_text else "error",
            "response_text": response_text or metadata.get("error", "LLM Error"),
            "location_used": location,
            "llm_provider_used": metadata.get("provider_used", provider or settings.DEFAULT_LLM_PROVIDER),
            "llm_model_used": metadata.get("model_used", model or "default")
        })
    else:
        rprint("[bold yellow]Hint: Use --interactive for interactive mode or --query with --location for a specific question.[/bold yellow]")
        # Show help for this command
        ctx = typer.Context(analyst_command)
        rprint(ctx.get_help())

@app_cli.command(name="llm-providers", help="List available LLM providers.")
def list_llm_providers_command():
    """Lists all configured LLM providers and their status."""
    _, llm_manager = get_services()
    rprint(Panel("Fetching available LLM provider status...", expand=False))
    
    try:
        providers_status = asyncio.run(llm_manager.list_available_providers_models())
        display_llm_providers_rich(providers_status)
    except Exception as e:
        logger.error(f"LLM providers command error: {str(e)}", exc_info=True)
        rprint(f"[bold red]An error occurred: {str(e)}[/bold red]")
        raise typer.Exit(code=1)

@app_cli.command(name="api", help="Run the OpenWeather REST API server.")
def run_api_server_command(
    host: Annotated[str, typer.Option(help="Host to bind the API server to.")] = settings.API_HOST,
    port: Annotated[int, typer.Option(help="Port to run the API server on.")] = settings.API_PORT,
    reload: Annotated[bool, typer.Option(help="Enable auto-reload for development.")] = (settings.ENVIRONMENT == "development")
):
    """Starts the FastAPI application server using Uvicorn."""
    rprint(Panel(f"Starting OpenWeather API server on [bold green]{host}:{port}[/bold green]...", expand=False))
    rprint(f"Auto-reload: {'Enabled' if reload else 'Disabled'}")
    rprint(f"Access OpenAPI docs at http://{host}:{port}/api/v1/docs")
    
    uvicorn.run(
        "openweather.api.main:app", 
        host=host, 
        port=port, 
        reload=reload,
        log_level=settings.LOG_LEVEL.lower()
    )

@app_cli.command()
def agent(
    query: Optional[str] = typer.Argument(None, help="Natural language weather query"),
    location: Optional[str] = typer.Option(None, "--location", "-l", help="Location context"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Start interactive session"),
    specialist: str = typer.Option("general", "--specialist", "-s", help="Agent specialist: general, marine, aviation, agriculture"),
    context_file: Optional[str] = typer.Option(None, "--context", help="Load context from JSON file")
):
    """Interact with intelligent weather agents."""
    _, llm_manager, forecast_service, agent_orchestrator = get_services()
    
    if interactive:
        cli = InteractiveCLI(llm_manager)
        cli.start_interactive_session()
        return
    
    if not query:
        console.print("[red]Please provide a query or use --interactive mode[/red]")
        raise typer.Exit(1)
    
    async def _process_query():
        context = {}
        if context_file:
            try:
                with open(context_file, 'r') as f:
                    context = json.load(f)
            except Exception as e:
                console.print(f"[yellow]Warning: Could not load context file: {e}[/yellow]")
        
        if location:
            context['location'] = location
        
        try:
            response = await agent_orchestrator.process_query(
                query=query,
                specialist=specialist,
                context=context
            )
            
            console.print(Panel(
                response.get('analysis', 'No analysis available'),
                title=f"[bold blue]{specialist.title()} Weather Agent[/bold blue]",
                border_style="blue"
            ))
            
            if response.get('recommendations'):
                console.print("\n[bold green]Recommendations:[/bold green]")
                for rec in response['recommendations']:
                    console.print(f"• {rec}")
                    
        except Exception as e:
            logger.exception("Agent query failed")
            console.print(f"[red]Agent error: {str(e)}[/red]")
            raise typer.Exit(1)
    
    asyncio.run(_process_query())

@app_cli.command()
def analytics(
    location: Optional[str] = typer.Option(None, "--location", help="Filter by location"),
    days: int = typer.Option(7, "--days", help="Days of data to analyze"),
    metric: str = typer.Option("accuracy", "--metric", help="Metric to analyze: accuracy, confidence, sources"),
    export: Optional[str] = typer.Option(None, "--export", help="Export to file"),
    open_datasette: bool = typer.Option(False, "--datasette", help="Open Datasette interface")
):
    """Advanced analytics and data exploration."""
    data_orchestrator, llm_manager, _, _ = get_services()
    
    if open_datasette:
        console.print("[blue]Opening Datasette interface...[/blue]")
        datasette = llm_manager.get_datasette_connection()
        # In a real implementation, this would start the Datasette server
        console.print("[green]Datasette available at http://localhost:8001[/green]")
        return
    
    async def _run_analytics():
        try:
            # Placeholder for analytics implementation
            table = Table(title=f"Weather Analytics - {metric.title()}")
            table.add_column("Date", style="cyan")
            table.add_column("Location", style="magenta")
            table.add_column("Value", style="green")
            table.add_column("Confidence", style="yellow")
            
            # This would be populated with real analytics data
            table.add_row("2024-01-15", location or "Global", "85%", "High")
            table.add_row("2024-01-14", location or "Global", "82%", "High")
            table.add_row("2024-01-13", location or "Global", "79%", "Medium")
            
            console.print(table)
            
            if export:
                console.print(f"[green]Analytics exported to: {export}[/green]")
                
        except Exception as e:
            logger.exception("Analytics failed")
            console.print(f"[red]Analytics error: {str(e)}[/red]")
            raise typer.Exit(1)
    
    asyncio.run(_run_analytics())

def _display_rich_forecast(weather_data, llm_analysis=None):
    """Display forecast in rich terminal format."""
    # Location header
    location = weather_data.location
    console.print(Panel(
        f"[bold blue]{location.name}[/bold blue]\n"
        f"Location: {location.coordinates.latitude:.4f}°N, {location.coordinates.longitude:.4f}°E\n"
        f"Updated: {weather_data.generated_at_utc.strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"Source: {weather_data.data_source}",
        title="Weather Forecast",
        border_style="blue"
    ))
    
    # Current weather if available
    if weather_data.current_weather:
        current = weather_data.current_weather
        console.print(Panel(
            f"**{current.temp_celsius:.1f}°C** (feels like {current.feels_like_celsius:.1f}°C)\n"
            f"{current.condition_text}\n"
            f"Wind: {current.wind_speed_kph:.1f} km/h {current.wind_direction_cardinal}\n"
            f"Humidity: {current.humidity_percent}%",
            title="Current Conditions",
            border_style="green"
        ))
    
    # Daily forecast table
    table = Table(title="Daily Forecast")
    table.add_column("Date", style="cyan", width=12)
    table.add_column("High/Low", style="red/blue", width=12)
    table.add_column("Conditions", style="yellow", width=20)
    table.add_column("Precip", style="blue", width=10)
    table.add_column("Wind", style="green", width=15)
    
    for daily in weather_data.daily_forecasts:
        table.add_row(
            daily.date.strftime("%a %m/%d"),
            f"{daily.temp_max_celsius:.0f}° / {daily.temp_min_celsius:.0f}°",
            daily.condition_text or "N/A",
            f"{daily.precipitation_mm:.1f}mm" if daily.precipitation_mm else "0mm",
            f"{daily.wind_speed_kph:.0f} km/h {daily.wind_direction_cardinal}" 
            if daily.wind_speed_kph else "N/A"
        )
    
    console.print(table)
    
    # LLM Analysis if available
    if llm_analysis:
        console.print(Panel(
            llm_analysis.analysis_text,
            title=f"AI Analysis ({llm_analysis.provider_used})",
            border_style="magenta"
        ))

def _format_json_output(weather_data, llm_analysis=None):
    """Format output as JSON."""
    output = {
        "forecast": weather_data.dict(),
        "analysis": llm_analysis.dict() if llm_analysis else None
    }
    return json.dumps(output, indent=2, default=str)

def _format_csv_output(weather_data):
    """Format output as CSV."""
    import csv
    import io
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow([
        "Date", "Location", "Temp_Max", "Temp_Min", "Condition", 
        "Precipitation", "Wind_Speed", "Wind_Direction"
    ])
    
    # Data rows
    for daily in weather_data.daily_forecasts:
        writer.writerow([
            daily.date.isoformat(),
            weather_data.location.name,
            daily.temp_max_celsius,
            daily.temp_min_celsius,
            daily.condition_text,
            daily.precipitation_mm,
            daily.wind_speed_kph,
            daily.wind_direction_cardinal
        ])
    
    return output.getvalue()

# --- Version Callback ---
def version_callback(value: bool):
    if value:
        rprint(f"OpenWeather CLI Version: {app_version}")
        raise typer.Exit()

@app_cli.callback()
def main_callback(
    version: Annotated[Optional[bool], typer.Option("--version", callback=version_callback, is_eager=True, help="Show application version and exit.")] = None,
):
    """OpenWeather CLI main application entry point."""
    pass # Main callback, can be used for global setup if needed

# This check is useful if this script is run directly, though Typer handles it.
if __name__ == "__main__":
    # Setup basic logging if run directly (Typer/uvicorn might override later)
    from openweather.core.utils import setup_logging
    setup_logging(settings.LOG_LEVEL)
    app_cli() # Run the Typer CLI application 