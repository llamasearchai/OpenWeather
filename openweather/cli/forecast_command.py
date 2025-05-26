import asyncio
from typing import Optional
from rich.console import Console
from rich.table import Table
from openweather.services.forecast_service import ForecastService
from openweather.data.data_loader import WeatherDataOrchestrator
from openweather.llm.llm_manager import LLMManager

console = Console()

async def run_forecast_command(
    location: str,
    days: int = 5,
    explain: bool = False,
    llm_provider: Optional[str] = None,
    output_format: str = "markdown",
    raw: bool = False
):
    """Run forecast command with rich output."""
    # Initialize services
    data_orchestrator = WeatherDataOrchestrator()
    llm_manager = LLMManager()
    forecast_service = ForecastService(data_orchestrator, llm_manager)

    # Get forecast
    forecast, analysis = await forecast_service.get_forecast_and_explain(
        location_str=location,
        num_days=days,
        explain_with_llm=explain,
        llm_provider=llm_provider,
        llm_output_format=output_format
    )

    if not forecast:
        console.print("[red]Error: Could not get forecast data[/red]")
        return

    if raw:
        console.print(forecast.json(indent=2))
        if analysis:
            console.print(analysis.json(indent=2))
        return

    # Display formatted forecast
    table = Table(title=f"Weather Forecast for {forecast.location.name}")
    table.add_column("Date")
    table.add_column("Condition")
    table.add_column("High (°C)")
    table.add_column("Low (°C)")
    table.add_column("Precip (mm)")

    for daily in forecast.daily_forecasts:
        table.add_row(
            str(daily.date),
            daily.condition_text,
            str(daily.temp_max_celsius),
            str(daily.temp_min_celsius),
            str(daily.precipitation_sum)
        )

    console.print(table)

    if analysis:
        console.print("\n[bold]Weather Analysis:[/bold]")
        console.print(analysis.analysis_text) 