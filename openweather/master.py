#!/usr/bin/env python3
"""
OpenWeather Master Program

This script serves as the main orchestration layer for the OpenWeather application,
providing a unified interface to access all major functionality including:
- Weather data retrieval and forecasting
- LLM-powered weather analysis
- Interactive weather agent
- API server management
"""
import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import uvicorn
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from openweather.core.config import settings
from openweather.core.utils import setup_logging, get_platform_info
from openweather.core import __version__
from openweather.data.data_loader import WeatherDataOrchestrator
from openweather.llm.llm_manager import LLMManager
from openweather.models.physics_model_stub import StubPhysicsEnhancedModel
from openweather.models.mlx_model_runner import MLXWeatherModelRunner, MLX_MODEL_RUNNER_AVAILABLE
from openweather.services.forecast_service import ForecastService
from openweather.api.main import app as fastapi_app
from openweather.agents.weather_analyst_agent import WeatherAnalystAgent

console = Console()
logger = logging.getLogger(__name__)

class OpenWeatherMaster:
    """Master class for orchestrating the OpenWeather application components."""
    
    def __init__(self):
        """Initialize the OpenWeather master program."""
        self.initialized = False
        self.llm_manager = None
        self.data_orchestrator = None
        self.forecast_service = None
        self.weather_model = None
        self.analyst_agent = None
        
    async def initialize(self) -> None:
        """Initialize all system components."""
        if self.initialized:
            return
            
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            # Initialize components
            init_task = progress.add_task("Initializing OpenWeather system...", total=5)
            
            # Initialize LLM manager
            progress.update(init_task, description="Initializing LLM providers...")
            self.llm_manager = LLMManager()
            progress.advance(init_task)
            
            # Initialize data orchestrator
            progress.update(init_task, description="Initializing data sources...")
            self.data_orchestrator = WeatherDataOrchestrator()
            progress.advance(init_task)
            
            # Initialize weather model
            progress.update(init_task, description="Loading weather prediction model...")
            try:
                if MLX_MODEL_RUNNER_AVAILABLE and settings.USE_MLX:
                    self.weather_model = MLXWeatherModelRunner()
                    if not self.weather_model.is_loaded:
                        self.weather_model = StubPhysicsEnhancedModel()
                else:
                    self.weather_model = StubPhysicsEnhancedModel()
            except Exception as e:
                logger.warning(f"Failed to load weather model: {str(e)}")
                self.weather_model = None
            progress.advance(init_task)
            
            # Initialize forecast service
            progress.update(init_task, description="Setting up forecast service...")
            self.forecast_service = ForecastService(
                data_orchestrator=self.data_orchestrator,
                llm_manager=self.llm_manager,
                weather_model=self.weather_model
            )
            progress.advance(init_task)
            
            # Initialize analyst agent
            progress.update(init_task, description="Initializing weather analyst agent...")
            self.analyst_agent = WeatherAnalystAgent(
                llm_manager=self.llm_manager,
                data_orchestrator=self.data_orchestrator,
                forecast_service=self.forecast_service
            )
            progress.advance(init_task)
            
        # Check available LLM providers
        provider_info = await self.llm_manager.list_available_providers_models()
        available_providers = [
            name for name, info in provider_info.items() 
            if info.get("status") == "configured"
        ]
        
        # Log system information
        logger.info(f"OpenWeather v{__version__} initialized successfully")
        logger.info(f"Platform: {get_platform_info()}")
        logger.info(f"Available LLM providers: {', '.join(available_providers) if available_providers else 'None'}")
        
        if self.weather_model:
            logger.info(f"Weather model: {self.weather_model.model_name} v{self.weather_model.model_version}")
        else:
            logger.info("No weather model loaded")
            
        self.initialized = True
        
    async def get_forecast(
        self, 
        location: str, 
        days: int = 7,
        explain: bool = False,
        llm_provider: Optional[str] = None,
        llm_output_format: str = "markdown",
        data_source: Optional[str] = None
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """Get weather forecast and optional LLM explanation."""
        if not self.initialized:
            await self.initialize()
            
        forecast, analysis = await self.forecast_service.get_forecast_and_explain(
            location_str=location,
            num_days=days,
            data_source_preference=data_source,
            explain_with_llm=explain,
            llm_provider=llm_provider,
            llm_output_format=llm_output_format
        )
        
        return (
            forecast.model_dump() if forecast else None,
            analysis.model_dump() if analysis else None
        )
        
    async def analyze_weather(
        self,
        query: str,
        location: str,
        llm_provider: Optional[str] = None,
        output_format: str = "markdown"
    ) -> Dict[str, Any]:
        """Analyze weather data with LLM."""
        if not self.initialized:
            await self.initialize()
            
        result = await self.analyst_agent.perform_task(
            task_description=query,
            location_str=location,
            llm_provider=llm_provider,
            output_format_preference=output_format
        )
        
        return result
        
    async def run_interactive_agent(self) -> None:
        """Run the interactive weather analyst agent."""
        if not self.initialized:
            await self.initialize()
            
        await self.analyst_agent.run_interactive_mode()
        
    def run_api_server(
        self, 
        host: Optional[str] = None, 
        port: Optional[int] = None,
        reload: bool = False
    ) -> None:
        """Run the FastAPI server."""
        api_host = host or settings.API_HOST
        api_port = port or settings.API_PORT
        
        console.print(Panel(
            f"Starting OpenWeather API server at http://{api_host}:{api_port}",
            title="OpenWeather API",
            expand=False
        ))
        
        uvicorn.run(
            "openweather.api.main:app",
            host=api_host,
            port=api_port,
            reload=reload
        )

def setup_parser() -> argparse.ArgumentParser:
    """Set up command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="OpenWeather - Complete Weather Analytics Platform"
    )
    
    parser.add_argument(
        "--version", "-v", 
        action="store_true", 
        help="Show version information and exit"
    )
    
    parser.add_argument(
        "--log-level", "-l",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level"
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # API server command
    api_parser = subparsers.add_parser("api", help="Run the API server")
    api_parser.add_argument(
        "--host", 
        type=str, 
        help=f"Host to bind the server to (default: {settings.API_HOST})"
    )
    api_parser.add_argument(
        "--port", 
        type=int, 
        help=f"Port to bind the server to (default: {settings.API_PORT})"
    )
    api_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    # Forecast command
    forecast_parser = subparsers.add_parser("forecast", help="Get weather forecast")
    forecast_parser.add_argument(
        "location", 
        type=str, 
        help="Location (city name or lat,lon)"
    )
    forecast_parser.add_argument(
        "--days", "-d",
        type=int,
        default=7,
        help="Number of days to forecast (1-16)"
    )
    forecast_parser.add_argument(
        "--explain", "-e",
        action="store_true",
        help="Include LLM explanation"
    )
    forecast_parser.add_argument(
        "--provider", "-p",
        type=str,
        help="LLM provider to use"
    )
    forecast_parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["markdown", "json", "text"],
        default="markdown",
        help="Output format for LLM explanation"
    )
    forecast_parser.add_argument(
        "--raw",
        action="store_true",
        help="Output raw JSON data"
    )
    
    # Analyst command
    analyst_parser = subparsers.add_parser("analyst", help="Weather analyst agent")
    analyst_parser.add_argument(
        "--query", "-q",
        type=str,
        help="Question about weather"
    )
    analyst_parser.add_argument(
        "--location", "-l",
        type=str,
        default="London",
        help="Location to analyze"
    )
    analyst_parser.add_argument(
        "--provider", "-p",
        type=str,
        help="LLM provider to use"
    )
    analyst_parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    
    return parser

async def run_forecast_command(args: argparse.Namespace) -> None:
    """Run the forecast command."""
    master = OpenWeatherMaster()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True
    ) as progress:
        task = progress.add_task("Fetching weather forecast...", total=None)
        forecast_data, llm_analysis = await master.get_forecast(
            location=args.location,
            days=args.days,
            explain=args.explain,
            llm_provider=args.provider,
            llm_output_format=args.format
        )
    
    if not forecast_data:
        console.print("[bold red]Error: Could not retrieve weather forecast.")
        return
        
    if args.raw:
        import json
        console.print(json.dumps(forecast_data, indent=2, default=str))
        if llm_analysis:
            console.print("\nLLM Analysis:")
            console.print(json.dumps(llm_analysis, indent=2, default=str))
    else:
        from openweather.cli.commands.forecast_cmd import _display_forecast_table, _display_llm_analysis
        from openweather.core.models_shared import WeatherForecastResponse, LLMAnalysisResponse
        from pydantic import parse_obj_as
        
        # Convert dict back to model for display
        forecast_model = parse_obj_as(WeatherForecastResponse, forecast_data)
        _display_forecast_table(console, forecast_model, forecast_model.location.name)
        
        if llm_analysis:
            analysis_model = parse_obj_as(LLMAnalysisResponse, llm_analysis)
            _display_llm_analysis(console, analysis_model)

async def run_analyst_command(args: argparse.Namespace) -> None:
    """Run the analyst command."""
    master = OpenWeatherMaster()
    
    if args.interactive:
        console.print(Panel("Starting interactive weather analyst agent...", title="Weather Analyst"))
        await master.run_interactive_agent()
    elif args.query:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            task = progress.add_task("Analyzing weather data...", total=None)
            result = await master.analyze_weather(
                query=args.query,
                location=args.location,
                llm_provider=args.provider
            )
        
        if result["status"] == "success":
            from rich.markdown import Markdown
            console.print(Markdown(result["response_text"]))
            console.print(f"[dim]Location: {result['location_used']}[/dim]")
            console.print(f"[dim]LLM: {result['llm_provider_used']}/{result['llm_model_used']}[/dim]")
        else:
            console.print(f"[bold red]Error: {result['error_message']}")
    else:
        console.print("[yellow]Please provide a query or use interactive mode[/yellow]")
        console.print("Example: openweather analyst -q 'What's the weather like?' -l 'London'")
        console.print("Or use: openweather analyst --interactive")

def main():
    """Main entry point for the master program."""
    parser = setup_parser()
    args = parser.parse_args()
    
    # Handle version flag
    if args.version:
        console.print(f"OpenWeather v{__version__}")
        platform_info = get_platform_info()
        console.print(f"Platform: {platform_info['os']} {platform_info['architecture']}")
        return
        
    # Setup logging
    if args.log_level:
        setup_logging(args.log_level)
    else:
        setup_logging(settings.LOG_LEVEL)
        
    # Show welcome banner if no command specified
    if not args.command:
        console.print(Panel(
            f"OpenWeather v{__version__} - Complete Weather Analytics Platform\n\n"
            "Usage:\n"
            "  openweather api         - Run the API server\n"
            "  openweather forecast    - Get weather forecast\n"
            "  openweather analyst     - Run weather analyst agent\n\n"
            "Use -h or --help with any command for more information.",
            title="OpenWeather",
            expand=False
        ))
        return
        
    # Handle commands
    if args.command == "api":
        master = OpenWeatherMaster()
        master.run_api_server(
            host=args.host,
            port=args.port,
            reload=args.reload
        )
    elif args.command == "forecast":
        asyncio.run(run_forecast_command(args))
    elif args.command == "analyst":
        asyncio.run(run_analyst_command(args))

if __name__ == "__main__":
    main()