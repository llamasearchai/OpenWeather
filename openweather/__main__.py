"""Main entry point for OpenWeather application."""
import asyncio
import logging
import sys
from typing import NoReturn
import typer
from openweather.cli import forecast_command, analyst_command

from openweather.core.config import settings
from openweather.core.utils import setup_logging
from openweather.cli.main import app_cli as main_typer_app

logger = logging.getLogger(__name__)

def run_application_cli() -> NoReturn:
    """Main entry point for CLI application."""
    try:
        # Setup logging based on configuration
        setup_logging(settings.LOG_LEVEL)
        
        # Execute CLI application
        app = typer.Typer()
        app.add_typer(forecast_command.app, name="forecast")
        app.add_typer(analyst_command.app, name="analyst")
        app()
        
    except ImportError as e:
        logger.critical(
            "Failed to start application: %s - Check dependencies and PYTHONPATH",
            str(e)
        )
        sys.exit(2)
    except Exception as e:
        logger.exception("Critical error during application startup: %s", str(e))
        sys.exit(1)

if __name__ == "__main__":
    run_application_cli() 