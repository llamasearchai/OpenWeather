"""Run the OpenWeather web interface."""
import argparse
import logging
import uvicorn
from openweather.core.utils import setup_logging
from openweather.core.config import settings
from openweather.web.app import app

def main():
    """Run the web interface server."""
    parser = argparse.ArgumentParser(description="OpenWeather Web Interface")
    parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0", 
        help="Host to bind the server to"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8080, 
        help="Port to bind the server to"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=settings.LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set log level"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Run server
    print(f"Running OpenWeather Web Interface at http://{args.host}:{args.port}")
    uvicorn.run(
        "openweather.web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main()