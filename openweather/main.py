"""Main application entry point for OpenWeather platform."""

import asyncio
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import RedirectResponse, Response

from openweather.api.routes import api_router_v1
from openweather.core.monitoring import monitoring
from openweather.services.weather_service import weather_service
from openweather.services.analytics_service import analytics_service
from openweather.web.dashboard import dashboard_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('openweather.log')
    ]
)

logger = logging.getLogger(__name__)

# Global shutdown event
shutdown_event = asyncio.Event()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting OpenWeather platform...")
    
    try:
        # Initialize monitoring system
        logger.info("Initializing monitoring system...")
        await monitoring.start()
        
        # Initialize weather service
        logger.info("Initializing weather service...")
        await weather_service.initialize()
        
        # Initialize analytics service
        logger.info("Initializing analytics service...")
        await analytics_service.initialize()
        
        # Start background tasks
        logger.info("Starting background monitoring tasks...")
        asyncio.create_task(run_background_tasks())
        
        logger.info("OpenWeather platform started successfully!")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start OpenWeather platform: {e}")
        raise
    
    finally:
        logger.info("Shutting down OpenWeather platform...")
        
        # Signal shutdown to background tasks
        shutdown_event.set()
        
        # Stop services
        try:
            await monitoring.stop()
            await weather_service.cleanup()
            await analytics_service.cleanup()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        
        logger.info("OpenWeather platform shutdown complete.")

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="OpenWeather Platform",
        description="Enterprise-grade weather analytics platform with AI, LLM integration, and drone support",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        lifespan=lifespan
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Add custom middleware for request logging
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = asyncio.get_event_loop().time()
        
        response = await call_next(request)
        
        process_time = asyncio.get_event_loop().time() - start_time
        
        # Log request details
        logger.info(
            f"{request.method} {request.url.path} "
            f"- {response.status_code} "
            f"- {process_time:.3f}s"
        )
        
        # Record metrics
        monitoring.metrics.record_histogram(
            "http_request_duration_seconds",
            process_time,
            tags={"method": request.method, "status": str(response.status_code)}
        )
        
        monitoring.metrics.increment_counter(
            "http_requests_total",
            tags={"method": request.method, "status": str(response.status_code)}
        )
        
        return response
    
    # Include API routes
    app.include_router(api_router_v1)
    
    # Mount dashboard application
    app.mount("/dashboard", dashboard_app, name="dashboard")
    
    # Mount static files
    if os.path.exists("openweather/web/static"):
        app.mount("/static", StaticFiles(directory="openweather/web/static"), name="static")
    
    # Root redirect to dashboard
    @app.get("/")
    async def root():
        """Redirect root to dashboard."""
        return RedirectResponse(url="/dashboard/")
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Application health check."""
        try:
            health_status = await monitoring.health_check_manager.get_overall_health()
            
            return {
                "status": health_status["status"],
                "timestamp": health_status["timestamp"],
                "services": health_status["checks"],
                "version": "1.0.0",
                "uptime": health_status.get("uptime", "unknown")
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "version": "1.0.0"
            }
    
    # Metrics endpoint (Prometheus format)
    @app.get("/metrics")
    async def metrics_endpoint():
        """Prometheus metrics endpoint."""
        try:
            metrics_data = monitoring.metrics.export_prometheus()
            return Response(content=metrics_data, media_type="text/plain")
        except Exception as e:
            logger.error(f"Metrics export failed: {e}")
            return {"error": "Metrics unavailable"}
    
    return app

async def run_background_tasks():
    """Run background maintenance tasks."""
    logger.info("Starting background tasks...")
    
    async def cleanup_task():
        """Periodic cleanup task."""
        while not shutdown_event.is_set():
            try:
                # Clean up expired cache entries
                await weather_service.cleanup_cache()
                
                # Clean up old metrics
                monitoring.metrics.cleanup_old_metrics()
                
                # Clean up old alerts
                monitoring.alert_manager.cleanup_old_alerts()
                
                logger.debug("Background cleanup completed")
                
            except Exception as e:
                logger.error(f"Background cleanup error: {e}")
            
            # Wait 5 minutes before next cleanup
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=300)
                break  # Shutdown requested
            except asyncio.TimeoutError:
                continue  # Continue cleanup cycle
    
    async def health_check_task():
        """Periodic health checks."""
        while not shutdown_event.is_set():
            try:
                # Run health checks
                await monitoring.health_check_manager.run_all_checks()
                
                # Check service health
                weather_health = await weather_service.health_check()
                analytics_health = await analytics_service.health_check()
                
                # Record health metrics
                monitoring.metrics.set_gauge(
                    "service_health",
                    1 if all(weather_health.values()) else 0,
                    tags={"service": "weather"}
                )
                
                monitoring.metrics.set_gauge(
                    "service_health",
                    1 if analytics_health else 0,
                    tags={"service": "analytics"}
                )
                
                logger.debug("Health checks completed")
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
            
            # Wait 30 seconds before next check
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=30)
                break  # Shutdown requested
            except asyncio.TimeoutError:
                continue  # Continue health check cycle
    
    # Start background tasks
    tasks = [
        asyncio.create_task(cleanup_task()),
        asyncio.create_task(health_check_task())
    ]
    
    # Wait for shutdown signal
    await shutdown_event.wait()
    
    # Cancel all background tasks
    for task in tasks:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    
    logger.info("Background tasks stopped")

def setup_signal_handlers():
    """Setup graceful shutdown signal handlers."""
    
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        shutdown_event.set()
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    if hasattr(signal, 'SIGHUP'):
        signal.signal(signal.SIGHUP, signal_handler)

def get_config() -> Dict[str, Any]:
    """Get application configuration from environment variables."""
    return {
        "host": os.getenv("HOST", "0.0.0.0"),
        "port": int(os.getenv("PORT", 8000)),
        "log_level": os.getenv("LOG_LEVEL", "info"),
        "workers": int(os.getenv("WORKERS", 1)),
        "reload": os.getenv("RELOAD", "false").lower() == "true",
        "access_log": os.getenv("ACCESS_LOG", "true").lower() == "true",
    }

def main():
    """Main application entry point."""
    try:
        # Setup signal handlers
        setup_signal_handlers()
        
        # Get configuration
        config = get_config()
        
        # Create application
        app = create_app()
        
        logger.info(f"Starting OpenWeather platform on {config['host']}:{config['port']}")
        
        # Run the application
        uvicorn.run(
            app,
            host=config["host"],
            port=config["port"],
            log_level=config["log_level"],
            access_log=config["access_log"],
            reload=config["reload"],
            workers=config["workers"] if not config["reload"] else 1
        )
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 