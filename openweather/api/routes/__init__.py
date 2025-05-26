"""API routes registration and main router."""
import logging
from fastapi import APIRouter

from openweather.api.routes import forecast_routes, health_routes, monitoring_routes

logger = logging.getLogger(__name__)

# Create main API router
api_router_v1 = APIRouter(prefix="/api/v1")

# Include route modules
api_router_v1.include_router(forecast_routes.router)
api_router_v1.include_router(health_routes.router)
api_router_v1.include_router(monitoring_routes.router)

# Try to include optional route modules if they exist
try:
    from openweather.api.routes import agent_routes
    api_router_v1.include_router(agent_routes.router)
    logger.info("Agent routes included")
except ImportError:
    logger.debug("Agent routes not available")

try:
    from openweather.api.routes import drone_routes
    api_router_v1.include_router(drone_routes.router)
    logger.info("Drone routes included")
except ImportError:
    logger.debug("Drone routes not available")

try:
    from openweather.api.routes import analytics_routes
    api_router_v1.include_router(analytics_routes.router)
    logger.info("Analytics routes included")
except ImportError:
    logger.debug("Analytics routes not available")

try:
    from openweather.api.routes import weather_routes
    api_router_v1.include_router(weather_routes.router)
    logger.info("Weather service routes included")
except ImportError:
    logger.debug("Weather service routes not available")

logger.info("API v1 routes initialized: forecast, health, monitoring, analytics, weather.")

__all__ = ["api_router_v1"] 