"""Main FastAPI application for OpenWeather API."""
import logging
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time

from openweather.core.config import settings
from openweather.core.utils import setup_logging
from openweather.api.routes import api_router_v1, forecast, agent
from openweather import __version__ as app_version

logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="OpenWeather API",
    description="Provides weather forecasts and LLM-powered analysis.",
    version=app_version,
    openapi_url="/api/v1/openapi.json",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc"
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.API_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware for request logging and timing
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    logger.info(f"Incoming request: {request.method} {request.url.path}")
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    logger.info(f"Response: {response.status_code} - Processed in {process_time:.2f}ms")
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Generic Exception Handler (example)
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception for {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected server error occurred."},
    )

# Include API routers
app.include_router(api_router_v1)
app.include_router(forecast.router, prefix="/forecast", tags=["Weather Forecasts"])
app.include_router(agent.router, prefix="/agent", tags=["Weather Agent"])

@app.on_event("startup")
async def startup_event():
    """Actions to perform on application startup."""
    setup_logging(settings.LOG_LEVEL)
    logger.info(f"OpenWeather API (v{app_version}) started on {settings.API_HOST}:{settings.API_PORT}")
    logger.info(f"Environment: {settings.ENVIRONMENT.value}")
    logger.info(f"Access OpenAPI docs at /api/v1/docs")

@app.get("/", tags=["Root"])
async def read_root():
    """Welcome endpoint for the OpenWeather API."""
    return {
        "message": f"Welcome to OpenWeather API Version {app_version}",
        "documentation": "/docs"
        }

# --- Main execution for development ---
if __name__ == "__main__":
    # Setup logging (primarily for dev mode direct run)
    setup_logging(settings.LOG_LEVEL)
    
    # Run Uvicorn server
    uvicorn.run(
        "openweather.api.main:app", 
        host=settings.API_HOST, 
        port=settings.API_PORT, 
        reload=(settings.ENVIRONMENT == "development"),
        log_level=settings.LOG_LEVEL.lower()
    ) 