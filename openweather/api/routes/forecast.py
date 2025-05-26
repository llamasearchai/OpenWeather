from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional
from pydantic import BaseModel

from openweather.services.forecast_service import ForecastService, ForecastError
from openweather.llm.llm_manager import LLMManager
from openweather.data.data_loader import WeatherDataOrchestrator
from openweather.core.models_shared import WeatherForecastResponse, LLMAnalysisResponse
from openweather.core.config import settings
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# --- Dependency Injection Setup (Simplified) ---
# In a larger app, use FastAPI's Depends for proper DI or a DI container.

def get_data_orchestrator() -> WeatherDataOrchestrator:
    return WeatherDataOrchestrator()

def get_llm_manager() -> LLMManager:
    return LLMManager()

def get_forecast_service(
    data_orchestrator: WeatherDataOrchestrator = Depends(get_data_orchestrator),
    llm_manager: LLMManager = Depends(get_llm_manager)
) -> ForecastService:
    return ForecastService(data_orchestrator, llm_manager)
# --- End Dependency Injection Setup ---


class ForecastAndAnalysisResponseAPI(BaseModel):
    forecast: WeatherForecastResponse
    analysis: Optional[LLMAnalysisResponse] = None
    message: Optional[str] = None


@router.get("/", response_model=ForecastAndAnalysisResponseAPI)
async def get_weather_forecast_api(
    location: str = Query(..., description="City name or coordinates (e.g., 'London' or '51.5074,-0.1278')"),
    days: int = Query(settings.DEFAULT_FORECAST_DAYS, ge=1, le=16, description="Number of forecast days"),
    explain: bool = Query(False, description="Enable LLM-based explanation of the forecast"),
    llm_provider: Optional[str] = Query(None, description=f"Specific LLM provider (e.g., openai, ollama). Defaults to: {settings.DEFAULT_LLM_PROVIDER}"),
    model_name: Optional[str] = Query(None, description="Specific LLM model name. Uses provider's default if not set."),
    forecast_service: ForecastService = Depends(get_forecast_service)
):
    """
    Retrieves the weather forecast for a given location.
    Optionally, it can provide an LLM-powered analysis of the weather conditions.
    """
    logger.info(f"API: Request for forecast. Location: {location}, Days: {days}, Explain: {explain}")
    try:
        weather_data, llm_analysis_obj = await forecast_service.get_forecast_and_explain(
            location_str=location,
            num_days=days,
            explain_with_llm=explain,
            llm_provider=llm_provider, # Pass None if not specified, service will use default
            model_name=model_name      # Pass None if not specified
        )

        if not weather_data:
            logger.warning(f"API: No weather data found for location: {location}")
            raise HTTPException(status_code=404, detail=f"Weather data not found for location: {location}.")

        return ForecastAndAnalysisResponseAPI(
            forecast=weather_data, 
            analysis=llm_analysis_obj,
            message="Forecast retrieved successfully."
        )

    except ForecastError as e:
        logger.error(f"API: ForecastError for location {location}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e: # Handles issues like invalid location string format from parse_location_string
        logger.warning(f"API: ValueError for location {location}: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid request parameter: {e}")
    except Exception as e:
        logger.exception(f"API: Unexpected error for location {location}: {e}") # Logs full stack trace
        raise HTTPException(status_code=500, detail="An unexpected internal server error occurred.") 