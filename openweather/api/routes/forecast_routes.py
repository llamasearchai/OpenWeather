"""API routes for weather forecasts."""
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from typing_extensions import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import HttpUrl

from openweather.core.models_shared import (
    WeatherForecastResponse, 
    LLMAnalysisResponse, 
    ForecastAndAnalysisApiResponse
)
from openweather.llm.llm_manager import LLMManager, ProviderType # Added LLMManager import
from openweather.api.dependencies import get_forecast_service, get_llm_manager_dependency # Added get_llm_manager_dependency
from openweather.services.forecast_service import ForecastService
from openweather.core.config import settings

router = APIRouter(prefix="/forecast", tags=["Weather Forecasts"])

@router.get(
    "/",
    response_model=ForecastAndAnalysisApiResponse,
    summary="Get weather forecast and optional LLM analysis"
)
async def get_weather_forecast_and_analysis_endpoint(
    location: str,
    days: Annotated[int, Query(ge=1, le=16)] = settings.FORECAST_DAYS,
    explain: bool = False,
    llm_provider: Optional[ProviderType] = None,
    llm_model: Optional[str] = None,
    data_source: Optional[str] = Query(None, description="Preferred data source (e.g. open-meteo, simulation)"), # Default to None
    llm_output_format: Annotated[str, Query(regex=r"^(markdown|json|text)$")] = "markdown",
    forecast_service: ForecastService = Depends(get_forecast_service)
):
    """Get weather forecast for a location with optional LLM explanation.
    
    Args:
        location: City name or lat,lon coordinates
        days: Number of days to forecast (1-16)
        explain: Whether to include LLM explanation
        llm_provider: LLM provider to use (if explain=True)
        llm_model: Specific LLM model to use (if explain=True)
        data_source: Preferred weather data source
        llm_output_format: Format for LLM output (markdown, json, text)
        forecast_service: Forecast service dependency
        
    Returns:
        Weather forecast with optional LLM analysis
    """
    try:
        # Call forecast service
        forecast_data, llm_analysis = await forecast_service.get_forecast_and_explain(
            location_str=location,
            num_days=days,
            data_source_preference=data_source,
            explain_with_llm=explain,
            llm_provider=llm_provider,
            llm_model_name=llm_model,
            llm_output_format=llm_output_format
        )
        
        if forecast_data is None:
            raise HTTPException(
                status_code=503,
                detail="Could not retrieve weather data from any source."
            )
            
        # Create response model instance
        response = ForecastAndAnalysisApiResponse(
            **forecast_data.model_dump(),
            llm_analysis=llm_analysis # This can be None
        )
        
        return response
        
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        # Catch any other unexpected errors from the service layer
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )

@router.get(
    "/providers",
    response_model=Dict[str, Dict[str, Any]],
    summary="List available LLM providers for analysis"
)
async def list_llm_providers_endpoint(
    llm_manager: LLMManager = Depends(get_llm_manager_dependency)
):
    """List all available LLM providers and their status."""
    try:
        providers = await llm_manager.list_available_providers_models()
        return providers
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing LLM providers: {str(e)}"
        ) 