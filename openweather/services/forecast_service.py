"""Forecast service providing core weather forecast functionality."""
import asyncio
import logging
from typing import Optional, Tuple, Any, Dict
from datetime import datetime, timezone

from openweather.core.config import settings
from openweather.core.models_shared import (
    WeatherForecastResponse, 
    LLMAnalysisRequest, 
    LLMAnalysisResponse,
    LocationInfo,
    DailyForecast
)
from openweather.core.utils import format_weather_data_for_llm
from openweather.models.weather_predictor import AbstractWeatherPredictor
from openweather.data.data_loader import WeatherDataOrchestrator
from openweather.llm.llm_manager import LLMManager, ProviderType

logger = logging.getLogger(__name__)

class ForecastService:
    """Service for weather forecasts and analysis."""
    
    def __init__(
        self,
        data_orchestrator: WeatherDataOrchestrator,
        llm_manager: LLMManager,
        custom_model=None
    ):
        """Initialize forecast service with required components."""
        self.data_orchestrator = data_orchestrator
        self.llm_manager = llm_manager
        self.custom_model = custom_model
        self.agent_specific_parameters = {
            "marine": ["wave_height", "swell_direction", "water_temp"],
            "aviation": ["visibility", "cloud_base", "wind_shear"],
            "agriculture": ["soil_moisture", "evapotranspiration"]
        }
        logger.info("ForecastService initialized")
        if custom_model:
            logger.info(f"Using custom weather model: {custom_model.model_name} v{custom_model.model_version}")
        
    async def get_forecast_and_explain(
        self,
        location_str: str,
        num_days: int = 5,
        data_source_preference: Optional[str] = None,
        explain_with_llm: bool = False,
        llm_provider: Optional[str] = None,
        llm_output_format: str = "markdown",
        agent_type: Optional[str] = None
    ) -> Tuple[Optional[WeatherForecastResponse], Optional[LLMAnalysisResponse]]:
        """Enhanced version supporting agent-specific data."""
        try:
            # Get parameters based on agent type
            extra_params = self.agent_specific_parameters.get(agent_type, [])
            
            # Get forecast with additional parameters if needed
            forecast = await self.data_orchestrator.get_weather_data(
                location_str=location_str,
                num_days=num_days,
                preferred_source=data_source_preference,
                extra_parameters=extra_params
            )

            if not forecast:
                return None, None

            # Apply custom model if available
            if self.custom_model and self.custom_model.is_loaded:
                forecast = await self.custom_model.predict(forecast)

            # Generate agent-specific analysis if requested
            analysis = None
            if explain_with_llm:
                analysis = await self._generate_agent_specific_analysis(
                    forecast, 
                    llm_provider,
                    llm_output_format,
                    agent_type
                )

            return forecast, analysis
            
        except Exception as e:
            logger.error(f"Forecast service error: {str(e)}")
            return None, None

    async def _generate_agent_specific_analysis(
        self,
        forecast: WeatherForecastResponse,
        provider: Optional[str],
        output_format: str,
        agent_type: Optional[str]
    ) -> LLMAnalysisResponse:
        """Generate analysis tailored to specific agent type."""
        prompt = self._build_agent_prompt(forecast, agent_type)
        system_prompt = self._get_agent_system_prompt(agent_type)
        
        analysis_text, metadata = await self.llm_manager.generate_text(
            prompt=prompt,
            provider=provider,
            system_prompt=system_prompt
        )

        return LLMAnalysisResponse(
            request_details={"agent_type": agent_type},
            analysis_text=analysis_text,
            provider_used=metadata.get("provider_used"),
            model_used=metadata.get("model_used"),
            output_format=output_format
        )

    def _build_agent_prompt(self, forecast: Any, agent_type: str) -> str:
        """Build prompt tailored to agent type."""
        if agent_type == "marine":
            return self._build_marine_prompt(forecast)
        elif agent_type == "aviation":
            return self._build_aviation_prompt(forecast)
        # ... other agent types ...
        else:
            return self._build_default_prompt(forecast)

    def _get_agent_system_prompt(self, agent_type: str) -> str:
        """Get system prompt tailored to agent type."""
        prompts = {
            "marine": "You are a marine weather expert. Analyze wave heights, swell...",
            "aviation": "You are an aviation meteorologist. Assess visibility, cloud...",
            # ... other agent types ...
        }
        return prompts.get(agent_type, self._default_system_prompt())

    async def _generate_llm_analysis(
        self,
        forecast: WeatherForecastResponse,
        provider: Optional[str],
        output_format: str
    ) -> LLMAnalysisResponse:
        """Generate LLM analysis of weather forecast."""
        prompt = self._build_llm_prompt(forecast, output_format)
        
        analysis_text, metadata = await self.llm_manager.generate_text(
            prompt=prompt,
            provider=provider,
            system_prompt="You are a professional meteorologist. Explain the weather forecast clearly and accurately."
        )

        return LLMAnalysisResponse(
            request_details={
                "location": forecast.location.name,
                "days": len(forecast.daily_forecasts)
            },
            analysis_text=analysis_text,
            provider_used=metadata.get("provider_used"),
            model_used=metadata.get("model_used"),
            generated_at_utc=datetime.now(timezone.utc),
            output_format=output_format
        )

    def _build_llm_prompt(self, forecast: WeatherForecastResponse, output_format: str) -> str:
        """Construct LLM prompt from forecast data."""
        prompt = f"""Explain this weather forecast for {forecast.location.name}:
        
Forecast Summary:
{self._format_forecast_summary(forecast)}

Provide a concise analysis in {output_format} format covering:
1. Key weather patterns
2. Notable temperature/precipitation trends
3. Any significant weather events
4. Recommendations if relevant"""
        return prompt

    def _format_forecast_summary(self, forecast: WeatherForecastResponse) -> str:
        """Format forecast data for LLM prompt."""
        summary = []
        for daily in forecast.daily_forecasts:
            summary.append(
                f"{daily.date}: High {daily.temp_max_celsius}°C, "
                f"Low {daily.temp_min_celsius}°C, {daily.condition_text}"
            )
        return "\n".join(summary) 