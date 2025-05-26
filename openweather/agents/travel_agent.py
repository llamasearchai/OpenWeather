from typing import Optional, Dict, Any
import asyncio
from openweather.agents.base_agent import BaseAgent
from openweather.core.models_shared import AgentResponse, WeatherContext
from openweather.services.forecast_service import ForecastService

class TravelPlanningAgent(BaseAgent):
    """Agent specialized in travel planning based on weather conditions."""
    
    def __init__(
        self,
        llm_manager: LLMManager,
        forecast_service: ForecastService
    ):
        super().__init__(
            llm_manager=llm_manager,
            name="Travel Planning Agent",
            description="Provides travel recommendations based on weather forecasts",
            capabilities=[
                "travel planning",
                "packing recommendations",
                "route optimization",
                "weather-based scheduling"
            ]
        )
        self.forecast_service = forecast_service

    async def perform_task(
        self,
        task: str,
        context: Optional[WeatherContext] = None,
        **kwargs
    ) -> AgentResponse:
        """Provide travel recommendations based on weather."""
        if not context or not context.location:
            return AgentResponse(
                status="error",
                response_text="Location required for travel planning"
            )

        try:
            # Get forecast data
            forecast, _ = await self.forecast_service.get_forecast_and_explain(
                location_str=context.location,
                num_days=context.days or 7  # Default to 7 days for travel
            )

            if not forecast:
                return AgentResponse(
                    status="error",
                    response_text="Could not retrieve forecast data"
                )

            # Generate travel recommendations
            recommendations = await self._generate_recommendations(task, forecast, context)
            
            return AgentResponse(
                status="success",
                response_text=recommendations,
                context=context,
                forecast_data=forecast
            )
            
        except Exception as e:
            return AgentResponse(
                status="error",
                response_text=f"Travel planning failed: {str(e)}",
                context=context
            )

    async def _generate_recommendations(
        self,
        task: str,
        forecast: Any,
        context: WeatherContext
    ) -> str:
        """Generate travel recommendations based on forecast."""
        prompt = self._build_travel_prompt(task, forecast, context)
        return await self.generate_response(prompt)

    def _build_travel_prompt(
        self,
        task: str,
        forecast: Any,
        context: WeatherContext
    ) -> str:
        """Construct prompt for travel recommendations."""
        prompt = f"Travel Planning Request: {task}\n\n"
        prompt += f"Destination: {forecast.location.name}\n"
        prompt += "Weather Forecast:\n"
        
        for daily in forecast.daily_forecasts:
            prompt += (
                f"{daily.date}: High {daily.temp_max_celsius}°C, "
                f"Low {daily.temp_min_celsius}°C, {daily.condition_text}\n"
            )
        
        if context.user_context:
            prompt += f"\nTraveler Context: {context.user_context}\n"
        
        prompt += "\nProvide detailed recommendations covering:\n"
        prompt += "1. Best travel dates/times\n2. Packing list\n"
        prompt += "3. Potential weather disruptions\n4. Alternative plans"
        
        return prompt 