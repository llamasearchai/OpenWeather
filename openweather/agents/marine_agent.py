from typing import Optional, Dict, Any, List
import asyncio
from openweather.agents.base_agent import BaseAgent
from openweather.core.models_shared import AgentResponse, WeatherContext
from openweather.services.forecast_service import ForecastService

class MarineWeatherAgent(BaseAgent):
    """Agent specialized in marine weather conditions and advisories."""
    
    def __init__(
        self,
        llm_manager: LLMManager,
        forecast_service: ForecastService
    ):
        super().__init__(
            llm_manager=llm_manager,
            name="Marine Weather Agent",
            description="Provides marine-specific weather analysis and warnings",
            capabilities=[
                "wave height analysis",
                "tide predictions",
                "marine warnings",
                "fishing conditions"
            ]
        )
        self.forecast_service = forecast_service
        self.marine_conditions = {
            "small_craft": {"min_wind": 20, "min_waves": 1.5},
            "gale": {"min_wind": 34, "min_waves": 4},
            "storm": {"min_wind": 48, "min_waves": 7}
        }

    async def perform_task(
        self,
        task: str,
        context: Optional[WeatherContext] = None,
        **kwargs
    ) -> AgentResponse:
        """Provide marine weather analysis."""
        if not context or not context.location:
            return AgentResponse(
                status="error",
                response_text="Location required for marine weather analysis"
            )

        try:
            # Get forecast with marine parameters
            forecast, _ = await self.forecast_service.get_forecast_and_explain(
                location_str=context.location,
                num_days=context.days or 3,
                marine_parameters=True
            )

            if not forecast:
                return AgentResponse(
                    status="error",
                    response_text="Could not retrieve marine forecast data"
                )

            # Generate marine analysis
            analysis = await self._generate_marine_analysis(task, forecast, context)
            
            return AgentResponse(
                status="success",
                response_text=analysis["report"],
                context=context,
                forecast_data=forecast,
                metadata={
                    "warnings": analysis["warnings"],
                    "conditions": analysis["conditions"]
                }
            )
            
        except Exception as e:
            return AgentResponse(
                status="error",
                response_text=f"Marine analysis failed: {str(e)}",
                context=context
            )

    async def _generate_marine_analysis(
        self,
        task: str,
        forecast: Any,
        context: WeatherContext
    ) -> Dict[str, Any]:
        """Generate marine weather analysis with warnings."""
        # Check for marine hazards
        warnings = self._check_marine_hazards(forecast)
        conditions = self._assess_marine_conditions(forecast)
        
        # Generate natural language report
        prompt = self._build_marine_prompt(task, forecast, context, warnings)
        report = await self.generate_response(prompt)
        
        return {
            "report": report,
            "warnings": warnings,
            "conditions": conditions
        }

    def _check_marine_hazards(self, forecast: Any) -> List[Dict[str, Any]]:
        """Check forecast for marine hazards."""
        warnings = []
        for daily in forecast.daily_forecasts:
            if hasattr(daily, 'wave_height') and hasattr(daily, 'wind_speed_kph'):
                for condition, thresholds in self.marine_conditions.items():
                    if (daily.wind_speed_kph >= thresholds["min_wind"] and
                        daily.wave_height >= thresholds["min_waves"]):
                        warnings.append({
                            "date": daily.date,
                            "condition": condition,
                            "wind_speed": daily.wind_speed_kph,
                            "wave_height": daily.wave_height
                        })
        return warnings

    def _assess_marine_conditions(self, forecast: Any) -> Dict[str, Any]:
        """Assess general marine conditions."""
        # Implementation would analyze wave height, swell, etc.
        return {"assessment": "moderate"}  # Simplified for example

    def _build_marine_prompt(
        self,
        task: str,
        forecast: Any,
        context: WeatherContext,
        warnings: List[Dict[str, Any]]
    ) -> str:
        """Construct prompt for marine weather analysis."""
        prompt = f"Marine Weather Request: {task}\n\n"
        prompt += f"Location: {forecast.location.name}\n"
        
        if warnings:
            prompt += "Marine Warnings:\n"
            for warning in warnings:
                prompt += (
                    f"{warning['date']}: {warning['condition'].replace('_', ' ').title()} "
                    f"(Wind: {warning['wind_speed']} kph, Waves: {warning['wave_height']} m)\n"
                )
        
        prompt += "\nProvide detailed marine weather analysis covering:\n"
        prompt += "1. Current marine conditions\n2. Safety recommendations\n"
        prompt += "3. Fishing/boating suitability\n4. 3-day outlook"
        
        return prompt 