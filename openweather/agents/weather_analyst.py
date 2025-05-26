from typing import Optional, Dict, Any
import asyncio
from openweather.agents.base_agent import BaseAgent
from openweather.core.models_shared import AgentResponse, WeatherContext
from openweather.services.forecast_service import ForecastService

class WeatherAnalystAgent(BaseAgent):
    """Agent for analyzing weather patterns and providing insights."""
    
    def __init__(
        self,
        llm_manager: LLMManager,
        forecast_service: ForecastService
    ):
        super().__init__(
            llm_manager=llm_manager,
            name="Weather Analyst",
            description="Provides expert analysis of weather patterns and forecasts",
            capabilities=[
                "weather interpretation",
                "forecast explanation",
                "climate trends",
                "weather impact analysis"
            ]
        )
        self.forecast_service = forecast_service

    async def perform_task(
        self,
        task: str,
        context: Optional[WeatherContext] = None,
        **kwargs
    ) -> AgentResponse:
        """Analyze weather data and provide insights."""
        try:
            # Get forecast if location provided
            forecast = None
            if context and context.location:
                forecast, _ = await self.forecast_service.get_forecast_and_explain(
                    location_str=context.location,
                    num_days=context.days or 3,
                    explain_with_llm=False
                )

            # Generate analysis
            analysis = await self._generate_analysis(task, forecast, context)
            
            return AgentResponse(
                status="success",
                response_text=analysis,
                context=context,
                forecast_data=forecast
            )
            
        except Exception as e:
            return AgentResponse(
                status="error",
                response_text=f"Analysis failed: {str(e)}",
                context=context
            )

    async def _generate_analysis(
        self,
        question: str,
        forecast: Optional[Any],
        context: Optional[WeatherContext]
    ) -> str:
        """Generate detailed weather analysis."""
        prompt = self._build_analysis_prompt(question, forecast, context)
        return await self.generate_response(prompt)

    def _build_analysis_prompt(
        self,
        question: str,
        forecast: Optional[Any],
        context: Optional[WeatherContext]
    ) -> str:
        """Construct analysis prompt with relevant context."""
        prompt = f"Question: {question}\n\n"
        
        if forecast:
            prompt += f"Location: {forecast.location.name}\n"
            prompt += "Forecast Summary:\n"
            for daily in forecast.daily_forecasts[:3]:  # Show next 3 days
                prompt += (
                    f"{daily.date}: High {daily.temp_max_celsius}°C, "
                    f"Low {daily.temp_min_celsius}°C, {daily.condition_text}\n"
                )
        
        if context and context.user_context:
            prompt += f"\nUser Context: {context.user_context}\n"
        
        prompt += "\nProvide a detailed analysis covering:\n"
        prompt += "1. Key weather patterns\n2. Notable trends\n"
        prompt += "3. Potential impacts\n4. Professional recommendations"
        
        return prompt 