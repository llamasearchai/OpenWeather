from typing import Dict, Any, Optional, List
from datetime import datetime
import json

class PromptEngineer:
    """Advanced prompt engineering for weather-related LLM queries."""
    
    TEMPLATES = {
        "forecast_analysis": """
        [Meteorological Analysis Task]
        Location: {location}
        Coordinates: {latitude}°N, {longitude}°E
        Time Period: {start_date} to {end_date}
        Data Source: {data_source}
        Model Used: {model_info}

        [Forecast Summary]
        {forecast_summary}

        [Analysis Requirements]
        1. Identify dominant weather patterns and systems
        2. Highlight significant temperature/precipitation anomalies
        3. Assess potential severe weather risks
        4. Provide professional recommendations
        5. Include relevant climate context
        6. Format response in {format} with clear sections
        """,
        "agent_task": """
        [Agent Task Brief]
        Task: {task}
        Context: {context}
        Available Data: {available_data}
        
        [Response Requirements]
        - Meteorological accuracy
        - Risk assessment
        - Actionable recommendations
        - Citations to relevant studies
        - {format} formatting
        """
    }

    FUNCTION_SPECS = [
        {
            "name": "get_historical_comparison",
            "description": "Get historical weather patterns for context",
            "parameters": {
                "location": {"type": "string", "description": "City/region name"},
                "date_range": {"type": "string", "description": "Date range in YYYY-MM-DD format"}
            }
        },
        {
            "name": "check_weather_alerts",
            "description": "Check for active weather alerts",
            "parameters": {
                "location": {"type": "string", "description": "City/region name"},
                "radius_km": {"type": "integer", "description": "Search radius in kilometers"}
            }
        }
    ]

    def build_forecast_prompt(
        self,
        forecast_data: Dict[str, Any],
        output_format: str = "markdown"
    ) -> str:
        """Build a detailed forecast analysis prompt."""
        summary = self._format_forecast_summary(forecast_data)
        return self.TEMPLATES["forecast_analysis"].format(
            location=forecast_data["location"]["name"],
            latitude=forecast_data["location"]["latitude"],
            longitude=forecast_data["location"]["longitude"],
            start_date=forecast_data["daily_forecasts"][0]["date"],
            end_date=forecast_data["daily_forecasts"][-1]["date"],
            data_source=forecast_data.get("data_source", "Unknown"),
            model_info=forecast_data.get("model_info", "Unknown"),
            forecast_summary=summary,
            format=output_format
        )

    def build_agent_prompt(
        self,
        task: str,
        context: Dict[str, Any],
        available_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build a prompt for agent tasks."""
        return self.TEMPLATES["agent_task"].format(
            task=task,
            context=json.dumps(context, indent=2),
            available_data=json.dumps(available_data or {}, indent=2),
            format=output_format
        )

    def _format_forecast_summary(self, forecast: Dict[str, Any]) -> str:
        """Enhanced forecast summary with more details"""
        summary = []
        for day in forecast["daily_forecasts"]:
            summary.append(
                f"### {day['date']}\n"
                f"- Temperature: {day['temp_max']}°C (max) / {day['temp_min']}°C (min)\n"
                f"- Conditions: {day['condition']}\n"
                f"- Precipitation: {day['precip']}mm ({day['precip_chance']}% chance)\n"
                f"- Wind: {day['wind_speed']} km/h from {day['wind_dir']}\n"
                f"- UV Index: {day['uv_index']}\n"
            )
        return "\n".join(summary)

    def build_function_prompt(
        self,
        user_query: str,
        available_functions: List[str],
        context: Optional[dict] = None
    ) -> str:
        """Build prompt for function-enabled queries"""
        return f"""
        [Function-Enabled Weather Analysis]
        User Query: {user_query}
        Available Functions: {', '.join(available_functions)}
        Context: {json.dumps(context or {}, indent=2)}
        
        Analyze the query and either:
        1. Provide direct analysis if sufficient information exists
        2. Request specific function calls to gather needed data
        3. Ask clarifying questions if the query is ambiguous
        
        Response must be in valid JSON format with either:
        - "analysis" field for direct responses
        - "function_calls" array for needed data
        - "clarification" field for ambiguous queries
        """ 