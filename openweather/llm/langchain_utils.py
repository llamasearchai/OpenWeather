"""LangChain utilities for OpenWeather application."""
import logging
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple
from typing_extensions import Annotated

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from openweather.core.models_shared import WeatherForecastResponse
from openweather.llm.llm_manager import LLMManager, ProviderType

logger = logging.getLogger(__name__)

# Define a template for weather summary generation
WEATHER_SUMMARY_PROMPT_LANGCHAIN = PromptTemplate.from_template(
    """Provide a concise weather summary for {location_name} based on the following data:

{formatted_weather_data}

Summary:
"""
)

class WeatherAnalysisLangChain:
    """LangChain integration for weather analysis."""
    
    def __init__(self, llm_manager: LLMManager):
        """Initialize with LLM manager."""
        self.llm_manager = llm_manager
        
    async def run_analysis(
        self,
        location_name: str,
        formatted_weather_data_str: str,
        llm_provider_override: Optional[ProviderType] = None,
        llm_model_override: Optional[str] = None
    ) -> str:
        """Run weather analysis using LangChain components."""
        try:
            # Format prompt using LangChain
            prompt_value = WEATHER_SUMMARY_PROMPT_LANGCHAIN.format_prompt(
                location_name=location_name,
                formatted_weather_data=formatted_weather_data_str
            )
            
            # Get LLM response using LLMManager
            generated_text, metadata = await self.llm_manager.generate_text(
                prompt=prompt_value.text,
                provider=llm_provider_override,
                model_name=llm_model_override,
                system_prompt="You are a helpful weather analyst. Provide clear, concise, and actionable information."
            )
            
            if not generated_text:
                error_msg = metadata.get("error", "Unknown error in LLM generation")
                logger.error(f"LLM generation failed: {error_msg}")
                return f"Error in LLM generation: {error_msg}"
                
            # Parse output
            output_parser = StrOutputParser()
            parsed_output = output_parser.parse(generated_text)
            
            return parsed_output
            
        except Exception as e:
            logger.exception("Error in LangChain weather analysis: %s", str(e))
            return f"Error in LangChain weather analysis: {str(e)}" 