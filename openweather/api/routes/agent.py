from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import logging

from openweather.llm.llm_manager import LLMManager
from openweather.llm.prompt_engineering import PromptEngineer
from openweather.core.config import settings
# For dependency injection, similar to forecast.py
from openweather.api.routes.forecast import get_llm_manager # Re-use if suitable, or define new

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Dependency Injection Setup (Simplified) ---
def get_prompt_engineer() -> PromptEngineer:
    return PromptEngineer()
# --- End Dependency Injection Setup ---

class AgentQueryRequest(BaseModel):
    query: str = Field(..., description="The natural language query for the weather agent.")
    location: Optional[str] = Field(None, description="Optional location context for the query (e.g., 'Paris').")
    user_context: Optional[Dict[str, Any]] = Field(None, description="Optional user-specific context (e.g., preferences).")

class AgentFunctionCall(BaseModel):
    name: str
    arguments: Dict[str, Any]

class AgentQueryResponseAPI(BaseModel):
    response_type: str = Field(..., description="Type of response: 'analysis', 'function_calls', or 'clarification'.")
    content: Any = Field(..., description="The content of the response, structure depends on response_type.")
    llm_metadata: Dict[str, Any] = Field(..., description="Metadata from the LLM call.")
    message: Optional[str] = None

@router.post("/query", response_model=AgentQueryResponseAPI)
async def query_agent_api(
    request: AgentQueryRequest, # This was AgentQueryRequest, correcting
    llm_manager: LLMManager = Depends(get_llm_manager), # Corrected type name
    prompt_engineer: PromptEngineer = Depends(get_prompt_engineer)
):
    """
    Processes a natural language query through the weather agent.
    The agent may respond with a direct analysis, request function calls for more data,
    or ask for clarification.
    """
    logger.info(f"API: Agent query received: '{request.query}', Location: {request.location}")

    # Define available functions based on PromptEngineer's specs
    available_function_names = [spec["name"] for spec in prompt_engineer.FUNCTION_SPECS]
    
    # Construct the context for the prompt engineer
    combined_context = {"location_hint": request.location}
    if request.user_context:
        combined_context.update(request.user_context)

    prompt = prompt_engineer.build_function_prompt(
        user_query=request.query,
        available_functions=available_function_names,
        context=combined_context
    )

    json_response_content, llm_meta = await llm_manager.generate_json_response(
        prompt=prompt,
        llm_provider=settings.DEFAULT_LLM_PROVIDER, # Allow override via request if needed
        model_name=settings.DEFAULT_LLM_MODEL # Allow override via request if needed
    )

    if not json_response_content:
        logger.error(f"API: Agent LLM failed to generate a valid JSON response for query: {request.query}")
        raise HTTPException(status_code=500, detail="Agent LLM failed to generate a valid JSON response.")

    # Determine response type based on JSON content
    resp_type = "unknown_format"
    actual_content = json_response_content # Default to full JSON if structure is unexpected

    if "analysis" in json_response_content:
        resp_type = "analysis"
        actual_content = json_response_content["analysis"]
    elif "function_calls" in json_response_content:
        # Validate function_calls structure
        calls = json_response_content["function_calls"]
        if isinstance(calls, list) and all(isinstance(call, dict) and "name" in call and "arguments" in call for call in calls):
            resp_type = "function_calls"
            actual_content = [AgentFunctionCall(**call) for call in calls]
            # In a real system, these function calls would be executed here,
            # and results potentially fed back to the LLM for a final analysis.
            # For this API, we return the request for functions.
        else:
            logger.warning(f"API: Agent LLM returned 'function_calls' but with unexpected structure: {calls}")
            resp_type = "malformed_function_calls" # Custom type for this case
            actual_content = calls

    elif "clarification" in json_response_content:
        resp_type = "clarification"
        actual_content = json_response_content["clarification"]
    
    logger.info(f"API: Agent response type: {resp_type} for query: {request.query}")

    return AgentQueryResponseAPI(
        response_type=resp_type,
        content=actual_content,
        llm_metadata=llm_meta,
        message="Agent query processed."
    ) 