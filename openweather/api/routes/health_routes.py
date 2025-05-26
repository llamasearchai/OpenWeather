"""API routes for health checks and application status."""
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from openweather import __version__ as app_version
from openweather.core.config import settings
from openweather.core.utils import get_platform_info, is_apple_silicon
from openweather.llm.llm_manager import LLMManager
from openweather.api.dependencies import get_llm_manager_dependency

router = APIRouter(tags=["Application Status"])

class HealthResponse(BaseModel):
    """Response model for health check.

    Includes application status, version, timestamp, and environment.
    """
    status: str = "Healthy"
    timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = app_version
    environment: str = settings.ENVIRONMENT.value
    platform: Dict[str, str] = Field(default_factory=get_platform_info)
    apple_silicon: bool = Field(default_factory=is_apple_silicon)
    llm_providers_status: Optional[Dict[str, Any]] = None

@router.get("/health", response_model=HealthResponse, summary="Application Health Check")
async def health_check_endpoint(
    llm_manager: LLMManager = Depends(get_llm_manager_dependency)
) -> HealthResponse:
    """Check application health and return status information."""
    llm_status = await llm_manager.list_available_providers_models()
    return HealthResponse(llm_providers_status=llm_status)

@router.get("/config", summary="View Application Configuration (Dev only)")
async def config_endpoint() -> Dict[str, Any]:
    """Return the current application configuration (for development/debug)."""
    if settings.ENVIRONMENT == "development":
        return settings.model_dump(mode='json')
    return {"message": "Configuration view is restricted to development environment."} 