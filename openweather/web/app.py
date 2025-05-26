"""Simple web interface for OpenWeather."""
import asyncio
import json
import logging
from typing import Optional, Dict, Any

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from openweather.master import OpenWeatherMaster
from openweather.llm.llm_manager import ProviderType
from openweather.core.config import settings

# Initialize FastAPI app
app = FastAPI(title="OpenWeather Web Interface")

# Setup templates and static files
templates = Jinja2Templates(directory="openweather/web/templates")
app.mount("/static", StaticFiles(directory="openweather/web/static"), name="static")

# Initialize OpenWeather master
master = OpenWeatherMaster()
loop = asyncio.get_event_loop()
loop.run_until_complete(master.initialize())

logger = logging.getLogger(__name__)

class WeatherQuery(BaseModel):
    """Weather query model."""
    location: str
    days: int = 5
    explain: bool = False
    provider: Optional[str] = None

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the main page."""
    # Get available LLM providers
    provider_info = await master.llm_manager.list_available_providers_models()
    available_providers = [
        name for name, info in provider_info.items() 
        if info.get("status") == "configured"
    ]
    
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "title": "OpenWeather",
            "providers": available_providers,
            "default_location": "London",
            "default_days": 5
        }
    )

@app.post("/forecast", response_class=HTMLResponse)
async def get_forecast(
    request: Request,
    location: str = Form(...),
    days: int = Form(5),
    explain: bool = Form(False),
    provider: Optional[str] = Form(None)
):
    """Get weather forecast and render the result."""
    try:
        # Get forecast
        forecast_data, llm_analysis = await master.get_forecast(
            location=location,
            days=days,
            explain=explain,
            llm_provider=provider
        )
        
        if not forecast_data:
            raise HTTPException(status_code=404, detail="Could not retrieve forecast")
            
        # Render template
        return templates.TemplateResponse(
            "forecast.html",
            {
                "request": request,
                "title": f"Weather for {forecast_data['location']['name']}",
                "forecast": forecast_data,
                "analysis": llm_analysis,
                "location": location,
                "days": days,
                "explain": explain,
                "provider": provider
            }
        )
    except Exception as e:
        logger.exception("Error in forecast: %s", str(e))
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "title": "Error",
                "error": str(e)
            }
        )

@app.get("/analyze", response_class=HTMLResponse)
async def analyze_form(request: Request):
    """Render the analysis form."""
    # Get available LLM providers
    provider_info = await master.llm_manager.list_available_providers_models()
    available_providers = [
        name for name, info in provider_info.items() 
        if info.get("status") == "configured"
    ]
    
    return templates.TemplateResponse(
        "analyze.html",
        {
            "request": request,
            "title": "Weather Analysis",
            "providers": available_providers,
            "default_location": "London"
        }
    )

@app.post("/analyze", response_class=HTMLResponse)
async def analyze_weather(
    request: Request,
    query: str = Form(...),
    location: str = Form(...),
    provider: Optional[str] = Form(None)
):
    """Analyze weather data and render the result."""
    try:
        # Analyze weather
        result = await master.analyze_weather(
            query=query,
            location=location,
            llm_provider=provider
        )
        
        if result["status"] != "success":
            raise HTTPException(status_code=500, detail=result["error_message"])
            
        # Render template
        return templates.TemplateResponse(
            "analysis_result.html",
            {
                "request": request,
                "title": "Weather Analysis",
                "result": result,
                "query": query,
                "location": location,
                "provider": provider
            }
        )
    except Exception as e:
        logger.exception("Error in analysis: %s", str(e))
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "title": "Error",
                "error": str(e)
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)