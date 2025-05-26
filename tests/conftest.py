"""Advanced Pytest configuration with fixtures for comprehensive testing."""
import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Optional, Dict, Any, List, Generator
from datetime import datetime, timezone, date
import logging

import httpx
from hypothesis import given, strategies as st, settings as hypothesis_settings
from hypothesis.errors import InvalidArgument

from openweather.core.config import settings, Settings
from openweather.core.models_shared import (
    WeatherForecastResponse, LocationInfo, Coordinate, 
    DailyForecast, CurrentWeather, LLMAnalysisResponse
)
from openweather.llm.llm_manager import LLMManager
from openweather.data.data_loader import WeatherDataOrchestrator
from openweather.services.forecast_service import ForecastService
from openweather.models.physics_model_stub import StubPhysicsEnhancedModel
from openweather.data.cache import WeatherCache

logger = logging.getLogger(__name__)

# Hypothesis strategies for property-based testing
@st.composite
def coordinates_strategy(draw):
    """Generate valid geographical coordinates."""
    lat = draw(st.floats(min_value=-90, max_value=90, allow_nan=False, allow_infinity=False))
    lon = draw(st.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False))
    return Coordinate(latitude=lat, longitude=lon)

@st.composite
def location_strategy(draw):
    """Generate valid location information."""
    coords = draw(coordinates_strategy())
    name = draw(st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc', 'Pd', 'Zs'))))
    country_code = draw(st.one_of(st.none(), st.text(min_size=2, max_size=2, alphabet=st.characters(whitelist_categories=('Lu',)))))
    return LocationInfo(name=name, coordinates=coords, country_code=country_code)

@st.composite
def daily_forecast_strategy(draw):
    """Generate valid daily forecast data."""
    forecast_date = draw(st.dates(min_value=date(2020, 1, 1), max_value=date(2030, 12, 31)))
    temp_min = draw(st.floats(min_value=-50, max_value=40, allow_nan=False))
    temp_max = draw(st.floats(min_value=temp_min, max_value=temp_min + 30, allow_nan=False))
    
    return DailyForecast(
        date=forecast_date,
        temp_max_celsius=temp_max,
        temp_min_celsius=temp_min,
        condition_code=draw(st.one_of(st.none(), st.integers(min_value=0, max_value=99))),
        condition_text=draw(st.one_of(st.none(), st.text(max_size=50))),
        precipitation_mm=draw(st.one_of(st.none(), st.floats(min_value=0, max_value=200, allow_nan=False))),
        precipitation_chance_percent=draw(st.one_of(st.none(), st.integers(min_value=0, max_value=100))),
        wind_speed_kph=draw(st.one_of(st.none(), st.floats(min_value=0, max_value=200, allow_nan=False))),
        wind_direction_degrees=draw(st.one_of(st.none(), st.integers(min_value=0, max_value=360))),
        uv_index=draw(st.one_of(st.none(), st.floats(min_value=0, max_value=15, allow_nan=False))),
        detailed_summary=draw(st.one_of(st.none(), st.text(max_size=200)))
    )

@st.composite
def weather_forecast_strategy(draw):
    """Generate complete weather forecast responses."""
    location = draw(location_strategy())
    num_forecasts = draw(st.integers(min_value=1, max_value=16))
    forecasts = draw(st.lists(daily_forecast_strategy(), min_size=num_forecasts, max_size=num_forecasts))
    
    return WeatherForecastResponse(
        location=location,
        daily_forecasts=forecasts,
        generated_at_utc=datetime.now(timezone.utc),
        data_source=draw(st.text(min_size=1, max_size=50)),
        model_info=draw(st.one_of(st.none(), st.dictionaries(st.text(min_size=1), st.text(max_size=50))))
    )

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for session scope."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def temp_dir_session():
    """Session-scoped temporary directory."""
    temp_dir = tempfile.mkdtemp(prefix="openweather_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture(scope="session")
def test_settings(temp_dir_session):
    """Test-specific settings configuration."""
    test_db_path = temp_dir_session / "test_weather.db"
    test_vector_store = temp_dir_session / "vector_store"
    test_app_data = temp_dir_session / "app_data"
    
    # Create directories
    test_vector_store.mkdir(exist_ok=True)
    test_app_data.mkdir(exist_ok=True)
    
    # Override settings for testing
    test_settings = Settings(
        ENVIRONMENT="testing",
        LOG_LEVEL="DEBUG",
        SQLITE_DB_PATH=test_db_path,
        VECTOR_STORE_PATH=test_vector_store,
        APP_DATA_PATH=test_app_data,
        USE_OLLAMA=False,
        USE_MLX=False,
        OPENAI_API_KEY=None,
        HF_API_KEY=None,
        API_HOST="127.0.0.1",
        API_PORT=8001,
        FORECAST_DAYS=5
    )
    return test_settings

@pytest.fixture
async def cache_fixture(test_settings):
    """Weather cache fixture with test database."""
    cache = WeatherCache(db_path=test_settings.SQLITE_DB_PATH)
    await cache.initialize()
    yield cache
    # Cleanup
    if test_settings.SQLITE_DB_PATH.exists():
        test_settings.SQLITE_DB_PATH.unlink()

@pytest.fixture
def mock_llm_manager():
    """Mock LLM manager with realistic responses."""
    manager = MagicMock(spec=LLMManager)
    manager.generate_text = AsyncMock(return_value=(
        "This is a mock weather analysis response with detailed meteorological insights.",
        {
            "provider_used": "mock_provider", 
            "model_used": "mock_model",
            "tokens_used": {"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300}
        }
    ))
    manager.generate_json_response = AsyncMock(return_value=(
        {"analysis": "Mock JSON analysis", "confidence": 0.85},
        {"provider_used": "mock_provider", "model_used": "mock_model"}
    ))
    manager.list_available_providers_models = AsyncMock(return_value={
        "mock_provider": {"status": "configured", "default_model": "mock_model"}
    })
    return manager

@pytest.fixture
def mock_data_orchestrator():
    """Mock data orchestrator with realistic forecast data."""
    orchestrator = MagicMock(spec=WeatherDataOrchestrator)
    
    # Create realistic mock forecast
    mock_location = LocationInfo(
        name="Test City",
        coordinates=Coordinate(latitude=51.5074, longitude=-0.1278)
    )
    
    mock_forecasts = [
        DailyForecast(
            date=date.today(),
            temp_max_celsius=20.0,
            temp_min_celsius=10.0,
            condition_text="Partly cloudy",
            precipitation_mm=0.0,
            wind_speed_kph=15.0,
            detailed_summary="Mock forecast for testing"
        )
    ]
    
    mock_response = WeatherForecastResponse(
        location=mock_location,
        daily_forecasts=mock_forecasts,
        generated_at_utc=datetime.now(timezone.utc),
        data_source="Mock Data Source"
    )
    
    orchestrator.get_weather_data = AsyncMock(return_value=mock_response)
    return orchestrator

@pytest.fixture
def mock_weather_model():
    """Mock custom weather model."""
    model = MagicMock(spec=StubPhysicsEnhancedModel)
    model.is_loaded = True
    model.model_name = "MockPhysicsModel"
    model.model_version = "1.0-test"
    
    async def mock_predict(input_data):
        # Return modified copy of input data
        output = input_data.model_copy(deep=True)
        output.data_source = f"[Enhanced by {model.model_name}] {output.data_source}"
        return output
    
    model.predict = AsyncMock(side_effect=mock_predict)
    return model

@pytest.fixture
def forecast_service(mock_data_orchestrator, mock_llm_manager, mock_weather_model):
    """Forecast service with mocked dependencies."""
    return ForecastService(
        data_orchestrator=mock_data_orchestrator,
        llm_manager=mock_llm_manager,
        weather_model=mock_weather_model
    )

@pytest.fixture
def mock_openmeteo_api_response():
    """Mock Open-Meteo API response."""
    return {
        "latitude": 51.5074,
        "longitude": -0.1278,
        "timezone": "UTC",
        "current": {
            "time": 1684941600,
            "temperature_2m": 15.3,
            "apparent_temperature": 14.8,
            "precipitation": 0.0,
            "weather_code": 2,
            "cloud_cover": 25,
            "pressure_msl": 1015.3,
            "wind_speed_10m": 12.4,
            "wind_direction_10m": 270,
            "relative_humidity_2m": 65
        },
        "daily": {
            "time": ["2023-05-24", "2023-05-25", "2023-05-26"],
            "temperature_2m_max": [19.8, 20.2, 18.5],
            "temperature_2m_min": [12.3, 13.1, 11.9],
            "weather_code": [2, 3, 61],
            "precipitation_sum": [0.0, 0.0, 2.3],
            "precipitation_probability_mean": [0, 10, 70],
            "wind_speed_10m_max": [15.2, 12.3, 18.9],
            "wind_direction_10m_dominant": [270, 225, 180],
            "uv_index_max": [6.2, 5.8, 4.1],
            "sunrise": ["2023-05-24T04:52", "2023-05-25T04:51", "2023-05-26T04:50"],
            "sunset": ["2023-05-24T19:47", "2023-05-25T19:48", "2023-05-26T19:49"]
        }
    }

@pytest.fixture
def mock_httpx_client():
    """Mock HTTPX client for testing API calls."""
    with patch('httpx.AsyncClient') as mock_client:
        yield mock_client

# Performance testing fixtures
@pytest.fixture
def performance_timer():
    """Timer for performance testing."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            
        def start(self):
            self.start_time = time.perf_counter()
            
        def stop(self):
            self.end_time = time.perf_counter()
            
        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return Timer()

# Hypothesis settings for property-based testing
hypothesis_settings.register_profile("default", max_examples=100, deadline=5000)
hypothesis_settings.register_profile("comprehensive", max_examples=1000, deadline=10000)
hypothesis_settings.load_profile("default")

# You can add more shared fixtures here as your test suite grows.
# For example, fixtures for FastAPI TestClient:
# from fastapi.testclient import TestClient
# from openweather.api.main import app
# @pytest.fixture
# def test_client() -> TestClient:
#     return TestClient(app) 