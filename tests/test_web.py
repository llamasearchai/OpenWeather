"""Tests for the web interface."""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

from openweather.web.app import app, master

@pytest.fixture
def client():
    """Create a test client for the web interface."""
    return TestClient(app)

@pytest.fixture
def mock_master():
    """Mock the OpenWeatherMaster class."""
    with patch('openweather.web.app.master') as mock:
        # Setup mock for get_forecast
        mock.get_forecast = AsyncMock()
        mock.get_forecast.return_value = (
            {
                "location": {
                    "name": "London",
                    "coordinates": {"latitude": 51.5074, "longitude": -0.1278}
                },
                "daily_forecasts": [
                    {
                        "date": "2023-01-01",
                        "temp_max_celsius": 10.0,
                        "temp_min_celsius": 5.0,
                        "condition_text": "Clear"
                    }
                ],
                "data_source": "Test"
            },
            None
        )
        
        # Setup mock for analyze_weather
        mock.analyze_weather = AsyncMock()
        mock.analyze_weather.return_value = {
            "status": "success",
            "response_text": "Test analysis response",
            "location_used": "London",
            "llm_provider_used": "test_provider",
            "llm_model_used": "test_model"
        }
        
        # Setup mock for llm_manager
        mock.llm_manager = MagicMock()
        mock.llm_manager.list_available_providers_models = AsyncMock()
        mock.llm_manager.list_available_providers_models.return_value = {
            "local_ollama": {"status": "configured"},
            "openai": {"status": "configured"}
        }
        
        yield mock

def test_index_page(client, mock_master):
    """Test the index page."""
    response = client.get("/")
    assert response.status_code == 200
    assert "OpenWeather" in response.text
    assert "Get Weather Forecast" in response.text

def test_analyze_form_page(client, mock_master):
    """Test the analyze form page."""
    response = client.get("/analyze")
    assert response.status_code == 200
    assert "Weather Analysis" in response.text
    assert "Your Question" in response.text

def test_get_forecast(client, mock_master):
    """Test getting a forecast."""
    response = client.post(
        "/forecast",
        data={
            "location": "London",
            "days": 5,
            "explain": False
        }
    )
    assert response.status_code == 200
    assert "Weather Forecast for London" in response.text
    mock_master.get_forecast.assert_called_once()

def test_analyze_weather(client, mock_master):
    """Test analyzing weather."""
    response = client.post(
        "/analyze",
        data={
            "query": "How's the weather?",
            "location": "London",
            "provider": ""
        }
    )
    assert response.status_code == 200
    assert "Your Question" in response.text
    assert "How's the weather?" in response.text
    mock_master.analyze_weather.assert_called_once()

def test_error_handling(client, mock_master):
    """Test error handling."""
    # Make get_forecast raise an exception
    mock_master.get_forecast.side_effect = Exception("Test error")
    
    response = client.post(
        "/forecast",
        data={
            "location": "London",
            "days": 5,
            "explain": False
        }
    )
    assert response.status_code == 200
    assert "Error" in response.text
    assert "Test error" in response.text