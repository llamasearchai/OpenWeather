"""Tests for weather data sources and orchestration."""
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

from openweather.data.data_loader import (
    OpenMeteoDataSource,
    WeatherDataOrchestrator,
    _map_wmo_code_to_text
)

def test_wmo_code_mapping():
    """Test WMO weather code mapping."""
    # Test valid codes
    assert _map_wmo_code_to_text(0) == "Clear sky"
    assert _map_wmo_code_to_text(61) == "Rain: Slight"
    assert _map_wmo_code_to_text(95) == "Thunderstorm: Slight or moderate"
    
    # Test invalid code
    assert _map_wmo_code_to_text(999) == "Unknown code 999"
    
    # Test None
    assert _map_wmo_code_to_text(None) is None

@pytest.mark.asyncio
async def test_openmeteo_data_source(mock_openmeteo_response, mock_httpx_response):
    """Test Open-Meteo data source."""
    data_source = OpenMeteoDataSource()
    
    # Mock the HTTP client
    mock_response = mock_httpx_response(status_code=200, json_data=mock_openmeteo_response)
    
    with patch('httpx.AsyncClient.get', return_value=mock_response):
        result = await data_source.get_forecast(
            latitude=51.5074,
            longitude=-0.1278,
            num_days=5,
            location_name="London"
        )
        
        # Verify result
        assert result is not None
        assert result.location.name == "London"
        assert result.location.coordinates.latitude == 51.5074
        assert result.location.coordinates.longitude == -0.1278
        assert len(result.daily_forecasts) == 5
        assert result.data_source == "Open-Meteo API"
        
        # Verify current weather
        assert result.current_weather is not None
        assert result.current_weather.temp_celsius == 15.3
        assert result.current_weather.feels_like_celsius == 14.8
        
        # Verify first forecast day
        first_day = result.daily_forecasts[0]
        assert first_day.temp_max_celsius == 19.8
        assert first_day.temp_min_celsius == 12.3
        assert first_day.condition_text == "Partly cloudy"

@pytest.mark.asyncio
async def test_weather_data_orchestrator():
    """Test weather data orchestrator."""
    # Create mock data source
    mock_source = MagicMock()
    mock_source.source_name = "test-source"
    mock_source.get_forecast = AsyncMock()
    
    # Create mock forecast response
    from openweather.core.models_shared import (
        WeatherForecastResponse, 
        LocationInfo, 
        Coordinate
    )
    from datetime import datetime, timezone
    
    mock_forecast = WeatherForecastResponse(
        location=LocationInfo(
            name="London",
            coordinates=Coordinate(latitude=51.5074, longitude=-0.1278)
        ),
        daily_forecasts=[],
        generated_at_utc=datetime.now(timezone.utc),
        data_source="Test Source"
    )
    
    mock_source.get_forecast.return_value = mock_forecast
    
    # Initialize orchestrator with mock source
    orchestrator = WeatherDataOrchestrator(data_sources=[mock_source])
    
    # Test get_weather_data
    result = await orchestrator.get_weather_data(
        location_str="London",
        num_days=5,
        preferred_source="test-source"
    )
    
    # Verify result
    assert result is mock_forecast
    mock_source.get_forecast.assert_called_once()
    
    # Test fallback to simulation
    mock_source.get_forecast.reset_mock()
    mock_source.get_forecast.return_value = None
    
    with patch('openweather.data.data_loader.get_simulated_weather_forecast') as mock_sim:
        mock_sim.return_value = mock_forecast
        
        result = await orchestrator.get_weather_data(
            location_str="London",
            num_days=5
        )
        
        # Verify fallback to simulation
        assert result is mock_forecast
        mock_sim.assert_called_once()