"""Tests for the ForecastService."""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone, date
from hypothesis import given, strategies as st

from openweather.services.forecast_service import ForecastService, ForecastError
from openweather.core.models_shared import (
    WeatherForecastResponse, LLMAnalysisResponse, LLMAnalysisRequest,
    LocationInfo, Coordinate, DailyForecast
)
from openweather.data.data_loader import WeatherDataOrchestrator
from openweather.llm.llm_manager import LLMManager
from openweather.models.physics_model_stub import StubPhysicsEnhancedModel

@pytest.mark.asyncio
async def test_get_forecast_and_explain_success(forecast_service: ForecastService):
    """Test successful forecast retrieval and LLM explanation."""
    # Mock dependencies
    mock_forecast_data = WeatherForecastResponse(
        location=LocationInfo(name="Test City", coordinates=Coordinate(latitude=0, longitude=0)),
        daily_forecasts=[],
        data_source="mock_source",
        generated_at_utc=datetime.now(timezone.utc)
    )
    mock_llm_response = LLMAnalysisResponse(
        request_details=MagicMock(), # Simplified for this test
        analysis_text="Sunny with a chance of tests.",
        provider_used="mock_provider",
        model_used="mock_model",
        generated_at_utc=datetime.now(timezone.utc)
    )
    
    forecast_service.data_orchestrator.get_weather_data = AsyncMock(return_value=mock_forecast_data)
    forecast_service.llm_manager.generate_text = AsyncMock(return_value=(
        mock_llm_response.analysis_text, 
        {"provider_used": mock_llm_response.provider_used, "model_used": mock_llm_response.model_used}
    ))
    
    # Call the service method
    forecast, analysis = await forecast_service.get_forecast_and_explain(
        location_str="Test City",
        num_days=3,
        explain_with_llm=True
    )
    
    assert forecast is not None
    assert forecast.location.name == "Test City"
    assert analysis is not None
    assert analysis.analysis_text == "Sunny with a chance of tests."
    forecast_service.data_orchestrator.get_weather_data.assert_called_once_with(
        location_str="Test City", num_days=3, preferred_source=None
    )
    forecast_service.llm_manager.generate_text.assert_called_once()

@pytest.mark.asyncio
async def test_get_forecast_no_data(forecast_service: ForecastService):
    """Test scenario where no forecast data is returned from orchestrator."""
    forecast_service.data_orchestrator.get_weather_data = AsyncMock(return_value=None)
    
    forecast, analysis = await forecast_service.get_forecast_and_explain(
        location_str="Nowhere",
        num_days=1
    )
    
    assert forecast is None
    assert analysis is None
    forecast_service.data_orchestrator.get_weather_data.assert_called_once_with(
        location_str="Nowhere", num_days=1, preferred_source=None
    )

@pytest.mark.asyncio
async def test_forecast_service_success_path(forecast_service):
    """Test successful forecast retrieval and analysis."""
    # Test the full success path
    forecast, analysis = await forecast_service.get_forecast_and_explain(
        location_str="London",
        num_days=5,
        explain_with_llm=True,
        llm_provider="mock_provider"
    )
    
    # Verify forecast data
    assert forecast is not None
    assert isinstance(forecast, WeatherForecastResponse)
    assert forecast.location.name == "Test City"
    assert len(forecast.daily_forecasts) == 1
    
    # Verify LLM analysis
    assert analysis is not None
    assert isinstance(analysis, LLMAnalysisResponse)
    assert "mock weather analysis" in analysis.analysis_text.lower()
    assert analysis.provider_used == "mock_provider"

@pytest.mark.asyncio
async def test_forecast_service_without_llm(forecast_service):
    """Test forecast retrieval without LLM explanation."""
    forecast, analysis = await forecast_service.get_forecast_and_explain(
        location_str="Paris",
        num_days=3,
        explain_with_llm=False
    )
    
    assert forecast is not None
    assert analysis is None  # No LLM analysis requested

@pytest.mark.asyncio
async def test_forecast_service_data_source_failure(mock_llm_manager):
    """Test handling of data source failures."""
    # Create orchestrator that returns None
    failing_orchestrator = MagicMock(spec=WeatherDataOrchestrator)
    failing_orchestrator.get_weather_data = AsyncMock(return_value=None)
    
    service = ForecastService(
        data_orchestrator=failing_orchestrator,
        llm_manager=mock_llm_manager
    )
    
    forecast, analysis = await service.get_forecast_and_explain(
        location_str="NonexistentCity",
        num_days=5
    )
    
    assert forecast is None
    assert analysis is None

@pytest.mark.asyncio
async def test_forecast_service_llm_failure(mock_data_orchestrator):
    """Test handling of LLM failures."""
    # Create LLM manager that fails
    failing_llm = MagicMock(spec=LLMManager)
    failing_llm.generate_text = AsyncMock(return_value=(None, {"error": "LLM failed"}))
    
    service = ForecastService(
        data_orchestrator=mock_data_orchestrator,
        llm_manager=failing_llm
    )
    
    forecast, analysis = await service.get_forecast_and_explain(
        location_str="London",
        num_days=5,
        explain_with_llm=True
    )
    
    assert forecast is not None  # Weather data should still work
    assert analysis is not None   # Should create error analysis
    assert "LLM analysis failed" in analysis.analysis_text

@pytest.mark.asyncio
async def test_custom_weather_model_integration(mock_data_orchestrator, mock_llm_manager):
    """Test integration with custom weather models."""
    # Create a mock custom model
    custom_model = MagicMock(spec=StubPhysicsEnhancedModel)
    custom_model.is_loaded = True
    custom_model.model_name = "TestPhysicsModel"
    custom_model.model_version = "2.0"
    
    async def mock_predict(input_data):
        # Modify the forecast data
        output = input_data.model_copy(deep=True)
        output.data_source = f"[Enhanced by {custom_model.model_name}] {output.data_source}"
        # Adjust temperatures slightly
        for forecast in output.daily_forecasts:
            forecast.temp_max_celsius += 1.0
            forecast.temp_min_celsius += 0.5
        return output
    
    custom_model.predict = AsyncMock(side_effect=mock_predict)
    
    service = ForecastService(
        data_orchestrator=mock_data_orchestrator,
        llm_manager=mock_llm_manager,
        weather_model=custom_model
    )
    
    forecast, _ = await service.get_forecast_and_explain(
        location_str="London",
        num_days=3
    )
    
    # Verify custom model was applied
    assert "[Enhanced by TestPhysicsModel]" in forecast.data_source
    assert forecast.daily_forecasts[0].temp_max_celsius == 21.0  # Original 20.0 + 1.0
    assert forecast.daily_forecasts[0].temp_min_celsius == 10.5  # Original 10.0 + 0.5
    
    # Verify model was called
    custom_model.predict.assert_called_once()

@pytest.mark.asyncio
async def test_custom_llm_queries(forecast_service):
    """Test custom LLM query functionality."""
    custom_query = "What should I wear for this weather?"
    
    forecast, analysis = await forecast_service.get_forecast_and_explain(
        location_str="London",
        num_days=3,
        explain_with_llm=True,
        custom_llm_query=custom_query
    )
    
    assert forecast is not None
    assert analysis is not None
    assert analysis.request_details.query == custom_query

@pytest.mark.asyncio
async def test_different_output_formats(forecast_service):
    """Test different LLM output formats."""
    formats = ["markdown", "json", "text"]
    
    for output_format in formats:
        forecast, analysis = await forecast_service.get_forecast_and_explain(
            location_str="London",
            num_days=3,
            explain_with_llm=True,
            llm_output_format=output_format
        )
        
        assert forecast is not None
        assert analysis is not None
        assert analysis.request_details.output_format == output_format

@pytest.mark.asyncio
async def test_concurrent_requests():
    """Test handling of concurrent forecast requests."""
    # Create service with mocks
    mock_orchestrator = MagicMock(spec=WeatherDataOrchestrator)
    mock_llm = MagicMock(spec=LLMManager)
    
    # Add delays to simulate real API calls
    async def delayed_weather_response(*args, **kwargs):
        await asyncio.sleep(0.1)
        return MagicMock(spec=WeatherForecastResponse)
    
    async def delayed_llm_response(*args, **kwargs):
        await asyncio.sleep(0.1)
        return "Mock analysis", {"provider_used": "mock", "model_used": "mock"}
    
    mock_orchestrator.get_weather_data = AsyncMock(side_effect=delayed_weather_response)
    mock_llm.generate_text = AsyncMock(side_effect=delayed_llm_response)
    
    service = ForecastService(mock_orchestrator, mock_llm)
    
    # Run multiple concurrent requests
    tasks = [
        service.get_forecast_and_explain(f"City{i}", 3, explain_with_llm=True)
        for i in range(5)
    ]
    
    results = await asyncio.gather(*tasks)
    
    # All requests should complete successfully
    assert len(results) == 5
    for forecast, analysis in results:
        assert forecast is not None
        assert analysis is not None

@given(
    location=st.text(min_size=1, max_size=50),
    num_days=st.integers(min_value=1, max_value=16),
    explain=st.booleans()
)
@pytest.mark.asyncio
async def test_forecast_service_property_based(forecast_service, location, num_days, explain):
    """Property-based test for forecast service."""
    forecast, analysis = await forecast_service.get_forecast_and_explain(
        location_str=location,
        num_days=num_days,
        explain_with_llm=explain
    )
    
    # Basic invariants
    if forecast:
        assert isinstance(forecast, WeatherForecastResponse)
        assert forecast.location is not None
        
    if explain and forecast:
        # If explanation was requested and forecast succeeded, analysis should exist
        assert analysis is not None
        assert isinstance(analysis, LLMAnalysisResponse)
    elif not explain:
        # If no explanation requested, analysis should be None
        assert analysis is None

@pytest.mark.performance
@pytest.mark.asyncio
async def test_forecast_service_performance(forecast_service, performance_timer):
    """Test forecast service performance."""
    performance_timer.start()
    
    forecast, analysis = await forecast_service.get_forecast_and_explain(
        location_str="London",
        num_days=7,
        explain_with_llm=True
    )
    
    performance_timer.stop()
    
    # Should complete within reasonable time (mocked services should be fast)
    assert performance_timer.elapsed < 1.0
    assert forecast is not None
    assert analysis is not None

@pytest.mark.asyncio
async def test_error_propagation(mock_data_orchestrator, mock_llm_manager):
    """Test proper error propagation and handling."""
    # Test with orchestrator that raises exception
    mock_data_orchestrator.get_weather_data = AsyncMock(side_effect=Exception("Network error"))
    
    service = ForecastService(mock_data_orchestrator, mock_llm_manager)
    
    forecast, analysis = await service.get_forecast_and_explain(
        location_str="London",
        num_days=5
    )
    
    # Service should handle exceptions gracefully
    assert forecast is None
    assert analysis is None

@pytest.mark.asyncio
async def test_metadata_preservation(forecast_service):
    """Test that metadata is properly preserved through the service."""
    forecast, analysis = await forecast_service.get_forecast_and_explain(
        location_str="London",
        num_days=5,
        explain_with_llm=True,
        llm_provider="mock_provider",
        llm_model_name="mock_model"
    )
    
    assert forecast is not None
    assert analysis is not None
    
    # Check that LLM metadata was preserved
    assert analysis.provider_used == "mock_provider"
    assert analysis.model_used == "mock_model"
    assert analysis.tokens_used is not None
    assert "prompt_tokens" in analysis.tokens_used
