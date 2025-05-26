"""Tests for weather prediction models."""
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from pydantic import ValidationError

from hypothesis import given, strategies as st, settings, HealthCheck
from hypothesis.errors import InvalidArgument # For more specific error handling if needed

from openweather.models.physics_model_stub import StubPhysicsEnhancedModel
from openweather.models.mlx_model_runner import MLXWeatherModelRunner, MLX_MODEL_RUNNER_AVAILABLE
from openweather.core.models_shared import (
    Coordinate, LocationInfo, DailyForecast, 
    CurrentWeather, WeatherForecastResponse, LLMAnalysisResponse
)

@pytest.mark.asyncio
async def test_stub_physics_model_initialization():
    """Test initialization of physics model stub."""
    model = StubPhysicsEnhancedModel()
    
    # Verify model properties
    assert model.model_name == "StubPhysicsEnhancedPredictor"
    assert model.model_version == "1.0-stub"
    assert model.is_loaded is True
    
    # Verify default config
    assert "adjustment_factor_temp_max" in model.config
    assert "adjustment_factor_temp_min" in model.config
    assert "random_noise_stddev" in model.config

@pytest.mark.asyncio
async def test_stub_physics_model_predict():
    """Test prediction with physics model stub."""
    # Create a mock forecast
    from openweather.core.models_shared import (
        WeatherForecastResponse, 
        LocationInfo, 
        Coordinate, 
        DailyForecast
    )
    from datetime import date, datetime, timezone
    
    location = LocationInfo(
        name="Test City",
        coordinates=Coordinate(latitude=35.0, longitude=-75.0)
    )
    
    test_forecast = WeatherForecastResponse(
        location=location,
        daily_forecasts=[
            DailyForecast(
                date=date.today(),
                temp_max_celsius=25.0,
                temp_min_celsius=15.0,
                condition_code=100,
                condition_text="Clear",
                detailed_summary="Original forecast"
            )
        ],
        generated_at_utc=datetime.now(timezone.utc),
        data_source="Test Source"
    )
    
    # Initialize model and predict
    model = StubPhysicsEnhancedModel()
    result = await model.predict(test_forecast)
    
    # Verify predictions were adjusted
    assert result.daily_forecasts[0].temp_max_celsius != 25.0
    assert result.daily_forecasts[0].temp_min_celsius != 15.0
    assert "adjusted by StubPhysicsModel" in result.daily_forecasts[0].detailed_summary
    assert "[Enhanced by StubPhysicsModel]" in result.data_source
    assert result.model_info["name"] == "StubPhysicsEnhancedPredictor"

@pytest.mark.asyncio
async def test_mlx_model_runner():
    """Test MLX model runner if available."""
    if not MLX_MODEL_RUNNER_AVAILABLE:
        pytest.skip("MLX not available on this system")
        
    with patch('openweather.models.mlx_model_runner.MLX_MODEL_RUNNER_AVAILABLE', True), \
         patch('asyncio.sleep'):
        
        model = MLXWeatherModelRunner()
        
        # Verify model properties
        assert model.model_name == "MLXWeatherModelRunner"
        assert model.model_version == "0.1-alpha"
        
        # Create test forecast
        from openweather.core.models_shared import (
            WeatherForecastResponse, 
            LocationInfo, 
            Coordinate, 
            DailyForecast
        )
        from datetime import date, datetime, timezone
        
        location = LocationInfo(
            name="Test City",
            coordinates=Coordinate(latitude=35.0, longitude=-75.0)
        )
        
        test_forecast = WeatherForecastResponse(
            location=location,
            daily_forecasts=[
                DailyForecast(
                    date=date.today(),
                    temp_max_celsius=25.0,
                    temp_min_celsius=15.0,
                    condition_code=100,
                    condition_text="Clear",
                    detailed_summary="Original forecast"
                )
            ],
            generated_at_utc=datetime.now(timezone.utc),
            data_source="Test Source"
        )
        
        # Test prediction
        result = await model.predict(test_forecast)
        
        # Verify result
        assert "enhanced by MLX model" in result.daily_forecasts[0].detailed_summary
        assert "[MLX Enhanced]" in result.data_source

# --- Strategies for Pydantic Models ---

st_latitude = st.floats(min_value=-90, max_value=90, allow_nan=False, allow_infinity=False)
st_longitude = st.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False)

st_coordinates = st.builds(
    Coordinate,
    latitude=st_latitude,
    longitude=st_longitude
)

st_location_info = st.builds(
    LocationInfo,
    name=st.text(min_size=1, max_size=100),
    coordinates=st_coordinates,
    country_code=st.text(min_size=2, max_size=2) | st.none(),
    timezone=st.timezones() | st.none() # Use hypothesis strategy for timezones
)

st_iso_datetime_str = st.datetimes(
    min_value=datetime(2000, 1, 1, tzinfo=timezone.utc),
    max_value=datetime(2050, 12, 31, tzinfo=timezone.utc)
).map(lambda dt: dt.isoformat())


st_dates_as_date_obj = st.dates(min_value=date(2000, 1, 1), max_value=date(2050, 12, 31))

@st.composite
def st_daily_forecast_temperatures(draw):
    temp_min = draw(st.floats(min_value=-70, max_value=50, allow_nan=False, allow_infinity=False))
    temp_max = draw(st.floats(min_value=temp_min, max_value=60, allow_nan=False, allow_infinity=False))
    return temp_min, temp_max

st_daily_forecast = st.builds(
    DailyForecast,
    date=st_dates_as_date_obj,
    temp_max_celsius=st_daily_forecast_temperatures().map(lambda temps: temps[1]),
    temp_min_celsius=st_daily_forecast_temperatures().map(lambda temps: temps[0]),
    condition_code=st.integers(min_value=0, max_value=99) | st.none(),
    condition_text=st.text(max_size=100) | st.none(),
    precipitation_mm=st.floats(min_value=0, max_value=500, allow_nan=False, allow_infinity=False) | st.none(),
    precipitation_chance_percent=st.integers(min_value=0, max_value=100) | st.none(),
    wind_speed_kph=st.floats(min_value=0, max_value=300, allow_nan=False, allow_infinity=False) | st.none(),
    wind_direction_degrees=st.integers(min_value=0, max_value=360) | st.none(),
    wind_direction_cardinal=st.text(min_size=1, max_size=3) | st.none(), # e.g. N, NE, SW
    uv_index=st.floats(min_value=0, max_value=15, allow_nan=False, allow_infinity=False) | st.none(),
    sunrise_utc=st_iso_datetime_str.map(lambda s: datetime.fromisoformat(s)) | st.none(),
    sunset_utc=st_iso_datetime_str.map(lambda s: datetime.fromisoformat(s)) | st.none(),
    detailed_summary=st.text(max_size=500) | st.none()
)

st_current_weather = st.builds(
    CurrentWeather,
    observed_at_utc=st_iso_datetime_str.map(lambda s: datetime.fromisoformat(s)),
    temp_celsius=st.floats(min_value=-70, max_value=60, allow_nan=False, allow_infinity=False),
    feels_like_celsius=st.floats(min_value=-80, max_value=70, allow_nan=False, allow_infinity=False) | st.none(),
    # Other fields similar to DailyForecast
)

st_weather_forecast_response = st.builds(
    WeatherForecastResponse,
    location=st_location_info,
    current_weather=st_current_weather | st.none(),
    daily_forecasts=st.lists(st_daily_forecast, min_size=0, max_size=16),
    generated_at_utc=st_iso_datetime_str.map(lambda s: datetime.fromisoformat(s)),
    data_source=st.text(min_size=3, max_size=50),
    model_info=st.dictionaries(st.text(min_size=1, max_size=20), st.text(max_size=50) | st.integers() | st.floats()) | st.none()
)

st_llm_analysis_response = st.builds(
    LLMAnalysisResponse,
    analysis_text=st.text(min_size=10),
    provider_used=st.text(min_size=3, max_size=30),
    model_used=st.text(min_size=3, max_size=50),
    prompt_tokens=st.integers(min_value=0) | st.none(),
    completion_tokens=st.integers(min_value=0) | st.none(),
    total_tokens=st.integers(min_value=0) | st.none(),
    processing_time_ms=st.floats(min_value=0) | st.none()
)


# --- Hypothesis Tests ---

@settings(suppress_health_check=[HealthCheck.data_too_large, HealthCheck.too_slow], deadline=None)
@given(data=st_coordinates)
def test_coordinate_serialization_hypothesis(data: Coordinate):
    serialized = data.dict()
    deserialized = Coordinate(**serialized)
    assert deserialized == data
    assert -90 <= data.latitude <= 90
    assert -180 <= data.longitude <= 180

@settings(suppress_health_check=[HealthCheck.data_too_large, HealthCheck.too_slow], deadline=None)
@given(data=st_daily_forecast)
def test_daily_forecast_serialization_hypothesis(data: DailyForecast):
    # Constraint: temp_min_celsius should be less than or equal to temp_max_celsius
    # This is handled by the st_daily_forecast_temperatures strategy.
    # If they are generated independently, we would need a filter or specific check here.
    assert data.temp_min_celsius <= data.temp_max_celsius
    
    serialized = data.dict()
    deserialized = DailyForecast(**serialized)
    assert deserialized == data
    if data.sunrise_utc and data.sunset_utc: # Basic sanity check
        assert data.sunrise_utc.tzinfo is not None
        assert data.sunset_utc.tzinfo is not None


@settings(suppress_health_check=[HealthCheck.data_too_large, HealthCheck.too_slow], deadline=None)
@given(data=st_weather_forecast_response)
def test_weather_forecast_response_serialization_hypothesis(data: WeatherForecastResponse):
    # Ensure all daily forecasts within the response also satisfy their constraints
    for daily in data.daily_forecasts:
        assert daily.temp_min_celsius <= daily.temp_max_celsius
        if daily.sunrise_utc and daily.sunset_utc:
             assert daily.sunrise_utc < daily.sunset_utc # A common expectation

    serialized = data.dict()
    deserialized = WeatherForecastResponse(**serialized)
    assert deserialized == data
    assert data.generated_at_utc.tzinfo is not None


@settings(suppress_health_check=[HealthCheck.data_too_large, HealthCheck.too_slow], deadline=None)
@given(data=st_llm_analysis_response)
def test_llm_analysis_response_serialization_hypothesis(data: LLMAnalysisResponse):
    serialized = data.dict()
    deserialized = LLMAnalysisResponse(**serialized)
    assert deserialized == data
    if data.prompt_tokens and data.completion_tokens and data.total_tokens:
        assert data.prompt_tokens + data.completion_tokens <= data.total_tokens + 5 # Allow small discrepancies for some models

# Example of testing validation errors (though Hypothesis tends to generate valid data)
def test_coordinate_invalid_latitude():
    with pytest.raises(ValidationError):
        Coordinate(latitude=91, longitude=0)
    with pytest.raises(ValidationError):
        Coordinate(latitude=-91, longitude=0)

# Add more specific tests for model validators if they exist