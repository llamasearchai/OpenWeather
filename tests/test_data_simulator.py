"""Tests for the OpenWeather data simulator module."""
import pytest
from datetime import date, datetime, timezone, timedelta
from hypothesis import given, strategies as st, assume, settings, HealthCheck
import statistics

from openweather.data.data_simulator import (
    get_simulated_weather_forecast,
    generate_simulated_daily_forecast,
    generate_simulated_current_weather,
    _map_condition_code_to_text_sim,
    WEATHER_CONDITIONS_SIM
)
from openweather.core.models_shared import (
    WeatherForecastResponse,
    DailyForecast,
    CurrentWeather,
    LocationInfo,
    Coordinate
)
from openweather.core.utils import parse_location_string

# Test parameters
TEST_LOCATIONS = ["London", "New York", "35.6895,139.6917", "InvalidLocation"]
TEST_NUM_DAYS = [1, 5, 16]

# Property-based testing strategies
location_names = st.one_of(
    st.just("London"), st.just("New York"), st.just("Tokyo"), st.just("Paris"),
    st.just("Berlin"), st.just("Sydney"), st.just("Moscow")
)

coordinate_strings = st.builds(
    lambda lat, lon: f"{lat},{lon}",
    st.floats(min_value=-85, max_value=85, allow_nan=False),
    st.floats(min_value=-180, max_value=180, allow_nan=False)
)

location_inputs = st.one_of(location_names, coordinate_strings)

def test_generate_simulated_daily_forecast():
    """Test the generation of a single simulated daily forecast."""
    today = date.today()
    base_temp = 20.0
    daily_data = generate_simulated_daily_forecast(today, 0, base_temp)

    assert isinstance(daily_data, DailyForecast)
    assert daily_data.date == today
    assert daily_data.temp_max_celsius is not None
    assert daily_data.temp_min_celsius is not None
    assert daily_data.temp_min_celsius <= daily_data.temp_max_celsius
    assert daily_data.condition_text is not None
    assert daily_data.wind_speed_kph is not None
    assert daily_data.wind_direction_cardinal is not None
    assert daily_data.uv_index is not None
    assert daily_data.sunrise_utc is not None
    assert daily_data.sunset_utc is not None
    assert "Simulated" in daily_data.detailed_summary

def test_generate_simulated_current_weather():
    """Test the generation of simulated current weather."""
    lat, lon, name = parse_location_string("Paris")
    location = LocationInfo(name=name, coordinates=Coordinate(latitude=lat, longitude=lon))
    base_temp = 15.0
    current_data = generate_simulated_current_weather(location, base_temp)

    assert isinstance(current_data, CurrentWeather)
    assert current_data.observed_at_utc <= datetime.now(timezone.utc)
    assert current_data.temp_celsius is not None
    assert current_data.condition_text is not None
    assert current_data.wind_speed_kph is not None
    assert current_data.pressure_mb is not None
    assert current_data.humidity_percent is not None

@pytest.mark.parametrize("location_str", TEST_LOCATIONS)
@pytest.mark.parametrize("num_days", TEST_NUM_DAYS)
def test_get_simulated_weather_forecast(location_str: str, num_days: int):
    """Test the main function for getting a simulated weather forecast."""
    forecast_response = get_simulated_weather_forecast(location_str, num_days)

    assert isinstance(forecast_response, WeatherForecastResponse)
    assert forecast_response.location is not None
    assert forecast_response.location.name is not None # parse_location_string ensures a name
    
    # If "InvalidLocation", it defaults to London.
    expected_name, _, _ = parse_location_string(location_str)
    if location_str == "InvalidLocation": # parse_location_string defaults to London
         assert forecast_response.location.name == "London"
    elif "," in location_str : # It's a coordinate string
        assert forecast_response.location.name.startswith("Lat:")
    else: # It's a known city name
        assert forecast_response.location.name == location_str


    assert forecast_response.current_weather is not None
    assert isinstance(forecast_response.current_weather, CurrentWeather)

    assert len(forecast_response.daily_forecasts) == num_days
    for df in forecast_response.daily_forecasts:
        assert isinstance(df, DailyForecast)
        assert df.date is not None

    assert forecast_response.data_source == "Local Simulation"
    assert forecast_response.model_info["name"] == "BasicSimulator"
    assert forecast_response.model_info["version"] == "1.0"

def test_get_simulated_weather_forecast_edge_cases():
    """Test edge cases for get_simulated_weather_forecast."""
    # Test with 0 days (should still produce valid structure, maybe 0 daily forecasts)
    # The function get_simulated_weather_forecast itself uses range(num_days)
    # so 0 days will result in an empty list, which is acceptable.
    forecast_0_days = get_simulated_weather_forecast("Berlin", 0)
    assert len(forecast_0_days.daily_forecasts) == 0
    assert forecast_0_days.location.name == "Berlin"

    # Test with a very large number of days (simulator might cap it, or produce many days)
    # The simulator does not cap num_days, so it will produce as many as requested.
    # Let's test a moderately large number.
    forecast_many_days = get_simulated_weather_forecast("Moscow", 20) # Open-Meteo caps at 16, sim does not
    assert len(forecast_many_days.daily_forecasts) == 20
    assert forecast_many_days.location.name == "Moscow"

    # Test with empty location string (should default to London)
    forecast_empty_loc = get_simulated_weather_forecast("", 3)
    assert forecast_empty_loc.location.name == "London"
    assert len(forecast_empty_loc.daily_forecasts) == 3

def test_simulated_data_consistency():
    """Test consistency between current and first day forecast (approximate)."""
    forecast = get_simulated_weather_forecast("Sydney", 1)
    
    current_temp = forecast.current_weather.temp_celsius
    first_day_min = forecast.daily_forecasts[0].temp_min_celsius
    first_day_max = forecast.daily_forecasts[0].temp_max_celsius

    # Current temperature should be roughly within the first day's min/max range
    # Allowing for some random variation, let's use a buffer
    buffer = 15 # Increased buffer due to seasonal and random effects.
    assert first_day_min - buffer <= current_temp <= first_day_max + buffer

    # Date of first daily forecast should be today or tomorrow
    # (depending on execution time relative to UTC day change)
    today = datetime.now(timezone.utc).date()
    tomorrow = today + timedelta(days=1)
    assert forecast.daily_forecasts[0].date == today or forecast.daily_forecasts[0].date == tomorrow 

@given(
    location_str=location_inputs,
    num_days=st.integers(min_value=1, max_value=16)
)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_simulated_forecast_properties(location_str, num_days):
    """Property-based test for simulated weather forecast generation."""
    forecast = get_simulated_weather_forecast(location_str, num_days)
    
    # Basic structure validation
    assert isinstance(forecast, WeatherForecastResponse)
    assert forecast.location is not None
    assert len(forecast.daily_forecasts) == num_days
    assert forecast.current_weather is not None
    assert forecast.data_source == "Local Simulation"
    
    # Temperature consistency
    for daily in forecast.daily_forecasts:
        assert daily.temp_min_celsius <= daily.temp_max_celsius
        assert -70 <= daily.temp_min_celsius <= 60  # Reasonable earth temperature range
        assert -70 <= daily.temp_max_celsius <= 60
        
    # Date sequence validation
    expected_date = date.today()
    for i, daily in enumerate(forecast.daily_forecasts):
        assert daily.date == expected_date + timedelta(days=i)
        
    # Current weather validation
    current = forecast.current_weather
    assert current.observed_at_utc <= datetime.now(timezone.utc)
    assert current.temp_celsius is not None
    assert -70 <= current.temp_celsius <= 60

@given(
    current_date=st.dates(min_value=date(2020, 1, 1), max_value=date(2030, 12, 31)),
    day_offset=st.integers(min_value=0, max_value=15),
    base_temp=st.floats(min_value=-40, max_value=40, allow_nan=False)
)
def test_daily_forecast_generation_properties(current_date, day_offset, base_temp):
    """Property-based test for daily forecast generation."""
    forecast = generate_simulated_daily_forecast(current_date, day_offset, base_temp)
    
    # Date calculation
    expected_date = current_date + timedelta(days=day_offset)
    assert forecast.date == expected_date
    
    # Temperature relationships
    assert forecast.temp_min_celsius <= forecast.temp_max_celsius
    temp_diff = forecast.temp_max_celsius - forecast.temp_min_celsius
    assert 0 <= temp_diff <= 30  # Reasonable daily temperature range
    
    # Value ranges
    assert 0 <= forecast.precipitation_chance_percent <= 100
    if forecast.precipitation_mm:
        assert forecast.precipitation_mm >= 0
    if forecast.wind_speed_kph:
        assert forecast.wind_speed_kph >= 0
    if forecast.uv_index:
        assert 0 <= forecast.uv_index <= 15
        
    # Condition consistency
    assert forecast.condition_code in [code for code, _ in WEATHER_CONDITIONS_SIM]
    assert forecast.condition_text == _map_condition_code_to_text_sim(forecast.condition_code)

def test_seasonal_temperature_variation():
    """Test that simulated temperatures show seasonal variation."""
    location = "London"
    num_days = 365
    
    # Generate forecast for a full year
    forecasts_by_month = {}
    current_date = date(2024, 1, 1)
    
    for month in range(1, 13):
        monthly_date = current_date.replace(month=month, day=15)
        forecast = generate_simulated_daily_forecast(monthly_date, 0, 15.0)  # Base temp 15°C
        forecasts_by_month[month] = forecast
    
    # Summer months should be warmer than winter months (Northern Hemisphere assumption)
    summer_temps = [forecasts_by_month[m].temp_max_celsius for m in [6, 7, 8]]  # June, July, August
    winter_temps = [forecasts_by_month[m].temp_max_celsius for m in [12, 1, 2]]  # Dec, Jan, Feb
    
    avg_summer = statistics.mean(summer_temps)
    avg_winter = statistics.mean(winter_temps)
    
    assert avg_summer > avg_winter, "Summer should be warmer than winter"

def test_coordinate_parsing_integration():
    """Test integration between coordinate parsing and simulation."""
    test_cases = [
        ("51.5074,-0.1278", "London area coordinates"),
        ("40.7128,-74.0060", "New York area coordinates"),
        ("35.6895,139.6917", "Tokyo area coordinates")
    ]
    
    for coord_str, description in test_cases:
        forecast = get_simulated_weather_forecast(coord_str, 3)
        
        # Parse coordinates manually
        lat, lon, name = parse_location_string(coord_str)
        
        # Check that simulation used correct coordinates
        assert forecast.location.coordinates.latitude == lat
        assert forecast.location.coordinates.longitude == lon
        assert forecast.location.name == name

def test_simulation_reproducibility():
    """Test that simulation produces different results on repeated calls."""
    location = "London"
    num_days = 5
    
    # Generate multiple forecasts
    forecasts = [get_simulated_weather_forecast(location, num_days) for _ in range(10)]
    
    # Check that not all forecasts are identical (due to randomness)
    first_day_temps = [f.daily_forecasts[0].temp_max_celsius for f in forecasts]
    
    # Should have some variation in temperatures
    assert len(set(first_day_temps)) > 1, "Simulation should produce varied results"

@pytest.mark.performance
def test_simulation_performance(performance_timer):
    """Test simulation performance."""
    location = "London"
    num_days = 16  # Maximum days
    
    performance_timer.start()
    forecast = get_simulated_weather_forecast(location, num_days)
    performance_timer.stop()
    
    # Should complete quickly
    assert performance_timer.elapsed < 1.0, "Simulation should complete within 1 second"
    assert len(forecast.daily_forecasts) == num_days

def test_extreme_conditions_handling():
    """Test simulation with extreme input conditions."""
    # Test with maximum allowed days
    forecast_max = get_simulated_weather_forecast("London", 16)
    assert len(forecast_max.daily_forecasts) == 16
    
    # Test with minimum days
    forecast_min = get_simulated_weather_forecast("London", 1)
    assert len(forecast_min.daily_forecasts) == 1
    
    # Test with zero days
    forecast_zero = get_simulated_weather_forecast("London", 0)
    assert len(forecast_zero.daily_forecasts) == 0
    
    # Test with empty location (should default)
    forecast_empty = get_simulated_weather_forecast("", 5)
    assert forecast_empty.location.name == "London"  # Default fallback

def test_weather_condition_coverage():
    """Test that simulation covers various weather conditions."""
    # Generate many forecasts to see condition variety
    conditions_seen = set()
    
    for _ in range(100):
        forecast = get_simulated_weather_forecast("London", 1)
        conditions_seen.add(forecast.daily_forecasts[0].condition_code)
    
    # Should see multiple different weather conditions
    assert len(conditions_seen) >= 3, "Should generate variety of weather conditions"
    
    # All conditions should be valid
    valid_codes = {code for code, _ in WEATHER_CONDITIONS_SIM}
    assert conditions_seen.issubset(valid_codes), "All condition codes should be valid" 