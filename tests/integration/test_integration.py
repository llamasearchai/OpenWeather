"""Integration tests for OpenWeather platform."""

import pytest
import asyncio
import httpx
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List
from unittest.mock import patch, Mock, AsyncMock

from openweather.api.main import app
from openweather.services.forecast_service import WeatherForecastService
from openweather.services.llm_manager import LLMManager
from openweather.drone.flight_planner import FlightPlanner
from openweather.drone.safety_analyzer import SafetyAnalyzer
from openweather.drone.mavlink_connector import MAVLinkConnector
from openweather.drone.models import DronePosition, WeatherConditions, FlightPlan
from openweather.core.config import settings
from openweather.data.weather_data import WeatherDataProvider
from openweather.agents.weather_analysis_agent import WeatherAnalysisAgent


@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for the FastAPI application."""
    
    @pytest.fixture
    async def client(self):
        """Create async HTTP client for testing."""
        async with httpx.AsyncClient(app=app, base_url="http://test") as client:
            yield client
    
    @pytest.mark.asyncio
    async def test_health_endpoints(self, client):
        """Test health check endpoints."""
        # Test root endpoint
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "OpenWeather API" in data["message"]
        assert "version" in data
        
        # Test health endpoint
        response = await client.get("/api/v1/health")
        assert response.status_code == 200
        health_data = response.json()
        assert health_data["status"] == "healthy"
        assert "timestamp" in health_data
        assert "uptime" in health_data
    
    @pytest.mark.asyncio
    async def test_forecast_endpoint_integration(self, client):
        """Test forecast API endpoint with real service integration."""
        # Mock external weather service
        with patch('openweather.services.forecast_service.WeatherDataProvider') as mock_provider:
            mock_data = {
                "current": {
                    "temperature": 22.5,
                    "humidity": 65,
                    "wind_speed": 8.2,
                    "wind_direction": 180,
                    "description": "Partly cloudy"
                },
                "daily": [
                    {
                        "date": datetime.now().isoformat(),
                        "temperature_max": 25.0,
                        "temperature_min": 18.0,
                        "humidity": 60,
                        "wind_speed": 10.0,
                        "precipitation": 0.0,
                        "description": "Sunny"
                    }
                ]
            }
            mock_provider.return_value.get_forecast.return_value = mock_data
            
            response = await client.get(
                "/api/v1/forecast/",
                params={"location": "San Francisco, CA", "days": 3}
            )
            
            assert response.status_code == 200
            forecast_data = response.json()
            assert forecast_data["location"] == "San Francisco, CA"
            assert "daily_forecasts" in forecast_data
            assert len(forecast_data["daily_forecasts"]) <= 3
    
    @pytest.mark.asyncio
    async def test_llm_agent_endpoint_integration(self, client):
        """Test LLM agent endpoint with real service integration."""
        with patch('openweather.services.llm_manager.LLMManager') as mock_llm:
            mock_response = "Based on the current weather data, it looks like partly cloudy conditions with mild temperatures."
            mock_llm.return_value.process_query.return_value = mock_response
            
            response = await client.post(
                "/api/v1/agent/query",
                json={"query": "What's the weather like today?"}
            )
            
            assert response.status_code == 200
            agent_data = response.json()
            assert agent_data["response"] == mock_response
            assert "query" in agent_data
            assert "timestamp" in agent_data
    
    @pytest.mark.asyncio
    async def test_drone_endpoints_integration(self, client):
        """Test drone-related API endpoints."""
        # Test flight planning endpoint
        waypoints_data = {
            "waypoints": [
                {"latitude": 37.7749, "longitude": -122.4194, "altitude": 100.0},
                {"latitude": 37.7849, "longitude": -122.4094, "altitude": 120.0}
            ],
            "max_altitude": 150.0,
            "max_speed": 20.0
        }
        
        with patch('openweather.drone.flight_planner.FlightPlanner') as mock_planner:
            mock_plan = Mock()
            mock_plan.drone_id = "TEST_DRONE"
            mock_plan.total_distance = 1000.0
            mock_plan.estimated_duration = 300
            mock_planner.return_value.create_flight_plan.return_value = mock_plan
            
            response = await client.post(
                "/api/v1/drone/plan-flight",
                json=waypoints_data
            )
            
            assert response.status_code == 200
            plan_data = response.json()
            assert "flight_plan_id" in plan_data
            assert plan_data["total_distance"] == 1000.0
        
        # Test safety assessment endpoint
        safety_data = {
            "position": {"latitude": 37.7749, "longitude": -122.4194, "altitude": 100.0},
            "weather_conditions": {
                "wind_speed": 10.0,
                "wind_direction": 180,
                "temperature": 20.0,
                "humidity": 60,
                "visibility": 10000,
                "precipitation": 0.0,
                "cloud_ceiling": 2000
            }
        }
        
        with patch('openweather.drone.safety_analyzer.SafetyAnalyzer') as mock_analyzer:
            mock_assessment = Mock()
            mock_assessment.is_safe = True
            mock_assessment.risk_level = "LOW"
            mock_assessment.warnings = []
            mock_analyzer.return_value.analyze_weather_conditions.return_value = mock_assessment
            
            response = await client.post(
                "/api/v1/drone/assess-safety",
                json=safety_data
            )
            
            assert response.status_code == 200
            safety_result = response.json()
            assert safety_result["is_safe"] is True
            assert safety_result["risk_level"] == "LOW"


@pytest.mark.integration
class TestServiceIntegration:
    """Integration tests for service layer interactions."""
    
    @pytest.mark.asyncio
    async def test_forecast_service_data_provider_integration(self):
        """Test integration between forecast service and data provider."""
        forecast_service = WeatherForecastService()
        
        # Mock the data provider to return consistent data
        with patch.object(forecast_service, '_data_provider') as mock_provider:
            mock_weather_data = {
                "current": {
                    "temperature": 22.5,
                    "humidity": 65,
                    "wind_speed": 8.2,
                    "wind_direction": 180,
                    "description": "Partly cloudy",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                },
                "daily": []
            }
            
            # Generate mock daily forecasts
            for i in range(7):
                date = datetime.now(timezone.utc) + timedelta(days=i)
                mock_weather_data["daily"].append({
                    "date": date.isoformat(),
                    "temperature_max": 25.0 + i,
                    "temperature_min": 18.0 + i,
                    "humidity": 60 - i,
                    "wind_speed": 10.0 + i * 0.5,
                    "precipitation": 0.0,
                    "description": f"Day {i+1} forecast"
                })
            
            mock_provider.get_forecast.return_value = mock_weather_data
            
            # Test forecast retrieval
            forecast = await forecast_service.get_forecast("San Francisco, CA", 7)
            
            assert forecast is not None
            assert forecast.location == "San Francisco, CA"
            assert len(forecast.daily_forecasts) == 7
            assert forecast.current_conditions.temperature == 22.5
            
            # Verify data provider was called correctly
            mock_provider.get_forecast.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_llm_weather_integration(self):
        """Test integration between LLM manager and weather services."""
        llm_manager = LLMManager()
        
        # Mock both LLM interface and weather service
        with patch.object(llm_manager, '_llm_interface') as mock_llm, \
             patch('openweather.services.forecast_service.WeatherForecastService') as mock_forecast:
            
            # Mock weather data
            mock_forecast_data = Mock()
            mock_forecast_data.location = "San Francisco, CA"
            mock_forecast_data.current_conditions.temperature = 22.5
            mock_forecast_data.current_conditions.description = "Partly cloudy"
            mock_forecast.return_value.get_forecast.return_value = mock_forecast_data
            
            # Mock LLM response
            mock_llm.generate_response.return_value = "The weather in San Francisco is currently 22.5°C and partly cloudy."
            
            # Test query processing
            query = "What's the weather like in San Francisco?"
            response = await llm_manager.process_query(query)
            
            assert response is not None
            assert "22.5" in response
            assert "San Francisco" in response
            assert "partly cloudy" in response.lower()
    
    @pytest.mark.asyncio
    async def test_drone_weather_integration(self):
        """Test integration between drone services and weather data."""
        flight_planner = FlightPlanner()
        safety_analyzer = SafetyAnalyzer()
        
        # Create test waypoints
        waypoints = [
            DronePosition(37.7749, -122.4194, 100.0, 50.0),
            DronePosition(37.7849, -122.4094, 120.0, 70.0)
        ]
        
        # Mock weather service for flight planning
        with patch.object(flight_planner, '_get_weather_forecast') as mock_weather:
            mock_weather.return_value = WeatherConditions(
                wind_speed=8.0, wind_direction=180, temperature=20.0,
                humidity=60, visibility=10000, precipitation=0.0, cloud_ceiling=2000
            )
            
            # Create flight plan
            flight_plan = await flight_planner.create_flight_plan(
                waypoints=waypoints,
                max_altitude=150.0,
                max_speed=20.0
            )
            
            assert flight_plan is not None
            assert len(flight_plan.segments) == 1
            
            # Validate flight plan with safety analyzer
            validation_result = await flight_planner.validate_flight_plan(flight_plan)
            
            assert validation_result.is_safe
            assert validation_result.risk_level == "LOW"


@pytest.mark.integration
class TestEndToEndWorkflows:
    """End-to-end integration tests for complete workflows."""
    
    @pytest.mark.asyncio
    async def test_weather_to_drone_workflow(self):
        """Test complete workflow from weather query to drone mission planning."""
        # Initialize all services
        forecast_service = WeatherForecastService()
        llm_manager = LLMManager()
        flight_planner = FlightPlanner()
        safety_analyzer = SafetyAnalyzer()
        mavlink_connector = MAVLinkConnector()
        
        # Step 1: Get weather forecast
        with patch.object(forecast_service, '_data_provider') as mock_provider:
            mock_weather_data = {
                "current": {
                    "temperature": 22.5,
                    "humidity": 65,
                    "wind_speed": 8.2,
                    "wind_direction": 180,
                    "description": "Clear skies"
                },
                "daily": [{
                    "date": datetime.now().isoformat(),
                    "temperature_max": 25.0,
                    "temperature_min": 18.0,
                    "wind_speed": 10.0,
                    "precipitation": 0.0
                }]
            }
            mock_provider.get_forecast.return_value = mock_weather_data
            
            forecast = await forecast_service.get_forecast("San Francisco, CA", 1)
            assert forecast is not None
        
        # Step 2: Get LLM analysis of weather suitability
        with patch.object(llm_manager, '_llm_interface') as mock_llm:
            mock_llm.generate_response.return_value = "Weather conditions are suitable for drone flight with clear skies and moderate winds."
            
            weather_analysis = await llm_manager.process_query(
                f"Analyze the weather conditions for drone flight: {forecast.current_conditions.description}, "
                f"wind speed {forecast.current_conditions.wind_speed} m/s"
            )
            assert "suitable" in weather_analysis.lower()
        
        # Step 3: Plan drone mission
        waypoints = [
            DronePosition(37.7749, -122.4194, 100.0, 50.0),
            DronePosition(37.7849, -122.4094, 120.0, 70.0),
            DronePosition(37.7949, -122.3994, 110.0, 60.0)
        ]
        
        with patch.object(flight_planner, '_get_weather_forecast') as mock_flight_weather:
            mock_flight_weather.return_value = WeatherConditions(
                wind_speed=8.2, wind_direction=180, temperature=22.5,
                humidity=65, visibility=10000, precipitation=0.0, cloud_ceiling=3000
            )
            
            flight_plan = await flight_planner.create_flight_plan(
                waypoints=waypoints,
                max_altitude=150.0,
                max_speed=20.0
            )
            assert flight_plan is not None
            assert len(flight_plan.segments) == 2
        
        # Step 4: Validate safety
        safety_conditions = WeatherConditions(
            wind_speed=8.2, wind_direction=180, temperature=22.5,
            humidity=65, visibility=10000, precipitation=0.0, cloud_ceiling=3000
        )
        
        safety_assessment = safety_analyzer.analyze_weather_conditions(safety_conditions)
        assert safety_assessment.is_safe
        assert safety_assessment.risk_level == "LOW"
        
        # Step 5: Simulate mission execution
        with patch.object(mavlink_connector, 'connect') as mock_connect, \
             patch.object(mavlink_connector, 'execute_flight_plan') as mock_execute:
            
            mock_connect.return_value = True
            mock_execute.return_value = True
            
            connection_success = await mavlink_connector.connect("tcp:127.0.0.1:5760")
            assert connection_success
            
            execution_success = await mavlink_connector.execute_flight_plan(flight_plan)
            assert execution_success
        
        print("Complete weather-to-drone workflow integration test passed")
    
    @pytest.mark.asyncio
    async def test_agent_driven_mission_planning(self):
        """Test agent-driven mission planning workflow."""
        agent = WeatherAnalysisAgent()
        
        # Mock all dependencies
        with patch('openweather.services.forecast_service.WeatherForecastService') as mock_forecast, \
             patch('openweather.services.llm_manager.LLMManager') as mock_llm, \
             patch('openweather.drone.flight_planner.FlightPlanner') as mock_planner:
            
            # Setup mocks
            mock_forecast_data = Mock()
            mock_forecast_data.location = "Mission Location"
            mock_forecast_data.current_conditions.wind_speed = 8.0
            mock_forecast_data.current_conditions.description = "Clear"
            mock_forecast.return_value.get_forecast.return_value = mock_forecast_data
            
            mock_llm.return_value.process_query.return_value = "Weather is suitable for drone operations. Recommend proceeding with mission."
            
            mock_flight_plan = Mock()
            mock_flight_plan.total_distance = 2000.0
            mock_flight_plan.estimated_duration = 600
            mock_planner.return_value.create_flight_plan.return_value = mock_flight_plan
            
            # Execute agent workflow
            mission_request = {
                "location": "San Francisco, CA",
                "mission_type": "surveillance",
                "waypoints": [
                    {"latitude": 37.7749, "longitude": -122.4194, "altitude": 100.0},
                    {"latitude": 37.7849, "longitude": -122.4094, "altitude": 120.0}
                ]
            }
            
            result = await agent.plan_mission(mission_request)
            
            assert result["status"] == "success"
            assert result["weather_suitable"] is True
            assert result["flight_plan_created"] is True
            assert "mission_analysis" in result
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self):
        """Test error handling across integrated services."""
        forecast_service = WeatherForecastService()
        flight_planner = FlightPlanner()
        
        # Test weather service failure
        with patch.object(forecast_service, '_data_provider') as mock_provider:
            mock_provider.get_forecast.side_effect = Exception("Weather service unavailable")
            
            with pytest.raises(Exception) as exc_info:
                await forecast_service.get_forecast("Invalid Location", 7)
            
            assert "Weather service unavailable" in str(exc_info.value)
        
        # Test flight planning with invalid waypoints
        invalid_waypoints = [
            DronePosition(200.0, 200.0, 100.0, 50.0),  # Invalid coordinates
        ]
        
        with pytest.raises(ValueError):
            await flight_planner.create_flight_plan(
                waypoints=invalid_waypoints,
                max_altitude=150.0,
                max_speed=20.0
            )
        
        print("Error handling integration test passed")


@pytest.mark.integration
class TestDataConsistency:
    """Integration tests for data consistency across services."""
    
    @pytest.mark.asyncio
    async def test_weather_data_consistency(self):
        """Test weather data consistency across different service calls."""
        forecast_service = WeatherForecastService()
        
        with patch.object(forecast_service, '_data_provider') as mock_provider:
            # Mock consistent weather data
            consistent_data = {
                "current": {
                    "temperature": 22.5,
                    "humidity": 65,
                    "wind_speed": 8.2,
                    "wind_direction": 180,
                    "description": "Partly cloudy",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                },
                "daily": []
            }
            mock_provider.get_forecast.return_value = consistent_data
            
            # Multiple calls should return consistent data
            forecast1 = await forecast_service.get_forecast("San Francisco, CA", 1)
            forecast2 = await forecast_service.get_forecast("San Francisco, CA", 1)
            
            assert forecast1.current_conditions.temperature == forecast2.current_conditions.temperature
            assert forecast1.current_conditions.wind_speed == forecast2.current_conditions.wind_speed
            assert forecast1.current_conditions.description == forecast2.current_conditions.description
    
    @pytest.mark.asyncio
    async def test_drone_position_consistency(self):
        """Test drone position data consistency across services."""
        flight_planner = FlightPlanner()
        safety_analyzer = SafetyAnalyzer()
        
        # Test position used in both flight planning and safety analysis
        test_position = DronePosition(37.7749, -122.4194, 100.0, 50.0)
        
        # Both services should handle the same position consistently
        altitude_assessment = safety_analyzer.assess_altitude_restrictions(test_position)
        no_fly_assessment = safety_analyzer.assess_no_fly_zones(test_position)
        
        # Position should be valid for both assessments (assuming safe test coordinates)
        assert altitude_assessment is not None
        assert no_fly_assessment is not None
        
        # Create waypoints including the test position
        waypoints = [
            test_position,
            DronePosition(37.7849, -122.4094, 120.0, 70.0)
        ]
        
        with patch.object(flight_planner, '_get_weather_forecast'):
            flight_plan = await flight_planner.create_flight_plan(
                waypoints=waypoints,
                max_altitude=150.0,
                max_speed=20.0
            )
            
            # Flight plan should include the test position
            assert flight_plan is not None
            assert flight_plan.segments[0].start_position.latitude == test_position.latitude
            assert flight_plan.segments[0].start_position.longitude == test_position.longitude


@pytest.mark.integration
class TestPerformanceIntegration:
    """Integration tests focusing on performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_concurrent_service_requests(self):
        """Test performance when multiple services are used concurrently."""
        forecast_service = WeatherForecastService()
        llm_manager = LLMManager()
        
        # Mock services for predictable performance
        with patch.object(forecast_service, '_data_provider') as mock_provider, \
             patch.object(llm_manager, '_llm_interface') as mock_llm:
            
            mock_provider.get_forecast.return_value = {"current": {}, "daily": []}
            mock_llm.generate_response.return_value = "Test response"
            
            # Concurrent requests to different services
            tasks = []
            
            # Weather requests
            for i in range(5):
                tasks.append(forecast_service.get_forecast(f"City {i}", 3))
            
            # LLM requests
            for i in range(5):
                tasks.append(llm_manager.process_query(f"Query {i}"))
            
            # Execute all tasks concurrently
            import time
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            # Verify all requests completed successfully
            successful_results = [r for r in results if not isinstance(r, Exception)]
            assert len(successful_results) >= 8  # Allow for some failures
            
            # Performance should be reasonable
            total_time = end_time - start_time
            assert total_time < 10.0, f"Concurrent requests took too long: {total_time:.3f}s"
            
            print(f"Concurrent service requests completed in {total_time:.3f}s")


# Utility functions for integration testing
async def setup_test_environment():
    """Setup test environment for integration tests."""
    # Initialize test database
    # Setup mock external services
    # Configure test settings
    pass


async def cleanup_test_environment():
    """Cleanup test environment after integration tests."""
    # Clear test database
    # Reset service states
    # Cleanup temporary files
    pass


# Fixtures for integration testing
@pytest.fixture(scope="session")
async def integration_setup():
    """Session-wide setup for integration tests."""
    await setup_test_environment()
    yield
    await cleanup_test_environment()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"]) 