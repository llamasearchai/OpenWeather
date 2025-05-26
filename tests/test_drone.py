"""Comprehensive test suite for drone functionality."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone
from typing import Dict, Any

from openweather.drone.models import (
    DronePosition, WeatherConditions, FlightPlan, FlightSegment,
    SafetyAssessment, DroneStatus, TelemetryData
)
from openweather.drone.safety_analyzer import SafetyAnalyzer
from openweather.drone.flight_planner import FlightPlanner
from openweather.drone.mavlink_connector import MAVLinkConnector


class TestDroneModels:
    """Test drone data models."""
    
    def test_drone_position_creation(self):
        """Test DronePosition model creation and validation."""
        position = DronePosition(
            latitude=37.7749,
            longitude=-122.4194,
            altitude_msl=100.0,
            altitude_rel=50.0
        )
        assert position.latitude == 37.7749
        assert position.longitude == -122.4194
        assert position.altitude_msl == 100.0
        assert position.altitude_rel == 50.0
    
    def test_weather_conditions_validation(self):
        """Test WeatherConditions model validation."""
        conditions = WeatherConditions(
            wind_speed=10.5,
            wind_direction=180,
            temperature=25.0,
            humidity=65,
            visibility=10000,
            precipitation=0.0,
            cloud_ceiling=1500
        )
        assert conditions.wind_speed == 10.5
        assert conditions.wind_direction == 180
        assert conditions.is_safe_for_flight()
    
    def test_weather_conditions_unsafe_wind(self):
        """Test unsafe weather conditions detection."""
        conditions = WeatherConditions(
            wind_speed=35.0,  # Unsafe wind speed
            wind_direction=180,
            temperature=25.0,
            humidity=65,
            visibility=10000,
            precipitation=0.0,
            cloud_ceiling=1500
        )
        assert not conditions.is_safe_for_flight()
    
    def test_flight_segment_creation(self):
        """Test FlightSegment model creation."""
        start_pos = DronePosition(
            latitude=37.7749, longitude=-122.4194,
            altitude_msl=100.0, altitude_rel=50.0
        )
        end_pos = DronePosition(
            latitude=37.7849, longitude=-122.4094,
            altitude_msl=120.0, altitude_rel=70.0
        )
        
        segment = FlightSegment(
            start_position=start_pos,
            end_position=end_pos,
            altitude=75.0,
            speed=15.0,
            estimated_duration=120
        )
        
        assert segment.start_position.latitude == 37.7749
        assert segment.end_position.latitude == 37.7849
        assert segment.altitude == 75.0
        assert segment.speed == 15.0


class TestSafetyAnalyzer:
    """Test safety analysis functionality."""
    
    @pytest.fixture
    def safety_analyzer(self):
        """Create SafetyAnalyzer instance for testing."""
        return SafetyAnalyzer()
    
    @pytest.fixture
    def safe_weather(self):
        """Create safe weather conditions for testing."""
        return WeatherConditions(
            wind_speed=8.0,
            wind_direction=180,
            temperature=20.0,
            humidity=60,
            visibility=10000,
            precipitation=0.0,
            cloud_ceiling=2000
        )
    
    @pytest.fixture
    def unsafe_weather(self):
        """Create unsafe weather conditions for testing."""
        return WeatherConditions(
            wind_speed=40.0,  # Too high
            wind_direction=180,
            temperature=20.0,
            humidity=60,
            visibility=1000,  # Too low
            precipitation=5.0,  # Rain
            cloud_ceiling=100   # Too low
        )
    
    def test_analyze_weather_conditions_safe(self, safety_analyzer, safe_weather):
        """Test safety analysis with safe weather conditions."""
        assessment = safety_analyzer.analyze_weather_conditions(safe_weather)
        
        assert assessment.is_safe
        assert assessment.risk_level == "LOW"
        assert len(assessment.warnings) == 0
        assert len(assessment.restrictions) == 0
    
    def test_analyze_weather_conditions_unsafe(self, safety_analyzer, unsafe_weather):
        """Test safety analysis with unsafe weather conditions."""
        assessment = safety_analyzer.analyze_weather_conditions(unsafe_weather)
        
        assert not assessment.is_safe
        assert assessment.risk_level == "HIGH"
        assert len(assessment.warnings) > 0
        assert len(assessment.restrictions) > 0
    
    def test_assess_no_fly_zone(self, safety_analyzer):
        """Test no-fly zone assessment."""
        # Test position near airport (should be restricted)
        airport_position = DronePosition(
            latitude=37.6213,  # Near SFO
            longitude=-122.3790,
            altitude_msl=100.0,
            altitude_rel=50.0
        )
        
        assessment = safety_analyzer.assess_no_fly_zones(airport_position)
        
        assert not assessment.is_safe
        assert "airport" in assessment.warnings[0].lower()
    
    def test_assess_altitude_restrictions(self, safety_analyzer):
        """Test altitude restriction assessment."""
        high_altitude_position = DronePosition(
            latitude=37.7749,
            longitude=-122.4194,
            altitude_msl=500.0,  # Above typical limit
            altitude_rel=450.0
        )
        
        assessment = safety_analyzer.assess_altitude_restrictions(high_altitude_position)
        
        assert not assessment.is_safe
        assert "altitude" in assessment.warnings[0].lower()


class TestFlightPlanner:
    """Test flight planning functionality."""
    
    @pytest.fixture
    def flight_planner(self):
        """Create FlightPlanner instance for testing."""
        return FlightPlanner()
    
    @pytest.fixture
    def waypoints(self):
        """Create test waypoints."""
        return [
            DronePosition(37.7749, -122.4194, 100.0, 50.0),
            DronePosition(37.7849, -122.4094, 120.0, 70.0),
            DronePosition(37.7949, -122.3994, 110.0, 60.0)
        ]
    
    @pytest.mark.asyncio
    async def test_create_flight_plan(self, flight_planner, waypoints):
        """Test flight plan creation."""
        flight_plan = await flight_planner.create_flight_plan(
            waypoints=waypoints,
            max_altitude=150.0,
            max_speed=20.0
        )
        
        assert isinstance(flight_plan, FlightPlan)
        assert len(flight_plan.segments) == len(waypoints) - 1
        assert flight_plan.total_distance > 0
        assert flight_plan.estimated_duration > 0
    
    @pytest.mark.asyncio
    async def test_optimize_route(self, flight_planner, waypoints):
        """Test route optimization."""
        original_plan = await flight_planner.create_flight_plan(
            waypoints=waypoints,
            max_altitude=150.0,
            max_speed=20.0
        )
        
        optimized_plan = await flight_planner.optimize_route(original_plan)
        
        assert isinstance(optimized_plan, FlightPlan)
        # Optimized plan should have same or better metrics
        assert optimized_plan.total_distance <= original_plan.total_distance * 1.1
    
    @pytest.mark.asyncio
    async def test_validate_flight_plan_safe(self, flight_planner, waypoints):
        """Test flight plan validation with safe conditions."""
        flight_plan = await flight_planner.create_flight_plan(
            waypoints=waypoints,
            max_altitude=150.0,
            max_speed=20.0
        )
        
        # Mock weather service to return safe conditions
        with patch.object(flight_planner, '_get_weather_forecast') as mock_weather:
            mock_weather.return_value = WeatherConditions(
                wind_speed=8.0, wind_direction=180, temperature=20.0,
                humidity=60, visibility=10000, precipitation=0.0, cloud_ceiling=2000
            )
            
            validation_result = await flight_planner.validate_flight_plan(flight_plan)
            
            assert validation_result.is_safe
            assert validation_result.risk_level == "LOW"


class TestMAVLinkConnector:
    """Test MAVLink communication functionality."""
    
    @pytest.fixture
    def mavlink_connector(self):
        """Create MAVLinkConnector instance for testing."""
        return MAVLinkConnector()
    
    @pytest.mark.asyncio
    async def test_connect_disconnect(self, mavlink_connector):
        """Test connection and disconnection."""
        # Mock the connection
        with patch.object(mavlink_connector, '_establish_connection') as mock_connect:
            mock_connect.return_value = True
            
            # Test connection
            success = await mavlink_connector.connect("tcp:127.0.0.1:5760")
            assert success
            assert mavlink_connector.is_connected
            
            # Test disconnection
            await mavlink_connector.disconnect()
            assert not mavlink_connector.is_connected
    
    @pytest.mark.asyncio
    async def test_send_command(self, mavlink_connector):
        """Test sending commands to drone."""
        # Mock connection
        mavlink_connector._connected = True
        mavlink_connector._connection = Mock()
        
        with patch.object(mavlink_connector, '_send_mavlink_command') as mock_send:
            mock_send.return_value = True
            
            success = await mavlink_connector.send_command("ARM", {})
            assert success
            mock_send.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_telemetry(self, mavlink_connector):
        """Test telemetry data retrieval."""
        # Mock connection and telemetry data
        mavlink_connector._connected = True
        
        mock_telemetry = TelemetryData(
            position=DronePosition(37.7749, -122.4194, 100.0, 50.0),
            velocity=(5.0, 3.0, -1.0),
            attitude=(0.1, 0.05, 1.57),
            battery_voltage=12.6,
            battery_current=5.5,
            battery_remaining=85,
            gps_satellites=12,
            flight_mode="GUIDED"
        )
        
        with patch.object(mavlink_connector, '_parse_telemetry_data') as mock_parse:
            mock_parse.return_value = mock_telemetry
            
            telemetry = await mavlink_connector.get_telemetry()
            
            assert isinstance(telemetry, TelemetryData)
            assert telemetry.position.latitude == 37.7749
            assert telemetry.battery_remaining == 85
    
    @pytest.mark.asyncio
    async def test_execute_flight_plan(self, mavlink_connector):
        """Test flight plan execution."""
        # Create test flight plan
        waypoints = [
            DronePosition(37.7749, -122.4194, 100.0, 50.0),
            DronePosition(37.7849, -122.4094, 120.0, 70.0)
        ]
        
        segments = [
            FlightSegment(
                start_position=waypoints[0],
                end_position=waypoints[1],
                altitude=75.0,
                speed=15.0,
                estimated_duration=120
            )
        ]
        
        flight_plan = FlightPlan(
            drone_id="TEST_DRONE",
            segments=segments,
            total_distance=1000.0,
            estimated_duration=300,
            max_altitude=150.0,
            created_at=datetime.now(timezone.utc)
        )
        
        # Mock connection and execution
        mavlink_connector._connected = True
        
        with patch.object(mavlink_connector, '_upload_mission') as mock_upload, \
             patch.object(mavlink_connector, '_start_mission') as mock_start:
            
            mock_upload.return_value = True
            mock_start.return_value = True
            
            success = await mavlink_connector.execute_flight_plan(flight_plan)
            assert success
            
            mock_upload.assert_called_once()
            mock_start.assert_called_once()


@pytest.mark.integration
class TestDroneIntegration:
    """Integration tests for drone system."""
    
    @pytest.mark.asyncio
    async def test_full_flight_workflow(self):
        """Test complete flight workflow from planning to execution."""
        # Initialize components
        safety_analyzer = SafetyAnalyzer()
        flight_planner = FlightPlanner()
        mavlink_connector = MAVLinkConnector()
        
        # Define mission waypoints
        waypoints = [
            DronePosition(37.7749, -122.4194, 100.0, 50.0),
            DronePosition(37.7849, -122.4094, 120.0, 70.0),
            DronePosition(37.7949, -122.3994, 110.0, 60.0)
        ]
        
        # 1. Create flight plan
        flight_plan = await flight_planner.create_flight_plan(
            waypoints=waypoints,
            max_altitude=150.0,
            max_speed=20.0
        )
        
        assert isinstance(flight_plan, FlightPlan)
        
        # 2. Validate safety
        with patch.object(flight_planner, '_get_weather_forecast') as mock_weather:
            mock_weather.return_value = WeatherConditions(
                wind_speed=8.0, wind_direction=180, temperature=20.0,
                humidity=60, visibility=10000, precipitation=0.0, cloud_ceiling=2000
            )
            
            safety_assessment = await flight_planner.validate_flight_plan(flight_plan)
            assert safety_assessment.is_safe
        
        # 3. Execute flight plan (mocked)
        with patch.object(mavlink_connector, 'connect') as mock_connect, \
             patch.object(mavlink_connector, 'execute_flight_plan') as mock_execute:
            
            mock_connect.return_value = True
            mock_execute.return_value = True
            
            # Connect to drone
            connection_success = await mavlink_connector.connect("tcp:127.0.0.1:5760")
            assert connection_success
            
            # Execute mission
            execution_success = await mavlink_connector.execute_flight_plan(flight_plan)
            assert execution_success


@pytest.mark.performance
class TestDronePerformance:
    """Performance tests for drone operations."""
    
    @pytest.mark.asyncio
    async def test_flight_planning_performance(self):
        """Test flight planning performance with large number of waypoints."""
        flight_planner = FlightPlanner()
        
        # Generate large number of waypoints
        waypoints = []
        for i in range(100):
            lat = 37.7749 + (i * 0.001)
            lon = -122.4194 + (i * 0.001)
            waypoints.append(DronePosition(lat, lon, 100.0, 50.0))
        
        import time
        start_time = time.time()
        
        flight_plan = await flight_planner.create_flight_plan(
            waypoints=waypoints,
            max_altitude=150.0,
            max_speed=20.0
        )
        
        end_time = time.time()
        planning_time = end_time - start_time
        
        # Should complete planning within reasonable time
        assert planning_time < 5.0  # 5 seconds max
        assert isinstance(flight_plan, FlightPlan)
        assert len(flight_plan.segments) == len(waypoints) - 1
    
    @pytest.mark.asyncio
    async def test_safety_analysis_performance(self):
        """Test safety analysis performance with multiple assessments."""
        safety_analyzer = SafetyAnalyzer()
        
        weather_conditions = WeatherConditions(
            wind_speed=15.0, wind_direction=180, temperature=20.0,
            humidity=60, visibility=8000, precipitation=0.5, cloud_ceiling=1200
        )
        
        import time
        start_time = time.time()
        
        # Perform multiple safety assessments
        for _ in range(1000):
            assessment = safety_analyzer.analyze_weather_conditions(weather_conditions)
            assert isinstance(assessment, SafetyAssessment)
        
        end_time = time.time()
        analysis_time = end_time - start_time
        
        # Should complete all assessments quickly
        assert analysis_time < 2.0  # 2 seconds max for 1000 assessments


# Test fixtures and utilities
@pytest.fixture(scope="session")
def drone_test_data():
    """Provide test data for drone operations."""
    return {
        "safe_waypoints": [
            DronePosition(37.7749, -122.4194, 100.0, 50.0),
            DronePosition(37.7849, -122.4094, 120.0, 70.0)
        ],
        "unsafe_waypoints": [
            DronePosition(37.6213, -122.3790, 100.0, 50.0),  # Near SFO
        ],
        "weather_scenarios": {
            "safe": WeatherConditions(
                wind_speed=8.0, wind_direction=180, temperature=20.0,
                humidity=60, visibility=10000, precipitation=0.0, cloud_ceiling=2000
            ),
            "marginal": WeatherConditions(
                wind_speed=15.0, wind_direction=180, temperature=20.0,
                humidity=75, visibility=5000, precipitation=1.0, cloud_ceiling=800
            ),
            "unsafe": WeatherConditions(
                wind_speed=35.0, wind_direction=180, temperature=20.0,
                humidity=90, visibility=1000, precipitation=10.0, cloud_ceiling=200
            )
        }
    }


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 