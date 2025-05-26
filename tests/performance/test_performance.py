"""Performance tests for OpenWeather platform."""

import pytest
import asyncio
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

import numpy as np
import psutil
import memory_profiler

from openweather.core.config import settings
from openweather.services.forecast_service import WeatherForecastService
from openweather.models.llm_interface import LLMInterface
from openweather.drone.flight_planner import FlightPlanner
from openweather.drone.safety_analyzer import SafetyAnalyzer
from openweather.drone.models import DronePosition, WeatherConditions
from openweather.data.weather_data import WeatherDataProvider
from openweather.services.llm_manager import LLMManager


@pytest.mark.performance
class TestForecastServicePerformance:
    """Performance tests for weather forecast service."""
    
    @pytest.fixture
    def forecast_service(self):
        """Create forecast service instance."""
        return WeatherForecastService()
    
    @pytest.mark.asyncio
    async def test_single_forecast_latency(self, forecast_service):
        """Test latency for single forecast request."""
        location = "San Francisco, CA"
        days = 7
        
        # Warm up
        await forecast_service.get_forecast(location, days)
        
        # Measure latency
        start_time = time.time()
        forecast = await forecast_service.get_forecast(location, days)
        end_time = time.time()
        
        latency = end_time - start_time
        
        # Should respond within 500ms for cached data
        assert latency < 0.5, f"Forecast latency too high: {latency:.3f}s"
        assert forecast is not None
        assert len(forecast.daily_forecasts) == days
    
    @pytest.mark.asyncio
    async def test_concurrent_forecast_requests(self, forecast_service):
        """Test performance under concurrent load."""
        locations = [
            "New York, NY", "Los Angeles, CA", "Chicago, IL",
            "Houston, TX", "Phoenix, AZ", "Philadelphia, PA",
            "San Antonio, TX", "San Diego, CA", "Dallas, TX",
            "San Jose, CA"
        ]
        
        concurrent_requests = 50
        
        async def get_forecast_wrapper(location: str) -> Dict[str, Any]:
            start_time = time.time()
            try:
                forecast = await forecast_service.get_forecast(location, 5)
                end_time = time.time()
                return {
                    "location": location,
                    "success": True,
                    "latency": end_time - start_time,
                    "forecast_days": len(forecast.daily_forecasts) if forecast else 0
                }
            except Exception as e:
                end_time = time.time()
                return {
                    "location": location,
                    "success": False,
                    "latency": end_time - start_time,
                    "error": str(e)
                }
        
        # Create concurrent tasks
        tasks = []
        for i in range(concurrent_requests):
            location = locations[i % len(locations)]
            tasks.append(get_forecast_wrapper(location))
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        total_time = end_time - start_time
        successful_requests = sum(1 for r in results if r["success"])
        failed_requests = concurrent_requests - successful_requests
        
        latencies = [r["latency"] for r in results if r["success"]]
        avg_latency = statistics.mean(latencies) if latencies else 0
        p95_latency = np.percentile(latencies, 95) if latencies else 0
        
        # Performance assertions
        assert successful_requests >= concurrent_requests * 0.95, \
            f"Too many failed requests: {failed_requests}/{concurrent_requests}"
        assert avg_latency < 2.0, f"Average latency too high: {avg_latency:.3f}s"
        assert p95_latency < 5.0, f"P95 latency too high: {p95_latency:.3f}s"
        
        # Throughput calculation
        throughput = successful_requests / total_time
        assert throughput > 10, f"Throughput too low: {throughput:.2f} req/s"
        
        print(f"\nConcurrent Forecast Performance:")
        print(f"  Total requests: {concurrent_requests}")
        print(f"  Successful: {successful_requests}")
        print(f"  Failed: {failed_requests}")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Throughput: {throughput:.2f} req/s")
        print(f"  Average latency: {avg_latency:.3f}s")
        print(f"  P95 latency: {p95_latency:.3f}s")
    
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, forecast_service):
        """Test memory usage during sustained load."""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Generate sustained load
        locations = ["San Francisco, CA", "New York, NY", "London, UK"]
        
        for round_num in range(10):
            tasks = []
            for location in locations:
                for _ in range(5):
                    tasks.append(forecast_service.get_forecast(location, 7))
            
            await asyncio.gather(*tasks)
            
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_growth = current_memory - initial_memory
            
            # Memory should not grow excessively
            assert memory_growth < 100, \
                f"Memory growth too high after round {round_num}: {memory_growth:.2f}MB"
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        total_growth = final_memory - initial_memory
        
        print(f"\nMemory Usage Test:")
        print(f"  Initial memory: {initial_memory:.2f}MB")
        print(f"  Final memory: {final_memory:.2f}MB")
        print(f"  Total growth: {total_growth:.2f}MB")
        
        # Total memory growth should be reasonable
        assert total_growth < 50, f"Total memory growth too high: {total_growth:.2f}MB"


@pytest.mark.performance
class TestLLMPerformance:
    """Performance tests for LLM services."""
    
    @pytest.fixture
    def llm_manager(self):
        """Create LLM manager instance."""
        return LLMManager()
    
    @pytest.mark.asyncio
    async def test_llm_response_time(self, llm_manager):
        """Test LLM response time performance."""
        query = "What's the weather forecast for tomorrow in San Francisco?"
        
        # Warm up
        await llm_manager.process_query(query)
        
        # Measure response time
        start_time = time.time()
        response = await llm_manager.process_query(query)
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Response should be reasonably fast
        assert response_time < 10.0, f"LLM response too slow: {response_time:.3f}s"
        assert response is not None
        assert len(response.strip()) > 0
        
        print(f"\nLLM Response Time: {response_time:.3f}s")
    
    @pytest.mark.asyncio
    async def test_llm_concurrent_requests(self, llm_manager):
        """Test LLM performance under concurrent load."""
        queries = [
            "What's the weather like today?",
            "Will it rain tomorrow?",
            "Is it safe to fly a drone now?",
            "What's the temperature forecast?",
            "Should I bring an umbrella?",
        ]
        
        concurrent_requests = 10
        
        async def process_query_wrapper(query: str) -> Dict[str, Any]:
            start_time = time.time()
            try:
                response = await llm_manager.process_query(query)
                end_time = time.time()
                return {
                    "query": query,
                    "success": True,
                    "response_time": end_time - start_time,
                    "response_length": len(response) if response else 0
                }
            except Exception as e:
                end_time = time.time()
                return {
                    "query": query,
                    "success": False,
                    "response_time": end_time - start_time,
                    "error": str(e)
                }
        
        # Create concurrent tasks
        tasks = []
        for i in range(concurrent_requests):
            query = queries[i % len(queries)]
            tasks.append(process_query_wrapper(query))
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        total_time = end_time - start_time
        successful_requests = sum(1 for r in results if r["success"])
        
        response_times = [r["response_time"] for r in results if r["success"]]
        avg_response_time = statistics.mean(response_times) if response_times else 0
        
        # Performance assertions
        assert successful_requests >= concurrent_requests * 0.8, \
            f"Too many failed LLM requests: {concurrent_requests - successful_requests}/{concurrent_requests}"
        assert avg_response_time < 15.0, f"Average LLM response time too high: {avg_response_time:.3f}s"
        
        print(f"\nConcurrent LLM Performance:")
        print(f"  Total requests: {concurrent_requests}")
        print(f"  Successful: {successful_requests}")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Average response time: {avg_response_time:.3f}s")


@pytest.mark.performance
class TestDronePerformance:
    """Performance tests for drone operations."""
    
    @pytest.fixture
    def flight_planner(self):
        """Create flight planner instance."""
        return FlightPlanner()
    
    @pytest.fixture
    def safety_analyzer(self):
        """Create safety analyzer instance."""
        return SafetyAnalyzer()
    
    @pytest.mark.asyncio
    async def test_flight_planning_scalability(self, flight_planner):
        """Test flight planning performance with increasing waypoint counts."""
        waypoint_counts = [10, 50, 100, 200, 500]
        planning_times = []
        
        for count in waypoint_counts:
            # Generate waypoints
            waypoints = []
            for i in range(count):
                lat = 37.7749 + (i * 0.001)
                lon = -122.4194 + (i * 0.001)
                waypoints.append(DronePosition(lat, lon, 100.0, 50.0))
            
            # Measure planning time
            start_time = time.time()
            flight_plan = await flight_planner.create_flight_plan(
                waypoints=waypoints,
                max_altitude=150.0,
                max_speed=20.0
            )
            end_time = time.time()
            
            planning_time = end_time - start_time
            planning_times.append(planning_time)
            
            # Verify plan was created successfully
            assert flight_plan is not None
            assert len(flight_plan.segments) == count - 1
            
            print(f"  {count} waypoints: {planning_time:.3f}s")
        
        # Check that planning time scales reasonably
        for i, time_taken in enumerate(planning_times):
            waypoint_count = waypoint_counts[i]
            
            # Should be roughly linear or sub-quadratic
            max_allowed_time = waypoint_count * 0.01  # 10ms per waypoint
            assert time_taken < max_allowed_time, \
                f"Planning time too high for {waypoint_count} waypoints: {time_taken:.3f}s"
        
        print(f"\nFlight Planning Scalability Test Completed")
    
    @pytest.mark.asyncio
    async def test_safety_analysis_throughput(self, safety_analyzer):
        """Test safety analysis throughput."""
        weather_conditions = WeatherConditions(
            wind_speed=15.0, wind_direction=180, temperature=20.0,
            humidity=60, visibility=8000, precipitation=0.5, cloud_ceiling=1200
        )
        
        positions = [
            DronePosition(37.7749 + i * 0.001, -122.4194 + i * 0.001, 100.0, 50.0)
            for i in range(1000)
        ]
        
        # Test weather analysis throughput
        start_time = time.time()
        for _ in range(1000):
            assessment = safety_analyzer.analyze_weather_conditions(weather_conditions)
            assert assessment is not None
        end_time = time.time()
        
        weather_analysis_time = end_time - start_time
        weather_throughput = 1000 / weather_analysis_time
        
        # Test position analysis throughput
        start_time = time.time()
        for position in positions:
            assessment = safety_analyzer.assess_altitude_restrictions(position)
            assert assessment is not None
        end_time = time.time()
        
        position_analysis_time = end_time - start_time
        position_throughput = 1000 / position_analysis_time
        
        # Performance assertions
        assert weather_throughput > 500, f"Weather analysis throughput too low: {weather_throughput:.2f}/s"
        assert position_throughput > 200, f"Position analysis throughput too low: {position_throughput:.2f}/s"
        
        print(f"\nSafety Analysis Performance:")
        print(f"  Weather analysis: {weather_throughput:.2f} assessments/s")
        print(f"  Position analysis: {position_throughput:.2f} assessments/s")


@pytest.mark.performance
class TestDataProviderPerformance:
    """Performance tests for data providers."""
    
    @pytest.fixture
    def weather_data_provider(self):
        """Create weather data provider instance."""
        return WeatherDataProvider()
    
    @pytest.mark.asyncio
    async def test_data_retrieval_performance(self, weather_data_provider):
        """Test weather data retrieval performance."""
        locations = [
            (37.7749, -122.4194),  # San Francisco
            (40.7128, -74.0060),   # New York
            (34.0522, -118.2437),  # Los Angeles
            (41.8781, -87.6298),   # Chicago
            (29.7604, -95.3698),   # Houston
        ]
        
        # Test sequential retrieval
        start_time = time.time()
        for lat, lon in locations:
            data = await weather_data_provider.get_current_weather(lat, lon)
            assert data is not None
        end_time = time.time()
        
        sequential_time = end_time - start_time
        
        # Test concurrent retrieval
        async def get_weather_wrapper(lat: float, lon: float):
            return await weather_data_provider.get_current_weather(lat, lon)
        
        start_time = time.time()
        tasks = [get_weather_wrapper(lat, lon) for lat, lon in locations]
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        concurrent_time = end_time - start_time
        
        # Verify all requests succeeded
        assert all(result is not None for result in results)
        
        # Concurrent should be faster than sequential
        speedup = sequential_time / concurrent_time
        assert speedup > 1.5, f"Insufficient speedup from concurrency: {speedup:.2f}x"
        
        print(f"\nData Retrieval Performance:")
        print(f"  Sequential time: {sequential_time:.3f}s")
        print(f"  Concurrent time: {concurrent_time:.3f}s")
        print(f"  Speedup: {speedup:.2f}x")
    
    @pytest.mark.asyncio
    async def test_cache_performance(self, weather_data_provider):
        """Test cache hit performance."""
        lat, lon = 37.7749, -122.4194
        
        # First request (cache miss)
        start_time = time.time()
        data1 = await weather_data_provider.get_current_weather(lat, lon)
        first_request_time = time.time() - start_time
        
        # Second request (cache hit)
        start_time = time.time()
        data2 = await weather_data_provider.get_current_weather(lat, lon)
        second_request_time = time.time() - start_time
        
        # Verify data consistency
        assert data1 is not None
        assert data2 is not None
        
        # Cache hit should be significantly faster
        if hasattr(weather_data_provider, '_cache'):
            speedup = first_request_time / second_request_time
            assert speedup > 5, f"Cache not providing sufficient speedup: {speedup:.2f}x"
            
            print(f"\nCache Performance:")
            print(f"  Cache miss: {first_request_time:.3f}s")
            print(f"  Cache hit: {second_request_time:.3f}s")
            print(f"  Speedup: {speedup:.2f}x")


@pytest.mark.performance
class TestSystemIntegrationPerformance:
    """End-to-end performance tests."""
    
    @pytest.mark.asyncio
    async def test_full_workflow_performance(self):
        """Test performance of complete weather-to-drone workflow."""
        # Initialize services
        forecast_service = WeatherForecastService()
        flight_planner = FlightPlanner()
        safety_analyzer = SafetyAnalyzer()
        
        # Define test scenario
        location = "San Francisco, CA"
        waypoints = [
            DronePosition(37.7749, -122.4194, 100.0, 50.0),
            DronePosition(37.7849, -122.4094, 120.0, 70.0),
            DronePosition(37.7949, -122.3994, 110.0, 60.0),
        ]
        
        start_time = time.time()
        
        # 1. Get weather forecast
        forecast = await forecast_service.get_forecast(location, 3)
        forecast_time = time.time()
        
        # 2. Create flight plan
        flight_plan = await flight_planner.create_flight_plan(
            waypoints=waypoints,
            max_altitude=150.0,
            max_speed=20.0
        )
        planning_time = time.time()
        
        # 3. Analyze safety
        weather_conditions = WeatherConditions(
            wind_speed=10.0, wind_direction=180, temperature=20.0,
            humidity=60, visibility=10000, precipitation=0.0, cloud_ceiling=2000
        )
        safety_assessment = safety_analyzer.analyze_weather_conditions(weather_conditions)
        safety_time = time.time()
        
        # Calculate timings
        total_time = safety_time - start_time
        forecast_duration = forecast_time - start_time
        planning_duration = planning_time - forecast_time
        safety_duration = safety_time - planning_time
        
        # Verify results
        assert forecast is not None
        assert flight_plan is not None
        assert safety_assessment is not None
        
        # Performance assertions
        assert total_time < 5.0, f"Total workflow time too high: {total_time:.3f}s"
        assert forecast_duration < 2.0, f"Forecast time too high: {forecast_duration:.3f}s"
        assert planning_duration < 2.0, f"Planning time too high: {planning_duration:.3f}s"
        assert safety_duration < 1.0, f"Safety analysis time too high: {safety_duration:.3f}s"
        
        print(f"\nFull Workflow Performance:")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Forecast: {forecast_duration:.3f}s")
        print(f"  Planning: {planning_duration:.3f}s")
        print(f"  Safety: {safety_duration:.3f}s")


# Benchmark utilities
def run_performance_benchmark():
    """Run all performance tests and generate report."""
    pytest.main([
        "tests/performance/",
        "-v",
        "--benchmark-only",
        "--benchmark-json=performance_report.json",
        "--benchmark-histogram=performance_histogram",
        "-m", "performance"
    ])


if __name__ == "__main__":
    run_performance_benchmark() 