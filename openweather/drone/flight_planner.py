"""
Flight planner for drone operations with weather optimization.

Creates optimal flight plans considering weather conditions, safety margins,
regulatory requirements, and mission objectives.
"""

import logging
import math
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from uuid import uuid4

import geopy.distance as distance
from geopy import Point

from openweather.core.models_shared import Location
from .models import (
    FlightPlan,
    Waypoint,
    DroneType,
    FlightRestrictions,
    WeatherConditions,
    SafetyReport,
)
from .safety_analyzer import SafetyAnalyzer

logger = logging.getLogger(__name__)


class FlightPlanner:
    """
    Advanced flight planner for drone operations.
    
    Features:
    - Weather-optimized route planning
    - Safety corridor calculation
    - Multi-objective optimization
    - Regulatory compliance checking
    - Real-time replanning capabilities
    """
    
    def __init__(self, safety_analyzer: Optional[SafetyAnalyzer] = None):
        """Initialize the flight planner."""
        self.safety_analyzer = safety_analyzer or SafetyAnalyzer()
        
        # Planning constraints
        self.max_waypoints = 100
        self.min_waypoint_separation = 10.0  # meters
        self.max_leg_distance = 2000.0  # meters
        
        # Performance parameters by drone type
        self.drone_performance = {
            DroneType.MULTIROTOR: {
                'max_speed': 15.0,  # m/s
                'cruise_speed': 8.0,
                'climb_rate': 3.0,
                'energy_consumption': 1.0,  # relative
                'wind_tolerance': 0.6  # fraction of max speed
            },
            DroneType.FIXED_WING: {
                'max_speed': 25.0,
                'cruise_speed': 15.0,
                'climb_rate': 5.0,
                'energy_consumption': 0.7,
                'wind_tolerance': 0.8
            },
            DroneType.VTOL: {
                'max_speed': 20.0,
                'cruise_speed': 12.0,
                'climb_rate': 4.0,
                'energy_consumption': 0.8,
                'wind_tolerance': 0.7
            },
            DroneType.HELICOPTER: {
                'max_speed': 18.0,
                'cruise_speed': 10.0,
                'climb_rate': 4.0,
                'energy_consumption': 1.2,
                'wind_tolerance': 0.5
            }
        }
    
    async def create_flight_plan(
        self,
        mission_id: str,
        drone_id: str,
        drone_type: DroneType,
        takeoff_location: Tuple[float, float, float],  # lat, lon, alt
        target_locations: List[Tuple[float, float, float]],
        landing_location: Optional[Tuple[float, float, float]] = None,
        mission_objectives: Optional[Dict[str, any]] = None,
        restrictions: Optional[FlightRestrictions] = None,
        optimize_for: str = "safety"  # "safety", "time", "energy", "weather"
    ) -> FlightPlan:
        """
        Create an optimized flight plan.
        
        Args:
            mission_id: Unique mission identifier
            drone_id: Drone identifier
            drone_type: Type of drone
            takeoff_location: Takeoff coordinates (lat, lon, alt)
            target_locations: List of target coordinates
            landing_location: Landing coordinates (defaults to takeoff)
            mission_objectives: Mission-specific parameters
            restrictions: Flight restrictions
            optimize_for: Optimization objective
            
        Returns:
            Optimized flight plan
        """
        try:
            logger.info(f"Creating flight plan for mission {mission_id}")
            
            # Set default landing location
            if landing_location is None:
                landing_location = takeoff_location
            
            # Get drone performance parameters
            performance = self.drone_performance.get(drone_type, 
                                                   self.drone_performance[DroneType.MULTIROTOR])
            
            # Create waypoints
            takeoff_wp = Waypoint(
                latitude=takeoff_location[0],
                longitude=takeoff_location[1], 
                altitude=takeoff_location[2],
                speed=performance['cruise_speed'],
                action="takeoff"
            )
            
            landing_wp = Waypoint(
                latitude=landing_location[0],
                longitude=landing_location[1],
                altitude=landing_location[2],
                speed=performance['cruise_speed'] * 0.5,
                action="land"
            )
            
            # Optimize waypoint sequence
            optimized_targets = await self._optimize_waypoint_sequence(
                takeoff_location, target_locations, landing_location,
                performance, optimize_for
            )
            
            # Create mission waypoints
            mission_waypoints = []
            for i, target in enumerate(optimized_targets):
                wp = Waypoint(
                    latitude=target[0],
                    longitude=target[1],
                    altitude=target[2],
                    speed=performance['cruise_speed'],
                    action=mission_objectives.get('waypoint_actions', {}).get(i, "survey")
                    if mission_objectives else "survey"
                )
                mission_waypoints.append(wp)
            
            # Add intermediate waypoints for long legs
            mission_waypoints = self._add_intermediate_waypoints(
                mission_waypoints, performance
            )
            
            # Calculate timing
            estimated_duration = self._calculate_flight_duration(
                takeoff_wp, mission_waypoints, landing_wp, performance
            )
            
            # Get weather requirements
            weather_conditions = await self._get_weather_requirements(
                takeoff_location, restrictions
            )
            
            # Create flight plan
            flight_plan = FlightPlan(
                mission_id=mission_id,
                drone_id=drone_id,
                takeoff_location=takeoff_wp,
                waypoints=mission_waypoints,
                landing_location=landing_wp,
                planned_start=datetime.utcnow() + timedelta(minutes=5),
                estimated_duration=estimated_duration,
                drone_type=drone_type,
                max_speed=performance['max_speed'],
                max_altitude=max(wp.altitude for wp in [takeoff_wp] + mission_waypoints + [landing_wp]),
                battery_capacity=mission_objectives.get('battery_capacity', 5000.0)
                if mission_objectives else 5000.0,
                payload_weight=mission_objectives.get('payload_weight', 0.0)
                if mission_objectives else 0.0,
                restrictions=restrictions or FlightRestrictions(),
                emergency_procedures=self._create_emergency_procedures(drone_type),
                required_weather_conditions=weather_conditions
            )
            
            # Validate flight plan
            await self._validate_flight_plan(flight_plan)
            
            logger.info(f"Flight plan created: {len(mission_waypoints)} waypoints, "
                       f"{estimated_duration.total_seconds()/60:.1f} minutes")
            
            return flight_plan
            
        except Exception as e:
            logger.error(f"Error creating flight plan: {str(e)}")
            raise
    
    async def analyze_flight_safety(
        self,
        waypoints: List[Tuple[float, float]],  # lat, lon pairs
        altitude: float,
        duration: int = 30
    ) -> SafetyReport:
        """
        Analyze flight safety for a set of waypoints.
        
        Args:
            waypoints: List of (latitude, longitude) pairs
            altitude: Flight altitude in meters
            duration: Flight duration in minutes
            
        Returns:
            Safety assessment report
        """
        if not waypoints:
            raise ValueError("At least one waypoint is required")
        
        # Use the center point for weather analysis
        center_lat = sum(wp[0] for wp in waypoints) / len(waypoints)
        center_lon = sum(wp[1] for wp in waypoints) / len(waypoints)
        
        location = Location(
            latitude=center_lat,
            longitude=center_lon,
            name=f"Flight Area ({center_lat:.4f}, {center_lon:.4f})"
        )
        
        return await self.safety_analyzer.analyze_flight_safety(
            location=location,
            altitude=altitude,
            duration_minutes=duration
        )
    
    async def replan_for_weather(
        self,
        original_plan: FlightPlan,
        current_weather: WeatherConditions,
        forecast: List[WeatherConditions]
    ) -> FlightPlan:
        """
        Replan flight considering updated weather conditions.
        
        Args:
            original_plan: Original flight plan
            current_weather: Current weather conditions
            forecast: Weather forecast
            
        Returns:
            Updated flight plan
        """
        try:
            logger.info(f"Replanning flight {original_plan.mission_id} for weather")
            
            # Assess if replanning is needed
            risk_assessment = await self._assess_weather_risk(
                original_plan, current_weather, forecast
            )
            
            if risk_assessment['risk_level'] <= 2:  # Low to moderate risk
                logger.info("Weather conditions acceptable, keeping original plan")
                return original_plan
            
            # Create modified plan
            modifications = risk_assessment['required_modifications']
            
            # Apply altitude modifications
            if 'reduce_altitude' in modifications:
                new_altitude = min(wp.altitude * 0.8 for wp in 
                                 [original_plan.takeoff_location] + 
                                 original_plan.waypoints + 
                                 [original_plan.landing_location])
                original_plan = self._modify_flight_altitude(original_plan, new_altitude)
            
            # Apply speed modifications
            if 'reduce_speed' in modifications:
                original_plan = self._modify_flight_speed(original_plan, 0.8)
            
            # Apply route modifications
            if 'simplify_route' in modifications:
                original_plan = await self._simplify_flight_route(original_plan)
            
            # Update timing
            performance = self.drone_performance.get(
                original_plan.drone_type,
                self.drone_performance[DroneType.MULTIROTOR]
            )
            
            original_plan.estimated_duration = self._calculate_flight_duration(
                original_plan.takeoff_location,
                original_plan.waypoints,
                original_plan.landing_location,
                performance
            )
            
            logger.info(f"Flight replanned with {len(modifications)} modifications")
            return original_plan
            
        except Exception as e:
            logger.error(f"Error replanning flight: {str(e)}")
            raise
    
    def calculate_flight_corridor(
        self,
        start_point: Tuple[float, float],
        end_point: Tuple[float, float],
        corridor_width: float = 100.0,
        safety_margin: float = 50.0
    ) -> List[Tuple[float, float]]:
        """
        Calculate a safe flight corridor between two points.
        
        Args:
            start_point: Starting coordinates (lat, lon)
            end_point: Ending coordinates (lat, lon)
            corridor_width: Width of corridor in meters
            safety_margin: Additional safety margin in meters
            
        Returns:
            List of corridor boundary points
        """
        try:
            start = Point(start_point[0], start_point[1])
            end = Point(end_point[0], end_point[1])
            
            # Calculate bearing and distance
            bearing = self._calculate_bearing(start_point, end_point)
            total_distance = distance.distance(start, end).meters
            
            # Calculate perpendicular bearings
            left_bearing = (bearing - 90) % 360
            right_bearing = (bearing + 90) % 360
            
            # Calculate corridor width including safety margin
            half_width = (corridor_width + safety_margin) / 2
            
            # Generate corridor points
            corridor_points = []
            num_segments = max(4, int(total_distance / 200))  # Point every 200m
            
            for i in range(num_segments + 1):
                # Calculate intermediate point along route
                segment_distance = (total_distance * i) / num_segments
                intermediate = distance.distance(meters=segment_distance).destination(start, bearing)
                
                # Calculate left and right boundary points
                left_point = distance.distance(meters=half_width).destination(
                    intermediate, left_bearing
                )
                right_point = distance.distance(meters=half_width).destination(
                    intermediate, right_bearing
                )
                
                corridor_points.extend([
                    (left_point.latitude, left_point.longitude),
                    (right_point.latitude, right_point.longitude)
                ])
            
            return corridor_points
            
        except Exception as e:
            logger.error(f"Error calculating flight corridor: {str(e)}")
            return []
    
    async def _optimize_waypoint_sequence(
        self,
        start: Tuple[float, float, float],
        targets: List[Tuple[float, float, float]], 
        end: Tuple[float, float, float],
        performance: Dict[str, float],
        optimize_for: str
    ) -> List[Tuple[float, float, float]]:
        """Optimize the sequence of waypoints using various algorithms."""
        if len(targets) <= 2:
            return targets
        
        if optimize_for == "time":
            return self._optimize_for_shortest_path(start, targets, end)
        elif optimize_for == "energy":
            return self._optimize_for_energy_efficiency(start, targets, end, performance)
        elif optimize_for == "weather":
            return await self._optimize_for_weather_conditions(start, targets, end)
        else:  # Default to safety
            return self._optimize_for_safety(start, targets, end)
    
    def _optimize_for_shortest_path(
        self,
        start: Tuple[float, float, float],
        targets: List[Tuple[float, float, float]],
        end: Tuple[float, float, float]
    ) -> List[Tuple[float, float, float]]:
        """Optimize waypoint sequence for shortest total distance."""
        if len(targets) <= 1:
            return targets
        
        # Simple nearest neighbor algorithm for TSP approximation
        unvisited = targets.copy()
        current = start
        optimized = []
        
        while unvisited:
            nearest_idx = 0
            nearest_dist = self._calculate_distance_3d(current, unvisited[0])
            
            for i, target in enumerate(unvisited[1:], 1):
                dist = self._calculate_distance_3d(current, target)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_idx = i
            
            nearest = unvisited.pop(nearest_idx)
            optimized.append(nearest)
            current = nearest
        
        return optimized
    
    def _optimize_for_energy_efficiency(
        self,
        start: Tuple[float, float, float],
        targets: List[Tuple[float, float, float]],
        end: Tuple[float, float, float],
        performance: Dict[str, float]
    ) -> List[Tuple[float, float, float]]:
        """Optimize for energy efficiency considering altitude changes."""
        if len(targets) <= 1:
            return targets
        
        # Consider altitude changes in energy calculation
        unvisited = targets.copy()
        current = start
        optimized = []
        
        while unvisited:
            best_idx = 0
            best_energy = float('inf')
            
            for i, target in enumerate(unvisited):
                horizontal_dist = self._calculate_distance_2d(current[:2], target[:2])
                altitude_change = abs(target[2] - current[2])
                
                # Energy cost includes horizontal travel and altitude change
                energy_cost = (
                    horizontal_dist * performance['energy_consumption'] +
                    altitude_change * performance['climb_rate'] * 2.0
                )
                
                if energy_cost < best_energy:
                    best_energy = energy_cost
                    best_idx = i
            
            best_target = unvisited.pop(best_idx)
            optimized.append(best_target)
            current = best_target
        
        return optimized
    
    async def _optimize_for_weather_conditions(
        self,
        start: Tuple[float, float, float],
        targets: List[Tuple[float, float, float]],
        end: Tuple[float, float, float]
    ) -> List[Tuple[float, float, float]]:
        """Optimize waypoint sequence considering weather conditions."""
        # For now, use distance optimization
        # In future versions, this could consider wind patterns, precipitation, etc.
        return self._optimize_for_shortest_path(start, targets, end)
    
    def _optimize_for_safety(
        self,
        start: Tuple[float, float, float],
        targets: List[Tuple[float, float, float]],
        end: Tuple[float, float, float]
    ) -> List[Tuple[float, float, float]]:
        """Optimize for safety by avoiding long legs and high altitudes."""
        if len(targets) <= 1:
            return targets
        
        # Prefer routes with shorter legs and lower altitudes
        unvisited = targets.copy()
        current = start
        optimized = []
        
        while unvisited:
            best_idx = 0
            best_score = float('inf')
            
            for i, target in enumerate(unvisited):
                distance = self._calculate_distance_3d(current, target)
                altitude_penalty = target[2] * 0.01  # Penalty for high altitude
                distance_penalty = max(0, distance - 1000) * 0.1  # Penalty for long legs
                
                safety_score = distance + altitude_penalty + distance_penalty
                
                if safety_score < best_score:
                    best_score = safety_score
                    best_idx = i
            
            best_target = unvisited.pop(best_idx)
            optimized.append(best_target)
            current = best_target
        
        return optimized
    
    def _add_intermediate_waypoints(
        self,
        waypoints: List[Waypoint],
        performance: Dict[str, float]
    ) -> List[Waypoint]:
        """Add intermediate waypoints for long legs."""
        if len(waypoints) <= 1:
            return waypoints
        
        expanded_waypoints = []
        
        for i in range(len(waypoints) - 1):
            current = waypoints[i]
            next_wp = waypoints[i + 1]
            
            expanded_waypoints.append(current)
            
            # Calculate distance between waypoints
            distance = self._calculate_distance_2d(
                (current.latitude, current.longitude),
                (next_wp.latitude, next_wp.longitude)
            )
            
            # Add intermediate waypoints if leg is too long
            if distance > self.max_leg_distance:
                num_intermediates = int(distance / self.max_leg_distance)
                
                for j in range(1, num_intermediates + 1):
                    fraction = j / (num_intermediates + 1)
                    
                    intermediate_lat = current.latitude + (next_wp.latitude - current.latitude) * fraction
                    intermediate_lon = current.longitude + (next_wp.longitude - current.longitude) * fraction
                    intermediate_alt = current.altitude + (next_wp.altitude - current.altitude) * fraction
                    
                    intermediate_wp = Waypoint(
                        latitude=intermediate_lat,
                        longitude=intermediate_lon,
                        altitude=intermediate_alt,
                        speed=performance['cruise_speed'],
                        action="transit"
                    )
                    
                    expanded_waypoints.append(intermediate_wp)
        
        # Add the last waypoint
        if waypoints:
            expanded_waypoints.append(waypoints[-1])
        
        return expanded_waypoints
    
    def _calculate_flight_duration(
        self,
        takeoff: Waypoint,
        waypoints: List[Waypoint],
        landing: Waypoint,
        performance: Dict[str, float]
    ) -> timedelta:
        """Calculate estimated flight duration."""
        total_time = 0.0
        all_waypoints = [takeoff] + waypoints + [landing]
        
        # Add time for each leg
        for i in range(len(all_waypoints) - 1):
            current = all_waypoints[i]
            next_wp = all_waypoints[i + 1]
            
            # Horizontal distance
            horizontal_dist = self._calculate_distance_2d(
                (current.latitude, current.longitude),
                (next_wp.latitude, next_wp.longitude)
            )
            
            # Altitude change
            altitude_change = abs(next_wp.altitude - current.altitude)
            
            # Time for horizontal travel
            speed = min(current.speed or performance['cruise_speed'], 
                       next_wp.speed or performance['cruise_speed'])
            horizontal_time = horizontal_dist / speed
            
            # Time for altitude change
            climb_time = altitude_change / performance['climb_rate']
            
            # Total time for this leg (considering concurrent operations)
            leg_time = max(horizontal_time, climb_time)
            total_time += leg_time
            
            # Add dwell time if specified
            if hasattr(next_wp, 'dwell_time') and next_wp.dwell_time:
                total_time += next_wp.dwell_time
        
        # Add takeoff and landing time
        total_time += 30.0  # 30 seconds for takeoff
        total_time += 60.0  # 60 seconds for landing
        
        return timedelta(seconds=total_time)
    
    async def _get_weather_requirements(
        self,
        location: Tuple[float, float, float],
        restrictions: Optional[FlightRestrictions]
    ) -> WeatherConditions:
        """Get required weather conditions for flight planning."""
        # Create conservative weather requirements
        active_restrictions = restrictions or FlightRestrictions()
        
        return WeatherConditions(
            wind_speed=active_restrictions.max_wind_speed * 0.8,
            wind_direction=0.0,  # Any direction acceptable
            visibility=active_restrictions.min_visibility * 1.2,
            precipitation=active_restrictions.max_precipitation * 0.5,
            temperature=20.0,  # Nominal temperature
            humidity=60.0,  # Nominal humidity
            pressure=1013.25,  # Standard pressure
            turbulence_intensity=0.2  # Low turbulence
        )
    
    def _create_emergency_procedures(self, drone_type: DroneType) -> Dict[str, str]:
        """Create emergency procedures based on drone type."""
        base_procedures = {
            "low_battery": "Return to launch immediately",
            "loss_of_signal": "Hover for 30 seconds, then return to launch",
            "high_wind": "Land at nearest safe location",
            "equipment_failure": "Execute emergency landing procedure"
        }
        
        if drone_type == DroneType.FIXED_WING:
            base_procedures.update({
                "engine_failure": "Attempt glide to landing site",
                "control_failure": "Deploy emergency parachute if available"
            })
        
        return base_procedures
    
    async def _validate_flight_plan(self, flight_plan: FlightPlan) -> None:
        """Validate the flight plan for safety and feasibility."""
        # Check waypoint count
        if len(flight_plan.waypoints) > self.max_waypoints:
            raise ValueError(f"Too many waypoints: {len(flight_plan.waypoints)} > {self.max_waypoints}")
        
        # Check altitude limits
        max_altitude = max(
            wp.altitude for wp in [flight_plan.takeoff_location] + 
            flight_plan.waypoints + [flight_plan.landing_location]
        )
        
        if max_altitude > flight_plan.restrictions.max_altitude:
            raise ValueError(f"Altitude exceeds limit: {max_altitude}m > {flight_plan.restrictions.max_altitude}m")
        
        # Check for minimum waypoint separation
        all_waypoints = [flight_plan.takeoff_location] + flight_plan.waypoints + [flight_plan.landing_location]
        for i in range(len(all_waypoints) - 1):
            current = all_waypoints[i]
            next_wp = all_waypoints[i + 1]
            
            distance = self._calculate_distance_2d(
                (current.latitude, current.longitude),
                (next_wp.latitude, next_wp.longitude)
            )
            
            if distance < self.min_waypoint_separation:
                logger.warning(f"Waypoints {i} and {i+1} are very close: {distance:.1f}m")
    
    async def _assess_weather_risk(
        self,
        flight_plan: FlightPlan,
        current_weather: WeatherConditions,
        forecast: List[WeatherConditions]
    ) -> Dict[str, any]:
        """Assess weather risk for flight plan."""
        risk_level = 0
        modifications = []
        
        # Wind risk assessment
        if current_weather.wind_speed > flight_plan.restrictions.max_wind_speed * 0.8:
            risk_level += 2
            modifications.append('reduce_speed')
        
        # Visibility risk assessment
        if current_weather.visibility < flight_plan.restrictions.min_visibility * 1.5:
            risk_level += 2
            modifications.append('reduce_altitude')
        
        # Precipitation risk assessment
        if current_weather.precipitation > flight_plan.restrictions.max_precipitation * 0.5:
            risk_level += 1
            modifications.append('simplify_route')
        
        # Forecast deterioration
        if forecast:
            future_conditions = forecast[0] if len(forecast) > 0 else current_weather
            if future_conditions.wind_speed > current_weather.wind_speed * 1.2:
                risk_level += 1
                modifications.append('reduce_speed')
        
        return {
            'risk_level': risk_level,
            'required_modifications': list(set(modifications))
        }
    
    def _modify_flight_altitude(self, flight_plan: FlightPlan, max_altitude: float) -> FlightPlan:
        """Modify flight plan to cap altitude."""
        # Update takeoff location
        if flight_plan.takeoff_location.altitude > max_altitude:
            flight_plan.takeoff_location.altitude = max_altitude
        
        # Update waypoints
        for waypoint in flight_plan.waypoints:
            if waypoint.altitude > max_altitude:
                waypoint.altitude = max_altitude
        
        # Update landing location
        if flight_plan.landing_location.altitude > max_altitude:
            flight_plan.landing_location.altitude = max_altitude
        
        return flight_plan
    
    def _modify_flight_speed(self, flight_plan: FlightPlan, speed_factor: float) -> FlightPlan:
        """Modify flight plan to reduce speeds."""
        # Update takeoff location
        if flight_plan.takeoff_location.speed:
            flight_plan.takeoff_location.speed *= speed_factor
        
        # Update waypoints
        for waypoint in flight_plan.waypoints:
            if waypoint.speed:
                waypoint.speed *= speed_factor
        
        # Update landing location
        if flight_plan.landing_location.speed:
            flight_plan.landing_location.speed *= speed_factor
        
        return flight_plan
    
    async def _simplify_flight_route(self, flight_plan: FlightPlan) -> FlightPlan:
        """Simplify flight route by removing non-essential waypoints."""
        if len(flight_plan.waypoints) <= 2:
            return flight_plan
        
        # Keep every other waypoint for simplification
        simplified_waypoints = []
        for i, waypoint in enumerate(flight_plan.waypoints):
            if i % 2 == 0 or waypoint.action in ['survey', 'photo']:
                simplified_waypoints.append(waypoint)
        
        flight_plan.waypoints = simplified_waypoints
        return flight_plan
    
    def _calculate_distance_2d(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate 2D distance between two points."""
        return distance.distance(point1, point2).meters
    
    def _calculate_distance_3d(self, point1: Tuple[float, float, float], point2: Tuple[float, float, float]) -> float:
        """Calculate 3D distance between two points."""
        horizontal_dist = self._calculate_distance_2d(point1[:2], point2[:2])
        vertical_dist = abs(point2[2] - point1[2])
        return math.sqrt(horizontal_dist**2 + vertical_dist**2)
    
    def _calculate_bearing(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate bearing between two points."""
        lat1, lon1 = math.radians(point1[0]), math.radians(point1[1])
        lat2, lon2 = math.radians(point2[0]), math.radians(point2[1])
        
        dlon = lon2 - lon1
        
        y = math.sin(dlon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        
        bearing = math.atan2(y, x)
        bearing = math.degrees(bearing)
        bearing = (bearing + 360) % 360
        
        return bearing 