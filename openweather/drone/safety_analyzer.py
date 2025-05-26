"""
Flight safety analyzer for drone operations.

Analyzes weather conditions, flight parameters, and environmental factors
to determine flight safety and provide recommendations.
"""

import logging
import math
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from uuid import uuid4

from openweather.core.models_shared import Location, WeatherData
from openweather.data.data_loader import WeatherDataOrchestrator
from .models import (
    WeatherConditions,
    FlightRestrictions,
    SafetyReport,
    RiskLevel,
    FlightPlan,
    Waypoint,
)

logger = logging.getLogger(__name__)


class SafetyAnalyzer:
    """
    Analyzes flight safety based on weather conditions and flight parameters.
    
    Provides comprehensive safety assessments including:
    - Wind and turbulence analysis
    - Visibility assessment
    - Precipitation impact
    - Flight corridor optimization
    - Risk mitigation recommendations
    """
    
    def __init__(self, data_orchestrator: Optional[WeatherDataOrchestrator] = None):
        """Initialize the safety analyzer."""
        self.data_orchestrator = data_orchestrator or WeatherDataOrchestrator()
        
        # Default safety thresholds
        self.default_restrictions = FlightRestrictions()
        
        # Risk calculation weights
        self.risk_weights = {
            'wind': 0.3,
            'visibility': 0.25,
            'precipitation': 0.2,
            'turbulence': 0.15,
            'temperature': 0.1
        }
    
    async def analyze_flight_safety(
        self,
        location: Location,
        altitude: float,
        duration_minutes: int = 30,
        restrictions: Optional[FlightRestrictions] = None,
        flight_plan: Optional[FlightPlan] = None
    ) -> SafetyReport:
        """
        Perform comprehensive flight safety analysis.
        
        Args:
            location: Flight location
            altitude: Flight altitude in meters
            duration_minutes: Expected flight duration
            restrictions: Custom flight restrictions
            flight_plan: Optional flight plan for detailed analysis
            
        Returns:
            Detailed safety report
        """
        try:
            # Use provided restrictions or defaults
            active_restrictions = restrictions or self.default_restrictions
            
            # Get current and forecast weather
            current_weather = await self._get_current_weather(location)
            forecast_weather = await self._get_forecast_weather(location, duration_minutes)
            
            # Analyze weather conditions
            current_conditions = self._weather_to_conditions(current_weather)
            forecast_conditions = [
                self._weather_to_conditions(w) for w in forecast_weather
            ]
            
            # Calculate risk assessments
            wind_risk = self._assess_wind_risk(current_conditions, active_restrictions)
            visibility_risk = self._assess_visibility_risk(current_conditions, active_restrictions)
            precipitation_risk = self._assess_precipitation_risk(current_conditions, active_restrictions)
            turbulence_risk = self._assess_turbulence_risk(current_conditions, altitude)
            
            # Calculate overall risk
            overall_risk = self._calculate_overall_risk({
                'wind': wind_risk,
                'visibility': visibility_risk, 
                'precipitation': precipitation_risk,
                'turbulence': turbulence_risk
            })
            
            # Generate warnings and recommendations
            warnings = self._generate_warnings(current_conditions, active_restrictions)
            recommendations = self._generate_recommendations(
                current_conditions, forecast_conditions, active_restrictions
            )
            restrictions_list = self._generate_restrictions(overall_risk, current_conditions)
            
            # Find optimal flight windows
            optimal_window = self._find_optimal_window(forecast_conditions)
            alternative_windows = self._find_alternative_windows(forecast_conditions)
            
            # Flight plan compatibility analysis
            compatible_plans = []
            modifications = []
            if flight_plan:
                compatible_plans, modifications = self._analyze_flight_plan_compatibility(
                    flight_plan, current_conditions, active_restrictions
                )
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                current_weather, forecast_weather, overall_risk
            )
            
            # Create safety report
            report = SafetyReport(
                assessment_id=str(uuid4()),
                location=location,
                assessment_time=datetime.utcnow(),
                valid_until=datetime.utcnow() + timedelta(hours=2),
                current_weather=current_conditions,
                forecast_weather=forecast_conditions,
                overall_risk=overall_risk,
                is_safe=overall_risk in [RiskLevel.LOW, RiskLevel.MODERATE],
                confidence_score=confidence_score,
                wind_risk=wind_risk,
                visibility_risk=visibility_risk,
                precipitation_risk=precipitation_risk,
                turbulence_risk=turbulence_risk,
                warnings=warnings,
                recommendations=recommendations,
                restrictions=restrictions_list,
                optimal_window=optimal_window,
                alternative_windows=alternative_windows,
                compatible_flight_plans=compatible_plans,
                recommended_modifications=modifications
            )
            
            logger.info(f"Safety analysis completed for {location.name}: {overall_risk.value}")
            return report
            
        except Exception as e:
            logger.error(f"Error in safety analysis: {str(e)}")
            raise
    
    async def _get_current_weather(self, location: Location) -> WeatherData:
        """Get current weather data for location."""
        try:
            return await self.data_orchestrator.get_current_weather(
                location.latitude, location.longitude
            )
        except Exception as e:
            logger.error(f"Failed to get current weather: {str(e)}")
            # Return default safe weather data
            return WeatherData(
                location=location,
                timestamp=datetime.utcnow(),
                temperature=20.0,
                humidity=50.0,
                pressure=1013.25,
                wind_speed=5.0,
                wind_direction=180.0,
                visibility=10000.0,
                weather_description="Unknown"
            )
    
    async def _get_forecast_weather(
        self, location: Location, duration_minutes: int
    ) -> List[WeatherData]:
        """Get forecast weather data for the flight duration."""
        try:
            # Get hourly forecast for the flight duration
            hours_needed = max(1, math.ceil(duration_minutes / 60))
            return await self.data_orchestrator.get_hourly_forecast(
                location.latitude, location.longitude, hours=hours_needed
            )
        except Exception as e:
            logger.error(f"Failed to get forecast weather: {str(e)}")
            return []
    
    def _weather_to_conditions(self, weather: WeatherData) -> WeatherConditions:
        """Convert WeatherData to WeatherConditions."""
        return WeatherConditions(
            wind_speed=weather.wind_speed or 0.0,
            wind_direction=weather.wind_direction or 0.0,
            wind_gust=getattr(weather, 'wind_gust', None),
            visibility=weather.visibility or 10000.0,
            ceiling=getattr(weather, 'cloud_ceiling', None),
            precipitation=getattr(weather, 'precipitation', 0.0),
            temperature=weather.temperature or 20.0,
            humidity=weather.humidity or 50.0,
            pressure=weather.pressure or 1013.25,
            turbulence_intensity=self._estimate_turbulence(weather)
        )
    
    def _estimate_turbulence(self, weather: WeatherData) -> float:
        """Estimate turbulence intensity from weather data."""
        # Simple turbulence estimation based on wind conditions
        wind_speed = weather.wind_speed or 0.0
        wind_gust = getattr(weather, 'wind_gust', wind_speed)
        
        # Turbulence increases with wind speed and gust factor
        gust_factor = (wind_gust - wind_speed) / max(wind_speed, 1.0)
        turbulence = min(1.0, (wind_speed * 0.05) + (gust_factor * 0.3))
        
        return turbulence
    
    def _assess_wind_risk(
        self, conditions: WeatherConditions, restrictions: FlightRestrictions
    ) -> RiskLevel:
        """Assess wind-related risk."""
        wind_speed = conditions.wind_speed
        wind_gust = conditions.wind_gust or wind_speed
        
        if wind_gust > restrictions.max_gust_speed:
            return RiskLevel.PROHIBITED
        elif wind_speed > restrictions.max_wind_speed:
            return RiskLevel.CRITICAL
        elif wind_speed > restrictions.max_wind_speed * 0.8:
            return RiskLevel.HIGH
        elif wind_speed > restrictions.max_wind_speed * 0.6:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW
    
    def _assess_visibility_risk(
        self, conditions: WeatherConditions, restrictions: FlightRestrictions
    ) -> RiskLevel:
        """Assess visibility-related risk."""
        visibility = conditions.visibility
        min_visibility = restrictions.min_visibility
        
        if visibility < min_visibility * 0.5:
            return RiskLevel.PROHIBITED
        elif visibility < min_visibility:
            return RiskLevel.CRITICAL
        elif visibility < min_visibility * 1.5:
            return RiskLevel.HIGH
        elif visibility < min_visibility * 2.0:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW
    
    def _assess_precipitation_risk(
        self, conditions: WeatherConditions, restrictions: FlightRestrictions
    ) -> RiskLevel:
        """Assess precipitation-related risk."""
        precipitation = conditions.precipitation
        max_precipitation = restrictions.max_precipitation
        
        if precipitation > max_precipitation * 2.0:
            return RiskLevel.PROHIBITED
        elif precipitation > max_precipitation:
            return RiskLevel.CRITICAL
        elif precipitation > max_precipitation * 0.7:
            return RiskLevel.HIGH
        elif precipitation > max_precipitation * 0.3:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW
    
    def _assess_turbulence_risk(
        self, conditions: WeatherConditions, altitude: float
    ) -> RiskLevel:
        """Assess turbulence-related risk."""
        turbulence = conditions.turbulence_intensity or 0.0
        
        # Turbulence risk increases with altitude in lower atmosphere
        altitude_factor = min(1.2, 1.0 + (altitude / 1000.0) * 0.2)
        adjusted_turbulence = turbulence * altitude_factor
        
        if adjusted_turbulence > 0.8:
            return RiskLevel.PROHIBITED
        elif adjusted_turbulence > 0.6:
            return RiskLevel.CRITICAL
        elif adjusted_turbulence > 0.4:
            return RiskLevel.HIGH
        elif adjusted_turbulence > 0.2:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW
    
    def _calculate_overall_risk(self, risk_assessments: Dict[str, RiskLevel]) -> RiskLevel:
        """Calculate overall risk from individual assessments."""
        # Convert risk levels to numeric scores
        risk_scores = {
            RiskLevel.LOW: 1,
            RiskLevel.MODERATE: 2,
            RiskLevel.HIGH: 3,
            RiskLevel.CRITICAL: 4,
            RiskLevel.PROHIBITED: 5
        }
        
        # If any assessment is PROHIBITED, overall risk is PROHIBITED
        if any(risk == RiskLevel.PROHIBITED for risk in risk_assessments.values()):
            return RiskLevel.PROHIBITED
        
        # Calculate weighted average
        weighted_score = sum(
            risk_scores[risk] * self.risk_weights.get(category, 0.1)
            for category, risk in risk_assessments.items()
        )
        
        # Convert back to risk level
        if weighted_score >= 4.5:
            return RiskLevel.PROHIBITED
        elif weighted_score >= 3.5:
            return RiskLevel.CRITICAL
        elif weighted_score >= 2.5:
            return RiskLevel.HIGH
        elif weighted_score >= 1.5:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW
    
    def _generate_warnings(
        self, conditions: WeatherConditions, restrictions: FlightRestrictions
    ) -> List[str]:
        """Generate safety warnings based on conditions."""
        warnings = []
        
        if conditions.wind_speed > restrictions.max_wind_speed * 0.8:
            warnings.append(f"High wind speed: {conditions.wind_speed:.1f} m/s")
        
        if conditions.wind_gust and conditions.wind_gust > restrictions.max_gust_speed * 0.8:
            warnings.append(f"Strong wind gusts: {conditions.wind_gust:.1f} m/s")
        
        if conditions.visibility < restrictions.min_visibility * 1.5:
            warnings.append(f"Reduced visibility: {conditions.visibility:.0f} m")
        
        if conditions.precipitation > restrictions.max_precipitation * 0.5:
            warnings.append(f"Precipitation present: {conditions.precipitation:.1f} mm/h")
        
        if conditions.turbulence_intensity and conditions.turbulence_intensity > 0.4:
            warnings.append("Moderate to severe turbulence expected")
        
        return warnings
    
    def _generate_recommendations(
        self,
        current: WeatherConditions,
        forecast: List[WeatherConditions],
        restrictions: FlightRestrictions
    ) -> List[str]:
        """Generate safety recommendations."""
        recommendations = []
        
        if current.wind_speed > restrictions.max_wind_speed * 0.6:
            recommendations.append("Consider postponing flight due to wind conditions")
            recommendations.append("If flying, use slower speeds and avoid aggressive maneuvers")
        
        if current.visibility < restrictions.min_visibility * 2.0:
            recommendations.append("Maintain visual contact with aircraft at all times")
            recommendations.append("Consider using LED lights for better visibility")
        
        if current.precipitation > 0:
            recommendations.append("Ensure aircraft is suitable for wet conditions")
            recommendations.append("Check battery and electronics for water resistance")
        
        if current.turbulence_intensity and current.turbulence_intensity > 0.3:
            recommendations.append("Fly at lower altitudes to avoid turbulence")
            recommendations.append("Use gentle control inputs")
        
        # Forecast-based recommendations
        if forecast and len(forecast) > 1:
            future_wind = max(w.wind_speed for w in forecast)
            if future_wind > current.wind_speed * 1.2:
                recommendations.append("Wind conditions expected to worsen - plan shorter flight")
        
        return recommendations
    
    def _generate_restrictions(
        self, overall_risk: RiskLevel, conditions: WeatherConditions
    ) -> List[str]:
        """Generate flight restrictions based on risk level."""
        restrictions = []
        
        if overall_risk == RiskLevel.PROHIBITED:
            restrictions.append("FLIGHT PROHIBITED - Do not fly")
            return restrictions
        
        if overall_risk == RiskLevel.CRITICAL:
            restrictions.append("Flight not recommended for inexperienced pilots")
            restrictions.append("Emergency landing sites must be identified")
        
        if overall_risk in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            restrictions.append("Reduced maximum flight altitude")
            restrictions.append("Shortened flight duration recommended")
            restrictions.append("Maintain closer proximity to pilot")
        
        if conditions.wind_speed > 8.0:
            restrictions.append("Avoid flights near obstacles")
            restrictions.append("Use manual flight mode for better control")
        
        if conditions.visibility < 3000:
            restrictions.append("Maintain visual line of sight at all times")
            restrictions.append("Avoid flights over water or uniform terrain")
        
        return restrictions
    
    def _find_optimal_window(
        self, forecast: List[WeatherConditions]
    ) -> Optional[Dict[str, datetime]]:
        """Find the optimal flight window in the forecast."""
        if not forecast:
            return None
        
        # Find the period with lowest combined risk
        best_score = float('inf')
        best_window = None
        
        for i, conditions in enumerate(forecast):
            score = (
                conditions.wind_speed * 2.0 +
                (1.0 / max(conditions.visibility / 1000.0, 0.1)) +
                conditions.precipitation * 10.0 +
                (conditions.turbulence_intensity or 0.0) * 5.0
            )
            
            if score < best_score:
                best_score = score
                best_window = {
                    'start': datetime.utcnow() + timedelta(hours=i),
                    'end': datetime.utcnow() + timedelta(hours=i+1)
                }
        
        return best_window
    
    def _find_alternative_windows(
        self, forecast: List[WeatherConditions]
    ) -> List[Dict[str, datetime]]:
        """Find alternative flight windows."""
        windows = []
        
        for i, conditions in enumerate(forecast):
            # Check if conditions are acceptable
            if (conditions.wind_speed <= 10.0 and
                conditions.visibility >= 1000.0 and
                conditions.precipitation <= 0.5):
                
                windows.append({
                    'start': datetime.utcnow() + timedelta(hours=i),
                    'end': datetime.utcnow() + timedelta(hours=i+1)
                })
        
        return windows[:3]  # Return up to 3 alternative windows
    
    def _analyze_flight_plan_compatibility(
        self,
        flight_plan: FlightPlan,
        conditions: WeatherConditions,
        restrictions: FlightRestrictions
    ) -> Tuple[List[str], List[str]]:
        """Analyze flight plan compatibility with weather conditions."""
        compatible_plans = []
        modifications = []
        
        # Check if current flight plan is compatible
        if self._is_flight_plan_safe(flight_plan, conditions, restrictions):
            compatible_plans.append("Current flight plan is safe to execute")
        else:
            modifications.extend(self._suggest_flight_plan_modifications(
                flight_plan, conditions, restrictions
            ))
        
        return compatible_plans, modifications
    
    def _is_flight_plan_safe(
        self,
        flight_plan: FlightPlan,
        conditions: WeatherConditions,
        restrictions: FlightRestrictions
    ) -> bool:
        """Check if flight plan is safe under current conditions."""
        # Check altitude restrictions
        max_altitude = max(
            wp.altitude for wp in [flight_plan.takeoff_location] + 
            flight_plan.waypoints + [flight_plan.landing_location]
        )
        
        if max_altitude > restrictions.max_altitude:
            return False
        
        # Check wind speed vs aircraft capabilities
        if conditions.wind_speed > flight_plan.max_speed * 0.5:
            return False
        
        return True
    
    def _suggest_flight_plan_modifications(
        self,
        flight_plan: FlightPlan,
        conditions: WeatherConditions,
        restrictions: FlightRestrictions
    ) -> List[str]:
        """Suggest modifications to make flight plan safer."""
        modifications = []
        
        # Altitude modifications
        max_altitude = max(
            wp.altitude for wp in [flight_plan.takeoff_location] + 
            flight_plan.waypoints + [flight_plan.landing_location]
        )
        
        if max_altitude > restrictions.max_altitude:
            modifications.append(
                f"Reduce maximum altitude to {restrictions.max_altitude}m"
            )
        
        # Speed modifications
        if conditions.wind_speed > 8.0:
            modifications.append("Reduce flight speed by 25% due to wind conditions")
        
        # Route modifications
        if conditions.visibility < 2000:
            modifications.append("Simplify route to maintain visual contact")
        
        return modifications
    
    def _calculate_confidence_score(
        self,
        current_weather: WeatherData,
        forecast_weather: List[WeatherData],
        overall_risk: RiskLevel
    ) -> float:
        """Calculate confidence score for the assessment."""
        base_confidence = 0.8
        
        # Reduce confidence if no forecast data
        if not forecast_weather:
            base_confidence -= 0.2
        
        # Reduce confidence for high-risk conditions
        risk_penalty = {
            RiskLevel.LOW: 0.0,
            RiskLevel.MODERATE: 0.05,
            RiskLevel.HIGH: 0.1,
            RiskLevel.CRITICAL: 0.15,
            RiskLevel.PROHIBITED: 0.2
        }
        
        base_confidence -= risk_penalty.get(overall_risk, 0.1)
        
        # Ensure confidence is between 0 and 1
        return max(0.0, min(1.0, base_confidence)) 