"""
Drone operation data models for type safety and validation.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from pydantic import BaseModel, Field, validator

from openweather.core.models_shared import Location, WeatherData


class FlightMode(str, Enum):
    """Drone flight modes."""
    MANUAL = "manual"
    AUTO = "auto"
    GUIDED = "guided"
    LOITER = "loiter"
    RTL = "return_to_launch"
    LAND = "land"
    STABILIZE = "stabilize"
    ALT_HOLD = "altitude_hold"


class DroneType(str, Enum):
    """Types of drones supported."""
    MULTIROTOR = "multirotor"
    FIXED_WING = "fixed_wing"
    HELICOPTER = "helicopter"
    VTOL = "vtol"
    ROVER = "rover"
    BOAT = "boat"


class FlightPhase(str, Enum):
    """Phases of flight."""
    PREFLIGHT = "preflight"
    TAKEOFF = "takeoff"
    CLIMB = "climb"
    CRUISE = "cruise"
    APPROACH = "approach"
    LANDING = "landing"
    EMERGENCY = "emergency"


class RiskLevel(str, Enum):
    """Risk assessment levels."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    PROHIBITED = "prohibited"


class WeatherConditions(BaseModel):
    """Weather conditions for drone operations."""
    
    wind_speed: float = Field(..., description="Wind speed in m/s")
    wind_direction: float = Field(..., description="Wind direction in degrees")
    wind_gust: Optional[float] = Field(None, description="Wind gust speed in m/s")
    visibility: float = Field(..., description="Visibility in meters")
    ceiling: Optional[float] = Field(None, description="Cloud ceiling in meters")
    precipitation: float = Field(0.0, description="Precipitation rate in mm/h")
    temperature: float = Field(..., description="Temperature in Celsius")
    humidity: float = Field(..., description="Relative humidity percentage")
    pressure: float = Field(..., description="Atmospheric pressure in hPa")
    turbulence_intensity: Optional[float] = Field(None, description="Turbulence intensity 0-1")
    
    @validator('wind_speed', 'wind_gust')
    def validate_wind_speeds(cls, v):
        if v is not None and v < 0:
            raise ValueError('Wind speeds cannot be negative')
        return v
    
    @validator('visibility')
    def validate_visibility(cls, v):
        if v < 0:
            raise ValueError('Visibility cannot be negative')
        return v


class FlightRestrictions(BaseModel):
    """Flight restrictions and safety margins."""
    
    max_wind_speed: float = Field(12.0, description="Maximum wind speed in m/s")
    max_gust_speed: float = Field(15.0, description="Maximum gust speed in m/s")
    min_visibility: float = Field(1000.0, description="Minimum visibility in meters")
    max_precipitation: float = Field(0.1, description="Maximum precipitation in mm/h")
    min_ceiling: Optional[float] = Field(150.0, description="Minimum ceiling in meters")
    max_altitude: float = Field(120.0, description="Maximum flight altitude in meters")
    no_fly_zones: List[Dict[str, Any]] = Field(default_factory=list)
    time_restrictions: List[Dict[str, Any]] = Field(default_factory=list)


class Waypoint(BaseModel):
    """A waypoint in a flight plan."""
    
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    altitude: float = Field(..., ge=0, le=500, description="Altitude in meters AGL")
    speed: Optional[float] = Field(None, description="Speed in m/s")
    heading: Optional[float] = Field(None, ge=0, lt=360, description="Heading in degrees")
    action: Optional[str] = Field(None, description="Action at waypoint")
    dwell_time: Optional[float] = Field(None, description="Time to remain at waypoint in seconds")


class FlightPlan(BaseModel):
    """Complete flight plan for drone mission."""
    
    mission_id: str = Field(..., description="Unique mission identifier")
    drone_id: str = Field(..., description="Drone identifier")
    pilot_id: Optional[str] = Field(None, description="Pilot identifier")
    
    # Flight details
    takeoff_location: Waypoint = Field(..., description="Takeoff waypoint")
    waypoints: List[Waypoint] = Field(..., description="Mission waypoints")
    landing_location: Waypoint = Field(..., description="Landing waypoint")
    
    # Timing
    planned_start: datetime = Field(..., description="Planned mission start time")
    estimated_duration: timedelta = Field(..., description="Estimated flight duration")
    
    # Aircraft details
    drone_type: DroneType = Field(..., description="Type of drone")
    max_speed: float = Field(..., description="Maximum speed in m/s")
    max_altitude: float = Field(..., description="Maximum altitude in meters")
    battery_capacity: float = Field(..., description="Battery capacity in mAh")
    payload_weight: float = Field(0.0, description="Payload weight in kg")
    
    # Safety and restrictions
    restrictions: FlightRestrictions = Field(default_factory=FlightRestrictions)
    emergency_procedures: Dict[str, str] = Field(default_factory=dict)
    
    # Weather requirements
    required_weather_conditions: WeatherConditions = Field(...)
    
    @validator('waypoints')
    def validate_waypoints(cls, v):
        if len(v) < 1:
            raise ValueError('At least one waypoint is required')
        return v


class DroneStatus(BaseModel):
    """Current status of a drone."""
    
    drone_id: str = Field(..., description="Drone identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Position and orientation
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    altitude: float = Field(..., description="Altitude in meters")
    heading: float = Field(..., ge=0, lt=360, description="Heading in degrees")
    
    # Velocity
    ground_speed: float = Field(..., description="Ground speed in m/s")
    vertical_speed: float = Field(..., description="Vertical speed in m/s")
    airspeed: Optional[float] = Field(None, description="Airspeed in m/s")
    
    # Flight state
    flight_mode: FlightMode = Field(..., description="Current flight mode")
    flight_phase: FlightPhase = Field(..., description="Current flight phase")
    is_armed: bool = Field(..., description="Whether drone is armed")
    is_flying: bool = Field(..., description="Whether drone is airborne")
    
    # System status
    battery_voltage: float = Field(..., description="Battery voltage")
    battery_percentage: float = Field(..., ge=0, le=100, description="Battery percentage")
    gps_fix: int = Field(..., description="GPS fix type (0=no fix, 3=3D fix)")
    satellites_visible: int = Field(..., description="Number of visible satellites")
    
    # Sensor readings
    temperature: Optional[float] = Field(None, description="Temperature in Celsius")
    barometric_pressure: Optional[float] = Field(None, description="Pressure in hPa")
    wind_speed_measured: Optional[float] = Field(None, description="Measured wind speed")
    
    # Mission status
    current_waypoint: Optional[int] = Field(None, description="Current waypoint index")
    mission_progress: float = Field(0.0, ge=0, le=100, description="Mission progress percentage")


class SafetyReport(BaseModel):
    """Safety assessment report for drone operations."""
    
    assessment_id: str = Field(..., description="Unique assessment identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Location and timing
    location: Location = Field(..., description="Assessment location")
    assessment_time: datetime = Field(..., description="Time of assessment")
    valid_until: datetime = Field(..., description="Assessment validity end time")
    
    # Weather conditions
    current_weather: WeatherConditions = Field(..., description="Current weather")
    forecast_weather: List[WeatherConditions] = Field(..., description="Forecast conditions")
    
    # Safety assessment
    overall_risk: RiskLevel = Field(..., description="Overall risk level")
    is_safe: bool = Field(..., description="Whether flight is safe")
    confidence_score: float = Field(..., ge=0, le=1, description="Confidence in assessment")
    
    # Detailed analysis
    wind_risk: RiskLevel = Field(..., description="Wind-related risk")
    visibility_risk: RiskLevel = Field(..., description="Visibility-related risk")
    precipitation_risk: RiskLevel = Field(..., description="Precipitation-related risk")
    turbulence_risk: RiskLevel = Field(..., description="Turbulence-related risk")
    
    # Warnings and recommendations
    warnings: List[str] = Field(default_factory=list, description="Safety warnings")
    recommendations: List[str] = Field(default_factory=list, description="Safety recommendations")
    restrictions: List[str] = Field(default_factory=list, description="Flight restrictions")
    
    # Optimal flight window
    optimal_window: Optional[Dict[str, datetime]] = Field(None, description="Optimal flight window")
    alternative_windows: List[Dict[str, datetime]] = Field(default_factory=list)
    
    # Flight plan compatibility
    compatible_flight_plans: List[str] = Field(default_factory=list)
    recommended_modifications: List[str] = Field(default_factory=list)


@dataclass
class MAVLinkMessage:
    """Represents a MAVLink message."""
    
    message_type: str
    system_id: int
    component_id: int
    timestamp: datetime
    data: Dict[str, Any]


@dataclass
class DroneCommand:
    """Command to be sent to drone."""
    
    command_type: str
    parameters: Dict[str, Any]
    target_system: int
    target_component: int
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class FlightMetrics(BaseModel):
    """Flight performance metrics."""
    
    flight_id: str = Field(..., description="Flight identifier")
    start_time: datetime = Field(..., description="Flight start time")
    end_time: Optional[datetime] = Field(None, description="Flight end time")
    
    # Distance and time
    total_distance: float = Field(0.0, description="Total distance flown in meters")
    flight_duration: Optional[timedelta] = Field(None, description="Total flight time")
    
    # Performance metrics
    average_speed: float = Field(0.0, description="Average speed in m/s")
    max_speed: float = Field(0.0, description="Maximum speed reached in m/s")
    max_altitude: float = Field(0.0, description="Maximum altitude reached in meters")
    
    # Battery usage
    battery_consumed: float = Field(0.0, description="Battery consumed in percentage")
    energy_efficiency: float = Field(0.0, description="Energy efficiency in mAh/km")
    
    # Weather exposure
    max_wind_encountered: float = Field(0.0, description="Maximum wind speed encountered")
    weather_delays: timedelta = Field(timedelta(), description="Weather-related delays")
    
    # Safety metrics
    safety_incidents: int = Field(0, description="Number of safety incidents")
    emergency_landings: int = Field(0, description="Number of emergency landings")
    route_deviations: int = Field(0, description="Number of route deviations") 