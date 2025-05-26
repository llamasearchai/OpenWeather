"""
MAVLink communication connector for drone integration.

Handles MAVLink protocol communication with drones including:
- Connection management
- Message parsing and sending
- Telemetry data extraction
- Command execution
- Real-time status monitoring
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from threading import Thread, Event
from queue import Queue, Empty

try:
    from pymavlink import mavutil
    from pymavlink.dialects.v20 import common as mavlink
    MAVLINK_AVAILABLE = True
except ImportError:
    MAVLINK_AVAILABLE = False
    mavutil = None
    mavlink = None

from .models import (
    DroneStatus,
    MAVLinkMessage,
    DroneCommand,
    FlightMode,
    FlightPhase,
    WeatherConditions,
)

logger = logging.getLogger(__name__)


class MAVLinkConnector:
    """
    MAVLink protocol connector for drone communication.
    
    Features:
    - Real-time telemetry streaming
    - Command and control interface
    - Mission upload/download
    - Parameter management
    - Heartbeat monitoring
    - Connection health management
    """
    
    def __init__(
        self,
        connection_string: str = "tcp:127.0.0.1:5760",
        system_id: int = 255,
        component_id: int = 1
    ):
        """
        Initialize MAVLink connector.
        
        Args:
            connection_string: MAVLink connection string
            system_id: Ground control system ID
            component_id: Ground control component ID
        """
        if not MAVLINK_AVAILABLE:
            raise ImportError("PyMAVLink is required for drone connectivity. Install with: pip install pymavlink")
        
        self.connection_string = connection_string
        self.system_id = system_id
        self.component_id = component_id
        
        # Connection state
        self.connection = None
        self.connected = False
        self.target_system = 1
        self.target_component = 1
        
        # Message handling
        self.message_queue = Queue()
        self.message_handlers: Dict[str, List[Callable]] = {}
        self.telemetry_data: Dict[str, Any] = {}
        
        # Threading
        self.receiver_thread = None
        self.processor_thread = None
        self.stop_event = Event()
        
        # Heartbeat monitoring
        self.last_heartbeat = None
        self.heartbeat_timeout = 10.0  # seconds
        
        # Flight mode mapping
        self.flight_mode_mapping = {
            0: FlightMode.MANUAL,
            1: FlightMode.ALT_HOLD,
            2: FlightMode.LOITER,
            3: FlightMode.AUTO,
            4: FlightMode.GUIDED,
            5: FlightMode.STABILIZE,
            6: FlightMode.RTL,
            7: FlightMode.LAND,
        }
    
    async def connect(self) -> bool:
        """
        Establish connection to drone.
        
        Returns:
            True if connection successful
        """
        try:
            logger.info(f"Connecting to drone at {self.connection_string}")
            
            # Create MAVLink connection
            self.connection = mavutil.mavlink_connection(
                self.connection_string,
                source_system=self.system_id,
                source_component=self.component_id,
                autoreconnect=True
            )
            
            # Wait for first heartbeat
            logger.info("Waiting for heartbeat...")
            heartbeat = self.connection.wait_heartbeat(timeout=10)
            
            if heartbeat:
                self.target_system = heartbeat.get_srcSystem()
                self.target_component = heartbeat.get_srcComponent()
                self.connected = True
                self.last_heartbeat = time.time()
                
                logger.info(f"Connected to system {self.target_system}, component {self.target_component}")
                
                # Start message processing threads
                self._start_background_threads()
                
                # Request data streams
                await self._request_data_streams()
                
                return True
            else:
                logger.error("No heartbeat received - connection failed")
                return False
                
        except Exception as e:
            logger.error(f"Connection failed: {str(e)}")
            self.connected = False
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from drone."""
        try:
            logger.info("Disconnecting from drone")
            
            # Stop background threads
            self.stop_event.set()
            
            if self.receiver_thread and self.receiver_thread.is_alive():
                self.receiver_thread.join(timeout=2.0)
            
            if self.processor_thread and self.processor_thread.is_alive():
                self.processor_thread.join(timeout=2.0)
            
            # Close connection
            if self.connection:
                self.connection.close()
            
            self.connected = False
            self.connection = None
            
            logger.info("Disconnected from drone")
            
        except Exception as e:
            logger.error(f"Error during disconnect: {str(e)}")
    
    async def get_drone_status(self) -> Optional[DroneStatus]:
        """
        Get current drone status.
        
        Returns:
            Current drone status or None if not available
        """
        if not self.connected or not self.telemetry_data:
            return None
        
        try:
            # Extract data from telemetry
            attitude = self.telemetry_data.get('ATTITUDE', {})
            global_pos = self.telemetry_data.get('GLOBAL_POSITION_INT', {})
            vfr_hud = self.telemetry_data.get('VFR_HUD', {})
            sys_status = self.telemetry_data.get('SYS_STATUS', {})
            gps_raw = self.telemetry_data.get('GPS_RAW_INT', {})
            heartbeat = self.telemetry_data.get('HEARTBEAT', {})
            
            # Check if we have minimum required data
            if not global_pos or not heartbeat:
                return None
            
            # Extract position
            latitude = global_pos.get('lat', 0) / 1e7
            longitude = global_pos.get('lon', 0) / 1e7
            altitude = global_pos.get('alt', 0) / 1000.0  # mm to m
            
            # Extract orientation
            heading = math.degrees(attitude.get('yaw', 0.0))
            if heading < 0:
                heading += 360
            
            # Extract velocities
            ground_speed = vfr_hud.get('groundspeed', 0.0)
            vertical_speed = global_pos.get('vz', 0) / 100.0  # cm/s to m/s
            airspeed = vfr_hud.get('airspeed', 0.0)
            
            # Extract flight state
            flight_mode_num = heartbeat.get('custom_mode', 0)
            flight_mode = self.flight_mode_mapping.get(flight_mode_num, FlightMode.MANUAL)
            
            is_armed = bool(heartbeat.get('base_mode', 0) & mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            is_flying = altitude > 1.0 and ground_speed > 0.5  # Simple heuristic
            
            # Extract system status
            battery_voltage = sys_status.get('voltage_battery', 0) / 1000.0  # mV to V
            battery_percentage = sys_status.get('battery_remaining', -1)
            
            # GPS status
            gps_fix = gps_raw.get('fix_type', 0)
            satellites_visible = gps_raw.get('satellites_visible', 0)
            
            # Determine flight phase
            flight_phase = self._determine_flight_phase(
                is_armed, is_flying, altitude, ground_speed, flight_mode
            )
            
            # Create status object
            status = DroneStatus(
                drone_id=f"drone_{self.target_system}",
                latitude=latitude,
                longitude=longitude,
                altitude=altitude,
                heading=heading,
                ground_speed=ground_speed,
                vertical_speed=vertical_speed,
                airspeed=airspeed,
                flight_mode=flight_mode,
                flight_phase=flight_phase,
                is_armed=is_armed,
                is_flying=is_flying,
                battery_voltage=battery_voltage,
                battery_percentage=max(0, battery_percentage),
                gps_fix=gps_fix,
                satellites_visible=satellites_visible,
                temperature=self.telemetry_data.get('temperature', 20.0),
                barometric_pressure=self.telemetry_data.get('pressure', 1013.25)
            )
            
            return status
            
        except Exception as e:
            logger.error(f"Error extracting drone status: {str(e)}")
            return None
    
    async def send_command(self, command: DroneCommand) -> bool:
        """
        Send command to drone.
        
        Args:
            command: Command to send
            
        Returns:
            True if command sent successfully
        """
        if not self.connected or not self.connection:
            logger.error("Not connected to drone")
            return False
        
        try:
            if command.command_type == "ARM":
                return await self._arm_disarm(True)
            elif command.command_type == "DISARM":
                return await self._arm_disarm(False)
            elif command.command_type == "TAKEOFF":
                altitude = command.parameters.get('altitude', 10.0)
                return await self._takeoff(altitude)
            elif command.command_type == "LAND":
                return await self._land()
            elif command.command_type == "RTL":
                return await self._return_to_launch()
            elif command.command_type == "GOTO":
                lat = command.parameters.get('latitude')
                lon = command.parameters.get('longitude')
                alt = command.parameters.get('altitude')
                return await self._goto_position(lat, lon, alt)
            elif command.command_type == "SET_MODE":
                mode = command.parameters.get('mode')
                return await self._set_flight_mode(mode)
            else:
                logger.warning(f"Unknown command type: {command.command_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending command: {str(e)}")
            return False
    
    def add_message_handler(self, message_type: str, handler: Callable[[MAVLinkMessage], None]) -> None:
        """
        Add handler for specific message type.
        
        Args:
            message_type: MAVLink message type
            handler: Handler function
        """
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        self.message_handlers[message_type].append(handler)
    
    def is_connected(self) -> bool:
        """Check if connected to drone."""
        if not self.connected:
            return False
        
        # Check heartbeat timeout
        if self.last_heartbeat and time.time() - self.last_heartbeat > self.heartbeat_timeout:
            logger.warning("Heartbeat timeout - connection may be lost")
            return False
        
        return True
    
    def _start_background_threads(self) -> None:
        """Start background message processing threads."""
        self.stop_event.clear()
        
        # Message receiver thread
        self.receiver_thread = Thread(target=self._message_receiver_loop, daemon=True)
        self.receiver_thread.start()
        
        # Message processor thread
        self.processor_thread = Thread(target=self._message_processor_loop, daemon=True)
        self.processor_thread.start()
    
    def _message_receiver_loop(self) -> None:
        """Background thread for receiving MAVLink messages."""
        while not self.stop_event.is_set() and self.connection:
            try:
                # Receive message with timeout
                msg = self.connection.recv_match(blocking=True, timeout=1.0)
                
                if msg:
                    # Create MAVLink message object
                    mavlink_msg = MAVLinkMessage(
                        message_type=msg.get_type(),
                        system_id=msg.get_srcSystem(),
                        component_id=msg.get_srcComponent(),
                        timestamp=datetime.utcnow(),
                        data=msg.to_dict()
                    )
                    
                    # Add to processing queue
                    self.message_queue.put(mavlink_msg)
                    
            except Exception as e:
                if not self.stop_event.is_set():
                    logger.error(f"Error receiving message: {str(e)}")
                break
    
    def _message_processor_loop(self) -> None:
        """Background thread for processing received messages."""
        while not self.stop_event.is_set():
            try:
                # Get message from queue
                try:
                    msg = self.message_queue.get(timeout=1.0)
                except Empty:
                    continue
                
                # Update telemetry data
                self.telemetry_data[msg.message_type] = msg.data
                
                # Handle heartbeat
                if msg.message_type == "HEARTBEAT":
                    self.last_heartbeat = time.time()
                
                # Call registered handlers
                handlers = self.message_handlers.get(msg.message_type, [])
                for handler in handlers:
                    try:
                        handler(msg)
                    except Exception as e:
                        logger.error(f"Error in message handler: {str(e)}")
                
                # Mark task as done
                self.message_queue.task_done()
                
            except Exception as e:
                if not self.stop_event.is_set():
                    logger.error(f"Error processing message: {str(e)}")
    
    async def _request_data_streams(self) -> None:
        """Request data streams from drone."""
        try:
            # Request telemetry streams
            streams = [
                (mavlink.MAV_DATA_STREAM_POSITION, 4),      # Position at 4Hz
                (mavlink.MAV_DATA_STREAM_RAW_SENSORS, 2),   # Sensors at 2Hz
                (mavlink.MAV_DATA_STREAM_EXTENDED_STATUS, 2), # Status at 2Hz
                (mavlink.MAV_DATA_STREAM_RC_CHANNELS, 2),   # RC at 2Hz
                (mavlink.MAV_DATA_STREAM_EXTRA1, 4),        # Attitude at 4Hz
                (mavlink.MAV_DATA_STREAM_EXTRA2, 4),        # VFR_HUD at 4Hz
            ]
            
            for stream_id, rate in streams:
                self.connection.mav.request_data_stream_send(
                    self.target_system,
                    self.target_component,
                    stream_id,
                    rate,
                    1  # start/stop (1=start, 0=stop)
                )
            
            logger.info("Requested data streams")
            
        except Exception as e:
            logger.error(f"Error requesting data streams: {str(e)}")
    
    async def _arm_disarm(self, arm: bool) -> bool:
        """Arm or disarm the drone."""
        try:
            self.connection.mav.command_long_send(
                self.target_system,
                self.target_component,
                mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,  # confirmation
                1 if arm else 0,  # arm/disarm
                0, 0, 0, 0, 0, 0  # unused parameters
            )
            
            action = "Armed" if arm else "Disarmed"
            logger.info(f"{action} command sent")
            return True
            
        except Exception as e:
            logger.error(f"Error sending arm/disarm command: {str(e)}")
            return False
    
    async def _takeoff(self, altitude: float) -> bool:
        """Command drone to takeoff to specified altitude."""
        try:
            self.connection.mav.command_long_send(
                self.target_system,
                self.target_component,
                mavlink.MAV_CMD_NAV_TAKEOFF,
                0,  # confirmation
                0,  # pitch
                0, 0, 0, 0, 0,
                altitude  # altitude
            )
            
            logger.info(f"Takeoff command sent (altitude: {altitude}m)")
            return True
            
        except Exception as e:
            logger.error(f"Error sending takeoff command: {str(e)}")
            return False
    
    async def _land(self) -> bool:
        """Command drone to land."""
        try:
            self.connection.mav.command_long_send(
                self.target_system,
                self.target_component,
                mavlink.MAV_CMD_NAV_LAND,
                0,  # confirmation
                0, 0, 0, 0, 0, 0, 0  # parameters
            )
            
            logger.info("Land command sent")
            return True
            
        except Exception as e:
            logger.error(f"Error sending land command: {str(e)}")
            return False
    
    async def _return_to_launch(self) -> bool:
        """Command drone to return to launch."""
        try:
            self.connection.mav.command_long_send(
                self.target_system,
                self.target_component,
                mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
                0,  # confirmation
                0, 0, 0, 0, 0, 0, 0  # parameters
            )
            
            logger.info("Return to launch command sent")
            return True
            
        except Exception as e:
            logger.error(f"Error sending RTL command: {str(e)}")
            return False
    
    async def _goto_position(self, latitude: float, longitude: float, altitude: float) -> bool:
        """Command drone to go to specified position."""
        try:
            self.connection.mav.set_position_target_global_int_send(
                0,  # time_boot_ms
                self.target_system,
                self.target_component,
                mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                0b0000111111111000,  # type_mask (position only)
                int(latitude * 1e7),
                int(longitude * 1e7),
                altitude,
                0, 0, 0,  # velocity
                0, 0, 0,  # acceleration
                0, 0  # yaw, yaw_rate
            )
            
            logger.info(f"Goto position command sent: {latitude}, {longitude}, {altitude}m")
            return True
            
        except Exception as e:
            logger.error(f"Error sending goto command: {str(e)}")
            return False
    
    async def _set_flight_mode(self, mode: str) -> bool:
        """Set flight mode."""
        try:
            # Map mode string to MAVLink mode
            mode_mapping = {
                'MANUAL': 0,
                'STABILIZE': 0,
                'ALT_HOLD': 1,
                'LOITER': 5,
                'AUTO': 3,
                'GUIDED': 4,
                'RTL': 6,
                'LAND': 9
            }
            
            mode_id = mode_mapping.get(mode.upper())
            if mode_id is None:
                logger.error(f"Unknown flight mode: {mode}")
                return False
            
            self.connection.mav.set_mode_send(
                self.target_system,
                mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                mode_id
            )
            
            logger.info(f"Set mode command sent: {mode}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting flight mode: {str(e)}")
            return False
    
    def _determine_flight_phase(
        self,
        is_armed: bool,
        is_flying: bool,
        altitude: float,
        ground_speed: float,
        flight_mode: FlightMode
    ) -> FlightPhase:
        """Determine current flight phase based on status."""
        if not is_armed:
            return FlightPhase.PREFLIGHT
        
        if not is_flying:
            if flight_mode == FlightMode.LAND:
                return FlightPhase.LANDING
            else:
                return FlightPhase.PREFLIGHT
        
        # Flying
        if altitude < 5.0 and ground_speed < 2.0:
            return FlightPhase.TAKEOFF
        elif flight_mode == FlightMode.LAND or (altitude < 10.0 and ground_speed < 2.0):
            return FlightPhase.LANDING
        elif ground_speed > 5.0:
            return FlightPhase.CRUISE
        else:
            return FlightPhase.LOITER


# Import guard for optional functionality
if not MAVLINK_AVAILABLE:
    import math
    
    class MAVLinkConnector:
        """Stub implementation when PyMAVLink is not available."""
        
        def __init__(self, *args, **kwargs):
            logger.warning("MAVLink not available - using stub implementation")
        
        async def connect(self) -> bool:
            logger.warning("MAVLink not available - simulating connection")
            return True
        
        async def disconnect(self) -> None:
            pass
        
        async def get_drone_status(self) -> Optional[DroneStatus]:
            # Return simulated status
            return DroneStatus(
                drone_id="simulated_drone",
                latitude=51.5074,
                longitude=-0.1278,
                altitude=50.0,
                heading=0.0,
                ground_speed=5.0,
                vertical_speed=0.0,
                flight_mode=FlightMode.AUTO,
                flight_phase=FlightPhase.CRUISE,
                is_armed=True,
                is_flying=True,
                battery_voltage=12.6,
                battery_percentage=75,
                gps_fix=3,
                satellites_visible=12
            )
        
        async def send_command(self, command: DroneCommand) -> bool:
            logger.info(f"Simulated command: {command.command_type}")
            return True
        
        def add_message_handler(self, message_type: str, handler: Callable) -> None:
            pass
        
        def is_connected(self) -> bool:
            return True 