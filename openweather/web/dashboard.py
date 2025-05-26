"""OpenWeather platform web dashboard."""

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from openweather.core.monitoring import monitoring
from openweather.services.weather_service import weather_service
from openweather.services.analytics_service import analytics_service
from openweather.models.weather import Location

logger = logging.getLogger(__name__)

class DashboardManager:
    """Manages the web dashboard for OpenWeather platform."""
    
    def __init__(self):
        self.app = FastAPI(title="OpenWeather Dashboard")
        self.templates = Jinja2Templates(directory="openweather/web/templates")
        self.websocket_connections: List[WebSocket] = []
        
        # Mount static files
        self.app.mount("/static", StaticFiles(directory="openweather/web/static"), name="static")
        
        # Setup routes
        self._setup_routes()
        
        # Start background tasks
        self._start_dashboard_tasks()
    
    def _setup_routes(self):
        """Setup dashboard routes."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home(request: Request):
            """Main dashboard page."""
            return self.templates.TemplateResponse("dashboard.html", {
                "request": request,
                "title": "OpenWeather Dashboard",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        
        @self.app.get("/monitoring", response_class=HTMLResponse)
        async def monitoring_dashboard(request: Request):
            """System monitoring dashboard."""
            return self.templates.TemplateResponse("monitoring.html", {
                "request": request,
                "title": "System Monitoring"
            })
        
        @self.app.get("/weather", response_class=HTMLResponse)
        async def weather_dashboard(request: Request):
            """Weather data dashboard."""
            return self.templates.TemplateResponse("weather.html", {
                "request": request,
                "title": "Weather Dashboard"
            })
        
        @self.app.get("/analytics", response_class=HTMLResponse)
        async def analytics_dashboard(request: Request):
            """Analytics dashboard."""
            return self.templates.TemplateResponse("analytics.html", {
                "request": request,
                "title": "Weather Analytics"
            })
        
        @self.app.get("/drone", response_class=HTMLResponse)
        async def drone_dashboard(request: Request):
            """Drone operations dashboard."""
            return self.templates.TemplateResponse("drone.html", {
                "request": request,
                "title": "Drone Operations"
            })
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates."""
            await self._handle_websocket(websocket)
        
        @self.app.get("/api/dashboard/summary")
        async def get_dashboard_summary():
            """Get dashboard summary data."""
            return await self._get_dashboard_summary()
        
        @self.app.get("/api/dashboard/metrics")
        async def get_metrics_data():
            """Get metrics data for charts."""
            return await self._get_metrics_data()
        
        @self.app.get("/api/dashboard/weather")
        async def get_weather_data():
            """Get weather data for dashboard."""
            return await self._get_weather_data()
        
        @self.app.get("/api/dashboard/analytics")
        async def get_analytics_data():
            """Get analytics data for dashboard."""
            return await self._get_analytics_data()
        
        @self.app.get("/api/dashboard/chart/weather")
        async def get_weather_chart(
            latitude: float = 37.7749,
            longitude: float = -122.4194,
            days: int = 7
        ):
            """Generate weather forecast chart."""
            return await self._create_weather_chart(latitude, longitude, days)
        
        @self.app.get("/api/dashboard/chart/system")
        async def get_system_chart():
            """Generate system metrics chart."""
            return await self._create_system_metrics_chart()
    
    async def _handle_websocket(self, websocket: WebSocket):
        """Handle WebSocket connections for real-time updates."""
        await websocket.accept()
        self.websocket_connections.append(websocket)
        
        try:
            while True:
                # Send periodic updates
                data = await self._get_realtime_data()
                await websocket.send_json(data)
                await asyncio.sleep(5)  # Update every 5 seconds
                
        except WebSocketDisconnect:
            self.websocket_connections.remove(websocket)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            if websocket in self.websocket_connections:
                self.websocket_connections.remove(websocket)
    
    async def _get_dashboard_summary(self) -> Dict[str, Any]:
        """Get dashboard summary data."""
        try:
            # Get monitoring data
            monitoring_data = monitoring.get_monitoring_dashboard()
            
            # Get weather service stats
            weather_stats = weather_service.get_cache_stats()
            
            # System metrics
            all_metrics = monitoring.metrics.get_all_metrics()
            
            summary = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "system_health": {
                    "overall_status": monitoring_data["health"]["overall_status"],
                    "active_alerts": len(monitoring_data["active_alerts"]),
                    "critical_alerts": len([
                        a for a in monitoring_data["active_alerts"] 
                        if a["level"] == "critical"
                    ])
                },
                "weather_service": {
                    "cache_hit_rate": weather_stats.get("hit_rate", 0),
                    "cache_entries": weather_stats.get("entries", 0),
                    "providers_available": len(weather_service.providers)
                },
                "system_resources": {
                    "cpu_usage": all_metrics.get("system.cpu.usage_percent", {}).get("current_value", 0),
                    "memory_usage": all_metrics.get("system.memory.usage_percent", {}).get("current_value", 0),
                    "disk_usage": all_metrics.get("system.disk.usage_percent", {}).get("current_value", 0)
                },
                "performance": {
                    "total_requests": sum(
                        m.get("current_value", 0) for name, m in all_metrics.items()
                        if name.endswith(".calls") and m.get("type") == "counter"
                    ),
                    "total_errors": sum(
                        m.get("current_value", 0) for name, m in all_metrics.items()
                        if name.endswith(".errors") and m.get("type") == "counter"
                    )
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting dashboard summary: {e}")
            return {"error": str(e)}
    
    async def _get_metrics_data(self) -> Dict[str, Any]:
        """Get metrics data for dashboard charts."""
        try:
            all_metrics = monitoring.metrics.get_all_metrics()
            
            # Organize metrics by category
            system_metrics = {}
            performance_metrics = {}
            application_metrics = {}
            
            for name, data in all_metrics.items():
                if name.startswith("system."):
                    system_metrics[name] = data
                elif any(keyword in name for keyword in ["duration", "latency", "calls", "errors"]):
                    performance_metrics[name] = data
                else:
                    application_metrics[name] = data
            
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "system_metrics": system_metrics,
                "performance_metrics": performance_metrics,
                "application_metrics": application_metrics
            }
            
        except Exception as e:
            logger.error(f"Error getting metrics data: {e}")
            return {"error": str(e)}
    
    async def _get_weather_data(self) -> Dict[str, Any]:
        """Get weather data for dashboard."""
        try:
            # Sample locations for dashboard
            sample_locations = [
                {"name": "San Francisco", "lat": 37.7749, "lon": -122.4194},
                {"name": "New York", "lat": 40.7128, "lon": -74.0060},
                {"name": "London", "lat": 51.5074, "lon": -0.1278},
                {"name": "Tokyo", "lat": 35.6762, "lon": 139.6503}
            ]
            
            weather_data = []
            
            for loc in sample_locations:
                try:
                    location = Location(
                        latitude=loc["lat"], 
                        longitude=loc["lon"], 
                        name=loc["name"]
                    )
                    
                    current_weather = await weather_service.get_current_weather(location)
                    
                    weather_data.append({
                        "location": loc["name"],
                        "latitude": loc["lat"],
                        "longitude": loc["lon"],
                        "temperature": current_weather.temperature,
                        "humidity": current_weather.humidity,
                        "wind_speed": current_weather.wind_speed,
                        "pressure": current_weather.pressure,
                        "timestamp": current_weather.timestamp.isoformat()
                    })
                    
                except Exception as e:
                    logger.warning(f"Error getting weather for {loc['name']}: {e}")
                    continue
            
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "locations": weather_data
            }
            
        except Exception as e:
            logger.error(f"Error getting weather data: {e}")
            return {"error": str(e)}
    
    async def _get_analytics_data(self) -> Dict[str, Any]:
        """Get analytics data for dashboard."""
        try:
            # Get analytics service statistics
            analytics_stats = {
                "analyses_completed": 0,
                "predictions_generated": 0,
                "anomalies_detected": 0,
                "models_active": len(analytics_service._models)
            }
            
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "statistics": analytics_stats
            }
            
        except Exception as e:
            logger.error(f"Error getting analytics data: {e}")
            return {"error": str(e)}
    
    async def _create_weather_chart(self, latitude: float, longitude: float, days: int) -> Dict[str, Any]:
        """Create weather forecast chart."""
        try:
            location = Location(latitude=latitude, longitude=longitude)
            forecast_data = await weather_service.get_forecast(location, days)
            
            # Extract data for chart
            dates = []
            temperatures = []
            humidity = []
            wind_speeds = []
            
            for forecast in forecast_data.forecasts:
                dates.append(forecast.timestamp)
                temperatures.append(forecast.temperature)
                humidity.append(forecast.humidity)
                wind_speeds.append(forecast.wind_speed)
            
            # Create Plotly chart
            fig = go.Figure()
            
            # Temperature line
            fig.add_trace(go.Scatter(
                x=dates,
                y=temperatures,
                mode='lines+markers',
                name='Temperature (°C)',
                line=dict(color='red'),
                yaxis='y'
            ))
            
            # Humidity line (secondary y-axis)
            fig.add_trace(go.Scatter(
                x=dates,
                y=humidity,
                mode='lines+markers',
                name='Humidity (%)',
                line=dict(color='blue'),
                yaxis='y2'
            ))
            
            # Wind speed bars
            fig.add_trace(go.Bar(
                x=dates,
                y=wind_speeds,
                name='Wind Speed (m/s)',
                opacity=0.6,
                yaxis='y3'
            ))
            
            fig.update_layout(
                title=f'Weather Forecast for {latitude:.2f}, {longitude:.2f}',
                xaxis_title='Date',
                yaxis=dict(title='Temperature (°C)', side='left'),
                yaxis2=dict(title='Humidity (%)', side='right', overlaying='y'),
                yaxis3=dict(title='Wind Speed (m/s)', side='right', overlaying='y', position=0.85),
                hovermode='x unified'
            )
            
            return {
                "chart": fig.to_json(),
                "data_points": len(dates),
                "location": {"latitude": latitude, "longitude": longitude}
            }
            
        except Exception as e:
            logger.error(f"Error creating weather chart: {e}")
            return {"error": str(e)}
    
    async def _create_system_metrics_chart(self) -> Dict[str, Any]:
        """Create system metrics chart."""
        try:
            all_metrics = monitoring.metrics.get_all_metrics()
            
            # Get system metrics
            cpu_data = all_metrics.get("system.cpu.usage_percent", {})
            memory_data = all_metrics.get("system.memory.usage_percent", {})
            disk_data = all_metrics.get("system.disk.usage_percent", {})
            
            # Create gauge charts for system resources
            fig = go.Figure()
            
            # CPU gauge
            fig.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=cpu_data.get("current_value", 0),
                domain={'x': [0, 0.33], 'y': [0, 1]},
                title={'text': "CPU Usage (%)"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkblue"},
                       'steps': [
                           {'range': [0, 50], 'color': "lightgray"},
                           {'range': [50, 80], 'color': "yellow"},
                           {'range': [80, 100], 'color': "red"}
                       ],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75, 'value': 90}}
            ))
            
            # Memory gauge
            fig.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=memory_data.get("current_value", 0),
                domain={'x': [0.33, 0.66], 'y': [0, 1]},
                title={'text': "Memory Usage (%)"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkgreen"},
                       'steps': [
                           {'range': [0, 50], 'color': "lightgray"},
                           {'range': [50, 80], 'color': "yellow"},
                           {'range': [80, 100], 'color': "red"}
                       ]}
            ))
            
            # Disk gauge
            fig.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=disk_data.get("current_value", 0),
                domain={'x': [0.66, 1], 'y': [0, 1]},
                title={'text': "Disk Usage (%)"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkorange"},
                       'steps': [
                           {'range': [0, 70], 'color': "lightgray"},
                           {'range': [70, 90], 'color': "yellow"},
                           {'range': [90, 100], 'color': "red"}
                       ]}
            ))
            
            fig.update_layout(title="System Resource Usage")
            
            return {
                "chart": fig.to_json(),
                "metrics": {
                    "cpu": cpu_data.get("current_value", 0),
                    "memory": memory_data.get("current_value", 0),
                    "disk": disk_data.get("current_value", 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating system metrics chart: {e}")
            return {"error": str(e)}
    
    async def _get_realtime_data(self) -> Dict[str, Any]:
        """Get real-time data for WebSocket updates."""
        try:
            dashboard_summary = await self._get_dashboard_summary()
            
            return {
                "type": "realtime_update",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": dashboard_summary
            }
            
        except Exception as e:
            logger.error(f"Error getting real-time data: {e}")
            return {
                "type": "error",
                "message": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    def _start_dashboard_tasks(self):
        """Start background tasks for dashboard."""
        
        async def broadcast_updates():
            """Broadcast updates to all connected WebSocket clients."""
            while True:
                try:
                    if self.websocket_connections:
                        data = await self._get_realtime_data()
                        
                        # Send to all connected clients
                        disconnected = []
                        for websocket in self.websocket_connections:
                            try:
                                await websocket.send_json(data)
                            except Exception:
                                disconnected.append(websocket)
                        
                        # Remove disconnected clients
                        for ws in disconnected:
                            if ws in self.websocket_connections:
                                self.websocket_connections.remove(ws)
                    
                    await asyncio.sleep(10)  # Broadcast every 10 seconds
                    
                except Exception as e:
                    logger.error(f"Error in broadcast task: {e}")
                    await asyncio.sleep(30)
        
        # Start background task
        asyncio.create_task(broadcast_updates())


# Create global dashboard instance
dashboard = DashboardManager()

# FastAPI app for dashboard
dashboard_app = dashboard.app 