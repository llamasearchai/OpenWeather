"""Comprehensive monitoring and observability system for OpenWeather platform."""

import asyncio
import logging
import time
import psutil
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
from collections import defaultdict, deque
import json

from openweather.core.config import settings


class MetricType(str, Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertLevel(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Metric:
    """Represents a single metric data point."""
    name: str
    value: Union[int, float]
    type: MetricType
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: Dict[str, str] = field(default_factory=dict)
    help_text: str = ""


@dataclass
class Alert:
    """Represents an alert condition."""
    id: str
    level: AlertLevel
    message: str
    metric_name: str
    threshold: float
    current_value: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None


@dataclass
class HealthCheck:
    """Represents a health check result."""
    name: str
    status: str  # "healthy", "degraded", "unhealthy"
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Collects and manages application metrics."""
    
    def __init__(self):
        self._metrics: Dict[str, List[Metric]] = defaultdict(list)
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = defaultdict(float)
        self._histograms: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._timers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._lock = threading.Lock()
        
        # Start system metrics collection
        self._system_metrics_task = None
        self._start_system_metrics_collection()
    
    def increment_counter(self, name: str, value: float = 1.0, tags: Dict[str, str] = None) -> None:
        """Increment a counter metric."""
        with self._lock:
            self._counters[name] += value
            metric = Metric(
                name=name,
                value=self._counters[name],
                type=MetricType.COUNTER,
                tags=tags or {}
            )
            self._metrics[name].append(metric)
    
    def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None) -> None:
        """Set a gauge metric value."""
        with self._lock:
            self._gauges[name] = value
            metric = Metric(
                name=name,
                value=value,
                type=MetricType.GAUGE,
                tags=tags or {}
            )
            self._metrics[name].append(metric)
    
    def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None) -> None:
        """Record a value in a histogram."""
        with self._lock:
            self._histograms[name].append(value)
            metric = Metric(
                name=name,
                value=value,
                type=MetricType.HISTOGRAM,
                tags=tags or {}
            )
            self._metrics[name].append(metric)
    
    def record_timer(self, name: str, duration_ms: float, tags: Dict[str, str] = None) -> None:
        """Record a timing measurement."""
        with self._lock:
            self._timers[name].append(duration_ms)
            metric = Metric(
                name=name,
                value=duration_ms,
                type=MetricType.TIMER,
                tags=tags or {}
            )
            self._metrics[name].append(metric)
    
    @asynccontextmanager
    async def timer(self, name: str, tags: Dict[str, str] = None):
        """Context manager for timing operations."""
        start_time = time.time()
        try:
            yield
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.record_timer(name, duration_ms, tags)
    
    def get_metric_summary(self, name: str) -> Dict[str, Any]:
        """Get summary statistics for a metric."""
        with self._lock:
            if name not in self._metrics:
                return {}
            
            metrics = self._metrics[name]
            if not metrics:
                return {}
            
            latest_metric = metrics[-1]
            
            if latest_metric.type == MetricType.COUNTER:
                return {
                    "type": "counter",
                    "current_value": self._counters[name],
                    "total_increments": len(metrics)
                }
            elif latest_metric.type == MetricType.GAUGE:
                return {
                    "type": "gauge",
                    "current_value": self._gauges[name],
                    "last_updated": latest_metric.timestamp.isoformat()
                }
            elif latest_metric.type in [MetricType.HISTOGRAM, MetricType.TIMER]:
                values = list(self._histograms[name] if latest_metric.type == MetricType.HISTOGRAM else self._timers[name])
                if values:
                    return {
                        "type": latest_metric.type.value,
                        "count": len(values),
                        "min": min(values),
                        "max": max(values),
                        "mean": sum(values) / len(values),
                        "p50": self._percentile(values, 50),
                        "p95": self._percentile(values, 95),
                        "p99": self._percentile(values, 99)
                    }
            
            return {}
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metrics."""
        result = {}
        for name in self._metrics.keys():
            result[name] = self.get_metric_summary(name)
        return result
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile value."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def _start_system_metrics_collection(self) -> None:
        """Start collecting system metrics in background."""
        def collect_system_metrics():
            while True:
                try:
                    # CPU usage
                    cpu_percent = psutil.cpu_percent(interval=1)
                    self.set_gauge("system.cpu.usage_percent", cpu_percent)
                    
                    # Memory usage
                    memory = psutil.virtual_memory()
                    self.set_gauge("system.memory.usage_percent", memory.percent)
                    self.set_gauge("system.memory.available_bytes", memory.available)
                    self.set_gauge("system.memory.used_bytes", memory.used)
                    
                    # Disk usage
                    disk = psutil.disk_usage('/')
                    self.set_gauge("system.disk.usage_percent", (disk.used / disk.total) * 100)
                    self.set_gauge("system.disk.free_bytes", disk.free)
                    
                    # Network I/O
                    net_io = psutil.net_io_counters()
                    self.increment_counter("system.network.bytes_sent", net_io.bytes_sent - getattr(self, '_last_bytes_sent', 0))
                    self.increment_counter("system.network.bytes_recv", net_io.bytes_recv - getattr(self, '_last_bytes_recv', 0))
                    self._last_bytes_sent = net_io.bytes_sent
                    self._last_bytes_recv = net_io.bytes_recv
                    
                    # Process info
                    process = psutil.Process()
                    self.set_gauge("process.memory.rss_bytes", process.memory_info().rss)
                    self.set_gauge("process.cpu.usage_percent", process.cpu_percent())
                    self.set_gauge("process.threads.count", process.num_threads())
                    
                except Exception as e:
                    logging.error(f"Error collecting system metrics: {e}")
                
                time.sleep(10)  # Collect every 10 seconds
        
        thread = threading.Thread(target=collect_system_metrics, daemon=True)
        thread.start()


class AlertManager:
    """Manages alerts and notifications."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self._alert_rules: Dict[str, Dict[str, Any]] = {}
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: List[Alert] = []
        self._notification_callbacks: List[Callable[[Alert], None]] = []
        
        # Default alert rules
        self._setup_default_alerts()
    
    def add_alert_rule(self, name: str, metric_name: str, threshold: float, 
                      level: AlertLevel, condition: str = "gt", 
                      window_minutes: int = 5) -> None:
        """Add an alert rule."""
        self._alert_rules[name] = {
            "metric_name": metric_name,
            "threshold": threshold,
            "level": level,
            "condition": condition,  # "gt", "lt", "eq"
            "window_minutes": window_minutes
        }
    
    def add_notification_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add a callback for alert notifications."""
        self._notification_callbacks.append(callback)
    
    def check_alerts(self) -> List[Alert]:
        """Check all alert rules and return any new alerts."""
        new_alerts = []
        
        for alert_name, rule in self._alert_rules.items():
            metric_summary = self.metrics_collector.get_metric_summary(rule["metric_name"])
            
            if not metric_summary:
                continue
            
            current_value = metric_summary.get("current_value", 0)
            threshold = rule["threshold"]
            condition = rule["condition"]
            
            # Check if alert condition is met
            alert_triggered = False
            if condition == "gt" and current_value > threshold:
                alert_triggered = True
            elif condition == "lt" and current_value < threshold:
                alert_triggered = True
            elif condition == "eq" and current_value == threshold:
                alert_triggered = True
            
            alert_id = f"{alert_name}_{rule['metric_name']}"
            
            if alert_triggered:
                if alert_id not in self._active_alerts:
                    # New alert
                    alert = Alert(
                        id=alert_id,
                        level=rule["level"],
                        message=f"Alert: {alert_name} - {rule['metric_name']} {condition} {threshold}",
                        metric_name=rule["metric_name"],
                        threshold=threshold,
                        current_value=current_value
                    )
                    
                    self._active_alerts[alert_id] = alert
                    self._alert_history.append(alert)
                    new_alerts.append(alert)
                    
                    # Send notifications
                    for callback in self._notification_callbacks:
                        try:
                            callback(alert)
                        except Exception as e:
                            logging.error(f"Error sending alert notification: {e}")
            else:
                # Check if we should resolve an active alert
                if alert_id in self._active_alerts:
                    alert = self._active_alerts[alert_id]
                    alert.resolved = True
                    alert.resolution_timestamp = datetime.now(timezone.utc)
                    del self._active_alerts[alert_id]
        
        return new_alerts
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self._active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for the specified number of hours."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        return [alert for alert in self._alert_history if alert.timestamp >= cutoff_time]
    
    def _setup_default_alerts(self) -> None:
        """Setup default alert rules."""
        # System resource alerts
        self.add_alert_rule(
            "high_cpu_usage",
            "system.cpu.usage_percent",
            80.0,
            AlertLevel.WARNING,
            "gt"
        )
        
        self.add_alert_rule(
            "critical_cpu_usage",
            "system.cpu.usage_percent",
            95.0,
            AlertLevel.CRITICAL,
            "gt"
        )
        
        self.add_alert_rule(
            "high_memory_usage",
            "system.memory.usage_percent",
            85.0,
            AlertLevel.WARNING,
            "gt"
        )
        
        self.add_alert_rule(
            "critical_memory_usage",
            "system.memory.usage_percent",
            95.0,
            AlertLevel.CRITICAL,
            "gt"
        )
        
        self.add_alert_rule(
            "high_disk_usage",
            "system.disk.usage_percent",
            90.0,
            AlertLevel.WARNING,
            "gt"
        )


class HealthCheckManager:
    """Manages application health checks."""
    
    def __init__(self):
        self._health_checks: Dict[str, Callable[[], HealthCheck]] = {}
        self._last_results: Dict[str, HealthCheck] = {}
    
    def register_health_check(self, name: str, check_func: Callable[[], HealthCheck]) -> None:
        """Register a health check function."""
        self._health_checks[name] = check_func
    
    async def run_health_checks(self) -> Dict[str, HealthCheck]:
        """Run all registered health checks."""
        results = {}
        
        for name, check_func in self._health_checks.items():
            start_time = time.time()
            try:
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()
                result.duration_ms = (time.time() - start_time) * 1000
                results[name] = result
                self._last_results[name] = result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                result = HealthCheck(
                    name=name,
                    status="unhealthy",
                    message=f"Health check failed: {str(e)}",
                    duration_ms=duration_ms
                )
                results[name] = result
                self._last_results[name] = result
        
        return results
    
    def get_overall_health(self) -> str:
        """Get overall system health status."""
        if not self._last_results:
            return "unknown"
        
        statuses = [check.status for check in self._last_results.values()]
        
        if any(status == "unhealthy" for status in statuses):
            return "unhealthy"
        elif any(status == "degraded" for status in statuses):
            return "degraded"
        else:
            return "healthy"
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health check summary."""
        return {
            "overall_status": self.get_overall_health(),
            "last_check": datetime.now(timezone.utc).isoformat(),
            "checks": {name: {
                "status": check.status,
                "message": check.message,
                "duration_ms": check.duration_ms,
                "last_check": check.timestamp.isoformat()
            } for name, check in self._last_results.items()}
        }


class MonitoringManager:
    """Main monitoring manager that coordinates all monitoring components."""
    
    def __init__(self):
        self.metrics = MetricsCollector()
        self.alerts = AlertManager(self.metrics)
        self.health = HealthCheckManager()
        
        # Setup default health checks
        self._setup_default_health_checks()
        
        # Start monitoring tasks
        self._monitoring_task = None
        self._start_monitoring_loop()
    
    def _setup_default_health_checks(self) -> None:
        """Setup default health checks."""
        
        def database_health_check() -> HealthCheck:
            """Check database connectivity."""
            try:
                # TODO: Implement actual database check
                return HealthCheck(
                    name="database",
                    status="healthy",
                    message="Database connection is healthy"
                )
            except Exception as e:
                return HealthCheck(
                    name="database",
                    status="unhealthy",
                    message=f"Database connection failed: {str(e)}"
                )
        
        def memory_health_check() -> HealthCheck:
            """Check memory usage."""
            memory = psutil.virtual_memory()
            if memory.percent > 95:
                status = "unhealthy"
                message = f"Critical memory usage: {memory.percent:.1f}%"
            elif memory.percent > 85:
                status = "degraded"
                message = f"High memory usage: {memory.percent:.1f}%"
            else:
                status = "healthy"
                message = f"Memory usage normal: {memory.percent:.1f}%"
            
            return HealthCheck(
                name="memory",
                status=status,
                message=message,
                metadata={"usage_percent": memory.percent}
            )
        
        def disk_health_check() -> HealthCheck:
            """Check disk usage."""
            disk = psutil.disk_usage('/')
            usage_percent = (disk.used / disk.total) * 100
            
            if usage_percent > 95:
                status = "unhealthy"
                message = f"Critical disk usage: {usage_percent:.1f}%"
            elif usage_percent > 90:
                status = "degraded"
                message = f"High disk usage: {usage_percent:.1f}%"
            else:
                status = "healthy"
                message = f"Disk usage normal: {usage_percent:.1f}%"
            
            return HealthCheck(
                name="disk",
                status=status,
                message=message,
                metadata={"usage_percent": usage_percent}
            )
        
        self.health.register_health_check("database", database_health_check)
        self.health.register_health_check("memory", memory_health_check)
        self.health.register_health_check("disk", disk_health_check)
    
    def _start_monitoring_loop(self) -> None:
        """Start the main monitoring loop."""
        async def monitoring_loop():
            while True:
                try:
                    # Check alerts
                    new_alerts = self.alerts.check_alerts()
                    if new_alerts:
                        logging.info(f"Generated {len(new_alerts)} new alerts")
                    
                    # Run health checks
                    await self.health.run_health_checks()
                    
                except Exception as e:
                    logging.error(f"Error in monitoring loop: {e}")
                
                await asyncio.sleep(30)  # Check every 30 seconds
        
        def start_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(monitoring_loop())
        
        thread = threading.Thread(target=start_loop, daemon=True)
        thread.start()
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data."""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "health": self.health.get_health_summary(),
            "metrics": self.metrics.get_all_metrics(),
            "active_alerts": [
                {
                    "id": alert.id,
                    "level": alert.level.value,
                    "message": alert.message,
                    "metric_name": alert.metric_name,
                    "current_value": alert.current_value,
                    "threshold": alert.threshold,
                    "timestamp": alert.timestamp.isoformat()
                }
                for alert in self.alerts.get_active_alerts()
            ],
            "alert_history": [
                {
                    "id": alert.id,
                    "level": alert.level.value,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                    "resolved": alert.resolved,
                    "resolution_timestamp": alert.resolution_timestamp.isoformat() if alert.resolution_timestamp else None
                }
                for alert in self.alerts.get_alert_history(hours=24)
            ]
        }
    
    def add_custom_metric(self, name: str, value: float, metric_type: MetricType, tags: Dict[str, str] = None) -> None:
        """Add a custom metric."""
        if metric_type == MetricType.COUNTER:
            self.metrics.increment_counter(name, value, tags)
        elif metric_type == MetricType.GAUGE:
            self.metrics.set_gauge(name, value, tags)
        elif metric_type == MetricType.HISTOGRAM:
            self.metrics.record_histogram(name, value, tags)
        elif metric_type == MetricType.TIMER:
            self.metrics.record_timer(name, value, tags)


# Global monitoring instance
monitoring = MonitoringManager()


# Decorators for automatic monitoring
def monitor_performance(metric_name: str = None):
    """Decorator to monitor function performance."""
    def decorator(func):
        name = metric_name or f"{func.__module__}.{func.__name__}"
        
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                async with monitoring.metrics.timer(f"{name}.duration"):
                    monitoring.metrics.increment_counter(f"{name}.calls")
                    try:
                        result = await func(*args, **kwargs)
                        monitoring.metrics.increment_counter(f"{name}.success")
                        return result
                    except Exception as e:
                        monitoring.metrics.increment_counter(f"{name}.errors")
                        raise
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                monitoring.metrics.increment_counter(f"{name}.calls")
                try:
                    result = func(*args, **kwargs)
                    monitoring.metrics.increment_counter(f"{name}.success")
                    duration_ms = (time.time() - start_time) * 1000
                    monitoring.metrics.record_timer(f"{name}.duration", duration_ms)
                    return result
                except Exception as e:
                    monitoring.metrics.increment_counter(f"{name}.errors")
                    duration_ms = (time.time() - start_time) * 1000
                    monitoring.metrics.record_timer(f"{name}.duration", duration_ms)
                    raise
            return sync_wrapper
    return decorator


def log_alert_notification(alert: Alert) -> None:
    """Default alert notification handler that logs alerts."""
    logging.warning(f"ALERT [{alert.level.value.upper()}]: {alert.message}")


# Setup default alert notification
monitoring.alerts.add_notification_callback(log_alert_notification) 