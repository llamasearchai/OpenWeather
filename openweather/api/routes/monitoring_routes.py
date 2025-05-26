"""API routes for monitoring and observability."""

from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from openweather.core.monitoring import monitoring, MetricType, AlertLevel
from openweather.core.auth import get_current_user  # Optional authentication


router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get("/health")
async def get_health_status() -> Dict[str, Any]:
    """Get overall system health status."""
    health_results = await monitoring.health.run_health_checks()
    overall_health = monitoring.health.get_overall_health()
    
    return {
        "status": overall_health,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": {
            name: {
                "status": check.status,
                "message": check.message,
                "duration_ms": check.duration_ms,
                "timestamp": check.timestamp.isoformat(),
                "metadata": check.metadata
            }
            for name, check in health_results.items()
        }
    }


@router.get("/health/{check_name}")
async def get_specific_health_check(check_name: str) -> Dict[str, Any]:
    """Get health status for a specific check."""
    health_results = await monitoring.health.run_health_checks()
    
    if check_name not in health_results:
        raise HTTPException(status_code=404, detail=f"Health check '{check_name}' not found")
    
    check = health_results[check_name]
    return {
        "name": check_name,
        "status": check.status,
        "message": check.message,
        "duration_ms": check.duration_ms,
        "timestamp": check.timestamp.isoformat(),
        "metadata": check.metadata
    }


@router.get("/metrics")
async def get_all_metrics() -> Dict[str, Any]:
    """Get all current metrics."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": monitoring.metrics.get_all_metrics()
    }


@router.get("/metrics/{metric_name}")
async def get_specific_metric(metric_name: str) -> Dict[str, Any]:
    """Get details for a specific metric."""
    metric_summary = monitoring.metrics.get_metric_summary(metric_name)
    
    if not metric_summary:
        raise HTTPException(status_code=404, detail=f"Metric '{metric_name}' not found")
    
    return {
        "name": metric_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **metric_summary
    }


@router.post("/metrics/{metric_name}")
async def add_custom_metric(
    metric_name: str,
    value: float,
    metric_type: MetricType,
    tags: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """Add a custom metric value."""
    try:
        monitoring.add_custom_metric(metric_name, value, metric_type, tags)
        return {
            "message": f"Metric '{metric_name}' recorded successfully",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to record metric: {str(e)}")


@router.get("/alerts")
async def get_active_alerts() -> Dict[str, Any]:
    """Get all active alerts."""
    active_alerts = monitoring.alerts.get_active_alerts()
    
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "count": len(active_alerts),
        "alerts": [
            {
                "id": alert.id,
                "level": alert.level.value,
                "message": alert.message,
                "metric_name": alert.metric_name,
                "current_value": alert.current_value,
                "threshold": alert.threshold,
                "timestamp": alert.timestamp.isoformat()
            }
            for alert in active_alerts
        ]
    }


@router.get("/alerts/history")
async def get_alert_history(
    hours: int = Query(24, description="Number of hours to look back", ge=1, le=168)
) -> Dict[str, Any]:
    """Get alert history for the specified time period."""
    alert_history = monitoring.alerts.get_alert_history(hours=hours)
    
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "period_hours": hours,
        "count": len(alert_history),
        "alerts": [
            {
                "id": alert.id,
                "level": alert.level.value,
                "message": alert.message,
                "metric_name": alert.metric_name,
                "current_value": alert.current_value,
                "threshold": alert.threshold,
                "timestamp": alert.timestamp.isoformat(),
                "resolved": alert.resolved,
                "resolution_timestamp": alert.resolution_timestamp.isoformat() if alert.resolution_timestamp else None
            }
            for alert in alert_history
        ]
    }


@router.post("/alerts/rules")
async def add_alert_rule(
    name: str,
    metric_name: str,
    threshold: float,
    level: AlertLevel,
    condition: str = "gt",
    window_minutes: int = 5
) -> Dict[str, str]:
    """Add a new alert rule."""
    try:
        monitoring.alerts.add_alert_rule(
            name=name,
            metric_name=metric_name,
            threshold=threshold,
            level=level,
            condition=condition,
            window_minutes=window_minutes
        )
        return {
            "message": f"Alert rule '{name}' added successfully",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to add alert rule: {str(e)}")


@router.get("/dashboard")
async def get_monitoring_dashboard() -> Dict[str, Any]:
    """Get comprehensive monitoring dashboard data."""
    return monitoring.get_monitoring_dashboard()


@router.get("/system")
async def get_system_metrics() -> Dict[str, Any]:
    """Get current system metrics (CPU, memory, disk, network)."""
    all_metrics = monitoring.metrics.get_all_metrics()
    
    # Filter system metrics
    system_metrics = {
        name: metrics for name, metrics in all_metrics.items()
        if name.startswith("system.") or name.startswith("process.")
    }
    
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system_metrics": system_metrics
    }


@router.get("/performance")
async def get_performance_metrics() -> Dict[str, Any]:
    """Get application performance metrics."""
    all_metrics = monitoring.metrics.get_all_metrics()
    
    # Filter performance-related metrics
    performance_metrics = {}
    for name, metrics in all_metrics.items():
        if any(keyword in name for keyword in ["duration", "latency", "response_time", "calls", "errors", "success"]):
            performance_metrics[name] = metrics
    
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "performance_metrics": performance_metrics
    }


@router.get("/status")
async def get_service_status() -> Dict[str, Any]:
    """Get overall service status summary."""
    health_summary = monitoring.health.get_health_summary()
    active_alerts = monitoring.alerts.get_active_alerts()
    all_metrics = monitoring.metrics.get_all_metrics()
    
    # Calculate uptime (simplified - would need actual start time tracking)
    import psutil
    boot_time = datetime.fromtimestamp(psutil.boot_time(), timezone.utc)
    uptime_seconds = (datetime.now(timezone.utc) - boot_time).total_seconds()
    
    # Service level indicators
    error_rate = 0.0
    avg_response_time = 0.0
    
    # Calculate error rate and response time from metrics
    total_calls = 0
    total_errors = 0
    total_duration = 0.0
    duration_count = 0
    
    for name, metric_data in all_metrics.items():
        if name.endswith(".calls") and metric_data.get("type") == "counter":
            total_calls += metric_data.get("current_value", 0)
        elif name.endswith(".errors") and metric_data.get("type") == "counter":
            total_errors += metric_data.get("current_value", 0)
        elif name.endswith(".duration") and metric_data.get("type") == "timer":
            avg_response_time += metric_data.get("mean", 0)
            duration_count += 1
    
    if total_calls > 0:
        error_rate = (total_errors / total_calls) * 100
    
    if duration_count > 0:
        avg_response_time = avg_response_time / duration_count
    
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_status": health_summary["overall_status"],
        "uptime_seconds": int(uptime_seconds),
        "active_alerts_count": len(active_alerts),
        "critical_alerts_count": len([a for a in active_alerts if a.level == AlertLevel.CRITICAL]),
        "service_level_indicators": {
            "error_rate_percent": round(error_rate, 2),
            "avg_response_time_ms": round(avg_response_time, 2),
            "total_requests": int(total_calls),
            "total_errors": int(total_errors)
        },
        "health_checks": health_summary["checks"]
    }


@router.post("/health/register")
async def register_health_check(
    name: str,
    endpoint: str,
    timeout_seconds: int = 30
) -> Dict[str, str]:
    """Register a new health check endpoint."""
    # This would typically register a health check that makes HTTP calls
    # For now, we'll return a placeholder response
    return {
        "message": f"Health check '{name}' registered for endpoint '{endpoint}'",
        "timeout_seconds": timeout_seconds,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@router.delete("/alerts/{alert_id}")
async def acknowledge_alert(alert_id: str) -> Dict[str, str]:
    """Acknowledge/dismiss an active alert."""
    active_alerts = monitoring.alerts._active_alerts
    
    if alert_id not in active_alerts:
        raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found")
    
    # Mark alert as resolved
    alert = active_alerts[alert_id]
    alert.resolved = True
    alert.resolution_timestamp = datetime.now(timezone.utc)
    del active_alerts[alert_id]
    
    return {
        "message": f"Alert '{alert_id}' acknowledged and resolved",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@router.get("/logs")
async def get_recent_logs(
    level: str = Query("INFO", description="Minimum log level"),
    lines: int = Query(100, description="Number of lines to retrieve", ge=1, le=1000)
) -> Dict[str, Any]:
    """Get recent application logs."""
    # This would typically read from log files or log aggregation system
    # For now, return a placeholder response
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "lines_requested": lines,
        "logs": [
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": "INFO",
                "message": "Sample log entry",
                "module": "openweather.api"
            }
        ],
        "note": "Log aggregation not implemented - this is a placeholder"
    }


@router.get("/export")
async def export_monitoring_data(
    format: str = Query("json", description="Export format: json, csv, prometheus"),
    hours: int = Query(1, description="Hours of data to export", ge=1, le=24)
) -> JSONResponse:
    """Export monitoring data in various formats."""
    if format not in ["json", "csv", "prometheus"]:
        raise HTTPException(status_code=400, detail="Unsupported export format")
    
    # Get comprehensive monitoring data
    dashboard_data = monitoring.get_monitoring_dashboard()
    
    if format == "json":
        return JSONResponse(content={
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "format": format,
            "period_hours": hours,
            "data": dashboard_data
        })
    elif format == "prometheus":
        # Convert metrics to Prometheus format
        prometheus_metrics = []
        for name, metric_data in dashboard_data["metrics"].items():
            if metric_data.get("type") == "gauge":
                prometheus_metrics.append(f'{name.replace(".", "_")} {metric_data.get("current_value", 0)}')
            elif metric_data.get("type") == "counter":
                prometheus_metrics.append(f'{name.replace(".", "_")}_total {metric_data.get("current_value", 0)}')
        
        prometheus_text = "\n".join(prometheus_metrics)
        return JSONResponse(content={
            "format": "prometheus",
            "metrics": prometheus_text
        })
    else:  # CSV format
        # This would generate CSV data
        return JSONResponse(content={
            "format": "csv",
            "message": "CSV export not fully implemented",
            "data": dashboard_data
        }) 