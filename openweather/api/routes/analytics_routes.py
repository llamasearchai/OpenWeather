"""API routes for weather analytics and machine learning services."""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Query, Path, Body, BackgroundTasks
from pydantic import BaseModel, Field, validator

from openweather.core.monitoring import MetricsCollector, performance_monitor
from openweather.models.weather import Location
from openweather.services.analytics_service import (
    analytics_service, 
    AnalysisType, 
    PredictionModel,
    TrendAnalysis,
    AnomalyDetection,
    WeatherPrediction,
    ClimateAnalysis
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analytics", tags=["Analytics"])
metrics = MetricsCollector()

# Request/Response Models
class LocationRequest(BaseModel):
    """Location request model."""
    latitude: float = Field(..., ge=-90, le=90, description="Latitude in degrees")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude in degrees")
    name: Optional[str] = Field(None, description="Location name")

class TrendAnalysisRequest(BaseModel):
    """Trend analysis request model."""
    location: LocationRequest
    parameter: str = Field(..., description="Weather parameter to analyze")
    start_date: datetime = Field(..., description="Start date for analysis")
    end_date: datetime = Field(..., description="End date for analysis")
    
    @validator('parameter')
    def validate_parameter(cls, v):
        valid_params = ['temperature', 'humidity', 'pressure', 'wind_speed', 'visibility']
        if v not in valid_params:
            raise ValueError(f"Parameter must be one of: {valid_params}")
        return v
    
    @validator('end_date')
    def validate_date_range(cls, v, values):
        if 'start_date' in values and v <= values['start_date']:
            raise ValueError("End date must be after start date")
        if v > datetime.utcnow():
            raise ValueError("End date cannot be in the future")
        return v

class AnomalyDetectionRequest(BaseModel):
    """Anomaly detection request model."""
    location: LocationRequest
    parameter: str = Field(..., description="Weather parameter to analyze")
    lookback_days: int = Field(30, ge=7, le=365, description="Number of days to look back")
    
    @validator('parameter')
    def validate_parameter(cls, v):
        valid_params = ['temperature', 'humidity', 'pressure', 'wind_speed', 'visibility']
        if v not in valid_params:
            raise ValueError(f"Parameter must be one of: {valid_params}")
        return v

class PredictionRequest(BaseModel):
    """Weather prediction request model."""
    location: LocationRequest
    forecast_hours: int = Field(24, ge=1, le=168, description="Forecast horizon in hours")
    model_type: PredictionModel = Field(PredictionModel.RANDOM_FOREST, description="ML model to use")

class CorrelationAnalysisRequest(BaseModel):
    """Correlation analysis request model."""
    location: LocationRequest
    parameters: List[str] = Field(..., min_items=2, description="Parameters to analyze")
    days: int = Field(30, ge=7, le=365, description="Number of days for analysis")
    
    @validator('parameters')
    def validate_parameters(cls, v):
        valid_params = {'temperature', 'humidity', 'pressure', 'wind_speed', 'visibility'}
        invalid_params = set(v) - valid_params
        if invalid_params:
            raise ValueError(f"Invalid parameters: {invalid_params}")
        if len(set(v)) != len(v):
            raise ValueError("Duplicate parameters not allowed")
        return v

class ClimateReportRequest(BaseModel):
    """Climate report request model."""
    location: LocationRequest
    years: int = Field(5, ge=1, le=10, description="Number of years for climate analysis")

# Response Models
class TrendAnalysisResponse(BaseModel):
    """Trend analysis response model."""
    parameter: str
    trend_direction: str
    trend_strength: float
    confidence: float
    period_start: datetime
    period_end: datetime
    seasonal_pattern: Optional[Dict[str, float]]
    statistical_significance: Optional[float]

class AnomalyDetectionResponse(BaseModel):
    """Anomaly detection response model."""
    timestamp: datetime
    parameter: str
    value: float
    expected_range: tuple
    anomaly_score: float
    severity: str
    description: str

class WeatherPredictionResponse(BaseModel):
    """Weather prediction response model."""
    location: LocationRequest
    prediction_time: datetime
    forecast_horizon_hours: int
    predicted_values: Dict[str, float]
    confidence_intervals: Dict[str, tuple]
    model_used: PredictionModel
    accuracy_metrics: Dict[str, float]
    feature_importance: Optional[Dict[str, Dict[str, float]]]

class ClimateAnalysisResponse(BaseModel):
    """Climate analysis response model."""
    location: LocationRequest
    analysis_period: tuple
    temperature_statistics: Dict[str, float]
    precipitation_statistics: Dict[str, float]
    wind_statistics: Dict[str, float]
    climate_indicators: Dict[str, float]
    trend_count: int

class AnalyticsStatsResponse(BaseModel):
    """Analytics service statistics response."""
    total_analyses: int
    trend_analyses: int
    anomaly_detections: int
    predictions: int
    climate_reports: int
    active_models: int
    cache_stats: Dict[str, Any]

# API Endpoints
@router.get("/health", summary="Analytics service health check")
@performance_monitor
async def health_check():
    """Check analytics service health."""
    try:
        # Basic health check
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "models_loaded": len(analytics_service.models),
            "scalers_initialized": len(analytics_service.scalers)
        }
        
        metrics.increment_counter("analytics_health_checks")
        return health_status
        
    except Exception as e:
        logger.error(f"Analytics health check failed: {e}")
        raise HTTPException(status_code=503, detail="Analytics service unhealthy")

@router.post("/trends", response_model=TrendAnalysisResponse, 
             summary="Analyze weather parameter trends")
@performance_monitor
async def analyze_trends(request: TrendAnalysisRequest):
    """Analyze trends in weather parameters over time."""
    try:
        location = Location(
            latitude=request.location.latitude,
            longitude=request.location.longitude,
            name=request.location.name
        )
        
        trend_result = await analytics_service.analyze_trends(
            location=location,
            parameter=request.parameter,
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        metrics.increment_counter("trend_analyses_requested")
        
        return TrendAnalysisResponse(
            parameter=trend_result.parameter,
            trend_direction=trend_result.trend_direction,
            trend_strength=trend_result.trend_strength,
            confidence=trend_result.confidence,
            period_start=trend_result.period_start,
            period_end=trend_result.period_end,
            seasonal_pattern=trend_result.seasonal_pattern,
            statistical_significance=trend_result.statistical_significance
        )
        
    except ValueError as e:
        logger.warning(f"Invalid trend analysis request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Trend analysis failed: {e}")
        metrics.increment_counter("trend_analysis_errors")
        raise HTTPException(status_code=500, detail="Trend analysis failed")

@router.post("/anomalies", response_model=List[AnomalyDetectionResponse],
             summary="Detect weather anomalies")
@performance_monitor
async def detect_anomalies(request: AnomalyDetectionRequest):
    """Detect anomalies in weather data."""
    try:
        location = Location(
            latitude=request.location.latitude,
            longitude=request.location.longitude,
            name=request.location.name
        )
        
        anomalies = await analytics_service.detect_anomalies(
            location=location,
            parameter=request.parameter,
            lookback_days=request.lookback_days
        )
        
        metrics.increment_counter("anomaly_detections_requested")
        metrics.set_gauge("anomalies_found", len(anomalies))
        
        return [
            AnomalyDetectionResponse(
                timestamp=anomaly.timestamp,
                parameter=anomaly.parameter,
                value=anomaly.value,
                expected_range=anomaly.expected_range,
                anomaly_score=anomaly.anomaly_score,
                severity=anomaly.severity,
                description=anomaly.description
            )
            for anomaly in anomalies
        ]
        
    except ValueError as e:
        logger.warning(f"Invalid anomaly detection request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}")
        metrics.increment_counter("anomaly_detection_errors")
        raise HTTPException(status_code=500, detail="Anomaly detection failed")

@router.post("/predict", response_model=WeatherPredictionResponse,
             summary="Generate ML weather predictions")
@performance_monitor
async def predict_weather(request: PredictionRequest):
    """Generate weather predictions using machine learning."""
    try:
        location = Location(
            latitude=request.location.latitude,
            longitude=request.location.longitude,
            name=request.location.name
        )
        
        prediction = await analytics_service.predict_weather(
            location=location,
            forecast_hours=request.forecast_hours,
            model_type=request.model_type
        )
        
        metrics.increment_counter("weather_predictions_requested")
        
        return WeatherPredictionResponse(
            location=request.location,
            prediction_time=prediction.prediction_time,
            forecast_horizon_hours=request.forecast_hours,
            predicted_values=prediction.predicted_values,
            confidence_intervals=prediction.confidence_intervals,
            model_used=prediction.model_used,
            accuracy_metrics=prediction.accuracy_metrics,
            feature_importance=prediction.feature_importance
        )
        
    except ValueError as e:
        logger.warning(f"Invalid prediction request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Weather prediction failed: {e}")
        metrics.increment_counter("weather_prediction_errors")
        raise HTTPException(status_code=500, detail="Weather prediction failed")

@router.post("/correlations", response_model=Dict[str, Dict[str, float]],
             summary="Analyze parameter correlations")
@performance_monitor
async def analyze_correlations(request: CorrelationAnalysisRequest):
    """Analyze correlations between weather parameters."""
    try:
        location = Location(
            latitude=request.location.latitude,
            longitude=request.location.longitude,
            name=request.location.name
        )
        
        correlations = await analytics_service.analyze_correlations(
            location=location,
            parameters=request.parameters,
            days=request.days
        )
        
        metrics.increment_counter("correlation_analyses_requested")
        
        return correlations
        
    except ValueError as e:
        logger.warning(f"Invalid correlation analysis request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Correlation analysis failed: {e}")
        metrics.increment_counter("correlation_analysis_errors")
        raise HTTPException(status_code=500, detail="Correlation analysis failed")

@router.post("/climate-report", response_model=ClimateAnalysisResponse,
             summary="Generate comprehensive climate report")
@performance_monitor
async def generate_climate_report(request: ClimateReportRequest):
    """Generate comprehensive climate analysis report."""
    try:
        location = Location(
            latitude=request.location.latitude,
            longitude=request.location.longitude,
            name=request.location.name
        )
        
        climate_analysis = await analytics_service.generate_climate_report(
            location=location,
            years=request.years
        )
        
        metrics.increment_counter("climate_reports_requested")
        
        return ClimateAnalysisResponse(
            location=request.location,
            analysis_period=climate_analysis.analysis_period,
            temperature_statistics=climate_analysis.temperature_statistics,
            precipitation_statistics=climate_analysis.precipitation_statistics,
            wind_statistics=climate_analysis.wind_statistics,
            climate_indicators=climate_analysis.climate_indicators,
            trend_count=len(climate_analysis.long_term_trends)
        )
        
    except ValueError as e:
        logger.warning(f"Invalid climate report request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Climate report generation failed: {e}")
        metrics.increment_counter("climate_report_errors")
        raise HTTPException(status_code=500, detail="Climate report generation failed")

@router.get("/stats", response_model=AnalyticsStatsResponse,
           summary="Get analytics service statistics")
@performance_monitor
async def get_analytics_stats():
    """Get analytics service statistics and metrics."""
    try:
        # Get weather service cache stats
        from openweather.services.weather_service import weather_service
        cache_stats = weather_service.get_cache_stats()
        
        stats = {
            "total_analyses": metrics.counters.get("trend_analyses_completed", 0) +
                            metrics.counters.get("anomaly_detections_completed", 0) +
                            metrics.counters.get("ml_predictions_completed", 0) +
                            metrics.counters.get("climate_reports_generated", 0),
            "trend_analyses": metrics.counters.get("trend_analyses_completed", 0),
            "anomaly_detections": metrics.counters.get("anomaly_detections_completed", 0),
            "predictions": metrics.counters.get("ml_predictions_completed", 0),
            "climate_reports": metrics.counters.get("climate_reports_generated", 0),
            "active_models": len(analytics_service.models),
            "cache_stats": cache_stats
        }
        
        return AnalyticsStatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"Failed to get analytics stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get analytics statistics")

@router.get("/parameters", summary="Get available analysis parameters")
async def get_available_parameters():
    """Get list of available weather parameters for analysis."""
    parameters = {
        "basic_parameters": [
            "temperature",
            "humidity", 
            "pressure",
            "wind_speed",
            "visibility"
        ],
        "derived_parameters": [
            "wind_direction",
            "cloud_cover"
        ],
        "analysis_types": [
            "trend_analysis",
            "anomaly_detection", 
            "correlation_analysis",
            "prediction",
            "climatology"
        ],
        "prediction_models": [
            "random_forest",
            "linear_regression", 
            "neural_network",
            "ensemble"
        ]
    }
    
    return parameters

@router.get("/models", summary="Get ML model information")
@performance_monitor
async def get_model_info():
    """Get information about loaded ML models."""
    try:
        model_info = {}
        
        for name, model in analytics_service.models.items():
            info = {
                "type": type(model).__name__,
                "trained": hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'),
                "parameters": {}
            }
            
            # Get model parameters if available
            if hasattr(model, 'get_params'):
                info["parameters"] = model.get_params()
            
            model_info[name] = info
        
        return {
            "models": model_info,
            "scalers": list(analytics_service.scalers.keys()),
            "total_models": len(analytics_service.models)
        }
        
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model information")

@router.delete("/cache", summary="Clear analytics cache")
@performance_monitor
async def clear_analytics_cache():
    """Clear all analytics caches."""
    try:
        from openweather.services.weather_service import weather_service
        weather_service.clear_cache()
        
        return {
            "status": "success",
            "message": "Analytics cache cleared",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear cache")

# Background task endpoints
@router.post("/batch/trends", summary="Submit batch trend analysis")
async def batch_trend_analysis(
    requests: List[TrendAnalysisRequest], 
    background_tasks: BackgroundTasks
):
    """Submit batch trend analysis requests for background processing."""
    try:
        task_id = f"batch_trends_{datetime.utcnow().isoformat()}"
        
        # Add background task
        background_tasks.add_task(
            _process_batch_trends, 
            requests, 
            task_id
        )
        
        metrics.increment_counter("batch_trend_analyses_submitted")
        
        return {
            "task_id": task_id,
            "status": "submitted",
            "request_count": len(requests),
            "estimated_completion": (datetime.utcnow() + 
                                   timedelta(minutes=len(requests) * 2)).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Batch trend analysis submission failed: {e}")
        raise HTTPException(status_code=500, detail="Batch submission failed")

async def _process_batch_trends(requests: List[TrendAnalysisRequest], task_id: str):
    """Process batch trend analysis requests."""
    logger.info(f"Starting batch trend analysis task {task_id} with {len(requests)} requests")
    
    results = []
    for i, request in enumerate(requests):
        try:
            location = Location(
                latitude=request.location.latitude,
                longitude=request.location.longitude,
                name=request.location.name
            )
            
            result = await analytics_service.analyze_trends(
                location=location,
                parameter=request.parameter,
                start_date=request.start_date,
                end_date=request.end_date
            )
            
            results.append({"index": i, "status": "success", "result": result})
            
        except Exception as e:
            logger.error(f"Batch trend analysis failed for request {i}: {e}")
            results.append({"index": i, "status": "error", "error": str(e)})
    
    logger.info(f"Completed batch trend analysis task {task_id}")
    # In a real implementation, you'd store results somewhere accessible
    return results 