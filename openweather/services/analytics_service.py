"""Advanced weather analytics and machine learning service."""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import statistics
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from openweather.core.monitoring import MetricsCollector, performance_monitor
from openweather.models.weather import WeatherData, ForecastData, Location
from openweather.services.weather_service import WeatherService, weather_service

logger = logging.getLogger(__name__)

class AnalysisType(str, Enum):
    """Types of weather analysis."""
    TREND_ANALYSIS = "trend_analysis"
    ANOMALY_DETECTION = "anomaly_detection"
    CORRELATION_ANALYSIS = "correlation_analysis"
    PREDICTION = "prediction"
    CLIMATOLOGY = "climatology"
    EXTREME_WEATHER = "extreme_weather"

class PredictionModel(str, Enum):
    """Machine learning models for predictions."""
    RANDOM_FOREST = "random_forest"
    LINEAR_REGRESSION = "linear_regression"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"

@dataclass
class TrendAnalysis:
    """Weather trend analysis results."""
    parameter: str
    trend_direction: str  # "increasing", "decreasing", "stable"
    trend_strength: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    period_start: datetime
    period_end: datetime
    seasonal_pattern: Optional[Dict[str, float]] = None
    statistical_significance: Optional[float] = None

@dataclass
class AnomalyDetection:
    """Anomaly detection results."""
    timestamp: datetime
    parameter: str
    value: float
    expected_range: Tuple[float, float]
    anomaly_score: float  # 0.0 to 1.0
    severity: str  # "low", "medium", "high", "critical"
    description: str

@dataclass
class WeatherPrediction:
    """Weather prediction results."""
    location: Location
    prediction_time: datetime
    forecast_horizon: timedelta
    predicted_values: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    model_used: PredictionModel
    accuracy_metrics: Dict[str, float]
    feature_importance: Optional[Dict[str, float]] = None

@dataclass
class ClimateAnalysis:
    """Climate analysis results."""
    location: Location
    analysis_period: Tuple[datetime, datetime]
    temperature_statistics: Dict[str, float]
    precipitation_statistics: Dict[str, float]
    wind_statistics: Dict[str, float]
    extreme_events: List[Dict[str, Any]]
    climate_indicators: Dict[str, float]
    long_term_trends: List[TrendAnalysis]

class WeatherAnalyticsService:
    """Advanced weather analytics and ML service."""
    
    def __init__(self, weather_service: WeatherService):
        self.weather_service = weather_service
        self.metrics = MetricsCollector()
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize ML models for predictions."""
        # Temperature prediction model
        self.models['temperature'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Humidity prediction model
        self.models['humidity'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            random_state=42
        )
        
        # Anomaly detection model
        self.models['anomaly'] = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        
        # Initialize scalers
        for param in ['temperature', 'humidity', 'pressure', 'wind_speed']:
            self.scalers[param] = StandardScaler()
    
    @performance_monitor
    async def analyze_trends(self, location: Location, 
                           parameter: str,
                           start_date: datetime,
                           end_date: datetime) -> TrendAnalysis:
        """Analyze weather parameter trends over time."""
        try:
            # Get historical data (placeholder - would come from database)
            historical_data = await self._get_historical_data(location, start_date, end_date)
            
            if not historical_data:
                raise ValueError("No historical data available for trend analysis")
            
            # Extract parameter values
            values = [getattr(data, parameter, 0) for data in historical_data]
            timestamps = [data.timestamp for data in historical_data]
            
            if len(values) < 10:
                raise ValueError("Insufficient data for trend analysis")
            
            # Calculate trend using linear regression
            x = np.array([(ts - timestamps[0]).total_seconds() for ts in timestamps])
            y = np.array(values)
            
            # Fit linear trend
            coeffs = np.polyfit(x, y, 1)
            trend_slope = coeffs[0]
            
            # Determine trend direction and strength
            if abs(trend_slope) < 1e-10:
                trend_direction = "stable"
                trend_strength = 0.0
            elif trend_slope > 0:
                trend_direction = "increasing"
                trend_strength = min(abs(trend_slope) * 1000, 1.0)
            else:
                trend_direction = "decreasing"
                trend_strength = -min(abs(trend_slope) * 1000, 1.0)
            
            # Calculate confidence (R-squared)
            y_pred = np.polyval(coeffs, x)
            confidence = r2_score(y, y_pred) if len(y) > 1 else 0.0
            
            # Seasonal pattern analysis
            seasonal_pattern = self._analyze_seasonal_pattern(timestamps, values)
            
            self.metrics.increment_counter("trend_analyses_completed")
            
            return TrendAnalysis(
                parameter=parameter,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                confidence=max(0.0, confidence),
                period_start=start_date,
                period_end=end_date,
                seasonal_pattern=seasonal_pattern
            )
            
        except Exception as e:
            self.metrics.increment_counter("trend_analysis_errors")
            logger.error(f"Trend analysis failed: {e}")
            raise
    
    def _analyze_seasonal_pattern(self, timestamps: List[datetime], 
                                values: List[float]) -> Dict[str, float]:
        """Analyze seasonal patterns in data."""
        if len(timestamps) < 12:  # Need at least a year of monthly data
            return {}
        
        # Group by month
        monthly_averages = {}
        for ts, value in zip(timestamps, values):
            month = ts.month
            if month not in monthly_averages:
                monthly_averages[month] = []
            monthly_averages[month].append(value)
        
        # Calculate monthly means
        seasonal_pattern = {}
        for month, month_values in monthly_averages.items():
            if month_values:
                seasonal_pattern[f"month_{month}"] = statistics.mean(month_values)
        
        return seasonal_pattern
    
    @performance_monitor
    async def detect_anomalies(self, location: Location,
                             parameter: str,
                             lookback_days: int = 30) -> List[AnomalyDetection]:
        """Detect anomalies in weather data."""
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=lookback_days)
            
            # Get recent data
            historical_data = await self._get_historical_data(location, start_date, end_date)
            
            if len(historical_data) < 10:
                logger.warning("Insufficient data for anomaly detection")
                return []
            
            # Extract parameter values
            values = np.array([getattr(data, parameter, 0) for data in historical_data])
            timestamps = [data.timestamp for data in historical_data]
            
            # Calculate statistical bounds (3-sigma rule)
            mean_val = np.mean(values)
            std_val = np.std(values)
            lower_bound = mean_val - 3 * std_val
            upper_bound = mean_val + 3 * std_val
            
            # Detect anomalies
            anomalies = []
            for i, (ts, value) in enumerate(zip(timestamps, values)):
                if value < lower_bound or value > upper_bound:
                    # Calculate anomaly score
                    deviation = abs(value - mean_val) / std_val if std_val > 0 else 0
                    anomaly_score = min(deviation / 3.0, 1.0)
                    
                    # Determine severity
                    if anomaly_score > 0.8:
                        severity = "critical"
                    elif anomaly_score > 0.6:
                        severity = "high"
                    elif anomaly_score > 0.4:
                        severity = "medium"
                    else:
                        severity = "low"
                    
                    description = f"{parameter} value {value:.2f} is outside normal range"
                    
                    anomalies.append(AnomalyDetection(
                        timestamp=ts,
                        parameter=parameter,
                        value=value,
                        expected_range=(lower_bound, upper_bound),
                        anomaly_score=anomaly_score,
                        severity=severity,
                        description=description
                    ))
            
            self.metrics.increment_counter("anomaly_detections_completed")
            self.metrics.set_gauge("anomalies_detected", len(anomalies))
            
            return anomalies
            
        except Exception as e:
            self.metrics.increment_counter("anomaly_detection_errors")
            logger.error(f"Anomaly detection failed: {e}")
            return []
    
    @performance_monitor
    async def predict_weather(self, location: Location,
                            forecast_hours: int = 24,
                            model_type: PredictionModel = PredictionModel.RANDOM_FOREST) -> WeatherPrediction:
        """Predict weather using machine learning models."""
        try:
            # Get recent data for training
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=30)
            
            historical_data = await self._get_historical_data(location, start_date, end_date)
            
            if len(historical_data) < 50:
                raise ValueError("Insufficient training data for ML prediction")
            
            # Prepare training data
            features, targets = self._prepare_ml_features(historical_data)
            
            if len(features) == 0:
                raise ValueError("No valid features extracted from historical data")
            
            # Train models
            predictions = {}
            confidence_intervals = {}
            accuracy_metrics = {}
            feature_importance = {}
            
            for param in ['temperature', 'humidity', 'pressure']:
                if param in targets and len(targets[param]) > 0:
                    # Train model
                    X_train, X_test, y_train, y_test = train_test_split(
                        features, targets[param], test_size=0.2, random_state=42
                    )
                    
                    model = self.models.get(param, self.models['temperature'])
                    model.fit(X_train, y_train)
                    
                    # Make prediction for future
                    current_features = features[-1:] if len(features) > 0 else [[0] * len(features[0])]
                    pred = model.predict(current_features)[0]
                    predictions[param] = pred
                    
                    # Calculate confidence interval (simplified)
                    if len(X_test) > 0:
                        y_pred_test = model.predict(X_test)
                        mae = mean_absolute_error(y_test, y_pred_test)
                        confidence_intervals[param] = (pred - mae, pred + mae)
                        accuracy_metrics[f"{param}_mae"] = mae
                        accuracy_metrics[f"{param}_r2"] = r2_score(y_test, y_pred_test)
                    else:
                        confidence_intervals[param] = (pred - 1, pred + 1)
                        accuracy_metrics[f"{param}_mae"] = 0.0
                        accuracy_metrics[f"{param}_r2"] = 0.0
                    
                    # Feature importance
                    if hasattr(model, 'feature_importances_'):
                        feature_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]
                        feature_importance[param] = dict(zip(feature_names, model.feature_importances_))
            
            self.metrics.increment_counter("ml_predictions_completed")
            
            return WeatherPrediction(
                location=location,
                prediction_time=datetime.utcnow(),
                forecast_horizon=timedelta(hours=forecast_hours),
                predicted_values=predictions,
                confidence_intervals=confidence_intervals,
                model_used=model_type,
                accuracy_metrics=accuracy_metrics,
                feature_importance=feature_importance
            )
            
        except Exception as e:
            self.metrics.increment_counter("ml_prediction_errors")
            logger.error(f"ML prediction failed: {e}")
            raise
    
    def _prepare_ml_features(self, historical_data: List[WeatherData]) -> Tuple[List[List[float]], Dict[str, List[float]]]:
        """Prepare features and targets for ML models."""
        features = []
        targets = {'temperature': [], 'humidity': [], 'pressure': []}
        
        for i in range(len(historical_data) - 1):
            current = historical_data[i]
            next_point = historical_data[i + 1]
            
            # Create features from current data point
            feature_vector = [
                current.temperature,
                current.humidity,
                current.pressure,
                current.wind_speed,
                current.wind_direction,
                current.visibility,
                current.conditions.cloud_cover if current.conditions else 0,
                current.timestamp.hour,
                current.timestamp.day,
                current.timestamp.month
            ]
            
            features.append(feature_vector)
            
            # Targets are next time step values
            targets['temperature'].append(next_point.temperature)
            targets['humidity'].append(next_point.humidity)
            targets['pressure'].append(next_point.pressure)
        
        return features, targets
    
    @performance_monitor
    async def analyze_correlations(self, location: Location,
                                 parameters: List[str],
                                 days: int = 30) -> Dict[str, Dict[str, float]]:
        """Analyze correlations between weather parameters."""
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            historical_data = await self._get_historical_data(location, start_date, end_date)
            
            if len(historical_data) < 10:
                logger.warning("Insufficient data for correlation analysis")
                return {}
            
            # Create DataFrame
            data_dict = {}
            for param in parameters:
                values = []
                for data_point in historical_data:
                    if hasattr(data_point, param):
                        values.append(getattr(data_point, param))
                    else:
                        values.append(0)
                data_dict[param] = values
            
            df = pd.DataFrame(data_dict)
            
            # Calculate correlation matrix
            correlation_matrix = df.corr()
            
            # Convert to nested dictionary
            correlations = {}
            for param1 in parameters:
                correlations[param1] = {}
                for param2 in parameters:
                    if param1 in correlation_matrix.index and param2 in correlation_matrix.columns:
                        correlations[param1][param2] = float(correlation_matrix.loc[param1, param2])
                    else:
                        correlations[param1][param2] = 0.0
            
            self.metrics.increment_counter("correlation_analyses_completed")
            
            return correlations
            
        except Exception as e:
            self.metrics.increment_counter("correlation_analysis_errors")
            logger.error(f"Correlation analysis failed: {e}")
            return {}
    
    @performance_monitor
    async def generate_climate_report(self, location: Location,
                                    years: int = 5) -> ClimateAnalysis:
        """Generate comprehensive climate analysis report."""
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=years * 365)
            
            historical_data = await self._get_historical_data(location, start_date, end_date)
            
            if len(historical_data) < 365:  # Need at least a year of data
                raise ValueError("Insufficient data for climate analysis")
            
            # Extract values
            temperatures = [d.temperature for d in historical_data]
            humidity_values = [d.humidity for d in historical_data]
            wind_speeds = [d.wind_speed for d in historical_data]
            
            # Temperature statistics
            temp_stats = {
                "mean": statistics.mean(temperatures),
                "median": statistics.median(temperatures),
                "min": min(temperatures),
                "max": max(temperatures),
                "std": statistics.stdev(temperatures) if len(temperatures) > 1 else 0,
                "percentile_90": np.percentile(temperatures, 90),
                "percentile_10": np.percentile(temperatures, 10)
            }
            
            # Humidity statistics
            humidity_stats = {
                "mean": statistics.mean(humidity_values),
                "median": statistics.median(humidity_values),
                "min": min(humidity_values),
                "max": max(humidity_values),
                "std": statistics.stdev(humidity_values) if len(humidity_values) > 1 else 0
            }
            
            # Wind statistics
            wind_stats = {
                "mean": statistics.mean(wind_speeds),
                "median": statistics.median(wind_speeds),
                "min": min(wind_speeds),
                "max": max(wind_speeds),
                "std": statistics.stdev(wind_speeds) if len(wind_speeds) > 1 else 0
            }
            
            # Climate indicators
            climate_indicators = {
                "temperature_variability": temp_stats["std"],
                "extreme_heat_days": len([t for t in temperatures if t > temp_stats["percentile_90"]]),
                "extreme_cold_days": len([t for t in temperatures if t < temp_stats["percentile_10"]]),
                "high_humidity_days": len([h for h in humidity_values if h > 80]),
                "windy_days": len([w for w in wind_speeds if w > 15])
            }
            
            # Trend analyses
            trends = []
            for param in ['temperature', 'humidity', 'wind_speed']:
                try:
                    trend = await self.analyze_trends(location, param, start_date, end_date)
                    trends.append(trend)
                except Exception as e:
                    logger.warning(f"Failed to analyze trend for {param}: {e}")
            
            self.metrics.increment_counter("climate_reports_generated")
            
            return ClimateAnalysis(
                location=location,
                analysis_period=(start_date, end_date),
                temperature_statistics=temp_stats,
                precipitation_statistics={},  # Placeholder
                wind_statistics=wind_stats,
                extreme_events=[],  # Placeholder
                climate_indicators=climate_indicators,
                long_term_trends=trends
            )
            
        except Exception as e:
            self.metrics.increment_counter("climate_analysis_errors")
            logger.error(f"Climate analysis failed: {e}")
            raise
    
    async def _get_historical_data(self, location: Location,
                                 start_date: datetime,
                                 end_date: datetime) -> List[WeatherData]:
        """Get historical weather data (placeholder implementation)."""
        # In a real implementation, this would query a database
        # For now, generate synthetic historical data
        data_points = []
        current_date = start_date
        
        while current_date <= end_date:
            # Generate synthetic data with some patterns
            base_temp = 20 + 10 * np.sin(2 * np.pi * current_date.day / 365)
            noise = np.random.normal(0, 2)
            
            weather_data = WeatherData(
                location=location,
                timestamp=current_date,
                temperature=base_temp + noise,
                humidity=60 + 20 * np.random.random(),
                pressure=1013 + 20 * np.random.random(),
                wind_speed=5 + 10 * np.random.random(),
                wind_direction=360 * np.random.random(),
                visibility=10.0,
                conditions=None,
                provider="synthetic"
            )
            
            data_points.append(weather_data)
            current_date += timedelta(hours=6)  # 4 data points per day
        
        return data_points

# Global analytics service instance
analytics_service = WeatherAnalyticsService(weather_service) 