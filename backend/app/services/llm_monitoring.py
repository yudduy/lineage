"""
LLM Monitoring and Analytics System - Comprehensive monitoring for LLM operations.

This module provides:
- Real-time performance monitoring
- Cost tracking and alerting
- Usage analytics and reporting
- Quality metrics and assessment
- Trend analysis and forecasting
- Dashboard data aggregation
- Alert management and notifications
"""

import asyncio
import json
import time
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import statistics
import logging

from ..core.config import Settings, get_settings
from ..services.llm_cost_manager import get_cost_manager, CostAnalytics, CostCategory
from ..services.llm_cache import get_cache_manager
from ..services.llm_fallback import get_fallback_service
from ..db.redis import RedisManager, get_redis_manager
from ..utils.logger import get_logger

logger = get_logger(__name__)


class MetricType(Enum):
    """Types of metrics collected."""
    PERFORMANCE = "performance"
    COST = "cost"
    QUALITY = "quality"
    USAGE = "usage"
    ERROR = "error"
    CACHE = "cache"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class PerformanceMetric:
    """Performance metric data point."""
    timestamp: datetime
    operation_type: str
    model: str
    provider: str
    response_time_ms: float
    tokens_processed: int
    success: bool
    error_type: Optional[str] = None
    fallback_used: bool = False
    cache_hit: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class QualityMetric:
    """Quality assessment metric."""
    timestamp: datetime
    operation_type: str
    model: str
    quality_score: float
    confidence_score: float
    content_length: int
    structured_output: bool
    error_indicators: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class SystemAlert:
    """System alert information."""
    alert_id: str
    timestamp: datetime
    severity: AlertSeverity
    category: str
    title: str
    description: str
    metrics: Dict[str, Any]
    acknowledged: bool = False
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['severity'] = self.severity.value
        return data


@dataclass
class DashboardData:
    """Aggregated dashboard data."""
    timestamp: datetime
    time_period: str  # 'last_hour', 'last_day', 'last_week'
    
    # Performance metrics
    avg_response_time_ms: float
    total_requests: int
    success_rate: float
    error_rate: float
    
    # Cost metrics
    total_cost: float
    cost_per_request: float
    budget_utilization: float
    
    # Quality metrics
    avg_quality_score: float
    avg_confidence_score: float
    
    # Usage patterns
    requests_by_model: Dict[str, int]
    requests_by_category: Dict[str, int]
    peak_usage_hour: Optional[int]
    
    # Cache metrics
    cache_hit_rate: float
    cache_savings: float
    
    # Fallback metrics
    fallback_usage_rate: float
    circuit_breaker_status: Dict[str, str]


class LLMMonitoringService:
    """
    Comprehensive monitoring and analytics service for LLM operations.
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        redis_manager: Optional[RedisManager] = None
    ):
        self.settings = settings or get_settings()
        self.redis_manager = redis_manager
        
        # In-memory metrics storage for real-time monitoring
        self._performance_buffer = deque(maxlen=1000)
        self._quality_buffer = deque(maxlen=500)
        self._alert_buffer = deque(maxlen=100)
        
        # Alert thresholds
        self.alert_thresholds = {
            'response_time_ms': 10000,      # 10 seconds
            'error_rate': 0.1,              # 10% error rate
            'cost_spike': 2.0,              # 2x normal cost
            'quality_drop': 0.5,            # Quality score below 50%
            'budget_utilization': 0.9       # 90% budget used
        }
        
        # Trend analysis windows
        self.trend_windows = {
            'short_term': timedelta(hours=1),
            'medium_term': timedelta(hours=6),
            'long_term': timedelta(days=1)
        }
        
        self._initialized = False
        self._monitoring_task = None
    
    async def initialize(self):
        """Initialize the monitoring service."""
        if self._initialized:
            return
        
        if not self.redis_manager:
            self.redis_manager = await get_redis_manager()
        
        # Start background monitoring
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        self._initialized = True
        logger.info("LLM Monitoring Service initialized")
    
    async def record_performance_metric(
        self,
        operation_type: str,
        model: str,
        provider: str,
        response_time_ms: float,
        tokens_processed: int,
        success: bool,
        error_type: Optional[str] = None,
        fallback_used: bool = False,
        cache_hit: bool = False
    ):
        """Record a performance metric."""
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            operation_type=operation_type,
            model=model,
            provider=provider,
            response_time_ms=response_time_ms,
            tokens_processed=tokens_processed,
            success=success,
            error_type=error_type,
            fallback_used=fallback_used,
            cache_hit=cache_hit
        )
        
        # Add to buffer
        self._performance_buffer.append(metric)
        
        # Store in Redis for persistence
        await self._store_metric(MetricType.PERFORMANCE, metric.to_dict())
        
        # Check for alerts
        await self._check_performance_alerts(metric)
    
    async def record_quality_metric(
        self,
        operation_type: str,
        model: str,
        quality_score: float,
        confidence_score: float,
        content_length: int,
        structured_output: bool,
        error_indicators: List[str]
    ):
        """Record a quality assessment metric."""
        metric = QualityMetric(
            timestamp=datetime.now(),
            operation_type=operation_type,
            model=model,
            quality_score=quality_score,
            confidence_score=confidence_score,
            content_length=content_length,
            structured_output=structured_output,
            error_indicators=error_indicators
        )
        
        # Add to buffer
        self._quality_buffer.append(metric)
        
        # Store in Redis
        await self._store_metric(MetricType.QUALITY, metric.to_dict())
        
        # Check for quality alerts
        await self._check_quality_alerts(metric)
    
    async def _store_metric(self, metric_type: MetricType, metric_data: Dict[str, Any]):
        """Store metric in Redis for persistence."""
        if not self.redis_manager:
            return
        
        try:
            # Create time-based key for efficient querying
            timestamp = datetime.now()
            hour_key = timestamp.strftime('%Y%m%d%H')
            day_key = timestamp.strftime('%Y%m%d')
            
            # Store in hourly bucket
            hourly_key = f"llm_metrics:{metric_type.value}:hourly:{hour_key}"
            await self.redis_manager.lpush(hourly_key, json.dumps(metric_data))
            await self.redis_manager.expire(hourly_key, 86400 * 7)  # 7 days
            
            # Store in daily aggregation
            daily_key = f"llm_metrics:{metric_type.value}:daily:{day_key}"
            await self.redis_manager.lpush(daily_key, json.dumps(metric_data))
            await self.redis_manager.expire(daily_key, 86400 * 30)  # 30 days
            
        except Exception as e:
            logger.warning(f"Failed to store metric: {e}")
    
    async def _check_performance_alerts(self, metric: PerformanceMetric):
        """Check performance metric against alert thresholds."""
        alerts = []
        
        # Response time alert
        if metric.response_time_ms > self.alert_thresholds['response_time_ms']:
            alerts.append(self._create_alert(
                AlertSeverity.WARNING,
                "performance",
                "High Response Time",
                f"Response time {metric.response_time_ms:.0f}ms exceeds threshold for {metric.model}",
                {"response_time_ms": metric.response_time_ms, "model": metric.model}
            ))
        
        # Error rate alert (check recent error rate)
        recent_metrics = [m for m in self._performance_buffer 
                         if (datetime.now() - m.timestamp).total_seconds() < 300]  # Last 5 minutes
        if len(recent_metrics) >= 10:
            error_rate = sum(1 for m in recent_metrics if not m.success) / len(recent_metrics)
            if error_rate > self.alert_thresholds['error_rate']:
                alerts.append(self._create_alert(
                    AlertSeverity.CRITICAL,
                    "performance",
                    "High Error Rate",
                    f"Error rate {error_rate:.1%} exceeds threshold",
                    {"error_rate": error_rate, "sample_size": len(recent_metrics)}
                ))
        
        # Process alerts
        for alert in alerts:
            await self._process_alert(alert)
    
    async def _check_quality_alerts(self, metric: QualityMetric):
        """Check quality metric against alert thresholds."""
        alerts = []
        
        # Quality drop alert
        if metric.quality_score < self.alert_thresholds['quality_drop']:
            alerts.append(self._create_alert(
                AlertSeverity.WARNING,
                "quality",
                "Low Quality Score",
                f"Quality score {metric.quality_score:.2f} below threshold for {metric.model}",
                {"quality_score": metric.quality_score, "model": metric.model}
            ))
        
        # Error indicators alert
        if len(metric.error_indicators) > 2:
            alerts.append(self._create_alert(
                AlertSeverity.WARNING,
                "quality",
                "Multiple Error Indicators",
                f"Found {len(metric.error_indicators)} error indicators in output",
                {"error_indicators": metric.error_indicators, "model": metric.model}
            ))
        
        # Process alerts
        for alert in alerts:
            await self._process_alert(alert)
    
    def _create_alert(
        self,
        severity: AlertSeverity,
        category: str,
        title: str,
        description: str,
        metrics: Dict[str, Any]
    ) -> SystemAlert:
        """Create a system alert."""
        return SystemAlert(
            alert_id=f"{category}_{int(time.time())}",
            timestamp=datetime.now(),
            severity=severity,
            category=category,
            title=title,
            description=description,
            metrics=metrics
        )
    
    async def _process_alert(self, alert: SystemAlert):
        """Process and store a system alert."""
        # Add to buffer
        self._alert_buffer.append(alert)
        
        # Store in Redis
        if self.redis_manager:
            try:
                alert_key = f"llm_alerts:{alert.alert_id}"
                await self.redis_manager.setex(
                    alert_key,
                    86400 * 7,  # 7 days
                    json.dumps(alert.to_dict())
                )
                
                # Add to alerts index
                alerts_index_key = f"llm_alerts_index:{datetime.now().strftime('%Y%m%d')}"
                await self.redis_manager.lpush(alerts_index_key, alert.alert_id)
                await self.redis_manager.expire(alerts_index_key, 86400 * 30)  # 30 days
                
            except Exception as e:
                logger.warning(f"Failed to store alert: {e}")
        
        # Log alert
        log_level = {
            AlertSeverity.INFO: logger.info,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.CRITICAL: logger.error,
            AlertSeverity.EMERGENCY: logger.critical
        }.get(alert.severity, logger.info)
        
        log_level(f"LLM Alert [{alert.severity.value.upper()}]: {alert.title} - {alert.description}")
    
    async def _monitoring_loop(self):
        """Background monitoring loop for continuous system health checks."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                # Check system health
                await self._check_system_health()
                
                # Check budget alerts
                await self._check_budget_alerts()
                
                # Clean up old metrics
                await self._cleanup_old_metrics()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)  # Continue after error
    
    async def _check_system_health(self):
        """Perform periodic system health checks."""
        try:
            # Check fallback service health
            fallback_service = await get_fallback_service()
            health_status = await fallback_service.get_system_health()
            
            # Check for circuit breakers in OPEN state
            for provider, status in health_status['providers'].items():
                if status['status'] == 'open':
                    alert = self._create_alert(
                        AlertSeverity.CRITICAL,
                        "system",
                        "Circuit Breaker Open",
                        f"Circuit breaker is open for provider {provider}",
                        {"provider": provider, "status": status}
                    )
                    await self._process_alert(alert)
            
            # Check cache health
            cache_manager = await get_cache_manager()
            cache_stats = await cache_manager.get_cache_stats()
            
            # Low cache hit rate alert
            if cache_stats.hit_rate < 0.3 and cache_stats.total_requests > 100:
                alert = self._create_alert(
                    AlertSeverity.WARNING,
                    "cache",
                    "Low Cache Hit Rate",
                    f"Cache hit rate {cache_stats.hit_rate:.1%} is below optimal threshold",
                    {"hit_rate": cache_stats.hit_rate, "total_requests": cache_stats.total_requests}
                )
                await self._process_alert(alert)
            
        except Exception as e:
            logger.warning(f"System health check failed: {e}")
    
    async def _check_budget_alerts(self):
        """Check budget utilization and generate alerts."""
        try:
            cost_manager = await get_cost_manager()
            
            daily_cost = await cost_manager.get_daily_cost()
            monthly_cost = await cost_manager.get_monthly_cost()
            
            daily_limit = self.settings.llm.daily_budget_limit
            monthly_limit = self.settings.llm.monthly_budget_limit
            
            # Daily budget alerts
            daily_utilization = daily_cost / daily_limit if daily_limit > 0 else 0
            if daily_utilization > self.alert_thresholds['budget_utilization']:
                alert = self._create_alert(
                    AlertSeverity.CRITICAL if daily_utilization >= 1.0 else AlertSeverity.WARNING,
                    "budget",
                    "Daily Budget Alert",
                    f"Daily budget utilization: {daily_utilization:.1%} (${daily_cost:.2f}/${daily_limit:.2f})",
                    {"daily_cost": daily_cost, "daily_limit": daily_limit, "utilization": daily_utilization}
                )
                await self._process_alert(alert)
            
            # Monthly budget alerts
            monthly_utilization = monthly_cost / monthly_limit if monthly_limit > 0 else 0
            if monthly_utilization > self.alert_thresholds['budget_utilization']:
                alert = self._create_alert(
                    AlertSeverity.CRITICAL if monthly_utilization >= 1.0 else AlertSeverity.WARNING,
                    "budget",
                    "Monthly Budget Alert",
                    f"Monthly budget utilization: {monthly_utilization:.1%} (${monthly_cost:.2f}/${monthly_limit:.2f})",
                    {"monthly_cost": monthly_cost, "monthly_limit": monthly_limit, "utilization": monthly_utilization}
                )
                await self._process_alert(alert)
                
        except Exception as e:
            logger.warning(f"Budget alert check failed: {e}")
    
    async def _cleanup_old_metrics(self):
        """Clean up old metrics to prevent memory leaks."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=1)
            
            # Clean performance buffer
            while (self._performance_buffer and 
                   self._performance_buffer[0].timestamp < cutoff_time):
                self._performance_buffer.popleft()
            
            # Clean quality buffer
            while (self._quality_buffer and 
                   self._quality_buffer[0].timestamp < cutoff_time):
                self._quality_buffer.popleft()
            
            # Clean alert buffer
            alert_cutoff = datetime.now() - timedelta(hours=24)
            while (self._alert_buffer and 
                   self._alert_buffer[0].timestamp < alert_cutoff):
                self._alert_buffer.popleft()
                
        except Exception as e:
            logger.warning(f"Metrics cleanup failed: {e}")
    
    async def get_dashboard_data(self, time_period: str = 'last_hour') -> DashboardData:
        """Get aggregated dashboard data for the specified time period."""
        await self.initialize()
        
        # Define time ranges
        time_ranges = {
            'last_hour': timedelta(hours=1),
            'last_day': timedelta(days=1),
            'last_week': timedelta(weeks=1),
            'last_month': timedelta(days=30)
        }
        
        if time_period not in time_ranges:
            time_period = 'last_hour'
        
        time_range = time_ranges[time_period]
        cutoff_time = datetime.now() - time_range
        
        # Filter metrics by time period
        performance_metrics = [
            m for m in self._performance_buffer 
            if m.timestamp >= cutoff_time
        ]
        
        quality_metrics = [
            m for m in self._quality_buffer 
            if m.timestamp >= cutoff_time
        ]
        
        # Calculate performance metrics
        total_requests = len(performance_metrics)
        successful_requests = sum(1 for m in performance_metrics if m.success)
        success_rate = successful_requests / total_requests if total_requests > 0 else 1.0
        error_rate = 1.0 - success_rate
        
        avg_response_time = (
            statistics.mean([m.response_time_ms for m in performance_metrics])
            if performance_metrics else 0.0
        )
        
        # Calculate cost metrics
        cost_manager = await get_cost_manager()
        analytics = await cost_manager.get_cost_analytics(
            start_date=(datetime.now() - time_range).date(),
            end_date=datetime.now().date()
        )
        
        # Calculate quality metrics
        avg_quality_score = (
            statistics.mean([m.quality_score for m in quality_metrics])
            if quality_metrics else 0.0
        )
        
        avg_confidence_score = (
            statistics.mean([m.confidence_score for m in quality_metrics])
            if quality_metrics else 0.0
        )
        
        # Calculate usage patterns
        requests_by_model = {}
        requests_by_category = {}
        hourly_requests = defaultdict(int)
        
        for metric in performance_metrics:
            requests_by_model[metric.model] = requests_by_model.get(metric.model, 0) + 1
            requests_by_category[metric.operation_type] = requests_by_category.get(metric.operation_type, 0) + 1
            hour = metric.timestamp.hour
            hourly_requests[hour] += 1
        
        peak_usage_hour = max(hourly_requests.keys(), key=hourly_requests.get) if hourly_requests else None
        
        # Calculate cache metrics
        cache_manager = await get_cache_manager()
        cache_stats = await cache_manager.get_cache_stats()
        
        # Calculate fallback metrics
        fallback_requests = sum(1 for m in performance_metrics if m.fallback_used)
        fallback_usage_rate = fallback_requests / total_requests if total_requests > 0 else 0.0
        
        # Get circuit breaker status
        fallback_service = await get_fallback_service()
        health_status = await fallback_service.get_system_health()
        circuit_breaker_status = {
            provider: status['status'] 
            for provider, status in health_status['providers'].items()
        }
        
        return DashboardData(
            timestamp=datetime.now(),
            time_period=time_period,
            avg_response_time_ms=avg_response_time,
            total_requests=total_requests,
            success_rate=success_rate,
            error_rate=error_rate,
            total_cost=analytics.total_cost,
            cost_per_request=analytics.avg_cost_per_request,
            budget_utilization=0.0,  # Would calculate based on current costs vs limits
            avg_quality_score=avg_quality_score,
            avg_confidence_score=avg_confidence_score,
            requests_by_model=requests_by_model,
            requests_by_category=requests_by_category,
            peak_usage_hour=peak_usage_hour,
            cache_hit_rate=cache_stats.hit_rate,
            cache_savings=cache_stats.cost_savings,
            fallback_usage_rate=fallback_usage_rate,
            circuit_breaker_status=circuit_breaker_status
        )
    
    async def get_trend_analysis(self, metric_type: str, days: int = 7) -> Dict[str, Any]:
        """Get trend analysis for specified metric over time period."""
        if not self.redis_manager:
            return {"error": "Redis not available for trend analysis"}
        
        try:
            trends = {
                'period_days': days,
                'metric_type': metric_type,
                'daily_values': [],
                'trend_direction': 'stable',
                'percentage_change': 0.0,
                'forecast': []
            }
            
            # Get daily metrics
            for i in range(days):
                date_obj = datetime.now().date() - timedelta(days=i)
                day_key = date_obj.strftime('%Y%m%d')
                
                daily_metrics_key = f"llm_metrics:{metric_type}:daily:{day_key}"
                daily_data = await self.redis_manager.lrange(daily_metrics_key, 0, -1)
                
                # Parse and aggregate daily metrics
                daily_value = 0.0
                if daily_data:
                    for data_point in daily_data:
                        try:
                            metric_data = json.loads(data_point)
                            # Extract relevant value based on metric type
                            if metric_type == 'performance':
                                daily_value += metric_data.get('response_time_ms', 0)
                            elif metric_type == 'quality':
                                daily_value += metric_data.get('quality_score', 0)
                        except Exception:
                            continue
                    
                    daily_value = daily_value / len(daily_data) if daily_data else 0.0
                
                trends['daily_values'].append({
                    'date': date_obj.isoformat(),
                    'value': daily_value
                })
            
            # Calculate trend direction
            if len(trends['daily_values']) >= 2:
                recent_avg = statistics.mean([d['value'] for d in trends['daily_values'][:3]])
                older_avg = statistics.mean([d['value'] for d in trends['daily_values'][-3:]])
                
                if recent_avg > older_avg * 1.1:
                    trends['trend_direction'] = 'increasing'
                elif recent_avg < older_avg * 0.9:
                    trends['trend_direction'] = 'decreasing'
                
                trends['percentage_change'] = ((recent_avg - older_avg) / older_avg * 100) if older_avg > 0 else 0.0
            
            return trends
            
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            return {"error": str(e)}
    
    async def get_active_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get active system alerts."""
        # Get recent alerts from buffer
        recent_alerts = list(self._alert_buffer)[-limit:]
        
        # Filter unresolved alerts
        active_alerts = [
            alert.to_dict() for alert in recent_alerts 
            if not alert.resolved
        ]
        
        # Sort by severity and timestamp
        severity_order = {
            AlertSeverity.EMERGENCY: 4,
            AlertSeverity.CRITICAL: 3,
            AlertSeverity.WARNING: 2,
            AlertSeverity.INFO: 1
        }
        
        active_alerts.sort(
            key=lambda a: (severity_order.get(AlertSeverity(a['severity']), 0), a['timestamp']),
            reverse=True
        )
        
        return active_alerts
    
    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        try:
            # Find alert in buffer
            for alert in self._alert_buffer:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    break
            
            # Update in Redis
            if self.redis_manager:
                alert_key = f"llm_alerts:{alert_id}"
                alert_data = await self.redis_manager.get(alert_key)
                
                if alert_data:
                    alert_dict = json.loads(alert_data)
                    alert_dict['acknowledged'] = True
                    await self.redis_manager.setex(alert_key, 86400 * 7, json.dumps(alert_dict))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to acknowledge alert {alert_id}: {e}")
            return False
    
    async def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        await self.initialize()
        
        return {
            'monitoring_status': 'active',
            'metrics_collected': {
                'performance_metrics': len(self._performance_buffer),
                'quality_metrics': len(self._quality_buffer),
                'active_alerts': len([a for a in self._alert_buffer if not a.resolved])
            },
            'alert_thresholds': self.alert_thresholds,
            'monitoring_uptime': 'N/A',  # Would track actual uptime
            'last_health_check': datetime.now().isoformat()
        }
    
    async def shutdown(self):
        """Shutdown the monitoring service."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("LLM Monitoring Service shutdown")


# Global monitoring service instance
_monitoring_service: Optional[LLMMonitoringService] = None


async def get_monitoring_service() -> LLMMonitoringService:
    """Get or create the global monitoring service."""
    global _monitoring_service
    
    if _monitoring_service is None:
        _monitoring_service = LLMMonitoringService()
        await _monitoring_service.initialize()
    
    return _monitoring_service