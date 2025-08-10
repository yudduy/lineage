"""
Metrics collection and monitoring service.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque
import asyncio

from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
from prometheus_client.core import CONTENT_TYPE_LATEST

from ..core.config import get_settings
from ..db.redis import RedisManager
from ..models.health import MetricsSnapshot, Alert, AlertRule
from ..utils.logger import get_logger

logger = get_logger(__name__)


class MetricsCollector:
    """Collect and store application metrics."""
    
    def __init__(self):
        self.settings = get_settings()
        self.registry = CollectorRegistry()
        self.logger = logger
        
        # Prometheus metrics
        self.request_count = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        self.active_connections = Gauge(
            'database_connections_active',
            'Active database connections',
            ['database'],
            registry=self.registry
        )
        
        self.cache_operations = Counter(
            'cache_operations_total',
            'Cache operations',
            ['operation', 'result'],
            registry=self.registry
        )
        
        self.external_api_calls = Counter(
            'external_api_calls_total',
            'External API calls',
            ['service', 'status_code'],
            registry=self.registry
        )
        
        self.external_api_duration = Histogram(
            'external_api_duration_seconds',
            'External API call duration',
            ['service'],
            registry=self.registry
        )
        
        self.search_operations = Counter(
            'search_operations_total',
            'Search operations',
            ['engine', 'status'],
            registry=self.registry
        )
        
        self.paper_operations = Counter(
            'paper_operations_total',
            'Paper operations',
            ['operation', 'status'],
            registry=self.registry
        )
        
        # In-memory metrics for quick access
        self.request_times = deque(maxlen=1000)  # Last 1000 request times
        self.error_count = 0
        self.total_requests = 0
        
        # Alert system
        self.alerts: List[Alert] = []
        self.alert_rules: List[AlertRule] = []
        
        self._setup_default_alert_rules()
    
    def _setup_default_alert_rules(self):
        """Setup default alerting rules."""
        self.alert_rules = [
            AlertRule(
                name="high_error_rate",
                metric="error_rate_percent",
                operator="gt",
                threshold=5.0,
                duration_minutes=5,
                severity="warning"
            ),
            AlertRule(
                name="high_response_time",
                metric="p95_response_time_ms",
                operator="gt",
                threshold=2000.0,
                duration_minutes=3,
                severity="warning"
            ),
            AlertRule(
                name="very_high_response_time",
                metric="p95_response_time_ms",
                operator="gt",
                threshold=5000.0,
                duration_minutes=1,
                severity="critical"
            ),
            AlertRule(
                name="database_connection_failure",
                metric="database_health",
                operator="eq",
                threshold=0.0,  # 0 = unhealthy
                duration_minutes=1,
                severity="critical"
            )
        ]
    
    def record_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration_seconds: float
    ):
        """Record HTTP request metrics."""
        # Prometheus metrics
        self.request_count.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code)
        ).inc()
        
        self.request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration_seconds)
        
        # In-memory metrics
        self.request_times.append({
            'timestamp': time.time(),
            'duration_ms': duration_seconds * 1000,
            'status_code': status_code
        })
        
        self.total_requests += 1
        if status_code >= 400:
            self.error_count += 1
    
    def record_database_connection(self, database: str, count: int):
        """Record database connection count."""
        self.active_connections.labels(database=database).set(count)
    
    def record_cache_operation(self, operation: str, hit: bool):
        """Record cache operation."""
        result = "hit" if hit else "miss"
        self.cache_operations.labels(operation=operation, result=result).inc()
    
    def record_external_api_call(
        self,
        service: str,
        status_code: int,
        duration_seconds: float
    ):
        """Record external API call metrics."""
        self.external_api_calls.labels(
            service=service,
            status_code=str(status_code)
        ).inc()
        
        self.external_api_duration.labels(service=service).observe(duration_seconds)
    
    def record_search_operation(self, engine: str, success: bool):
        """Record search operation metrics."""
        status = "success" if success else "error"
        self.search_operations.labels(engine=engine, status=status).inc()
    
    def record_paper_operation(self, operation: str, success: bool):
        """Record paper operation metrics."""
        status = "success" if success else "error"
        self.paper_operations.labels(operation=operation, status=status).inc()
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus-formatted metrics."""
        return generate_latest(self.registry).decode('utf-8')
    
    async def get_metrics_snapshot(self) -> MetricsSnapshot:
        """Get current metrics snapshot."""
        now = time.time()
        
        # Calculate request metrics from recent data
        recent_requests = [
            r for r in self.request_times
            if now - r['timestamp'] <= 60  # Last minute
        ]
        
        if recent_requests:
            requests_per_second = len(recent_requests) / 60.0
            response_times = [r['duration_ms'] for r in recent_requests]
            avg_response_time = sum(response_times) / len(response_times)
            
            # Calculate percentiles
            sorted_times = sorted(response_times)
            p95_index = int(0.95 * len(sorted_times))
            p99_index = int(0.99 * len(sorted_times))
            p95_response_time = sorted_times[p95_index] if sorted_times else 0
            p99_response_time = sorted_times[p99_index] if sorted_times else 0
            
            # Calculate error rate
            recent_errors = sum(1 for r in recent_requests if r['status_code'] >= 400)
            error_rate = (recent_errors / len(recent_requests)) * 100 if recent_requests else 0
        else:
            requests_per_second = 0
            avg_response_time = 0
            p95_response_time = 0
            p99_response_time = 0
            error_rate = 0
        
        return MetricsSnapshot(
            timestamp=datetime.utcnow(),
            total_requests=self.total_requests,
            requests_per_second=requests_per_second,
            average_response_time_ms=avg_response_time,
            p95_response_time_ms=p95_response_time,
            p99_response_time_ms=p99_response_time,
            total_errors=self.error_count,
            error_rate_percent=error_rate
        )


class MetricsService:
    """Service for managing metrics collection and alerting."""
    
    def __init__(self):
        self.settings = get_settings()
        self.collector = MetricsCollector()
        self.redis_manager: Optional[RedisManager] = None
        self.logger = logger
        
        # Background task for alert checking
        self._alert_check_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self):
        """Start the metrics service."""
        self.redis_manager = RedisManager()
        await self.redis_manager.connect()
        
        # Start background alert checking
        self._running = True
        self._alert_check_task = asyncio.create_task(self._alert_check_loop())
        
        self.logger.info("Metrics service started")
    
    async def stop(self):
        """Stop the metrics service."""
        self._running = False
        
        if self._alert_check_task:
            self._alert_check_task.cancel()
            try:
                await self._alert_check_task
            except asyncio.CancelledError:
                pass
        
        if self.redis_manager:
            await self.redis_manager.disconnect()
        
        self.logger.info("Metrics service stopped")
    
    async def _alert_check_loop(self):
        """Background loop for checking alert conditions."""
        while self._running:
            try:
                await self._check_alerts()
                await asyncio.sleep(30)  # Check every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in alert check loop: {e}", exc_info=True)
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _check_alerts(self):
        """Check all alert rules and trigger alerts if needed."""
        if not self.redis_manager:
            return
        
        current_metrics = await self.collector.get_metrics_snapshot()
        
        for rule in self.collector.alert_rules:
            if not rule.enabled:
                continue
            
            try:
                # Get metric value
                metric_value = getattr(current_metrics, rule.metric, None)
                if metric_value is None:
                    continue
                
                # Check threshold
                threshold_breached = False
                if rule.operator == "gt":
                    threshold_breached = metric_value > rule.threshold
                elif rule.operator == "gte":
                    threshold_breached = metric_value >= rule.threshold
                elif rule.operator == "lt":
                    threshold_breached = metric_value < rule.threshold
                elif rule.operator == "lte":
                    threshold_breached = metric_value <= rule.threshold
                elif rule.operator == "eq":
                    threshold_breached = metric_value == rule.threshold
                elif rule.operator == "neq":
                    threshold_breached = metric_value != rule.threshold
                
                if threshold_breached:
                    await self._trigger_alert(rule, metric_value, current_metrics.timestamp)
                else:
                    await self._resolve_alert(rule.name)
                    
            except Exception as e:
                self.logger.error(f"Error checking alert rule {rule.name}: {e}")
    
    async def _trigger_alert(
        self,
        rule: AlertRule,
        current_value: float,
        timestamp: datetime
    ):
        """Trigger an alert for a rule."""
        if not self.redis_manager:
            return
        
        alert_key = f"alert:{rule.name}"
        
        # Check if alert is already active
        existing_alert = await self.redis_manager.cache_get(alert_key)
        
        if not existing_alert:
            # Create new alert
            alert = Alert(
                id=f"{rule.name}_{int(timestamp.timestamp())}",
                rule_name=rule.name,
                metric=rule.metric,
                current_value=current_value,
                threshold=rule.threshold,
                severity=rule.severity,
                message=f"{rule.metric} is {current_value}, threshold is {rule.threshold}",
                started_at=timestamp
            )
            
            # Store alert
            await self.redis_manager.cache_set(
                alert_key,
                alert.model_dump(),
                expire=rule.duration_minutes * 60
            )
            
            self.collector.alerts.append(alert)
            
            # Log alert
            self.logger.warning(
                "Alert triggered",
                alert_id=alert.id,
                rule_name=rule.name,
                metric=rule.metric,
                current_value=current_value,
                threshold=rule.threshold,
                severity=rule.severity
            )
    
    async def _resolve_alert(self, rule_name: str):
        """Resolve an active alert."""
        if not self.redis_manager:
            return
        
        alert_key = f"alert:{rule_name}"
        
        # Check if alert exists
        existing_alert = await self.redis_manager.cache_get(alert_key)
        
        if existing_alert:
            # Remove from cache
            await self.redis_manager.cache_delete(alert_key)
            
            # Update in-memory alerts
            for alert in self.collector.alerts:
                if alert.rule_name == rule_name and alert.is_active:
                    alert.resolved_at = datetime.utcnow()
                    alert.is_active = False
                    
                    self.logger.info(
                        "Alert resolved",
                        alert_id=alert.id,
                        rule_name=rule_name,
                        duration_minutes=alert.duration_minutes
                    )
                    break
    
    # Metric recording methods (delegate to collector)
    def record_request(self, method: str, endpoint: str, status_code: int, duration_seconds: float):
        """Record HTTP request metrics."""
        self.collector.record_request(method, endpoint, status_code, duration_seconds)
    
    def record_database_connection(self, database: str, count: int):
        """Record database connection count."""
        self.collector.record_database_connection(database, count)
    
    def record_cache_operation(self, operation: str, hit: bool):
        """Record cache operation."""
        self.collector.record_cache_operation(operation, hit)
    
    def record_external_api_call(self, service: str, status_code: int, duration_seconds: float):
        """Record external API call metrics."""
        self.collector.record_external_api_call(service, status_code, duration_seconds)
    
    def record_search_operation(self, engine: str, success: bool):
        """Record search operation metrics."""
        self.collector.record_search_operation(engine, success)
    
    def record_paper_operation(self, operation: str, success: bool):
        """Record paper operation metrics."""
        self.collector.record_paper_operation(operation, success)
    
    async def get_metrics_snapshot(self) -> MetricsSnapshot:
        """Get current metrics snapshot."""
        return await self.collector.get_metrics_snapshot()
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus-formatted metrics."""
        return self.collector.get_prometheus_metrics()
    
    def get_active_alerts(self) -> List[Alert]:
        """Get currently active alerts."""
        return [alert for alert in self.collector.alerts if alert.is_active]
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history."""
        return sorted(
            self.collector.alerts,
            key=lambda a: a.started_at,
            reverse=True
        )[:limit]


# Global metrics service instance
metrics_service = MetricsService()


def get_metrics_service() -> MetricsService:
    """Dependency function to get metrics service."""
    return metrics_service