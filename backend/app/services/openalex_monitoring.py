"""
Monitoring and metrics collection for OpenAlex API client.

Provides comprehensive monitoring, alerting, and performance metrics
for the OpenAlex integration including API health, rate limiting,
cache performance, and data quality metrics.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque

from ..core.config import get_settings
from ..db.redis import get_redis_manager
from ..services.openalex import get_openalex_client
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MetricWindow:
    """Rolling window for metric tracking."""
    
    values: deque = field(default_factory=lambda: deque(maxlen=100))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def add_value(self, value: float, timestamp: Optional[datetime] = None):
        """Add a value to the window."""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        self.values.append(value)
        self.timestamps.append(timestamp)
    
    def get_recent_values(self, minutes: int = 60) -> List[float]:
        """Get values from the last N minutes."""
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        recent_values = []
        
        for value, timestamp in zip(self.values, self.timestamps):
            if timestamp >= cutoff:
                recent_values.append(value)
        
        return recent_values
    
    def get_avg(self, minutes: int = 60) -> Optional[float]:
        """Get average value over the last N minutes."""
        recent = self.get_recent_values(minutes)
        return sum(recent) / len(recent) if recent else None
    
    def get_percentile(self, percentile: float, minutes: int = 60) -> Optional[float]:
        """Get percentile value over the last N minutes."""
        recent = self.get_recent_values(minutes)
        if not recent:
            return None
        
        sorted_values = sorted(recent)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]


class OpenAlexMonitor:
    """Comprehensive monitoring for OpenAlex client operations."""
    
    def __init__(self):
        self.settings = get_settings()
        
        # Metric windows for different types of measurements
        self.response_times = MetricWindow()
        self.error_rates = MetricWindow()
        self.cache_hit_rates = MetricWindow()
        self.rate_limit_usage = MetricWindow()
        self.throughput = MetricWindow()
        
        # Counters and tracking
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.rate_limit_hits = 0
        
        # Error tracking
        self.error_types = defaultdict(int)
        self.recent_errors = deque(maxlen=50)
        
        # Performance tracking
        self.slowest_endpoints = defaultdict(list)
        self.endpoint_stats = defaultdict(lambda: {"count": 0, "total_time": 0, "errors": 0})
        
        # Alerts and thresholds
        self.alert_thresholds = {
            "error_rate_percent": 5.0,  # Alert if error rate > 5%
            "response_time_p95_ms": 5000,  # Alert if P95 response time > 5s
            "cache_hit_rate_percent": 80.0,  # Alert if cache hit rate < 80%
            "rate_limit_usage_percent": 90.0,  # Alert if rate limit usage > 90%
        }
        
        self.active_alerts = set()
        self.last_health_check = None
        
        logger.info("OpenAlex monitor initialized")
    
    def record_request_start(self, endpoint: str, method: str = "GET") -> str:
        """Record the start of an API request."""
        request_id = f"{endpoint}_{int(time.time() * 1000)}_{id(self)}"
        
        # Store request start time
        self._request_starts = getattr(self, '_request_starts', {})
        self._request_starts[request_id] = {
            "endpoint": endpoint,
            "method": method,
            "start_time": time.time(),
            "timestamp": datetime.utcnow()
        }
        
        return request_id
    
    def record_request_success(
        self,
        request_id: str,
        response_size: Optional[int] = None,
        cache_hit: bool = False
    ):
        """Record a successful API request."""
        if not hasattr(self, '_request_starts') or request_id not in self._request_starts:
            return
        
        start_info = self._request_starts.pop(request_id)
        response_time = (time.time() - start_info["start_time"]) * 1000  # Convert to milliseconds
        
        # Update counters
        self.total_requests += 1
        self.successful_requests += 1
        
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        
        # Update metrics
        self.response_times.add_value(response_time, start_info["timestamp"])
        self.cache_hit_rates.add_value(100.0 if cache_hit else 0.0, start_info["timestamp"])
        self.throughput.add_value(1.0, start_info["timestamp"])  # 1 request per timestamp
        
        # Update endpoint stats
        endpoint = start_info["endpoint"]
        self.endpoint_stats[endpoint]["count"] += 1
        self.endpoint_stats[endpoint]["total_time"] += response_time
        
        # Track slow endpoints
        if response_time > 2000:  # > 2 seconds
            self.slowest_endpoints[endpoint].append({
                "response_time": response_time,
                "timestamp": start_info["timestamp"],
                "cache_hit": cache_hit
            })
            # Keep only last 10 slow requests per endpoint
            if len(self.slowest_endpoints[endpoint]) > 10:
                self.slowest_endpoints[endpoint].pop(0)
        
        logger.debug(f"Request {request_id} completed in {response_time:.2f}ms (cache_hit: {cache_hit})")
    
    def record_request_error(
        self,
        request_id: str,
        error_type: str,
        error_message: str,
        status_code: Optional[int] = None
    ):
        """Record a failed API request."""
        if not hasattr(self, '_request_starts') or request_id not in self._request_starts:
            return
        
        start_info = self._request_starts.pop(request_id)
        response_time = (time.time() - start_info["start_time"]) * 1000
        
        # Update counters
        self.total_requests += 1
        self.failed_requests += 1
        self.error_types[error_type] += 1
        
        # Update metrics
        self.response_times.add_value(response_time, start_info["timestamp"])
        self.error_rates.add_value(100.0, start_info["timestamp"])  # 100% error for this request
        
        # Update endpoint stats
        endpoint = start_info["endpoint"]
        self.endpoint_stats[endpoint]["count"] += 1
        self.endpoint_stats[endpoint]["total_time"] += response_time
        self.endpoint_stats[endpoint]["errors"] += 1
        
        # Record error details
        error_info = {
            "timestamp": start_info["timestamp"],
            "endpoint": endpoint,
            "error_type": error_type,
            "error_message": error_message,
            "status_code": status_code,
            "response_time": response_time
        }
        self.recent_errors.append(error_info)
        
        # Check for rate limiting
        if status_code == 429:
            self.rate_limit_hits += 1
        
        logger.warning(f"Request {request_id} failed: {error_type} - {error_message}")
    
    def record_rate_limit_status(self, remaining: int, total: int):
        """Record current rate limit status."""
        if total > 0:
            usage_percent = ((total - remaining) / total) * 100
            self.rate_limit_usage.add_value(usage_percent)
    
    async def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        now = datetime.utcnow()
        
        # Calculate rates and averages
        error_rate = 0.0
        if self.total_requests > 0:
            error_rate = (self.failed_requests / self.total_requests) * 100
        
        cache_hit_rate = 0.0
        total_cache_operations = self.cache_hits + self.cache_misses
        if total_cache_operations > 0:
            cache_hit_rate = (self.cache_hits / total_cache_operations) * 100
        
        # Get recent metric averages
        avg_response_time = self.response_times.get_avg(60)  # Last 60 minutes
        p95_response_time = self.response_times.get_percentile(95, 60)
        p99_response_time = self.response_times.get_percentile(99, 60)
        
        recent_error_rate = None
        recent_errors = self.error_rates.get_recent_values(60)
        if recent_errors:
            recent_error_rate = sum(recent_errors) / len(recent_errors)
        
        recent_throughput = len(self.throughput.get_recent_values(60))  # Requests per hour
        
        # Calculate endpoint performance
        endpoint_performance = {}
        for endpoint, stats in self.endpoint_stats.items():
            if stats["count"] > 0:
                endpoint_performance[endpoint] = {
                    "total_requests": stats["count"],
                    "total_errors": stats["errors"],
                    "error_rate": (stats["errors"] / stats["count"]) * 100,
                    "avg_response_time": stats["total_time"] / stats["count"],
                    "last_hour_requests": sum(1 for _, ts in zip(self.response_times.values, self.response_times.timestamps)
                                            if ts >= now - timedelta(hours=1))
                }
        
        return {
            "timestamp": now.isoformat(),
            "uptime_hours": (now - (self.last_health_check or now)).total_seconds() / 3600,
            
            # Request statistics
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "error_rate_percent": error_rate,
            "recent_error_rate_percent": recent_error_rate,
            
            # Performance metrics
            "avg_response_time_ms": avg_response_time,
            "p95_response_time_ms": p95_response_time,
            "p99_response_time_ms": p99_response_time,
            "throughput_requests_per_hour": recent_throughput,
            
            # Cache performance
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate_percent": cache_hit_rate,
            
            # Rate limiting
            "rate_limit_hits": self.rate_limit_hits,
            "current_rate_limit_usage_percent": self.rate_limit_usage.get_recent_values(5)[-1] if self.rate_limit_usage.values else 0,
            
            # Error breakdown
            "error_types": dict(self.error_types),
            "recent_errors": list(self.recent_errors)[-10:],  # Last 10 errors
            
            # Endpoint performance
            "endpoint_performance": endpoint_performance,
            
            # Slow requests
            "slowest_endpoints": {
                endpoint: requests[-5:]  # Last 5 slow requests per endpoint
                for endpoint, requests in self.slowest_endpoints.items()
            },
            
            # Alert status
            "active_alerts": list(self.active_alerts),
            "alert_thresholds": self.alert_thresholds
        }
    
    async def check_health_and_alerts(self) -> Dict[str, Any]:
        """Check system health and update alerts."""
        metrics = await self.get_current_metrics()
        self.last_health_check = datetime.utcnow()
        
        # Clear old alerts
        old_alerts = self.active_alerts.copy()
        self.active_alerts.clear()
        
        # Check thresholds and generate alerts
        if metrics["recent_error_rate_percent"] and metrics["recent_error_rate_percent"] > self.alert_thresholds["error_rate_percent"]:
            self.active_alerts.add("high_error_rate")
        
        if metrics["p95_response_time_ms"] and metrics["p95_response_time_ms"] > self.alert_thresholds["response_time_p95_ms"]:
            self.active_alerts.add("high_response_time")
        
        if metrics["cache_hit_rate_percent"] < self.alert_thresholds["cache_hit_rate_percent"]:
            self.active_alerts.add("low_cache_hit_rate")
        
        if metrics["current_rate_limit_usage_percent"] > self.alert_thresholds["rate_limit_usage_percent"]:
            self.active_alerts.add("high_rate_limit_usage")
        
        # Log new alerts
        new_alerts = self.active_alerts - old_alerts
        if new_alerts:
            logger.warning(f"New alerts triggered: {new_alerts}")
        
        resolved_alerts = old_alerts - self.active_alerts
        if resolved_alerts:
            logger.info(f"Alerts resolved: {resolved_alerts}")
        
        # Determine overall health
        health_status = "healthy"
        if "high_error_rate" in self.active_alerts or "high_response_time" in self.active_alerts:
            health_status = "degraded"
        
        if metrics["recent_error_rate_percent"] and metrics["recent_error_rate_percent"] > 20:
            health_status = "unhealthy"
        
        return {
            "status": health_status,
            "timestamp": self.last_health_check.isoformat(),
            "metrics": metrics,
            "alerts": {
                "active": list(self.active_alerts),
                "new": list(new_alerts),
                "resolved": list(resolved_alerts)
            },
            "recommendations": self._generate_recommendations(metrics)
        }
    
    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations based on metrics."""
        recommendations = []
        
        if metrics["cache_hit_rate_percent"] < 70:
            recommendations.append("Consider increasing cache TTL or reviewing cache key strategy")
        
        if metrics["avg_response_time_ms"] and metrics["avg_response_time_ms"] > 3000:
            recommendations.append("High average response times detected - check network connectivity and API performance")
        
        if metrics["current_rate_limit_usage_percent"] > 80:
            recommendations.append("Approaching rate limits - consider implementing request queuing or upgrading to polite pool")
        
        if metrics["error_rate_percent"] > 10:
            recommendations.append("High error rates detected - check API credentials and network connectivity")
        
        # Check for endpoint-specific issues
        for endpoint, perf in metrics["endpoint_performance"].items():
            if perf["error_rate"] > 20:
                recommendations.append(f"High error rate for {endpoint} endpoint - investigate specific issues")
            
            if perf["avg_response_time"] > 5000:
                recommendations.append(f"Slow performance for {endpoint} endpoint - consider optimization")
        
        return recommendations
    
    async def export_metrics_to_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        metrics = await self.get_current_metrics()
        prometheus_lines = []
        
        # Request metrics
        prometheus_lines.extend([
            f"# HELP openalex_requests_total Total number of requests to OpenAlex API",
            f"# TYPE openalex_requests_total counter",
            f"openalex_requests_total {metrics['total_requests']}",
            f"",
            f"# HELP openalex_request_duration_milliseconds Request duration in milliseconds",
            f"# TYPE openalex_request_duration_milliseconds histogram",
            f"openalex_request_duration_milliseconds_sum {sum(self.response_times.values)}",
            f"openalex_request_duration_milliseconds_count {len(self.response_times.values)}",
        ])
        
        # Error metrics
        prometheus_lines.extend([
            f"# HELP openalex_errors_total Total number of API errors",
            f"# TYPE openalex_errors_total counter",
            f"openalex_errors_total {metrics['failed_requests']}",
            f"",
            f"# HELP openalex_error_rate_percent Current error rate percentage",
            f"# TYPE openalex_error_rate_percent gauge",
            f"openalex_error_rate_percent {metrics['error_rate_percent']}",
        ])
        
        # Cache metrics
        prometheus_lines.extend([
            f"# HELP openalex_cache_hits_total Total cache hits",
            f"# TYPE openalex_cache_hits_total counter",
            f"openalex_cache_hits_total {metrics['cache_hits']}",
            f"",
            f"# HELP openalex_cache_hit_rate_percent Cache hit rate percentage",
            f"# TYPE openalex_cache_hit_rate_percent gauge",
            f"openalex_cache_hit_rate_percent {metrics['cache_hit_rate_percent']}",
        ])
        
        # Rate limit metrics
        prometheus_lines.extend([
            f"# HELP openalex_rate_limit_usage_percent Current rate limit usage percentage",
            f"# TYPE openalex_rate_limit_usage_percent gauge",
            f"openalex_rate_limit_usage_percent {metrics['current_rate_limit_usage_percent']}",
        ])
        
        return "\n".join(prometheus_lines)


# Global monitor instance
_openalex_monitor: Optional[OpenAlexMonitor] = None


def get_openalex_monitor() -> OpenAlexMonitor:
    """Get or create the global OpenAlex monitor instance."""
    global _openalex_monitor
    
    if _openalex_monitor is None:
        _openalex_monitor = OpenAlexMonitor()
    
    return _openalex_monitor


async def collect_openalex_health_metrics() -> Dict[str, Any]:
    """Collect comprehensive health metrics for OpenAlex integration."""
    monitor = get_openalex_monitor()
    
    try:
        # Get OpenAlex client health
        client = await get_openalex_client()
        client_health = await client.health_check()
        
        # Get monitor metrics
        monitor_health = await monitor.check_health_and_alerts()
        
        # Get Redis health (for caching)
        redis_manager = await get_redis_manager()
        redis_health = await redis_manager.health_check()
        
        # Combine all health data
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": _determine_overall_status([
                client_health.get("status", "unknown"),
                monitor_health.get("status", "unknown"),
                redis_health.get("status", "unknown")
            ]),
            "components": {
                "openalex_api": client_health,
                "monitoring": monitor_health,
                "cache": redis_health
            },
            "integration_metrics": {
                "total_papers_imported": await _get_paper_import_stats(),
                "citation_networks_built": await _get_network_stats(),
                "data_freshness": await _get_data_freshness_stats()
            }
        }
        
    except Exception as e:
        logger.error(f"Error collecting OpenAlex health metrics: {e}")
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "error",
            "error": str(e),
            "components": {}
        }


def _determine_overall_status(statuses: List[str]) -> str:
    """Determine overall health status from component statuses."""
    if "unhealthy" in statuses or "error" in statuses:
        return "unhealthy"
    elif "degraded" in statuses:
        return "degraded"
    elif all(status == "healthy" for status in statuses):
        return "healthy"
    else:
        return "unknown"


async def _get_paper_import_stats() -> Dict[str, Any]:
    """Get statistics about paper imports from OpenAlex."""
    # This would query the database for import statistics
    # Implementation depends on your database schema
    return {
        "total_imported": 0,  # Placeholder
        "last_24h": 0,
        "success_rate": 100.0
    }


async def _get_network_stats() -> Dict[str, Any]:
    """Get statistics about citation network building."""
    return {
        "networks_built": 0,  # Placeholder
        "avg_network_size": 0,
        "largest_network": 0
    }


async def _get_data_freshness_stats() -> Dict[str, Any]:
    """Get statistics about data freshness."""
    return {
        "oldest_paper_days": 0,  # Placeholder
        "avg_citation_age_days": 0,
        "last_sync_hours_ago": 0
    }