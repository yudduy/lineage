"""
Metrics and monitoring endpoints.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, Query
from prometheus_client.core import CONTENT_TYPE_LATEST
from fastapi.responses import Response

from ....models.health import MetricsSnapshot, Alert
from ....models.common import APIResponse
from ....models.user import User
from ....middleware.auth import get_current_user, require_scope
from ....services.metrics import MetricsService, get_metrics_service
from ....utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/", response_model=APIResponse[MetricsSnapshot])
async def get_metrics_snapshot(
    metrics_service: MetricsService = Depends(get_metrics_service),
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Get current application metrics snapshot.
    
    Returns real-time performance metrics including:
    - Request counts and response times
    - Error rates and status codes
    - Database and cache performance
    - System resource usage
    """
    logger.info(
        "Getting metrics snapshot",
        user_id=current_user.id if current_user else None
    )
    
    snapshot = await metrics_service.get_metrics_snapshot()
    
    return APIResponse(
        success=True,
        message="Metrics snapshot retrieved successfully",
        data=snapshot
    )


@router.get("/prometheus")
async def get_prometheus_metrics(
    metrics_service: MetricsService = Depends(get_metrics_service)
):
    """
    Get metrics in Prometheus format.
    
    This endpoint returns metrics in the standard Prometheus text format
    for scraping by Prometheus monitoring systems.
    
    No authentication required for monitoring systems.
    """
    metrics_data = metrics_service.get_prometheus_metrics()
    
    return Response(
        content=metrics_data,
        media_type=CONTENT_TYPE_LATEST,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )


@router.get("/alerts", response_model=APIResponse[List[Alert]])
async def get_active_alerts(
    severity: Optional[str] = Query(None, pattern="^(info|warning|error|critical)$", description="Filter by severity"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of alerts to return"),
    metrics_service: MetricsService = Depends(get_metrics_service),
    current_user: User = Depends(require_scope("admin"))
):
    """
    Get currently active alerts.
    
    Returns list of active monitoring alerts with their severity levels,
    trigger conditions, and current values.
    
    Requires admin permissions.
    """
    logger.info(
        "Getting active alerts",
        severity=severity,
        limit=limit,
        user_id=current_user.id
    )
    
    alerts = metrics_service.get_active_alerts()
    
    # Filter by severity if specified
    if severity:
        alerts = [alert for alert in alerts if alert.severity == severity]
    
    # Limit results
    alerts = alerts[:limit]
    
    return APIResponse(
        success=True,
        message=f"Retrieved {len(alerts)} active alerts",
        data=alerts
    )


@router.get("/alerts/history", response_model=APIResponse[List[Alert]])
async def get_alert_history(
    limit: int = Query(100, ge=1, le=500, description="Maximum number of alerts to return"),
    include_resolved: bool = Query(True, description="Include resolved alerts"),
    metrics_service: MetricsService = Depends(get_metrics_service),
    current_user: User = Depends(require_scope("admin"))
):
    """
    Get alert history.
    
    Returns historical alert data including resolved alerts,
    alert duration, and trigger patterns.
    
    Requires admin permissions.
    """
    logger.info(
        "Getting alert history",
        limit=limit,
        include_resolved=include_resolved,
        user_id=current_user.id
    )
    
    alert_history = metrics_service.get_alert_history(limit)
    
    # Filter resolved alerts if not requested
    if not include_resolved:
        alert_history = [alert for alert in alert_history if alert.is_active]
    
    return APIResponse(
        success=True,
        message=f"Retrieved {len(alert_history)} alerts from history",
        data=alert_history
    )


@router.get("/performance", response_model=APIResponse[dict])
async def get_performance_metrics(
    time_range: str = Query("1h", pattern="^(15m|1h|6h|24h|7d)$", description="Time range for metrics"),
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Get detailed performance metrics.
    
    Returns performance data over specified time range including:
    - Response time percentiles
    - Throughput metrics
    - Error rates by endpoint
    - Database query performance
    """
    logger.info(
        "Getting performance metrics",
        time_range=time_range,
        user_id=current_user.id if current_user else None
    )
    
    # TODO: Implement time-series performance data retrieval
    # This would typically come from a time-series database like InfluxDB or Prometheus
    
    # Mock performance data
    performance_data = {
        "time_range": time_range,
        "request_metrics": {
            "total_requests": 15420,
            "requests_per_minute": 85.6,
            "average_response_time_ms": 245.3,
            "p50_response_time_ms": 180.2,
            "p95_response_time_ms": 520.1,
            "p99_response_time_ms": 850.4
        },
        "error_metrics": {
            "total_errors": 123,
            "error_rate_percent": 0.8,
            "4xx_errors": 89,
            "5xx_errors": 34
        },
        "endpoint_performance": [
            {
                "endpoint": "/api/v1/papers/search",
                "request_count": 2340,
                "average_response_time_ms": 320.5,
                "error_count": 12
            },
            {
                "endpoint": "/api/v1/papers/{id}",
                "request_count": 5670,
                "average_response_time_ms": 150.2,
                "error_count": 8
            }
        ],
        "database_metrics": {
            "neo4j": {
                "query_count": 8920,
                "average_query_time_ms": 89.3,
                "slow_queries": 15
            },
            "redis": {
                "operations": 25600,
                "cache_hit_rate_percent": 87.5,
                "average_operation_time_ms": 2.1
            }
        }
    }
    
    return APIResponse(
        success=True,
        message=f"Performance metrics retrieved for {time_range}",
        data=performance_data
    )


@router.get("/usage", response_model=APIResponse[dict])
async def get_usage_statistics(
    period: str = Query("week", pattern="^(day|week|month|year)$", description="Usage period"),
    current_user: User = Depends(require_scope("admin"))
):
    """
    Get application usage statistics.
    
    Returns usage data including:
    - User activity metrics
    - Feature usage patterns
    - Popular search queries
    - API endpoint usage
    
    Requires admin permissions.
    """
    logger.info(
        "Getting usage statistics",
        period=period,
        user_id=current_user.id
    )
    
    # TODO: Implement usage statistics collection
    # This would aggregate data from user activity logs
    
    # Mock usage data
    usage_data = {
        "period": period,
        "user_metrics": {
            "total_users": 1250,
            "active_users": 890,
            "new_users": 45,
            "returning_users": 845
        },
        "search_metrics": {
            "total_searches": 12450,
            "unique_queries": 3200,
            "average_results_per_search": 8.5,
            "top_queries": [
                "machine learning",
                "neural networks",
                "deep learning",
                "artificial intelligence",
                "natural language processing"
            ]
        },
        "feature_usage": {
            "paper_searches": 8920,
            "citation_network_views": 3450,
            "zotero_exports": 234,
            "bulk_operations": 89
        },
        "api_usage": {
            "total_api_calls": 45600,
            "authenticated_calls": 32100,
            "rate_limited_calls": 156,
            "top_endpoints": [
                "/api/v1/papers/search",
                "/api/v1/papers/{id}",
                "/api/v1/papers/{id}/citations",
                "/api/v1/auth/token"
            ]
        }
    }
    
    return APIResponse(
        success=True,
        message=f"Usage statistics retrieved for {period}",
        data=usage_data
    )


@router.get("/system", response_model=APIResponse[dict])
async def get_system_metrics(
    current_user: User = Depends(require_scope("admin"))
):
    """
    Get system resource metrics.
    
    Returns system-level performance data including:
    - CPU and memory usage
    - Disk space utilization
    - Network statistics
    - Process information
    
    Requires admin permissions.
    """
    logger.info(
        "Getting system metrics",
        user_id=current_user.id
    )
    
    # TODO: Get actual system metrics using psutil or similar
    
    # Mock system data
    system_data = {
        "cpu": {
            "usage_percent": 45.2,
            "load_average": [1.2, 1.1, 0.9],
            "core_count": 8
        },
        "memory": {
            "total_mb": 16384,
            "used_mb": 8192,
            "available_mb": 8192,
            "usage_percent": 50.0
        },
        "disk": {
            "total_gb": 500,
            "used_gb": 200,
            "available_gb": 300,
            "usage_percent": 40.0
        },
        "network": {
            "bytes_sent": 1048576000,
            "bytes_received": 2097152000,
            "packets_sent": 500000,
            "packets_received": 750000
        },
        "processes": {
            "total_processes": 156,
            "running_processes": 3,
            "sleeping_processes": 148,
            "zombie_processes": 0
        }
    }
    
    return APIResponse(
        success=True,
        message="System metrics retrieved successfully",
        data=system_data
    )