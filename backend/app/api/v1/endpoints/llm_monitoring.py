"""
LLM Monitoring API Endpoints - REST API for monitoring and analytics.

This module provides endpoints for:
- System health and status monitoring
- Performance metrics and analytics  
- Cost tracking and budget monitoring
- Alert management and notifications
- Trend analysis and forecasting
- Dashboard data aggregation
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, Query, Path, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import logging

from ....services.llm_monitoring import get_monitoring_service, MetricType, AlertSeverity
from ....services.llm_cost_manager import get_cost_manager
from ....services.llm_cache import get_cache_manager  
from ....services.llm_fallback import get_fallback_service
from ....services.llm_service_enhanced import get_enhanced_llm_service
from ....core.config import get_settings
from ....utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


# Request/Response Models

class AlertAcknowledgeRequest(BaseModel):
    """Request to acknowledge an alert."""
    alert_id: str = Field(..., description="Alert ID to acknowledge")
    acknowledged_by: Optional[str] = Field(None, description="User who acknowledged the alert")


class DashboardResponse(BaseModel):
    """Dashboard data response."""
    timestamp: str
    time_period: str
    performance: Dict[str, Any]
    costs: Dict[str, Any]  
    quality: Dict[str, Any]
    usage: Dict[str, Any]
    cache: Dict[str, Any]
    fallback: Dict[str, Any]
    alerts_summary: Dict[str, Any]


# Dependency functions

async def get_settings_dep():
    """Dependency to get application settings."""
    return get_settings()


# API Endpoints

@router.get("/dashboard",
            response_model=DashboardResponse,
            summary="Get Dashboard Data",
            description="Get comprehensive dashboard data with system metrics and analytics")
async def get_dashboard(
    time_period: str = Query("last_hour", description="Time period (last_hour, last_day, last_week, last_month)"),
    include_trends: bool = Query(False, description="Include trend analysis"),
    settings = Depends(get_settings_dep)
):
    """Get dashboard data for monitoring interface."""
    
    try:
        monitoring_service = await get_monitoring_service()
        dashboard_data = await monitoring_service.get_dashboard_data(time_period)
        
        # Get active alerts summary
        active_alerts = await monitoring_service.get_active_alerts(limit=10)
        alerts_by_severity = {}
        for alert in active_alerts:
            severity = alert['severity']
            alerts_by_severity[severity] = alerts_by_severity.get(severity, 0) + 1
        
        # Prepare response
        response_data = {
            "timestamp": dashboard_data.timestamp.isoformat(),
            "time_period": dashboard_data.time_period,
            "performance": {
                "avg_response_time_ms": dashboard_data.avg_response_time_ms,
                "total_requests": dashboard_data.total_requests,
                "success_rate": dashboard_data.success_rate,
                "error_rate": dashboard_data.error_rate,
                "requests_by_model": dashboard_data.requests_by_model,
                "requests_by_category": dashboard_data.requests_by_category,
                "peak_usage_hour": dashboard_data.peak_usage_hour
            },
            "costs": {
                "total_cost": dashboard_data.total_cost,
                "cost_per_request": dashboard_data.cost_per_request,
                "budget_utilization": dashboard_data.budget_utilization
            },
            "quality": {
                "avg_quality_score": dashboard_data.avg_quality_score,
                "avg_confidence_score": dashboard_data.avg_confidence_score
            },
            "usage": {
                "requests_by_model": dashboard_data.requests_by_model,
                "requests_by_category": dashboard_data.requests_by_category,
                "peak_usage_hour": dashboard_data.peak_usage_hour
            },
            "cache": {
                "hit_rate": dashboard_data.cache_hit_rate,
                "savings": dashboard_data.cache_savings
            },
            "fallback": {
                "usage_rate": dashboard_data.fallback_usage_rate,
                "circuit_breaker_status": dashboard_data.circuit_breaker_status
            },
            "alerts_summary": {
                "total_active": len(active_alerts),
                "by_severity": alerts_by_severity
            }
        }
        
        # Add trend analysis if requested
        if include_trends:
            try:
                performance_trends = await monitoring_service.get_trend_analysis("performance", days=7)
                quality_trends = await monitoring_service.get_trend_analysis("quality", days=7)
                
                response_data["trends"] = {
                    "performance": performance_trends,
                    "quality": quality_trends
                }
            except Exception as e:
                logger.warning(f"Failed to include trend analysis: {e}")
                response_data["trends"] = {"error": "Trend analysis unavailable"}
        
        return DashboardResponse(**response_data)
    
    except Exception as e:
        logger.error(f"Dashboard data retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve dashboard data")


@router.get("/health",
            summary="System Health Check",
            description="Get comprehensive health status of all LLM services")
async def get_system_health(
    include_details: bool = Query(True, description="Include detailed health information"),
    settings = Depends(get_settings_dep)
):
    """Get system health status."""
    
    try:
        # Get enhanced service health (includes fallback system)
        enhanced_service = await get_enhanced_llm_service()
        health_data = await enhanced_service.health_check_with_fallback()
        
        # Get cost manager status
        cost_manager = await get_cost_manager()
        daily_cost = await cost_manager.get_daily_cost()
        monthly_cost = await cost_manager.get_monthly_cost()
        budget_status = cost_manager.get_budget_status()
        
        # Get monitoring service status
        monitoring_service = await get_monitoring_service()
        monitoring_stats = await monitoring_service.get_system_statistics()
        
        health_response = {
            "overall_status": health_data.get("status", "unknown"),
            "timestamp": datetime.now().isoformat(),
            "services": {
                "llm_service": {
                    "status": "healthy" if health_data.get("status") == "healthy" else "degraded",
                    "fallback_enabled": health_data.get("fallback_enabled", False),
                    "providers": health_data.get("fallback_system", {}).get("providers", {})
                },
                "monitoring": {
                    "status": "active" if monitoring_stats.get("monitoring_status") == "active" else "inactive",
                    "metrics_collected": monitoring_stats.get("metrics_collected", {}),
                    "active_alerts": monitoring_stats.get("metrics_collected", {}).get("active_alerts", 0)
                },
                "cost_tracking": {
                    "status": "enabled" if budget_status.get("cost_tracking_enabled") else "disabled",
                    "daily_cost": daily_cost,
                    "monthly_cost": monthly_cost,
                    "budget_limits": {
                        "daily": budget_status.get("daily_budget_limit", 0),
                        "monthly": budget_status.get("monthly_budget_limit", 0)
                    }
                }
            }
        }
        
        if include_details:
            health_response["detailed_status"] = health_data
            health_response["monitoring_details"] = monitoring_stats
        
        return health_response
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "overall_status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
        )


@router.get("/metrics/performance",
            summary="Get Performance Metrics",
            description="Get detailed performance metrics and statistics")
async def get_performance_metrics(
    time_period: str = Query("last_hour", description="Time period for metrics"),
    include_breakdown: bool = Query(True, description="Include breakdown by model/provider"),
    settings = Depends(get_settings_dep)
):
    """Get performance metrics."""
    
    try:
        monitoring_service = await get_monitoring_service()
        dashboard_data = await monitoring_service.get_dashboard_data(time_period)
        
        metrics = {
            "time_period": time_period,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_requests": dashboard_data.total_requests,
                "avg_response_time_ms": dashboard_data.avg_response_time_ms,
                "success_rate": dashboard_data.success_rate,
                "error_rate": dashboard_data.error_rate
            },
            "usage_patterns": {
                "requests_by_model": dashboard_data.requests_by_model,
                "requests_by_category": dashboard_data.requests_by_category,
                "peak_usage_hour": dashboard_data.peak_usage_hour
            }
        }
        
        if include_breakdown:
            # Get trend analysis for performance
            trends = await monitoring_service.get_trend_analysis("performance", days=7)
            metrics["trends"] = trends
        
        return metrics
    
    except Exception as e:
        logger.error(f"Performance metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve performance metrics")


@router.get("/metrics/costs",
            summary="Get Cost Metrics",
            description="Get detailed cost metrics and budget analysis")
async def get_cost_metrics(
    days: int = Query(30, ge=1, le=365, description="Number of days for cost analysis"),
    include_optimization: bool = Query(True, description="Include cost optimization recommendations"),
    settings = Depends(get_settings_dep)
):
    """Get cost metrics and analysis."""
    
    try:
        cost_manager = await get_cost_manager()
        
        # Get cost analytics
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        analytics = await cost_manager.get_cost_analytics(start_date, end_date)
        
        # Get budget status
        budget_status = cost_manager.get_budget_status()
        
        # Get current costs
        daily_cost = await cost_manager.get_daily_cost()
        monthly_cost = await cost_manager.get_monthly_cost()
        
        cost_metrics = {
            "period": {
                "start_date": analytics.period_start.isoformat(),
                "end_date": analytics.period_end.isoformat(),
                "days": days
            },
            "current_costs": {
                "daily_cost": daily_cost,
                "monthly_cost": monthly_cost,
                "daily_budget_utilization": daily_cost / budget_status["daily_budget_limit"] if budget_status["daily_budget_limit"] > 0 else 0,
                "monthly_budget_utilization": monthly_cost / budget_status["monthly_budget_limit"] if budget_status["monthly_budget_limit"] > 0 else 0
            },
            "analytics": {
                "total_cost": analytics.total_cost,
                "total_requests": analytics.total_requests,
                "avg_cost_per_request": analytics.avg_cost_per_request,
                "cost_by_category": {k.value if hasattr(k, 'value') else str(k): v for k, v in analytics.cost_by_category.items()},
                "cost_by_model": analytics.cost_by_model,
                "cost_by_provider": analytics.cost_by_provider
            },
            "efficiency": {
                "cache_hit_rate": analytics.cache_hit_rate,
                "cost_savings_from_cache": analytics.cost_savings_from_cache,
                "token_usage": analytics.token_usage
            },
            "budget_settings": budget_status
        }
        
        if include_optimization:
            recommendations = await cost_manager.get_cost_optimization_recommendations()
            cost_metrics["optimization_recommendations"] = recommendations
        
        return cost_metrics
    
    except Exception as e:
        logger.error(f"Cost metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve cost metrics")


@router.get("/alerts",
            summary="Get System Alerts",
            description="Get active and recent system alerts")
async def get_alerts(
    limit: int = Query(50, ge=1, le=200, description="Maximum number of alerts to return"),
    severity: Optional[str] = Query(None, description="Filter by severity (info, warning, critical, emergency)"),
    category: Optional[str] = Query(None, description="Filter by category"),
    unresolved_only: bool = Query(True, description="Show only unresolved alerts"),
    settings = Depends(get_settings_dep)
):
    """Get system alerts."""
    
    try:
        monitoring_service = await get_monitoring_service()
        alerts = await monitoring_service.get_active_alerts(limit=limit)
        
        # Apply filters
        if severity:
            alerts = [a for a in alerts if a.get('severity') == severity]
        
        if category:
            alerts = [a for a in alerts if a.get('category') == category]
        
        if unresolved_only:
            alerts = [a for a in alerts if not a.get('resolved', False)]
        
        # Group alerts by category for summary
        alerts_by_category = {}
        alerts_by_severity = {}
        
        for alert in alerts:
            cat = alert.get('category', 'unknown')
            sev = alert.get('severity', 'info')
            
            alerts_by_category[cat] = alerts_by_category.get(cat, 0) + 1
            alerts_by_severity[sev] = alerts_by_severity.get(sev, 0) + 1
        
        return {
            "total_alerts": len(alerts),
            "alerts": alerts[:limit],
            "summary": {
                "by_category": alerts_by_category,
                "by_severity": alerts_by_severity
            },
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Alerts retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve alerts")


@router.post("/alerts/{alert_id}/acknowledge",
             summary="Acknowledge Alert",
             description="Acknowledge a system alert")
async def acknowledge_alert(
    alert_id: str = Path(..., description="Alert ID"),
    request: Optional[AlertAcknowledgeRequest] = None,
    settings = Depends(get_settings_dep)
):
    """Acknowledge an alert."""
    
    try:
        monitoring_service = await get_monitoring_service()
        success = await monitoring_service.acknowledge_alert(alert_id)
        
        if success:
            return {
                "status": "acknowledged",
                "alert_id": alert_id,
                "acknowledged_at": datetime.now().isoformat(),
                "acknowledged_by": request.acknowledged_by if request else "unknown"
            }
        else:
            raise HTTPException(status_code=404, detail="Alert not found")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Alert acknowledgment failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to acknowledge alert")


@router.get("/trends/{metric_type}",
            summary="Get Trend Analysis",
            description="Get trend analysis for specific metric types")
async def get_trend_analysis(
    metric_type: str = Path(..., description="Metric type (performance, quality, cost)"),
    days: int = Query(7, ge=1, le=90, description="Number of days for trend analysis"),
    include_forecast: bool = Query(False, description="Include simple forecast"),
    settings = Depends(get_settings_dep)
):
    """Get trend analysis for metrics."""
    
    try:
        monitoring_service = await get_monitoring_service()
        trends = await monitoring_service.get_trend_analysis(metric_type, days=days)
        
        if "error" in trends:
            raise HTTPException(status_code=500, detail=trends["error"])
        
        # Add simple forecast if requested
        if include_forecast and len(trends.get("daily_values", [])) >= 3:
            try:
                # Simple linear trend forecast (basic implementation)
                values = [d["value"] for d in trends["daily_values"][-7:]]  # Last 7 days
                if len(values) >= 3:
                    # Simple moving average forecast
                    recent_avg = sum(values[-3:]) / 3
                    overall_avg = sum(values) / len(values)
                    trend_factor = recent_avg / overall_avg if overall_avg > 0 else 1.0
                    
                    forecast = []
                    for i in range(1, 4):  # 3-day forecast
                        forecast_date = datetime.now().date() + timedelta(days=i)
                        forecast_value = recent_avg * (trend_factor ** i)
                        forecast.append({
                            "date": forecast_date.isoformat(),
                            "value": forecast_value,
                            "type": "forecast"
                        })
                    
                    trends["forecast"] = forecast
            except Exception as e:
                logger.warning(f"Forecast calculation failed: {e}")
                trends["forecast"] = []
        
        return trends
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Trend analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve trend analysis")


@router.get("/cache/stats",
            summary="Get Cache Statistics", 
            description="Get detailed cache performance statistics")
async def get_cache_statistics(
    reset_stats: bool = Query(False, description="Reset cache statistics after retrieval"),
    settings = Depends(get_settings_dep)
):
    """Get cache performance statistics."""
    
    try:
        cache_manager = await get_cache_manager()
        cache_stats = await cache_manager.get_cache_stats()
        
        stats = {
            "performance": {
                "hit_rate": cache_stats.hit_rate,
                "semantic_hit_rate": cache_stats.semantic_hit_rate,
                "total_requests": cache_stats.total_requests,
                "cache_hits": cache_stats.cache_hits,
                "cache_misses": cache_stats.cache_misses,
                "exact_hits": cache_stats.exact_hits,
                "semantic_hits": cache_stats.semantic_hits,
                "avg_response_time_ms": cache_stats.avg_response_time_ms
            },
            "storage": {
                "total_cache_entries": cache_stats.total_cache_entries,
                "cache_size_mb": cache_stats.cache_size_mb
            },
            "economics": {
                "cost_savings": cache_stats.cost_savings,
                "estimated_requests_saved": cache_stats.cache_hits
            },
            "distribution": {
                "top_cached_models": cache_stats.top_cached_models
            },
            "timestamp": datetime.now().isoformat()
        }
        
        if reset_stats:
            try:
                cache_manager.reset_stats()
                stats["stats_reset"] = True
            except Exception as e:
                logger.warning(f"Failed to reset cache stats: {e}")
                stats["stats_reset"] = False
        
        return stats
    
    except Exception as e:
        logger.error(f"Cache statistics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve cache statistics")


@router.post("/circuit-breaker/{provider}/reset",
             summary="Reset Circuit Breaker",
             description="Manually reset circuit breaker for a provider")
async def reset_circuit_breaker(
    provider: str = Path(..., description="Provider name (openai, anthropic, ollama)"),
    settings = Depends(get_settings_dep)
):
    """Reset circuit breaker for a provider."""
    
    try:
        enhanced_service = await get_enhanced_llm_service()
        success = await enhanced_service.reset_circuit_breaker(provider)
        
        if success:
            return {
                "status": "reset",
                "provider": provider,
                "reset_at": datetime.now().isoformat(),
                "message": f"Circuit breaker reset for provider {provider}"
            }
        else:
            raise HTTPException(status_code=404, detail=f"Provider {provider} not found or reset failed")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Circuit breaker reset failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset circuit breaker")


@router.get("/system/stats",
            summary="Get System Statistics",
            description="Get comprehensive system statistics and monitoring info")
async def get_system_statistics(
    settings = Depends(get_settings_dep)
):
    """Get system statistics."""
    
    try:
        monitoring_service = await get_monitoring_service()
        stats = await monitoring_service.get_system_statistics()
        
        # Get enhanced service stats
        enhanced_service = await get_enhanced_llm_service()
        usage_stats = enhanced_service.get_enhanced_usage_stats()
        
        return {
            "monitoring": stats,
            "llm_service": usage_stats,
            "system_info": {
                "timestamp": datetime.now().isoformat(),
                "monitoring_active": stats.get("monitoring_status") == "active",
                "fallback_enabled": usage_stats.get("enhanced_features", {}).get("fallback_enabled", False)
            }
        }
    
    except Exception as e:
        logger.error(f"System statistics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system statistics")