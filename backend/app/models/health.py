"""
Health check and monitoring Pydantic models.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum
from pydantic import Field
from .base import BaseModel


class HealthStatus(str, Enum):
    """Health check status values."""
    
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    UNHEALTHY = "unhealthy"


class ServiceHealth(BaseModel):
    """Health status for individual service."""
    
    name: str = Field(description="Service name")
    status: HealthStatus = Field(description="Service health status")
    response_time_ms: Optional[float] = Field(default=None, description="Response time in milliseconds")
    last_check: datetime = Field(description="Last health check timestamp")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional health details")
    error: Optional[str] = Field(default=None, description="Error message if unhealthy")


class DatabaseHealth(BaseModel):
    """Database connection health status."""
    
    neo4j: ServiceHealth = Field(description="Neo4j database health")
    redis: ServiceHealth = Field(description="Redis cache health") 


class ExternalAPIHealth(BaseModel):
    """External API health status."""
    
    openalex: Optional[ServiceHealth] = Field(default=None, description="OpenAlex API health")
    semantic_scholar: Optional[ServiceHealth] = Field(default=None, description="Semantic Scholar API health")
    crossref: Optional[ServiceHealth] = Field(default=None, description="CrossRef API health")
    zotero: Optional[ServiceHealth] = Field(default=None, description="Zotero API health")


class SystemMetrics(BaseModel):
    """System performance metrics."""
    
    cpu_usage_percent: Optional[float] = Field(default=None, description="CPU usage percentage")
    memory_usage_percent: Optional[float] = Field(default=None, description="Memory usage percentage")
    disk_usage_percent: Optional[float] = Field(default=None, description="Disk usage percentage")
    active_connections: Optional[int] = Field(default=None, description="Active database connections")
    cache_hit_rate: Optional[float] = Field(default=None, description="Cache hit rate percentage")
    requests_per_minute: Optional[float] = Field(default=None, description="Requests per minute")
    average_response_time_ms: Optional[float] = Field(default=None, description="Average response time")


class HealthCheck(BaseModel):
    """Complete health check response."""
    
    status: HealthStatus = Field(description="Overall system health status")
    timestamp: datetime = Field(description="Health check timestamp")
    version: str = Field(description="Application version")
    uptime_seconds: int = Field(description="Application uptime in seconds")
    
    # Service health
    database: DatabaseHealth = Field(description="Database health status")
    external_apis: ExternalAPIHealth = Field(description="External API health status")
    
    # System metrics
    metrics: Optional[SystemMetrics] = Field(default=None, description="System performance metrics")
    
    # Additional details
    checks_passed: int = Field(description="Number of health checks passed")
    checks_total: int = Field(description="Total number of health checks")
    warnings: List[str] = Field(default_factory=list, description="Health warnings")
    errors: List[str] = Field(default_factory=list, description="Health errors")
    
    @property
    def is_healthy(self) -> bool:
        """Check if system is healthy."""
        return self.status == HealthStatus.HEALTHY
    
    @property
    def success_rate(self) -> float:
        """Calculate health check success rate."""
        if self.checks_total == 0:
            return 0.0
        return (self.checks_passed / self.checks_total) * 100


class ReadinessCheck(BaseModel):
    """Application readiness check."""
    
    ready: bool = Field(description="Application is ready to serve requests")
    timestamp: datetime = Field(description="Readiness check timestamp")
    services_ready: Dict[str, bool] = Field(description="Individual service readiness")
    startup_time_seconds: Optional[float] = Field(default=None, description="Application startup time")


class LivenessCheck(BaseModel):
    """Application liveness check."""
    
    alive: bool = Field(description="Application is alive and running")
    timestamp: datetime = Field(description="Liveness check timestamp")
    pid: Optional[int] = Field(default=None, description="Process ID")
    
    
class MetricsSnapshot(BaseModel):
    """Performance metrics snapshot."""
    
    timestamp: datetime = Field(description="Metrics snapshot timestamp")
    
    # Request metrics
    total_requests: int = Field(default=0, description="Total requests processed")
    requests_per_second: float = Field(default=0.0, description="Current requests per second")
    average_response_time_ms: float = Field(default=0.0, description="Average response time")
    p95_response_time_ms: float = Field(default=0.0, description="95th percentile response time")
    p99_response_time_ms: float = Field(default=0.0, description="99th percentile response time")
    
    # Error metrics
    total_errors: int = Field(default=0, description="Total errors encountered")
    error_rate_percent: float = Field(default=0.0, description="Current error rate percentage")
    
    # Database metrics
    database_connections_active: int = Field(default=0, description="Active database connections")
    database_connections_idle: int = Field(default=0, description="Idle database connections")
    cache_hits: int = Field(default=0, description="Cache hits")
    cache_misses: int = Field(default=0, description="Cache misses")
    
    # System metrics
    memory_usage_mb: Optional[float] = Field(default=None, description="Memory usage in MB")
    cpu_usage_percent: Optional[float] = Field(default=None, description="CPU usage percentage")
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate percentage."""
        total_requests = self.cache_hits + self.cache_misses
        if total_requests == 0:
            return 0.0
        return (self.cache_hits / total_requests) * 100


class AlertRule(BaseModel):
    """Performance alert rule."""
    
    name: str = Field(description="Alert rule name")
    metric: str = Field(description="Metric to monitor")
    operator: str = Field(description="Comparison operator", pattern="^(gt|gte|lt|lte|eq|neq)$")
    threshold: float = Field(description="Alert threshold value")
    duration_minutes: int = Field(description="Alert duration in minutes")
    severity: str = Field(description="Alert severity", pattern="^(info|warning|error|critical)$")
    enabled: bool = Field(default=True, description="Alert rule is enabled")


class Alert(BaseModel):
    """Performance alert."""
    
    id: str = Field(description="Alert ID")
    rule_name: str = Field(description="Alert rule name")
    metric: str = Field(description="Metric that triggered alert")
    current_value: float = Field(description="Current metric value")
    threshold: float = Field(description="Alert threshold")
    severity: str = Field(description="Alert severity")
    message: str = Field(description="Alert message")
    started_at: datetime = Field(description="Alert start time")
    resolved_at: Optional[datetime] = Field(default=None, description="Alert resolution time")
    is_active: bool = Field(default=True, description="Alert is currently active")
    
    @property
    def duration_minutes(self) -> float:
        """Calculate alert duration in minutes."""
        end_time = self.resolved_at or datetime.utcnow()
        duration = end_time - self.started_at
        return duration.total_seconds() / 60