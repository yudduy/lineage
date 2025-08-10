"""
Health check service for monitoring system status.
"""

import asyncio
import psutil
from datetime import datetime, timedelta
from typing import Dict, Any, List

from ..core.config import get_settings
from ..db.neo4j import Neo4jManager
from ..db.redis import RedisManager
from ..models.health import (
    HealthCheck,
    HealthStatus,
    ServiceHealth,
    DatabaseHealth,
    ExternalAPIHealth,
    SystemMetrics,
    ReadinessCheck,
    LivenessCheck
)
from ..utils.logger import get_logger

logger = get_logger(__name__)


class HealthService:
    """Service for performing health checks and monitoring."""
    
    def __init__(self):
        self.settings = get_settings()
        self.startup_time = datetime.utcnow()
        self.logger = logger
    
    async def get_basic_health(self) -> Dict[str, Any]:
        """Get basic health status for minimal demo."""
        try:
            from ..db.neo4j import get_neo4j_manager
            from ..db.redis import get_redis_manager
            
            # Check Neo4j
            neo4j_status = "unknown"
            try:
                neo4j_manager = await get_neo4j_manager()
                health = await neo4j_manager.health_check()
                neo4j_status = health.get("status", "unknown")
            except Exception as e:
                neo4j_status = f"error: {str(e)}"
            
            # Check Redis (optional)
            redis_status = "not_configured"
            if self.settings.redis_url:
                try:
                    redis_manager = await get_redis_manager()
                    if redis_manager.client:
                        health = await redis_manager.health_check()
                        redis_status = health.get("status", "unknown")
                    else:
                        redis_status = "not_connected"
                except Exception as e:
                    redis_status = f"error: {str(e)}"
            
            # Calculate uptime
            uptime_seconds = int((datetime.utcnow() - self.startup_time).total_seconds())
            
            return {
                "status": "healthy" if neo4j_status == "healthy" else "degraded",
                "timestamp": datetime.utcnow().isoformat(),
                "uptime_seconds": uptime_seconds,
                "services": {
                    "neo4j": neo4j_status,
                    "redis": redis_status
                },
                "version": self.settings.app_version
            }
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        
    async def get_health_status(self) -> HealthCheck:
        """Get comprehensive health status."""
        try:
            # Get individual service health
            database_health = await self._check_database_health()
            external_apis_health = await self._check_external_apis_health()
            system_metrics = await self._get_system_metrics()
            
            # Determine overall status
            all_checks = []
            
            # Add database health checks
            all_checks.append(database_health.neo4j.status)
            all_checks.append(database_health.redis.status)
            
            # Add external API checks (if configured)
            if external_apis_health.openalex:
                all_checks.append(external_apis_health.openalex.status)
            if external_apis_health.semantic_scholar:
                all_checks.append(external_apis_health.semantic_scholar.status)
            if external_apis_health.crossref:
                all_checks.append(external_apis_health.crossref.status)
            if external_apis_health.zotero:
                all_checks.append(external_apis_health.zotero.status)
            
            # Calculate overall status
            healthy_count = sum(1 for status in all_checks if status == HealthStatus.HEALTHY)
            degraded_count = sum(1 for status in all_checks if status == HealthStatus.DEGRADED)
            unhealthy_count = sum(1 for status in all_checks if status == HealthStatus.UNHEALTHY)
            
            if unhealthy_count > 0:
                overall_status = HealthStatus.UNHEALTHY
            elif degraded_count > 0:
                overall_status = HealthStatus.DEGRADED
            else:
                overall_status = HealthStatus.HEALTHY
            
            # Collect warnings and errors
            warnings = []
            errors = []
            
            for check in all_checks:
                if check == HealthStatus.DEGRADED:
                    warnings.append(f"Service is degraded")
                elif check == HealthStatus.UNHEALTHY:
                    errors.append(f"Service is unhealthy")
            
            # Calculate uptime
            uptime_seconds = int((datetime.utcnow() - self.startup_time).total_seconds())
            
            return HealthCheck(
                status=overall_status,
                timestamp=datetime.utcnow(),
                version=self.settings.app_version,
                uptime_seconds=uptime_seconds,
                database=database_health,
                external_apis=external_apis_health,
                metrics=system_metrics,
                checks_passed=healthy_count,
                checks_total=len(all_checks),
                warnings=warnings,
                errors=errors
            )
            
        except Exception as e:
            self.logger.error(f"Error getting health status: {e}", exc_info=True)
            
            return HealthCheck(
                status=HealthStatus.UNHEALTHY,
                timestamp=datetime.utcnow(),
                version=self.settings.app_version,
                uptime_seconds=0,
                database=DatabaseHealth(
                    neo4j=ServiceHealth(
                        name="neo4j",
                        status=HealthStatus.UNHEALTHY,
                        last_check=datetime.utcnow(),
                        error="Health check failed"
                    ),
                    redis=ServiceHealth(
                        name="redis",
                        status=HealthStatus.UNHEALTHY,
                        last_check=datetime.utcnow(),
                        error="Health check failed"
                    )
                ),
                external_apis=ExternalAPIHealth(),
                checks_passed=0,
                checks_total=1,
                errors=[str(e)]
            )
    
    async def _check_database_health(self) -> DatabaseHealth:
        """Check health of database connections."""
        
        # Check Neo4j
        neo4j_manager = Neo4jManager()
        try:
            neo4j_health_data = await neo4j_manager.health_check()
            
            if neo4j_health_data["status"] == "healthy":
                neo4j_health = ServiceHealth(
                    name="neo4j",
                    status=HealthStatus.HEALTHY,
                    response_time_ms=neo4j_health_data.get("response_time_ms"),
                    last_check=datetime.utcnow(),
                    details=neo4j_health_data
                )
            else:
                neo4j_health = ServiceHealth(
                    name="neo4j",
                    status=HealthStatus.UNHEALTHY,
                    last_check=datetime.utcnow(),
                    error=neo4j_health_data.get("error", "Unknown error")
                )
                
        except Exception as e:
            neo4j_health = ServiceHealth(
                name="neo4j",
                status=HealthStatus.UNHEALTHY,
                last_check=datetime.utcnow(),
                error=str(e)
            )
        
        # Check Redis
        redis_manager = RedisManager()
        try:
            redis_health_data = await redis_manager.health_check()
            
            if redis_health_data["status"] == "healthy":
                redis_health = ServiceHealth(
                    name="redis",
                    status=HealthStatus.HEALTHY,
                    response_time_ms=redis_health_data.get("response_time_ms"),
                    last_check=datetime.utcnow(),
                    details=redis_health_data
                )
            else:
                redis_health = ServiceHealth(
                    name="redis",
                    status=HealthStatus.UNHEALTHY,
                    last_check=datetime.utcnow(),
                    error=redis_health_data.get("error", "Unknown error")
                )
                
        except Exception as e:
            redis_health = ServiceHealth(
                name="redis",
                status=HealthStatus.UNHEALTHY,
                last_check=datetime.utcnow(),
                error=str(e)
            )
        
        return DatabaseHealth(neo4j=neo4j_health, redis=redis_health)
    
    async def _check_external_apis_health(self) -> ExternalAPIHealth:
        """Check health of external APIs."""
        
        external_health = ExternalAPIHealth()
        
        # Check OpenAlex (if configured)
        if self.settings.external_apis.openalex_email:
            external_health.openalex = await self._check_openalex_health()
        
        # Check Semantic Scholar (if configured)
        if self.settings.external_apis.semantic_scholar_api_key:
            external_health.semantic_scholar = await self._check_semantic_scholar_health()
        
        # Check CrossRef (if configured)
        if self.settings.external_apis.crossref_email:
            external_health.crossref = await self._check_crossref_health()
        
        # Check Zotero (if configured)
        if self.settings.external_apis.zotero_app_key:
            external_health.zotero = await self._check_zotero_health()
        
        return external_health
    
    async def _check_openalex_health(self) -> ServiceHealth:
        """Check OpenAlex API health."""
        try:
            import httpx
            start_time = datetime.utcnow()
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.openalex.org/works",
                    params={"per-page": 1},
                    timeout=5.0
                )
            
            end_time = datetime.utcnow()
            response_time_ms = (end_time - start_time).total_seconds() * 1000
            
            if response.status_code == 200:
                return ServiceHealth(
                    name="openalex",
                    status=HealthStatus.HEALTHY,
                    response_time_ms=response_time_ms,
                    last_check=datetime.utcnow(),
                    details={"status_code": response.status_code}
                )
            else:
                return ServiceHealth(
                    name="openalex",
                    status=HealthStatus.DEGRADED,
                    response_time_ms=response_time_ms,
                    last_check=datetime.utcnow(),
                    error=f"HTTP {response.status_code}"
                )
                
        except Exception as e:
            return ServiceHealth(
                name="openalex",
                status=HealthStatus.UNHEALTHY,
                last_check=datetime.utcnow(),
                error=str(e)
            )
    
    async def _check_semantic_scholar_health(self) -> ServiceHealth:
        """Check Semantic Scholar API health."""
        try:
            import httpx
            start_time = datetime.utcnow()
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.semanticscholar.org/graph/v1/paper/search",
                    params={"query": "test", "limit": 1},
                    timeout=5.0
                )
            
            end_time = datetime.utcnow()
            response_time_ms = (end_time - start_time).total_seconds() * 1000
            
            if response.status_code == 200:
                return ServiceHealth(
                    name="semantic_scholar",
                    status=HealthStatus.HEALTHY,
                    response_time_ms=response_time_ms,
                    last_check=datetime.utcnow(),
                    details={"status_code": response.status_code}
                )
            else:
                return ServiceHealth(
                    name="semantic_scholar",
                    status=HealthStatus.DEGRADED,
                    response_time_ms=response_time_ms,
                    last_check=datetime.utcnow(),
                    error=f"HTTP {response.status_code}"
                )
                
        except Exception as e:
            return ServiceHealth(
                name="semantic_scholar",
                status=HealthStatus.UNHEALTHY,
                last_check=datetime.utcnow(),
                error=str(e)
            )
    
    async def _check_crossref_health(self) -> ServiceHealth:
        """Check CrossRef API health."""
        try:
            import httpx
            start_time = datetime.utcnow()
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.crossref.org/works",
                    params={"rows": 1},
                    timeout=5.0
                )
            
            end_time = datetime.utcnow()
            response_time_ms = (end_time - start_time).total_seconds() * 1000
            
            if response.status_code == 200:
                return ServiceHealth(
                    name="crossref",
                    status=HealthStatus.HEALTHY,
                    response_time_ms=response_time_ms,
                    last_check=datetime.utcnow(),
                    details={"status_code": response.status_code}
                )
            else:
                return ServiceHealth(
                    name="crossref",
                    status=HealthStatus.DEGRADED,
                    response_time_ms=response_time_ms,
                    last_check=datetime.utcnow(),
                    error=f"HTTP {response.status_code}"
                )
                
        except Exception as e:
            return ServiceHealth(
                name="crossref",
                status=HealthStatus.UNHEALTHY,
                last_check=datetime.utcnow(),
                error=str(e)
            )
    
    async def _check_zotero_health(self) -> ServiceHealth:
        """Check Zotero API health."""
        try:
            import httpx
            start_time = datetime.utcnow()
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.zotero.org/",
                    timeout=5.0
                )
            
            end_time = datetime.utcnow()
            response_time_ms = (end_time - start_time).total_seconds() * 1000
            
            if response.status_code == 200:
                return ServiceHealth(
                    name="zotero",
                    status=HealthStatus.HEALTHY,
                    response_time_ms=response_time_ms,
                    last_check=datetime.utcnow(),
                    details={"status_code": response.status_code}
                )
            else:
                return ServiceHealth(
                    name="zotero",
                    status=HealthStatus.DEGRADED,
                    response_time_ms=response_time_ms,
                    last_check=datetime.utcnow(),
                    error=f"HTTP {response.status_code}"
                )
                
        except Exception as e:
            return ServiceHealth(
                name="zotero",
                status=HealthStatus.UNHEALTHY,
                last_check=datetime.utcnow(),
                error=str(e)
            )
    
    async def _get_system_metrics(self) -> SystemMetrics:
        """Get system performance metrics."""
        try:
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Get memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Get disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Try to get database connection counts (simplified)
            active_connections = 1  # Placeholder
            
            return SystemMetrics(
                cpu_usage_percent=cpu_percent,
                memory_usage_percent=memory_percent,
                disk_usage_percent=disk_percent,
                active_connections=active_connections
            )
            
        except Exception as e:
            self.logger.warning(f"Error getting system metrics: {e}")
            return SystemMetrics()
    
    async def get_readiness_check(self) -> ReadinessCheck:
        """Check if application is ready to serve requests."""
        try:
            # Check essential services
            services_ready = {}
            
            # Check Neo4j
            neo4j_manager = Neo4jManager()
            neo4j_health = await neo4j_manager.health_check()
            services_ready["neo4j"] = neo4j_health["status"] == "healthy"
            
            # Check Redis
            redis_manager = RedisManager()
            redis_health = await redis_manager.health_check()
            services_ready["redis"] = redis_health["status"] == "healthy"
            
            # Application is ready if essential services are available
            ready = services_ready.get("neo4j", False) and services_ready.get("redis", False)
            
            startup_time_seconds = (datetime.utcnow() - self.startup_time).total_seconds()
            
            return ReadinessCheck(
                ready=ready,
                timestamp=datetime.utcnow(),
                services_ready=services_ready,
                startup_time_seconds=startup_time_seconds
            )
            
        except Exception as e:
            self.logger.error(f"Error in readiness check: {e}", exc_info=True)
            
            return ReadinessCheck(
                ready=False,
                timestamp=datetime.utcnow(),
                services_ready={}
            )
    
    async def get_liveness_check(self) -> LivenessCheck:
        """Check if application is alive and running."""
        try:
            import os
            
            return LivenessCheck(
                alive=True,
                timestamp=datetime.utcnow(),
                pid=os.getpid()
            )
            
        except Exception as e:
            self.logger.error(f"Error in liveness check: {e}", exc_info=True)
            
            return LivenessCheck(
                alive=False,
                timestamp=datetime.utcnow()
            )


# Global health service instance
health_service = HealthService()


def get_health_service() -> HealthService:
    """Dependency function to get health service."""
    return health_service