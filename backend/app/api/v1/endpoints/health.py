"""
Health check endpoints - Minimal demo version.
"""

from fastapi import APIRouter, Depends
from ....services.health import HealthService, get_health_service

router = APIRouter()


@router.get("/")
async def get_health_status(
    health_service: HealthService = Depends(get_health_service)
):
    """
    Get basic health status of the application.
    
    Checks database connections and basic service availability.
    """
    health_status = await health_service.get_basic_health()
    return health_status


@router.get("/ready")
async def readiness_check(
    health_service: HealthService = Depends(get_health_service)
):
    """
    Simple readiness check for minimal demo.
    """
    return await health_service.get_basic_health()


@router.get("/live")
async def liveness_check():
    """
    Simple liveness check - always returns OK.
    """
    return {"status": "alive", "service": "citation-network-explorer-demo"}