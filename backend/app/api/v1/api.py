"""
Main API v1 router - Minimal demo version with essential endpoints only.
"""

from fastapi import APIRouter

from .endpoints import (
    health,
    papers,
    search,
    openalex
)

# Create main API router
api_router = APIRouter(prefix="/api/v1")

# Include only essential endpoint routers for minimal demo
api_router.include_router(
    health.router,
    prefix="/health",
    tags=["health"]
)

api_router.include_router(
    papers.router,
    prefix="/papers",
    tags=["papers"]
)

api_router.include_router(
    search.router,
    prefix="/search",
    tags=["search"]
)

api_router.include_router(
    openalex.router,
    prefix="/openalex",
    tags=["openalex"]
)