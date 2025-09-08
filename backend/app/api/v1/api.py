"""
Main API router with core endpoints for citation network exploration.
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

# Core endpoints for the citation network explorer
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