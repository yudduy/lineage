"""
OpenAlex-specific API endpoints.

Provides endpoints for OpenAlex data retrieval and network building.
Minimal demo version without authentication.
"""

from typing import Any, Dict, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ....services.openalex import get_openalex_client
from ....db.redis import get_redis_manager
from ....utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


class BuildNetworkRequest(BaseModel):
    """Request model for building citation network."""
    identifier: str = Field(..., description="DOI, OpenAlex URL/ID, or paper title")
    direction: str = Field(default="both", description="Direction: backward, forward, or both")
    max_depth: int = Field(default=2, ge=1, le=3, description="Maximum depth (1-3)")
    max_per_level: int = Field(default=20, ge=1, le=50, description="Maximum papers per level (1-50)")


class NetworkResponse(BaseModel):
    """Response model for network data."""
    center_paper_id: str
    total_nodes: int
    total_edges: int
    max_depth_reached: int
    nodes: list
    edges: list


@router.post("/network/build-sync", response_model=NetworkResponse)
async def build_network_sync(request: BuildNetworkRequest):
    """
    Build citation network synchronously and persist to Neo4j.
    
    This endpoint:
    1. Accepts a DOI, OpenAlex URL/ID, or paper title
    2. Builds the citation network via OpenAlex API
    3. Persists the network to Neo4j
    4. Returns the network data for visualization
    
    Args:
        request: Network building parameters
        
    Returns:
        NetworkResponse with nodes and edges for visualization
    """
    try:
        logger.info(f"Building network for: {request.identifier}")
        
        # Get OpenAlex client (with optional Redis caching)
        redis_manager = None
        try:
            redis_manager = await get_redis_manager()
        except Exception as e:
            logger.warning(f"Redis not available, continuing without cache: {e}")
        
        client = await get_openalex_client(redis_manager)
        
        # Build the network
        network_data = await client.build_citation_network_sync(
            center_identifier=request.identifier,
            direction=request.direction,
            max_depth=request.max_depth,
            max_per_level=request.max_per_level
        )
        
        logger.info(f"Network built: {network_data['total_nodes']} nodes, {network_data['total_edges']} edges")
        
        return NetworkResponse(**network_data)
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error building network: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to build network: {str(e)}")


@router.get("/health")
async def health_check():
    """
    Simple health check for OpenAlex integration.
    
    Returns:
        Health status and basic metrics
    """
    try:
        redis_manager = None
        try:
            redis_manager = await get_redis_manager()
        except:
            pass
        
        client = await get_openalex_client(redis_manager)
        health_status = await client.health_check()
        
        return {
            "status": health_status.get("status", "unknown"),
            "openalex_available": health_status.get("status") == "healthy",
            "response_time_ms": health_status.get("response_time_ms"),
            "rate_limit": health_status.get("rate_limit", {}),
            "cache_available": redis_manager is not None
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "openalex_available": False,
            "cache_available": False
        }