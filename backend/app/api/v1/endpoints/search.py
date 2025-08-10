"""
Search endpoints - Minimal demo version.
"""

from typing import Optional
from fastapi import APIRouter, Query
from ....db.neo4j import get_neo4j_manager
from ....utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/papers")
async def search_papers(
    q: str = Query(..., description="Search query"),
    limit: int = Query(20, ge=1, le=100, description="Number of results to return"),
    skip: int = Query(0, ge=0, description="Number of results to skip")
):
    """
    Search for papers in the Neo4j database.
    """
    try:
        neo4j_manager = await get_neo4j_manager()
        results = await neo4j_manager.search_papers(q, limit, skip)
        
        return {
            "query": q,
            "total": len(results),
            "papers": results
        }
    except Exception as e:
        logger.error(f"Search error: {e}")
        return {
            "query": q,
            "total": 0,
            "papers": [],
            "error": str(e)
        }