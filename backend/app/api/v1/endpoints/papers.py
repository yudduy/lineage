"""
Paper management endpoints - Minimal demo version.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException
from ....db.neo4j import get_neo4j_manager
from ....utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/{paper_id}")
async def get_paper_by_id(paper_id: str):
    """
    Get a paper by its ID from Neo4j.
    """
    try:
        neo4j_manager = await get_neo4j_manager()
        paper = await neo4j_manager.get_paper_by_id(paper_id)
        
        if not paper:
            raise HTTPException(status_code=404, detail="Paper not found")
        
        return paper
    except Exception as e:
        logger.error(f"Error getting paper {paper_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/doi/{doi:path}")
async def get_paper_by_doi(doi: str):
    """
    Get a paper by its DOI from Neo4j.
    """
    try:
        neo4j_manager = await get_neo4j_manager()
        paper = await neo4j_manager.get_paper_by_doi(doi)
        
        if not paper:
            raise HTTPException(status_code=404, detail="Paper not found")
        
        return paper
    except Exception as e:
        logger.error(f"Error getting paper with DOI {doi}: {e}")
        raise HTTPException(status_code=500, detail=str(e))