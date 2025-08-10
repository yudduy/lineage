"""
Background task management endpoints.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, Query, HTTPException, status

from ....models.user import User
from ....models.common import APIResponse
from ....middleware.auth import get_current_active_user, require_scope
from ....services.tasks import CeleryTaskService, get_task_service
from ....services.openalex_tasks import (
    get_task_status as get_openalex_task_status,
    cancel_task as cancel_openalex_task,
    get_active_tasks
)
from ....utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/papers/{paper_id}/fetch-metadata", response_model=APIResponse[Dict[str, str]])
async def fetch_paper_metadata(
    paper_id: str,
    sources: Optional[List[str]] = Query(None, description="Data sources to query"),
    current_user: User = Depends(get_current_active_user),
    task_service: CeleryTaskService = Depends(get_task_service)
):
    """
    Start background task to fetch paper metadata from external sources.
    
    This will query various academic databases to gather complete
    metadata for the specified paper.
    """
    logger.info(
        "Starting metadata fetch task",
        paper_id=paper_id,
        sources=sources,
        user_id=current_user.id
    )
    
    task_id = await task_service.submit_task(
        "citation_network_explorer.tasks.paper.fetch_paper_metadata",
        args=(paper_id,),
        kwargs={"sources": sources or ["openalex", "semantic_scholar", "crossref"]},
        queue="papers"
    )
    
    return APIResponse(
        success=True,
        message="Metadata fetch task started",
        data={"task_id": task_id, "paper_id": paper_id}
    )


@router.post("/papers/{paper_id}/build-network", response_model=APIResponse[Dict[str, str]])
async def build_citation_network(
    paper_id: str,
    max_depth: int = Query(2, ge=1, le=5, description="Maximum network depth"),
    current_user: User = Depends(get_current_active_user),
    task_service: CeleryTaskService = Depends(get_task_service)
):
    """
    Start background task to build citation network for a paper.
    
    This will traverse citations and references to build a complete
    citation network around the specified paper.
    """
    logger.info(
        "Starting citation network build task",
        paper_id=paper_id,
        max_depth=max_depth,
        user_id=current_user.id
    )
    
    task_id = await task_service.submit_task(
        "citation_network_explorer.tasks.paper.build_citation_network",
        args=(paper_id,),
        kwargs={"max_depth": max_depth},
        queue="papers"
    )
    
    return APIResponse(
        success=True,
        message="Citation network build task started",
        data={"task_id": task_id, "paper_id": paper_id}
    )


@router.post("/zotero/sync/{collection_key}", response_model=APIResponse[Dict[str, str]])
async def sync_zotero_collection(
    collection_key: str,
    current_user: User = Depends(get_current_active_user),
    task_service: CeleryTaskService = Depends(get_task_service)
):
    """
    Start background task to sync a Zotero collection.
    
    This will fetch all items from the specified Zotero collection
    and update the local database.
    """
    logger.info(
        "Starting Zotero sync task",
        collection_key=collection_key,
        user_id=current_user.id
    )
    
    task_id = await task_service.submit_task(
        "citation_network_explorer.tasks.zotero.sync_collection",
        args=(current_user.id, collection_key),
        queue="zotero"
    )
    
    return APIResponse(
        success=True,
        message="Zotero sync task started",
        data={"task_id": task_id, "collection_key": collection_key}
    )


@router.post("/export/bibtex", response_model=APIResponse[Dict[str, str]])
async def export_bibtex(
    paper_ids: List[str],
    current_user: User = Depends(get_current_active_user),
    task_service: CeleryTaskService = Depends(get_task_service)
):
    """
    Start background task to generate BibTeX export.
    
    This will create a BibTeX file containing all specified papers
    and provide a download URL when complete.
    """
    if not paper_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one paper ID is required"
        )
    
    if len(paper_ids) > 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 1000 papers allowed per export"
        )
    
    logger.info(
        "Starting BibTeX export task",
        paper_count=len(paper_ids),
        user_id=current_user.id
    )
    
    task_id = await task_service.submit_task(
        "citation_network_explorer.tasks.export.generate_bibtex",
        args=(current_user.id, paper_ids),
        queue="export"
    )
    
    return APIResponse(
        success=True,
        message="BibTeX export task started",
        data={"task_id": task_id, "paper_count": len(paper_ids)}
    )


@router.post("/search/reindex", response_model=APIResponse[Dict[str, str]])
async def reindex_search(
    paper_ids: Optional[List[str]] = None,
    current_user: User = Depends(require_scope("admin")),
    task_service: CeleryTaskService = Depends(get_task_service)
):
    """
    Start background task to update search index.
    
    This will reindex papers in the search system. If no paper IDs
    are provided, will reindex all papers.
    
    Requires admin permissions.
    """
    logger.info(
        "Starting search reindex task",
        paper_count=len(paper_ids) if paper_ids else "all",
        user_id=current_user.id
    )
    
    task_id = await task_service.submit_task(
        "citation_network_explorer.tasks.search.update_search_index",
        kwargs={"paper_ids": paper_ids},
        queue="search",
        priority=3  # Lower priority for admin tasks
    )
    
    return APIResponse(
        success=True,
        message="Search reindex task started",
        data={"task_id": task_id, "scope": "partial" if paper_ids else "full"}
    )


@router.get("/{task_id}/status", response_model=APIResponse[Dict[str, Any]])
async def get_task_status(
    task_id: str,
    current_user: User = Depends(get_current_active_user),
    task_service: CeleryTaskService = Depends(get_task_service)
):
    """
    Get the status of a background task.
    
    Returns current task status, progress information,
    and results if the task is complete. Supports both
    legacy task service and OpenAlex task tracking.
    """
    logger.info(
        "Getting task status",
        task_id=task_id,
        user_id=current_user.id
    )
    
    # Try OpenAlex task status first (for new tasks)
    try:
        openalex_status = await get_openalex_task_status(task_id)
        if openalex_status:
            return APIResponse(
                success=True,
                message="Task status retrieved successfully",
                data=openalex_status
            )
    except Exception as e:
        logger.debug(f"OpenAlex task status lookup failed: {e}")
    
    # Fall back to legacy task service
    status_info = await task_service.get_task_status(task_id)
    
    return APIResponse(
        success=True,
        message="Task status retrieved successfully",
        data=status_info
    )


@router.delete("/{task_id}", response_model=APIResponse[Dict[str, str]])
async def cancel_task(
    task_id: str,
    terminate: bool = Query(False, description="Force terminate the task"),
    current_user: User = Depends(get_current_active_user),
    task_service: CeleryTaskService = Depends(get_task_service)
):
    """
    Cancel a background task.
    
    This will attempt to cancel the specified task. If terminate is True,
    will forcefully kill the task process. Supports both legacy and
    OpenAlex task cancellation.
    """
    logger.info(
        "Cancelling task",
        task_id=task_id,
        terminate=terminate,
        user_id=current_user.id
    )
    
    # TODO: Check if user owns the task or has admin permissions
    
    # Try OpenAlex task cancellation first
    try:
        openalex_success = await cancel_openalex_task(task_id)
        if openalex_success:
            return APIResponse(
                success=True,
                message="Task cancelled successfully (OpenAlex)",
                data={"task_id": task_id, "cancelled": True}
            )
    except Exception as e:
        logger.debug(f"OpenAlex task cancellation failed: {e}")
    
    # Fall back to legacy task service
    success = await task_service.cancel_task(task_id, terminate=terminate)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel task"
        )
    
    return APIResponse(
        success=True,
        message="Task cancelled successfully",
        data={"task_id": task_id, "cancelled": True}
    )


@router.get("/queues/{queue_name}", response_model=APIResponse[Dict[str, Any]])
async def get_queue_info(
    queue_name: str,
    current_user: User = Depends(require_scope("admin")),
    task_service: CeleryTaskService = Depends(get_task_service)
):
    """
    Get information about a task queue.
    
    Returns queue statistics including task counts and worker information.
    Requires admin permissions.
    """
    logger.info(
        "Getting queue info",
        queue_name=queue_name,
        user_id=current_user.id
    )
    
    queue_info = await task_service.get_queue_info(queue_name)
    
    return APIResponse(
        success=True,
        message=f"Queue information retrieved for {queue_name}",
        data=queue_info
    )


@router.get("/workers", response_model=APIResponse[Dict[str, Any]])
async def get_worker_stats(
    current_user: User = Depends(require_scope("admin")),
    task_service: CeleryTaskService = Depends(get_task_service)
):
    """
    Get statistics about Celery workers.
    
    Returns worker health, performance metrics, and task processing information.
    Requires admin permissions.
    """
    logger.info(
        "Getting worker stats",
        user_id=current_user.id
    )
    
    worker_stats = await task_service.get_worker_stats()
    
    return APIResponse(
        success=True,
        message="Worker statistics retrieved successfully",
        data=worker_stats
    )


@router.get("/openalex/active", response_model=APIResponse[List[Dict[str, Any]]])
async def get_active_openalex_tasks(
    current_user: User = Depends(get_current_active_user)
):
    """
    Get list of active OpenAlex tasks.
    
    Returns information about currently running OpenAlex import,
    citation network building, and sync tasks.
    """
    logger.info(
        "Getting active OpenAlex tasks",
        user_id=current_user.id
    )
    
    try:
        active_tasks = get_active_tasks()
        
        # Filter to only OpenAlex-related tasks
        openalex_tasks = [
            task for task in active_tasks 
            if any(keyword in task.get("name", "") 
                  for keyword in ["openalex", "citation", "import", "sync"])
        ]
        
        return APIResponse(
            success=True,
            message=f"Found {len(openalex_tasks)} active OpenAlex tasks",
            data=openalex_tasks
        )
        
    except Exception as e:
        logger.error(f"Error getting active OpenAlex tasks: {e}")
        return APIResponse(
            success=True,
            message="Could not retrieve active tasks",
            data=[]
        )


@router.post("/openalex/batch-import", response_model=APIResponse[Dict[str, Any]])
async def batch_import_papers_by_dois_endpoint(
    dois: List[str] = Query(..., description="List of DOIs to import"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Start background task to import multiple papers by DOI from OpenAlex.
    
    This endpoint takes a list of DOIs and starts a background task
    to import the corresponding papers from OpenAlex.
    """
    if not dois:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one DOI is required"
        )
    
    if len(dois) > 500:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 500 DOIs allowed per batch import"
        )
    
    logger.info(
        "Starting OpenAlex batch DOI import",
        doi_count=len(dois),
        user_id=current_user.id
    )
    
    # Import the batch import task
    from ....services.openalex_tasks import batch_import_papers_by_dois
    
    # Start background task
    task = batch_import_papers_by_dois.delay(
        dois=dois,
        user_id=current_user.id
    )
    
    return APIResponse(
        success=True,
        message=f"Started batch import task for {len(dois)} DOIs",
        data={
            "task_id": task.id,
            "doi_count": len(dois),
            "status": "importing",
            "message": "Batch import started. Use GET /api/v1/tasks/{task_id}/status to check progress."
        }
    )