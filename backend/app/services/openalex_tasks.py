"""
Background tasks for OpenAlex data operations.

Celery tasks for handling large-scale OpenAlex operations like batch imports,
citation network building, and data synchronization.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from celery import Celery, Task

from ..core.config import get_settings
from ..db.redis import get_redis_manager
from ..db.dependencies import get_db_service
from ..services.openalex import get_openalex_client
from ..services.openalex_converter import OpenAlexConverter
from ..models.openalex import OpenAlexWork, OpenAlexSearchFilters
from ..models.paper import Paper, PaperEdge
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Initialize Celery app
settings = get_settings()
celery_app = Celery(
    "openalex_tasks",
    broker=settings.database.redis_url,
    backend=settings.database.redis_url
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    result_expires=3600,  # Results expire after 1 hour
    task_routes={
        "app.services.openalex_tasks.*": {"queue": "openalex"}
    }
)


class OpenAlexTask(Task):
    """Base class for OpenAlex tasks with shared functionality."""
    
    def __init__(self):
        self._redis_manager = None
        self._db_service = None
        self._openalex_client = None
    
    async def get_redis_manager(self):
        """Get Redis manager instance."""
        if self._redis_manager is None:
            self._redis_manager = await get_redis_manager()
        return self._redis_manager
    
    async def get_db_service(self):
        """Get database service instance."""
        if self._db_service is None:
            from ..db.dependencies import get_db_service
            self._db_service = await get_db_service()
        return self._db_service
    
    async def get_openalex_client(self):
        """Get OpenAlex client instance."""
        if self._openalex_client is None:
            redis_manager = await self.get_redis_manager()
            self._openalex_client = await get_openalex_client(redis_manager)
        return self._openalex_client
    
    async def update_task_progress(self, task_id: str, progress: Dict[str, Any]):
        """Update task progress in Redis."""
        redis_manager = await self.get_redis_manager()
        progress_key = f"task_progress:{task_id}"
        
        progress_data = {
            **progress,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        await redis_manager.cache_set(progress_key, progress_data, expire=3600)
    
    async def get_task_progress(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task progress from Redis."""
        redis_manager = await self.get_redis_manager()
        progress_key = f"task_progress:{task_id}"
        return await redis_manager.cache_get(progress_key)


def run_async_task(coro):
    """Helper to run async coroutines in Celery tasks."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)


@celery_app.task(base=OpenAlexTask, bind=True)
def batch_import_papers_by_dois(self, dois: List[str], user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Import multiple papers by DOI in batches.
    
    Args:
        dois: List of DOIs to import
        user_id: User ID for tracking
        
    Returns:
        Dictionary with import results
    """
    async def _batch_import():
        task_id = self.request.id
        client = await self.get_openalex_client()
        db_service = await self.get_db_service()
        
        total_dois = len(dois)
        batch_size = 50
        imported_papers = []
        failed_dois = []
        skipped_dois = []
        
        logger.info(f"Starting batch import of {total_dois} papers for user {user_id}")
        
        # Update initial progress
        await self.update_task_progress(task_id, {
            "status": "running",
            "total_items": total_dois,
            "processed_items": 0,
            "imported_count": 0,
            "failed_count": 0,
            "skipped_count": 0,
            "current_batch": 0
        })
        
        # Process DOIs in batches
        for i in range(0, total_dois, batch_size):
            batch_dois = dois[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            logger.info(f"Processing batch {batch_num}: DOIs {i+1}-{min(i+batch_size, total_dois)}")
            
            try:
                # Get works from OpenAlex
                works = await client.get_works_batch(batch_dois)
                
                # Convert to papers and save
                for work in works:
                    try:
                        paper = OpenAlexConverter.convert_openalex_work_to_paper(work)
                        
                        # Check if paper already exists
                        existing_paper = await db_service.db.get_paper_by_doi(paper.doi) if paper.doi else None
                        
                        if existing_paper:
                            # Merge data and update
                            existing_paper_obj = Paper(**existing_paper)
                            merged_paper = OpenAlexConverter.merge_paper_data(existing_paper_obj, paper)
                            await db_service.db.update_paper(merged_paper.dict())
                            skipped_dois.append(paper.doi or work.id)
                        else:
                            # Save new paper
                            await db_service.db.create_paper(paper.dict())
                            imported_papers.append(paper)
                        
                    except Exception as e:
                        logger.error(f"Error processing work {work.id}: {e}")
                        failed_dois.append(work.ids.doi or work.id)
                
                # Find DOIs that weren't found in OpenAlex
                found_dois = set([work.ids.doi for work in works if work.ids.doi])
                missing_dois = set(batch_dois) - found_dois
                failed_dois.extend(missing_dois)
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {e}")
                failed_dois.extend(batch_dois)
            
            # Update progress
            processed_items = min(i + batch_size, total_dois)
            await self.update_task_progress(task_id, {
                "status": "running",
                "total_items": total_dois,
                "processed_items": processed_items,
                "imported_count": len(imported_papers),
                "failed_count": len(failed_dois),
                "skipped_count": len(skipped_dois),
                "current_batch": batch_num,
                "progress_percent": (processed_items / total_dois) * 100
            })
        
        # Final progress update
        await self.update_task_progress(task_id, {
            "status": "completed",
            "total_items": total_dois,
            "processed_items": total_dois,
            "imported_count": len(imported_papers),
            "failed_count": len(failed_dois),
            "skipped_count": len(skipped_dois),
            "progress_percent": 100,
            "completed_at": datetime.utcnow().isoformat()
        })
        
        logger.info(f"Batch import completed: {len(imported_papers)} imported, {len(failed_dois)} failed, {len(skipped_dois)} skipped")
        
        return {
            "task_id": task_id,
            "status": "completed",
            "imported_papers": [p.dict() for p in imported_papers],
            "failed_dois": failed_dois,
            "skipped_dois": skipped_dois,
            "summary": {
                "total": total_dois,
                "imported": len(imported_papers),
                "failed": len(failed_dois),
                "skipped": len(skipped_dois)
            }
        }
    
    return run_async_task(_batch_import())


@celery_app.task(base=OpenAlexTask, bind=True)
def build_citation_network(
    self,
    center_paper_id: str,
    max_depth: int = 2,
    max_papers_per_level: int = 50,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Build citation network for a paper using background processing.
    
    Args:
        center_paper_id: Center paper ID (DOI or OpenAlex ID)
        max_depth: Maximum traversal depth
        max_papers_per_level: Maximum papers per level
        user_id: User ID for tracking
        
    Returns:
        Citation network data
    """
    async def _build_network():
        task_id = self.request.id
        client = await self.get_openalex_client()
        db_service = await self.get_db_service()
        
        logger.info(f"Building citation network for {center_paper_id} (depth: {max_depth})")
        
        # Update initial progress
        await self.update_task_progress(task_id, {
            "status": "running",
            "center_paper_id": center_paper_id,
            "max_depth": max_depth,
            "current_depth": 0,
            "total_papers": 0,
            "total_edges": 0
        })
        
        try:
            # Build citation network
            network_data = await client.traverse_citation_network(
                center_paper_id,
                direction="both",
                max_depth=max_depth,
                max_works_per_level=max_papers_per_level
            )
            
            # Convert works to papers
            papers = OpenAlexConverter.batch_convert_works_to_papers(network_data["nodes"])
            
            # Save papers to database
            saved_papers = []
            for paper in papers:
                try:
                    # Check if paper already exists
                    existing_paper = await db_service.db.get_paper_by_doi(paper.doi) if paper.doi else None
                    
                    if existing_paper:
                        # Update existing paper
                        existing_paper_obj = Paper(**existing_paper)
                        merged_paper = OpenAlexConverter.merge_paper_data(existing_paper_obj, paper)
                        await db_service.db.update_paper(merged_paper.dict())
                        saved_papers.append(merged_paper)
                    else:
                        # Create new paper
                        await db_service.db.create_paper(paper.dict())
                        saved_papers.append(paper)
                        
                except Exception as e:
                    logger.warning(f"Error saving paper {paper.id}: {e}")
            
            # Save citation edges
            edges = network_data["edges"]
            for edge in edges:
                try:
                    await db_service.db.create_citation_edge(
                        edge["source_id"],
                        edge["target_id"],
                        edge["edge_type"],
                        edge.get("weight", 1.0)
                    )
                except Exception as e:
                    logger.warning(f"Error saving edge: {e}")
            
            # Cache network for quick access
            cache_key = f"citation_network:{center_paper_id}:{max_depth}"
            redis_manager = await self.get_redis_manager()
            await redis_manager.cache_set(cache_key, network_data, expire=3600)
            
            # Final progress update
            await self.update_task_progress(task_id, {
                "status": "completed",
                "center_paper_id": center_paper_id,
                "max_depth": max_depth,
                "total_papers": len(saved_papers),
                "total_edges": len(edges),
                "completed_at": datetime.utcnow().isoformat()
            })
            
            logger.info(f"Citation network built: {len(saved_papers)} papers, {len(edges)} edges")
            
            return {
                "task_id": task_id,
                "status": "completed",
                "network_data": {
                    "center_paper_id": center_paper_id,
                    "papers": [p.dict() for p in saved_papers],
                    "edges": edges,
                    "total_papers": len(saved_papers),
                    "total_edges": len(edges),
                    "max_depth_reached": network_data.get("max_depth_reached", 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Error building citation network: {e}")
            
            await self.update_task_progress(task_id, {
                "status": "failed",
                "error": str(e),
                "failed_at": datetime.utcnow().isoformat()
            })
            
            raise
    
    return run_async_task(_build_network())


@celery_app.task(base=OpenAlexTask, bind=True)
def search_and_import_papers(
    self,
    search_query: str,
    filters: Optional[Dict[str, Any]] = None,
    max_results: int = 100,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Search OpenAlex and import matching papers.
    
    Args:
        search_query: Search query string
        filters: Search filters
        max_results: Maximum results to import
        user_id: User ID for tracking
        
    Returns:
        Import results
    """
    async def _search_and_import():
        task_id = self.request.id
        client = await self.get_openalex_client()
        db_service = await self.get_db_service()
        
        logger.info(f"Searching and importing papers: {search_query} (max: {max_results})")
        
        # Update initial progress
        await self.update_task_progress(task_id, {
            "status": "running",
            "search_query": search_query,
            "max_results": max_results,
            "processed_results": 0,
            "imported_count": 0,
            "skipped_count": 0
        })
        
        try:
            # Parse filters
            search_filters = None
            if filters:
                search_filters = OpenAlexSearchFilters(**filters)
            
            # Search works
            imported_papers = []
            skipped_papers = []
            page = 1
            per_page = min(200, max_results)
            
            while len(imported_papers) + len(skipped_papers) < max_results:
                # Perform search
                search_response = await client.search_works(
                    query=search_query,
                    filters=search_filters,
                    page=page,
                    per_page=per_page
                )
                
                if not search_response.results:
                    break
                
                # Process results
                for work in search_response.results:
                    if len(imported_papers) + len(skipped_papers) >= max_results:
                        break
                    
                    try:
                        paper = OpenAlexConverter.convert_openalex_work_to_paper(work)
                        
                        # Check if paper already exists
                        existing_paper = None
                        if paper.doi:
                            existing_paper = await db_service.db.get_paper_by_doi(paper.doi)
                        
                        if existing_paper:
                            skipped_papers.append(paper)
                        else:
                            await db_service.db.create_paper(paper.dict())
                            imported_papers.append(paper)
                        
                    except Exception as e:
                        logger.warning(f"Error processing search result: {e}")
                
                # Update progress
                total_processed = len(imported_papers) + len(skipped_papers)
                await self.update_task_progress(task_id, {
                    "status": "running",
                    "search_query": search_query,
                    "max_results": max_results,
                    "processed_results": total_processed,
                    "imported_count": len(imported_papers),
                    "skipped_count": len(skipped_papers),
                    "current_page": page,
                    "progress_percent": (total_processed / max_results) * 100
                })
                
                # Check if we have more results
                if len(search_response.results) < per_page:
                    break
                
                page += 1
            
            # Final progress update
            await self.update_task_progress(task_id, {
                "status": "completed",
                "search_query": search_query,
                "max_results": max_results,
                "processed_results": len(imported_papers) + len(skipped_papers),
                "imported_count": len(imported_papers),
                "skipped_count": len(skipped_papers),
                "progress_percent": 100,
                "completed_at": datetime.utcnow().isoformat()
            })
            
            logger.info(f"Search import completed: {len(imported_papers)} imported, {len(skipped_papers)} skipped")
            
            return {
                "task_id": task_id,
                "status": "completed",
                "search_query": search_query,
                "imported_papers": [p.dict() for p in imported_papers],
                "skipped_papers": [p.dict() for p in skipped_papers],
                "summary": {
                    "total_processed": len(imported_papers) + len(skipped_papers),
                    "imported": len(imported_papers),
                    "skipped": len(skipped_papers)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in search and import: {e}")
            
            await self.update_task_progress(task_id, {
                "status": "failed",
                "error": str(e),
                "failed_at": datetime.utcnow().isoformat()
            })
            
            raise
    
    return run_async_task(_search_and_import())


@celery_app.task(base=OpenAlexTask, bind=True)
def update_paper_citations(self, paper_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Update citation counts and relationships for a specific paper.
    
    Args:
        paper_id: Paper ID to update
        user_id: User ID for tracking
        
    Returns:
        Update results
    """
    async def _update_citations():
        task_id = self.request.id
        client = await self.get_openalex_client()
        db_service = await self.get_db_service()
        
        logger.info(f"Updating citations for paper {paper_id}")
        
        try:
            # Get paper from database
            paper_data = await db_service.get_paper_with_cache(paper_id)
            if not paper_data:
                raise ValueError(f"Paper {paper_id} not found")
            
            paper = Paper(**paper_data)
            
            # Get OpenAlex work
            openalex_id = paper.openalex_id or paper.doi
            if not openalex_id:
                raise ValueError(f"No OpenAlex ID or DOI for paper {paper_id}")
            
            openalex_work = await client.get_work_by_id(openalex_id)
            if not openalex_work:
                raise ValueError(f"Paper not found in OpenAlex: {openalex_id}")
            
            # Update citation count
            paper.citation_count.openalex = openalex_work.cited_by_count
            paper.citation_count.total = openalex_work.cited_by_count
            paper.citation_count.last_updated = datetime.utcnow()
            
            # Get citing papers (limited to 100 for performance)
            citing_works = await client.get_citations(openalex_id, "cited_by", max_results=100)
            citing_paper_ids = []
            
            for citing_work in citing_works:
                try:
                    citing_paper = OpenAlexConverter.convert_openalex_work_to_paper(citing_work)
                    
                    # Save or update citing paper
                    existing_citing = await db_service.db.get_paper_by_doi(citing_paper.doi) if citing_paper.doi else None
                    
                    if existing_citing:
                        existing_citing_obj = Paper(**existing_citing)
                        merged_citing = OpenAlexConverter.merge_paper_data(existing_citing_obj, citing_paper)
                        await db_service.db.update_paper(merged_citing.dict())
                        citing_paper_ids.append(merged_citing.id)
                    else:
                        await db_service.db.create_paper(citing_paper.dict())
                        citing_paper_ids.append(citing_paper.id)
                    
                    # Create citation edge
                    await db_service.db.create_citation_edge(
                        citing_paper.id,
                        paper_id,
                        "cites",
                        1.0
                    )
                    
                except Exception as e:
                    logger.warning(f"Error processing citing work: {e}")
            
            # Update paper's cited_by list
            paper.cited_by = citing_paper_ids
            paper.updated_at = datetime.utcnow()
            
            # Save updated paper
            await db_service.db.update_paper(paper.dict())
            
            logger.info(f"Updated citations for paper {paper_id}: {len(citing_paper_ids)} citing papers")
            
            return {
                "task_id": task_id,
                "status": "completed",
                "paper_id": paper_id,
                "citation_count": openalex_work.cited_by_count,
                "citing_papers_found": len(citing_paper_ids),
                "updated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error updating citations for paper {paper_id}: {e}")
            raise
    
    return run_async_task(_update_citations())


@celery_app.task(base=OpenAlexTask, bind=True)
def sync_openalex_data(
    self,
    paper_ids: List[str],
    sync_type: str = "metadata",
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Synchronize paper data with OpenAlex.
    
    Args:
        paper_ids: List of paper IDs to sync
        sync_type: Type of sync ("metadata", "citations", "full")
        user_id: User ID for tracking
        
    Returns:
        Sync results
    """
    async def _sync_data():
        task_id = self.request.id
        client = await self.get_openalex_client()
        db_service = await self.get_db_service()
        
        logger.info(f"Syncing {len(paper_ids)} papers with OpenAlex ({sync_type})")
        
        updated_papers = []
        failed_papers = []
        
        # Update progress
        await self.update_task_progress(task_id, {
            "status": "running",
            "sync_type": sync_type,
            "total_papers": len(paper_ids),
            "processed_papers": 0,
            "updated_count": 0,
            "failed_count": 0
        })
        
        for i, paper_id in enumerate(paper_ids):
            try:
                # Get paper from database
                paper_data = await db_service.get_paper_with_cache(paper_id)
                if not paper_data:
                    failed_papers.append({"paper_id": paper_id, "error": "Paper not found"})
                    continue
                
                paper = Paper(**paper_data)
                openalex_id = paper.openalex_id or paper.doi
                
                if not openalex_id:
                    failed_papers.append({"paper_id": paper_id, "error": "No OpenAlex ID or DOI"})
                    continue
                
                # Get latest data from OpenAlex
                openalex_work = await client.get_work_by_id(openalex_id)
                if not openalex_work:
                    failed_papers.append({"paper_id": paper_id, "error": "Not found in OpenAlex"})
                    continue
                
                # Convert and merge data
                openalex_paper = OpenAlexConverter.convert_openalex_work_to_paper(openalex_work)
                
                if sync_type == "metadata":
                    # Sync only metadata
                    merged_paper = OpenAlexConverter.merge_paper_data(paper, openalex_paper)
                    await db_service.db.update_paper(merged_paper.dict())
                    updated_papers.append(merged_paper.dict())
                
                elif sync_type == "citations":
                    # Update citation data
                    paper.citation_count.openalex = openalex_work.cited_by_count
                    paper.citation_count.total = openalex_work.cited_by_count
                    paper.citation_count.last_updated = datetime.utcnow()
                    paper.updated_at = datetime.utcnow()
                    await db_service.db.update_paper(paper.dict())
                    updated_papers.append(paper.dict())
                
                elif sync_type == "full":
                    # Full sync including citations
                    merged_paper = OpenAlexConverter.merge_paper_data(paper, openalex_paper)
                    await db_service.db.update_paper(merged_paper.dict())
                    updated_papers.append(merged_paper.dict())
                
            except Exception as e:
                logger.error(f"Error syncing paper {paper_id}: {e}")
                failed_papers.append({"paper_id": paper_id, "error": str(e)})
            
            # Update progress
            await self.update_task_progress(task_id, {
                "status": "running",
                "sync_type": sync_type,
                "total_papers": len(paper_ids),
                "processed_papers": i + 1,
                "updated_count": len(updated_papers),
                "failed_count": len(failed_papers),
                "progress_percent": ((i + 1) / len(paper_ids)) * 100
            })
        
        # Final progress update
        await self.update_task_progress(task_id, {
            "status": "completed",
            "sync_type": sync_type,
            "total_papers": len(paper_ids),
            "processed_papers": len(paper_ids),
            "updated_count": len(updated_papers),
            "failed_count": len(failed_papers),
            "progress_percent": 100,
            "completed_at": datetime.utcnow().isoformat()
        })
        
        logger.info(f"Sync completed: {len(updated_papers)} updated, {len(failed_papers)} failed")
        
        return {
            "task_id": task_id,
            "status": "completed",
            "sync_type": sync_type,
            "updated_papers": updated_papers,
            "failed_papers": failed_papers,
            "summary": {
                "total": len(paper_ids),
                "updated": len(updated_papers),
                "failed": len(failed_papers)
            }
        }
    
    return run_async_task(_sync_data())


# Helper functions for task management

async def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """Get task status and progress."""
    redis_manager = await get_redis_manager()
    
    # Get task result from Celery
    result = celery_app.AsyncResult(task_id)
    
    # Get progress from Redis
    progress_key = f"task_progress:{task_id}"
    progress = await redis_manager.cache_get(progress_key)
    
    return {
        "task_id": task_id,
        "state": result.state,
        "result": result.result if result.ready() else None,
        "progress": progress,
        "ready": result.ready(),
        "successful": result.successful() if result.ready() else None,
        "failed": result.failed() if result.ready() else None
    }


async def cancel_task(task_id: str) -> bool:
    """Cancel a running task."""
    try:
        celery_app.control.revoke(task_id, terminate=True)
        
        # Update progress to cancelled
        redis_manager = await get_redis_manager()
        progress_key = f"task_progress:{task_id}"
        await redis_manager.cache_set(progress_key, {
            "status": "cancelled",
            "cancelled_at": datetime.utcnow().isoformat()
        }, expire=3600)
        
        return True
    except Exception as e:
        logger.error(f"Error cancelling task {task_id}: {e}")
        return False


def get_active_tasks() -> List[Dict[str, Any]]:
    """Get list of active tasks."""
    inspect = celery_app.control.inspect()
    
    active_tasks = []
    
    # Get active tasks from all workers
    active = inspect.active()
    if active:
        for worker, tasks in active.items():
            for task in tasks:
                active_tasks.append({
                    "task_id": task["id"],
                    "name": task["name"],
                    "worker": worker,
                    "args": task.get("args", []),
                    "kwargs": task.get("kwargs", {}),
                    "time_start": task.get("time_start")
                })
    
    return active_tasks