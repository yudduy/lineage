"""
Background task service using Celery for asynchronous processing.
"""

import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from celery import Celery
from celery.result import AsyncResult

from ..core.config import get_settings
from ..db.redis import get_redis_manager
from ..utils.logger import get_logger

logger = get_logger(__name__)


class CeleryTaskService:
    """Service for managing background tasks with Celery."""
    
    def __init__(self):
        self.settings = get_settings()
        self.celery_app: Optional[Celery] = None
        self.logger = logger
        
    def create_celery_app(self) -> Celery:
        """Create and configure Celery application."""
        if self.celery_app:
            return self.celery_app
        
        # Create Celery app
        celery_app = Celery(
            "citation_network_explorer",
            broker=self.settings.celery.broker_url,
            backend=self.settings.celery.result_backend
        )
        
        # Configure Celery
        celery_app.conf.update(
            task_serializer=self.settings.celery.task_serializer,
            result_serializer=self.settings.celery.result_serializer,
            accept_content=self.settings.celery.accept_content,
            timezone=self.settings.celery.timezone,
            enable_utc=self.settings.celery.enable_utc,
            
            # Task routing and execution
            task_routes={
                'citation_network_explorer.tasks.paper.*': {'queue': 'papers'},
                'citation_network_explorer.tasks.zotero.*': {'queue': 'zotero'},
                'citation_network_explorer.tasks.search.*': {'queue': 'search'},
                'citation_network_explorer.tasks.export.*': {'queue': 'export'},
            },
            
            # Task time limits
            task_soft_time_limit=300,  # 5 minutes
            task_time_limit=600,       # 10 minutes
            
            # Worker settings
            worker_prefetch_multiplier=1,
            worker_max_tasks_per_child=1000,
            
            # Result backend settings
            result_expires=3600,  # 1 hour
            result_persistent=True,
            
            # Monitoring
            worker_send_task_events=True,
            task_send_sent_event=True,
        )
        
        self.celery_app = celery_app
        self.logger.info("Celery application configured")
        
        return celery_app
    
    def get_celery_app(self) -> Celery:
        """Get or create Celery application."""
        if not self.celery_app:
            return self.create_celery_app()
        return self.celery_app
    
    async def submit_task(
        self,
        task_name: str,
        args: tuple = (),
        kwargs: dict = None,
        queue: str = "default",
        priority: int = 5,
        eta: Optional[datetime] = None,
        countdown: Optional[int] = None
    ) -> str:
        """
        Submit a background task.
        
        Args:
            task_name: Name of the task to execute
            args: Positional arguments for the task
            kwargs: Keyword arguments for the task
            queue: Queue to submit task to
            priority: Task priority (0-9, higher is more priority)
            eta: Specific datetime to execute task
            countdown: Delay in seconds before executing task
            
        Returns:
            Task ID for tracking
        """
        celery_app = self.get_celery_app()
        
        kwargs = kwargs or {}
        
        # Submit task
        result = celery_app.send_task(
            task_name,
            args=args,
            kwargs=kwargs,
            queue=queue,
            priority=priority,
            eta=eta,
            countdown=countdown
        )
        
        self.logger.info(
            "Task submitted",
            task_id=result.id,
            task_name=task_name,
            queue=queue,
            priority=priority
        )
        
        return result.id
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get status of a background task.
        
        Args:
            task_id: Task ID to check
            
        Returns:
            Task status information
        """
        celery_app = self.get_celery_app()
        
        result = AsyncResult(task_id, app=celery_app)
        
        status_info = {
            "task_id": task_id,
            "status": result.status,
            "ready": result.ready(),
            "successful": result.successful() if result.ready() else None,
            "failed": result.failed() if result.ready() else None,
            "result": None,
            "traceback": None,
            "meta": result.info if hasattr(result, 'info') else None
        }
        
        if result.ready():
            if result.successful():
                status_info["result"] = result.result
            elif result.failed():
                status_info["traceback"] = result.traceback
        
        return status_info
    
    async def cancel_task(self, task_id: str, terminate: bool = False) -> bool:
        """
        Cancel a background task.
        
        Args:
            task_id: Task ID to cancel
            terminate: Whether to terminate forcefully
            
        Returns:
            True if task was cancelled successfully
        """
        celery_app = self.get_celery_app()
        
        try:
            celery_app.control.revoke(task_id, terminate=terminate, signal='SIGKILL' if terminate else 'SIGTERM')
            
            self.logger.info(
                "Task cancelled",
                task_id=task_id,
                terminate=terminate
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cancel task {task_id}: {e}")
            return False
    
    async def get_queue_info(self, queue_name: str = "default") -> Dict[str, Any]:
        """
        Get information about a task queue.
        
        Args:
            queue_name: Name of the queue to inspect
            
        Returns:
            Queue information including task counts
        """
        celery_app = self.get_celery_app()
        
        try:
            # Get active tasks
            inspect = celery_app.control.inspect()
            active_tasks = inspect.active()
            scheduled_tasks = inspect.scheduled()
            reserved_tasks = inspect.reserved()
            
            queue_info = {
                "queue_name": queue_name,
                "active_tasks": 0,
                "scheduled_tasks": 0,
                "reserved_tasks": 0,
                "total_tasks": 0
            }
            
            # Count tasks for the specified queue
            if active_tasks:
                for worker, tasks in active_tasks.items():
                    queue_info["active_tasks"] += len([t for t in tasks if t.get("delivery_info", {}).get("routing_key") == queue_name])
            
            if scheduled_tasks:
                for worker, tasks in scheduled_tasks.items():
                    queue_info["scheduled_tasks"] += len([t for t in tasks if t.get("delivery_info", {}).get("routing_key") == queue_name])
            
            if reserved_tasks:
                for worker, tasks in reserved_tasks.items():
                    queue_info["reserved_tasks"] += len([t for t in tasks if t.get("delivery_info", {}).get("routing_key") == queue_name])
            
            queue_info["total_tasks"] = queue_info["active_tasks"] + queue_info["scheduled_tasks"] + queue_info["reserved_tasks"]
            
            return queue_info
            
        except Exception as e:
            self.logger.error(f"Failed to get queue info for {queue_name}: {e}")
            return {"queue_name": queue_name, "error": str(e)}
    
    async def get_worker_stats(self) -> Dict[str, Any]:
        """
        Get statistics about Celery workers.
        
        Returns:
            Worker statistics and health information
        """
        celery_app = self.get_celery_app()
        
        try:
            inspect = celery_app.control.inspect()
            
            # Get worker information
            stats = inspect.stats()
            active = inspect.active()
            registered = inspect.registered()
            
            worker_info = {
                "total_workers": len(stats) if stats else 0,
                "workers": []
            }
            
            if stats:
                for worker_name, worker_stats in stats.items():
                    worker_data = {
                        "name": worker_name,
                        "status": "online",
                        "pool": worker_stats.get("pool", {}),
                        "total_tasks": worker_stats.get("total", 0),
                        "active_tasks": len(active.get(worker_name, [])) if active else 0,
                        "registered_tasks": len(registered.get(worker_name, [])) if registered else 0,
                        "load_average": worker_stats.get("rusage", {}).get("utime", 0),
                        "memory_usage": worker_stats.get("rusage", {}).get("maxrss", 0)
                    }
                    worker_info["workers"].append(worker_data)
            
            return worker_info
            
        except Exception as e:
            self.logger.error(f"Failed to get worker stats: {e}")
            return {"error": str(e), "total_workers": 0, "workers": []}


# Task definitions
def create_task_definitions(celery_app: Celery):
    """Create Celery task definitions."""
    
    @celery_app.task(name="citation_network_explorer.tasks.paper.fetch_paper_metadata")
    def fetch_paper_metadata(paper_id: str, sources: List[str] = None) -> Dict[str, Any]:
        """
        Fetch paper metadata from external sources.
        
        Args:
            paper_id: Paper ID to fetch metadata for
            sources: List of sources to query (openalex, semantic_scholar, etc.)
            
        Returns:
            Updated paper metadata
        """
        logger.info(f"Fetching metadata for paper {paper_id}")
        
        try:
            # TODO: Implement metadata fetching from external APIs
            # This would query OpenAlex, Semantic Scholar, CrossRef, etc.
            
            # Mock implementation
            metadata = {
                "paper_id": paper_id,
                "title": "Example Paper Title",
                "authors": ["Author One", "Author Two"],
                "abstract": "This is an example abstract...",
                "publication_date": "2023-01-01",
                "journal": "Example Journal",
                "citation_count": 25,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Successfully fetched metadata for paper {paper_id}")
            return {"success": True, "data": metadata}
            
        except Exception as e:
            logger.error(f"Failed to fetch metadata for paper {paper_id}: {e}")
            return {"success": False, "error": str(e)}
    
    @celery_app.task(name="citation_network_explorer.tasks.paper.build_citation_network")
    def build_citation_network(paper_id: str, max_depth: int = 2) -> Dict[str, Any]:
        """
        Build citation network for a paper.
        
        Args:
            paper_id: Central paper ID
            max_depth: Maximum depth to traverse
            
        Returns:
            Citation network data
        """
        logger.info(f"Building citation network for paper {paper_id}")
        
        try:
            # TODO: Implement citation network building
            # This would traverse citations and build the network graph
            
            # Mock implementation
            network = {
                "center_paper_id": paper_id,
                "nodes": [
                    {"id": paper_id, "title": "Center Paper"},
                    {"id": "ref1", "title": "Reference 1"},
                    {"id": "ref2", "title": "Reference 2"},
                    {"id": "cite1", "title": "Citing Paper 1"}
                ],
                "edges": [
                    {"source": paper_id, "target": "ref1", "type": "cites"},
                    {"source": paper_id, "target": "ref2", "type": "cites"},
                    {"source": "cite1", "target": paper_id, "type": "cites"}
                ],
                "depth": max_depth,
                "built_at": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Successfully built citation network for paper {paper_id}")
            return {"success": True, "data": network}
            
        except Exception as e:
            logger.error(f"Failed to build citation network for paper {paper_id}: {e}")
            return {"success": False, "error": str(e)}
    
    @celery_app.task(name="citation_network_explorer.tasks.zotero.sync_collection")
    def sync_zotero_collection(user_id: str, collection_key: str) -> Dict[str, Any]:
        """
        Synchronize a Zotero collection.
        
        Args:
            user_id: User ID
            collection_key: Zotero collection key
            
        Returns:
            Sync results
        """
        logger.info(f"Syncing Zotero collection {collection_key} for user {user_id}")
        
        try:
            # TODO: Implement Zotero collection sync
            # This would fetch items from Zotero API and update local database
            
            # Mock implementation
            sync_result = {
                "user_id": user_id,
                "collection_key": collection_key,
                "items_synced": 15,
                "items_added": 3,
                "items_updated": 2,
                "items_deleted": 1,
                "sync_time": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Successfully synced Zotero collection {collection_key}")
            return {"success": True, "data": sync_result}
            
        except Exception as e:
            logger.error(f"Failed to sync Zotero collection {collection_key}: {e}")
            return {"success": False, "error": str(e)}
    
    @celery_app.task(name="citation_network_explorer.tasks.export.generate_bibtex")
    def generate_bibtex_export(user_id: str, paper_ids: List[str]) -> Dict[str, Any]:
        """
        Generate BibTeX export for papers.
        
        Args:
            user_id: User ID requesting export
            paper_ids: List of paper IDs to export
            
        Returns:
            Export results with download URL
        """
        logger.info(f"Generating BibTeX export for user {user_id} ({len(paper_ids)} papers)")
        
        try:
            # TODO: Implement BibTeX generation
            # This would fetch paper data and format as BibTeX
            
            # Mock implementation
            bibtex_content = """
@article{example2023,
  title={Example Paper Title},
  author={Author, One and Two, Author},
  journal={Example Journal},
  year={2023},
  publisher={Example Publisher}
}
"""
            
            # TODO: Save to file storage and generate download URL
            download_url = f"/exports/bibtex_{user_id}_{datetime.utcnow().timestamp()}.bib"
            
            export_result = {
                "user_id": user_id,
                "paper_count": len(paper_ids),
                "format": "bibtex",
                "file_size": len(bibtex_content),
                "download_url": download_url,
                "expires_at": (datetime.utcnow() + timedelta(hours=24)).isoformat(),
                "generated_at": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Successfully generated BibTeX export for user {user_id}")
            return {"success": True, "data": export_result}
            
        except Exception as e:
            logger.error(f"Failed to generate BibTeX export for user {user_id}: {e}")
            return {"success": False, "error": str(e)}
    
    @celery_app.task(name="citation_network_explorer.tasks.search.update_search_index")
    def update_search_index(paper_ids: List[str] = None) -> Dict[str, Any]:
        """
        Update search index for papers.
        
        Args:
            paper_ids: Specific paper IDs to update, or None for full reindex
            
        Returns:
            Index update results
        """
        logger.info(f"Updating search index for {len(paper_ids) if paper_ids else 'all'} papers")
        
        try:
            # TODO: Implement search index update
            # This would update Elasticsearch or other search backend
            
            # Mock implementation
            if paper_ids:
                papers_updated = len(paper_ids)
            else:
                papers_updated = 10000  # Mock full index size
            
            index_result = {
                "papers_updated": papers_updated,
                "index_size": papers_updated + 5000,  # Mock existing size
                "update_time_seconds": 45.2,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Successfully updated search index ({papers_updated} papers)")
            return {"success": True, "data": index_result}
            
        except Exception as e:
            logger.error(f"Failed to update search index: {e}")
            return {"success": False, "error": str(e)}


# Global task service instance
task_service = CeleryTaskService()


def get_task_service() -> CeleryTaskService:
    """Get the global task service instance."""
    return task_service


def get_celery_app() -> Celery:
    """Get configured Celery application."""
    celery_app = task_service.get_celery_app()
    create_task_definitions(celery_app)
    return celery_app