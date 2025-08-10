"""
LLM Background Tasks - Celery tasks for batch processing and async enrichment.

This module provides:
- Async paper enrichment tasks
- Batch citation analysis processing
- Research trajectory analysis tasks
- Cost-aware task scheduling
- Error handling and retry logic
- Progress tracking and reporting
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from celery import Celery, Task
from celery.exceptions import Retry, WorkerLostError
from celery.result import AsyncResult
from celery.signals import task_prerun, task_postrun, task_failure
import logging

from ..core.config import get_settings
from ..services.content_enrichment import get_enrichment_service, EnrichmentRequest, BatchEnrichmentResult
from ..services.citation_analysis import get_citation_analysis_service, CitationAnalysisRequest
from ..services.research_trajectory import get_trajectory_service
from ..services.llm_cost_manager import get_cost_manager, CostCategory
from ..db.redis import get_redis_manager
from ..utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Create Celery app
celery_app = Celery(
    'llm_tasks',
    broker=settings.celery.broker_url,
    backend=settings.celery.result_backend,
    include=['app.services.llm_tasks']
)

# Configure Celery
celery_app.conf.update(
    task_serializer=settings.celery.task_serializer,
    result_serializer=settings.celery.result_serializer,
    accept_content=settings.celery.accept_content,
    timezone=settings.celery.timezone,
    enable_utc=settings.celery.enable_utc,
    task_routes={
        'app.services.llm_tasks.enrich_paper_async': {'queue': 'enrichment'},
        'app.services.llm_tasks.batch_enrich_papers_async': {'queue': 'batch_enrichment'},
        'app.services.llm_tasks.analyze_citation_async': {'queue': 'citation_analysis'},
        'app.services.llm_tasks.trace_lineage_async': {'queue': 'trajectory_analysis'},
    },
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,  # Important for memory management with LLMs
    task_acks_late=True,
    worker_disable_rate_limits=False,
    task_default_retry_delay=60,  # 1 minute
    task_max_retries=3,
)


class LLMTask(Task):
    """Base class for LLM tasks with common functionality."""
    
    abstract = True
    
    def __init__(self):
        self._services_initialized = False
        self._enrichment_service = None
        self._citation_service = None
        self._trajectory_service = None
        self._cost_manager = None
        self._redis_manager = None
    
    async def _ensure_services_initialized(self):
        """Ensure all services are initialized."""
        if not self._services_initialized:
            self._enrichment_service = await get_enrichment_service()
            self._citation_service = await get_citation_analysis_service()
            self._trajectory_service = await get_trajectory_service()
            self._cost_manager = await get_cost_manager()
            self._redis_manager = await get_redis_manager()
            self._services_initialized = True
    
    def retry_with_countdown(self, exc=None, countdown=None, max_retries=None):
        """Retry task with exponential backoff."""
        if countdown is None:
            # Exponential backoff: 1min, 2min, 4min, 8min
            countdown = min(60 * (2 ** self.request.retries), 480)
        
        if max_retries is None:
            max_retries = 3
        
        raise self.retry(exc=exc, countdown=countdown, max_retries=max_retries)
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure."""
        logger.error(f"Task {task_id} failed: {exc}")
        
        # Update task status in Redis if available
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def update_status():
                redis_manager = await get_redis_manager()
                if redis_manager:
                    status_key = f"task_status:{task_id}"
                    await redis_manager.setex(
                        status_key,
                        3600,  # 1 hour TTL
                        json.dumps({
                            'status': 'FAILED',
                            'error': str(exc),
                            'timestamp': datetime.now().isoformat()
                        })
                    )
            
            loop.run_until_complete(update_status())
            loop.close()
        except Exception as e:
            logger.warning(f"Failed to update task status: {e}")
    
    def on_success(self, retval, task_id, args, kwargs):
        """Handle task success."""
        logger.info(f"Task {task_id} completed successfully")
        
        # Update task status in Redis
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def update_status():
                redis_manager = await get_redis_manager()
                if redis_manager:
                    status_key = f"task_status:{task_id}"
                    await redis_manager.setex(
                        status_key,
                        3600,  # 1 hour TTL
                        json.dumps({
                            'status': 'SUCCESS',
                            'result_summary': f"Completed at {datetime.now().isoformat()}",
                            'timestamp': datetime.now().isoformat()
                        })
                    )
            
            loop.run_until_complete(update_status())
            loop.close()
        except Exception as e:
            logger.warning(f"Failed to update task status: {e}")


@celery_app.task(base=LLMTask, bind=True, name='app.services.llm_tasks.enrich_paper_async')
def enrich_paper_async(
    self,
    paper_id: str,
    paper_data: Optional[Dict[str, Any]] = None,
    force_refresh: bool = False,
    priority: str = 'normal'
):
    """
    Asynchronously enrich a single paper with LLM analysis.
    
    Args:
        paper_id: Unique paper identifier
        paper_data: Optional paper metadata
        force_refresh: Force new analysis even if cached
        priority: Task priority (high/normal/low)
    
    Returns:
        Enriched content dictionary
    """
    
    async def _enrich_paper():
        try:
            await self._ensure_services_initialized()
            
            # Check budget before proceeding
            budget_available, message = await self._cost_manager.is_budget_available(estimated_cost=0.05)
            if not budget_available:
                logger.warning(f"Budget limit reached, skipping enrichment for {paper_id}: {message}")
                return {
                    'status': 'SKIPPED',
                    'reason': 'Budget limit reached',
                    'paper_id': paper_id
                }
            
            # Perform enrichment
            enriched_content = await self._enrichment_service.enrich_paper(
                paper_id=paper_id,
                paper_data=paper_data,
                force_refresh=force_refresh,
                use_cache=True
            )
            
            # Record task completion
            await self._cost_manager.record_usage(
                model=enriched_content.enrichment_model or 'unknown',
                provider='unknown',
                input_tokens=enriched_content.enrichment_tokens // 2,  # Estimate input tokens
                output_tokens=enriched_content.enrichment_tokens // 2,  # Estimate output tokens
                category=CostCategory.BACKGROUND_TASK,
                cached=False
            )
            
            logger.info(f"Successfully enriched paper {paper_id} in background task")
            
            return {
                'status': 'SUCCESS',
                'paper_id': paper_id,
                'enrichment_quality': enriched_content.content_quality.value,
                'cost': enriched_content.enrichment_cost,
                'tokens': enriched_content.enrichment_tokens,
                'confidence': enriched_content.confidence_score
            }
            
        except Exception as exc:
            logger.error(f"Failed to enrich paper {paper_id}: {exc}")
            
            # Retry on transient errors
            if isinstance(exc, (ConnectionError, TimeoutError)) and self.request.retries < 3:
                self.retry_with_countdown(exc=exc)
            
            return {
                'status': 'FAILED',
                'paper_id': paper_id,
                'error': str(exc)
            }
    
    # Run async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(_enrich_paper())
        return result
    finally:
        loop.close()


@celery_app.task(base=LLMTask, bind=True, name='app.services.llm_tasks.batch_enrich_papers_async')
def batch_enrich_papers_async(
    self,
    paper_requests: List[Dict[str, Any]],
    max_concurrency: int = 3,
    priority: str = 'normal'
):
    """
    Asynchronously enrich multiple papers in batch.
    
    Args:
        paper_requests: List of paper enrichment requests
        max_concurrency: Maximum concurrent enrichments
        priority: Task priority
        
    Returns:
        Batch enrichment result summary
    """
    
    async def _batch_enrich():
        try:
            await self._ensure_services_initialized()
            
            # Convert requests to EnrichmentRequest objects
            enrichment_requests = []
            for req_data in paper_requests:
                request = EnrichmentRequest(
                    paper_id=req_data['paper_id'],
                    paper_data=req_data.get('paper_data'),
                    priority=req_data.get('priority', 1),
                    force_refresh=req_data.get('force_refresh', False)
                )
                enrichment_requests.append(request)
            
            # Check overall budget
            estimated_cost = len(enrichment_requests) * 0.05  # Estimate per paper
            budget_available, message = await self._cost_manager.is_budget_available(estimated_cost)
            
            if not budget_available:
                logger.warning(f"Insufficient budget for batch enrichment: {message}")
                return {
                    'status': 'FAILED',
                    'reason': 'Insufficient budget',
                    'requested_papers': len(paper_requests),
                    'processed_papers': 0
                }
            
            # Perform batch enrichment
            batch_result = await self._enrichment_service.batch_enrich_papers(
                paper_requests=enrichment_requests,
                max_concurrency=max_concurrency
            )
            
            logger.info(f"Batch enrichment completed: {batch_result.successful_enrichments}/"
                       f"{batch_result.total_papers} papers, ${batch_result.total_cost:.4f}")
            
            return {
                'status': 'SUCCESS',
                'total_papers': batch_result.total_papers,
                'successful_enrichments': batch_result.successful_enrichments,
                'failed_enrichments': batch_result.failed_enrichments,
                'cached_results': batch_result.cached_results,
                'total_cost': batch_result.total_cost,
                'total_tokens': batch_result.total_tokens,
                'processing_time_seconds': batch_result.processing_time_seconds,
                'errors': batch_result.error_summary[:5]  # Limit error details
            }
            
        except Exception as exc:
            logger.error(f"Batch enrichment failed: {exc}")
            
            if isinstance(exc, (ConnectionError, TimeoutError)) and self.request.retries < 2:
                self.retry_with_countdown(exc=exc, max_retries=2)
            
            return {
                'status': 'FAILED',
                'error': str(exc),
                'requested_papers': len(paper_requests),
                'processed_papers': 0
            }
    
    # Run async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(_batch_enrich())
        return result
    finally:
        loop.close()


@celery_app.task(base=LLMTask, bind=True, name='app.services.llm_tasks.analyze_citation_async')
def analyze_citation_async(
    self,
    citing_paper_id: str,
    cited_paper_id: str,
    citation_context: Optional[str] = None,
    force_refresh: bool = False
):
    """
    Asynchronously analyze citation relationship.
    
    Args:
        citing_paper_id: Paper that cites
        cited_paper_id: Paper being cited
        citation_context: Optional citation context
        force_refresh: Force new analysis
        
    Returns:
        Citation analysis result
    """
    
    async def _analyze_citation():
        try:
            await self._ensure_services_initialized()
            
            # Check budget
            budget_available, message = await self._cost_manager.is_budget_available(estimated_cost=0.03)
            if not budget_available:
                logger.warning(f"Budget limit reached, skipping citation analysis: {message}")
                return {
                    'status': 'SKIPPED',
                    'reason': 'Budget limit reached',
                    'citing_paper_id': citing_paper_id,
                    'cited_paper_id': cited_paper_id
                }
            
            # Perform analysis
            relationship = await self._citation_service.analyze_citation_relationship(
                citing_paper_id=citing_paper_id,
                cited_paper_id=cited_paper_id,
                citation_context=citation_context,
                force_refresh=force_refresh
            )
            
            logger.info(f"Successfully analyzed citation: {citing_paper_id} -> {cited_paper_id}")
            
            return {
                'status': 'SUCCESS',
                'citing_paper_id': citing_paper_id,
                'cited_paper_id': cited_paper_id,
                'citation_type': relationship.citation_type.value if relationship.citation_type else None,
                'influence_level': relationship.influence_level.value if relationship.influence_level else None,
                'analysis_confidence': relationship.analysis_confidence,
                'analysis_cost': relationship.analysis_cost
            }
            
        except Exception as exc:
            logger.error(f"Citation analysis failed: {citing_paper_id} -> {cited_paper_id}: {exc}")
            
            if isinstance(exc, (ConnectionError, TimeoutError)) and self.request.retries < 3:
                self.retry_with_countdown(exc=exc)
            
            return {
                'status': 'FAILED',
                'citing_paper_id': citing_paper_id,
                'cited_paper_id': cited_paper_id,
                'error': str(exc)
            }
    
    # Run async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(_analyze_citation())
        return result
    finally:
        loop.close()


@celery_app.task(base=LLMTask, bind=True, name='app.services.llm_tasks.trace_lineage_async')
def trace_lineage_async(
    self,
    seed_paper_ids: List[str],
    max_generations: int = 3,
    include_future_work: bool = True,
    generate_timeline: bool = False
):
    """
    Asynchronously trace intellectual lineage.
    
    Args:
        seed_paper_ids: Starting papers
        max_generations: Maximum generations to trace
        include_future_work: Include citing papers
        generate_timeline: Generate timeline narrative
        
    Returns:
        Lineage analysis result
    """
    
    async def _trace_lineage():
        try:
            await self._ensure_services_initialized()
            
            # Estimate cost based on expected analysis complexity
            estimated_cost = len(seed_paper_ids) * max_generations * 0.10
            budget_available, message = await self._cost_manager.is_budget_available(estimated_cost)
            
            if not budget_available:
                logger.warning(f"Insufficient budget for lineage tracing: {message}")
                return {
                    'status': 'FAILED',
                    'reason': 'Insufficient budget',
                    'seed_papers': seed_paper_ids
                }
            
            # Trace lineage
            lineage = await self._trajectory_service.trace_intellectual_lineage(
                seed_paper_ids=seed_paper_ids,
                max_generations=max_generations,
                include_future_work=include_future_work
            )
            
            result = {
                'status': 'SUCCESS',
                'lineage_id': lineage.lineage_id,
                'total_papers': lineage.total_papers,
                'time_span_years': lineage.time_span_years,
                'milestone_count': len(lineage.milestones),
                'trajectory_type': lineage.trajectory_type.value if lineage.trajectory_type else None,
                'analysis_confidence': lineage.analysis_confidence,
                'analysis_cost': lineage.analysis_cost
            }
            
            # Generate timeline if requested
            if generate_timeline and lineage.milestones:
                try:
                    timeline = await self._trajectory_service.generate_timeline_narrative(
                        lineage=lineage,
                        include_context=True
                    )
                    result['timeline_generated'] = True
                    result['timeline_id'] = timeline.timeline_id
                    result['timeline_periods'] = len(timeline.periods)
                except Exception as e:
                    logger.warning(f"Failed to generate timeline: {e}")
                    result['timeline_generated'] = False
                    result['timeline_error'] = str(e)
            
            logger.info(f"Successfully traced lineage: {lineage.total_papers} papers, "
                       f"{len(lineage.milestones)} milestones")
            
            return result
            
        except Exception as exc:
            logger.error(f"Lineage tracing failed for {seed_paper_ids}: {exc}")
            
            if isinstance(exc, (ConnectionError, TimeoutError)) and self.request.retries < 2:
                self.retry_with_countdown(exc=exc, max_retries=2)
            
            return {
                'status': 'FAILED',
                'seed_papers': seed_paper_ids,
                'error': str(exc)
            }
    
    # Run async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(_trace_lineage())
        return result
    finally:
        loop.close()


@celery_app.task(name='app.services.llm_tasks.cleanup_expired_tasks')
def cleanup_expired_tasks():
    """Clean up expired task results and status entries."""
    
    async def _cleanup():
        try:
            redis_manager = await get_redis_manager()
            if not redis_manager:
                return
            
            # Clean up expired task statuses
            task_keys = await redis_manager.keys("task_status:*")
            
            cleanup_count = 0
            for key in task_keys:
                try:
                    status_data = await redis_manager.get(key)
                    if status_data:
                        status = json.loads(status_data)
                        timestamp_str = status.get('timestamp')
                        
                        if timestamp_str:
                            timestamp = datetime.fromisoformat(timestamp_str)
                            age = datetime.now() - timestamp
                            
                            # Remove statuses older than 24 hours
                            if age > timedelta(hours=24):
                                await redis_manager.delete(key)
                                cleanup_count += 1
                except Exception as e:
                    logger.warning(f"Failed to process task status {key}: {e}")
            
            if cleanup_count > 0:
                logger.info(f"Cleaned up {cleanup_count} expired task statuses")
            
            return {'cleaned_up': cleanup_count}
            
        except Exception as e:
            logger.error(f"Task cleanup failed: {e}")
            return {'error': str(e)}
    
    # Run async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(_cleanup())
        return result
    finally:
        loop.close()


# Task management utilities

class TaskManager:
    """Utility class for managing LLM background tasks."""
    
    def __init__(self):
        self.redis_manager = None
    
    async def initialize(self):
        """Initialize task manager."""
        if not self.redis_manager:
            self.redis_manager = await get_redis_manager()
    
    async def submit_paper_enrichment(
        self,
        paper_id: str,
        paper_data: Optional[Dict[str, Any]] = None,
        priority: str = 'normal',
        force_refresh: bool = False
    ) -> str:
        """Submit paper enrichment task."""
        task = enrich_paper_async.apply_async(
            args=[paper_id],
            kwargs={
                'paper_data': paper_data,
                'force_refresh': force_refresh,
                'priority': priority
            },
            queue='enrichment',
            priority=self._get_priority_value(priority)
        )
        
        await self._track_task(task.id, 'paper_enrichment', {
            'paper_id': paper_id,
            'priority': priority
        })
        
        return task.id
    
    async def submit_batch_enrichment(
        self,
        paper_requests: List[Dict[str, Any]],
        max_concurrency: int = 3,
        priority: str = 'normal'
    ) -> str:
        """Submit batch enrichment task."""
        task = batch_enrich_papers_async.apply_async(
            args=[paper_requests],
            kwargs={
                'max_concurrency': max_concurrency,
                'priority': priority
            },
            queue='batch_enrichment',
            priority=self._get_priority_value(priority)
        )
        
        await self._track_task(task.id, 'batch_enrichment', {
            'paper_count': len(paper_requests),
            'max_concurrency': max_concurrency,
            'priority': priority
        })
        
        return task.id
    
    async def submit_citation_analysis(
        self,
        citing_paper_id: str,
        cited_paper_id: str,
        citation_context: Optional[str] = None,
        force_refresh: bool = False
    ) -> str:
        """Submit citation analysis task."""
        task = analyze_citation_async.apply_async(
            args=[citing_paper_id, cited_paper_id],
            kwargs={
                'citation_context': citation_context,
                'force_refresh': force_refresh
            },
            queue='citation_analysis'
        )
        
        await self._track_task(task.id, 'citation_analysis', {
            'citing_paper_id': citing_paper_id,
            'cited_paper_id': cited_paper_id
        })
        
        return task.id
    
    async def submit_lineage_tracing(
        self,
        seed_paper_ids: List[str],
        max_generations: int = 3,
        include_future_work: bool = True,
        generate_timeline: bool = False
    ) -> str:
        """Submit lineage tracing task."""
        task = trace_lineage_async.apply_async(
            args=[seed_paper_ids],
            kwargs={
                'max_generations': max_generations,
                'include_future_work': include_future_work,
                'generate_timeline': generate_timeline
            },
            queue='trajectory_analysis'
        )
        
        await self._track_task(task.id, 'lineage_tracing', {
            'seed_papers': seed_paper_ids,
            'max_generations': max_generations,
            'generate_timeline': generate_timeline
        })
        
        return task.id
    
    def _get_priority_value(self, priority: str) -> int:
        """Convert priority string to numeric value."""
        priority_map = {
            'high': 9,
            'normal': 5,
            'low': 1
        }
        return priority_map.get(priority.lower(), 5)
    
    async def _track_task(self, task_id: str, task_type: str, metadata: Dict[str, Any]):
        """Track task in Redis."""
        await self.initialize()
        
        if self.redis_manager:
            try:
                task_info = {
                    'task_id': task_id,
                    'task_type': task_type,
                    'metadata': metadata,
                    'submitted_at': datetime.now().isoformat(),
                    'status': 'PENDING'
                }
                
                await self.redis_manager.setex(
                    f"task_info:{task_id}",
                    86400,  # 24 hours
                    json.dumps(task_info)
                )
            except Exception as e:
                logger.warning(f"Failed to track task {task_id}: {e}")
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get task status and result."""
        await self.initialize()
        
        # Get Celery task result
        result = AsyncResult(task_id, app=celery_app)
        
        task_info = {
            'task_id': task_id,
            'status': result.status,
            'ready': result.ready(),
            'successful': result.successful() if result.ready() else None,
            'failed': result.failed() if result.ready() else None,
        }
        
        if result.ready():
            if result.successful():
                task_info['result'] = result.result
            else:
                task_info['error'] = str(result.result)
        
        # Get additional info from Redis
        if self.redis_manager:
            try:
                task_data = await self.redis_manager.get(f"task_info:{task_id}")
                if task_data:
                    additional_info = json.loads(task_data)
                    task_info.update(additional_info)
            except Exception as e:
                logger.warning(f"Failed to get additional task info: {e}")
        
        return task_info
    
    async def get_active_tasks(self) -> List[Dict[str, Any]]:
        """Get list of active tasks."""
        await self.initialize()
        
        active_tasks = []
        
        if self.redis_manager:
            try:
                task_keys = await self.redis_manager.keys("task_info:*")
                
                for key in task_keys:
                    task_data = await self.redis_manager.get(key)
                    if task_data:
                        task_info = json.loads(task_data)
                        task_id = task_info['task_id']
                        
                        # Get current status from Celery
                        result = AsyncResult(task_id, app=celery_app)
                        task_info['current_status'] = result.status
                        task_info['ready'] = result.ready()
                        
                        active_tasks.append(task_info)
                        
            except Exception as e:
                logger.warning(f"Failed to get active tasks: {e}")
        
        return active_tasks


# Global task manager instance
_task_manager: Optional[TaskManager] = None


async def get_task_manager() -> TaskManager:
    """Get or create the global task manager."""
    global _task_manager
    
    if _task_manager is None:
        _task_manager = TaskManager()
        await _task_manager.initialize()
    
    return _task_manager


# Periodic task for cleanup
@celery_app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    """Set up periodic tasks."""
    # Clean up expired tasks every hour
    sender.add_periodic_task(
        3600.0,  # 1 hour
        cleanup_expired_tasks.s(),
        name='cleanup_expired_tasks'
    )