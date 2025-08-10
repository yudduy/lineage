"""
Semantic Scholar background tasks using Celery.

Provides asynchronous task processing for semantic enrichment, analysis,
and large-scale data processing operations.
"""

import asyncio
import time
import json
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from celery import Celery
from celery.utils.log import get_task_logger

from ..core.config import get_settings
from ..services.semantic_scholar import get_semantic_scholar_client
from ..services.semantic_analysis import get_semantic_analysis_service
from ..services.enrichment_pipeline import (
    get_enrichment_pipeline,
    EnrichmentPriority,
    EnrichmentStatus
)
from ..models.semantic_scholar import EnrichedPaper
from ..db.redis import get_redis_manager
from ..utils.exceptions import APIError, ValidationError

# Initialize settings and Celery
settings = get_settings()
logger = get_task_logger(__name__)

# Create Celery app
celery_app = Celery(
    "semantic_scholar_tasks",
    broker=settings.celery.broker_url,
    backend=settings.celery.result_backend,
    include=["app.services.semantic_scholar_tasks"]
)

# Configure Celery
celery_app.conf.update(
    task_serializer=settings.celery.task_serializer,
    accept_content=settings.celery.accept_content,
    result_serializer=settings.celery.result_serializer,
    timezone=settings.celery.timezone,
    enable_utc=settings.celery.enable_utc,
    task_track_started=True,
    task_time_limit=1800,  # 30 minutes
    task_soft_time_limit=1500,  # 25 minutes
    worker_prefetch_multiplier=4,
    task_routes={
        "semantic_scholar_tasks.enrich_paper": {"queue": "semantic_enrichment"},
        "semantic_scholar_tasks.enrich_papers_batch": {"queue": "semantic_enrichment"},
        "semantic_scholar_tasks.analyze_research_trajectory": {"queue": "semantic_analysis"},
        "semantic_scholar_tasks.identify_emerging_trends": {"queue": "semantic_analysis"},
        "semantic_scholar_tasks.build_citation_network": {"queue": "semantic_analysis"},
        "semantic_scholar_tasks.process_enrichment_queue": {"queue": "semantic_pipeline"}
    }
)


def run_async_task(coro):
    """Helper to run async function in Celery task."""
    try:
        # Create new event loop for the task
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@celery_app.task(bind=True, name="semantic_scholar_tasks.enrich_paper")
def enrich_paper_task(
    self,
    paper_identifier: str,
    priority: str = "medium",
    include_citations: bool = True,
    include_references: bool = True,
    include_embeddings: bool = True,
    include_semantic_analysis: bool = True,
    max_citations: int = 100,
    max_references: int = 100
) -> Dict[str, Any]:
    """
    Celery task for enriching a single paper with semantic features.
    
    Args:
        paper_identifier: Paper DOI, OpenAlex ID, or Semantic Scholar ID
        priority: Task priority level
        include_citations: Whether to include citation analysis
        include_references: Whether to include reference analysis
        include_embeddings: Whether to include semantic embeddings
        include_semantic_analysis: Whether to include advanced semantic analysis
        max_citations: Maximum citations to analyze
        max_references: Maximum references to analyze
        
    Returns:
        Dictionary with enrichment results
    """
    task_start = time.time()
    
    try:
        logger.info(f"Starting paper enrichment task for: {paper_identifier}")
        
        # Update task state
        self.update_state(
            state="PROGRESS",
            meta={"current": 0, "total": 100, "status": "Initializing enrichment"}
        )
        
        async def _enrich():
            # Get services
            semantic_service = await get_semantic_analysis_service()
            pipeline = await get_enrichment_pipeline()
            
            # Update progress
            self.update_state(
                state="PROGRESS",
                meta={"current": 20, "total": 100, "status": "Enriching with semantic features"}
            )
            
            # Perform enrichment
            enriched_paper = await semantic_service.enrich_paper_with_semantic_features(
                paper_identifier,
                use_cache=True
            )
            
            if not enriched_paper:
                raise APIError(f"Failed to enrich paper: {paper_identifier}")
            
            # Update progress
            self.update_state(
                state="PROGRESS",
                meta={"current": 60, "total": 100, "status": "Processing additional features"}
            )
            
            # Additional processing based on options
            if include_citations and enriched_paper.semantic_scholar_id:
                try:
                    client = await get_semantic_scholar_client()
                    citations = await client.get_paper_citations(
                        enriched_paper.semantic_scholar_id,
                        limit=max_citations,
                        use_cache=True
                    )
                    
                    # Store citation analysis
                    if not enriched_paper.citation_intent_analysis:
                        enriched_paper.citation_intent_analysis = {}
                    
                    enriched_paper.citation_intent_analysis["citations_processed"] = len(citations)
                    
                except Exception as e:
                    logger.warning(f"Citation analysis failed: {e}")
            
            # Update progress
            self.update_state(
                state="PROGRESS",
                meta={"current": 80, "total": 100, "status": "Storing results"}
            )
            
            # Store enriched paper
            await pipeline._store_enriched_paper(enriched_paper)
            
            return enriched_paper
        
        # Run async enrichment
        enriched_paper = run_async_task(_enrich())
        
        processing_time = time.time() - task_start
        
        result = {
            "status": "completed",
            "paper_identifier": paper_identifier,
            "enrichment_sources": enriched_paper.enrichment_sources,
            "has_semantic_features": enriched_paper.has_semantic_features(),
            "processing_time_seconds": processing_time,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Update final state
        self.update_state(
            state="SUCCESS",
            meta={"current": 100, "total": 100, "status": "Completed", "result": result}
        )
        
        logger.info(f"Paper enrichment completed in {processing_time:.2f}s: {paper_identifier}")
        return result
        
    except Exception as e:
        processing_time = time.time() - task_start
        error_msg = f"Paper enrichment failed: {str(e)}"
        
        logger.error(error_msg)
        
        # Update error state
        self.update_state(
            state="FAILURE",
            meta={"error": error_msg, "processing_time": processing_time}
        )
        
        return {
            "status": "failed",
            "paper_identifier": paper_identifier,
            "error": error_msg,
            "processing_time_seconds": processing_time,
            "timestamp": datetime.utcnow().isoformat()
        }


@celery_app.task(bind=True, name="semantic_scholar_tasks.enrich_papers_batch")
def enrich_papers_batch_task(
    self,
    paper_identifiers: List[str],
    priority: str = "medium",
    include_citations: bool = True,
    include_references: bool = True,
    include_embeddings: bool = True,
    include_semantic_analysis: bool = True,
    max_citations: int = 50,
    max_references: int = 50
) -> Dict[str, Any]:
    """
    Celery task for batch enrichment of multiple papers.
    
    Args:
        paper_identifiers: List of paper identifiers
        priority: Task priority level
        include_citations: Whether to include citation analysis
        include_references: Whether to include reference analysis
        include_embeddings: Whether to include semantic embeddings
        include_semantic_analysis: Whether to include advanced semantic analysis
        max_citations: Maximum citations to analyze per paper
        max_references: Maximum references to analyze per paper
        
    Returns:
        Dictionary with batch enrichment results
    """
    task_start = time.time()
    total_papers = len(paper_identifiers)
    
    try:
        logger.info(f"Starting batch enrichment for {total_papers} papers")
        
        # Update task state
        self.update_state(
            state="PROGRESS",
            meta={"current": 0, "total": total_papers, "status": "Starting batch enrichment"}
        )
        
        async def _batch_enrich():
            semantic_service = await get_semantic_analysis_service()
            pipeline = await get_enrichment_pipeline()
            
            results = []
            successful = 0
            failed = 0
            
            for i, paper_id in enumerate(paper_identifiers):
                try:
                    # Update progress
                    self.update_state(
                        state="PROGRESS",
                        meta={
                            "current": i,
                            "total": total_papers,
                            "status": f"Processing paper {i+1}/{total_papers}: {paper_id[:50]}..."
                        }
                    )
                    
                    # Enrich individual paper
                    enriched_paper = await semantic_service.enrich_paper_with_semantic_features(
                        paper_id,
                        use_cache=True
                    )
                    
                    if enriched_paper:
                        # Store enriched paper
                        await pipeline._store_enriched_paper(enriched_paper)
                        
                        results.append({
                            "paper_identifier": paper_id,
                            "status": "completed",
                            "enrichment_sources": enriched_paper.enrichment_sources,
                            "has_semantic_features": enriched_paper.has_semantic_features()
                        })
                        successful += 1
                    else:
                        results.append({
                            "paper_identifier": paper_id,
                            "status": "failed",
                            "error": "No data retrieved"
                        })
                        failed += 1
                
                except Exception as e:
                    logger.warning(f"Failed to enrich paper {paper_id}: {e}")
                    results.append({
                        "paper_identifier": paper_id,
                        "status": "failed",
                        "error": str(e)
                    })
                    failed += 1
                
                # Brief pause to avoid overwhelming APIs
                await asyncio.sleep(0.1)
            
            return results, successful, failed
        
        # Run batch enrichment
        results, successful, failed = run_async_task(_batch_enrich())
        
        processing_time = time.time() - task_start
        
        batch_result = {
            "status": "completed",
            "total_papers": total_papers,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total_papers if total_papers > 0 else 0,
            "processing_time_seconds": processing_time,
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Update final state
        self.update_state(
            state="SUCCESS",
            meta={
                "current": total_papers,
                "total": total_papers,
                "status": "Batch enrichment completed",
                "result": batch_result
            }
        )
        
        logger.info(
            f"Batch enrichment completed in {processing_time:.2f}s: "
            f"{successful}/{total_papers} successful"
        )
        
        return batch_result
        
    except Exception as e:
        processing_time = time.time() - task_start
        error_msg = f"Batch enrichment failed: {str(e)}"
        
        logger.error(error_msg)
        
        # Update error state
        self.update_state(
            state="FAILURE",
            meta={"error": error_msg, "processing_time": processing_time}
        )
        
        return {
            "status": "failed",
            "total_papers": total_papers,
            "error": error_msg,
            "processing_time_seconds": processing_time,
            "timestamp": datetime.utcnow().isoformat()
        }


@celery_app.task(bind=True, name="semantic_scholar_tasks.analyze_research_trajectory")
def analyze_research_trajectory_task(
    self,
    author_papers: List[str],
    time_window_years: int = 5
) -> Dict[str, Any]:
    """
    Celery task for analyzing research trajectory.
    
    Args:
        author_papers: List of author's paper identifiers
        time_window_years: Time window for analysis
        
    Returns:
        Research trajectory analysis results
    """
    task_start = time.time()
    
    try:
        logger.info(f"Starting research trajectory analysis for {len(author_papers)} papers")
        
        # Update task state
        self.update_state(
            state="PROGRESS",
            meta={"current": 0, "total": 100, "status": "Starting trajectory analysis"}
        )
        
        async def _analyze():
            semantic_service = await get_semantic_analysis_service()
            
            # Update progress
            self.update_state(
                state="PROGRESS",
                meta={"current": 30, "total": 100, "status": "Analyzing semantic evolution"}
            )
            
            # Perform analysis
            analysis = await semantic_service.analyze_research_trajectory(
                author_papers=author_papers,
                time_window_years=time_window_years,
                use_cache=True
            )
            
            return analysis
        
        # Run analysis
        analysis = run_async_task(_analyze())
        
        processing_time = time.time() - task_start
        
        result = {
            "status": "completed",
            "analysis": analysis,
            "processing_time_seconds": processing_time,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Update final state
        self.update_state(
            state="SUCCESS",
            meta={"current": 100, "total": 100, "status": "Analysis completed", "result": result}
        )
        
        logger.info(f"Research trajectory analysis completed in {processing_time:.2f}s")
        return result
        
    except Exception as e:
        processing_time = time.time() - task_start
        error_msg = f"Research trajectory analysis failed: {str(e)}"
        
        logger.error(error_msg)
        
        # Update error state
        self.update_state(
            state="FAILURE",
            meta={"error": error_msg, "processing_time": processing_time}
        )
        
        return {
            "status": "failed",
            "error": error_msg,
            "processing_time_seconds": processing_time,
            "timestamp": datetime.utcnow().isoformat()
        }


@celery_app.task(bind=True, name="semantic_scholar_tasks.identify_emerging_trends")
def identify_emerging_trends_task(
    self,
    field_of_study: str,
    time_window_months: int = 12,
    min_papers: int = 50
) -> Dict[str, Any]:
    """
    Celery task for identifying emerging research trends.
    
    Args:
        field_of_study: Research field to analyze
        time_window_months: Time window in months
        min_papers: Minimum papers required
        
    Returns:
        Emerging trends analysis results
    """
    task_start = time.time()
    
    try:
        logger.info(f"Starting trend analysis for field: {field_of_study}")
        
        # Update task state
        self.update_state(
            state="PROGRESS",
            meta={"current": 0, "total": 100, "status": "Starting trend analysis"}
        )
        
        async def _analyze():
            semantic_service = await get_semantic_analysis_service()
            
            # Update progress
            self.update_state(
                state="PROGRESS",
                meta={"current": 20, "total": 100, "status": "Collecting recent papers"}
            )
            
            # Perform analysis
            trends = await semantic_service.identify_emerging_research_trends(
                field_of_study=field_of_study,
                time_window_months=time_window_months,
                min_papers=min_papers,
                use_cache=True
            )
            
            return trends
        
        # Run analysis
        trends = run_async_task(_analyze())
        
        processing_time = time.time() - task_start
        
        result = {
            "status": "completed",
            "trends": trends,
            "processing_time_seconds": processing_time,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Update final state
        self.update_state(
            state="SUCCESS",
            meta={"current": 100, "total": 100, "status": "Trend analysis completed", "result": result}
        )
        
        logger.info(f"Trend analysis completed in {processing_time:.2f}s for field: {field_of_study}")
        return result
        
    except Exception as e:
        processing_time = time.time() - task_start
        error_msg = f"Trend analysis failed: {str(e)}"
        
        logger.error(error_msg)
        
        # Update error state
        self.update_state(
            state="FAILURE",
            meta={"error": error_msg, "processing_time": processing_time}
        )
        
        return {
            "status": "failed",
            "field_of_study": field_of_study,
            "error": error_msg,
            "processing_time_seconds": processing_time,
            "timestamp": datetime.utcnow().isoformat()
        }


@celery_app.task(bind=True, name="semantic_scholar_tasks.build_citation_network")
def build_citation_network_task(
    self,
    center_paper_id: str,
    max_depth: int = 2,
    max_papers_per_level: int = 20,
    similarity_threshold: float = 0.3
) -> Dict[str, Any]:
    """
    Celery task for building semantic citation network.
    
    Args:
        center_paper_id: Central paper identifier
        max_depth: Maximum network depth
        max_papers_per_level: Maximum papers per level
        similarity_threshold: Similarity threshold
        
    Returns:
        Citation network with semantic features
    """
    task_start = time.time()
    
    try:
        logger.info(f"Building semantic citation network for: {center_paper_id}")
        
        # Update task state
        self.update_state(
            state="PROGRESS",
            meta={"current": 0, "total": 100, "status": "Building citation network"}
        )
        
        async def _build():
            client = await get_semantic_scholar_client()
            
            # Update progress
            self.update_state(
                state="PROGRESS",
                meta={"current": 30, "total": 100, "status": "Collecting network nodes"}
            )
            
            # Build network
            network = await client.build_semantic_citation_network(
                center_paper_id=center_paper_id,
                max_depth=max_depth,
                max_papers_per_level=max_papers_per_level,
                similarity_threshold=similarity_threshold,
                use_cache=True
            )
            
            return network
        
        # Run network building
        network = run_async_task(_build())
        
        processing_time = time.time() - task_start
        
        result = {
            "status": "completed",
            "network": network.dict(),
            "processing_time_seconds": processing_time,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Update final state
        self.update_state(
            state="SUCCESS",
            meta={"current": 100, "total": 100, "status": "Network building completed", "result": result}
        )
        
        logger.info(f"Citation network built in {processing_time:.2f}s for: {center_paper_id}")
        return result
        
    except Exception as e:
        processing_time = time.time() - task_start
        error_msg = f"Citation network building failed: {str(e)}"
        
        logger.error(error_msg)
        
        # Update error state
        self.update_state(
            state="FAILURE",
            meta={"error": error_msg, "processing_time": processing_time}
        )
        
        return {
            "status": "failed",
            "center_paper_id": center_paper_id,
            "error": error_msg,
            "processing_time_seconds": processing_time,
            "timestamp": datetime.utcnow().isoformat()
        }


@celery_app.task(bind=True, name="semantic_scholar_tasks.process_enrichment_queue")
def process_enrichment_queue_task(self) -> Dict[str, Any]:
    """
    Celery task for processing enrichment queue.
    
    This task runs periodically to process queued enrichment tasks.
    """
    task_start = time.time()
    
    try:
        logger.info("Starting enrichment queue processing")
        
        async def _process():
            pipeline = await get_enrichment_pipeline()
            
            # Process queue for a limited time (10 minutes)
            end_time = time.time() + 600
            processed_tasks = 0
            
            while time.time() < end_time:
                try:
                    # Check if there are tasks to process
                    if pipeline.task_queue.empty():
                        await asyncio.sleep(5)  # Wait for new tasks
                        continue
                    
                    # Get and process task
                    task = await asyncio.wait_for(
                        pipeline.task_queue.get(),
                        timeout=1.0
                    )
                    
                    # Process task
                    await pipeline._process_enrichment_task(task)
                    processed_tasks += 1
                    
                    # Update progress periodically
                    if processed_tasks % 5 == 0:
                        self.update_state(
                            state="PROGRESS",
                            meta={
                                "processed_tasks": processed_tasks,
                                "queue_size": pipeline.task_queue.qsize(),
                                "status": f"Processed {processed_tasks} tasks"
                            }
                        )
                
                except asyncio.TimeoutError:
                    # No tasks available, continue
                    continue
                except Exception as e:
                    logger.warning(f"Error processing task: {e}")
                    continue
            
            return processed_tasks
        
        # Run queue processing
        processed_tasks = run_async_task(_process())
        
        processing_time = time.time() - task_start
        
        result = {
            "status": "completed",
            "processed_tasks": processed_tasks,
            "processing_time_seconds": processing_time,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(f"Queue processing completed: {processed_tasks} tasks in {processing_time:.2f}s")
        return result
        
    except Exception as e:
        processing_time = time.time() - task_start
        error_msg = f"Queue processing failed: {str(e)}"
        
        logger.error(error_msg)
        
        return {
            "status": "failed",
            "error": error_msg,
            "processing_time_seconds": processing_time,
            "timestamp": datetime.utcnow().isoformat()
        }


# Periodic tasks
@celery_app.task(bind=True)
def cleanup_expired_tasks(self):
    """Clean up expired enrichment tasks and cached data."""
    try:
        logger.info("Starting cleanup of expired tasks")
        
        async def _cleanup():
            redis_manager = await get_redis_manager()
            
            # Clean up expired enrichment tasks (older than 24 hours)
            pattern = "enrichment_task:*"
            keys = await redis_manager.scan_keys(pattern)
            
            cleaned_count = 0
            for key in keys:
                try:
                    task_data = await redis_manager.cache_get(key)
                    if task_data and "created_at" in task_data:
                        created_at = datetime.fromisoformat(task_data["created_at"])
                        if datetime.utcnow() - created_at > timedelta(days=1):
                            await redis_manager.delete_key(key)
                            cleaned_count += 1
                except Exception:
                    # Delete corrupted task data
                    await redis_manager.delete_key(key)
                    cleaned_count += 1
            
            return cleaned_count
        
        cleaned_count = run_async_task(_cleanup())
        
        logger.info(f"Cleanup completed: {cleaned_count} expired tasks removed")
        return {"status": "completed", "cleaned_tasks": cleaned_count}
        
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return {"status": "failed", "error": str(e)}


# Configure periodic tasks
celery_app.conf.beat_schedule = {
    'process-enrichment-queue': {
        'task': 'semantic_scholar_tasks.process_enrichment_queue',
        'schedule': 300.0,  # Every 5 minutes
    },
    'cleanup-expired-tasks': {
        'task': 'semantic_scholar_tasks.cleanup_expired_tasks',
        'schedule': 3600.0,  # Every hour
    },
}

celery_app.conf.timezone = 'UTC'