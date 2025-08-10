"""
Enrichment pipeline service for combining and processing multi-source research data.

Orchestrates the integration of OpenAlex and Semantic Scholar data with background
processing capabilities, incremental enrichment, and comprehensive error handling.
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
from collections import defaultdict

from ..models.semantic_scholar import (
    SemanticScholarPaper,
    EnrichedPaper,
    CitationIntent
)
from ..models.openalex import OpenAlexWork
from ..services.semantic_scholar import SemanticScholarClient, get_semantic_scholar_client
from ..services.openalex import OpenAlexClient, get_openalex_client
from ..services.semantic_analysis import SemanticAnalysisService, get_semantic_analysis_service
from ..db.redis import RedisManager, get_redis_manager
from ..db.neo4j import Neo4jManager, get_neo4j_manager
from ..utils.logger import get_logger
from ..utils.exceptions import ValidationError, APIError

logger = get_logger(__name__)


class EnrichmentStatus(str, Enum):
    """Status of enrichment process."""
    
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class EnrichmentPriority(str, Enum):
    """Priority levels for enrichment tasks."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class EnrichmentTask:
    """Individual enrichment task."""
    
    task_id: str
    paper_identifier: str
    priority: EnrichmentPriority = EnrichmentPriority.MEDIUM
    status: EnrichmentStatus = EnrichmentStatus.PENDING
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    # Enrichment options
    include_citations: bool = True
    include_references: bool = True
    include_embeddings: bool = True
    include_semantic_analysis: bool = True
    citation_depth: int = 1
    max_citations: int = 100
    max_references: int = 100
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnrichmentTask':
        """Create from dictionary."""
        # Handle datetime conversion
        for field in ['created_at', 'started_at', 'completed_at']:
            if data.get(field) and isinstance(data[field], str):
                data[field] = datetime.fromisoformat(data[field])
        
        return cls(**data)


@dataclass
class EnrichmentResult:
    """Result of enrichment process."""
    
    task_id: str
    status: EnrichmentStatus
    enriched_paper: Optional[EnrichedPaper] = None
    processing_time_seconds: float = 0
    sources_used: List[str] = None
    error_details: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.sources_used is None:
            self.sources_used = []
        if self.metadata is None:
            self.metadata = {}


class EnrichmentPipeline:
    """
    Comprehensive enrichment pipeline for research data integration.
    
    Provides orchestrated enrichment of papers using multiple data sources
    with background processing, caching, error handling, and monitoring.
    """
    
    def __init__(
        self,
        semantic_scholar_client: Optional[SemanticScholarClient] = None,
        openalex_client: Optional[OpenAlexClient] = None,
        semantic_analysis_service: Optional[SemanticAnalysisService] = None,
        redis_manager: Optional[RedisManager] = None,
        neo4j_manager: Optional[Neo4jManager] = None,
        max_concurrent_tasks: int = 10,
        batch_size: int = 20
    ):
        self.semantic_scholar_client = semantic_scholar_client
        self.openalex_client = openalex_client
        self.semantic_analysis_service = semantic_analysis_service
        self.redis_manager = redis_manager
        self.neo4j_manager = neo4j_manager
        self.max_concurrent_tasks = max_concurrent_tasks
        self.batch_size = batch_size
        
        # Task management
        self.active_tasks: Dict[str, EnrichmentTask] = {}
        self.task_semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self.task_queue = asyncio.Queue()
        
        # Metrics
        self.total_tasks_processed = 0
        self.successful_enrichments = 0
        self.failed_enrichments = 0
        self.partial_enrichments = 0
        self.total_processing_time = 0
        
        self._clients_initialized = False
    
    async def _ensure_clients(self):
        """Ensure all clients are initialized."""
        if not self._clients_initialized:
            if not self.semantic_scholar_client:
                self.semantic_scholar_client = await get_semantic_scholar_client(self.redis_manager)
            if not self.openalex_client:
                self.openalex_client = await get_openalex_client(self.redis_manager)
            if not self.semantic_analysis_service:
                self.semantic_analysis_service = await get_semantic_analysis_service()
            if not self.redis_manager:
                self.redis_manager = await get_redis_manager()
            if not self.neo4j_manager:
                self.neo4j_manager = await get_neo4j_manager()
            
            self._clients_initialized = True
    
    async def enrich_paper(
        self,
        paper_identifier: str,
        priority: EnrichmentPriority = EnrichmentPriority.MEDIUM,
        **enrichment_options
    ) -> str:
        """
        Submit a paper for enrichment.
        
        Args:
            paper_identifier: Paper DOI, OpenAlex ID, or Semantic Scholar ID
            priority: Task priority
            **enrichment_options: Additional enrichment options
            
        Returns:
            Task ID for tracking
        """
        task_id = f"enrich_{int(time.time() * 1000)}_{hash(paper_identifier) % 10000}"
        
        task = EnrichmentTask(
            task_id=task_id,
            paper_identifier=paper_identifier,
            priority=priority,
            **enrichment_options
        )
        
        # Store task
        await self._store_task(task)
        
        # Queue for processing
        await self.task_queue.put(task)
        
        logger.info(f"Queued enrichment task {task_id} for {paper_identifier}")
        return task_id
    
    async def enrich_papers_batch(
        self,
        paper_identifiers: List[str],
        priority: EnrichmentPriority = EnrichmentPriority.MEDIUM,
        **enrichment_options
    ) -> List[str]:
        """
        Submit multiple papers for batch enrichment.
        
        Args:
            paper_identifiers: List of paper identifiers
            priority: Task priority
            **enrichment_options: Additional enrichment options
            
        Returns:
            List of task IDs
        """
        if len(paper_identifiers) > 100:
            raise ValidationError("Maximum 100 papers allowed per batch")
        
        task_ids = []
        
        for paper_identifier in paper_identifiers:
            task_id = await self.enrich_paper(
                paper_identifier,
                priority=priority,
                **enrichment_options
            )
            task_ids.append(task_id)
        
        logger.info(f"Queued {len(task_ids)} papers for batch enrichment")
        return task_ids
    
    async def process_enrichment_queue(self):
        """
        Process enrichment queue with concurrent task execution.
        
        This method should be run as a background task.
        """
        await self._ensure_clients()
        logger.info("Starting enrichment queue processor")
        
        while True:
            try:
                # Get task from queue (with timeout to allow graceful shutdown)
                task = await asyncio.wait_for(
                    self.task_queue.get(),
                    timeout=1.0
                )
                
                # Process task concurrently
                asyncio.create_task(self._process_task_with_semaphore(task))
                
            except asyncio.TimeoutError:
                # No task available, continue loop
                continue
            except Exception as e:
                logger.error(f"Error in queue processor: {e}")
                await asyncio.sleep(5)  # Brief pause before retry
    
    async def _process_task_with_semaphore(self, task: EnrichmentTask):
        """Process task with semaphore for concurrency control."""
        async with self.task_semaphore:
            await self._process_enrichment_task(task)
    
    async def _process_enrichment_task(self, task: EnrichmentTask) -> EnrichmentResult:
        """
        Process individual enrichment task.
        
        Args:
            task: Enrichment task to process
            
        Returns:
            Enrichment result
        """
        start_time = time.time()
        
        # Update task status
        task.status = EnrichmentStatus.IN_PROGRESS
        task.started_at = datetime.utcnow()
        await self._store_task(task)
        
        self.active_tasks[task.task_id] = task
        
        try:
            logger.info(f"Processing enrichment task {task.task_id}")
            
            # Perform enrichment
            enriched_paper = await self._perform_enrichment(task)
            
            # Store results
            if enriched_paper:
                await self._store_enriched_paper(enriched_paper)
                
                # Update Neo4j if configured
                if self.neo4j_manager:
                    await self._update_neo4j_graph(enriched_paper)
            
            # Determine final status
            status = self._determine_enrichment_status(enriched_paper, task)
            
            # Update task completion
            task.status = status
            task.completed_at = datetime.utcnow()
            await self._store_task(task)
            
            processing_time = time.time() - start_time
            
            result = EnrichmentResult(
                task_id=task.task_id,
                status=status,
                enriched_paper=enriched_paper,
                processing_time_seconds=processing_time,
                sources_used=enriched_paper.enrichment_sources if enriched_paper else [],
                metadata={
                    "task_options": {
                        "include_citations": task.include_citations,
                        "include_references": task.include_references,
                        "include_embeddings": task.include_embeddings,
                        "include_semantic_analysis": task.include_semantic_analysis
                    }
                }
            )
            
            # Update metrics
            self.total_tasks_processed += 1
            self.total_processing_time += processing_time
            
            if status == EnrichmentStatus.COMPLETED:
                self.successful_enrichments += 1
            elif status == EnrichmentStatus.PARTIAL:
                self.partial_enrichments += 1
            else:
                self.failed_enrichments += 1
            
            logger.info(
                f"Completed enrichment task {task.task_id} in {processing_time:.2f}s "
                f"with status {status}"
            )
            
            return result
            
        except Exception as e:
            # Handle task failure
            task.status = EnrichmentStatus.FAILED
            task.completed_at = datetime.utcnow()
            task.error_message = str(e)
            task.retry_count += 1
            
            await self._store_task(task)
            
            processing_time = time.time() - start_time
            self.failed_enrichments += 1
            self.total_processing_time += processing_time
            
            logger.error(f"Enrichment task {task.task_id} failed: {e}")
            
            # Retry if under limit
            if task.retry_count < task.max_retries:
                logger.info(f"Retrying task {task.task_id} (attempt {task.retry_count + 1})")
                # Reset status and requeue
                task.status = EnrichmentStatus.PENDING
                task.started_at = None
                await self.task_queue.put(task)
            
            return EnrichmentResult(
                task_id=task.task_id,
                status=EnrichmentStatus.FAILED,
                processing_time_seconds=processing_time,
                error_details={"error": str(e), "retry_count": task.retry_count}
            )
        
        finally:
            # Clean up active tasks
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
    
    async def _perform_enrichment(self, task: EnrichmentTask) -> Optional[EnrichedPaper]:
        """Perform the actual enrichment process."""
        # Use semantic analysis service for comprehensive enrichment
        enriched_paper = await self.semantic_analysis_service.enrich_paper_with_semantic_features(
            task.paper_identifier,
            use_cache=True
        )
        
        if not enriched_paper:
            return None
        
        # Additional enrichments based on task options
        if task.include_citations and enriched_paper.semantic_scholar_id:
            await self._enrich_with_citations(enriched_paper, task)
        
        if task.include_references and enriched_paper.semantic_scholar_id:
            await self._enrich_with_references(enriched_paper, task)
        
        if task.include_embeddings and enriched_paper.semantic_scholar_id:
            await self._enrich_with_embeddings(enriched_paper)
        
        return enriched_paper
    
    async def _enrich_with_citations(self, enriched_paper: EnrichedPaper, task: EnrichmentTask):
        """Enrich paper with detailed citation information."""
        if not enriched_paper.semantic_scholar_id:
            return
        
        try:
            citations = await self.semantic_scholar_client.get_paper_citations(
                enriched_paper.semantic_scholar_id,
                limit=task.max_citations,
                use_cache=True
            )
            
            # Analyze citation patterns
            citation_analysis = {
                "total_citations": len(citations),
                "citing_years": [c.year for c in citations if c.year],
                "citation_contexts": []
            }
            
            # Extract citation contexts and intents
            for citation in citations:
                if hasattr(citation, 'contexts') and citation.contexts:
                    citation_analysis["citation_contexts"].extend(citation.contexts)
            
            # Store citation analysis
            if not enriched_paper.citation_intent_analysis:
                enriched_paper.citation_intent_analysis = {}
            
            enriched_paper.citation_intent_analysis.update(citation_analysis)
            
        except Exception as e:
            logger.warning(f"Failed to enrich with citations: {e}")
    
    async def _enrich_with_references(self, enriched_paper: EnrichedPaper, task: EnrichmentTask):
        """Enrich paper with detailed reference information."""
        if not enriched_paper.semantic_scholar_id:
            return
        
        try:
            references = await self.semantic_scholar_client.get_paper_references(
                enriched_paper.semantic_scholar_id,
                limit=task.max_references,
                use_cache=True
            )
            
            # Analyze reference patterns
            reference_analysis = {
                "total_references": len(references),
                "reference_years": [r.year for r in references if r.year],
                "reference_fields": []
            }
            
            # Extract fields from references
            for reference in references:
                if reference.fields_of_study:
                    reference_analysis["reference_fields"].extend(reference.fields_of_study)
            
            # Store reference analysis
            if not enriched_paper.citation_intent_analysis:
                enriched_paper.citation_intent_analysis = {}
            
            enriched_paper.citation_intent_analysis["reference_analysis"] = reference_analysis
            
        except Exception as e:
            logger.warning(f"Failed to enrich with references: {e}")
    
    async def _enrich_with_embeddings(self, enriched_paper: EnrichedPaper):
        """Enrich paper with semantic embeddings."""
        if not enriched_paper.semantic_scholar_id:
            return
        
        try:
            embedding = await self.semantic_scholar_client.get_paper_embedding(
                enriched_paper.semantic_scholar_id,
                use_cache=True
            )
            
            if embedding and enriched_paper.semantic_scholar_data:
                enriched_paper.semantic_scholar_data.embedding = embedding
            
        except Exception as e:
            logger.warning(f"Failed to enrich with embeddings: {e}")
    
    def _determine_enrichment_status(
        self, 
        enriched_paper: Optional[EnrichedPaper], 
        task: EnrichmentTask
    ) -> EnrichmentStatus:
        """Determine the final status of enrichment."""
        if not enriched_paper:
            return EnrichmentStatus.FAILED
        
        # Check if we have data from at least one source
        if not enriched_paper.enrichment_sources:
            return EnrichmentStatus.FAILED
        
        # Check if we have both sources when expected
        has_semantic_scholar = "semantic_scholar" in enriched_paper.enrichment_sources
        has_openalex = "openalex" in enriched_paper.enrichment_sources
        
        # If we have semantic features (the main goal), consider it successful
        if enriched_paper.has_semantic_features():
            return EnrichmentStatus.COMPLETED
        
        # If we have some data but missing semantic features, it's partial
        if has_semantic_scholar or has_openalex:
            return EnrichmentStatus.PARTIAL
        
        return EnrichmentStatus.FAILED
    
    async def _store_task(self, task: EnrichmentTask):
        """Store task state in Redis."""
        if not self.redis_manager:
            return
        
        try:
            cache_key = f"enrichment_task:{task.task_id}"
            await self.redis_manager.cache_set(
                cache_key,
                task.to_dict(),
                ttl=86400  # Store for 24 hours
            )
        except Exception as e:
            logger.warning(f"Failed to store task {task.task_id}: {e}")
    
    async def _store_enriched_paper(self, enriched_paper: EnrichedPaper):
        """Store enriched paper in Redis cache."""
        if not self.redis_manager:
            return
        
        try:
            # Store by different identifiers for flexible retrieval
            paper_data = enriched_paper.dict()
            
            cache_keys = []
            if enriched_paper.doi:
                cache_keys.append(f"enriched_paper:doi:{enriched_paper.doi}")
            if enriched_paper.openalex_id:
                cache_key = enriched_paper.openalex_id.replace("https://openalex.org/", "")
                cache_keys.append(f"enriched_paper:openalex:{cache_key}")
            if enriched_paper.semantic_scholar_id:
                cache_keys.append(f"enriched_paper:s2:{enriched_paper.semantic_scholar_id}")
            
            # Store under all applicable keys
            for cache_key in cache_keys:
                await self.redis_manager.cache_set(
                    cache_key,
                    paper_data,
                    ttl=7200  # Store for 2 hours
                )
                
        except Exception as e:
            logger.warning(f"Failed to store enriched paper: {e}")
    
    async def _update_neo4j_graph(self, enriched_paper: EnrichedPaper):
        """Update Neo4j graph with enriched paper data."""
        if not self.neo4j_manager:
            return
        
        try:
            # Create or update paper node
            paper_properties = {
                "title": enriched_paper.title,
                "doi": enriched_paper.doi,
                "openalex_id": enriched_paper.openalex_id,
                "semantic_scholar_id": enriched_paper.semantic_scholar_id,
                "enrichment_timestamp": enriched_paper.enrichment_timestamp.isoformat(),
                "has_semantic_features": enriched_paper.has_semantic_features()
            }
            
            # Add citation metrics
            if enriched_paper.semantic_scholar_data:
                paper_properties.update({
                    "citation_count": enriched_paper.semantic_scholar_data.citation_count,
                    "influential_citation_count": enriched_paper.semantic_scholar_data.influential_citation_count,
                    "publication_year": enriched_paper.semantic_scholar_data.year
                })
            
            # Create/update paper node
            query = """
            MERGE (p:Paper {id: $paper_id})
            SET p += $properties
            SET p.updated_at = datetime()
            """
            
            paper_id = enriched_paper.doi or enriched_paper.openalex_id or enriched_paper.semantic_scholar_id
            
            await self.neo4j_manager.execute_write(
                query,
                paper_id=paper_id,
                properties=paper_properties
            )
            
        except Exception as e:
            logger.warning(f"Failed to update Neo4j graph: {e}")
    
    async def get_task_status(self, task_id: str) -> Optional[EnrichmentTask]:
        """Get current status of an enrichment task."""
        if not self.redis_manager:
            return None
        
        try:
            cache_key = f"enrichment_task:{task_id}"
            task_data = await self.redis_manager.cache_get(cache_key)
            
            if task_data:
                return EnrichmentTask.from_dict(task_data)
            
        except Exception as e:
            logger.warning(f"Failed to get task status for {task_id}: {e}")
        
        return None
    
    async def get_enriched_paper(
        self,
        paper_identifier: str,
        identifier_type: str = "auto"
    ) -> Optional[EnrichedPaper]:
        """
        Retrieve enriched paper from cache.
        
        Args:
            paper_identifier: Paper identifier
            identifier_type: Type of identifier ("auto", "doi", "openalex", "s2")
            
        Returns:
            Enriched paper if available
        """
        if not self.redis_manager:
            return None
        
        try:
            cache_key = None
            
            if identifier_type == "doi" or (identifier_type == "auto" and paper_identifier.startswith("10.")):
                cache_key = f"enriched_paper:doi:{paper_identifier}"
            elif identifier_type == "openalex" or (identifier_type == "auto" and "openalex.org" in paper_identifier):
                clean_id = paper_identifier.replace("https://openalex.org/", "")
                cache_key = f"enriched_paper:openalex:{clean_id}"
            elif identifier_type == "s2":
                cache_key = f"enriched_paper:s2:{paper_identifier}"
            else:
                # Try all cache patterns
                for prefix in ["doi", "openalex", "s2"]:
                    test_key = f"enriched_paper:{prefix}:{paper_identifier}"
                    result = await self.redis_manager.cache_get(test_key)
                    if result:
                        return EnrichedPaper(**result)
                return None
            
            if cache_key:
                paper_data = await self.redis_manager.cache_get(cache_key)
                if paper_data:
                    return EnrichedPaper(**paper_data)
            
        except Exception as e:
            logger.warning(f"Failed to get enriched paper {paper_identifier}: {e}")
        
        return None
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status and metrics."""
        return {
            "queue_size": self.task_queue.qsize(),
            "active_tasks": len(self.active_tasks),
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "metrics": {
                "total_tasks_processed": self.total_tasks_processed,
                "successful_enrichments": self.successful_enrichments,
                "failed_enrichments": self.failed_enrichments,
                "partial_enrichments": self.partial_enrichments,
                "avg_processing_time": (
                    self.total_processing_time / max(self.total_tasks_processed, 1)
                ),
                "success_rate": (
                    self.successful_enrichments / max(self.total_tasks_processed, 1)
                )
            },
            "active_task_ids": list(self.active_tasks.keys())
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on enrichment pipeline."""
        await self._ensure_clients()
        
        health_status = {
            "status": "healthy",
            "clients_initialized": self._clients_initialized,
            "queue_status": await self.get_queue_status()
        }
        
        # Check client health
        if self.semantic_scholar_client:
            try:
                s2_health = await self.semantic_scholar_client.health_check()
                health_status["semantic_scholar"] = s2_health
            except Exception as e:
                health_status["semantic_scholar"] = {"status": "unhealthy", "error": str(e)}
        
        if self.openalex_client:
            try:
                oa_health = await self.openalex_client.health_check()
                health_status["openalex"] = oa_health
            except Exception as e:
                health_status["openalex"] = {"status": "unhealthy", "error": str(e)}
        
        # Check if any critical component is unhealthy
        if (health_status.get("semantic_scholar", {}).get("status") == "unhealthy" or
            health_status.get("openalex", {}).get("status") == "unhealthy"):
            health_status["status"] = "degraded"
        
        return health_status


# Global pipeline instance
_enrichment_pipeline: Optional[EnrichmentPipeline] = None


async def get_enrichment_pipeline() -> EnrichmentPipeline:
    """Get or create enrichment pipeline instance."""
    global _enrichment_pipeline
    
    if _enrichment_pipeline is None:
        _enrichment_pipeline = EnrichmentPipeline()
    
    return _enrichment_pipeline