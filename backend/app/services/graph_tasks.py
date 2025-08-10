"""
Background graph processing tasks using Celery.
Handles long-running graph algorithms, community detection, and metrics calculation.
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import asdict

from celery import Celery
from celery.result import AsyncResult
from celery.exceptions import Retry

from ..core.config import get_settings
from ..db.neo4j_advanced import (
    AdvancedNeo4jManager, 
    GraphProjection,
    GraphAlgorithm,
    CommunityAlgorithm,
    CentralityMetric
)
from ..services.graph_operations import GraphCRUDOperations
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Initialize Celery app
settings = get_settings()
celery_app = Celery(
    "graph_tasks",
    broker=settings.celery.broker_url,
    backend=settings.celery.result_backend
)

celery_app.conf.update(
    task_serializer=settings.celery.task_serializer,
    result_serializer=settings.celery.result_serializer,
    accept_content=settings.celery.accept_content,
    timezone=settings.celery.timezone,
    enable_utc=settings.celery.enable_utc,
    task_routes={
        "graph_tasks.*": {"queue": "graph_processing"},
        "quick_tasks.*": {"queue": "quick_processing"},
        "heavy_tasks.*": {"queue": "heavy_processing"}
    },
    task_time_limit=7200,  # 2 hours max
    task_soft_time_limit=6600,  # 1 hour 50 minutes soft limit
)


class GraphTaskManager:
    """Manager for graph processing tasks."""
    
    def __init__(self):
        self.neo4j_manager = AdvancedNeo4jManager()
        self.crud_ops = GraphCRUDOperations(self.neo4j_manager)
        
    async def ensure_connection(self):
        """Ensure Neo4j connection is established."""
        if not self.neo4j_manager.driver:
            await self.neo4j_manager.connect()


# Global task manager instance
task_manager = GraphTaskManager()


def run_async_task(coro):
    """Helper to run async coroutines in Celery tasks."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ==================== GRAPH PROJECTION TASKS ====================

@celery_app.task(bind=True, name="graph_tasks.create_citation_projection")
def create_citation_projection_task(
    self,
    projection_name: str,
    node_labels: List[str] = None,
    relationship_types: List[str] = None,
    node_properties: Dict = None,
    orientation: str = "NATURAL"
):
    """Create a graph projection for citation network analysis."""
    try:
        if node_labels is None:
            node_labels = ["Paper"]
        if relationship_types is None:
            relationship_types = ["CITES"]
        
        projection = GraphProjection(
            name=projection_name,
            node_labels=node_labels,
            relationship_types=relationship_types,
            node_properties=node_properties or {},
            orientation=orientation
        )
        
        async def _create_projection():
            await task_manager.ensure_connection()
            success = await task_manager.neo4j_manager.create_graph_projection(projection)
            if success:
                # Get projection statistics
                projections = await task_manager.neo4j_manager.list_graph_projections()
                proj_info = next((p for p in projections if p["graphName"] == projection_name), {})
                return {
                    "projection_name": projection_name,
                    "success": True,
                    "node_count": proj_info.get("nodeCount", 0),
                    "relationship_count": proj_info.get("relationshipCount", 0),
                    "memory_usage": proj_info.get("memoryUsage", "0 Bytes")
                }
            else:
                return {"projection_name": projection_name, "success": False, "error": "Failed to create projection"}
        
        return run_async_task(_create_projection())
        
    except Exception as e:
        logger.error(f"Graph projection task failed: {e}")
        raise self.retry(countdown=60, max_retries=3)


@celery_app.task(bind=True, name="graph_tasks.cleanup_projections")
def cleanup_projections_task(self, max_age_hours: int = 24):
    """Clean up old graph projections."""
    try:
        async def _cleanup():
            await task_manager.ensure_connection()
            projections = await task_manager.neo4j_manager.list_graph_projections()
            
            cleaned_count = 0
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            for projection in projections:
                # Check if projection is old (this would need to be tracked separately)
                # For now, just clean up projections with specific naming patterns
                graph_name = projection["graphName"]
                if graph_name.startswith("temp_") or "analysis_" in graph_name:
                    await task_manager.neo4j_manager.drop_graph_projection(graph_name)
                    cleaned_count += 1
            
            return {"cleaned_projections": cleaned_count}
        
        return run_async_task(_cleanup())
        
    except Exception as e:
        logger.error(f"Projection cleanup task failed: {e}")
        return {"cleaned_projections": 0, "error": str(e)}


# ==================== CENTRALITY CALCULATION TASKS ====================

@celery_app.task(bind=True, name="graph_tasks.calculate_pagerank")
def calculate_pagerank_task(
    self,
    graph_name: str,
    max_iterations: int = 20,
    damping_factor: float = 0.85,
    tolerance: float = 1e-6,
    write_property: str = None
):
    """Calculate PageRank centrality scores."""
    try:
        async def _calculate():
            await task_manager.ensure_connection()
            result = await task_manager.neo4j_manager.calculate_pagerank(
                graph_name=graph_name,
                max_iterations=max_iterations,
                damping_factor=damping_factor,
                tolerance=tolerance,
                write_property=write_property
            )
            
            return {
                "algorithm": result.algorithm,
                "execution_time": result.execution_time,
                "top_papers": result.top_nodes[:10],  # Top 10 most influential
                "statistics": result.statistics,
                "total_papers": len(result.scores) if result.scores else 0
            }
        
        return run_async_task(_calculate())
        
    except Exception as e:
        logger.error(f"PageRank calculation failed: {e}")
        raise self.retry(countdown=120, max_retries=2)


@celery_app.task(bind=True, name="graph_tasks.calculate_all_centralities")
def calculate_all_centralities_task(
    self,
    graph_name: str,
    write_properties: bool = True
):
    """Calculate all centrality measures for comprehensive analysis."""
    try:
        async def _calculate():
            await task_manager.ensure_connection()
            
            results = {}
            
            # PageRank
            try:
                pagerank_result = await task_manager.neo4j_manager.calculate_pagerank(
                    graph_name=graph_name,
                    write_property="pagerank_score" if write_properties else None
                )
                results["pagerank"] = {
                    "execution_time": pagerank_result.execution_time,
                    "top_nodes": pagerank_result.top_nodes[:10],
                    "statistics": pagerank_result.statistics
                }
            except Exception as e:
                logger.error(f"PageRank failed: {e}")
                results["pagerank"] = {"error": str(e)}
            
            # Betweenness Centrality
            try:
                betweenness_result = await task_manager.neo4j_manager.calculate_betweenness_centrality(
                    graph_name=graph_name,
                    write_property="betweenness_score" if write_properties else None
                )
                results["betweenness"] = {
                    "execution_time": betweenness_result.execution_time,
                    "top_nodes": betweenness_result.top_nodes[:10],
                    "statistics": betweenness_result.statistics
                }
            except Exception as e:
                logger.error(f"Betweenness failed: {e}")
                results["betweenness"] = {"error": str(e)}
            
            # Closeness Centrality
            try:
                closeness_result = await task_manager.neo4j_manager.calculate_closeness_centrality(
                    graph_name=graph_name,
                    write_property="closeness_score" if write_properties else None
                )
                results["closeness"] = {
                    "execution_time": closeness_result.execution_time,
                    "top_nodes": closeness_result.top_nodes[:10],
                    "statistics": closeness_result.statistics
                }
            except Exception as e:
                logger.error(f"Closeness failed: {e}")
                results["closeness"] = {"error": str(e)}
            
            return results
        
        return run_async_task(_calculate())
        
    except Exception as e:
        logger.error(f"Centrality calculations failed: {e}")
        raise self.retry(countdown=180, max_retries=2)


# ==================== COMMUNITY DETECTION TASKS ====================

@celery_app.task(bind=True, name="graph_tasks.detect_communities_louvain")
def detect_communities_louvain_task(
    self,
    graph_name: str,
    max_iterations: int = 10,
    tolerance: float = 1e-6,
    write_property: str = None
):
    """Detect communities using Louvain algorithm."""
    try:
        async def _detect():
            await task_manager.ensure_connection()
            result = await task_manager.neo4j_manager.detect_communities_louvain(
                graph_name=graph_name,
                max_iterations=max_iterations,
                tolerance=tolerance,
                write_property=write_property
            )
            
            # Analyze community sizes
            community_sizes = {
                community_id: len(members) 
                for community_id, members in result.communities.items()
            }
            
            return {
                "algorithm": result.algorithm,
                "community_count": result.community_count,
                "modularity": result.modularity,
                "execution_time": result.execution_time,
                "community_sizes": community_sizes,
                "largest_community_size": max(community_sizes.values()) if community_sizes else 0,
                "average_community_size": sum(community_sizes.values()) / len(community_sizes) if community_sizes else 0
            }
        
        return run_async_task(_detect())
        
    except Exception as e:
        logger.error(f"Louvain community detection failed: {e}")
        raise self.retry(countdown=120, max_retries=2)


@celery_app.task(bind=True, name="graph_tasks.detect_communities_leiden")
def detect_communities_leiden_task(
    self,
    graph_name: str,
    max_iterations: int = 10,
    gamma: float = 1.0,
    theta: float = 0.01,
    write_property: str = None
):
    """Detect communities using Leiden algorithm."""
    try:
        async def _detect():
            await task_manager.ensure_connection()
            result = await task_manager.neo4j_manager.detect_communities_leiden(
                graph_name=graph_name,
                max_iterations=max_iterations,
                gamma=gamma,
                theta=theta,
                write_property=write_property
            )
            
            community_sizes = {
                community_id: len(members) 
                for community_id, members in result.communities.items()
            }
            
            return {
                "algorithm": result.algorithm,
                "community_count": result.community_count,
                "modularity": result.modularity,
                "execution_time": result.execution_time,
                "community_sizes": community_sizes,
                "largest_community_size": max(community_sizes.values()) if community_sizes else 0,
                "average_community_size": sum(community_sizes.values()) / len(community_sizes) if community_sizes else 0
            }
        
        return run_async_task(_detect())
        
    except Exception as e:
        logger.error(f"Leiden community detection failed: {e}")
        raise self.retry(countdown=120, max_retries=2)


# ==================== LINEAGE ANALYSIS TASKS ====================

@celery_app.task(bind=True, name="graph_tasks.trace_research_lineage")
def trace_research_lineage_task(
    self,
    paper_id: str,
    direction: str = "backward",
    max_depth: int = 5
):
    """Trace intellectual lineage of a paper."""
    try:
        async def _trace():
            await task_manager.ensure_connection()
            lineage = await task_manager.neo4j_manager.find_research_lineage(
                paper_id=paper_id,
                direction=direction,
                max_depth=max_depth
            )
            
            # Add additional analysis
            lineage_analysis = {
                "paper_id": paper_id,
                "direction": direction,
                "max_depth": max_depth,
                "lineage": lineage,
                "analysis": {
                    "total_papers": lineage.get("total_lineage_papers", 0),
                    "backward_count": len(lineage.get("backward_lineage", [])),
                    "forward_count": len(lineage.get("forward_lineage", [])),
                    "lineage_depth": lineage.get("lineage_depth", 0)
                }
            }
            
            # Calculate influence metrics
            if direction in ["backward", "both"] and lineage.get("backward_lineage"):
                backward_years = [p["year"] for p in lineage["backward_lineage"] if p.get("year")]
                if backward_years:
                    lineage_analysis["analysis"]["earliest_influence"] = min(backward_years)
                    lineage_analysis["analysis"]["latest_influence"] = max(backward_years)
            
            return lineage_analysis
        
        return run_async_task(_trace())
        
    except Exception as e:
        logger.error(f"Research lineage tracing failed: {e}")
        raise self.retry(countdown=60, max_retries=3)


@celery_app.task(bind=True, name="graph_tasks.analyze_citation_influence")
def analyze_citation_influence_task(
    self,
    paper_id: str,
    max_depth: int = 3
):
    """Analyze citation influence patterns for a paper."""
    try:
        async def _analyze():
            await task_manager.ensure_connection()
            
            # First create a temporary projection for this analysis
            projection_name = f"influence_analysis_{paper_id}_{int(time.time())}"
            projection = GraphProjection(
                name=projection_name,
                node_labels=["Paper"],
                relationship_types=["CITES"],
                orientation="NATURAL"
            )
            
            try:
                # Create projection
                await task_manager.neo4j_manager.create_graph_projection(projection)
                
                # Analyze influence
                influence = await task_manager.neo4j_manager.analyze_citation_influence(
                    graph_name=projection_name,
                    paper_id=paper_id,
                    max_depth=max_depth
                )
                
                return influence
                
            finally:
                # Clean up temporary projection
                await task_manager.neo4j_manager.drop_graph_projection(projection_name)
        
        return run_async_task(_analyze())
        
    except Exception as e:
        logger.error(f"Citation influence analysis failed: {e}")
        raise self.retry(countdown=60, max_retries=3)


# ==================== GRAPH METRICS TASKS ====================

@celery_app.task(bind=True, name="graph_tasks.calculate_comprehensive_metrics")
def calculate_comprehensive_metrics_task(
    self,
    graph_name: str,
    include_centrality: bool = True,
    include_communities: bool = True
):
    """Calculate comprehensive graph metrics."""
    try:
        async def _calculate():
            await task_manager.ensure_connection()
            
            results = {
                "graph_name": graph_name,
                "timestamp": datetime.utcnow().isoformat(),
                "basic_metrics": {},
                "centrality_metrics": {},
                "community_metrics": {},
                "path_metrics": {}
            }
            
            # Basic graph metrics
            try:
                basic_metrics = await task_manager.neo4j_manager.calculate_graph_metrics(graph_name)
                results["basic_metrics"] = basic_metrics
            except Exception as e:
                logger.error(f"Basic metrics calculation failed: {e}")
                results["basic_metrics"]["error"] = str(e)
            
            # Centrality metrics
            if include_centrality:
                try:
                    # Calculate PageRank for influence scoring
                    pagerank_result = await task_manager.neo4j_manager.calculate_pagerank(
                        graph_name=graph_name
                    )
                    results["centrality_metrics"]["pagerank"] = {
                        "execution_time": pagerank_result.execution_time,
                        "statistics": pagerank_result.statistics,
                        "top_influential": pagerank_result.top_nodes[:5]
                    }
                except Exception as e:
                    logger.error(f"Centrality metrics calculation failed: {e}")
                    results["centrality_metrics"]["error"] = str(e)
            
            # Community metrics
            if include_communities:
                try:
                    louvain_result = await task_manager.neo4j_manager.detect_communities_louvain(
                        graph_name=graph_name
                    )
                    results["community_metrics"] = {
                        "algorithm": louvain_result.algorithm,
                        "community_count": louvain_result.community_count,
                        "modularity": louvain_result.modularity,
                        "execution_time": louvain_result.execution_time
                    }
                except Exception as e:
                    logger.error(f"Community metrics calculation failed: {e}")
                    results["community_metrics"]["error"] = str(e)
            
            return results
        
        return run_async_task(_calculate())
        
    except Exception as e:
        logger.error(f"Comprehensive metrics calculation failed: {e}")
        raise self.retry(countdown=180, max_retries=2)


# ==================== DATA MIGRATION TASKS ====================

@celery_app.task(bind=True, name="graph_tasks.migrate_papers_batch")
def migrate_papers_batch_task(
    self,
    papers_data: List[Dict],
    batch_size: int = 100
):
    """Migrate papers data in batches."""
    try:
        async def _migrate():
            await task_manager.ensure_connection()
            
            migrated_count = 0
            failed_count = 0
            
            # Process in batches
            for i in range(0, len(papers_data), batch_size):
                batch = papers_data[i:i + batch_size]
                
                try:
                    created_ids = await task_manager.crud_ops.batch_create_papers(batch)
                    migrated_count += len(created_ids)
                except Exception as e:
                    logger.error(f"Batch migration failed for batch {i//batch_size + 1}: {e}")
                    failed_count += len(batch)
            
            return {
                "total_papers": len(papers_data),
                "migrated_count": migrated_count,
                "failed_count": failed_count,
                "success_rate": migrated_count / len(papers_data) if papers_data else 0
            }
        
        return run_async_task(_migrate())
        
    except Exception as e:
        logger.error(f"Papers batch migration failed: {e}")
        raise self.retry(countdown=60, max_retries=3)


@celery_app.task(bind=True, name="graph_tasks.migrate_citations_batch")
def migrate_citations_batch_task(
    self,
    citations_data: List[Tuple[str, str]],
    batch_size: int = 500
):
    """Migrate citation relationships in batches."""
    try:
        async def _migrate():
            await task_manager.ensure_connection()
            
            migrated_count = 0
            
            # Process in batches
            for i in range(0, len(citations_data), batch_size):
                batch = citations_data[i:i + batch_size]
                
                try:
                    created_count = await task_manager.crud_ops.batch_create_citations(batch)
                    migrated_count += created_count
                except Exception as e:
                    logger.error(f"Citations batch migration failed for batch {i//batch_size + 1}: {e}")
            
            return {
                "total_citations": len(citations_data),
                "migrated_count": migrated_count,
                "success_rate": migrated_count / len(citations_data) if citations_data else 0
            }
        
        return run_async_task(_migrate())
        
    except Exception as e:
        logger.error(f"Citations batch migration failed: {e}")
        raise self.retry(countdown=60, max_retries=3)


# ==================== TASK MONITORING AND UTILITIES ====================

class GraphTaskMonitor:
    """Monitor and manage graph processing tasks."""
    
    @staticmethod
    def get_task_status(task_id: str) -> Dict[str, Any]:
        """Get status of a specific task."""
        result = AsyncResult(task_id, app=celery_app)
        
        return {
            "task_id": task_id,
            "status": result.status,
            "result": result.result if result.ready() else None,
            "traceback": result.traceback if result.failed() else None,
            "progress": getattr(result, 'info', {}) if result.status == 'PROGRESS' else None
        }
    
    @staticmethod
    def cancel_task(task_id: str) -> bool:
        """Cancel a running task."""
        try:
            celery_app.control.revoke(task_id, terminate=True)
            return True
        except Exception as e:
            logger.error(f"Failed to cancel task {task_id}: {e}")
            return False
    
    @staticmethod
    def get_active_tasks() -> List[Dict[str, Any]]:
        """Get list of currently active tasks."""
        active_tasks = []
        
        try:
            active = celery_app.control.inspect().active()
            if active:
                for worker, tasks in active.items():
                    for task in tasks:
                        active_tasks.append({
                            "worker": worker,
                            "task_id": task["id"],
                            "name": task["name"],
                            "args": task.get("args", []),
                            "kwargs": task.get("kwargs", {}),
                            "time_start": task.get("time_start")
                        })
        except Exception as e:
            logger.error(f"Failed to get active tasks: {e}")
        
        return active_tasks
    
    @staticmethod
    def get_queue_length(queue_name: str = "graph_processing") -> int:
        """Get length of a specific queue."""
        try:
            inspect = celery_app.control.inspect()
            reserved = inspect.reserved()
            
            if reserved:
                for worker, tasks in reserved.items():
                    # This is a simplified approach; actual queue length 
                    # monitoring might need Redis inspection
                    return len(tasks)
            
            return 0
        except Exception as e:
            logger.error(f"Failed to get queue length for {queue_name}: {e}")
            return 0


# Initialize monitor instance
task_monitor = GraphTaskMonitor()