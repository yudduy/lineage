"""
Graph algorithms and analysis API endpoints.
Provides access to Neo4j graph algorithms, community detection, and citation network analysis.
"""

from typing import Any, Dict, List, Optional, Union
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ....db.neo4j_advanced import AdvancedNeo4jManager, get_advanced_neo4j_manager, GraphProjection
from ....services.graph_operations import GraphCRUDOperations
from ....services.graph_tasks import (
    create_citation_projection_task,
    calculate_pagerank_task,
    calculate_all_centralities_task,
    detect_communities_louvain_task,
    detect_communities_leiden_task,
    trace_research_lineage_task,
    analyze_citation_influence_task,
    calculate_comprehensive_metrics_task,
    task_monitor
)
from ....utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


# ==================== REQUEST/RESPONSE MODELS ====================

class GraphProjectionRequest(BaseModel):
    """Request model for creating graph projections."""
    name: str = Field(..., description="Unique name for the graph projection")
    node_labels: List[str] = Field(default=["Paper"], description="Node labels to include")
    relationship_types: List[str] = Field(default=["CITES"], description="Relationship types to include")
    node_properties: Optional[Dict[str, Any]] = Field(default=None, description="Node properties to include")
    relationship_properties: Optional[Dict[str, Any]] = Field(default=None, description="Relationship properties to include")
    orientation: str = Field(default="NATURAL", description="Graph orientation: NATURAL, REVERSE, UNDIRECTED")


class CentralityRequest(BaseModel):
    """Request model for centrality calculations."""
    graph_name: str = Field(..., description="Name of the graph projection")
    algorithm: str = Field(..., description="Centrality algorithm: pagerank, betweenness, closeness")
    write_property: Optional[str] = Field(default=None, description="Property name to write results to")
    max_iterations: Optional[int] = Field(default=20, description="Maximum iterations for iterative algorithms")
    damping_factor: Optional[float] = Field(default=0.85, description="Damping factor for PageRank")
    tolerance: Optional[float] = Field(default=1e-6, description="Convergence tolerance")


class CommunityDetectionRequest(BaseModel):
    """Request model for community detection."""
    graph_name: str = Field(..., description="Name of the graph projection")
    algorithm: str = Field(..., description="Community algorithm: louvain, leiden")
    write_property: Optional[str] = Field(default=None, description="Property name to write results to")
    max_iterations: Optional[int] = Field(default=10, description="Maximum iterations")
    tolerance: Optional[float] = Field(default=1e-6, description="Convergence tolerance")
    gamma: Optional[float] = Field(default=1.0, description="Resolution parameter for Leiden")
    theta: Optional[float] = Field(default=0.01, description="Threshold parameter for Leiden")


class LineageAnalysisRequest(BaseModel):
    """Request model for lineage analysis."""
    paper_id: str = Field(..., description="ID of the paper to analyze")
    direction: str = Field(default="backward", description="Direction: backward, forward, both")
    max_depth: int = Field(default=5, description="Maximum traversal depth")


class PathfindingRequest(BaseModel):
    """Request model for pathfinding."""
    graph_name: str = Field(..., description="Name of the graph projection")
    source_node_id: str = Field(..., description="Source node ID")
    target_node_id: str = Field(..., description="Target node ID")
    relationship_weight_property: Optional[str] = Field(default=None, description="Weight property for weighted paths")
    k: Optional[int] = Field(default=1, description="Number of paths to find")


class GraphSearchRequest(BaseModel):
    """Request model for graph search."""
    query: str = Field(..., description="Search query")
    node_types: Optional[List[str]] = Field(default=["Paper"], description="Node types to search")
    limit: int = Field(default=20, description="Maximum results to return")
    skip: int = Field(default=0, description="Number of results to skip")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Additional filters")


class TaskResponse(BaseModel):
    """Response model for background tasks."""
    task_id: str = Field(..., description="Unique task ID")
    status: str = Field(..., description="Task status")
    message: str = Field(..., description="Status message")


# ==================== GRAPH PROJECTION ENDPOINTS ====================

@router.post("/projections", response_model=Dict[str, Any])
async def create_graph_projection(
    request: GraphProjectionRequest,
    background_tasks: BackgroundTasks,
    neo4j: AdvancedNeo4jManager = Depends(get_advanced_neo4j_manager)
):
    """Create a graph projection for algorithm execution."""
    try:
        # Start background task for projection creation
        task = create_citation_projection_task.delay(
            projection_name=request.name,
            node_labels=request.node_labels,
            relationship_types=request.relationship_types,
            node_properties=request.node_properties,
            orientation=request.orientation
        )
        
        return {
            "task_id": task.id,
            "status": "started",
            "projection_name": request.name,
            "message": f"Graph projection '{request.name}' creation started"
        }
        
    except Exception as e:
        logger.error(f"Failed to create graph projection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projections", response_model=List[Dict[str, Any]])
async def list_graph_projections(
    neo4j: AdvancedNeo4jManager = Depends(get_advanced_neo4j_manager)
):
    """List all graph projections."""
    try:
        projections = await neo4j.list_graph_projections()
        return projections
        
    except Exception as e:
        logger.error(f"Failed to list graph projections: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/projections/{projection_name}")
async def drop_graph_projection(
    projection_name: str,
    neo4j: AdvancedNeo4jManager = Depends(get_advanced_neo4j_manager)
):
    """Drop a graph projection."""
    try:
        success = await neo4j.drop_graph_projection(projection_name)
        
        if success:
            return {"message": f"Graph projection '{projection_name}' dropped successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Graph projection '{projection_name}' not found")
            
    except Exception as e:
        logger.error(f"Failed to drop graph projection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== CENTRALITY ANALYSIS ENDPOINTS ====================

@router.post("/algorithms/centrality", response_model=TaskResponse)
async def calculate_centrality(
    request: CentralityRequest,
    background_tasks: BackgroundTasks
):
    """Calculate centrality measures."""
    try:
        if request.algorithm.lower() == "pagerank":
            task = calculate_pagerank_task.delay(
                graph_name=request.graph_name,
                max_iterations=request.max_iterations,
                damping_factor=request.damping_factor,
                tolerance=request.tolerance,
                write_property=request.write_property
            )
        elif request.algorithm.lower() == "all":
            task = calculate_all_centralities_task.delay(
                graph_name=request.graph_name,
                write_properties=bool(request.write_property)
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported centrality algorithm: {request.algorithm}"
            )
        
        return TaskResponse(
            task_id=task.id,
            status="started",
            message=f"{request.algorithm.title()} centrality calculation started"
        )
        
    except Exception as e:
        logger.error(f"Failed to start centrality calculation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/algorithms/centrality/{graph_name}/pagerank")
async def get_pagerank_scores(
    graph_name: str,
    limit: int = Query(default=20, description="Number of top scores to return"),
    neo4j: AdvancedNeo4jManager = Depends(get_advanced_neo4j_manager)
):
    """Get PageRank scores for a graph projection."""
    try:
        result = await neo4j.calculate_pagerank(graph_name=graph_name)
        
        return {
            "algorithm": result.algorithm,
            "execution_time": result.execution_time,
            "top_papers": result.top_nodes[:limit],
            "statistics": result.statistics,
            "total_papers": len(result.scores) if result.scores else 0
        }
        
    except Exception as e:
        logger.error(f"Failed to get PageRank scores: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== COMMUNITY DETECTION ENDPOINTS ====================

@router.post("/algorithms/communities", response_model=TaskResponse)
async def detect_communities(
    request: CommunityDetectionRequest,
    background_tasks: BackgroundTasks
):
    """Detect communities in the graph."""
    try:
        if request.algorithm.lower() == "louvain":
            task = detect_communities_louvain_task.delay(
                graph_name=request.graph_name,
                max_iterations=request.max_iterations,
                tolerance=request.tolerance,
                write_property=request.write_property
            )
        elif request.algorithm.lower() == "leiden":
            task = detect_communities_leiden_task.delay(
                graph_name=request.graph_name,
                max_iterations=request.max_iterations,
                gamma=request.gamma,
                theta=request.theta,
                write_property=request.write_property
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported community detection algorithm: {request.algorithm}"
            )
        
        return TaskResponse(
            task_id=task.id,
            status="started",
            message=f"{request.algorithm.title()} community detection started"
        )
        
    except Exception as e:
        logger.error(f"Failed to start community detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/algorithms/communities/{graph_name}/louvain")
async def get_louvain_communities(
    graph_name: str,
    neo4j: AdvancedNeo4jManager = Depends(get_advanced_neo4j_manager)
):
    """Get Louvain community detection results."""
    try:
        result = await neo4j.detect_communities_louvain(graph_name=graph_name)
        
        # Calculate community statistics
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
        
    except Exception as e:
        logger.error(f"Failed to get Louvain communities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== LINEAGE ANALYSIS ENDPOINTS ====================

@router.post("/analysis/lineage", response_model=TaskResponse)
async def trace_intellectual_lineage(
    request: LineageAnalysisRequest,
    background_tasks: BackgroundTasks
):
    """Trace intellectual lineage of a paper."""
    try:
        task = trace_research_lineage_task.delay(
            paper_id=request.paper_id,
            direction=request.direction,
            max_depth=request.max_depth
        )
        
        return TaskResponse(
            task_id=task.id,
            status="started",
            message=f"Intellectual lineage tracing started for paper {request.paper_id}"
        )
        
    except Exception as e:
        logger.error(f"Failed to start lineage tracing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analysis/lineage/{paper_id}")
async def get_research_lineage(
    paper_id: str,
    direction: str = Query(default="backward", description="Direction: backward, forward, both"),
    max_depth: int = Query(default=5, description="Maximum traversal depth"),
    neo4j: AdvancedNeo4jManager = Depends(get_advanced_neo4j_manager)
):
    """Get research lineage for a paper."""
    try:
        lineage = await neo4j.find_research_lineage(
            paper_id=paper_id,
            direction=direction,
            max_depth=max_depth
        )
        
        return lineage
        
    except Exception as e:
        logger.error(f"Failed to get research lineage: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analysis/influence", response_model=TaskResponse)
async def analyze_citation_influence(
    request: LineageAnalysisRequest,
    background_tasks: BackgroundTasks
):
    """Analyze citation influence patterns."""
    try:
        task = analyze_citation_influence_task.delay(
            paper_id=request.paper_id,
            max_depth=request.max_depth
        )
        
        return TaskResponse(
            task_id=task.id,
            status="started",
            message=f"Citation influence analysis started for paper {request.paper_id}"
        )
        
    except Exception as e:
        logger.error(f"Failed to start influence analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== PATHFINDING ENDPOINTS ====================

@router.post("/algorithms/paths/shortest")
async def find_shortest_path(
    request: PathfindingRequest,
    neo4j: AdvancedNeo4jManager = Depends(get_advanced_neo4j_manager)
):
    """Find shortest path between two nodes."""
    try:
        if request.k and request.k > 1:
            paths = await neo4j.find_all_shortest_paths(
                graph_name=request.graph_name,
                source_node_id=request.source_node_id,
                target_node_id=request.target_node_id,
                k=request.k
            )
            return {"paths": paths}
        else:
            path = await neo4j.find_shortest_path(
                graph_name=request.graph_name,
                source_node_id=request.source_node_id,
                target_node_id=request.target_node_id,
                relationship_weight_property=request.relationship_weight_property
            )
            return {"path": path}
        
    except Exception as e:
        logger.error(f"Failed to find shortest path: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== GRAPH METRICS ENDPOINTS ====================

@router.get("/metrics/{graph_name}")
async def get_graph_metrics(
    graph_name: str,
    neo4j: AdvancedNeo4jManager = Depends(get_advanced_neo4j_manager)
):
    """Get basic graph metrics."""
    try:
        metrics = await neo4j.calculate_graph_metrics(graph_name)
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get graph metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/metrics/{graph_name}/comprehensive", response_model=TaskResponse)
async def calculate_comprehensive_metrics(
    graph_name: str,
    include_centrality: bool = Query(default=True, description="Include centrality metrics"),
    include_communities: bool = Query(default=True, description="Include community metrics"),
    background_tasks: BackgroundTasks = None
):
    """Calculate comprehensive graph metrics."""
    try:
        task = calculate_comprehensive_metrics_task.delay(
            graph_name=graph_name,
            include_centrality=include_centrality,
            include_communities=include_communities
        )
        
        return TaskResponse(
            task_id=task.id,
            status="started",
            message=f"Comprehensive metrics calculation started for graph '{graph_name}'"
        )
        
    except Exception as e:
        logger.error(f"Failed to start comprehensive metrics calculation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== SEARCH ENDPOINTS ====================

@router.post("/search")
async def search_graph(
    request: GraphSearchRequest,
    neo4j: AdvancedNeo4jManager = Depends(get_advanced_neo4j_manager)
):
    """Search the graph with full-text search."""
    try:
        crud_ops = GraphCRUDOperations(neo4j)
        
        if "Paper" in request.node_types:
            results = await crud_ops.search_papers(
                query_text=request.query,
                limit=request.limit,
                skip=request.skip,
                filters=request.filters
            )
            return {"papers": results}
        else:
            # For now, only paper search is implemented
            return {"message": "Only paper search is currently supported"}
        
    except Exception as e:
        logger.error(f"Failed to search graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== TASK MANAGEMENT ENDPOINTS ====================

@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get status of a background task."""
    try:
        status = task_monitor.get_task_status(task_id)
        return status
        
    except Exception as e:
        logger.error(f"Failed to get task status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/tasks/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a running task."""
    try:
        success = task_monitor.cancel_task(task_id)
        
        if success:
            return {"message": f"Task {task_id} cancelled successfully"}
        else:
            return {"message": f"Failed to cancel task {task_id}"}
            
    except Exception as e:
        logger.error(f"Failed to cancel task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks/active")
async def get_active_tasks():
    """Get list of active tasks."""
    try:
        active_tasks = task_monitor.get_active_tasks()
        return {"active_tasks": active_tasks}
        
    except Exception as e:
        logger.error(f"Failed to get active tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== UTILITY ENDPOINTS ====================

@router.get("/statistics")
async def get_graph_statistics(
    neo4j: AdvancedNeo4jManager = Depends(get_advanced_neo4j_manager)
):
    """Get overall graph database statistics."""
    try:
        crud_ops = GraphCRUDOperations(neo4j)
        stats = await crud_ops.get_node_statistics()
        
        return {
            "node_counts": stats,
            "timestamp": "2025-08-08T00:00:00Z"  # Current timestamp would be dynamic
        }
        
    except Exception as e:
        logger.error(f"Failed to get graph statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/maintenance/cleanup")
async def cleanup_orphaned_nodes(
    neo4j: AdvancedNeo4jManager = Depends(get_advanced_neo4j_manager)
):
    """Clean up orphaned nodes with no relationships."""
    try:
        crud_ops = GraphCRUDOperations(neo4j)
        cleanup_results = await crud_ops.cleanup_orphaned_nodes()
        
        return {
            "message": "Cleanup completed",
            "cleanup_results": cleanup_results
        }
        
    except Exception as e:
        logger.error(f"Failed to cleanup orphaned nodes: {e}")
        raise HTTPException(status_code=500, detail=str(e))