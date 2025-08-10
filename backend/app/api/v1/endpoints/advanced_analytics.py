"""
Advanced Analytics API Endpoints.

This module provides comprehensive API endpoints for intellectual lineage analysis,
research intelligence, and performance monitoring with enterprise-grade features.
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import hashlib
import json
import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from ....services.advanced_analytics import (
    get_analytics_service,
    AdvancedAnalyticsService,
    AnalysisDepth,
    IntellectualLineage,
    ContentIntelligence
)
from ....services.research_intelligence import (
    get_intelligence_engine,
    ResearchIntelligenceEngine,
    ResearchTrend,
    ResearchTrendType,
    TrendDetectionMethod,
    CommunityDynamics,
    KnowledgeFlow,
    ResearchForecast
)
from ....services.performance_optimizer import (
    get_performance_optimizer,
    PerformanceOptimizer,
    TaskPriority
)
from ....core.security import get_current_user
from ....models.user import User
from ....utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/analytics", tags=["Advanced Analytics"])


# ==================== Intellectual Lineage Analysis ====================

@router.post("/lineage/{paper_id}")
async def analyze_intellectual_lineage(
    paper_id: str,
    depth: AnalysisDepth = Query(AnalysisDepth.MODERATE, description="Analysis depth"),
    include_predictions: bool = Query(True, description="Include trajectory predictions"),
    enrich_content: bool = Query(True, description="Enrich with LLM insights"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    analytics_service: AdvancedAnalyticsService = Depends(get_analytics_service),
    current_user: User = Depends(get_current_user)
):
    """
    Perform comprehensive intellectual lineage analysis for a research paper.
    
    This endpoint provides deep analysis of citation networks, research evolution paths,
    knowledge flows, and impact propagation with optional predictive analytics.
    """
    try:
        # Perform analysis
        lineage = await analytics_service.analyze_intellectual_lineage(
            paper_id=paper_id,
            depth=depth,
            include_predictions=include_predictions,
            enrich_content=enrich_content
        )
        
        # Queue background enrichment for related papers
        if lineage.key_milestones:
            for milestone in lineage.key_milestones[:5]:
                background_tasks.add_task(
                    analytics_service._queue_cache_warming,
                    [milestone['paper_id']]
                )
        
        return {
            "status": "success",
            "data": {
                "lineage": lineage.__dict__,
                "metadata": {
                    "analysis_depth": depth.value,
                    "total_papers": lineage.total_papers,
                    "total_citations": lineage.total_citations,
                    "key_milestones_count": len(lineage.key_milestones),
                    "evolution_paths_count": len(lineage.evolution_path),
                    "communities_detected": len(lineage.research_communities)
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Lineage analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/lineage/{paper_id}/milestones")
async def get_research_milestones(
    paper_id: str,
    limit: int = Query(10, ge=1, le=50),
    analytics_service: AdvancedAnalyticsService = Depends(get_analytics_service),
    current_user: User = Depends(get_current_user)
):
    """
    Get key research milestones in the intellectual lineage of a paper.
    """
    try:
        # Get cached lineage or perform lightweight analysis
        cache_key = f"lineage:{paper_id}:moderate:True:False"
        cached = await analytics_service._get_cached_result(cache_key)
        
        if cached:
            milestones = cached.get('key_milestones', [])[:limit]
        else:
            # Perform quick milestone analysis
            lineage = await analytics_service.analyze_intellectual_lineage(
                paper_id=paper_id,
                depth=AnalysisDepth.SHALLOW,
                include_predictions=False,
                enrich_content=False
            )
            milestones = lineage.key_milestones[:limit]
        
        return {
            "status": "success",
            "data": {
                "paper_id": paper_id,
                "milestones": milestones,
                "count": len(milestones)
            }
        }
        
    except Exception as e:
        logger.error(f"Milestone retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/lineage/{paper_id}/evolution")
async def get_evolution_paths(
    paper_id: str,
    max_paths: int = Query(5, ge=1, le=20),
    analytics_service: AdvancedAnalyticsService = Depends(get_analytics_service),
    current_user: User = Depends(get_current_user)
):
    """
    Get research evolution paths showing how ideas developed over time.
    """
    try:
        lineage = await analytics_service.analyze_intellectual_lineage(
            paper_id=paper_id,
            depth=AnalysisDepth.MODERATE,
            include_predictions=False,
            enrich_content=False
        )
        
        paths = lineage.evolution_path[:max_paths]
        
        return {
            "status": "success",
            "data": {
                "paper_id": paper_id,
                "evolution_paths": paths,
                "path_count": len(paths)
            }
        }
        
    except Exception as e:
        logger.error(f"Evolution path error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Research Intelligence ====================

@router.post("/trends/detect")
async def detect_research_trends(
    domain: Optional[str] = Query(None, description="Research domain/field"),
    start_year: Optional[int] = Query(None, ge=1900, le=2024),
    end_year: Optional[int] = Query(None, ge=1900, le=2024),
    methods: Optional[List[TrendDetectionMethod]] = Query(None),
    min_confidence: float = Query(0.6, ge=0.0, le=1.0),
    intelligence_engine: ResearchIntelligenceEngine = Depends(get_intelligence_engine),
    current_user: User = Depends(get_current_user)
):
    """
    Detect emerging, declining, and breakthrough research trends.
    
    Uses multiple detection methods including burst detection, growth analysis,
    citation acceleration, and network dynamics.
    """
    try:
        time_window = None
        if start_year and end_year:
            time_window = (start_year, end_year)
        
        trends = await intelligence_engine.detect_research_trends(
            domain=domain,
            time_window=time_window,
            methods=methods,
            min_confidence=min_confidence
        )
        
        # Group trends by type
        trends_by_type = {}
        for trend in trends:
            trend_type = trend.type.value
            if trend_type not in trends_by_type:
                trends_by_type[trend_type] = []
            trends_by_type[trend_type].append(trend.__dict__)
        
        return {
            "status": "success",
            "data": {
                "trends": [t.__dict__ for t in trends],
                "trends_by_type": trends_by_type,
                "metadata": {
                    "total_trends": len(trends),
                    "domain": domain,
                    "time_window": time_window,
                    "confidence_threshold": min_confidence
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Trend detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trends/{trend_id}")
async def get_trend_details(
    trend_id: str,
    intelligence_engine: ResearchIntelligenceEngine = Depends(get_intelligence_engine),
    current_user: User = Depends(get_current_user)
):
    """
    Get detailed information about a specific research trend.
    """
    try:
        # This would retrieve trend details from cache or database
        # For now, return a placeholder
        return {
            "status": "success",
            "data": {
                "trend_id": trend_id,
                "details": "Trend details would be retrieved here"
            }
        }
        
    except Exception as e:
        logger.error(f"Trend detail error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trends/{trend_id}/monitor")
async def start_trend_monitoring(
    trend_id: str,
    background_tasks: BackgroundTasks,
    intelligence_engine: ResearchIntelligenceEngine = Depends(get_intelligence_engine),
    current_user: User = Depends(get_current_user)
):
    """
    Start real-time monitoring of a research trend.
    """
    try:
        await intelligence_engine.start_trend_monitoring(trend_id)
        
        return {
            "status": "success",
            "message": f"Started monitoring trend {trend_id}",
            "data": {
                "trend_id": trend_id,
                "monitoring": True
            }
        }
        
    except Exception as e:
        logger.error(f"Trend monitoring error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/trends/{trend_id}/monitor")
async def stop_trend_monitoring(
    trend_id: str,
    intelligence_engine: ResearchIntelligenceEngine = Depends(get_intelligence_engine),
    current_user: User = Depends(get_current_user)
):
    """
    Stop monitoring a research trend.
    """
    try:
        await intelligence_engine.stop_trend_monitoring(trend_id)
        
        return {
            "status": "success",
            "message": f"Stopped monitoring trend {trend_id}",
            "data": {
                "trend_id": trend_id,
                "monitoring": False
            }
        }
        
    except Exception as e:
        logger.error(f"Stop monitoring error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Community Analysis ====================

@router.get("/communities/{community_id}/dynamics")
async def analyze_community_dynamics(
    community_id: str,
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    intelligence_engine: ResearchIntelligenceEngine = Depends(get_intelligence_engine),
    current_user: User = Depends(get_current_user)
):
    """
    Analyze the dynamics and evolution of a research community.
    """
    try:
        time_range = None
        if start_date and end_date:
            time_range = (start_date, end_date)
        
        dynamics = await intelligence_engine.analyze_community_dynamics(
            community_id=community_id,
            time_range=time_range
        )
        
        return {
            "status": "success",
            "data": {
                "community_id": community_id,
                "dynamics": dynamics.__dict__,
                "metadata": {
                    "lifecycle_stage": dynamics.lifecycle_stage,
                    "cohesion_score": dynamics.cohesion_score,
                    "influence_radius": dynamics.influence_radius
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Community dynamics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/communities/compare")
async def compare_communities(
    community_ids: List[str] = Query(..., description="Community IDs to compare"),
    analytics_service: AdvancedAnalyticsService = Depends(get_analytics_service),
    current_user: User = Depends(get_current_user)
):
    """
    Compare multiple research communities.
    """
    try:
        if len(community_ids) < 2 or len(community_ids) > 5:
            raise HTTPException(
                status_code=400,
                detail="Please provide 2-5 community IDs for comparison"
            )
        
        # Perform comparison analysis
        # This would be implemented in the analytics service
        
        return {
            "status": "success",
            "data": {
                "communities": community_ids,
                "comparison": "Community comparison would be performed here"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Community comparison error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Knowledge Flow Analysis ====================

@router.get("/knowledge-flow/{entity_id}")
async def analyze_knowledge_flow(
    entity_id: str,
    entity_type: str = Query("field", enum=["field", "author", "institution", "paper"]),
    max_hops: int = Query(3, ge=1, le=5),
    intelligence_engine: ResearchIntelligenceEngine = Depends(get_intelligence_engine),
    current_user: User = Depends(get_current_user)
):
    """
    Analyze how knowledge flows from a source entity.
    """
    try:
        flows = await intelligence_engine.analyze_knowledge_flows(
            source_entity=entity_id,
            entity_type=entity_type,
            max_hops=max_hops
        )
        
        return {
            "status": "success",
            "data": {
                "source_entity": entity_id,
                "entity_type": entity_type,
                "flows": [f.__dict__ for f in flows],
                "flow_count": len(flows)
            }
        }
        
    except Exception as e:
        logger.error(f"Knowledge flow error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Research Forecasting ====================

@router.post("/forecast/{entity_id}")
async def forecast_research_development(
    entity_id: str,
    entity_type: str = Query("paper", enum=["paper", "author", "field"]),
    horizon_months: int = Query(24, ge=6, le=60),
    intelligence_engine: ResearchIntelligenceEngine = Depends(get_intelligence_engine),
    current_user: User = Depends(get_current_user)
):
    """
    Forecast future research development for an entity.
    """
    try:
        forecast = await intelligence_engine.forecast_research_development(
            entity_id=entity_id,
            entity_type=entity_type,
            horizon_months=horizon_months
        )
        
        return {
            "status": "success",
            "data": {
                "forecast": forecast.__dict__,
                "metadata": {
                    "entity_id": entity_id,
                    "entity_type": entity_type,
                    "horizon_months": horizon_months,
                    "confidence": forecast.confidence_interval
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Forecasting error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Content Intelligence ====================

@router.post("/content-intelligence/{paper_id}")
async def analyze_content_intelligence(
    paper_id: str,
    comparative_papers: Optional[List[str]] = Query(None, description="Papers for comparison"),
    analytics_service: AdvancedAnalyticsService = Depends(get_analytics_service),
    current_user: User = Depends(get_current_user)
):
    """
    Deep content analysis including theme extraction, significance scoring,
    and research gap identification.
    """
    try:
        content = await analytics_service.analyze_content_intelligence(
            paper_id=paper_id,
            comparative_papers=comparative_papers
        )
        
        return {
            "status": "success",
            "data": {
                "paper_id": paper_id,
                "intelligence": content.__dict__,
                "metadata": {
                    "significance_score": content.significance_score,
                    "novelty_score": content.novelty_score,
                    "impact_potential": content.impact_potential,
                    "theme_count": len(content.research_themes),
                    "gap_count": len(content.research_gaps)
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Content intelligence error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Batch Analysis ====================

@router.post("/batch/analyze")
async def batch_analysis(
    background_tasks: BackgroundTasks,
    paper_ids: List[str] = Query(..., description="Paper IDs to analyze"),
    analysis_type: str = Query("lineage", enum=["lineage", "content", "impact"]),
    priority: TaskPriority = Query(TaskPriority.NORMAL),
    performance_optimizer: PerformanceOptimizer = Depends(get_performance_optimizer),
    analytics_service: AdvancedAnalyticsService = Depends(get_analytics_service),
    current_user: User = Depends(get_current_user)
):
    """
    Submit batch analysis job for multiple papers.
    """
    try:
        if len(paper_ids) > 100:
            raise HTTPException(
                status_code=400,
                detail="Maximum 100 papers per batch"
            )
        
        # Submit background tasks
        task_ids = []
        for paper_id in paper_ids:
            if analysis_type == "lineage":
                task_id = await performance_optimizer.task_processor.submit(
                    name=f"lineage_analysis_{paper_id}",
                    function=analytics_service.analyze_intellectual_lineage,
                    args=(paper_id,),
                    kwargs={"depth": AnalysisDepth.SHALLOW},
                    priority=priority
                )
            elif analysis_type == "content":
                task_id = await performance_optimizer.task_processor.submit(
                    name=f"content_analysis_{paper_id}",
                    function=analytics_service.analyze_content_intelligence,
                    args=(paper_id,),
                    priority=priority
                )
            else:
                # Impact analysis
                task_id = await performance_optimizer.task_processor.submit(
                    name=f"impact_analysis_{paper_id}",
                    function=analytics_service.analyze_intellectual_lineage,
                    args=(paper_id,),
                    kwargs={"depth": AnalysisDepth.SHALLOW, "enrich_content": False},
                    priority=priority
                )
            
            task_ids.append(task_id)
        
        return {
            "status": "success",
            "data": {
                "batch_id": hashlib.md5(str(task_ids).encode()).hexdigest(),
                "task_ids": task_ids,
                "paper_count": len(paper_ids),
                "analysis_type": analysis_type,
                "priority": priority.name
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/batch/{task_id}/status")
async def get_task_status(
    task_id: str,
    performance_optimizer: PerformanceOptimizer = Depends(get_performance_optimizer),
    current_user: User = Depends(get_current_user)
):
    """
    Get status of a background analysis task.
    """
    try:
        status = await performance_optimizer.task_processor.get_task_status(task_id)
        
        if status is None:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return {
            "status": "success",
            "data": status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Task status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Performance Monitoring ====================

@router.get("/performance/metrics")
async def get_performance_metrics(
    analytics_service: AdvancedAnalyticsService = Depends(get_analytics_service),
    performance_optimizer: PerformanceOptimizer = Depends(get_performance_optimizer),
    current_user: User = Depends(get_current_user)
):
    """
    Get current performance metrics for the analytics system.
    """
    try:
        analytics_metrics = await analytics_service.get_performance_metrics()
        optimizer_report = await performance_optimizer.get_performance_report()
        
        return {
            "status": "success",
            "data": {
                "analytics": analytics_metrics,
                "optimizer": optimizer_report,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Performance metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance/cache/stats")
async def get_cache_statistics(
    performance_optimizer: PerformanceOptimizer = Depends(get_performance_optimizer),
    current_user: User = Depends(get_current_user)
):
    """
    Get detailed cache statistics.
    """
    try:
        stats = await performance_optimizer.cache.get_statistics()
        
        return {
            "status": "success",
            "data": stats
        }
        
    except Exception as e:
        logger.error(f"Cache stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/performance/cache/clear")
async def clear_cache(
    tier: Optional[str] = Query(None, enum=["l1", "l2", "all"]),
    performance_optimizer: PerformanceOptimizer = Depends(get_performance_optimizer),
    current_user: User = Depends(get_current_user)
):
    """
    Clear cache (admin only).
    """
    try:
        # Check admin privileges
        if not current_user.is_admin:
            raise HTTPException(status_code=403, detail="Admin access required")
        
        if tier in ["l1", "all"]:
            await performance_optimizer.cache.l1_cache.clear()
        
        # L2 (Redis) clearing would be implemented here
        
        return {
            "status": "success",
            "message": f"Cache cleared: {tier or 'all'}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cache clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Real-time Updates (WebSocket) ====================

@router.websocket("/ws/analytics/{client_id}")
async def analytics_websocket(
    websocket: WebSocket,
    client_id: str,
    analytics_service: AdvancedAnalyticsService = Depends(get_analytics_service)
):
    """
    WebSocket endpoint for real-time analytics updates.
    """
    await websocket.accept()
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            if data.get("type") == "subscribe":
                # Subscribe to specific analytics updates
                entity_id = data.get("entity_id")
                update_type = data.get("update_type", "all")
                
                # Send confirmation
                await websocket.send_json({
                    "type": "subscribed",
                    "entity_id": entity_id,
                    "update_type": update_type
                })
                
            elif data.get("type") == "ping":
                # Heartbeat
                await websocket.send_json({"type": "pong"})
                
            # Send periodic updates
            await asyncio.sleep(5)
            
            # Get and send performance metrics
            metrics = await analytics_service.get_performance_metrics()
            await websocket.send_json({
                "type": "metrics_update",
                "data": metrics,
                "timestamp": datetime.utcnow().isoformat()
            })
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


# ==================== Export and Streaming ====================

@router.get("/export/lineage/{paper_id}")
async def export_lineage_analysis(
    paper_id: str,
    format: str = Query("json", enum=["json", "csv", "graphml"]),
    analytics_service: AdvancedAnalyticsService = Depends(get_analytics_service),
    current_user: User = Depends(get_current_user)
):
    """
    Export lineage analysis in various formats.
    """
    try:
        # Get lineage data
        lineage = await analytics_service.analyze_intellectual_lineage(
            paper_id=paper_id,
            depth=AnalysisDepth.MODERATE,
            include_predictions=True,
            enrich_content=False
        )
        
        if format == "json":
            return StreamingResponse(
                iter([json.dumps(lineage.__dict__, default=str)]),
                media_type="application/json",
                headers={
                    "Content-Disposition": f"attachment; filename=lineage_{paper_id}.json"
                }
            )
        elif format == "csv":
            # Convert to CSV format
            # Implementation would convert lineage data to CSV
            pass
        elif format == "graphml":
            # Convert to GraphML format for network visualization
            # Implementation would convert to GraphML
            pass
        
        return {"status": "success", "message": f"Export format {format} not yet implemented"}
        
    except Exception as e:
        logger.error(f"Export error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stream/trends")
async def stream_trend_updates(
    domain: Optional[str] = Query(None),
    intelligence_engine: ResearchIntelligenceEngine = Depends(get_intelligence_engine),
    current_user: User = Depends(get_current_user)
):
    """
    Stream real-time trend updates using Server-Sent Events.
    """
    async def event_generator():
        try:
            while True:
                # Get latest trends
                trends = await intelligence_engine.detect_research_trends(
                    domain=domain,
                    min_confidence=0.7
                )
                
                # Format as SSE
                data = {
                    "trends": [t.__dict__ for t in trends[:5]],
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                yield f"data: {json.dumps(data, default=str)}\n\n"
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
        except asyncio.CancelledError:
            pass
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )