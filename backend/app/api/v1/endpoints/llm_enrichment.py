"""
LLM Enrichment API Endpoints - REST API for LLM-powered research analysis.

This module provides endpoints for:
- On-demand paper enrichment and analysis
- Batch processing of multiple papers
- Citation relationship analysis
- Research trajectory and lineage tracing
- Cost monitoring and usage analytics
- Task status tracking and management
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, Path, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import logging

from ....services.content_enrichment import (
    get_enrichment_service, 
    EnrichmentRequest, 
    EnrichedContent,
    ContentQuality
)
from ....services.citation_analysis import (
    get_citation_analysis_service,
    CitationAnalysisRequest,
    CitationRelationship,
    CitationType,
    InfluenceLevel
)
from ....services.research_trajectory import (
    get_trajectory_service,
    IntellectualLineage,
    ResearchTimeline,
    TrajectoryType
)
from ....services.llm_cost_manager import get_cost_manager, CostAnalytics
from ....services.llm_cache import get_cache_manager
from ....services.llm_tasks import get_task_manager
from ....core.config import get_settings
from ....utils.logger import get_logger
from ....utils.exceptions import ValidationError

logger = get_logger(__name__)
router = APIRouter()


# Request/Response Models

class PaperEnrichmentRequest(BaseModel):
    """Request for paper enrichment."""
    paper_id: str = Field(..., description="Unique paper identifier")
    paper_data: Optional[Dict[str, Any]] = Field(None, description="Optional paper metadata")
    force_refresh: bool = Field(False, description="Force new analysis even if cached")
    use_cache: bool = Field(True, description="Whether to use LLM response caching")
    priority: str = Field("normal", description="Processing priority (high/normal/low)")
    
    @validator('priority')
    def validate_priority(cls, v):
        if v not in ['high', 'normal', 'low']:
            raise ValueError('Priority must be high, normal, or low')
        return v


class BatchEnrichmentRequest(BaseModel):
    """Request for batch paper enrichment."""
    paper_requests: List[Dict[str, Any]] = Field(..., description="List of paper enrichment requests")
    max_concurrency: int = Field(3, ge=1, le=10, description="Maximum concurrent processes")
    async_processing: bool = Field(False, description="Process asynchronously in background")
    priority: str = Field("normal", description="Processing priority")
    
    @validator('paper_requests')
    def validate_paper_requests(cls, v):
        if not v:
            raise ValueError('Must provide at least one paper request')
        if len(v) > 100:
            raise ValueError('Cannot process more than 100 papers in a single batch')
        return v


class CitationAnalysisRequest(BaseModel):
    """Request for citation analysis."""
    citing_paper_id: str = Field(..., description="ID of paper that cites")
    cited_paper_id: str = Field(..., description="ID of paper being cited")
    citation_context: Optional[str] = Field(None, description="Context text where citation appears")
    force_refresh: bool = Field(False, description="Force new analysis")
    include_network_analysis: bool = Field(False, description="Include broader network analysis")


class LineageTracingRequest(BaseModel):
    """Request for intellectual lineage tracing."""
    seed_paper_ids: List[str] = Field(..., description="Starting papers for lineage tracing")
    max_generations: int = Field(3, ge=1, le=10, description="Maximum generations to trace")
    include_future_work: bool = Field(True, description="Include papers citing seed papers")
    generate_timeline: bool = Field(False, description="Generate timeline narrative")
    async_processing: bool = Field(True, description="Process asynchronously")
    
    @validator('seed_paper_ids')
    def validate_seed_papers(cls, v):
        if not v:
            raise ValueError('Must provide at least one seed paper')
        if len(v) > 10:
            raise ValueError('Cannot start with more than 10 seed papers')
        return v


class EnrichmentResponse(BaseModel):
    """Response for paper enrichment."""
    status: str
    paper_id: str
    enrichment_quality: Optional[str] = None
    enhanced_summary: Optional[str] = None
    key_contributions: Optional[List[str]] = None
    methodology_summary: Optional[str] = None
    key_findings: Optional[List[str]] = None
    significance_assessment: Optional[str] = None
    limitations: Optional[List[str]] = None
    future_directions: Optional[List[str]] = None
    confidence_score: Optional[float] = None
    enrichment_cost: Optional[float] = None
    enrichment_tokens: Optional[int] = None
    cached: Optional[bool] = None
    processing_time_ms: Optional[float] = None


class CitationAnalysisResponse(BaseModel):
    """Response for citation analysis."""
    status: str
    citing_paper_id: str
    cited_paper_id: str
    citation_purpose: Optional[str] = None
    intellectual_relationship: Optional[str] = None
    knowledge_flow_description: Optional[str] = None
    impact_assessment: Optional[str] = None
    citation_type: Optional[str] = None
    influence_level: Optional[str] = None
    analysis_confidence: Optional[float] = None
    analysis_cost: Optional[float] = None
    processing_time_ms: Optional[float] = None


class TaskStatusResponse(BaseModel):
    """Response for task status."""
    task_id: str
    status: str
    ready: bool
    successful: Optional[bool] = None
    failed: Optional[bool] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    submitted_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress: Optional[Dict[str, Any]] = None


# Dependency functions

async def get_settings_dep():
    """Dependency to get application settings."""
    return get_settings()


# API Endpoints

@router.post("/papers/{paper_id}/enrich", 
             response_model=EnrichmentResponse,
             summary="Enrich Single Paper",
             description="Perform comprehensive LLM-based analysis of a single research paper")
async def enrich_paper(
    paper_id: str = Path(..., description="Unique paper identifier"),
    request: Optional[PaperEnrichmentRequest] = None,
    force_refresh: bool = Query(False, description="Force new analysis"),
    use_cache: bool = Query(True, description="Use LLM caching"),
    settings = Depends(get_settings_dep)
):
    """Enrich a single paper with comprehensive LLM analysis."""
    
    start_time = datetime.now()
    
    try:
        # Get enrichment service
        enrichment_service = await get_enrichment_service()
        
        # Prepare request data
        if request:
            paper_data = request.paper_data
            force_refresh = request.force_refresh
            use_cache = request.use_cache
        else:
            paper_data = None
        
        # Check if async processing requested
        if request and request.priority == 'high':
            # Process synchronously for high priority
            enriched_content = await enrichment_service.enrich_paper(
                paper_id=paper_id,
                paper_data=paper_data,
                force_refresh=force_refresh,
                use_cache=use_cache
            )
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return EnrichmentResponse(
                status="completed",
                paper_id=paper_id,
                enrichment_quality=enriched_content.content_quality.value,
                enhanced_summary=enriched_content.enhanced_summary,
                key_contributions=enriched_content.key_contributions,
                methodology_summary=enriched_content.methodology_summary,
                key_findings=enriched_content.key_findings,
                significance_assessment=enriched_content.significance_assessment,
                limitations=enriched_content.limitations,
                future_directions=enriched_content.future_directions,
                confidence_score=enriched_content.confidence_score,
                enrichment_cost=enriched_content.enrichment_cost,
                enrichment_tokens=enriched_content.enrichment_tokens,
                processing_time_ms=processing_time
            )
        else:
            # Submit as background task for normal/low priority
            task_manager = await get_task_manager()
            task_id = await task_manager.submit_paper_enrichment(
                paper_id=paper_id,
                paper_data=paper_data,
                priority=request.priority if request else 'normal',
                force_refresh=force_refresh
            )
            
            return JSONResponse(
                status_code=status.HTTP_202_ACCEPTED,
                content={
                    "status": "accepted",
                    "message": "Paper enrichment submitted for background processing",
                    "task_id": task_id,
                    "paper_id": paper_id
                }
            )
    
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Paper enrichment failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/papers/enrich/batch",
             summary="Batch Paper Enrichment",
             description="Enrich multiple papers in batch with controlled concurrency")
async def batch_enrich_papers(
    request: BatchEnrichmentRequest,
    settings = Depends(get_settings_dep)
):
    """Enrich multiple papers in batch."""
    
    try:
        task_manager = await get_task_manager()
        
        if request.async_processing:
            # Submit as background task
            task_id = await task_manager.submit_batch_enrichment(
                paper_requests=request.paper_requests,
                max_concurrency=request.max_concurrency,
                priority=request.priority
            )
            
            return JSONResponse(
                status_code=status.HTTP_202_ACCEPTED,
                content={
                    "status": "accepted",
                    "message": "Batch enrichment submitted for background processing",
                    "task_id": task_id,
                    "paper_count": len(request.paper_requests),
                    "max_concurrency": request.max_concurrency
                }
            )
        else:
            # Process synchronously (limited batch size)
            if len(request.paper_requests) > 20:
                raise HTTPException(
                    status_code=400, 
                    detail="Synchronous batch processing limited to 20 papers. Use async_processing=true for larger batches."
                )
            
            enrichment_service = await get_enrichment_service()
            
            # Convert to EnrichmentRequest objects
            enrichment_requests = []
            for req_data in request.paper_requests:
                enrichment_req = EnrichmentRequest(
                    paper_id=req_data['paper_id'],
                    paper_data=req_data.get('paper_data'),
                    priority=req_data.get('priority', 1),
                    force_refresh=req_data.get('force_refresh', False)
                )
                enrichment_requests.append(enrichment_req)
            
            # Process batch
            batch_result = await enrichment_service.batch_enrich_papers(
                paper_requests=enrichment_requests,
                max_concurrency=request.max_concurrency
            )
            
            return {
                "status": "completed",
                "total_papers": batch_result.total_papers,
                "successful_enrichments": batch_result.successful_enrichments,
                "failed_enrichments": batch_result.failed_enrichments,
                "cached_results": batch_result.cached_results,
                "total_cost": batch_result.total_cost,
                "total_tokens": batch_result.total_tokens,
                "processing_time_seconds": batch_result.processing_time_seconds,
                "errors": batch_result.error_summary
            }
    
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Batch enrichment failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/citations/analyze",
             response_model=CitationAnalysisResponse,
             summary="Analyze Citation Relationship",
             description="Analyze the intellectual relationship between two papers")
async def analyze_citation(
    request: CitationAnalysisRequest,
    async_processing: bool = Query(False, description="Process asynchronously"),
    settings = Depends(get_settings_dep)
):
    """Analyze citation relationship between two papers."""
    
    start_time = datetime.now()
    
    try:
        if async_processing:
            # Submit as background task
            task_manager = await get_task_manager()
            task_id = await task_manager.submit_citation_analysis(
                citing_paper_id=request.citing_paper_id,
                cited_paper_id=request.cited_paper_id,
                citation_context=request.citation_context,
                force_refresh=request.force_refresh
            )
            
            return JSONResponse(
                status_code=status.HTTP_202_ACCEPTED,
                content={
                    "status": "accepted",
                    "message": "Citation analysis submitted for background processing",
                    "task_id": task_id,
                    "citing_paper_id": request.citing_paper_id,
                    "cited_paper_id": request.cited_paper_id
                }
            )
        else:
            # Process synchronously
            citation_service = await get_citation_analysis_service()
            
            relationship = await citation_service.analyze_citation_relationship(
                citing_paper_id=request.citing_paper_id,
                cited_paper_id=request.cited_paper_id,
                citation_context=request.citation_context,
                force_refresh=request.force_refresh
            )
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return CitationAnalysisResponse(
                status="completed",
                citing_paper_id=relationship.citing_paper_id,
                cited_paper_id=relationship.cited_paper_id,
                citation_purpose=relationship.citation_purpose,
                intellectual_relationship=relationship.intellectual_relationship,
                knowledge_flow_description=relationship.knowledge_flow_description,
                impact_assessment=relationship.impact_assessment,
                citation_type=relationship.citation_type.value if relationship.citation_type else None,
                influence_level=relationship.influence_level.value if relationship.influence_level else None,
                analysis_confidence=relationship.analysis_confidence,
                analysis_cost=relationship.analysis_cost,
                processing_time_ms=processing_time
            )
    
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Citation analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/research/lineage/trace",
             summary="Trace Intellectual Lineage",
             description="Trace the intellectual lineage and evolution of research ideas")
async def trace_research_lineage(
    request: LineageTracingRequest,
    settings = Depends(get_settings_dep)
):
    """Trace intellectual lineage from seed papers."""
    
    try:
        if request.async_processing:
            # Submit as background task
            task_manager = await get_task_manager()
            task_id = await task_manager.submit_lineage_tracing(
                seed_paper_ids=request.seed_paper_ids,
                max_generations=request.max_generations,
                include_future_work=request.include_future_work,
                generate_timeline=request.generate_timeline
            )
            
            return JSONResponse(
                status_code=status.HTTP_202_ACCEPTED,
                content={
                    "status": "accepted",
                    "message": "Lineage tracing submitted for background processing",
                    "task_id": task_id,
                    "seed_papers": request.seed_paper_ids,
                    "max_generations": request.max_generations,
                    "generate_timeline": request.generate_timeline
                }
            )
        else:
            # Process synchronously (limited scope)
            if len(request.seed_paper_ids) > 3 or request.max_generations > 2:
                raise HTTPException(
                    status_code=400,
                    detail="Synchronous lineage tracing limited to 3 seed papers and 2 generations. Use async_processing=true for larger scope."
                )
            
            trajectory_service = await get_trajectory_service()
            
            lineage = await trajectory_service.trace_intellectual_lineage(
                seed_paper_ids=request.seed_paper_ids,
                max_generations=request.max_generations,
                include_future_work=request.include_future_work
            )
            
            result = {
                "status": "completed",
                "lineage_id": lineage.lineage_id,
                "total_papers": lineage.total_papers,
                "time_span_years": lineage.time_span_years,
                "generation_count": lineage.generation_count,
                "trajectory_type": lineage.trajectory_type.value if lineage.trajectory_type else None,
                "milestone_count": len(lineage.milestones),
                "key_researchers": lineage.key_researchers,
                "dominant_venues": lineage.dominant_venues,
                "intellectual_evolution": lineage.intellectual_evolution,
                "key_insights": lineage.key_insights,
                "future_directions": lineage.future_directions,
                "analysis_confidence": lineage.analysis_confidence,
                "analysis_cost": lineage.analysis_cost
            }
            
            # Generate timeline if requested
            if request.generate_timeline and lineage.milestones:
                try:
                    timeline = await trajectory_service.generate_timeline_narrative(
                        lineage=lineage,
                        include_context=True
                    )
                    result["timeline"] = {
                        "timeline_id": timeline.timeline_id,
                        "domain": timeline.domain,
                        "start_year": timeline.start_year,
                        "end_year": timeline.end_year,
                        "periods": timeline.periods,
                        "breakthrough_moments": timeline.breakthrough_moments,
                        "timeline_narrative": timeline.timeline_narrative,
                        "analysis_confidence": timeline.analysis_confidence
                    }
                except Exception as e:
                    logger.warning(f"Timeline generation failed: {e}")
                    result["timeline_error"] = str(e)
            
            return result
    
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Lineage tracing failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/tasks/{task_id}/status",
            response_model=TaskStatusResponse,
            summary="Get Task Status",
            description="Get the status and result of a background task")
async def get_task_status(
    task_id: str = Path(..., description="Task identifier"),
    settings = Depends(get_settings_dep)
):
    """Get status of a background task."""
    
    try:
        task_manager = await get_task_manager()
        task_info = await task_manager.get_task_status(task_id)
        
        return TaskStatusResponse(**task_info)
    
    except Exception as e:
        logger.error(f"Failed to get task status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve task status")


@router.get("/tasks/active",
            summary="Get Active Tasks",
            description="Get list of all active background tasks")
async def get_active_tasks(
    limit: int = Query(50, ge=1, le=200, description="Maximum number of tasks to return"),
    settings = Depends(get_settings_dep)
):
    """Get list of active background tasks."""
    
    try:
        task_manager = await get_task_manager()
        active_tasks = await task_manager.get_active_tasks()
        
        # Limit results
        limited_tasks = active_tasks[:limit]
        
        return {
            "total_tasks": len(active_tasks),
            "returned_tasks": len(limited_tasks),
            "tasks": limited_tasks
        }
    
    except Exception as e:
        logger.error(f"Failed to get active tasks: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve active tasks")


@router.get("/analytics/costs",
            summary="Get Cost Analytics",
            description="Get comprehensive cost analytics and usage statistics")
async def get_cost_analytics(
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    settings = Depends(get_settings_dep)
):
    """Get LLM cost analytics and usage statistics."""
    
    try:
        cost_manager = await get_cost_manager()
        
        # Get analytics for specified period
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        analytics = await cost_manager.get_cost_analytics(start_date, end_date)
        
        # Get current budget status
        budget_status = cost_manager.get_budget_status()
        
        # Get optimization recommendations
        recommendations = await cost_manager.get_cost_optimization_recommendations()
        
        return {
            "period": {
                "start_date": analytics.period_start.isoformat(),
                "end_date": analytics.period_end.isoformat(),
                "days": days
            },
            "costs": {
                "total_cost": analytics.total_cost,
                "average_cost_per_request": analytics.avg_cost_per_request,
                "cost_by_category": analytics.cost_by_category,
                "cost_by_model": analytics.cost_by_model,
                "cost_by_provider": analytics.cost_by_provider
            },
            "usage": {
                "total_requests": analytics.total_requests,
                "token_usage": analytics.token_usage,
                "cache_hit_rate": analytics.cache_hit_rate,
                "cost_savings_from_cache": analytics.cost_savings_from_cache
            },
            "budget": budget_status,
            "optimization_recommendations": recommendations,
            "trends": analytics.usage_trends
        }
    
    except Exception as e:
        logger.error(f"Failed to get cost analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve cost analytics")


@router.get("/analytics/cache",
            summary="Get Cache Analytics",
            description="Get cache performance statistics and metrics")
async def get_cache_analytics(
    settings = Depends(get_settings_dep)
):
    """Get cache performance analytics."""
    
    try:
        cache_manager = await get_cache_manager()
        cache_stats = await cache_manager.get_cache_stats()
        
        return {
            "performance": {
                "hit_rate": cache_stats.hit_rate,
                "semantic_hit_rate": cache_stats.semantic_hit_rate,
                "total_requests": cache_stats.total_requests,
                "cache_hits": cache_stats.cache_hits,
                "cache_misses": cache_stats.cache_misses,
                "avg_response_time_ms": cache_stats.avg_response_time_ms
            },
            "storage": {
                "total_cache_entries": cache_stats.total_cache_entries,
                "cache_size_mb": cache_stats.cache_size_mb,
                "cost_savings": cache_stats.cost_savings
            },
            "distribution": {
                "exact_hits": cache_stats.exact_hits,
                "semantic_hits": cache_stats.semantic_hits,
                "top_cached_models": cache_stats.top_cached_models
            }
        }
    
    except Exception as e:
        logger.error(f"Failed to get cache analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve cache analytics")


@router.delete("/cache",
               summary="Invalidate Cache",
               description="Invalidate LLM response cache entries")
async def invalidate_cache(
    pattern: Optional[str] = Query(None, description="Cache key pattern to invalidate (optional)"),
    settings = Depends(get_settings_dep)
):
    """Invalidate cache entries."""
    
    try:
        cache_manager = await get_cache_manager()
        await cache_manager.invalidate_cache(pattern=pattern)
        
        return {
            "status": "success",
            "message": f"Cache invalidated{'for pattern: ' + pattern if pattern else ' (all entries)'}"
        }
    
    except Exception as e:
        logger.error(f"Failed to invalidate cache: {e}")
        raise HTTPException(status_code=500, detail="Failed to invalidate cache")


@router.get("/health",
            summary="Health Check",
            description="Check health status of LLM services")
async def health_check(
    settings = Depends(get_settings_dep)
):
    """Perform health check on LLM services."""
    
    try:
        # Check individual services
        from ....services.llm_service import get_llm_service
        
        llm_service = await get_llm_service()
        health_result = await llm_service.health_check()
        
        # Get usage stats
        usage_stats = llm_service.get_usage_stats()
        
        # Get cost manager status
        cost_manager = await get_cost_manager()
        daily_cost = await cost_manager.get_daily_cost()
        monthly_cost = await cost_manager.get_monthly_cost()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": health_result,
            "usage": usage_stats,
            "costs": {
                "daily_cost": daily_cost,
                "monthly_cost": monthly_cost,
                "daily_budget_limit": settings.llm.daily_budget_limit,
                "monthly_budget_limit": settings.llm.monthly_budget_limit
            },
            "configuration": {
                "cost_tracking_enabled": settings.llm.enable_cost_tracking,
                "semantic_caching_enabled": settings.llm.enable_semantic_caching,
                "local_fallback_enabled": settings.llm.enable_local_fallback
            }
        }
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )