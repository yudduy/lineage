"""
Semantic Scholar API endpoints.

FastAPI endpoints for accessing Semantic Scholar data with advanced semantic
features including citation intent analysis, influential citations, semantic
similarity, and research trend identification.
"""

from typing import Any, Dict, List, Optional, Union
from fastapi import APIRouter, Depends, HTTPException, Query, Path, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ....services.semantic_scholar import SemanticScholarClient, get_semantic_scholar_client
from ....services.semantic_analysis import SemanticAnalysisService, get_semantic_analysis_service
from ....services.enrichment_pipeline import (
    EnrichmentPipeline, 
    get_enrichment_pipeline, 
    EnrichmentPriority,
    EnrichmentTask
)
from ....models.semantic_scholar import (
    SemanticScholarPaper,
    SemanticScholarPapersResponse,
    SemanticScholarCitationNetwork,
    SemanticScholarSimilarityResult,
    SemanticScholarInfluentialCitation,
    SemanticScholarSearchFilters,
    EnrichedPaper,
    CitationIntent
)
from ....db.redis import get_redis_manager
from ....utils.logger import get_logger
from ....utils.exceptions import APIError, ValidationError

logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/semantic-scholar", tags=["Semantic Scholar"])


# Request/Response models
class PaperEnrichmentRequest(BaseModel):
    """Request model for paper enrichment."""
    
    paper_identifier: str = Field(description="Paper DOI, OpenAlex ID, or Semantic Scholar ID")
    priority: EnrichmentPriority = Field(default=EnrichmentPriority.MEDIUM, description="Task priority")
    include_citations: bool = Field(default=True, description="Include citation analysis")
    include_references: bool = Field(default=True, description="Include reference analysis")
    include_embeddings: bool = Field(default=True, description="Include semantic embeddings")
    include_semantic_analysis: bool = Field(default=True, description="Include advanced semantic analysis")
    max_citations: int = Field(default=100, ge=1, le=1000, description="Maximum citations to analyze")
    max_references: int = Field(default=100, ge=1, le=1000, description="Maximum references to analyze")


class BatchEnrichmentRequest(BaseModel):
    """Request model for batch paper enrichment."""
    
    paper_identifiers: List[str] = Field(description="List of paper identifiers", min_items=1, max_items=50)
    priority: EnrichmentPriority = Field(default=EnrichmentPriority.MEDIUM, description="Task priority")
    include_citations: bool = Field(default=True, description="Include citation analysis")
    include_references: bool = Field(default=True, description="Include reference analysis")
    include_embeddings: bool = Field(default=True, description="Include semantic embeddings")
    include_semantic_analysis: bool = Field(default=True, description="Include advanced semantic analysis")
    max_citations: int = Field(default=50, ge=1, le=500, description="Maximum citations to analyze per paper")
    max_references: int = Field(default=50, ge=1, le=500, description="Maximum references to analyze per paper")


class SimilarityAnalysisRequest(BaseModel):
    """Request model for semantic similarity analysis."""
    
    paper_id: str = Field(description="Reference paper identifier")
    candidate_papers: List[str] = Field(description="Candidate paper identifiers", max_items=100)
    similarity_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum similarity threshold")


class ResearchTrajectoryRequest(BaseModel):
    """Request model for research trajectory analysis."""
    
    author_papers: List[str] = Field(description="List of author's paper identifiers", min_items=3, max_items=50)
    time_window_years: int = Field(default=5, ge=1, le=20, description="Time window for analysis")


class TrendAnalysisRequest(BaseModel):
    """Request model for trend analysis."""
    
    field_of_study: str = Field(description="Research field to analyze")
    time_window_months: int = Field(default=12, ge=3, le=60, description="Time window in months")
    min_papers: int = Field(default=50, ge=10, le=1000, description="Minimum papers required")


# Basic Semantic Scholar API endpoints

@router.get("/paper/{paper_id}", response_model=SemanticScholarPaper)
async def get_paper(
    paper_id: str = Path(description="Paper ID, DOI, or other identifier"),
    include_embeddings: bool = Query(default=False, description="Include SPECTER embeddings"),
    client: SemanticScholarClient = Depends(get_semantic_scholar_client)
):
    """
    Get paper information from Semantic Scholar.
    
    Retrieves comprehensive paper data including metadata, citations, references,
    and optionally semantic embeddings.
    """
    try:
        fields = client.PAPER_FIELDS.copy()
        if include_embeddings:
            fields.append("embedding")
        
        paper = await client.get_paper_by_id(
            paper_id,
            fields=fields,
            use_cache=True
        )
        
        if not paper:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Paper not found: {paper_id}"
            )
        
        return paper
    
    except APIError as e:
        raise HTTPException(
            status_code=e.status_code or status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/papers/batch", response_model=List[SemanticScholarPaper])
async def get_papers_batch(
    paper_ids: List[str] = Field(description="List of paper identifiers", max_items=500),
    include_embeddings: bool = Query(default=False, description="Include SPECTER embeddings"),
    client: SemanticScholarClient = Depends(get_semantic_scholar_client)
):
    """
    Get multiple papers in a single batch request.
    
    Efficiently retrieves data for up to 500 papers simultaneously.
    """
    try:
        fields = client.PAPER_FIELDS.copy()
        if include_embeddings:
            fields.append("embedding")
        
        papers = await client.get_papers_batch(
            paper_ids,
            fields=fields,
            use_cache=True
        )
        
        return papers
    
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except APIError as e:
        raise HTTPException(
            status_code=e.status_code or status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/search", response_model=SemanticScholarPapersResponse)
async def search_papers(
    query: str = Query(description="Search query"),
    limit: int = Query(default=10, ge=1, le=100, description="Maximum results"),
    offset: int = Query(default=0, ge=0, description="Result offset"),
    year: Optional[str] = Query(default=None, description="Publication year filter"),
    venue: Optional[str] = Query(default=None, description="Venue filter"),
    fields_of_study: Optional[str] = Query(default=None, description="Fields of study (comma-separated)"),
    min_citation_count: Optional[int] = Query(default=None, ge=0, description="Minimum citations"),
    open_access_pdf: Optional[bool] = Query(default=None, description="Open access PDF filter"),
    client: SemanticScholarClient = Depends(get_semantic_scholar_client)
):
    """
    Search for papers using Semantic Scholar's search API.
    
    Supports advanced filtering by year, venue, field of study, citation count,
    and open access availability.
    """
    try:
        # Build search filters
        filters = SemanticScholarSearchFilters()
        
        if year:
            filters.year = year
        if venue:
            filters.venue = venue
        if fields_of_study:
            filters.fields_of_study = [f.strip() for f in fields_of_study.split(",")]
        if min_citation_count is not None:
            filters.min_citation_count = min_citation_count
        if open_access_pdf is not None:
            filters.open_access_pdf = open_access_pdf
        
        results = await client.search_papers(
            query=query,
            filters=filters,
            limit=limit,
            offset=offset,
            use_cache=True
        )
        
        return results
    
    except APIError as e:
        raise HTTPException(
            status_code=e.status_code or status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/paper/{paper_id}/citations", response_model=List[SemanticScholarPaper])
async def get_paper_citations(
    paper_id: str = Path(description="Paper identifier"),
    limit: int = Query(default=50, ge=1, le=1000, description="Maximum citations"),
    offset: int = Query(default=0, ge=0, description="Result offset"),
    client: SemanticScholarClient = Depends(get_semantic_scholar_client)
):
    """
    Get papers that cite the specified paper.
    
    Includes citation context and intent information where available.
    """
    try:
        citations = await client.get_paper_citations(
            paper_id,
            limit=limit,
            offset=offset,
            use_cache=True
        )
        
        return citations
    
    except APIError as e:
        raise HTTPException(
            status_code=e.status_code or status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/paper/{paper_id}/references", response_model=List[SemanticScholarPaper])
async def get_paper_references(
    paper_id: str = Path(description="Paper identifier"),
    limit: int = Query(default=50, ge=1, le=1000, description="Maximum references"),
    offset: int = Query(default=0, ge=0, description="Result offset"),
    client: SemanticScholarClient = Depends(get_semantic_scholar_client)
):
    """
    Get papers referenced by the specified paper.
    
    Includes citation context and intent information where available.
    """
    try:
        references = await client.get_paper_references(
            paper_id,
            limit=limit,
            offset=offset,
            use_cache=True
        )
        
        return references
    
    except APIError as e:
        raise HTTPException(
            status_code=e.status_code or status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# Advanced semantic features endpoints

@router.get("/paper/{paper_id}/influential-citations", response_model=List[SemanticScholarInfluentialCitation])
async def get_influential_citations(
    paper_id: str = Path(description="Paper identifier"),
    limit: int = Query(default=20, ge=1, le=100, description="Maximum influential citations"),
    client: SemanticScholarClient = Depends(get_semantic_scholar_client)
):
    """
    Get influential citations for a paper.
    
    Returns citations that have been identified as particularly impactful
    using machine learning models.
    """
    try:
        influential_citations = await client.get_influential_citations(
            paper_id,
            limit=limit,
            use_cache=True
        )
        
        return influential_citations
    
    except APIError as e:
        raise HTTPException(
            status_code=e.status_code or status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/similarity/analyze", response_model=List[SemanticScholarSimilarityResult])
async def analyze_semantic_similarity(
    request: SimilarityAnalysisRequest,
    client: SemanticScholarClient = Depends(get_semantic_scholar_client)
):
    """
    Analyze semantic similarity between papers using SPECTER embeddings.
    
    Computes cosine similarity between paper embeddings to identify
    semantically related research.
    """
    try:
        similarity_results = await client.find_similar_papers(
            request.paper_id,
            request.candidate_papers,
            similarity_threshold=request.similarity_threshold,
            use_cache=True
        )
        
        return similarity_results
    
    except APIError as e:
        raise HTTPException(
            status_code=e.status_code or status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/paper/{paper_id}/embedding", response_model=Dict[str, Any])
async def get_paper_embedding(
    paper_id: str = Path(description="Paper identifier"),
    client: SemanticScholarClient = Depends(get_semantic_scholar_client)
):
    """
    Get SPECTER embedding for a paper.
    
    Returns the 768-dimensional semantic embedding vector that can be used
    for similarity computation and clustering.
    """
    try:
        embedding = await client.get_paper_embedding(
            paper_id,
            use_cache=True
        )
        
        if not embedding:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No embedding available for paper: {paper_id}"
            )
        
        return embedding.dict()
    
    except APIError as e:
        raise HTTPException(
            status_code=e.status_code or status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/network/build", response_model=SemanticScholarCitationNetwork)
async def build_semantic_citation_network(
    paper_id: str = Field(description="Central paper identifier"),
    max_depth: int = Field(default=2, ge=1, le=3, description="Maximum network depth"),
    max_papers_per_level: int = Field(default=20, ge=5, le=50, description="Maximum papers per level"),
    similarity_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Similarity threshold"),
    client: SemanticScholarClient = Depends(get_semantic_scholar_client)
):
    """
    Build a semantically-enriched citation network.
    
    Creates a citation network with semantic similarity scores, citation intent
    classification, and influential citation detection.
    """
    try:
        network = await client.build_semantic_citation_network(
            center_paper_id=paper_id,
            max_depth=max_depth,
            max_papers_per_level=max_papers_per_level,
            similarity_threshold=similarity_threshold,
            use_cache=True
        )
        
        return network
    
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except APIError as e:
        raise HTTPException(
            status_code=e.status_code or status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# Paper enrichment endpoints

@router.post("/enrich/paper", response_model=Dict[str, str])
async def enrich_paper(
    request: PaperEnrichmentRequest,
    pipeline: EnrichmentPipeline = Depends(get_enrichment_pipeline)
):
    """
    Submit a paper for comprehensive enrichment.
    
    Combines data from multiple sources and performs advanced semantic analysis
    including citation intent classification and influence detection.
    Returns a task ID for tracking progress.
    """
    try:
        task_id = await pipeline.enrich_paper(
            paper_identifier=request.paper_identifier,
            priority=request.priority,
            include_citations=request.include_citations,
            include_references=request.include_references,
            include_embeddings=request.include_embeddings,
            include_semantic_analysis=request.include_semantic_analysis,
            max_citations=request.max_citations,
            max_references=request.max_references
        )
        
        return {"task_id": task_id, "status": "queued"}
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to queue enrichment task: {str(e)}"
        )


@router.post("/enrich/batch", response_model=Dict[str, Any])
async def enrich_papers_batch(
    request: BatchEnrichmentRequest,
    pipeline: EnrichmentPipeline = Depends(get_enrichment_pipeline)
):
    """
    Submit multiple papers for batch enrichment.
    
    Efficiently processes multiple papers with shared configuration.
    Returns list of task IDs for tracking progress.
    """
    try:
        task_ids = await pipeline.enrich_papers_batch(
            paper_identifiers=request.paper_identifiers,
            priority=request.priority,
            include_citations=request.include_citations,
            include_references=request.include_references,
            include_embeddings=request.include_embeddings,
            include_semantic_analysis=request.include_semantic_analysis,
            max_citations=request.max_citations,
            max_references=request.max_references
        )
        
        return {
            "task_ids": task_ids,
            "count": len(task_ids),
            "status": "queued"
        }
    
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to queue batch enrichment: {str(e)}"
        )


@router.get("/enrich/status/{task_id}", response_model=Dict[str, Any])
async def get_enrichment_status(
    task_id: str = Path(description="Enrichment task ID"),
    pipeline: EnrichmentPipeline = Depends(get_enrichment_pipeline)
):
    """
    Get status of an enrichment task.
    
    Returns current status, progress information, and any error details.
    """
    try:
        task = await pipeline.get_task_status(task_id)
        
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task not found: {task_id}"
            )
        
        return task.to_dict()
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get task status: {str(e)}"
        )


@router.get("/enrich/result/{paper_identifier}", response_model=EnrichedPaper)
async def get_enriched_paper(
    paper_identifier: str = Path(description="Paper identifier"),
    identifier_type: str = Query(default="auto", description="Identifier type (auto, doi, openalex, s2)"),
    pipeline: EnrichmentPipeline = Depends(get_enrichment_pipeline)
):
    """
    Get enriched paper data.
    
    Retrieves comprehensive enriched paper data combining multiple sources
    with advanced semantic analysis results.
    """
    try:
        enriched_paper = await pipeline.get_enriched_paper(
            paper_identifier,
            identifier_type=identifier_type
        )
        
        if not enriched_paper:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Enriched paper not found: {paper_identifier}"
            )
        
        return enriched_paper
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get enriched paper: {str(e)}"
        )


# Advanced analysis endpoints

@router.post("/analysis/research-trajectory", response_model=Dict[str, Any])
async def analyze_research_trajectory(
    request: ResearchTrajectoryRequest,
    service: SemanticAnalysisService = Depends(get_semantic_analysis_service)
):
    """
    Analyze research trajectory using semantic embeddings.
    
    Examines how an author's research focus evolves over time using
    semantic similarity analysis and field classification.
    """
    try:
        analysis = await service.analyze_research_trajectory(
            author_papers=request.author_papers,
            time_window_years=request.time_window_years,
            use_cache=True
        )
        
        return analysis
    
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Research trajectory analysis failed: {str(e)}"
        )


@router.post("/analysis/emerging-trends", response_model=Dict[str, Any])
async def identify_emerging_trends(
    request: TrendAnalysisRequest,
    service: SemanticAnalysisService = Depends(get_semantic_analysis_service)
):
    """
    Identify emerging research trends in a field.
    
    Uses semantic clustering and citation analysis to identify emerging
    research directions and hot topics in a specific field of study.
    """
    try:
        trends = await service.identify_emerging_research_trends(
            field_of_study=request.field_of_study,
            time_window_months=request.time_window_months,
            min_papers=request.min_papers,
            use_cache=True
        )
        
        return trends
    
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Trend analysis failed: {str(e)}"
        )


# Background task endpoints

@router.post("/tasks/enrich/paper", response_model=Dict[str, Any])
async def enrich_paper_async(
    request: PaperEnrichmentRequest
):
    """
    Submit a paper for asynchronous enrichment using Celery.
    
    Returns a Celery task ID that can be used to track progress.
    """
    try:
        from ....services.semantic_scholar_tasks import enrich_paper_task
        
        task = enrich_paper_task.delay(
            paper_identifier=request.paper_identifier,
            priority=request.priority.value,
            include_citations=request.include_citations,
            include_references=request.include_references,
            include_embeddings=request.include_embeddings,
            include_semantic_analysis=request.include_semantic_analysis,
            max_citations=request.max_citations,
            max_references=request.max_references
        )
        
        return {
            "task_id": task.id,
            "status": "queued",
            "task_type": "paper_enrichment"
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to queue async enrichment task: {str(e)}"
        )


@router.post("/tasks/enrich/batch", response_model=Dict[str, Any])
async def enrich_papers_batch_async(
    request: BatchEnrichmentRequest
):
    """
    Submit multiple papers for asynchronous batch enrichment using Celery.
    
    Returns a Celery task ID for tracking the batch operation.
    """
    try:
        from ....services.semantic_scholar_tasks import enrich_papers_batch_task
        
        task = enrich_papers_batch_task.delay(
            paper_identifiers=request.paper_identifiers,
            priority=request.priority.value,
            include_citations=request.include_citations,
            include_references=request.include_references,
            include_embeddings=request.include_embeddings,
            include_semantic_analysis=request.include_semantic_analysis,
            max_citations=request.max_citations,
            max_references=request.max_references
        )
        
        return {
            "task_id": task.id,
            "status": "queued",
            "task_type": "batch_enrichment",
            "paper_count": len(request.paper_identifiers)
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to queue batch enrichment task: {str(e)}"
        )


@router.post("/tasks/analysis/trajectory", response_model=Dict[str, Any])
async def analyze_research_trajectory_async(
    request: ResearchTrajectoryRequest
):
    """
    Submit research trajectory analysis as asynchronous Celery task.
    
    Returns a task ID for tracking the analysis progress.
    """
    try:
        from ....services.semantic_scholar_tasks import analyze_research_trajectory_task
        
        task = analyze_research_trajectory_task.delay(
            author_papers=request.author_papers,
            time_window_years=request.time_window_years
        )
        
        return {
            "task_id": task.id,
            "status": "queued",
            "task_type": "research_trajectory_analysis",
            "paper_count": len(request.author_papers)
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to queue trajectory analysis task: {str(e)}"
        )


@router.post("/tasks/analysis/trends", response_model=Dict[str, Any])
async def identify_emerging_trends_async(
    request: TrendAnalysisRequest
):
    """
    Submit emerging trends analysis as asynchronous Celery task.
    
    Returns a task ID for tracking the analysis progress.
    """
    try:
        from ....services.semantic_scholar_tasks import identify_emerging_trends_task
        
        task = identify_emerging_trends_task.delay(
            field_of_study=request.field_of_study,
            time_window_months=request.time_window_months,
            min_papers=request.min_papers
        )
        
        return {
            "task_id": task.id,
            "status": "queued",
            "task_type": "trend_analysis",
            "field_of_study": request.field_of_study
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to queue trend analysis task: {str(e)}"
        )


@router.post("/tasks/network/build", response_model=Dict[str, Any])
async def build_citation_network_async(
    paper_id: str = Field(description="Central paper identifier"),
    max_depth: int = Field(default=2, ge=1, le=3, description="Maximum network depth"),
    max_papers_per_level: int = Field(default=20, ge=5, le=50, description="Maximum papers per level"),
    similarity_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Similarity threshold")
):
    """
    Submit citation network building as asynchronous Celery task.
    
    Returns a task ID for tracking the network building progress.
    """
    try:
        from ....services.semantic_scholar_tasks import build_citation_network_task
        
        task = build_citation_network_task.delay(
            center_paper_id=paper_id,
            max_depth=max_depth,
            max_papers_per_level=max_papers_per_level,
            similarity_threshold=similarity_threshold
        )
        
        return {
            "task_id": task.id,
            "status": "queued",
            "task_type": "citation_network_building",
            "center_paper_id": paper_id
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to queue network building task: {str(e)}"
        )


@router.get("/tasks/{task_id}/status", response_model=Dict[str, Any])
async def get_task_status(
    task_id: str = Path(description="Celery task ID")
):
    """
    Get status of a Celery background task.
    
    Returns current status, progress information, and results if completed.
    """
    try:
        from ....services.semantic_scholar_tasks import celery_app
        
        # Get task result
        result = celery_app.AsyncResult(task_id)
        
        response = {
            "task_id": task_id,
            "status": result.status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if result.status == "PENDING":
            response["message"] = "Task is waiting to be processed"
        elif result.status == "PROGRESS":
            response["progress"] = result.info
        elif result.status == "SUCCESS":
            response["result"] = result.result
        elif result.status == "FAILURE":
            response["error"] = str(result.info)
        
        return response
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get task status: {str(e)}"
        )


@router.delete("/tasks/{task_id}/cancel", response_model=Dict[str, Any])
async def cancel_task(
    task_id: str = Path(description="Celery task ID")
):
    """
    Cancel a running Celery background task.
    
    Attempts to revoke the task if it hasn't started or is still running.
    """
    try:
        from ....services.semantic_scholar_tasks import celery_app
        
        # Revoke the task
        celery_app.control.revoke(task_id, terminate=True)
        
        return {
            "task_id": task_id,
            "status": "cancelled",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel task: {str(e)}"
        )


# System status endpoints

@router.get("/status/queue", response_model=Dict[str, Any])
async def get_queue_status(
    pipeline: EnrichmentPipeline = Depends(get_enrichment_pipeline)
):
    """
    Get current enrichment queue status.
    
    Returns information about active tasks, queue size, and processing metrics.
    """
    try:
        status = await pipeline.get_queue_status()
        return status
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get queue status: {str(e)}"
        )


@router.get("/status/celery", response_model=Dict[str, Any])
async def get_celery_status():
    """
    Get Celery worker and queue status.
    
    Returns information about active workers, queued tasks, and system health.
    """
    try:
        from ....services.semantic_scholar_tasks import celery_app
        
        # Get worker stats
        inspect = celery_app.control.inspect()
        
        active_tasks = inspect.active()
        scheduled_tasks = inspect.scheduled()
        registered_tasks = inspect.registered()
        worker_stats = inspect.stats()
        
        return {
            "status": "healthy" if active_tasks else "idle",
            "active_workers": len(active_tasks) if active_tasks else 0,
            "active_tasks": active_tasks,
            "scheduled_tasks": scheduled_tasks,
            "registered_tasks": registered_tasks,
            "worker_stats": worker_stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@router.get("/status/health", response_model=Dict[str, Any])
async def health_check(
    client: SemanticScholarClient = Depends(get_semantic_scholar_client),
    service: SemanticAnalysisService = Depends(get_semantic_analysis_service),
    pipeline: EnrichmentPipeline = Depends(get_enrichment_pipeline)
):
    """
    Comprehensive health check for Semantic Scholar services.
    
    Checks the status of all components including API client, analysis service,
    and enrichment pipeline.
    """
    try:
        # Check individual components
        client_health = await client.health_check()
        service_metrics = service.get_metrics()
        pipeline_health = await pipeline.health_check()
        
        # Determine overall status
        overall_status = "healthy"
        if (client_health.get("status") != "healthy" or 
            pipeline_health.get("status") in ["unhealthy", "degraded"]):
            overall_status = "degraded"
        
        return {
            "status": overall_status,
            "timestamp": pipeline.redis_manager and await pipeline.redis_manager.get_current_time(),
            "components": {
                "semantic_scholar_client": client_health,
                "semantic_analysis_service": service_metrics,
                "enrichment_pipeline": pipeline_health
            }
        }
    
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": None
        }


@router.get("/metrics", response_model=Dict[str, Any])
async def get_metrics(
    client: SemanticScholarClient = Depends(get_semantic_scholar_client),
    service: SemanticAnalysisService = Depends(get_semantic_analysis_service),
    pipeline: EnrichmentPipeline = Depends(get_enrichment_pipeline)
):
    """
    Get comprehensive metrics for Semantic Scholar services.
    
    Returns performance metrics, usage statistics, and operational data
    for monitoring and optimization.
    """
    try:
        return {
            "semantic_scholar_client": client.get_metrics(),
            "semantic_analysis_service": service.get_metrics(),
            "enrichment_pipeline": await pipeline.get_queue_status()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get metrics: {str(e)}"
        )