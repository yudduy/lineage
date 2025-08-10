"""
Content Enrichment Service - Intelligent paper analysis and summarization using LLMs.

This service provides:
- Comprehensive paper summarization with key contributions
- Abstract enhancement and clarification
- Methodology and results analysis
- Significance assessment and impact evaluation
- Content quality scoring and validation
- Batch processing for large-scale enrichment
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging

from ..models.paper import Paper
from ..models.semantic_scholar import SemanticScholarPaper
from ..models.openalex import OpenAlexWork
from ..services.llm_service import LLMService, get_llm_service, LLMResponse
from ..services.llm_prompts import PromptType, get_template_manager
from ..services.llm_cost_manager import CostCategory, get_cost_manager
from ..services.llm_cache import get_cache_manager
from ..db.redis import RedisManager, get_redis_manager
from ..db.neo4j import Neo4jManager, get_neo4j_manager
from ..utils.logger import get_logger
from ..utils.exceptions import ValidationError

logger = get_logger(__name__)


class EnrichmentStatus(Enum):
    """Status of content enrichment."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"


class ContentQuality(Enum):
    """Content quality levels."""
    EXCELLENT = "excellent"  # >90% confidence
    GOOD = "good"           # 70-90% confidence
    FAIR = "fair"           # 50-70% confidence
    POOR = "poor"           # <50% confidence


@dataclass
class EnrichedContent:
    """Enriched paper content with LLM analysis."""
    paper_id: str
    paper_title: str
    
    # Core enrichment
    enhanced_summary: Optional[str] = None
    key_contributions: Optional[List[str]] = None
    methodology_summary: Optional[str] = None
    key_findings: Optional[List[str]] = None
    significance_assessment: Optional[str] = None
    limitations: Optional[List[str]] = None
    future_directions: Optional[List[str]] = None
    
    # Quality and metadata
    content_quality: ContentQuality = ContentQuality.FAIR
    confidence_score: float = 0.0
    enrichment_model: Optional[str] = None
    enrichment_cost: float = 0.0
    enrichment_tokens: int = 0
    enrichment_timestamp: Optional[datetime] = None
    status: EnrichmentStatus = EnrichmentStatus.PENDING
    
    # Source attribution
    source_abstract: Optional[str] = None
    source_venue: Optional[str] = None
    source_year: Optional[int] = None
    source_authors: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['content_quality'] = self.content_quality.value
        data['status'] = self.status.value
        if self.enrichment_timestamp:
            data['enrichment_timestamp'] = self.enrichment_timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnrichedContent':
        """Create from dictionary."""
        if 'content_quality' in data:
            data['content_quality'] = ContentQuality(data['content_quality'])
        if 'status' in data:
            data['status'] = EnrichmentStatus(data['status'])
        if 'enrichment_timestamp' in data and data['enrichment_timestamp']:
            data['enrichment_timestamp'] = datetime.fromisoformat(data['enrichment_timestamp'])
        return cls(**data)


@dataclass
class EnrichmentRequest:
    """Request for content enrichment."""
    paper_id: str
    paper_data: Dict[str, Any]
    priority: int = 1  # 1=high, 2=medium, 3=low
    force_refresh: bool = False
    requested_by: Optional[str] = None
    request_timestamp: datetime = None
    
    def __post_init__(self):
        if self.request_timestamp is None:
            self.request_timestamp = datetime.now()


@dataclass
class BatchEnrichmentResult:
    """Result of batch enrichment operation."""
    total_papers: int
    successful_enrichments: int
    failed_enrichments: int
    cached_results: int
    total_cost: float
    total_tokens: int
    processing_time_seconds: float
    error_summary: List[str]
    enriched_papers: List[EnrichedContent]


class ContentEnrichmentService:
    """
    Service for intelligent content enrichment of research papers using LLMs.
    """
    
    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        redis_manager: Optional[RedisManager] = None,
        neo4j_manager: Optional[Neo4jManager] = None
    ):
        self.llm_service = llm_service
        self.redis_manager = redis_manager
        self.neo4j_manager = neo4j_manager
        self._template_manager = get_template_manager()
        self._initialized = False
        
        # Processing configuration
        self.max_concurrent_enrichments = 5
        self.enrichment_timeout_seconds = 300  # 5 minutes
        self.quality_threshold = 0.7
        
    async def initialize(self):
        """Initialize the content enrichment service."""
        if self._initialized:
            return
        
        if not self.llm_service:
            self.llm_service = await get_llm_service()
        
        if not self.redis_manager:
            self.redis_manager = await get_redis_manager()
        
        if not self.neo4j_manager:
            self.neo4j_manager = await get_neo4j_manager()
        
        self._initialized = True
        logger.info("Content Enrichment Service initialized")
    
    async def enrich_paper(
        self,
        paper_id: str,
        paper_data: Optional[Dict[str, Any]] = None,
        force_refresh: bool = False,
        use_cache: bool = True
    ) -> EnrichedContent:
        """
        Enrich a single paper with comprehensive LLM analysis.
        
        Args:
            paper_id: Unique identifier for the paper
            paper_data: Paper metadata (if not provided, will fetch from database)
            force_refresh: Force new analysis even if cached result exists
            use_cache: Whether to use LLM response caching
            
        Returns:
            EnrichedContent with comprehensive analysis
        """
        await self.initialize()
        
        # Check for existing enrichment first
        if not force_refresh:
            cached_enrichment = await self._get_cached_enrichment(paper_id)
            if cached_enrichment and not self._is_enrichment_stale(cached_enrichment):
                logger.debug(f"Using cached enrichment for paper {paper_id}")
                return cached_enrichment
        
        # Get or validate paper data
        if not paper_data:
            paper_data = await self._fetch_paper_data(paper_id)
        
        if not paper_data:
            raise ValidationError(f"Could not find paper data for {paper_id}")
        
        # Validate paper has sufficient content for enrichment
        if not self._validate_paper_for_enrichment(paper_data):
            raise ValidationError(f"Paper {paper_id} lacks sufficient content for enrichment")
        
        # Create enrichment request
        enrichment = EnrichedContent(
            paper_id=paper_id,
            paper_title=paper_data.get('title', 'Unknown Title'),
            status=EnrichmentStatus.IN_PROGRESS,
            source_abstract=paper_data.get('abstract'),
            source_venue=paper_data.get('venue', {}).get('name') if paper_data.get('venue') else None,
            source_year=paper_data.get('year'),
            source_authors=[author.get('name', '') for author in paper_data.get('authors', [])],
            enrichment_timestamp=datetime.now()
        )
        
        try:
            # Generate comprehensive summary
            await self._enrich_with_summary_analysis(enrichment, paper_data, use_cache)
            
            # Assess content quality
            self._assess_content_quality(enrichment)
            
            # Mark as completed
            enrichment.status = EnrichmentStatus.COMPLETED
            
            # Cache the enriched content
            await self._cache_enriched_content(enrichment)
            
            # Store in Neo4j for long-term persistence
            await self._store_enrichment_in_neo4j(enrichment)
            
            logger.info(f"Successfully enriched paper {paper_id} with quality {enrichment.content_quality.value}")
            
            return enrichment
            
        except Exception as e:
            enrichment.status = EnrichmentStatus.FAILED
            await self._cache_enriched_content(enrichment)  # Cache failure to avoid retries
            
            logger.error(f"Failed to enrich paper {paper_id}: {e}")
            raise
    
    async def _enrich_with_summary_analysis(
        self,
        enrichment: EnrichedContent,
        paper_data: Dict[str, Any],
        use_cache: bool = True
    ):
        """Perform comprehensive summary analysis using LLM."""
        # Get paper summary template
        template = self._template_manager.get_template(PromptType.PAPER_SUMMARY)
        
        # Prepare template variables
        template_vars = {
            'title': paper_data.get('title', ''),
            'abstract': paper_data.get('abstract', ''),
            'year': paper_data.get('year', 'Unknown'),
            'authors': ', '.join([author.get('name', '') for author in paper_data.get('authors', [])]),
            'venue': paper_data.get('venue', {}).get('name', 'Unknown') if paper_data.get('venue') else 'Unknown',
            'additional_content': self._prepare_additional_content(paper_data)
        }
        
        # Make LLM request
        response = await self.llm_service.complete(
            template,
            use_cache=use_cache,
            **template_vars
        )
        
        # Record cost
        cost_manager = await get_cost_manager()
        await cost_manager.record_usage(
            model=response.model,
            provider=response.provider,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            category=CostCategory.PAPER_ANALYSIS,
            cached=response.cached
        )
        
        # Update enrichment with cost info
        enrichment.enrichment_cost = response.usage.cost
        enrichment.enrichment_tokens = response.usage.total_tokens
        enrichment.enrichment_model = response.model
        
        # Parse the structured response
        await self._parse_summary_response(enrichment, response.content)
    
    def _prepare_additional_content(self, paper_data: Dict[str, Any]) -> str:
        """Prepare additional content for analysis."""
        additional = []
        
        # Add citation counts if available
        if paper_data.get('citation_count'):
            additional.append(f"**Citation Count:** {paper_data['citation_count']}")
        
        # Add fields of study
        fields = []
        if paper_data.get('s2_fields_of_study'):
            fields.extend([field.get('category', '') for field in paper_data['s2_fields_of_study']])
        if paper_data.get('fields_of_study'):
            fields.extend(paper_data['fields_of_study'])
        
        if fields:
            unique_fields = list(set(fields))
            additional.append(f"**Fields of Study:** {', '.join(unique_fields)}")
        
        # Add DOI if available
        if paper_data.get('doi'):
            additional.append(f"**DOI:** {paper_data['doi']}")
        
        return '\n'.join(additional)
    
    async def _parse_summary_response(self, enrichment: EnrichedContent, response_content: str):
        """Parse the structured LLM response into enrichment fields."""
        try:
            sections = self._split_response_into_sections(response_content)
            
            # Extract each section
            enrichment.enhanced_summary = sections.get('summary', '').strip()
            
            # Parse key contributions (bullet points)
            contributions_text = sections.get('key contributions', '')
            enrichment.key_contributions = self._parse_bullet_points(contributions_text)
            
            # Parse methodology
            enrichment.methodology_summary = sections.get('methodology', '').strip()
            
            # Parse key findings
            findings_text = sections.get('key findings', '')
            enrichment.key_findings = self._parse_bullet_points(findings_text)
            
            # Parse significance
            enrichment.significance_assessment = sections.get('impact and significance', '').strip()
            
            # Parse limitations
            limitations_text = sections.get('limitations', '')
            enrichment.limitations = self._parse_bullet_points(limitations_text)
            
            # Parse future directions
            future_text = sections.get('future directions', '')
            enrichment.future_directions = self._parse_bullet_points(future_text)
            
        except Exception as e:
            logger.warning(f"Failed to parse summary response structure, using raw content: {e}")
            # Fallback to raw content
            enrichment.enhanced_summary = response_content[:1000]  # Truncate if too long
    
    def _split_response_into_sections(self, content: str) -> Dict[str, str]:
        """Split LLM response into structured sections."""
        sections = {}
        current_section = None
        current_content = []
        
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Check if this is a section header (starts with ## or **section name**)
            if line.startswith('##') or (line.startswith('**') and line.endswith('**')):
                # Save previous section
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Start new section
                current_section = line.replace('##', '').replace('**', '').strip().lower()
                current_content = []
            else:
                if current_section and line:
                    current_content.append(line)
        
        # Save last section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def _parse_bullet_points(self, text: str) -> List[str]:
        """Parse bullet points from text."""
        if not text:
            return []
        
        bullet_points = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Remove common bullet point markers
            if line.startswith(('-', '•', '*', '+')):
                line = line[1:].strip()
            elif line.startswith(tuple(f"{i}." for i in range(1, 20))):
                # Remove numbered list markers
                line = line.split('.', 1)[1].strip() if '.' in line else line
            
            if line and len(line) > 10:  # Only include substantial points
                bullet_points.append(line)
        
        return bullet_points[:10]  # Limit to top 10 points
    
    def _assess_content_quality(self, enrichment: EnrichedContent):
        """Assess the quality of the enriched content."""
        quality_score = 0.0
        max_score = 7.0  # Total possible score
        
        # Check for presence of key sections
        if enrichment.enhanced_summary and len(enrichment.enhanced_summary) > 100:
            quality_score += 1.0
        
        if enrichment.key_contributions and len(enrichment.key_contributions) >= 2:
            quality_score += 1.0
        
        if enrichment.methodology_summary and len(enrichment.methodology_summary) > 50:
            quality_score += 1.0
        
        if enrichment.key_findings and len(enrichment.key_findings) >= 1:
            quality_score += 1.0
        
        if enrichment.significance_assessment and len(enrichment.significance_assessment) > 50:
            quality_score += 1.0
        
        if enrichment.limitations and len(enrichment.limitations) >= 1:
            quality_score += 1.0
        
        if enrichment.future_directions and len(enrichment.future_directions) >= 1:
            quality_score += 1.0
        
        # Calculate quality percentage
        quality_percentage = quality_score / max_score
        enrichment.confidence_score = quality_percentage
        
        # Assign quality level
        if quality_percentage >= 0.9:
            enrichment.content_quality = ContentQuality.EXCELLENT
        elif quality_percentage >= 0.7:
            enrichment.content_quality = ContentQuality.GOOD
        elif quality_percentage >= 0.5:
            enrichment.content_quality = ContentQuality.FAIR
        else:
            enrichment.content_quality = ContentQuality.POOR
    
    async def _fetch_paper_data(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Fetch paper data from available sources."""
        # This would integrate with existing data sources
        # For now, return None to indicate not found
        return None
    
    def _validate_paper_for_enrichment(self, paper_data: Dict[str, Any]) -> bool:
        """Validate that paper has sufficient content for enrichment."""
        # Must have title
        if not paper_data.get('title'):
            return False
        
        # Must have abstract or substantial other content
        abstract = paper_data.get('abstract', '')
        if len(abstract) < 100:  # Minimum abstract length
            return False
        
        return True
    
    async def _get_cached_enrichment(self, paper_id: str) -> Optional[EnrichedContent]:
        """Get cached enrichment if available."""
        if not self.redis_manager:
            return None
        
        try:
            cache_key = f"paper_enrichment:{paper_id}"
            cached_data = await self.redis_manager.get(cache_key)
            
            if cached_data:
                data = json.loads(cached_data)
                return EnrichedContent.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to retrieve cached enrichment: {e}")
        
        return None
    
    def _is_enrichment_stale(self, enrichment: EnrichedContent) -> bool:
        """Check if enrichment is stale and needs refresh."""
        if not enrichment.enrichment_timestamp:
            return True
        
        # Consider enrichment stale after 30 days
        stale_threshold = datetime.now() - timedelta(days=30)
        return enrichment.enrichment_timestamp < stale_threshold
    
    async def _cache_enriched_content(self, enrichment: EnrichedContent):
        """Cache enriched content in Redis."""
        if not self.redis_manager:
            return
        
        try:
            cache_key = f"paper_enrichment:{enrichment.paper_id}"
            ttl_seconds = 30 * 24 * 3600  # 30 days
            
            await self.redis_manager.setex(
                cache_key,
                ttl_seconds,
                json.dumps(enrichment.to_dict())
            )
        except Exception as e:
            logger.error(f"Failed to cache enriched content: {e}")
    
    async def _store_enrichment_in_neo4j(self, enrichment: EnrichedContent):
        """Store enriched content in Neo4j for long-term persistence."""
        if not self.neo4j_manager:
            return
        
        try:
            # This would store enriched content in Neo4j
            # Implementation would depend on the graph schema
            logger.debug(f"Would store enrichment for {enrichment.paper_id} in Neo4j")
        except Exception as e:
            logger.error(f"Failed to store enrichment in Neo4j: {e}")
    
    async def batch_enrich_papers(
        self,
        paper_requests: List[EnrichmentRequest],
        max_concurrency: Optional[int] = None
    ) -> BatchEnrichmentResult:
        """
        Enrich multiple papers in batch with controlled concurrency.
        
        Args:
            paper_requests: List of enrichment requests
            max_concurrency: Maximum concurrent enrichments
            
        Returns:
            BatchEnrichmentResult with comprehensive results
        """
        await self.initialize()
        
        if max_concurrency is None:
            max_concurrency = self.max_concurrent_enrichments
        
        start_time = datetime.now()
        semaphore = asyncio.Semaphore(max_concurrency)
        
        results = []
        errors = []
        total_cost = 0.0
        total_tokens = 0
        cached_count = 0
        
        async def process_request(request: EnrichmentRequest):
            async with semaphore:
                try:
                    enrichment = await self.enrich_paper(
                        request.paper_id,
                        request.paper_data,
                        request.force_refresh,
                        use_cache=True
                    )
                    
                    nonlocal total_cost, total_tokens, cached_count
                    total_cost += enrichment.enrichment_cost
                    total_tokens += enrichment.enrichment_tokens
                    
                    if enrichment.status == EnrichmentStatus.CACHED:
                        cached_count += 1
                    
                    return enrichment
                    
                except Exception as e:
                    error_msg = f"Failed to enrich paper {request.paper_id}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
                    return None
        
        # Process all requests
        logger.info(f"Starting batch enrichment of {len(paper_requests)} papers")
        
        tasks = [process_request(request) for request in paper_requests]
        completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect results
        for task_result in completed_tasks:
            if isinstance(task_result, Exception):
                errors.append(f"Task exception: {str(task_result)}")
            elif task_result is not None:
                results.append(task_result)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Create batch result
        batch_result = BatchEnrichmentResult(
            total_papers=len(paper_requests),
            successful_enrichments=len(results),
            failed_enrichments=len(errors),
            cached_results=cached_count,
            total_cost=total_cost,
            total_tokens=total_tokens,
            processing_time_seconds=processing_time,
            error_summary=errors,
            enriched_papers=results
        )
        
        logger.info(
            f"Batch enrichment completed: {batch_result.successful_enrichments}/{batch_result.total_papers} "
            f"successful, ${batch_result.total_cost:.4f} cost, {processing_time:.2f}s"
        )
        
        return batch_result
    
    async def get_enrichment_stats(self) -> Dict[str, Any]:
        """Get comprehensive enrichment statistics."""
        # This would query Redis and Neo4j for statistics
        # For now, return placeholder data
        return {
            'total_enriched_papers': 0,
            'enrichments_today': 0,
            'avg_enrichment_cost': 0.0,
            'quality_distribution': {
                'excellent': 0,
                'good': 0,
                'fair': 0,
                'poor': 0
            },
            'cache_hit_rate': 0.0,
            'avg_processing_time_seconds': 0.0
        }


# Global service instance
_enrichment_service: Optional[ContentEnrichmentService] = None


async def get_enrichment_service() -> ContentEnrichmentService:
    """Get or create the global content enrichment service."""
    global _enrichment_service
    
    if _enrichment_service is None:
        _enrichment_service = ContentEnrichmentService()
        await _enrichment_service.initialize()
    
    return _enrichment_service