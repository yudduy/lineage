"""
Citation Analysis Service - Intelligent analysis of citation relationships and contexts.

This service provides:
- Citation context analysis and explanation
- Citation intent classification and reasoning
- Intellectual relationship mapping between papers
- Citation influence assessment
- Research lineage and knowledge flow analysis
- Community citation pattern analysis
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, Counter
import logging

from ..services.llm_service import LLMService, get_llm_service
from ..services.llm_prompts import PromptType, get_template_manager
from ..services.llm_cost_manager import CostCategory, get_cost_manager
from ..services.content_enrichment import get_enrichment_service
from ..db.redis import RedisManager, get_redis_manager
from ..db.neo4j import Neo4jManager, get_neo4j_manager
from ..utils.logger import get_logger
from ..utils.exceptions import ValidationError

logger = get_logger(__name__)


class CitationType(Enum):
    """Types of citation relationships."""
    BACKGROUND = "background"
    METHODOLOGY = "methodology"
    COMPARISON = "comparison"
    EXTENSION = "extension"
    CRITICISM = "criticism"
    SUPPORTING_EVIDENCE = "supporting_evidence"
    CONTRADICTING = "contradicting"
    REVIEW = "review"
    DATA_SOURCE = "data_source"
    TOOL_USAGE = "tool_usage"


class InfluenceLevel(Enum):
    """Levels of citation influence."""
    TRANSFORMATIVE = "transformative"  # Fundamentally changed the field
    HIGH_IMPACT = "high_impact"        # Significant methodological or theoretical impact
    MODERATE_IMPACT = "moderate_impact" # Notable contribution to specific area
    INCREMENTAL = "incremental"        # Small but meaningful contribution
    MINIMAL = "minimal"                # Limited direct impact


class CitationContext(Enum):
    """Context where citation appears."""
    INTRODUCTION = "introduction"
    RELATED_WORK = "related_work"
    METHODOLOGY = "methodology"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    UNKNOWN = "unknown"


@dataclass
class CitationRelationship:
    """Detailed analysis of a citation relationship."""
    citing_paper_id: str
    cited_paper_id: str
    
    # Basic metadata
    citing_title: str
    cited_title: str
    citation_year: int
    time_gap_years: int
    
    # LLM Analysis
    citation_purpose: Optional[str] = None
    intellectual_relationship: Optional[str] = None
    knowledge_flow_description: Optional[str] = None
    impact_assessment: Optional[str] = None
    citation_type: Optional[CitationType] = None
    influence_level: Optional[InfluenceLevel] = None
    
    # Context information
    citation_context: Optional[str] = None
    context_location: CitationContext = CitationContext.UNKNOWN
    is_self_citation: bool = False
    is_influential: bool = False
    
    # Quality metrics
    analysis_confidence: float = 0.0
    analysis_model: Optional[str] = None
    analysis_cost: float = 0.0
    analysis_timestamp: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        if self.citation_type:
            data['citation_type'] = self.citation_type.value
        if self.influence_level:
            data['influence_level'] = self.influence_level.value
        data['context_location'] = self.context_location.value
        if self.analysis_timestamp:
            data['analysis_timestamp'] = self.analysis_timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CitationRelationship':
        """Create from dictionary."""
        if 'citation_type' in data and data['citation_type']:
            data['citation_type'] = CitationType(data['citation_type'])
        if 'influence_level' in data and data['influence_level']:
            data['influence_level'] = InfluenceLevel(data['influence_level'])
        if 'context_location' in data:
            data['context_location'] = CitationContext(data['context_location'])
        if 'analysis_timestamp' in data and data['analysis_timestamp']:
            data['analysis_timestamp'] = datetime.fromisoformat(data['analysis_timestamp'])
        return cls(**data)


@dataclass
class CitationNetwork:
    """Analysis of a citation network or subgraph."""
    network_id: str
    center_paper_id: Optional[str] = None
    paper_ids: List[str] = None
    
    # Network metrics
    total_papers: int = 0
    total_citations: int = 0
    network_density: float = 0.0
    avg_citation_age: float = 0.0
    
    # LLM Analysis
    dominant_themes: List[str] = None
    research_evolution: Optional[str] = None
    key_breakthroughs: List[str] = None
    methodological_trends: List[str] = None
    community_structure: Optional[str] = None
    
    # Influence analysis
    most_influential_papers: List[str] = None
    knowledge_flow_patterns: Dict[str, Any] = None
    citation_type_distribution: Dict[CitationType, int] = None
    
    # Quality metrics
    analysis_confidence: float = 0.0
    analysis_cost: float = 0.0
    analysis_timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.paper_ids is None:
            self.paper_ids = []
        if self.dominant_themes is None:
            self.dominant_themes = []
        if self.key_breakthroughs is None:
            self.key_breakthroughs = []
        if self.methodological_trends is None:
            self.methodological_trends = []
        if self.most_influential_papers is None:
            self.most_influential_papers = []
        if self.knowledge_flow_patterns is None:
            self.knowledge_flow_patterns = {}
        if self.citation_type_distribution is None:
            self.citation_type_distribution = {}


@dataclass
class CitationAnalysisRequest:
    """Request for citation analysis."""
    citing_paper_id: str
    cited_paper_id: str
    citation_context: Optional[str] = None
    priority: int = 1  # 1=high, 2=medium, 3=low
    force_refresh: bool = False
    include_network_analysis: bool = False


class CitationAnalysisService:
    """
    Service for intelligent analysis of citation relationships and networks.
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
        self._enrichment_service = None
        self._initialized = False
        
        # Analysis configuration
        self.max_concurrent_analyses = 3
        self.analysis_timeout_seconds = 180  # 3 minutes
        self.confidence_threshold = 0.7
        
    async def initialize(self):
        """Initialize the citation analysis service."""
        if self._initialized:
            return
        
        if not self.llm_service:
            self.llm_service = await get_llm_service()
        
        if not self.redis_manager:
            self.redis_manager = await get_redis_manager()
        
        if not self.neo4j_manager:
            self.neo4j_manager = await get_neo4j_manager()
        
        if not self._enrichment_service:
            self._enrichment_service = await get_enrichment_service()
        
        self._initialized = True
        logger.info("Citation Analysis Service initialized")
    
    async def analyze_citation_relationship(
        self,
        citing_paper_id: str,
        cited_paper_id: str,
        citation_context: Optional[str] = None,
        force_refresh: bool = False
    ) -> CitationRelationship:
        """
        Analyze the relationship between two papers in a citation.
        
        Args:
            citing_paper_id: ID of the paper that cites
            cited_paper_id: ID of the paper being cited
            citation_context: Optional context text where citation appears
            force_refresh: Force new analysis even if cached result exists
            
        Returns:
            CitationRelationship with comprehensive analysis
        """
        await self.initialize()
        
        # Check for existing analysis
        if not force_refresh:
            cached_analysis = await self._get_cached_citation_analysis(citing_paper_id, cited_paper_id)
            if cached_analysis and not self._is_analysis_stale(cached_analysis):
                logger.debug(f"Using cached citation analysis: {citing_paper_id} -> {cited_paper_id}")
                return cached_analysis
        
        # Fetch paper data for both papers
        citing_paper = await self._fetch_paper_data(citing_paper_id)
        cited_paper = await self._fetch_paper_data(cited_paper_id)
        
        if not citing_paper or not cited_paper:
            raise ValidationError(f"Could not fetch paper data for citation analysis")
        
        # Create initial relationship object
        relationship = CitationRelationship(
            citing_paper_id=citing_paper_id,
            cited_paper_id=cited_paper_id,
            citing_title=citing_paper.get('title', 'Unknown'),
            cited_title=cited_paper.get('title', 'Unknown'),
            citation_year=citing_paper.get('year', 0),
            time_gap_years=self._calculate_time_gap(citing_paper, cited_paper),
            citation_context=citation_context,
            analysis_timestamp=datetime.now()
        )
        
        # Detect self-citation
        relationship.is_self_citation = self._is_self_citation(citing_paper, cited_paper)
        
        try:
            # Perform LLM analysis
            await self._analyze_with_llm(relationship, citing_paper, cited_paper)
            
            # Cache the analysis
            await self._cache_citation_analysis(relationship)
            
            # Store in Neo4j
            await self._store_citation_analysis(relationship)
            
            logger.info(f"Analyzed citation: {citing_paper_id} -> {cited_paper_id} "
                       f"(type: {relationship.citation_type.value if relationship.citation_type else 'unknown'})")
            
            return relationship
            
        except Exception as e:
            logger.error(f"Failed to analyze citation {citing_paper_id} -> {cited_paper_id}: {e}")
            raise
    
    async def _analyze_with_llm(
        self,
        relationship: CitationRelationship,
        citing_paper: Dict[str, Any],
        cited_paper: Dict[str, Any]
    ):
        """Perform LLM analysis of the citation relationship."""
        # Get citation analysis template
        template = self._template_manager.get_template(PromptType.CITATION_ANALYSIS)
        
        # Prepare template variables
        template_vars = {
            'citing_title': citing_paper.get('title', ''),
            'citing_authors': ', '.join([author.get('name', '') for author in citing_paper.get('authors', [])]),
            'citing_year': citing_paper.get('year', 'Unknown'),
            'citing_abstract': citing_paper.get('abstract', ''),
            'cited_title': cited_paper.get('title', ''),
            'cited_authors': ', '.join([author.get('name', '') for author in cited_paper.get('authors', [])]),
            'cited_year': cited_paper.get('year', 'Unknown'),
            'cited_abstract': cited_paper.get('abstract', ''),
            'citation_context': relationship.citation_context or 'Not available',
            'citation_intent': 'Unknown',  # Would be populated from Semantic Scholar data
            'is_influential': str(relationship.is_influential)
        }
        
        # Make LLM request
        response = await self.llm_service.complete(
            template,
            use_cache=True,
            **template_vars
        )
        
        # Record cost
        cost_manager = await get_cost_manager()
        await cost_manager.record_usage(
            model=response.model,
            provider=response.provider,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            category=CostCategory.CITATION_ANALYSIS,
            cached=response.cached
        )
        
        # Update relationship with cost info
        relationship.analysis_cost = response.usage.cost
        relationship.analysis_model = response.model
        
        # Parse the structured response
        await self._parse_citation_analysis_response(relationship, response.content)
    
    async def _parse_citation_analysis_response(
        self,
        relationship: CitationRelationship,
        response_content: str
    ):
        """Parse the structured LLM response into relationship fields."""
        try:
            sections = self._split_response_into_sections(response_content)
            
            # Extract analysis fields
            relationship.citation_purpose = sections.get('citation purpose', '').strip()
            relationship.intellectual_relationship = sections.get('intellectual relationship', '').strip()
            relationship.knowledge_flow_description = sections.get('knowledge flow', '').strip()
            relationship.impact_assessment = sections.get('impact assessment', '').strip()
            
            # Parse citation type
            citation_type_text = sections.get('citation type', '').lower()
            relationship.citation_type = self._parse_citation_type(citation_type_text)
            
            # Assess analysis quality
            relationship.analysis_confidence = self._assess_analysis_quality(relationship, sections)
            
            # Infer influence level from impact assessment
            relationship.influence_level = self._infer_influence_level(
                relationship.impact_assessment,
                relationship.time_gap_years
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse citation analysis response: {e}")
            # Fallback - use raw content
            relationship.citation_purpose = response_content[:500]
            relationship.analysis_confidence = 0.3  # Low confidence for unparsed content
    
    def _split_response_into_sections(self, content: str) -> Dict[str, str]:
        """Split LLM response into structured sections."""
        sections = {}
        current_section = None
        current_content = []
        
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Check if this is a section header
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
    
    def _parse_citation_type(self, type_text: str) -> Optional[CitationType]:
        """Parse citation type from text."""
        type_text = type_text.lower()
        
        # Map common phrases to citation types
        type_mappings = {
            'background': CitationType.BACKGROUND,
            'methodology': CitationType.METHODOLOGY,
            'comparison': CitationType.COMPARISON,
            'extension': CitationType.EXTENSION,
            'criticism': CitationType.CRITICISM,
            'supporting': CitationType.SUPPORTING_EVIDENCE,
            'evidence': CitationType.SUPPORTING_EVIDENCE,
            'contradicting': CitationType.CONTRADICTING,
            'review': CitationType.REVIEW,
            'data': CitationType.DATA_SOURCE,
            'tool': CitationType.TOOL_USAGE,
        }
        
        for keyword, citation_type in type_mappings.items():
            if keyword in type_text:
                return citation_type
        
        return None
    
    def _assess_analysis_quality(
        self,
        relationship: CitationRelationship,
        sections: Dict[str, str]
    ) -> float:
        """Assess the quality/confidence of the analysis."""
        quality_score = 0.0
        max_score = 5.0
        
        # Check for presence and quality of key sections
        if relationship.citation_purpose and len(relationship.citation_purpose) > 30:
            quality_score += 1.0
        
        if relationship.intellectual_relationship and len(relationship.intellectual_relationship) > 30:
            quality_score += 1.0
        
        if relationship.knowledge_flow_description and len(relationship.knowledge_flow_description) > 30:
            quality_score += 1.0
        
        if relationship.impact_assessment and len(relationship.impact_assessment) > 30:
            quality_score += 1.0
        
        if relationship.citation_type is not None:
            quality_score += 1.0
        
        return quality_score / max_score
    
    def _infer_influence_level(self, impact_text: str, time_gap: int) -> InfluenceLevel:
        """Infer influence level from impact assessment text."""
        if not impact_text:
            return InfluenceLevel.MINIMAL
        
        impact_lower = impact_text.lower()
        
        # Keywords indicating different influence levels
        transformative_keywords = ['transformative', 'revolutionary', 'paradigm', 'breakthrough', 'foundational']
        high_impact_keywords = ['significant', 'major', 'important', 'influential', 'substantial']
        moderate_keywords = ['notable', 'valuable', 'useful', 'contributes', 'advances']
        
        # Check for transformative impact
        if any(keyword in impact_lower for keyword in transformative_keywords):
            return InfluenceLevel.TRANSFORMATIVE
        
        # High impact indicators
        if any(keyword in impact_lower for keyword in high_impact_keywords):
            return InfluenceLevel.HIGH_IMPACT
        
        # Moderate impact indicators
        if any(keyword in impact_lower for keyword in moderate_keywords):
            return InfluenceLevel.MODERATE_IMPACT
        
        # Consider time gap - newer citations might be more incremental
        if time_gap <= 2:
            return InfluenceLevel.INCREMENTAL
        
        return InfluenceLevel.MINIMAL
    
    def _calculate_time_gap(self, citing_paper: Dict[str, Any], cited_paper: Dict[str, Any]) -> int:
        """Calculate time gap between papers in years."""
        citing_year = citing_paper.get('year', 0)
        cited_year = cited_paper.get('year', 0)
        
        if citing_year and cited_year:
            return max(0, citing_year - cited_year)
        
        return 0
    
    def _is_self_citation(self, citing_paper: Dict[str, Any], cited_paper: Dict[str, Any]) -> bool:
        """Check if this is a self-citation."""
        citing_authors = {author.get('name', '').lower() for author in citing_paper.get('authors', [])}
        cited_authors = {author.get('name', '').lower() for author in cited_paper.get('authors', [])}
        
        # If any author appears in both papers, it's a self-citation
        return len(citing_authors.intersection(cited_authors)) > 0
    
    async def analyze_citation_network(
        self,
        center_paper_id: str,
        depth: int = 2,
        max_papers: int = 50
    ) -> CitationNetwork:
        """
        Analyze a citation network centered on a specific paper.
        
        Args:
            center_paper_id: Central paper for network analysis
            depth: Citation depth to explore (1 = direct citations only)
            max_papers: Maximum papers to include in analysis
            
        Returns:
            CitationNetwork with comprehensive network analysis
        """
        await self.initialize()
        
        # Build citation network
        network_papers = await self._build_citation_network(center_paper_id, depth, max_papers)
        
        if len(network_papers) < 3:
            raise ValidationError(f"Insufficient papers in citation network (found {len(network_papers)})")
        
        # Create network object
        network = CitationNetwork(
            network_id=f"network_{center_paper_id}_{depth}",
            center_paper_id=center_paper_id,
            paper_ids=list(network_papers.keys()),
            total_papers=len(network_papers),
            analysis_timestamp=datetime.now()
        )
        
        # Calculate network metrics
        await self._calculate_network_metrics(network, network_papers)
        
        # Perform LLM analysis of the network
        await self._analyze_network_with_llm(network, network_papers)
        
        # Cache network analysis
        await self._cache_network_analysis(network)
        
        logger.info(f"Analyzed citation network for {center_paper_id}: "
                   f"{network.total_papers} papers, {network.total_citations} citations")
        
        return network
    
    async def _build_citation_network(
        self,
        center_paper_id: str,
        depth: int,
        max_papers: int
    ) -> Dict[str, Dict[str, Any]]:
        """Build citation network data structure."""
        # This would query Neo4j or other data sources to build the network
        # For now, return placeholder structure
        network_papers = {}
        
        # Add center paper
        center_paper = await self._fetch_paper_data(center_paper_id)
        if center_paper:
            network_papers[center_paper_id] = center_paper
        
        # In a real implementation, would recursively add cited/citing papers
        # up to the specified depth and max_papers limit
        
        return network_papers
    
    async def _calculate_network_metrics(
        self,
        network: CitationNetwork,
        network_papers: Dict[str, Dict[str, Any]]
    ):
        """Calculate basic network metrics."""
        # Count total citations in network
        total_citations = sum(
            paper.get('citation_count', 0) for paper in network_papers.values()
        )
        network.total_citations = total_citations
        
        # Calculate average citation age
        current_year = datetime.now().year
        years = [paper.get('year', current_year) for paper in network_papers.values()]
        if years:
            network.avg_citation_age = current_year - sum(years) / len(years)
        
        # Network density would require citation graph structure
        network.network_density = 0.0  # Placeholder
    
    async def _analyze_network_with_llm(
        self,
        network: CitationNetwork,
        network_papers: Dict[str, Dict[str, Any]]
    ):
        """Perform LLM analysis of the citation network."""
        # Get theme identification template
        template = self._template_manager.get_template(PromptType.THEME_IDENTIFICATION)
        
        # Prepare paper sample for analysis
        paper_sample = []
        for paper_id, paper_data in list(network_papers.items())[:10]:  # Limit to 10 papers
            paper_sample.append({
                'title': paper_data.get('title', ''),
                'abstract': paper_data.get('abstract', '')[:200] + '...',  # Truncate abstract
                'year': paper_data.get('year', 'Unknown')
            })
        
        template_vars = {
            'num_papers': network.total_papers,
            'time_range': f"{min(p.get('year', 2024) for p in network_papers.values())} - "
                         f"{max(p.get('year', 2024) for p in network_papers.values())}",
            'primary_field': 'Computer Science',  # Would detect from paper data
            'key_venues': ['Unknown'],  # Would extract from paper data
            'papers_sample': json.dumps(paper_sample, indent=2),
            'total_citations': network.total_citations,
            'avg_citations': network.total_citations // network.total_papers if network.total_papers > 0 else 0,
            'top_cited': ['Unknown'],  # Would calculate from data
            'total_authors': 0,  # Would calculate from data
            'collaboration_info': 'Unknown'  # Would analyze collaboration patterns
        }
        
        try:
            # Make LLM request
            response = await self.llm_service.complete(
                template,
                use_cache=True,
                **template_vars
            )
            
            # Record cost
            cost_manager = await get_cost_manager()
            await cost_manager.record_usage(
                model=response.model,
                provider=response.provider,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                category=CostCategory.RESEARCH_TRAJECTORY,
                cached=response.cached
            )
            
            network.analysis_cost = response.usage.cost
            
            # Parse network analysis response
            await self._parse_network_analysis_response(network, response.content)
            
        except Exception as e:
            logger.warning(f"Failed to analyze network with LLM: {e}")
            network.dominant_themes = ['Analysis failed']
            network.analysis_confidence = 0.0
    
    async def _parse_network_analysis_response(
        self,
        network: CitationNetwork,
        response_content: str
    ):
        """Parse network analysis response."""
        try:
            sections = self._split_response_into_sections(response_content)
            
            # Extract themes
            themes_text = sections.get('major research themes', '')
            network.dominant_themes = self._extract_themes_from_text(themes_text)
            
            # Extract evolution description
            network.research_evolution = sections.get('research maturity', '').strip()
            
            # Extract breakthroughs
            breakthroughs_text = sections.get('emerging trends', '')
            network.key_breakthroughs = self._extract_bullet_points(breakthroughs_text)
            
            # Extract methodological trends
            methods_text = sections.get('methodological approaches', '')
            network.methodological_trends = self._extract_bullet_points(methods_text)
            
            # Assess analysis quality
            network.analysis_confidence = self._assess_network_analysis_quality(network)
            
        except Exception as e:
            logger.warning(f"Failed to parse network analysis: {e}")
            network.analysis_confidence = 0.3
    
    def _extract_themes_from_text(self, text: str) -> List[str]:
        """Extract research themes from text."""
        if not text:
            return []
        
        themes = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 10:
                # Clean up theme text
                if line.startswith(('-', '•', '*', '+')):
                    line = line[1:].strip()
                elif line.startswith(tuple(f"{i}." for i in range(1, 20))):
                    line = line.split('.', 1)[1].strip() if '.' in line else line
                
                themes.append(line)
        
        return themes[:8]  # Limit to top 8 themes
    
    def _extract_bullet_points(self, text: str) -> List[str]:
        """Extract bullet points from text."""
        if not text:
            return []
        
        points = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 15:
                # Clean up bullet point
                if line.startswith(('-', '•', '*', '+')):
                    line = line[1:].strip()
                elif line.startswith(tuple(f"{i}." for i in range(1, 20))):
                    line = line.split('.', 1)[1].strip() if '.' in line else line
                
                points.append(line)
        
        return points[:6]  # Limit to top 6 points
    
    def _assess_network_analysis_quality(self, network: CitationNetwork) -> float:
        """Assess quality of network analysis."""
        quality_score = 0.0
        max_score = 4.0
        
        if network.dominant_themes and len(network.dominant_themes) >= 3:
            quality_score += 1.0
        
        if network.research_evolution and len(network.research_evolution) > 50:
            quality_score += 1.0
        
        if network.key_breakthroughs and len(network.key_breakthroughs) >= 2:
            quality_score += 1.0
        
        if network.methodological_trends and len(network.methodological_trends) >= 2:
            quality_score += 1.0
        
        return quality_score / max_score
    
    async def _fetch_paper_data(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Fetch paper data from available sources."""
        # Would integrate with existing data sources
        return None
    
    def _is_analysis_stale(self, relationship: CitationRelationship) -> bool:
        """Check if analysis is stale and needs refresh."""
        if not relationship.analysis_timestamp:
            return True
        
        # Consider analysis stale after 60 days
        stale_threshold = datetime.now() - timedelta(days=60)
        return relationship.analysis_timestamp < stale_threshold
    
    async def _get_cached_citation_analysis(
        self,
        citing_paper_id: str,
        cited_paper_id: str
    ) -> Optional[CitationRelationship]:
        """Get cached citation analysis if available."""
        if not self.redis_manager:
            return None
        
        try:
            cache_key = f"citation_analysis:{citing_paper_id}:{cited_paper_id}"
            cached_data = await self.redis_manager.get(cache_key)
            
            if cached_data:
                data = json.loads(cached_data)
                return CitationRelationship.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to retrieve cached citation analysis: {e}")
        
        return None
    
    async def _cache_citation_analysis(self, relationship: CitationRelationship):
        """Cache citation analysis in Redis."""
        if not self.redis_manager:
            return
        
        try:
            cache_key = f"citation_analysis:{relationship.citing_paper_id}:{relationship.cited_paper_id}"
            ttl_seconds = 60 * 24 * 3600  # 60 days
            
            await self.redis_manager.setex(
                cache_key,
                ttl_seconds,
                json.dumps(relationship.to_dict())
            )
        except Exception as e:
            logger.error(f"Failed to cache citation analysis: {e}")
    
    async def _cache_network_analysis(self, network: CitationNetwork):
        """Cache network analysis in Redis."""
        if not self.redis_manager:
            return
        
        try:
            cache_key = f"citation_network:{network.network_id}"
            ttl_seconds = 30 * 24 * 3600  # 30 days
            
            network_dict = asdict(network)
            network_dict['analysis_timestamp'] = network.analysis_timestamp.isoformat() if network.analysis_timestamp else None
            
            await self.redis_manager.setex(
                cache_key,
                ttl_seconds,
                json.dumps(network_dict)
            )
        except Exception as e:
            logger.error(f"Failed to cache network analysis: {e}")
    
    async def _store_citation_analysis(self, relationship: CitationRelationship):
        """Store citation analysis in Neo4j."""
        if not self.neo4j_manager:
            return
        
        try:
            # Would store citation relationship in Neo4j graph
            logger.debug(f"Would store citation analysis in Neo4j: "
                        f"{relationship.citing_paper_id} -> {relationship.cited_paper_id}")
        except Exception as e:
            logger.error(f"Failed to store citation analysis in Neo4j: {e}")


# Global service instance
_citation_analysis_service: Optional[CitationAnalysisService] = None


async def get_citation_analysis_service() -> CitationAnalysisService:
    """Get or create the global citation analysis service."""
    global _citation_analysis_service
    
    if _citation_analysis_service is None:
        _citation_analysis_service = CitationAnalysisService()
        await _citation_analysis_service.initialize()
    
    return _citation_analysis_service