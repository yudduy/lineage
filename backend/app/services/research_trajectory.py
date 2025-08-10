"""
Research Trajectory & Intellectual Lineage Analysis Service.

This service provides:
- Intellectual lineage tracing through citation networks
- Research trajectory analysis over time
- Timeline narrative generation for research evolution
- Community and field evolution analysis
- Key milestone and breakthrough identification
- Influence pathway mapping
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
from ..services.citation_analysis import get_citation_analysis_service, CitationRelationship
from ..services.content_enrichment import get_enrichment_service
from ..db.redis import RedisManager, get_redis_manager
from ..db.neo4j import Neo4jManager, get_neo4j_manager
from ..utils.logger import get_logger
from ..utils.exceptions import ValidationError

logger = get_logger(__name__)


class TrajectoryType(Enum):
    """Types of research trajectories."""
    LINEAR_PROGRESSION = "linear_progression"      # Steady incremental progress
    BREAKTHROUGH_DRIVEN = "breakthrough_driven"   # Progress through major breakthroughs
    CONVERGENT_EVOLUTION = "convergent_evolution" # Multiple paths converging
    DIVERGENT_BRANCHING = "divergent_branching"   # Single idea branching into multiple directions
    CYCLICAL_REVIVAL = "cyclical_revival"         # Ideas that come back after dormancy
    INTERDISCIPLINARY = "interdisciplinary"      # Cross-field knowledge transfer


class MilestoneType(Enum):
    """Types of research milestones."""
    FOUNDATIONAL_WORK = "foundational_work"       # Early foundational papers
    METHODOLOGICAL_BREAKTHROUGH = "methodological_breakthrough"  # New methods
    THEORETICAL_ADVANCE = "theoretical_advance"   # New theories or frameworks
    EMPIRICAL_VALIDATION = "empirical_validation" # Key experimental validations
    PRACTICAL_APPLICATION = "practical_application"  # Applied implementations
    PARADIGM_SHIFT = "paradigm_shift"            # Fundamental changes in thinking
    SYNTHESIS = "synthesis"                      # Integration of multiple streams


@dataclass
class ResearchMilestone:
    """A significant milestone in research trajectory."""
    paper_id: str
    title: str
    year: int
    milestone_type: MilestoneType
    significance_description: str
    impact_score: float
    citation_count: int
    influenced_papers: List[str]
    key_contributions: List[str]
    
    # Analysis metadata
    confidence_score: float = 0.0
    analysis_model: Optional[str] = None
    analysis_timestamp: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['milestone_type'] = self.milestone_type.value
        if self.analysis_timestamp:
            data['analysis_timestamp'] = self.analysis_timestamp.isoformat()
        return data


@dataclass
class IntellectualLineage:
    """Intellectual lineage tracing a line of research."""
    lineage_id: str
    root_paper_ids: List[str]
    leaf_paper_ids: List[str]
    
    # Trajectory analysis
    trajectory_type: Optional[TrajectoryType] = None
    total_papers: int = 0
    time_span_years: int = 0
    generation_count: int = 0  # Number of "generations" in the lineage
    
    # Key components
    milestones: List[ResearchMilestone] = None
    key_researchers: List[str] = None
    dominant_venues: List[str] = None
    field_evolution: List[str] = None
    
    # LLM Analysis
    lineage_narrative: Optional[str] = None
    intellectual_evolution: Optional[str] = None
    key_insights: List[str] = None
    future_directions: List[str] = None
    cross_field_influences: List[str] = None
    
    # Quality metrics
    analysis_confidence: float = 0.0
    analysis_cost: float = 0.0
    analysis_model: Optional[str] = None
    analysis_timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.milestones is None:
            self.milestones = []
        if self.key_researchers is None:
            self.key_researchers = []
        if self.dominant_venues is None:
            self.dominant_venues = []
        if self.field_evolution is None:
            self.field_evolution = []
        if self.key_insights is None:
            self.key_insights = []
        if self.future_directions is None:
            self.future_directions = []
        if self.cross_field_influences is None:
            self.cross_field_influences = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        if self.trajectory_type:
            data['trajectory_type'] = self.trajectory_type.value
        data['milestones'] = [milestone.to_dict() for milestone in self.milestones]
        if self.analysis_timestamp:
            data['analysis_timestamp'] = self.analysis_timestamp.isoformat()
        return data


@dataclass
class ResearchTimeline:
    """Timeline representation of research evolution."""
    timeline_id: str
    domain: str
    start_year: int
    end_year: int
    
    # Timeline structure
    periods: List[Dict[str, Any]] = None  # Time periods with descriptions
    breakthrough_moments: List[Dict[str, Any]] = None
    key_transitions: List[Dict[str, Any]] = None
    
    # LLM-generated narrative
    timeline_narrative: Optional[str] = None
    period_descriptions: Dict[str, str] = None
    causal_relationships: List[str] = None
    
    # Analysis metadata
    analysis_confidence: float = 0.0
    analysis_cost: float = 0.0
    analysis_timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.periods is None:
            self.periods = []
        if self.breakthrough_moments is None:
            self.breakthrough_moments = []
        if self.key_transitions is None:
            self.key_transitions = []
        if self.period_descriptions is None:
            self.period_descriptions = {}
        if self.causal_relationships is None:
            self.causal_relationships = []


class ResearchTrajectoryService:
    """
    Service for analyzing research trajectories and intellectual lineages.
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
        self._citation_service = None
        self._enrichment_service = None
        self._initialized = False
        
        # Analysis configuration
        self.max_lineage_papers = 100
        self.max_timeline_years = 50
        self.min_milestone_impact = 0.7
        
    async def initialize(self):
        """Initialize the research trajectory service."""
        if self._initialized:
            return
        
        if not self.llm_service:
            self.llm_service = await get_llm_service()
        
        if not self.redis_manager:
            self.redis_manager = await get_redis_manager()
        
        if not self.neo4j_manager:
            self.neo4j_manager = await get_neo4j_manager()
        
        if not self._citation_service:
            self._citation_service = await get_citation_analysis_service()
        
        if not self._enrichment_service:
            self._enrichment_service = await get_enrichment_service()
        
        self._initialized = True
        logger.info("Research Trajectory Service initialized")
    
    async def trace_intellectual_lineage(
        self,
        seed_paper_ids: List[str],
        max_generations: int = 5,
        include_future_work: bool = True
    ) -> IntellectualLineage:
        """
        Trace the intellectual lineage from seed papers.
        
        Args:
            seed_paper_ids: Starting papers for lineage tracing
            max_generations: Maximum generations to trace
            include_future_work: Whether to include papers citing the seed papers
            
        Returns:
            IntellectualLineage with comprehensive analysis
        """
        await self.initialize()
        
        if not seed_paper_ids:
            raise ValidationError("Must provide at least one seed paper ID")
        
        # Build lineage graph
        lineage_graph = await self._build_lineage_graph(
            seed_paper_ids,
            max_generations,
            include_future_work
        )
        
        if len(lineage_graph) < 3:
            raise ValidationError(f"Insufficient papers in lineage (found {len(lineage_graph)})")
        
        # Create lineage object
        lineage = IntellectualLineage(
            lineage_id=f"lineage_{'_'.join(seed_paper_ids[:3])}",
            root_paper_ids=seed_paper_ids,
            total_papers=len(lineage_graph),
            analysis_timestamp=datetime.now()
        )
        
        # Analyze lineage structure
        await self._analyze_lineage_structure(lineage, lineage_graph)
        
        # Identify key milestones
        await self._identify_milestones(lineage, lineage_graph)
        
        # Perform LLM analysis
        await self._analyze_lineage_with_llm(lineage, lineage_graph)
        
        # Cache the analysis
        await self._cache_lineage_analysis(lineage)
        
        logger.info(f"Traced intellectual lineage: {lineage.total_papers} papers, "
                   f"{len(lineage.milestones)} milestones, "
                   f"{lineage.time_span_years} year span")
        
        return lineage
    
    async def _build_lineage_graph(
        self,
        seed_paper_ids: List[str],
        max_generations: int,
        include_future_work: bool
    ) -> Dict[str, Dict[str, Any]]:
        """Build lineage graph through citation relationships."""
        lineage_papers = {}
        
        # Start with seed papers
        current_generation = set(seed_paper_ids)
        
        for generation in range(max_generations):
            next_generation = set()
            
            for paper_id in current_generation:
                # Get paper data
                paper_data = await self._fetch_paper_data(paper_id)
                if paper_data:
                    lineage_papers[paper_id] = {
                        **paper_data,
                        'generation': generation,
                        'is_seed': generation == 0
                    }
                    
                    # Add referenced papers (past influences)
                    references = await self._get_paper_references(paper_id)
                    for ref_id in references[:10]:  # Limit references per paper
                        if ref_id not in lineage_papers and len(lineage_papers) < self.max_lineage_papers:
                            next_generation.add(ref_id)
                    
                    # Add citing papers (future influence) if requested
                    if include_future_work:
                        citations = await self._get_paper_citations(paper_id)
                        for cite_id in citations[:5]:  # Limit citations per paper
                            if cite_id not in lineage_papers and len(lineage_papers) < self.max_lineage_papers:
                                next_generation.add(cite_id)
            
            if not next_generation or len(lineage_papers) >= self.max_lineage_papers:
                break
                
            current_generation = next_generation
        
        return lineage_papers
    
    async def _analyze_lineage_structure(
        self,
        lineage: IntellectualLineage,
        lineage_graph: Dict[str, Dict[str, Any]]
    ):
        """Analyze the structure of the lineage."""
        papers = list(lineage_graph.values())
        
        # Calculate time span
        years = [paper.get('year', 0) for paper in papers if paper.get('year')]
        if years:
            lineage.time_span_years = max(years) - min(years)
        
        # Count generations
        generations = [paper.get('generation', 0) for paper in papers]
        lineage.generation_count = max(generations) + 1 if generations else 1
        
        # Identify leaf papers (papers with no citing papers in the lineage)
        citing_relationships = set()
        for paper_id in lineage_graph:
            citations = await self._get_paper_citations(paper_id)
            for cite_id in citations:
                if cite_id in lineage_graph:
                    citing_relationships.add(cite_id)
        
        lineage.leaf_paper_ids = [
            paper_id for paper_id in lineage_graph
            if paper_id not in citing_relationships
        ]
        
        # Extract key researchers
        all_authors = []
        for paper in papers:
            authors = paper.get('authors', [])
            all_authors.extend([author.get('name', '') for author in authors])
        
        author_counts = Counter(all_authors)
        lineage.key_researchers = [
            author for author, count in author_counts.most_common(10)
            if count > 1 and author
        ]
        
        # Extract dominant venues
        venues = [paper.get('venue', {}).get('name', '') for paper in papers]
        venue_counts = Counter(venues)
        lineage.dominant_venues = [
            venue for venue, count in venue_counts.most_common(5)
            if venue
        ]
        
        # Classify trajectory type
        lineage.trajectory_type = self._classify_trajectory_type(lineage, papers)
    
    def _classify_trajectory_type(
        self,
        lineage: IntellectualLineage,
        papers: List[Dict[str, Any]]
    ) -> TrajectoryType:
        """Classify the type of research trajectory."""
        # Simple heuristics for trajectory classification
        # In production, this could be more sophisticated
        
        # Check temporal distribution
        years = [paper.get('year', 0) for paper in papers if paper.get('year')]
        if not years:
            return TrajectoryType.LINEAR_PROGRESSION
        
        year_range = max(years) - min(years)
        
        # If short time span with many papers, likely breakthrough-driven
        if year_range < 5 and len(papers) > 20:
            return TrajectoryType.BREAKTHROUGH_DRIVEN
        
        # If many different venues/fields, likely interdisciplinary
        if len(lineage.dominant_venues) > 8:
            return TrajectoryType.INTERDISCIPLINARY
        
        # If high generation count relative to papers, likely linear
        if lineage.generation_count > len(papers) / 10:
            return TrajectoryType.LINEAR_PROGRESSION
        
        # Default to linear progression
        return TrajectoryType.LINEAR_PROGRESSION
    
    async def _identify_milestones(
        self,
        lineage: IntellectualLineage,
        lineage_graph: Dict[str, Dict[str, Any]]
    ):
        """Identify key milestones in the research lineage."""
        milestones = []
        
        for paper_id, paper_data in lineage_graph.items():
            # Calculate impact score based on citations and other factors
            citation_count = paper_data.get('citation_count', 0)
            year = paper_data.get('year', 0)
            is_seed = paper_data.get('is_seed', False)
            
            # Simple impact scoring (could be more sophisticated)
            impact_score = 0.0
            
            if citation_count > 100:
                impact_score += 0.4
            elif citation_count > 50:
                impact_score += 0.3
            elif citation_count > 20:
                impact_score += 0.2
            
            if is_seed:
                impact_score += 0.2
            
            if year and year < 2000:  # Older foundational work
                impact_score += 0.1
            
            # Only consider high-impact papers as milestones
            if impact_score >= self.min_milestone_impact:
                milestone_type = self._classify_milestone_type(paper_data, is_seed)
                
                milestone = ResearchMilestone(
                    paper_id=paper_id,
                    title=paper_data.get('title', 'Unknown'),
                    year=year,
                    milestone_type=milestone_type,
                    significance_description=f"High-impact paper with {citation_count} citations",
                    impact_score=impact_score,
                    citation_count=citation_count,
                    influenced_papers=await self._get_paper_citations(paper_id),
                    key_contributions=[],  # Would be filled by enrichment service
                    confidence_score=0.8,  # Basic confidence
                    analysis_timestamp=datetime.now()
                )
                
                milestones.append(milestone)
        
        # Sort by year and impact
        milestones.sort(key=lambda m: (m.year, -m.impact_score))
        lineage.milestones = milestones[:20]  # Limit to top 20 milestones
    
    def _classify_milestone_type(
        self,
        paper_data: Dict[str, Any],
        is_seed: bool
    ) -> MilestoneType:
        """Classify the type of milestone based on paper characteristics."""
        year = paper_data.get('year', 0)
        title = paper_data.get('title', '').lower()
        
        # Simple classification based on patterns
        if is_seed or year < 1990:
            return MilestoneType.FOUNDATIONAL_WORK
        
        if any(keyword in title for keyword in ['method', 'algorithm', 'approach']):
            return MilestoneType.METHODOLOGICAL_BREAKTHROUGH
        
        if any(keyword in title for keyword in ['theory', 'framework', 'model']):
            return MilestoneType.THEORETICAL_ADVANCE
        
        if any(keyword in title for keyword in ['application', 'system', 'implementation']):
            return MilestoneType.PRACTICAL_APPLICATION
        
        if any(keyword in title for keyword in ['evaluation', 'experiment', 'study']):
            return MilestoneType.EMPIRICAL_VALIDATION
        
        if any(keyword in title for keyword in ['survey', 'review', 'synthesis']):
            return MilestoneType.SYNTHESIS
        
        return MilestoneType.THEORETICAL_ADVANCE  # Default
    
    async def _analyze_lineage_with_llm(
        self,
        lineage: IntellectualLineage,
        lineage_graph: Dict[str, Dict[str, Any]]
    ):
        """Perform LLM analysis of the intellectual lineage."""
        # Get research trajectory template
        template = self._template_manager.get_template(PromptType.RESEARCH_TRAJECTORY)
        
        # Prepare papers data for analysis
        papers_data = []
        for paper_id, paper_data in list(lineage_graph.items())[:20]:  # Limit to 20 papers
            papers_data.append({
                'id': paper_id,
                'title': paper_data.get('title', ''),
                'year': paper_data.get('year', 'Unknown'),
                'authors': ', '.join([author.get('name', '') for author in paper_data.get('authors', [])]),
                'citation_count': paper_data.get('citation_count', 0),
                'generation': paper_data.get('generation', 0)
            })
        
        # Sort by year
        papers_data.sort(key=lambda p: p.get('year', 0))
        
        template_vars = {
            'papers_data': json.dumps(papers_data, indent=2),
            'field': self._infer_research_field(lineage_graph),
            'time_span': f"{lineage.time_span_years} years",
            'num_papers': lineage.total_papers
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
            
            lineage.analysis_cost = response.usage.cost
            lineage.analysis_model = response.model
            
            # Parse trajectory analysis response
            await self._parse_trajectory_analysis_response(lineage, response.content)
            
        except Exception as e:
            logger.warning(f"Failed to analyze lineage with LLM: {e}")
            lineage.analysis_confidence = 0.3
            lineage.intellectual_evolution = "LLM analysis failed"
    
    def _infer_research_field(self, lineage_graph: Dict[str, Dict[str, Any]]) -> str:
        """Infer the primary research field from papers."""
        # Extract fields from papers
        all_fields = []
        for paper_data in lineage_graph.values():
            fields = paper_data.get('fields_of_study', [])
            if isinstance(fields, list):
                all_fields.extend(fields)
        
        if not all_fields:
            return "Computer Science"  # Default
        
        field_counts = Counter(all_fields)
        return field_counts.most_common(1)[0][0] if field_counts else "Computer Science"
    
    async def _parse_trajectory_analysis_response(
        self,
        lineage: IntellectualLineage,
        response_content: str
    ):
        """Parse trajectory analysis response."""
        try:
            sections = self._split_response_into_sections(response_content)
            
            # Extract key sections
            lineage.intellectual_evolution = sections.get('intellectual evolution', '').strip()
            lineage.lineage_narrative = sections.get('significance assessment', '').strip()
            
            # Extract key insights
            insights_text = sections.get('innovation patterns', '')
            lineage.key_insights = self._extract_bullet_points(insights_text)
            
            # Extract future directions
            future_text = sections.get('future trajectory', '')
            lineage.future_directions = self._extract_bullet_points(future_text)
            
            # Assess analysis quality
            lineage.analysis_confidence = self._assess_trajectory_analysis_quality(lineage)
            
        except Exception as e:
            logger.warning(f"Failed to parse trajectory analysis: {e}")
            lineage.analysis_confidence = 0.3
    
    def _split_response_into_sections(self, content: str) -> Dict[str, str]:
        """Split LLM response into structured sections."""
        sections = {}
        current_section = None
        current_content = []
        
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('##') or (line.startswith('**') and line.endswith('**')):
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                current_section = line.replace('##', '').replace('**', '').strip().lower()
                current_content = []
            else:
                if current_section and line:
                    current_content.append(line)
        
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def _extract_bullet_points(self, text: str) -> List[str]:
        """Extract bullet points from text."""
        if not text:
            return []
        
        points = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 15:
                if line.startswith(('-', '•', '*', '+')):
                    line = line[1:].strip()
                elif line.startswith(tuple(f"{i}." for i in range(1, 20))):
                    line = line.split('.', 1)[1].strip() if '.' in line else line
                
                points.append(line)
        
        return points[:8]
    
    def _assess_trajectory_analysis_quality(self, lineage: IntellectualLineage) -> float:
        """Assess quality of trajectory analysis."""
        quality_score = 0.0
        max_score = 4.0
        
        if lineage.intellectual_evolution and len(lineage.intellectual_evolution) > 100:
            quality_score += 1.0
        
        if lineage.key_insights and len(lineage.key_insights) >= 3:
            quality_score += 1.0
        
        if lineage.future_directions and len(lineage.future_directions) >= 2:
            quality_score += 1.0
        
        if len(lineage.milestones) >= 3:
            quality_score += 1.0
        
        return quality_score / max_score
    
    async def generate_timeline_narrative(
        self,
        lineage: IntellectualLineage,
        include_context: bool = True
    ) -> ResearchTimeline:
        """
        Generate a narrative timeline from the intellectual lineage.
        
        Args:
            lineage: IntellectualLineage to create timeline for
            include_context: Whether to include broader research context
            
        Returns:
            ResearchTimeline with narrative structure
        """
        await self.initialize()
        
        if not lineage.milestones:
            raise ValidationError("Lineage must have milestones to generate timeline")
        
        # Create timeline object
        timeline = ResearchTimeline(
            timeline_id=f"timeline_{lineage.lineage_id}",
            domain=self._infer_research_field({}),  # Would extract from lineage
            start_year=min(milestone.year for milestone in lineage.milestones if milestone.year),
            end_year=max(milestone.year for milestone in lineage.milestones if milestone.year),
            analysis_timestamp=datetime.now()
        )
        
        # Organize milestones into periods
        await self._organize_timeline_periods(timeline, lineage)
        
        # Generate LLM narrative
        await self._generate_timeline_narrative_with_llm(timeline, lineage)
        
        # Cache timeline
        await self._cache_timeline_analysis(timeline)
        
        logger.info(f"Generated timeline narrative: {timeline.start_year}-{timeline.end_year}, "
                   f"{len(timeline.periods)} periods")
        
        return timeline
    
    async def _organize_timeline_periods(
        self,
        timeline: ResearchTimeline,
        lineage: IntellectualLineage
    ):
        """Organize milestones into coherent time periods."""
        if not lineage.milestones:
            return
        
        # Group milestones by decade
        period_milestones = defaultdict(list)
        for milestone in lineage.milestones:
            if milestone.year:
                decade = (milestone.year // 10) * 10
                period_milestones[decade].append(milestone)
        
        # Create periods
        for decade in sorted(period_milestones.keys()):
            milestones = period_milestones[decade]
            
            period = {
                'start_year': decade,
                'end_year': decade + 9,
                'title': f"{decade}s",
                'milestone_count': len(milestones),
                'key_milestones': [
                    {
                        'paper_id': m.paper_id,
                        'title': m.title,
                        'year': m.year,
                        'significance': m.significance_description
                    }
                    for m in sorted(milestones, key=lambda m: -m.impact_score)[:3]
                ]
            }
            
            timeline.periods.append(period)
        
        # Identify breakthrough moments
        high_impact_milestones = [
            m for m in lineage.milestones
            if m.impact_score > 0.8 and m.milestone_type in [
                MilestoneType.METHODOLOGICAL_BREAKTHROUGH,
                MilestoneType.PARADIGM_SHIFT,
                MilestoneType.THEORETICAL_ADVANCE
            ]
        ]
        
        timeline.breakthrough_moments = [
            {
                'year': m.year,
                'title': m.title,
                'type': m.milestone_type.value,
                'description': m.significance_description
            }
            for m in high_impact_milestones
        ]
    
    async def _generate_timeline_narrative_with_llm(
        self,
        timeline: ResearchTimeline,
        lineage: IntellectualLineage
    ):
        """Generate timeline narrative using LLM."""
        # Get timeline narrative template
        template = self._template_manager.get_template(PromptType.TIMELINE_NARRATIVE)
        
        # Prepare timeline data
        timeline_data = {
            'periods': timeline.periods,
            'breakthrough_moments': timeline.breakthrough_moments,
            'total_span': timeline.end_year - timeline.start_year
        }
        
        template_vars = {
            'timeline_data': json.dumps(timeline_data, indent=2),
            'domain': timeline.domain,
            'key_researchers': ', '.join(lineage.key_researchers[:5]),
            'timeline_span': f"{timeline.start_year}-{timeline.end_year}",
            'breakthroughs': [bm['title'] for bm in timeline.breakthrough_moments[:3]],
            'early_years': f"{timeline.start_year}-{timeline.start_year+10}",
            'early_papers_context': 'Foundational work establishing key concepts',
            'middle_years': f"{timeline.start_year+10}-{timeline.end_year-10}" if timeline.end_year - timeline.start_year > 20 else '',
            'middle_papers_context': 'Methodological development and expansion',
            'breakthrough_years': f"{timeline.end_year-15}-{timeline.end_year-5}" if timeline.end_year - timeline.start_year > 15 else '',
            'breakthrough_papers_context': 'Major breakthroughs and applications',
            'recent_years': f"{timeline.end_year-10}-{timeline.end_year}",
            'recent_papers_context': 'Recent developments and new directions'
        }
        
        try:
            # Make LLM request
            response = await self.llm_service.complete(
                template,
                use_cache=True,
                temperature=0.3,  # Slightly higher for narrative generation
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
            
            timeline.analysis_cost = response.usage.cost
            
            # Store the narrative
            timeline.timeline_narrative = response.content
            timeline.analysis_confidence = 0.8  # Narratives are generally good
            
        except Exception as e:
            logger.warning(f"Failed to generate timeline narrative: {e}")
            timeline.timeline_narrative = "Failed to generate narrative"
            timeline.analysis_confidence = 0.0
    
    async def _fetch_paper_data(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Fetch paper data from available sources."""
        # Would integrate with existing data sources
        return None
    
    async def _get_paper_references(self, paper_id: str) -> List[str]:
        """Get list of papers this paper references."""
        # Would query citation database
        return []
    
    async def _get_paper_citations(self, paper_id: str) -> List[str]:
        """Get list of papers that cite this paper."""
        # Would query citation database
        return []
    
    async def _cache_lineage_analysis(self, lineage: IntellectualLineage):
        """Cache lineage analysis in Redis."""
        if not self.redis_manager:
            return
        
        try:
            cache_key = f"research_lineage:{lineage.lineage_id}"
            ttl_seconds = 30 * 24 * 3600  # 30 days
            
            await self.redis_manager.setex(
                cache_key,
                ttl_seconds,
                json.dumps(lineage.to_dict())
            )
        except Exception as e:
            logger.error(f"Failed to cache lineage analysis: {e}")
    
    async def _cache_timeline_analysis(self, timeline: ResearchTimeline):
        """Cache timeline analysis in Redis."""
        if not self.redis_manager:
            return
        
        try:
            cache_key = f"research_timeline:{timeline.timeline_id}"
            ttl_seconds = 30 * 24 * 3600  # 30 days
            
            timeline_dict = asdict(timeline)
            timeline_dict['analysis_timestamp'] = timeline.analysis_timestamp.isoformat() if timeline.analysis_timestamp else None
            
            await self.redis_manager.setex(
                cache_key,
                ttl_seconds,
                json.dumps(timeline_dict)
            )
        except Exception as e:
            logger.error(f"Failed to cache timeline analysis: {e}")
    
    async def get_trajectory_analytics(self) -> Dict[str, Any]:
        """Get analytics for trajectory analysis."""
        return {
            'total_lineages_analyzed': 0,
            'avg_lineage_size': 0,
            'avg_analysis_cost': 0.0,
            'trajectory_type_distribution': {},
            'milestone_type_distribution': {},
            'avg_analysis_confidence': 0.0
        }


# Global service instance
_trajectory_service: Optional[ResearchTrajectoryService] = None


async def get_trajectory_service() -> ResearchTrajectoryService:
    """Get or create the global research trajectory service."""
    global _trajectory_service
    
    if _trajectory_service is None:
        _trajectory_service = ResearchTrajectoryService()
        await _trajectory_service.initialize()
    
    return _trajectory_service