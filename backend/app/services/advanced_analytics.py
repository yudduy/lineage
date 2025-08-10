"""
Advanced Analytics and Content Enrichment System.

This module provides a unified intellectual lineage analysis platform that combines
OpenAlex, Semantic Scholar, LLM services, and Neo4j graph database into a comprehensive
research intelligence system with enterprise-grade performance and scalability.
"""

import asyncio
import json
import time
import hashlib
from typing import Any, Dict, List, Optional, Set, Tuple, Union, AsyncIterator
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from ..core.config import get_settings
from ..db.redis import RedisManager, get_redis_manager
from ..db.neo4j_advanced import AdvancedNeo4jManager, GraphAlgorithm, CommunityAlgorithm
from ..services.openalex import OpenAlexClient
from ..services.semantic_scholar import SemanticScholarClient
from ..services.llm_service_enhanced import EnhancedLLMService
from ..services.content_enrichment import ContentEnrichmentService
from ..utils.logger import get_logger
from ..utils.exceptions import APIError, ValidationError

logger = get_logger(__name__)


class AnalysisDepth(str, Enum):
    """Depth levels for citation network analysis."""
    SHALLOW = "shallow"  # 1-2 levels
    MODERATE = "moderate"  # 3-4 levels
    DEEP = "deep"  # 5-6 levels
    COMPREHENSIVE = "comprehensive"  # 7+ levels


class ResearchTrendType(str, Enum):
    """Types of research trends to detect."""
    EMERGING = "emerging"
    DECLINING = "declining"
    STABLE = "stable"
    BREAKTHROUGH = "breakthrough"
    CONVERGENT = "convergent"


@dataclass
class IntellectualLineage:
    """Represents the intellectual lineage of a research work."""
    root_paper_id: str
    depth: int
    total_papers: int
    total_citations: int
    key_milestones: List[Dict[str, Any]]
    evolution_path: List[List[str]]  # Ordered paths of paper IDs
    knowledge_flows: Dict[str, Dict[str, float]]  # Source -> Target -> Weight
    temporal_patterns: Dict[str, Any]
    impact_propagation: Dict[str, float]
    research_communities: List[Dict[str, Any]]
    predicted_trajectory: Optional[Dict[str, Any]] = None
    
    
@dataclass
class ResearchCommunity:
    """Represents a research community or cluster."""
    community_id: str
    name: str
    size: int
    core_members: List[str]  # Author IDs
    key_papers: List[str]  # Paper IDs
    research_themes: List[str]
    temporal_activity: Dict[str, int]  # Year -> Activity count
    cross_disciplinary_links: Dict[str, float]  # Other community ID -> Strength
    influence_score: float
    emerging_topics: List[str]
    collaboration_density: float
    

@dataclass
class ContentIntelligence:
    """Enhanced content analysis results."""
    paper_id: str
    research_themes: List[Dict[str, float]]  # Theme -> Relevance score
    significance_score: float
    novelty_score: float
    impact_potential: float
    research_gaps: List[str]
    methodological_approaches: List[str]
    comparative_advantages: Dict[str, Any]
    timeline_narrative: str
    key_contributions: List[str]
    future_directions: List[str]


@dataclass
class PerformanceMetrics:
    """Performance tracking for analytics operations."""
    operation_id: str
    start_time: float
    end_time: Optional[float] = None
    total_papers_processed: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    api_calls: Dict[str, int] = field(default_factory=dict)
    processing_stages: Dict[str, float] = field(default_factory=dict)
    memory_usage_mb: float = 0
    
    @property
    def duration(self) -> float:
        """Calculate operation duration."""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0


class AdvancedAnalyticsService:
    """
    Unified advanced analytics service orchestrating all data sources and providing
    comprehensive intellectual lineage analysis capabilities.
    """
    
    def __init__(
        self,
        redis_manager: Optional[RedisManager] = None,
        neo4j_manager: Optional[AdvancedNeo4jManager] = None,
        openalex_client: Optional[OpenAlexClient] = None,
        semantic_scholar_client: Optional[SemanticScholarClient] = None,
        llm_service: Optional[EnhancedLLMService] = None,
        content_enrichment: Optional[ContentEnrichmentService] = None
    ):
        self.redis = redis_manager
        self.neo4j = neo4j_manager
        self.openalex = openalex_client
        self.semantic_scholar = semantic_scholar_client
        self.llm = llm_service
        self.content_enrichment = content_enrichment
        
        # Performance optimization
        self._cache_ttl = 3600  # 1 hour default
        self._batch_size = 50
        self._max_concurrent_tasks = 10
        self._executor = ThreadPoolExecutor(max_workers=20)
        
        # Monitoring
        self._active_operations: Dict[str, PerformanceMetrics] = {}
        self._completed_operations: deque = deque(maxlen=1000)
        
        # Cache warming queue
        self._cache_warm_queue: asyncio.Queue = asyncio.Queue()
        self._cache_warm_task: Optional[asyncio.Task] = None
        
        self.settings = get_settings()
        
    async def initialize(self):
        """Initialize all service dependencies."""
        # Initialize services if not provided
        if not self.redis:
            self.redis = await get_redis_manager()
            
        if not self.neo4j:
            self.neo4j = AdvancedNeo4jManager()
            await self.neo4j.initialize()
            
        if not self.openalex:
            self.openalex = OpenAlexClient()
            await self.openalex.initialize()
            
        if not self.semantic_scholar:
            self.semantic_scholar = SemanticScholarClient()
            await self.semantic_scholar.initialize()
            
        if not self.llm:
            self.llm = EnhancedLLMService()
            await self.llm.initialize()
            
        if not self.content_enrichment:
            self.content_enrichment = ContentEnrichmentService()
            await self.content_enrichment.initialize()
            
        # Start background tasks
        self._cache_warm_task = asyncio.create_task(self._cache_warming_worker())
        
        logger.info("Advanced Analytics Service initialized successfully")
        
    async def close(self):
        """Clean up resources."""
        if self._cache_warm_task:
            self._cache_warm_task.cancel()
            
        if self.neo4j:
            await self.neo4j.close()
            
        if self.openalex:
            await self.openalex.close()
            
        if self.semantic_scholar:
            await self.semantic_scholar.close()
            
        self._executor.shutdown(wait=False)
        
    # ==================== Intellectual Lineage Analysis ====================
    
    async def analyze_intellectual_lineage(
        self,
        paper_id: str,
        depth: AnalysisDepth = AnalysisDepth.MODERATE,
        include_predictions: bool = True,
        enrich_content: bool = True
    ) -> IntellectualLineage:
        """
        Comprehensive intellectual lineage analysis combining citation networks,
        temporal patterns, and research evolution.
        """
        operation_id = f"lineage_{paper_id}_{int(time.time())}"
        metrics = PerformanceMetrics(
            operation_id=operation_id,
            start_time=time.time()
        )
        self._active_operations[operation_id] = metrics
        
        try:
            # Check cache first
            cache_key = f"lineage:{paper_id}:{depth.value}:{include_predictions}:{enrich_content}"
            cached = await self._get_cached_result(cache_key)
            if cached:
                metrics.cache_hits += 1
                return IntellectualLineage(**cached)
            
            metrics.cache_misses += 1
            
            # Determine analysis depth
            max_depth = self._get_max_depth(depth)
            
            # Phase 1: Build citation network
            metrics.processing_stages['network_building'] = time.time()
            citation_network = await self._build_citation_network(
                paper_id, max_depth, metrics
            )
            
            # Phase 2: Identify key milestones and evolution paths
            metrics.processing_stages['milestone_identification'] = time.time()
            milestones = await self._identify_research_milestones(
                citation_network, metrics
            )
            evolution_paths = await self._trace_evolution_paths(
                citation_network, paper_id, metrics
            )
            
            # Phase 3: Analyze knowledge flows
            metrics.processing_stages['knowledge_flow_analysis'] = time.time()
            knowledge_flows = await self._analyze_knowledge_flows(
                citation_network, metrics
            )
            
            # Phase 4: Temporal pattern analysis
            metrics.processing_stages['temporal_analysis'] = time.time()
            temporal_patterns = await self._analyze_temporal_patterns(
                citation_network, metrics
            )
            
            # Phase 5: Impact propagation modeling
            metrics.processing_stages['impact_modeling'] = time.time()
            impact_propagation = await self._model_impact_propagation(
                citation_network, paper_id, metrics
            )
            
            # Phase 6: Research community detection
            metrics.processing_stages['community_detection'] = time.time()
            communities = await self._detect_research_communities(
                citation_network, metrics
            )
            
            # Phase 7: Trajectory prediction (optional)
            predicted_trajectory = None
            if include_predictions:
                metrics.processing_stages['trajectory_prediction'] = time.time()
                predicted_trajectory = await self._predict_research_trajectory(
                    citation_network, temporal_patterns, metrics
                )
            
            # Phase 8: Content enrichment (optional)
            if enrich_content and self.llm:
                metrics.processing_stages['content_enrichment'] = time.time()
                await self._enrich_lineage_content(
                    citation_network, milestones, metrics
                )
            
            # Build result
            result = IntellectualLineage(
                root_paper_id=paper_id,
                depth=max_depth,
                total_papers=len(citation_network['nodes']),
                total_citations=sum(n.get('citations', 0) for n in citation_network['nodes'].values()),
                key_milestones=milestones,
                evolution_path=evolution_paths,
                knowledge_flows=knowledge_flows,
                temporal_patterns=temporal_patterns,
                impact_propagation=impact_propagation,
                research_communities=communities,
                predicted_trajectory=predicted_trajectory
            )
            
            # Cache result
            await self._cache_result(cache_key, result.__dict__, ttl=self._cache_ttl * 2)
            
            # Update metrics
            metrics.end_time = time.time()
            metrics.total_papers_processed = len(citation_network['nodes'])
            self._completed_operations.append(metrics)
            del self._active_operations[operation_id]
            
            # Trigger background cache warming for related papers
            await self._queue_cache_warming(citation_network['nodes'].keys())
            
            return result
            
        except Exception as e:
            logger.error(f"Error in intellectual lineage analysis: {e}")
            metrics.end_time = time.time()
            self._completed_operations.append(metrics)
            if operation_id in self._active_operations:
                del self._active_operations[operation_id]
            raise
    
    async def _build_citation_network(
        self,
        root_paper_id: str,
        max_depth: int,
        metrics: PerformanceMetrics
    ) -> Dict[str, Any]:
        """Build multi-depth citation network."""
        network = {
            'nodes': {},
            'edges': [],
            'levels': defaultdict(list)
        }
        
        visited = set()
        queue = deque([(root_paper_id, 0)])
        
        while queue:
            # Process in batches for efficiency
            batch = []
            for _ in range(min(self._batch_size, len(queue))):
                if queue:
                    batch.append(queue.popleft())
            
            if not batch:
                break
            
            # Parallel processing of batch
            tasks = []
            for paper_id, level in batch:
                if paper_id not in visited and level <= max_depth:
                    visited.add(paper_id)
                    tasks.append(self._process_paper_node(
                        paper_id, level, network, queue, metrics
                    ))
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        
        return network
    
    async def _process_paper_node(
        self,
        paper_id: str,
        level: int,
        network: Dict,
        queue: deque,
        metrics: PerformanceMetrics
    ):
        """Process individual paper node in citation network."""
        try:
            # Fetch paper data from multiple sources
            paper_data = await self._fetch_paper_data(paper_id, metrics)
            
            if paper_data:
                # Add to network
                network['nodes'][paper_id] = {
                    'id': paper_id,
                    'level': level,
                    'data': paper_data,
                    'citations': paper_data.get('citation_count', 0),
                    'year': paper_data.get('year'),
                    'title': paper_data.get('title'),
                    'authors': paper_data.get('authors', []),
                    'venue': paper_data.get('venue'),
                    'fields': paper_data.get('fields', [])
                }
                network['levels'][level].append(paper_id)
                
                # Add citations and references to queue
                if level < 6:  # Limit depth to prevent explosion
                    for ref_id in paper_data.get('references', [])[:20]:  # Limit refs
                        network['edges'].append({
                            'source': paper_id,
                            'target': ref_id,
                            'type': 'cites'
                        })
                        queue.append((ref_id, level + 1))
                    
                    for cite_id in paper_data.get('citations', [])[:10]:  # Limit citations
                        network['edges'].append({
                            'source': cite_id,
                            'target': paper_id,
                            'type': 'cites'
                        })
                        queue.append((cite_id, level + 1))
                        
        except Exception as e:
            logger.warning(f"Error processing paper {paper_id}: {e}")
    
    async def _fetch_paper_data(
        self,
        paper_id: str,
        metrics: PerformanceMetrics
    ) -> Optional[Dict[str, Any]]:
        """Fetch paper data from multiple sources with fallback."""
        # Try cache first
        cache_key = f"paper_data:{paper_id}"
        cached = await self._get_cached_result(cache_key)
        if cached:
            metrics.cache_hits += 1
            return cached
        
        metrics.cache_misses += 1
        paper_data = {}
        
        # Try OpenAlex first
        try:
            if self.openalex:
                metrics.api_calls['openalex'] = metrics.api_calls.get('openalex', 0) + 1
                openalex_data = await self.openalex.get_work(paper_id)
                if openalex_data:
                    paper_data.update(self._normalize_openalex_data(openalex_data))
        except Exception as e:
            logger.debug(f"OpenAlex fetch failed for {paper_id}: {e}")
        
        # Enhance with Semantic Scholar
        try:
            if self.semantic_scholar and not paper_data:
                metrics.api_calls['semantic_scholar'] = metrics.api_calls.get('semantic_scholar', 0) + 1
                ss_data = await self.semantic_scholar.get_paper(paper_id)
                if ss_data:
                    paper_data.update(self._normalize_semantic_scholar_data(ss_data))
        except Exception as e:
            logger.debug(f"Semantic Scholar fetch failed for {paper_id}: {e}")
        
        # Cache if we got data
        if paper_data:
            await self._cache_result(cache_key, paper_data, ttl=self._cache_ttl)
        
        return paper_data if paper_data else None
    
    async def _identify_research_milestones(
        self,
        network: Dict[str, Any],
        metrics: PerformanceMetrics
    ) -> List[Dict[str, Any]]:
        """Identify key milestone papers in the network."""
        milestones = []
        
        # Use graph metrics to identify important papers
        if self.neo4j:
            try:
                # Create temporary graph in Neo4j
                graph_id = await self._create_temp_graph(network)
                
                # Run centrality algorithms
                pagerank = await self.neo4j.run_algorithm(
                    GraphAlgorithm.PAGERANK,
                    graph_name=graph_id
                )
                
                betweenness = await self.neo4j.run_algorithm(
                    GraphAlgorithm.BETWEENNESS_CENTRALITY,
                    graph_name=graph_id
                )
                
                # Identify milestones based on multiple criteria
                for node_id, node_data in network['nodes'].items():
                    score = 0
                    reasons = []
                    
                    # High PageRank
                    if node_id in pagerank and pagerank[node_id] > 0.05:
                        score += pagerank[node_id] * 100
                        reasons.append("high_influence")
                    
                    # High betweenness (bridge papers)
                    if node_id in betweenness and betweenness[node_id] > 0.1:
                        score += betweenness[node_id] * 50
                        reasons.append("bridge_paper")
                    
                    # High citation count
                    if node_data.get('citations', 0) > 100:
                        score += min(node_data['citations'] / 10, 50)
                        reasons.append("highly_cited")
                    
                    # First papers in field
                    if node_data.get('year') and node_data['year'] < 2000:
                        score += 20
                        reasons.append("foundational")
                    
                    if score > 30:  # Threshold for milestone
                        milestones.append({
                            'paper_id': node_id,
                            'title': node_data.get('title'),
                            'year': node_data.get('year'),
                            'score': score,
                            'reasons': reasons,
                            'metrics': {
                                'pagerank': pagerank.get(node_id, 0),
                                'betweenness': betweenness.get(node_id, 0),
                                'citations': node_data.get('citations', 0)
                            }
                        })
                
                # Clean up temp graph
                await self._cleanup_temp_graph(graph_id)
                
            except Exception as e:
                logger.error(f"Error identifying milestones with Neo4j: {e}")
        
        # Fallback to simple heuristics if Neo4j not available
        if not milestones:
            for node_id, node_data in network['nodes'].items():
                if node_data.get('citations', 0) > 50:
                    milestones.append({
                        'paper_id': node_id,
                        'title': node_data.get('title'),
                        'year': node_data.get('year'),
                        'score': node_data['citations'],
                        'reasons': ['highly_cited']
                    })
        
        # Sort by score and limit
        milestones.sort(key=lambda x: x['score'], reverse=True)
        return milestones[:20]
    
    async def _trace_evolution_paths(
        self,
        network: Dict[str, Any],
        root_id: str,
        metrics: PerformanceMetrics
    ) -> List[List[str]]:
        """Trace research evolution paths through citation network."""
        paths = []
        
        # Build adjacency list
        adj_list = defaultdict(list)
        for edge in network['edges']:
            if edge['type'] == 'cites':
                adj_list[edge['source']].append(edge['target'])
        
        # Find paths using DFS with temporal ordering
        def find_paths(current: str, target: str, path: List[str], visited: Set[str]):
            if len(path) > 10:  # Limit path length
                return
            
            if current == target:
                paths.append(path.copy())
                return
            
            visited.add(current)
            
            # Get references sorted by year
            refs = adj_list.get(current, [])
            refs_with_year = []
            for ref in refs:
                if ref in network['nodes'] and ref not in visited:
                    year = network['nodes'][ref].get('year', 9999)
                    refs_with_year.append((ref, year))
            
            # Sort by year (oldest first for evolution)
            refs_with_year.sort(key=lambda x: x[1])
            
            for ref, _ in refs_with_year[:5]:  # Limit branching
                path.append(ref)
                find_paths(ref, target, path, visited.copy())
                path.pop()
        
        # Find influential end nodes
        end_nodes = [
            node_id for node_id, node_data in network['nodes'].items()
            if node_data.get('citations', 0) > 100 and node_id != root_id
        ][:5]
        
        # Trace paths to influential papers
        for end_node in end_nodes:
            find_paths(root_id, end_node, [root_id], set())
        
        # Sort by path importance (sum of citations along path)
        paths_with_score = []
        for path in paths:
            score = sum(
                network['nodes'].get(p, {}).get('citations', 0)
                for p in path
            )
            paths_with_score.append((path, score))
        
        paths_with_score.sort(key=lambda x: x[1], reverse=True)
        
        return [path for path, _ in paths_with_score[:10]]
    
    async def _analyze_knowledge_flows(
        self,
        network: Dict[str, Any],
        metrics: PerformanceMetrics
    ) -> Dict[str, Dict[str, float]]:
        """Analyze knowledge flows between research areas."""
        flows = defaultdict(lambda: defaultdict(float))
        
        # Group papers by field/topic
        field_papers = defaultdict(list)
        for node_id, node_data in network['nodes'].items():
            fields = node_data.get('fields', ['unknown'])
            for field in fields:
                field_papers[field].append(node_id)
        
        # Calculate flow strength between fields
        for edge in network['edges']:
            source = edge['source']
            target = edge['target']
            
            if source in network['nodes'] and target in network['nodes']:
                source_fields = network['nodes'][source].get('fields', ['unknown'])
                target_fields = network['nodes'][target].get('fields', ['unknown'])
                
                # Weight by citation count and year difference
                source_citations = network['nodes'][source].get('citations', 1)
                target_citations = network['nodes'][target].get('citations', 1)
                
                weight = np.log1p(source_citations) * np.log1p(target_citations)
                
                for s_field in source_fields:
                    for t_field in target_fields:
                        if s_field != t_field:  # Inter-field flow
                            flows[s_field][t_field] += weight
        
        # Normalize flows
        max_flow = max(
            max(field_flows.values()) if field_flows else 0
            for field_flows in flows.values()
        )
        
        if max_flow > 0:
            for source_field in flows:
                for target_field in flows[source_field]:
                    flows[source_field][target_field] /= max_flow
        
        return dict(flows)
    
    async def _analyze_temporal_patterns(
        self,
        network: Dict[str, Any],
        metrics: PerformanceMetrics
    ) -> Dict[str, Any]:
        """Analyze temporal patterns in the citation network."""
        patterns = {
            'publication_timeline': defaultdict(int),
            'citation_accumulation': defaultdict(int),
            'field_evolution': defaultdict(lambda: defaultdict(int)),
            'burst_periods': [],
            'growth_rate': {}
        }
        
        # Publication timeline
        for node_data in network['nodes'].values():
            year = node_data.get('year')
            if year:
                patterns['publication_timeline'][year] += 1
                
                # Field evolution
                for field in node_data.get('fields', []):
                    patterns['field_evolution'][year][field] += 1
                
                # Citation accumulation
                patterns['citation_accumulation'][year] += node_data.get('citations', 0)
        
        # Identify burst periods
        years = sorted(patterns['publication_timeline'].keys())
        if len(years) > 2:
            for i in range(1, len(years) - 1):
                prev_year = years[i - 1]
                curr_year = years[i]
                next_year = years[i + 1]
                
                prev_count = patterns['publication_timeline'][prev_year]
                curr_count = patterns['publication_timeline'][curr_year]
                next_count = patterns['publication_timeline'][next_year]
                
                # Detect burst (significant increase)
                if curr_count > prev_count * 1.5 and curr_count > next_count * 0.8:
                    patterns['burst_periods'].append({
                        'year': curr_year,
                        'intensity': curr_count / max(prev_count, 1),
                        'papers': curr_count
                    })
        
        # Calculate growth rates
        for i in range(1, len(years)):
            prev_year = years[i - 1]
            curr_year = years[i]
            
            prev_count = patterns['publication_timeline'][prev_year]
            curr_count = patterns['publication_timeline'][curr_year]
            
            if prev_count > 0:
                growth_rate = (curr_count - prev_count) / prev_count
                patterns['growth_rate'][curr_year] = growth_rate
        
        return patterns
    
    async def _model_impact_propagation(
        self,
        network: Dict[str, Any],
        root_id: str,
        metrics: PerformanceMetrics
    ) -> Dict[str, float]:
        """Model how impact propagates through the citation network."""
        impact_scores = defaultdict(float)
        
        # Initialize root paper with impact 1.0
        impact_scores[root_id] = 1.0
        
        # Build reverse adjacency list (who cites whom)
        cited_by = defaultdict(list)
        for edge in network['edges']:
            if edge['type'] == 'cites':
                cited_by[edge['target']].append(edge['source'])
        
        # Propagate impact using modified PageRank algorithm
        damping = 0.85
        iterations = 10
        
        for _ in range(iterations):
            new_scores = defaultdict(float)
            
            for node_id in network['nodes']:
                # Base impact
                new_scores[node_id] = (1 - damping) * impact_scores.get(node_id, 0)
                
                # Propagated impact from papers citing this one
                for citer in cited_by.get(node_id, []):
                    if citer in impact_scores:
                        # Weight by citation count and year
                        citer_data = network['nodes'].get(citer, {})
                        weight = np.log1p(citer_data.get('citations', 0)) / 10
                        new_scores[node_id] += damping * impact_scores[citer] * weight
            
            impact_scores = new_scores
        
        # Normalize scores
        max_score = max(impact_scores.values()) if impact_scores else 1
        for node_id in impact_scores:
            impact_scores[node_id] /= max_score
        
        return dict(impact_scores)
    
    async def _detect_research_communities(
        self,
        network: Dict[str, Any],
        metrics: PerformanceMetrics
    ) -> List[Dict[str, Any]]:
        """Detect and characterize research communities."""
        communities = []
        
        if self.neo4j:
            try:
                # Create temporary graph
                graph_id = await self._create_temp_graph(network)
                
                # Run community detection
                community_result = await self.neo4j.detect_communities(
                    algorithm=CommunityAlgorithm.LOUVAIN,
                    graph_name=graph_id
                )
                
                # Group nodes by community
                community_nodes = defaultdict(list)
                for node_id, comm_id in community_result.items():
                    if node_id in network['nodes']:
                        community_nodes[comm_id].append(node_id)
                
                # Characterize each community
                for comm_id, node_ids in community_nodes.items():
                    if len(node_ids) < 3:  # Skip tiny communities
                        continue
                    
                    # Collect community statistics
                    total_citations = 0
                    all_fields = []
                    all_years = []
                    all_authors = set()
                    
                    for node_id in node_ids:
                        node_data = network['nodes'][node_id]
                        total_citations += node_data.get('citations', 0)
                        all_fields.extend(node_data.get('fields', []))
                        if node_data.get('year'):
                            all_years.append(node_data['year'])
                        for author in node_data.get('authors', []):
                            if isinstance(author, dict):
                                all_authors.add(author.get('id', author.get('name')))
                            else:
                                all_authors.add(author)
                    
                    # Find dominant research themes
                    field_counts = defaultdict(int)
                    for field in all_fields:
                        field_counts[field] += 1
                    
                    top_fields = sorted(
                        field_counts.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:5]
                    
                    # Generate community name using LLM if available
                    community_name = f"Community_{comm_id}"
                    if self.llm and top_fields:
                        try:
                            prompt = f"Generate a concise name for a research community focused on: {', '.join([f[0] for f in top_fields[:3]])}"
                            llm_response = await self.llm.generate(prompt, max_tokens=20)
                            if llm_response:
                                community_name = llm_response.strip()
                        except Exception as e:
                            logger.debug(f"Failed to generate community name: {e}")
                    
                    communities.append({
                        'id': str(comm_id),
                        'name': community_name,
                        'size': len(node_ids),
                        'total_citations': total_citations,
                        'avg_citations': total_citations / len(node_ids),
                        'research_themes': [f[0] for f in top_fields],
                        'year_range': (min(all_years), max(all_years)) if all_years else None,
                        'key_papers': sorted(
                            node_ids,
                            key=lambda x: network['nodes'][x].get('citations', 0),
                            reverse=True
                        )[:5],
                        'author_count': len(all_authors)
                    })
                
                # Clean up
                await self._cleanup_temp_graph(graph_id)
                
            except Exception as e:
                logger.error(f"Error detecting communities: {e}")
        
        # Sort by size and importance
        communities.sort(key=lambda x: x['total_citations'], reverse=True)
        
        return communities[:20]
    
    async def _predict_research_trajectory(
        self,
        network: Dict[str, Any],
        temporal_patterns: Dict[str, Any],
        metrics: PerformanceMetrics
    ) -> Dict[str, Any]:
        """Predict future research trajectory based on patterns."""
        prediction = {
            'next_breakthrough_areas': [],
            'emerging_collaborations': [],
            'predicted_citation_growth': {},
            'convergence_points': [],
            'confidence': 0.0
        }
        
        if not self.llm:
            return prediction
        
        try:
            # Analyze growth patterns
            growth_rates = temporal_patterns.get('growth_rate', {})
            field_evolution = temporal_patterns.get('field_evolution', {})
            
            # Identify accelerating fields
            recent_years = sorted(field_evolution.keys())[-5:]
            field_acceleration = defaultdict(list)
            
            for year in recent_years:
                for field, count in field_evolution[year].items():
                    field_acceleration[field].append(count)
            
            # Find fields with increasing growth
            emerging_fields = []
            for field, counts in field_acceleration.items():
                if len(counts) >= 3:
                    # Simple linear regression for trend
                    x = np.arange(len(counts))
                    y = np.array(counts)
                    if len(x) > 1:
                        slope = np.polyfit(x, y, 1)[0]
                        if slope > 0:
                            emerging_fields.append({
                                'field': field,
                                'growth_rate': float(slope),
                                'recent_papers': counts[-1]
                            })
            
            # Sort by growth rate
            emerging_fields.sort(key=lambda x: x['growth_rate'], reverse=True)
            prediction['next_breakthrough_areas'] = emerging_fields[:5]
            
            # Predict citation growth using time series analysis
            citation_timeline = temporal_patterns.get('citation_accumulation', {})
            if len(citation_timeline) >= 3:
                years = sorted(citation_timeline.keys())
                citations = [citation_timeline[y] for y in years]
                
                # Simple exponential smoothing for prediction
                alpha = 0.3
                smoothed = [citations[0]]
                for i in range(1, len(citations)):
                    smoothed.append(alpha * citations[i] + (1 - alpha) * smoothed[-1])
                
                # Predict next 3 years
                last_value = smoothed[-1]
                last_year = years[-1]
                for i in range(1, 4):
                    predicted_value = last_value * (1 + growth_rates.get(last_year, 0.1))
                    prediction['predicted_citation_growth'][last_year + i] = int(predicted_value)
                    last_value = predicted_value
            
            # Identify convergence points using knowledge flows
            # (Simplified - would need more sophisticated analysis in production)
            prediction['convergence_points'] = []
            
            # Calculate confidence based on data quality
            data_points = len(network['nodes'])
            time_span = len(temporal_patterns.get('publication_timeline', {}))
            prediction['confidence'] = min(
                0.95,
                0.3 + (min(data_points, 100) / 100) * 0.35 + 
                (min(time_span, 20) / 20) * 0.35
            )
            
        except Exception as e:
            logger.error(f"Error predicting trajectory: {e}")
        
        return prediction
    
    async def _enrich_lineage_content(
        self,
        network: Dict[str, Any],
        milestones: List[Dict[str, Any]],
        metrics: PerformanceMetrics
    ):
        """Enrich lineage analysis with LLM-generated insights."""
        if not self.llm:
            return
        
        try:
            # Enrich milestone descriptions
            for milestone in milestones[:5]:  # Limit to top 5 for cost
                paper_id = milestone['paper_id']
                if paper_id in network['nodes']:
                    node_data = network['nodes'][paper_id]
                    
                    prompt = f"""
                    Analyze this milestone paper in a research lineage:
                    Title: {node_data.get('title')}
                    Year: {node_data.get('year')}
                    Citations: {node_data.get('citations')}
                    Fields: {', '.join(node_data.get('fields', []))}
                    
                    Explain why this is a milestone paper in 2-3 sentences.
                    """
                    
                    response = await self.llm.generate(prompt, max_tokens=100)
                    if response:
                        milestone['significance'] = response.strip()
                        
        except Exception as e:
            logger.error(f"Error enriching content: {e}")
    
    # ==================== Research Community Intelligence ====================
    
    async def analyze_research_community(
        self,
        community_identifier: Union[str, List[str]],
        depth: str = "moderate"
    ) -> ResearchCommunity:
        """
        Comprehensive analysis of a research community including members,
        themes, evolution, and cross-disciplinary connections.
        """
        operation_id = f"community_{hashlib.md5(str(community_identifier).encode()).hexdigest()}"
        metrics = PerformanceMetrics(
            operation_id=operation_id,
            start_time=time.time()
        )
        
        try:
            # Implementation would follow similar pattern to lineage analysis
            # This is a placeholder for the community analysis logic
            pass
            
        except Exception as e:
            logger.error(f"Error in community analysis: {e}")
            raise
    
    # ==================== Content Intelligence ====================
    
    async def analyze_content_intelligence(
        self,
        paper_id: str,
        comparative_papers: Optional[List[str]] = None
    ) -> ContentIntelligence:
        """
        Deep content analysis including theme extraction, significance scoring,
        research gap identification, and comparative analysis.
        """
        operation_id = f"content_{paper_id}_{int(time.time())}"
        metrics = PerformanceMetrics(
            operation_id=operation_id,
            start_time=time.time()
        )
        
        try:
            # Check cache
            cache_key = f"content_intelligence:{paper_id}"
            cached = await self._get_cached_result(cache_key)
            if cached:
                metrics.cache_hits += 1
                return ContentIntelligence(**cached)
            
            # Fetch paper data
            paper_data = await self._fetch_paper_data(paper_id, metrics)
            if not paper_data:
                raise ValueError(f"Paper {paper_id} not found")
            
            # Extract research themes
            themes = await self._extract_research_themes(paper_data, metrics)
            
            # Calculate significance scores
            significance = await self._calculate_significance_score(paper_data, metrics)
            novelty = await self._calculate_novelty_score(paper_data, metrics)
            impact_potential = await self._calculate_impact_potential(paper_data, metrics)
            
            # Identify research gaps
            gaps = await self._identify_research_gaps(paper_data, metrics)
            
            # Extract methodological approaches
            methods = await self._extract_methodological_approaches(paper_data, metrics)
            
            # Comparative analysis if requested
            comparative_advantages = {}
            if comparative_papers:
                comparative_advantages = await self._perform_comparative_analysis(
                    paper_id, comparative_papers, metrics
                )
            
            # Generate timeline narrative
            narrative = await self._generate_timeline_narrative(paper_data, metrics)
            
            # Extract key contributions and future directions
            contributions = await self._extract_key_contributions(paper_data, metrics)
            future_directions = await self._identify_future_directions(paper_data, metrics)
            
            result = ContentIntelligence(
                paper_id=paper_id,
                research_themes=themes,
                significance_score=significance,
                novelty_score=novelty,
                impact_potential=impact_potential,
                research_gaps=gaps,
                methodological_approaches=methods,
                comparative_advantages=comparative_advantages,
                timeline_narrative=narrative,
                key_contributions=contributions,
                future_directions=future_directions
            )
            
            # Cache result
            await self._cache_result(cache_key, result.__dict__, ttl=self._cache_ttl)
            
            metrics.end_time = time.time()
            self._completed_operations.append(metrics)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in content intelligence analysis: {e}")
            metrics.end_time = time.time()
            self._completed_operations.append(metrics)
            raise
    
    # ==================== Performance Optimization ====================
    
    async def _cache_warming_worker(self):
        """Background worker for cache warming."""
        while True:
            try:
                # Get paper IDs to warm
                paper_ids = []
                for _ in range(10):  # Process up to 10 at a time
                    try:
                        paper_id = await asyncio.wait_for(
                            self._cache_warm_queue.get(),
                            timeout=1.0
                        )
                        paper_ids.append(paper_id)
                    except asyncio.TimeoutError:
                        break
                
                if paper_ids:
                    # Warm cache for these papers
                    tasks = [
                        self._fetch_paper_data(pid, PerformanceMetrics(
                            operation_id=f"cache_warm_{pid}",
                            start_time=time.time()
                        ))
                        for pid in paper_ids
                    ]
                    await asyncio.gather(*tasks, return_exceptions=True)
                
                await asyncio.sleep(5)  # Wait before next batch
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache warming: {e}")
                await asyncio.sleep(10)
    
    async def _queue_cache_warming(self, paper_ids: Union[List[str], Set[str]]):
        """Queue papers for cache warming."""
        for paper_id in list(paper_ids)[:50]:  # Limit queue size
            try:
                self._cache_warm_queue.put_nowait(paper_id)
            except asyncio.QueueFull:
                break
    
    async def _get_cached_result(self, key: str) -> Optional[Any]:
        """Get cached result."""
        if not self.redis:
            return None
        
        try:
            result = await self.redis.get(key)
            if result:
                return json.loads(result)
        except Exception as e:
            logger.debug(f"Cache get error: {e}")
        
        return None
    
    async def _cache_result(self, key: str, value: Any, ttl: int = None):
        """Cache result."""
        if not self.redis:
            return
        
        try:
            ttl = ttl or self._cache_ttl
            await self.redis.set(
                key,
                json.dumps(value, default=str),
                ttl=ttl
            )
        except Exception as e:
            logger.debug(f"Cache set error: {e}")
    
    # ==================== Helper Methods ====================
    
    def _get_max_depth(self, depth: AnalysisDepth) -> int:
        """Convert analysis depth enum to max traversal depth."""
        depth_map = {
            AnalysisDepth.SHALLOW: 2,
            AnalysisDepth.MODERATE: 4,
            AnalysisDepth.DEEP: 6,
            AnalysisDepth.COMPREHENSIVE: 8
        }
        return depth_map.get(depth, 4)
    
    def _normalize_openalex_data(self, data: Dict) -> Dict:
        """Normalize OpenAlex data format."""
        return {
            'id': data.get('id'),
            'title': data.get('title'),
            'year': data.get('publication_year'),
            'authors': [
                {'id': a.get('author', {}).get('id'), 'name': a.get('author', {}).get('display_name')}
                for a in data.get('authorships', [])
            ],
            'venue': data.get('host_venue', {}).get('display_name'),
            'fields': [c.get('display_name') for c in data.get('concepts', [])],
            'citation_count': data.get('cited_by_count', 0),
            'references': [r for r in data.get('referenced_works', [])],
            'citations': []  # Would need separate call
        }
    
    def _normalize_semantic_scholar_data(self, data: Dict) -> Dict:
        """Normalize Semantic Scholar data format."""
        return {
            'id': data.get('paperId'),
            'title': data.get('title'),
            'year': data.get('year'),
            'authors': [
                {'id': a.get('authorId'), 'name': a.get('name')}
                for a in data.get('authors', [])
            ],
            'venue': data.get('venue'),
            'fields': data.get('fieldsOfStudy', []),
            'citation_count': data.get('citationCount', 0),
            'references': [r.get('paperId') for r in data.get('references', [])],
            'citations': [c.get('paperId') for c in data.get('citations', [])]
        }
    
    async def _create_temp_graph(self, network: Dict) -> str:
        """Create temporary graph in Neo4j for analysis."""
        # This would create a temporary graph projection in Neo4j
        # Implementation depends on Neo4j setup
        return f"temp_graph_{int(time.time())}"
    
    async def _cleanup_temp_graph(self, graph_id: str):
        """Clean up temporary graph from Neo4j."""
        # This would remove the temporary graph projection
        pass
    
    async def _extract_research_themes(
        self,
        paper_data: Dict,
        metrics: PerformanceMetrics
    ) -> List[Dict[str, float]]:
        """Extract research themes from paper."""
        themes = []
        
        # Use fields/concepts as base themes
        for field in paper_data.get('fields', []):
            themes.append({'theme': field, 'score': 0.8})
        
        # Use LLM for deeper theme extraction if available
        if self.llm and paper_data.get('title'):
            try:
                prompt = f"Extract 3-5 key research themes from this paper title: {paper_data['title']}"
                response = await self.llm.generate(prompt, max_tokens=100)
                if response:
                    # Parse LLM response (simplified)
                    for line in response.split('\n'):
                        if line.strip():
                            themes.append({'theme': line.strip(), 'score': 0.6})
            except Exception as e:
                logger.debug(f"LLM theme extraction failed: {e}")
        
        return themes[:10]
    
    async def _calculate_significance_score(
        self,
        paper_data: Dict,
        metrics: PerformanceMetrics
    ) -> float:
        """Calculate paper significance score."""
        score = 0.0
        
        # Citation-based significance
        citations = paper_data.get('citation_count', 0)
        score += min(citations / 100, 0.5)  # Max 0.5 from citations
        
        # Venue significance (would need venue ranking data)
        if paper_data.get('venue'):
            score += 0.2
        
        # Author influence (would need author metrics)
        score += min(len(paper_data.get('authors', [])) * 0.05, 0.3)
        
        return min(score, 1.0)
    
    async def _calculate_novelty_score(
        self,
        paper_data: Dict,
        metrics: PerformanceMetrics
    ) -> float:
        """Calculate research novelty score."""
        # Simplified novelty calculation
        # In production, would compare with existing literature
        score = 0.5  # Base score
        
        # Adjust based on year (newer = potentially more novel)
        year = paper_data.get('year')
        if year and year > 2020:
            score += 0.2
        
        # Adjust based on field diversity
        fields = paper_data.get('fields', [])
        if len(fields) > 3:
            score += 0.2  # Interdisciplinary bonus
        
        return min(score, 1.0)
    
    async def _calculate_impact_potential(
        self,
        paper_data: Dict,
        metrics: PerformanceMetrics
    ) -> float:
        """Calculate potential future impact."""
        # Simplified - would use more sophisticated prediction in production
        base_score = 0.5
        
        # Recent papers with early citations show promise
        year = paper_data.get('year')
        citations = paper_data.get('citation_count', 0)
        
        if year and year > 2020 and citations > 10:
            years_since = 2024 - year
            citation_rate = citations / max(years_since, 1)
            base_score += min(citation_rate / 50, 0.5)
        
        return min(base_score, 1.0)
    
    async def _identify_research_gaps(
        self,
        paper_data: Dict,
        metrics: PerformanceMetrics
    ) -> List[str]:
        """Identify research gaps mentioned or implied."""
        gaps = []
        
        # Use LLM if available
        if self.llm and paper_data.get('title'):
            try:
                prompt = f"""
                Based on this paper title: {paper_data['title']}
                What research gaps might this paper be addressing? List 2-3 gaps.
                """
                response = await self.llm.generate(prompt, max_tokens=150)
                if response:
                    for line in response.split('\n'):
                        if line.strip() and not line.startswith('#'):
                            gaps.append(line.strip())
            except Exception as e:
                logger.debug(f"Gap identification failed: {e}")
        
        # Fallback to generic gaps
        if not gaps:
            gaps = ["Further empirical validation needed", "Cross-domain applications unexplored"]
        
        return gaps[:5]
    
    async def _extract_methodological_approaches(
        self,
        paper_data: Dict,
        metrics: PerformanceMetrics
    ) -> List[str]:
        """Extract methodological approaches used."""
        # Simplified - would use NLP on abstract/full text in production
        methods = []
        
        title = paper_data.get('title', '').lower()
        
        # Check for common methodological keywords
        method_keywords = {
            'empirical': 'Empirical Analysis',
            'theoretical': 'Theoretical Framework',
            'experimental': 'Experimental Study',
            'survey': 'Survey Research',
            'case study': 'Case Study Analysis',
            'meta-analysis': 'Meta-Analysis',
            'simulation': 'Simulation Study',
            'machine learning': 'Machine Learning',
            'deep learning': 'Deep Learning'
        }
        
        for keyword, method in method_keywords.items():
            if keyword in title:
                methods.append(method)
        
        if not methods:
            methods = ['Analytical Study']
        
        return methods
    
    async def _perform_comparative_analysis(
        self,
        paper_id: str,
        comparative_papers: List[str],
        metrics: PerformanceMetrics
    ) -> Dict[str, Any]:
        """Perform comparative analysis between papers."""
        # Simplified comparative analysis
        comparison = {
            'citation_advantage': 0,
            'novelty_advantage': 0,
            'unique_contributions': [],
            'shared_themes': []
        }
        
        # Would implement detailed comparison logic
        return comparison
    
    async def _generate_timeline_narrative(
        self,
        paper_data: Dict,
        metrics: PerformanceMetrics
    ) -> str:
        """Generate narrative description of research timeline."""
        year = paper_data.get('year', 'Unknown')
        title = paper_data.get('title', 'Unknown')
        citations = paper_data.get('citation_count', 0)
        
        narrative = f"Published in {year}, '{title}' has garnered {citations} citations."
        
        if self.llm:
            try:
                prompt = f"Write a brief narrative about this paper's place in research history: {title} ({year})"
                response = await self.llm.generate(prompt, max_tokens=100)
                if response:
                    narrative = response.strip()
            except Exception as e:
                logger.debug(f"Narrative generation failed: {e}")
        
        return narrative
    
    async def _extract_key_contributions(
        self,
        paper_data: Dict,
        metrics: PerformanceMetrics
    ) -> List[str]:
        """Extract key contributions of the paper."""
        contributions = []
        
        # Use LLM if available
        if self.llm and paper_data.get('title'):
            try:
                prompt = f"List 3 key contributions of a paper titled: {paper_data['title']}"
                response = await self.llm.generate(prompt, max_tokens=150)
                if response:
                    for line in response.split('\n'):
                        if line.strip() and not line.startswith('#'):
                            contributions.append(line.strip())
            except Exception as e:
                logger.debug(f"Contribution extraction failed: {e}")
        
        if not contributions:
            contributions = ["Novel approach to the problem", "Empirical validation of theory"]
        
        return contributions[:5]
    
    async def _identify_future_directions(
        self,
        paper_data: Dict,
        metrics: PerformanceMetrics
    ) -> List[str]:
        """Identify future research directions."""
        directions = []
        
        # Use LLM if available
        if self.llm and paper_data.get('title'):
            try:
                prompt = f"Suggest 3 future research directions based on: {paper_data['title']}"
                response = await self.llm.generate(prompt, max_tokens=150)
                if response:
                    for line in response.split('\n'):
                        if line.strip() and not line.startswith('#'):
                            directions.append(line.strip())
            except Exception as e:
                logger.debug(f"Future directions identification failed: {e}")
        
        if not directions:
            directions = ["Extension to other domains", "Longitudinal validation studies"]
        
        return directions[:5]
    
    # ==================== Performance Analytics ====================
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        active_ops = len(self._active_operations)
        completed_ops = len(self._completed_operations)
        
        # Calculate aggregate metrics
        total_duration = sum(op.duration for op in self._completed_operations)
        avg_duration = total_duration / completed_ops if completed_ops > 0 else 0
        
        total_papers = sum(op.total_papers_processed for op in self._completed_operations)
        
        cache_hits = sum(op.cache_hits for op in self._completed_operations)
        cache_misses = sum(op.cache_misses for op in self._completed_operations)
        cache_hit_rate = cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0
        
        # API call statistics
        api_calls = defaultdict(int)
        for op in self._completed_operations:
            for api, count in op.api_calls.items():
                api_calls[api] += count
        
        return {
            'active_operations': active_ops,
            'completed_operations': completed_ops,
            'average_duration_seconds': avg_duration,
            'total_papers_processed': total_papers,
            'cache_hit_rate': cache_hit_rate,
            'api_calls': dict(api_calls),
            'queue_size': self._cache_warm_queue.qsize()
        }


# Singleton instance management
_analytics_service: Optional[AdvancedAnalyticsService] = None


async def get_analytics_service() -> AdvancedAnalyticsService:
    """Get or create the analytics service singleton."""
    global _analytics_service
    
    if _analytics_service is None:
        _analytics_service = AdvancedAnalyticsService()
        await _analytics_service.initialize()
    
    return _analytics_service