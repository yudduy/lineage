"""
Research Intelligence Engine for automated trend analysis and community detection.

This module provides advanced research intelligence capabilities including
automated trend detection, community characterization, knowledge flow analysis,
and research trajectory modeling with real-time processing capabilities.
"""

import asyncio
import json
import time
import hashlib
import numpy as np
from typing import Any, Dict, List, Optional, Set, Tuple, Union, AsyncIterator
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from ..core.config import get_settings
from ..db.redis import RedisManager, get_redis_manager
from ..db.neo4j_advanced import AdvancedNeo4jManager
from ..services.advanced_analytics import (
    AdvancedAnalyticsService,
    ResearchTrendType,
    ResearchCommunity
)
from ..utils.logger import get_logger
from ..utils.exceptions import APIError, ValidationError

logger = get_logger(__name__)


@dataclass
class ResearchTrend:
    """Represents a detected research trend."""
    trend_id: str
    name: str
    type: ResearchTrendType
    strength: float  # 0-1 indicating trend strength
    confidence: float  # 0-1 confidence in detection
    start_period: str
    peak_period: Optional[str]
    key_papers: List[str]
    key_authors: List[str]
    key_institutions: List[str]
    growth_rate: float  # Percentage growth rate
    momentum: float  # Current momentum indicator
    related_trends: List[str]
    enabling_factors: List[str]
    predicted_duration: Optional[int]  # Months
    impact_areas: List[str]


@dataclass
class CommunityDynamics:
    """Represents dynamics within and between research communities."""
    community_id: str
    formation_date: datetime
    growth_trajectory: List[Dict[str, Any]]  # Time series of size/activity
    key_events: List[Dict[str, Any]]  # Significant events in community history
    leadership_changes: List[Dict[str, Any]]
    collaboration_patterns: Dict[str, float]  # With other communities
    knowledge_imports: Dict[str, float]  # Ideas from other fields
    knowledge_exports: Dict[str, float]  # Ideas to other fields
    cohesion_score: float  # Internal cohesion metric
    influence_radius: float  # External influence metric
    lifecycle_stage: str  # emerging, growing, mature, declining


@dataclass
class KnowledgeFlow:
    """Represents knowledge flow between entities."""
    source_id: str
    source_type: str  # paper, author, institution, field
    target_id: str
    target_type: str
    flow_type: str  # citation, collaboration, concept_transfer
    strength: float
    temporal_pattern: str  # continuous, burst, declining
    key_carriers: List[str]  # Papers or authors carrying knowledge
    concepts_transferred: List[str]
    impact_score: float


@dataclass
class ResearchForecast:
    """Forecast for research development."""
    forecast_id: str
    target_entity: str  # Paper, author, field, etc.
    entity_type: str
    forecast_horizon: int  # Months
    predicted_citations: Optional[int]
    predicted_collaborations: List[str]
    emerging_topics: List[Dict[str, float]]  # Topic -> Probability
    breakthrough_probability: float
    decline_risk: float
    confidence_interval: Tuple[float, float]
    key_assumptions: List[str]
    risk_factors: List[str]


class TrendDetectionMethod(str, Enum):
    """Methods for trend detection."""
    BURST_DETECTION = "burst_detection"
    GROWTH_ANALYSIS = "growth_analysis"
    CITATION_ACCELERATION = "citation_acceleration"
    TOPIC_MODELING = "topic_modeling"
    NETWORK_DYNAMICS = "network_dynamics"


class ResearchIntelligenceEngine:
    """
    Advanced research intelligence engine for automated analysis of
    research trends, communities, and knowledge flows.
    """
    
    def __init__(
        self,
        analytics_service: Optional[AdvancedAnalyticsService] = None,
        redis_manager: Optional[RedisManager] = None,
        neo4j_manager: Optional[AdvancedNeo4jManager] = None
    ):
        self.analytics = analytics_service
        self.redis = redis_manager
        self.neo4j = neo4j_manager
        
        # Configuration
        self.min_trend_confidence = 0.6
        self.min_community_size = 5
        self.forecast_confidence_threshold = 0.7
        
        # Caching
        self._cache_ttl = 1800  # 30 minutes for intelligence data
        
        # Real-time processing
        self._trend_monitors: Dict[str, asyncio.Task] = {}
        self._community_monitors: Dict[str, asyncio.Task] = {}
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._event_processor: Optional[asyncio.Task] = None
        
        self.settings = get_settings()
        
    async def initialize(self):
        """Initialize the intelligence engine."""
        # Initialize dependencies
        if not self.analytics:
            from .advanced_analytics import get_analytics_service
            self.analytics = await get_analytics_service()
        
        if not self.redis:
            self.redis = await get_redis_manager()
        
        if not self.neo4j:
            self.neo4j = AdvancedNeo4jManager()
            await self.neo4j.initialize()
        
        # Start event processor
        self._event_processor = asyncio.create_task(self._process_events())
        
        logger.info("Research Intelligence Engine initialized")
    
    async def close(self):
        """Clean up resources."""
        # Cancel monitoring tasks
        for task in self._trend_monitors.values():
            task.cancel()
        for task in self._community_monitors.values():
            task.cancel()
        
        if self._event_processor:
            self._event_processor.cancel()
        
        if self.neo4j:
            await self.neo4j.close()
    
    # ==================== Trend Detection ====================
    
    async def detect_research_trends(
        self,
        domain: Optional[str] = None,
        time_window: Optional[Tuple[int, int]] = None,
        methods: Optional[List[TrendDetectionMethod]] = None,
        min_confidence: float = 0.6
    ) -> List[ResearchTrend]:
        """
        Detect research trends using multiple methods and data sources.
        """
        cache_key = f"trends:{domain}:{time_window}:{min_confidence}"
        cached = await self._get_cached_result(cache_key)
        if cached:
            return [ResearchTrend(**t) for t in cached]
        
        trends = []
        methods = methods or [
            TrendDetectionMethod.BURST_DETECTION,
            TrendDetectionMethod.GROWTH_ANALYSIS,
            TrendDetectionMethod.CITATION_ACCELERATION
        ]
        
        # Run different detection methods in parallel
        detection_tasks = []
        
        if TrendDetectionMethod.BURST_DETECTION in methods:
            detection_tasks.append(
                self._detect_burst_trends(domain, time_window)
            )
        
        if TrendDetectionMethod.GROWTH_ANALYSIS in methods:
            detection_tasks.append(
                self._detect_growth_trends(domain, time_window)
            )
        
        if TrendDetectionMethod.CITATION_ACCELERATION in methods:
            detection_tasks.append(
                self._detect_citation_acceleration(domain, time_window)
            )
        
        if TrendDetectionMethod.TOPIC_MODELING in methods:
            detection_tasks.append(
                self._detect_topic_trends(domain, time_window)
            )
        
        if TrendDetectionMethod.NETWORK_DYNAMICS in methods:
            detection_tasks.append(
                self._detect_network_trends(domain, time_window)
            )
        
        # Gather results
        method_results = await asyncio.gather(*detection_tasks, return_exceptions=True)
        
        # Merge and deduplicate trends
        all_trends = []
        for result in method_results:
            if isinstance(result, list):
                all_trends.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Trend detection method failed: {result}")
        
        # Deduplicate and filter by confidence
        seen_trends = set()
        for trend in all_trends:
            if trend.confidence >= min_confidence:
                trend_key = f"{trend.name}_{trend.type}"
                if trend_key not in seen_trends:
                    trends.append(trend)
                    seen_trends.add(trend_key)
        
        # Sort by strength and confidence
        trends.sort(key=lambda x: x.strength * x.confidence, reverse=True)
        
        # Cache results
        await self._cache_result(
            cache_key,
            [t.__dict__ for t in trends],
            ttl=self._cache_ttl
        )
        
        return trends
    
    async def _detect_burst_trends(
        self,
        domain: Optional[str],
        time_window: Optional[Tuple[int, int]]
    ) -> List[ResearchTrend]:
        """Detect sudden bursts in research activity."""
        trends = []
        
        try:
            # Query for publication time series data
            query = """
            MATCH (p:Paper)
            WHERE p.year >= $start_year AND p.year <= $end_year
            """
            if domain:
                query += " AND $domain IN p.fields"
            
            query += """
            WITH p.year as year, p.fields as fields, count(*) as paper_count
            RETURN year, fields, paper_count
            ORDER BY year
            """
            
            start_year = time_window[0] if time_window else 2010
            end_year = time_window[1] if time_window else 2024
            
            results = await self.neo4j.execute_query(
                query,
                parameters={
                    'start_year': start_year,
                    'end_year': end_year,
                    'domain': domain
                }
            )
            
            # Analyze time series for each field
            field_series = defaultdict(list)
            for record in results:
                year = record['year']
                fields = record['fields']
                count = record['paper_count']
                
                for field in fields:
                    field_series[field].append((year, count))
            
            # Detect bursts using Kleinberg's algorithm (simplified)
            for field, series in field_series.items():
                if len(series) < 3:
                    continue
                
                # Calculate burst score
                counts = [c for _, c in series]
                years = [y for y, _ in series]
                
                if len(counts) >= 3:
                    # Simple burst detection: significant increase
                    for i in range(2, len(counts)):
                        prev_avg = np.mean(counts[max(0, i-3):i])
                        current = counts[i]
                        
                        if prev_avg > 0 and current > prev_avg * 2:
                            # Burst detected
                            burst_strength = (current - prev_avg) / prev_avg
                            
                            trend = ResearchTrend(
                                trend_id=hashlib.md5(f"burst_{field}_{years[i]}".encode()).hexdigest(),
                                name=f"{field} Research Burst",
                                type=ResearchTrendType.BREAKTHROUGH,
                                strength=min(burst_strength / 5, 1.0),
                                confidence=0.7,
                                start_period=str(years[i]),
                                peak_period=str(years[i]),
                                key_papers=[],  # Would populate from actual data
                                key_authors=[],
                                key_institutions=[],
                                growth_rate=burst_strength * 100,
                                momentum=burst_strength * 0.5,
                                related_trends=[],
                                enabling_factors=["Technological advancement", "Increased funding"],
                                predicted_duration=24,
                                impact_areas=[field]
                            )
                            trends.append(trend)
            
        except Exception as e:
            logger.error(f"Error in burst detection: {e}")
        
        return trends
    
    async def _detect_growth_trends(
        self,
        domain: Optional[str],
        time_window: Optional[Tuple[int, int]]
    ) -> List[ResearchTrend]:
        """Detect sustained growth trends."""
        trends = []
        
        try:
            # Similar to burst detection but looking for sustained growth
            # Implementation would analyze growth patterns over time
            pass
            
        except Exception as e:
            logger.error(f"Error in growth trend detection: {e}")
        
        return trends
    
    async def _detect_citation_acceleration(
        self,
        domain: Optional[str],
        time_window: Optional[Tuple[int, int]]
    ) -> List[ResearchTrend]:
        """Detect trends based on citation acceleration."""
        trends = []
        
        try:
            # Query for papers with accelerating citation rates
            query = """
            MATCH (p:Paper)<-[:CITES]-(c:Paper)
            WHERE p.year >= $start_year AND p.year <= $end_year
            """
            if domain:
                query += " AND $domain IN p.fields"
            
            query += """
            WITH p, count(c) as citations, 
                 collect(c.year) as citing_years
            WHERE citations > 10
            RETURN p.id as paper_id, p.title as title, 
                   p.year as year, p.fields as fields,
                   citations, citing_years
            ORDER BY citations DESC
            LIMIT 100
            """
            
            start_year = time_window[0] if time_window else 2015
            end_year = time_window[1] if time_window else 2024
            
            results = await self.neo4j.execute_query(
                query,
                parameters={
                    'start_year': start_year,
                    'end_year': end_year,
                    'domain': domain
                }
            )
            
            # Analyze citation acceleration
            for record in results:
                citing_years = record['citing_years']
                if not citing_years:
                    continue
                
                # Calculate citation rate over time
                year_counts = Counter(citing_years)
                years_sorted = sorted(year_counts.keys())
                
                if len(years_sorted) >= 3:
                    # Check for acceleration
                    recent_rate = sum(
                        year_counts[y] for y in years_sorted[-2:]
                    ) / 2
                    older_rate = sum(
                        year_counts[y] for y in years_sorted[:-2]
                    ) / len(years_sorted[:-2])
                    
                    if older_rate > 0 and recent_rate > older_rate * 1.5:
                        acceleration = (recent_rate - older_rate) / older_rate
                        
                        # Create trend for this paper/topic
                        fields = record['fields']
                        for field in fields[:1]:  # Primary field
                            trend = ResearchTrend(
                                trend_id=hashlib.md5(f"accel_{field}_{record['paper_id']}".encode()).hexdigest(),
                                name=f"Accelerating Interest in {field}",
                                type=ResearchTrendType.EMERGING,
                                strength=min(acceleration / 3, 1.0),
                                confidence=0.75,
                                start_period=str(years_sorted[0]),
                                peak_period=str(years_sorted[-1]),
                                key_papers=[record['paper_id']],
                                key_authors=[],
                                key_institutions=[],
                                growth_rate=acceleration * 100,
                                momentum=acceleration * 0.7,
                                related_trends=[],
                                enabling_factors=["High-impact publication"],
                                predicted_duration=18,
                                impact_areas=fields
                            )
                            trends.append(trend)
            
        except Exception as e:
            logger.error(f"Error in citation acceleration detection: {e}")
        
        return trends
    
    async def _detect_topic_trends(
        self,
        domain: Optional[str],
        time_window: Optional[Tuple[int, int]]
    ) -> List[ResearchTrend]:
        """Detect trends using topic modeling."""
        trends = []
        
        # This would use LDA or other topic modeling techniques
        # on paper abstracts/titles to identify emerging topics
        
        return trends
    
    async def _detect_network_trends(
        self,
        domain: Optional[str],
        time_window: Optional[Tuple[int, int]]
    ) -> List[ResearchTrend]:
        """Detect trends based on network dynamics."""
        trends = []
        
        try:
            # Analyze changes in collaboration networks
            query = """
            MATCH (a1:Author)-[:AUTHORED]->(p:Paper)<-[:AUTHORED]-(a2:Author)
            WHERE p.year >= $start_year AND p.year <= $end_year
            """
            if domain:
                query += " AND $domain IN p.fields"
            
            query += """
            WITH p.year as year, count(DISTINCT a1) + count(DISTINCT a2) as collab_count
            RETURN year, collab_count
            ORDER BY year
            """
            
            start_year = time_window[0] if time_window else 2015
            end_year = time_window[1] if time_window else 2024
            
            results = await self.neo4j.execute_query(
                query,
                parameters={
                    'start_year': start_year,
                    'end_year': end_year,
                    'domain': domain
                }
            )
            
            # Analyze collaboration growth
            if len(results) >= 3:
                years = [r['year'] for r in results]
                collabs = [r['collab_count'] for r in results]
                
                # Fit trend line
                z = np.polyfit(range(len(years)), collabs, 1)
                slope = z[0]
                
                if slope > 0:
                    growth_rate = slope / np.mean(collabs) * 100
                    
                    trend = ResearchTrend(
                        trend_id=hashlib.md5(f"network_{domain}_{time_window}".encode()).hexdigest(),
                        name=f"Growing Collaboration in {domain or 'Research'}",
                        type=ResearchTrendType.CONVERGENT,
                        strength=min(growth_rate / 20, 1.0),
                        confidence=0.65,
                        start_period=str(years[0]),
                        peak_period=str(years[-1]),
                        key_papers=[],
                        key_authors=[],
                        key_institutions=[],
                        growth_rate=growth_rate,
                        momentum=growth_rate * 0.3,
                        related_trends=[],
                        enabling_factors=["Increased interdisciplinary research"],
                        predicted_duration=36,
                        impact_areas=[domain] if domain else ["General"]
                    )
                    trends.append(trend)
            
        except Exception as e:
            logger.error(f"Error in network trend detection: {e}")
        
        return trends
    
    # ==================== Community Analysis ====================
    
    async def analyze_community_dynamics(
        self,
        community_id: str,
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> CommunityDynamics:
        """
        Analyze the dynamics and evolution of a research community.
        """
        cache_key = f"community_dynamics:{community_id}:{time_range}"
        cached = await self._get_cached_result(cache_key)
        if cached:
            return CommunityDynamics(**cached)
        
        try:
            # Get community data from graph
            query = """
            MATCH (p:Paper)-[:IN_COMMUNITY]->(c:Community {id: $community_id})
            OPTIONAL MATCH (p)<-[:AUTHORED]-(a:Author)
            OPTIONAL MATCH (p)<-[:CITES]-(citing:Paper)
            RETURN p.year as year, p.id as paper_id, 
                   collect(DISTINCT a.id) as authors,
                   count(DISTINCT citing) as citations
            ORDER BY p.year
            """
            
            results = await self.neo4j.execute_query(
                query,
                parameters={'community_id': community_id}
            )
            
            # Build timeline
            timeline = defaultdict(lambda: {
                'papers': 0,
                'authors': set(),
                'citations': 0
            })
            
            for record in results:
                year = record['year']
                if year:
                    timeline[year]['papers'] += 1
                    timeline[year]['authors'].update(record['authors'])
                    timeline[year]['citations'] += record['citations']
            
            # Convert to growth trajectory
            growth_trajectory = []
            for year in sorted(timeline.keys()):
                growth_trajectory.append({
                    'year': year,
                    'paper_count': timeline[year]['papers'],
                    'author_count': len(timeline[year]['authors']),
                    'citation_count': timeline[year]['citations']
                })
            
            # Identify key events (simplified)
            key_events = []
            for i in range(1, len(growth_trajectory)):
                curr = growth_trajectory[i]
                prev = growth_trajectory[i-1]
                
                # Detect significant changes
                if curr['paper_count'] > prev['paper_count'] * 1.5:
                    key_events.append({
                        'year': curr['year'],
                        'type': 'publication_burst',
                        'description': f"Publication surge: {curr['paper_count']} papers"
                    })
            
            # Calculate dynamics metrics
            cohesion_score = self._calculate_cohesion(timeline)
            influence_radius = self._calculate_influence(timeline)
            lifecycle_stage = self._determine_lifecycle(growth_trajectory)
            
            dynamics = CommunityDynamics(
                community_id=community_id,
                formation_date=datetime(min(timeline.keys()), 1, 1),
                growth_trajectory=growth_trajectory,
                key_events=key_events,
                leadership_changes=[],  # Would need author-level analysis
                collaboration_patterns={},  # Would need cross-community analysis
                knowledge_imports={},
                knowledge_exports={},
                cohesion_score=cohesion_score,
                influence_radius=influence_radius,
                lifecycle_stage=lifecycle_stage
            )
            
            # Cache result
            await self._cache_result(cache_key, dynamics.__dict__, ttl=self._cache_ttl)
            
            return dynamics
            
        except Exception as e:
            logger.error(f"Error analyzing community dynamics: {e}")
            raise
    
    def _calculate_cohesion(self, timeline: Dict) -> float:
        """Calculate community cohesion score."""
        # Simplified: ratio of internal citations to total citations
        # In production, would use more sophisticated network metrics
        return 0.7  # Placeholder
    
    def _calculate_influence(self, timeline: Dict) -> float:
        """Calculate community influence radius."""
        # Based on external citations and cross-community collaborations
        total_citations = sum(t['citations'] for t in timeline.values())
        return min(total_citations / 1000, 1.0)
    
    def _determine_lifecycle(self, trajectory: List[Dict]) -> str:
        """Determine community lifecycle stage."""
        if len(trajectory) < 3:
            return "emerging"
        
        # Analyze recent growth
        recent = trajectory[-3:]
        growth_rates = []
        for i in range(1, len(recent)):
            prev = recent[i-1]['paper_count']
            curr = recent[i]['paper_count']
            if prev > 0:
                growth_rates.append((curr - prev) / prev)
        
        avg_growth = np.mean(growth_rates) if growth_rates else 0
        
        if avg_growth > 0.2:
            return "growing"
        elif avg_growth > -0.1:
            return "mature"
        else:
            return "declining"
    
    # ==================== Knowledge Flow Analysis ====================
    
    async def analyze_knowledge_flows(
        self,
        source_entity: str,
        entity_type: str = "field",
        max_hops: int = 3
    ) -> List[KnowledgeFlow]:
        """
        Analyze how knowledge flows from a source entity.
        """
        flows = []
        
        try:
            if entity_type == "field":
                # Analyze field-to-field knowledge transfer
                query = """
                MATCH (p1:Paper)-[:HAS_FIELD]->(:Field {name: $source})
                MATCH (p1)<-[:CITES]-(p2:Paper)-[:HAS_FIELD]->(f2:Field)
                WHERE f2.name <> $source
                WITH f2.name as target_field, 
                     count(DISTINCT p2) as flow_count,
                     collect(DISTINCT p2.id)[..5] as sample_papers
                RETURN target_field, flow_count, sample_papers
                ORDER BY flow_count DESC
                LIMIT 20
                """
                
                results = await self.neo4j.execute_query(
                    query,
                    parameters={'source': source_entity}
                )
                
                for record in results:
                    flow = KnowledgeFlow(
                        source_id=source_entity,
                        source_type="field",
                        target_id=record['target_field'],
                        target_type="field",
                        flow_type="citation",
                        strength=min(record['flow_count'] / 100, 1.0),
                        temporal_pattern="continuous",
                        key_carriers=record['sample_papers'],
                        concepts_transferred=[],  # Would need concept extraction
                        impact_score=record['flow_count'] * 0.01
                    )
                    flows.append(flow)
            
            elif entity_type == "author":
                # Analyze author's knowledge dissemination
                query = """
                MATCH (a:Author {id: $source})-[:AUTHORED]->(p1:Paper)
                MATCH (p1)<-[:CITES]-(p2:Paper)<-[:AUTHORED]-(a2:Author)
                WHERE a2.id <> $source
                WITH a2.id as target_author, 
                     count(DISTINCT p2) as flow_count
                RETURN target_author, flow_count
                ORDER BY flow_count DESC
                LIMIT 20
                """
                
                results = await self.neo4j.execute_query(
                    query,
                    parameters={'source': source_entity}
                )
                
                for record in results:
                    flow = KnowledgeFlow(
                        source_id=source_entity,
                        source_type="author",
                        target_id=record['target_author'],
                        target_type="author",
                        flow_type="citation",
                        strength=min(record['flow_count'] / 50, 1.0),
                        temporal_pattern="continuous",
                        key_carriers=[],
                        concepts_transferred=[],
                        impact_score=record['flow_count'] * 0.02
                    )
                    flows.append(flow)
            
        except Exception as e:
            logger.error(f"Error analyzing knowledge flows: {e}")
        
        return flows
    
    # ==================== Research Forecasting ====================
    
    async def forecast_research_development(
        self,
        entity_id: str,
        entity_type: str,
        horizon_months: int = 24
    ) -> ResearchForecast:
        """
        Forecast future research development for an entity.
        """
        cache_key = f"forecast:{entity_id}:{entity_type}:{horizon_months}"
        cached = await self._get_cached_result(cache_key)
        if cached:
            return ResearchForecast(**cached)
        
        try:
            # Get historical data
            historical_data = await self._get_historical_data(entity_id, entity_type)
            
            if not historical_data:
                raise ValueError(f"No historical data for {entity_type} {entity_id}")
            
            # Perform time series forecasting
            citations_forecast = self._forecast_citations(historical_data, horizon_months)
            
            # Predict collaborations
            predicted_collaborations = await self._predict_collaborations(
                entity_id, entity_type, historical_data
            )
            
            # Predict emerging topics
            emerging_topics = await self._predict_emerging_topics(
                entity_id, entity_type, historical_data
            )
            
            # Calculate probabilities
            breakthrough_prob = self._calculate_breakthrough_probability(historical_data)
            decline_risk = self._calculate_decline_risk(historical_data)
            
            # Create forecast
            forecast = ResearchForecast(
                forecast_id=hashlib.md5(f"{entity_id}_{horizon_months}_{time.time()}".encode()).hexdigest(),
                target_entity=entity_id,
                entity_type=entity_type,
                forecast_horizon=horizon_months,
                predicted_citations=citations_forecast,
                predicted_collaborations=predicted_collaborations,
                emerging_topics=emerging_topics,
                breakthrough_probability=breakthrough_prob,
                decline_risk=decline_risk,
                confidence_interval=(0.6, 0.9),
                key_assumptions=[
                    "Historical patterns continue",
                    "No major disruptions",
                    "Funding levels remain stable"
                ],
                risk_factors=[
                    "Technology disruption",
                    "Funding changes",
                    "Researcher mobility"
                ]
            )
            
            # Cache result
            await self._cache_result(cache_key, forecast.__dict__, ttl=self._cache_ttl)
            
            return forecast
            
        except Exception as e:
            logger.error(f"Error in research forecasting: {e}")
            raise
    
    async def _get_historical_data(
        self,
        entity_id: str,
        entity_type: str
    ) -> Dict[str, Any]:
        """Get historical data for an entity."""
        data = {}
        
        if entity_type == "paper":
            query = """
            MATCH (p:Paper {id: $entity_id})
            OPTIONAL MATCH (p)<-[:CITES]-(citing:Paper)
            WITH p, citing.year as cite_year, count(citing) as cite_count
            RETURN p.year as year, p.citation_count as total_citations,
                   collect({year: cite_year, count: cite_count}) as citation_timeline
            """
        elif entity_type == "author":
            query = """
            MATCH (a:Author {id: $entity_id})-[:AUTHORED]->(p:Paper)
            WITH a, p.year as year, count(p) as paper_count,
                 sum(p.citation_count) as citations
            RETURN year, paper_count, citations
            ORDER BY year
            """
        else:
            return data
        
        results = await self.neo4j.execute_query(
            query,
            parameters={'entity_id': entity_id}
        )
        
        if results:
            data = {
                'entity_id': entity_id,
                'entity_type': entity_type,
                'timeline': results
            }
        
        return data
    
    def _forecast_citations(
        self,
        historical_data: Dict[str, Any],
        horizon_months: int
    ) -> Optional[int]:
        """Forecast future citations using time series analysis."""
        # Simplified linear extrapolation
        # In production, would use ARIMA, Prophet, or other sophisticated methods
        
        timeline = historical_data.get('timeline', [])
        if len(timeline) < 3:
            return None
        
        # Extract citation counts over time
        citations = []
        for record in timeline:
            if 'citations' in record:
                citations.append(record['citations'])
        
        if len(citations) < 3:
            return None
        
        # Simple linear regression
        x = np.arange(len(citations))
        y = np.array(citations)
        z = np.polyfit(x, y, 1)
        
        # Extrapolate
        future_periods = horizon_months // 12  # Convert to years
        predicted = z[0] * (len(citations) + future_periods) + z[1]
        
        return max(int(predicted), 0)
    
    async def _predict_collaborations(
        self,
        entity_id: str,
        entity_type: str,
        historical_data: Dict[str, Any]
    ) -> List[str]:
        """Predict future collaborations."""
        # Simplified: return top potential collaborators based on network
        collaborations = []
        
        if entity_type == "author":
            query = """
            MATCH (a:Author {id: $entity_id})-[:AUTHORED]->(p1:Paper)
            MATCH (p1)<-[:CITES|CITED_BY]-(p2:Paper)<-[:AUTHORED]-(a2:Author)
            WHERE a2.id <> $entity_id
            AND NOT EXISTS((a)-[:COLLABORATED_WITH]-(a2))
            WITH a2.id as potential_collaborator, count(*) as connection_strength
            RETURN potential_collaborator
            ORDER BY connection_strength DESC
            LIMIT 5
            """
            
            results = await self.neo4j.execute_query(
                query,
                parameters={'entity_id': entity_id}
            )
            
            collaborations = [r['potential_collaborator'] for r in results]
        
        return collaborations
    
    async def _predict_emerging_topics(
        self,
        entity_id: str,
        entity_type: str,
        historical_data: Dict[str, Any]
    ) -> List[Dict[str, float]]:
        """Predict emerging topics for the entity."""
        topics = []
        
        # Analyze recent paper topics and trends
        if entity_type in ["author", "field"]:
            query = """
            MATCH (p:Paper)-[:HAS_FIELD]->(f:Field)
            WHERE p.year >= $recent_year
            WITH f.name as field, count(p) as paper_count,
                 avg(p.citation_count) as avg_citations
            WHERE paper_count > 5
            RETURN field, paper_count, avg_citations
            ORDER BY paper_count * avg_citations DESC
            LIMIT 10
            """
            
            results = await self.neo4j.execute_query(
                query,
                parameters={'recent_year': 2020}
            )
            
            total_score = sum(
                r['paper_count'] * r['avg_citations'] for r in results
            )
            
            for record in results:
                score = (record['paper_count'] * record['avg_citations']) / total_score
                topics.append({
                    'topic': record['field'],
                    'probability': min(score, 1.0)
                })
        
        return topics[:5]
    
    def _calculate_breakthrough_probability(
        self,
        historical_data: Dict[str, Any]
    ) -> float:
        """Calculate probability of breakthrough based on patterns."""
        # Simplified heuristic
        # In production, would use ML models trained on breakthrough papers
        
        timeline = historical_data.get('timeline', [])
        if len(timeline) < 3:
            return 0.1
        
        # Check for accelerating metrics
        recent_growth = 0
        if len(timeline) >= 2:
            if 'citations' in timeline[-1] and 'citations' in timeline[-2]:
                if timeline[-2]['citations'] > 0:
                    recent_growth = (
                        timeline[-1]['citations'] - timeline[-2]['citations']
                    ) / timeline[-2]['citations']
        
        # Base probability with growth modifier
        base_prob = 0.15
        if recent_growth > 0.5:
            base_prob += 0.2
        elif recent_growth > 0.2:
            base_prob += 0.1
        
        return min(base_prob, 0.8)
    
    def _calculate_decline_risk(
        self,
        historical_data: Dict[str, Any]
    ) -> float:
        """Calculate risk of decline."""
        timeline = historical_data.get('timeline', [])
        if len(timeline) < 3:
            return 0.3
        
        # Check for declining metrics
        recent_decline = 0
        if len(timeline) >= 2:
            if 'citations' in timeline[-1] and 'citations' in timeline[-2]:
                if timeline[-2]['citations'] > 0:
                    recent_decline = (
                        timeline[-2]['citations'] - timeline[-1]['citations']
                    ) / timeline[-2]['citations']
        
        # Base risk with decline modifier
        base_risk = 0.2
        if recent_decline > 0.3:
            base_risk += 0.3
        elif recent_decline > 0.1:
            base_risk += 0.15
        
        return min(base_risk, 0.9)
    
    # ==================== Real-time Monitoring ====================
    
    async def start_trend_monitoring(
        self,
        trend_id: str,
        callback: Optional[callable] = None
    ):
        """Start real-time monitoring of a trend."""
        if trend_id in self._trend_monitors:
            logger.warning(f"Trend {trend_id} already being monitored")
            return
        
        async def monitor():
            while True:
                try:
                    # Check for updates
                    updates = await self._check_trend_updates(trend_id)
                    
                    if updates and callback:
                        await callback(trend_id, updates)
                    
                    # Emit event
                    await self._event_queue.put({
                        'type': 'trend_update',
                        'trend_id': trend_id,
                        'updates': updates,
                        'timestamp': datetime.utcnow()
                    })
                    
                    await asyncio.sleep(300)  # Check every 5 minutes
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error monitoring trend {trend_id}: {e}")
                    await asyncio.sleep(60)
        
        self._trend_monitors[trend_id] = asyncio.create_task(monitor())
    
    async def stop_trend_monitoring(self, trend_id: str):
        """Stop monitoring a trend."""
        if trend_id in self._trend_monitors:
            self._trend_monitors[trend_id].cancel()
            del self._trend_monitors[trend_id]
    
    async def _check_trend_updates(self, trend_id: str) -> Dict[str, Any]:
        """Check for updates to a trend."""
        # Implementation would query for new papers, citations, etc.
        return {}
    
    async def _process_events(self):
        """Process intelligence events."""
        while True:
            try:
                event = await self._event_queue.get()
                
                # Process different event types
                if event['type'] == 'trend_update':
                    await self._handle_trend_update(event)
                elif event['type'] == 'community_change':
                    await self._handle_community_change(event)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    async def _handle_trend_update(self, event: Dict[str, Any]):
        """Handle trend update event."""
        # Implementation would update caches, trigger alerts, etc.
        pass
    
    async def _handle_community_change(self, event: Dict[str, Any]):
        """Handle community change event."""
        # Implementation would update community metrics, notify subscribers
        pass
    
    # ==================== Helper Methods ====================
    
    async def _get_cached_result(self, key: str) -> Optional[Any]:
        """Get cached result."""
        if not self.redis:
            return None
        
        try:
            result = await self.redis.get(f"intelligence:{key}")
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
                f"intelligence:{key}",
                json.dumps(value, default=str),
                ttl=ttl
            )
        except Exception as e:
            logger.debug(f"Cache set error: {e}")


# Singleton instance management
_intelligence_engine: Optional[ResearchIntelligenceEngine] = None


async def get_intelligence_engine() -> ResearchIntelligenceEngine:
    """Get or create the intelligence engine singleton."""
    global _intelligence_engine
    
    if _intelligence_engine is None:
        _intelligence_engine = ResearchIntelligenceEngine()
        await _intelligence_engine.initialize()
    
    return _intelligence_engine