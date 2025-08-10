"""
LLM Integration Service - Integration with existing Neo4j and Redis infrastructure.

This module provides:
- Neo4j integration for enriched content storage
- Redis integration for caching and session management
- Graph-aware enrichment strategies
- Citation network analysis integration
- Research lineage storage in graph database
- Semantic search integration
- Data consistency and synchronization
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import asdict
import logging

from ..db.neo4j import Neo4jManager, get_neo4j_manager
from ..db.redis import RedisManager, get_redis_manager
from ..services.content_enrichment import EnrichedContent, ContentQuality
from ..services.citation_analysis import CitationRelationship, CitationType, InfluenceLevel
from ..services.research_trajectory import IntellectualLineage, ResearchMilestone, TrajectoryType
from ..models.paper import Paper
from ..utils.logger import get_logger
from ..utils.exceptions import ValidationError

logger = get_logger(__name__)


class LLMNeo4jIntegration:
    """
    Integration service for storing LLM-enriched content in Neo4j graph database.
    """
    
    def __init__(self, neo4j_manager: Optional[Neo4jManager] = None):
        self.neo4j_manager = neo4j_manager
        self._initialized = False
    
    async def initialize(self):
        """Initialize the Neo4j integration service."""
        if self._initialized:
            return
        
        if not self.neo4j_manager:
            self.neo4j_manager = await get_neo4j_manager()
        
        # Create indexes and constraints for LLM-enriched data
        await self._create_schema()
        
        self._initialized = True
        logger.info("LLM Neo4j Integration initialized")
    
    async def _create_schema(self):
        """Create Neo4j schema for LLM-enriched data."""
        if not self.neo4j_manager:
            return
        
        try:
            # Create constraints and indexes
            constraints_and_indexes = [
                # Enriched content constraints
                "CREATE CONSTRAINT enriched_content_id IF NOT EXISTS FOR (e:EnrichedContent) REQUIRE e.paper_id IS UNIQUE",
                
                # Citation relationship constraints  
                "CREATE CONSTRAINT citation_relationship_id IF NOT EXISTS FOR (c:CitationRelationship) REQUIRE (c.citing_paper_id, c.cited_paper_id) IS UNIQUE",
                
                # Research lineage constraints
                "CREATE CONSTRAINT research_lineage_id IF NOT EXISTS FOR (l:ResearchLineage) REQUIRE l.lineage_id IS UNIQUE",
                
                # Indexes for performance
                "CREATE INDEX enriched_content_quality IF NOT EXISTS FOR (e:EnrichedContent) ON e.content_quality",
                "CREATE INDEX enriched_content_timestamp IF NOT EXISTS FOR (e:EnrichedContent) ON e.enrichment_timestamp",
                "CREATE INDEX citation_type IF NOT EXISTS FOR (c:CitationRelationship) ON c.citation_type",
                "CREATE INDEX citation_influence IF NOT EXISTS FOR (c:CitationRelationship) ON c.influence_level",
                "CREATE INDEX lineage_trajectory_type IF NOT EXISTS FOR (l:ResearchLineage) ON l.trajectory_type",
            ]
            
            for query in constraints_and_indexes:
                try:
                    await self.neo4j_manager.execute_query(query)
                except Exception as e:
                    # Some constraints might already exist
                    logger.debug(f"Schema creation warning: {e}")
            
            logger.info("Neo4j schema for LLM data created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create Neo4j schema: {e}")
    
    async def store_enriched_content(self, enriched_content: EnrichedContent) -> bool:
        """
        Store enriched content in Neo4j with relationships to existing paper nodes.
        
        Args:
            enriched_content: The enriched content to store
            
        Returns:
            True if successful, False otherwise
        """
        await self.initialize()
        
        if not self.neo4j_manager:
            logger.error("Neo4j manager not available")
            return False
        
        try:
            # Prepare enriched content data
            content_data = {
                'paper_id': enriched_content.paper_id,
                'paper_title': enriched_content.paper_title,
                'enhanced_summary': enriched_content.enhanced_summary,
                'key_contributions': enriched_content.key_contributions or [],
                'methodology_summary': enriched_content.methodology_summary,
                'key_findings': enriched_content.key_findings or [],
                'significance_assessment': enriched_content.significance_assessment,
                'limitations': enriched_content.limitations or [],
                'future_directions': enriched_content.future_directions or [],
                'content_quality': enriched_content.content_quality.value,
                'confidence_score': enriched_content.confidence_score,
                'enrichment_model': enriched_content.enrichment_model,
                'enrichment_cost': enriched_content.enrichment_cost,
                'enrichment_tokens': enriched_content.enrichment_tokens,
                'enrichment_timestamp': enriched_content.enrichment_timestamp.isoformat() if enriched_content.enrichment_timestamp else None,
                'source_abstract': enriched_content.source_abstract,
                'source_venue': enriched_content.source_venue,
                'source_year': enriched_content.source_year,
                'source_authors': enriched_content.source_authors or []
            }
            
            # Cypher query to create/update enriched content and link to paper
            query = """
            MATCH (p:Paper {paper_id: $paper_id})
            MERGE (e:EnrichedContent {paper_id: $paper_id})
            SET e += $content_data
            MERGE (p)-[:HAS_ENRICHED_CONTENT]->(e)
            
            // Create contribution nodes
            WITH e, $key_contributions AS contributions
            UNWIND contributions AS contribution
            MERGE (c:Contribution {text: contribution})
            MERGE (e)-[:HAS_CONTRIBUTION]->(c)
            
            // Create finding nodes
            WITH e, $key_findings AS findings
            UNWIND findings AS finding
            MERGE (f:Finding {text: finding})
            MERGE (e)-[:HAS_FINDING]->(f)
            
            RETURN e.paper_id as paper_id
            """
            
            result = await self.neo4j_manager.execute_query(
                query,
                paper_id=enriched_content.paper_id,
                content_data=content_data,
                key_contributions=enriched_content.key_contributions or [],
                key_findings=enriched_content.key_findings or []
            )
            
            if result:
                logger.info(f"Stored enriched content for paper {enriched_content.paper_id}")
                return True
            else:
                logger.warning(f"No paper node found for {enriched_content.paper_id}")
                return False
            
        except Exception as e:
            logger.error(f"Failed to store enriched content in Neo4j: {e}")
            return False
    
    async def store_citation_relationship(self, relationship: CitationRelationship) -> bool:
        """
        Store citation relationship analysis in Neo4j.
        
        Args:
            relationship: The citation relationship to store
            
        Returns:
            True if successful, False otherwise
        """
        await self.initialize()
        
        if not self.neo4j_manager:
            return False
        
        try:
            # Prepare relationship data
            relationship_data = {
                'citing_paper_id': relationship.citing_paper_id,
                'cited_paper_id': relationship.cited_paper_id,
                'citing_title': relationship.citing_title,
                'cited_title': relationship.cited_title,
                'citation_year': relationship.citation_year,
                'time_gap_years': relationship.time_gap_years,
                'citation_purpose': relationship.citation_purpose,
                'intellectual_relationship': relationship.intellectual_relationship,
                'knowledge_flow_description': relationship.knowledge_flow_description,
                'impact_assessment': relationship.impact_assessment,
                'citation_type': relationship.citation_type.value if relationship.citation_type else None,
                'influence_level': relationship.influence_level.value if relationship.influence_level else None,
                'citation_context': relationship.citation_context,
                'is_self_citation': relationship.is_self_citation,
                'is_influential': relationship.is_influential,
                'analysis_confidence': relationship.analysis_confidence,
                'analysis_model': relationship.analysis_model,
                'analysis_cost': relationship.analysis_cost,
                'analysis_timestamp': relationship.analysis_timestamp.isoformat() if relationship.analysis_timestamp else None
            }
            
            # Cypher query to create enriched citation relationship
            query = """
            MATCH (citing:Paper {paper_id: $citing_paper_id})
            MATCH (cited:Paper {paper_id: $cited_paper_id})
            
            // Create or update the citation relationship
            MERGE (citing)-[c:CITES]->(cited)
            
            // Create enriched citation analysis node
            MERGE (analysis:CitationAnalysis {
                citing_paper_id: $citing_paper_id,
                cited_paper_id: $cited_paper_id
            })
            SET analysis += $relationship_data
            
            // Link citation to analysis
            MERGE (c)-[:HAS_ANALYSIS]->(analysis)
            
            // Create knowledge flow relationship if significant
            WITH citing, cited, analysis, $influence_level AS influence
            WHERE influence IN ['high_impact', 'transformative']
            MERGE (cited)-[:INFLUENCES {
                strength: influence,
                description: analysis.knowledge_flow_description
            }]->(citing)
            
            RETURN analysis.citing_paper_id as citing_id
            """
            
            result = await self.neo4j_manager.execute_query(
                query,
                citing_paper_id=relationship.citing_paper_id,
                cited_paper_id=relationship.cited_paper_id,
                relationship_data=relationship_data,
                influence_level=relationship.influence_level.value if relationship.influence_level else None
            )
            
            if result:
                logger.info(f"Stored citation analysis: {relationship.citing_paper_id} -> {relationship.cited_paper_id}")
                return True
            else:
                logger.warning(f"Failed to find papers for citation relationship")
                return False
            
        except Exception as e:
            logger.error(f"Failed to store citation relationship in Neo4j: {e}")
            return False
    
    async def store_research_lineage(self, lineage: IntellectualLineage) -> bool:
        """
        Store research lineage in Neo4j with milestone and relationship information.
        
        Args:
            lineage: The research lineage to store
            
        Returns:
            True if successful, False otherwise
        """
        await self.initialize()
        
        if not self.neo4j_manager:
            return False
        
        try:
            # Prepare lineage data
            lineage_data = {
                'lineage_id': lineage.lineage_id,
                'root_paper_ids': lineage.root_paper_ids,
                'total_papers': lineage.total_papers,
                'time_span_years': lineage.time_span_years,
                'generation_count': lineage.generation_count,
                'trajectory_type': lineage.trajectory_type.value if lineage.trajectory_type else None,
                'key_researchers': lineage.key_researchers or [],
                'dominant_venues': lineage.dominant_venues or [],
                'lineage_narrative': lineage.lineage_narrative,
                'intellectual_evolution': lineage.intellectual_evolution,
                'key_insights': lineage.key_insights or [],
                'future_directions': lineage.future_directions or [],
                'cross_field_influences': lineage.cross_field_influences or [],
                'analysis_confidence': lineage.analysis_confidence,
                'analysis_cost': lineage.analysis_cost,
                'analysis_model': lineage.analysis_model,
                'analysis_timestamp': lineage.analysis_timestamp.isoformat() if lineage.analysis_timestamp else None
            }
            
            # Create research lineage node
            lineage_query = """
            CREATE (l:ResearchLineage)
            SET l += $lineage_data
            
            // Connect to root papers
            WITH l, $root_paper_ids AS roots
            UNWIND roots AS root_id
            MATCH (p:Paper {paper_id: root_id})
            MERGE (l)-[:STARTS_FROM]->(p)
            """
            
            await self.neo4j_manager.execute_query(
                lineage_query,
                lineage_data=lineage_data,
                root_paper_ids=lineage.root_paper_ids
            )
            
            # Store milestones
            for milestone in lineage.milestones:
                milestone_data = {
                    'paper_id': milestone.paper_id,
                    'title': milestone.title,
                    'year': milestone.year,
                    'milestone_type': milestone.milestone_type.value,
                    'significance_description': milestone.significance_description,
                    'impact_score': milestone.impact_score,
                    'citation_count': milestone.citation_count,
                    'influenced_papers': milestone.influenced_papers,
                    'key_contributions': milestone.key_contributions,
                    'confidence_score': milestone.confidence_score,
                    'analysis_model': milestone.analysis_model,
                    'analysis_timestamp': milestone.analysis_timestamp.isoformat() if milestone.analysis_timestamp else None
                }
                
                milestone_query = """
                MATCH (l:ResearchLineage {lineage_id: $lineage_id})
                MATCH (p:Paper {paper_id: $paper_id})
                
                MERGE (m:ResearchMilestone {
                    paper_id: $paper_id,
                    lineage_id: $lineage_id
                })
                SET m += $milestone_data
                
                MERGE (l)-[:HAS_MILESTONE]->(m)
                MERGE (m)-[:REPRESENTS]->(p)
                """
                
                await self.neo4j_manager.execute_query(
                    milestone_query,
                    lineage_id=lineage.lineage_id,
                    paper_id=milestone.paper_id,
                    milestone_data=milestone_data
                )
            
            logger.info(f"Stored research lineage {lineage.lineage_id} with {len(lineage.milestones)} milestones")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store research lineage in Neo4j: {e}")
            return False
    
    async def get_enriched_papers_by_quality(
        self,
        quality_threshold: float = 0.7,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get enriched papers by quality score."""
        await self.initialize()
        
        if not self.neo4j_manager:
            return []
        
        try:
            query = """
            MATCH (e:EnrichedContent)
            WHERE e.confidence_score >= $quality_threshold
            MATCH (p:Paper)-[:HAS_ENRICHED_CONTENT]->(e)
            RETURN p.paper_id as paper_id,
                   p.title as title,
                   e.content_quality as quality,
                   e.confidence_score as confidence,
                   e.enrichment_timestamp as timestamp
            ORDER BY e.confidence_score DESC
            LIMIT $limit
            """
            
            result = await self.neo4j_manager.execute_query(
                query,
                quality_threshold=quality_threshold,
                limit=limit
            )
            
            return [dict(record) for record in result] if result else []
            
        except Exception as e:
            logger.error(f"Failed to get enriched papers by quality: {e}")
            return []
    
    async def get_citation_network_with_analysis(
        self,
        center_paper_id: str,
        depth: int = 2
    ) -> Dict[str, Any]:
        """Get citation network with LLM analysis data."""
        await self.initialize()
        
        if not self.neo4j_manager:
            return {}
        
        try:
            query = """
            MATCH (center:Paper {paper_id: $center_paper_id})
            CALL apoc.path.subgraphAll(center, {
                relationshipFilter: "CITES",
                maxLevel: $depth
            })
            YIELD nodes, relationships
            
            // Get enriched data for papers
            UNWIND nodes AS node
            OPTIONAL MATCH (node)-[:HAS_ENRICHED_CONTENT]->(e:EnrichedContent)
            
            // Get analysis for relationships
            UNWIND relationships AS rel
            OPTIONAL MATCH (rel)-[:HAS_ANALYSIS]->(a:CitationAnalysis)
            
            RETURN {
                papers: collect(DISTINCT {
                    paper_id: node.paper_id,
                    title: node.title,
                    enriched: e IS NOT NULL,
                    quality: e.content_quality,
                    summary: e.enhanced_summary
                }),
                citations: collect(DISTINCT {
                    source: startNode(rel).paper_id,
                    target: endNode(rel).paper_id,
                    has_analysis: a IS NOT NULL,
                    citation_type: a.citation_type,
                    influence_level: a.influence_level,
                    purpose: a.citation_purpose
                })
            } AS network
            """
            
            result = await self.neo4j_manager.execute_query(
                query,
                center_paper_id=center_paper_id,
                depth=depth
            )
            
            return result[0]['network'] if result else {}
            
        except Exception as e:
            logger.error(f"Failed to get citation network with analysis: {e}")
            return {}
    
    async def get_research_trajectories_by_type(
        self,
        trajectory_type: TrajectoryType,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get research lineages by trajectory type."""
        await self.initialize()
        
        if not self.neo4j_manager:
            return []
        
        try:
            query = """
            MATCH (l:ResearchLineage {trajectory_type: $trajectory_type})
            MATCH (l)-[:HAS_MILESTONE]->(m:ResearchMilestone)
            WITH l, collect(m) as milestones
            RETURN {
                lineage_id: l.lineage_id,
                total_papers: l.total_papers,
                time_span_years: l.time_span_years,
                trajectory_type: l.trajectory_type,
                key_insights: l.key_insights,
                analysis_confidence: l.analysis_confidence,
                milestone_count: size(milestones),
                dominant_venues: l.dominant_venues
            } AS trajectory
            ORDER BY l.analysis_confidence DESC
            LIMIT $limit
            """
            
            result = await self.neo4j_manager.execute_query(
                query,
                trajectory_type=trajectory_type.value,
                limit=limit
            )
            
            return [record['trajectory'] for record in result] if result else []
            
        except Exception as e:
            logger.error(f"Failed to get research trajectories: {e}")
            return []


class LLMRedisIntegration:
    """
    Integration service for LLM caching and session management with Redis.
    """
    
    def __init__(self, redis_manager: Optional[RedisManager] = None):
        self.redis_manager = redis_manager
        self._initialized = False
    
    async def initialize(self):
        """Initialize the Redis integration service."""
        if self._initialized:
            return
        
        if not self.redis_manager:
            self.redis_manager = await get_redis_manager()
        
        self._initialized = True
        logger.info("LLM Redis Integration initialized")
    
    async def cache_semantic_embeddings(
        self,
        content: str,
        embedding: List[float],
        content_type: str = "paper_content",
        ttl_hours: int = 72
    ) -> bool:
        """Cache semantic embeddings for content."""
        await self.initialize()
        
        if not self.redis_manager:
            return False
        
        try:
            content_hash = hash(content)
            embedding_key = f"semantic_embedding:{content_type}:{content_hash}"
            
            embedding_data = {
                'content': content[:500],  # Store truncated content for reference
                'embedding': embedding,
                'content_type': content_type,
                'timestamp': datetime.now().isoformat()
            }
            
            await self.redis_manager.setex(
                embedding_key,
                ttl_hours * 3600,
                json.dumps(embedding_data)
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache semantic embedding: {e}")
            return False
    
    async def get_semantic_embedding(
        self,
        content: str,
        content_type: str = "paper_content"
    ) -> Optional[List[float]]:
        """Get cached semantic embedding for content."""
        await self.initialize()
        
        if not self.redis_manager:
            return None
        
        try:
            content_hash = hash(content)
            embedding_key = f"semantic_embedding:{content_type}:{content_hash}"
            
            embedding_data = await self.redis_manager.get(embedding_key)
            if embedding_data:
                data = json.loads(embedding_data)
                return data['embedding']
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get semantic embedding: {e}")
            return None
    
    async def store_enrichment_session(
        self,
        session_id: str,
        session_data: Dict[str, Any],
        ttl_hours: int = 24
    ) -> bool:
        """Store enrichment session data."""
        await self.initialize()
        
        if not self.redis_manager:
            return False
        
        try:
            session_key = f"enrichment_session:{session_id}"
            
            session_info = {
                'session_id': session_id,
                'created_at': datetime.now().isoformat(),
                'data': session_data
            }
            
            await self.redis_manager.setex(
                session_key,
                ttl_hours * 3600,
                json.dumps(session_info)
            )
            
            # Add to active sessions index
            await self.redis_manager.sadd("active_enrichment_sessions", session_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store enrichment session: {e}")
            return False
    
    async def get_enrichment_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get enrichment session data."""
        await self.initialize()
        
        if not self.redis_manager:
            return None
        
        try:
            session_key = f"enrichment_session:{session_id}"
            session_data = await self.redis_manager.get(session_key)
            
            if session_data:
                return json.loads(session_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get enrichment session: {e}")
            return None
    
    async def cache_research_insights(
        self,
        insight_type: str,
        insight_key: str,
        insights: List[str],
        confidence_score: float,
        ttl_hours: int = 168  # 1 week
    ) -> bool:
        """Cache research insights for reuse."""
        await self.initialize()
        
        if not self.redis_manager:
            return False
        
        try:
            cache_key = f"research_insights:{insight_type}:{insight_key}"
            
            insight_data = {
                'insights': insights,
                'confidence_score': confidence_score,
                'timestamp': datetime.now().isoformat(),
                'insight_type': insight_type
            }
            
            await self.redis_manager.setex(
                cache_key,
                ttl_hours * 3600,
                json.dumps(insight_data)
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache research insights: {e}")
            return False
    
    async def get_research_insights(
        self,
        insight_type: str,
        insight_key: str
    ) -> Optional[Tuple[List[str], float]]:
        """Get cached research insights."""
        await self.initialize()
        
        if not self.redis_manager:
            return None
        
        try:
            cache_key = f"research_insights:{insight_type}:{insight_key}"
            insight_data = await self.redis_manager.get(cache_key)
            
            if insight_data:
                data = json.loads(insight_data)
                return data['insights'], data['confidence_score']
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get research insights: {e}")
            return None


class LLMIntegrationService:
    """
    Main integration service that coordinates between LLM services and existing infrastructure.
    """
    
    def __init__(
        self,
        neo4j_manager: Optional[Neo4jManager] = None,
        redis_manager: Optional[RedisManager] = None
    ):
        self.neo4j_integration = LLMNeo4jIntegration(neo4j_manager)
        self.redis_integration = LLMRedisIntegration(redis_manager)
        self._initialized = False
    
    async def initialize(self):
        """Initialize all integration services."""
        if self._initialized:
            return
        
        await self.neo4j_integration.initialize()
        await self.redis_integration.initialize()
        
        self._initialized = True
        logger.info("LLM Integration Service initialized")
    
    async def store_complete_enrichment(
        self,
        enriched_content: EnrichedContent,
        related_citations: List[CitationRelationship] = None,
        lineage: Optional[IntellectualLineage] = None
    ) -> Dict[str, bool]:
        """
        Store complete enrichment data across Neo4j and Redis.
        
        Returns:
            Dictionary with success status for each storage operation
        """
        await self.initialize()
        
        results = {}
        
        # Store enriched content in Neo4j
        results['enriched_content'] = await self.neo4j_integration.store_enriched_content(enriched_content)
        
        # Store citation relationships if provided
        if related_citations:
            citation_results = []
            for citation in related_citations:
                success = await self.neo4j_integration.store_citation_relationship(citation)
                citation_results.append(success)
            results['citations'] = all(citation_results)
        else:
            results['citations'] = True
        
        # Store research lineage if provided
        if lineage:
            results['lineage'] = await self.neo4j_integration.store_research_lineage(lineage)
        else:
            results['lineage'] = True
        
        # Cache enrichment data in Redis for quick access
        if results['enriched_content']:
            session_data = {
                'paper_id': enriched_content.paper_id,
                'quality': enriched_content.content_quality.value,
                'confidence': enriched_content.confidence_score,
                'timestamp': enriched_content.enrichment_timestamp.isoformat() if enriched_content.enrichment_timestamp else None
            }
            
            session_id = f"enrich_{enriched_content.paper_id}_{int(datetime.now().timestamp())}"
            results['session_cache'] = await self.redis_integration.store_enrichment_session(
                session_id, session_data
            )
        else:
            results['session_cache'] = False
        
        # Log overall success
        overall_success = all(results.values())
        if overall_success:
            logger.info(f"Successfully stored complete enrichment for paper {enriched_content.paper_id}")
        else:
            logger.warning(f"Partial success storing enrichment for paper {enriched_content.paper_id}: {results}")
        
        return results
    
    async def get_enrichment_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics from both Neo4j and Redis."""
        await self.initialize()
        
        analytics = {
            'neo4j': {},
            'redis': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Get Neo4j analytics
        try:
            high_quality_papers = await self.neo4j_integration.get_enriched_papers_by_quality(0.8, 10)
            analytics['neo4j'] = {
                'high_quality_papers': len(high_quality_papers),
                'sample_papers': high_quality_papers[:5]  # Sample for display
            }
        except Exception as e:
            logger.error(f"Failed to get Neo4j analytics: {e}")
            analytics['neo4j'] = {'error': str(e)}
        
        # Get Redis analytics
        try:
            if self.redis_integration.redis_manager:
                active_sessions = await self.redis_integration.redis_manager.scard("active_enrichment_sessions")
                analytics['redis'] = {
                    'active_sessions': active_sessions or 0
                }
        except Exception as e:
            logger.error(f"Failed to get Redis analytics: {e}")
            analytics['redis'] = {'error': str(e)}
        
        return analytics
    
    async def cleanup_old_data(self, days_to_keep: int = 90) -> Dict[str, int]:
        """Clean up old enrichment data from both stores."""
        await self.initialize()
        
        cleanup_results = {
            'neo4j_cleaned': 0,
            'redis_cleaned': 0,
            'errors': []
        }
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # Clean up old Neo4j data
        try:
            if self.neo4j_integration.neo4j_manager:
                cleanup_query = """
                MATCH (e:EnrichedContent)
                WHERE e.enrichment_timestamp < $cutoff_date
                OPTIONAL MATCH (e)-[r]-()
                DELETE r, e
                RETURN count(e) as cleaned_count
                """
                
                result = await self.neo4j_integration.neo4j_manager.execute_query(
                    cleanup_query,
                    cutoff_date=cutoff_date.isoformat()
                )
                
                if result:
                    cleanup_results['neo4j_cleaned'] = result[0]['cleaned_count']
                    
        except Exception as e:
            cleanup_results['errors'].append(f"Neo4j cleanup error: {str(e)}")
        
        # Clean up old Redis data would be handled by TTL, but we can clean sessions
        try:
            if self.redis_integration.redis_manager:
                # Clean up expired sessions from active sessions set
                active_sessions = await self.redis_integration.redis_manager.smembers("active_enrichment_sessions")
                cleaned_sessions = 0
                
                for session_id in active_sessions or []:
                    session_key = f"enrichment_session:{session_id}"
                    exists = await self.redis_integration.redis_manager.exists(session_key)
                    
                    if not exists:
                        await self.redis_integration.redis_manager.srem("active_enrichment_sessions", session_id)
                        cleaned_sessions += 1
                
                cleanup_results['redis_cleaned'] = cleaned_sessions
                
        except Exception as e:
            cleanup_results['errors'].append(f"Redis cleanup error: {str(e)}")
        
        logger.info(f"Cleanup completed: {cleanup_results}")
        return cleanup_results


# Global integration service instance
_integration_service: Optional[LLMIntegrationService] = None


async def get_integration_service() -> LLMIntegrationService:
    """Get or create the global integration service."""
    global _integration_service
    
    if _integration_service is None:
        _integration_service = LLMIntegrationService()
        await _integration_service.initialize()
    
    return _integration_service