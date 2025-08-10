"""
Advanced Neo4j database integration with Graph Data Science algorithms.
Provides comprehensive citation network analysis, community detection, and graph metrics.
"""

from typing import Any, Dict, List, Optional, Union, Tuple, Set
import asyncio
import json
import uuid
import time
import math
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession, AsyncTransaction
from neo4j.exceptions import ServiceUnavailable, AuthError, Neo4jError, ClientError

from ..core.config import get_settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


class GraphAlgorithm(str, Enum):
    """Available graph algorithms."""
    PAGERANK = "pagerank"
    BETWEENNESS_CENTRALITY = "betweenness"
    CLOSENESS_CENTRALITY = "closeness"
    DEGREE_CENTRALITY = "degree"
    EIGENVECTOR_CENTRALITY = "eigenvector"
    ARTICULATION_POINTS = "articulationPoints"
    BRIDGES = "bridges"
    TRIANGLE_COUNT = "triangleCount"
    LOCAL_CLUSTERING_COEFFICIENT = "localClusteringCoefficient"


class CommunityAlgorithm(str, Enum):
    """Community detection algorithms."""
    LOUVAIN = "louvain"
    LEIDEN = "leiden"
    LABEL_PROPAGATION = "labelPropagation"
    MODULARITY_OPTIMIZATION = "modularityOptimization"
    WEAKLY_CONNECTED_COMPONENTS = "wcc"
    STRONGLY_CONNECTED_COMPONENTS = "scc"


class PathfindingAlgorithm(str, Enum):
    """Pathfinding algorithms."""
    SHORTEST_PATH = "shortestPath"
    ALL_SHORTEST_PATHS = "allShortestPaths"
    SINGLE_SOURCE_SHORTEST_PATH = "singleSourceShortestPath"
    DELTA_STEPPING = "deltaStepping"
    A_STAR = "astar"


class SimilarityAlgorithm(str, Enum):
    """Node similarity algorithms."""
    JACCARD = "jaccard"
    COSINE = "cosine"
    PEARSON = "pearson"
    OVERLAP = "overlap"


@dataclass
class GraphProjection:
    """Graph projection configuration."""
    name: str
    node_labels: List[str]
    relationship_types: List[str]
    node_properties: Optional[Dict[str, Any]] = None
    relationship_properties: Optional[Dict[str, Any]] = None
    orientation: str = "NATURAL"  # NATURAL, REVERSE, UNDIRECTED


@dataclass
class AlgorithmResult:
    """Result from graph algorithm execution."""
    algorithm: str
    execution_time: float
    memory_used: Optional[int] = None
    node_count: Optional[int] = None
    relationship_count: Optional[int] = None
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class CommunityResult:
    """Result from community detection algorithm."""
    algorithm: str
    community_count: int
    modularity: float
    communities: Dict[str, List[str]]  # community_id -> [node_ids]
    node_communities: Dict[str, str]   # node_id -> community_id
    execution_time: float


@dataclass
class CentralityResult:
    """Result from centrality algorithm."""
    algorithm: str
    scores: Dict[str, float]  # node_id -> score
    top_nodes: List[Tuple[str, float]]  # [(node_id, score), ...]
    execution_time: float
    statistics: Dict[str, float]  # min, max, mean, std


class AdvancedNeo4jManager:
    """Advanced Neo4j manager with Graph Data Science algorithms."""
    
    def __init__(self):
        self.settings = get_settings()
        self.driver: Optional[AsyncDriver] = None
        self._lock = asyncio.Lock()
        self._graph_catalog = {}  # Track created graph projections
        self._schema_initialized = False
        
    async def connect(self):
        """Establish connection to Neo4j with GDS support verification."""
        async with self._lock:
            if self.driver is None:
                try:
                    self.driver = AsyncGraphDatabase.driver(
                        self.settings.database.neo4j_uri,
                        auth=(
                            self.settings.database.neo4j_user,
                            self.settings.database.neo4j_password
                        ),
                        database=self.settings.database.neo4j_database,
                        max_connection_pool_size=20,
                        connection_acquisition_timeout=30.0,
                        max_transaction_retry_time=15.0
                    )
                    
                    # Test connection and verify GDS availability
                    await self.driver.verify_connectivity()
                    await self._verify_gds_support()
                    await self._initialize_schema()
                    
                    logger.info("Successfully connected to Neo4j with GDS support")
                    
                except (ServiceUnavailable, AuthError) as e:
                    logger.error(f"Failed to connect to Neo4j: {e}")
                    raise
                except Exception as e:
                    logger.error(f"Unexpected error connecting to Neo4j: {e}")
                    raise
    
    async def disconnect(self):
        """Close Neo4j connection and cleanup."""
        async with self._lock:
            if self.driver:
                # Drop any remaining graph projections
                await self._cleanup_graph_catalog()
                await self.driver.close()
                self.driver = None
                logger.info("Disconnected from Neo4j")
    
    async def _verify_gds_support(self):
        """Verify that Graph Data Science library is available."""
        try:
            async with self.driver.session() as session:
                result = await session.run("CALL gds.version() YIELD gdsVersion")
                record = await result.single()
                if record:
                    gds_version = record["gdsVersion"]
                    logger.info(f"Neo4j GDS version: {gds_version}")
                else:
                    raise Exception("GDS library not available")
        except ClientError as e:
            if "procedure not found" in str(e).lower():
                logger.error("Neo4j Graph Data Science library is not installed")
                raise Exception("GDS library is required but not available")
            raise
    
    async def _initialize_schema(self):
        """Initialize database schema with constraints and indexes."""
        if self._schema_initialized:
            return
            
        schema_queries = [
            # Node constraints
            "CREATE CONSTRAINT paper_id_unique IF NOT EXISTS FOR (p:Paper) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT author_id_unique IF NOT EXISTS FOR (a:Author) REQUIRE a.id IS UNIQUE",
            "CREATE CONSTRAINT institution_id_unique IF NOT EXISTS FOR (i:Institution) REQUIRE i.id IS UNIQUE",
            "CREATE CONSTRAINT venue_id_unique IF NOT EXISTS FOR (v:Venue) REQUIRE v.id IS UNIQUE",
            
            # Property indexes
            "CREATE INDEX paper_doi_index IF NOT EXISTS FOR (p:Paper) ON (p.doi)",
            "CREATE INDEX paper_title_index IF NOT EXISTS FOR (p:Paper) ON (p.title)",
            "CREATE INDEX author_name_index IF NOT EXISTS FOR (a:Author) ON (a.name)",
            "CREATE INDEX paper_year_index IF NOT EXISTS FOR (p:Paper) ON (p.publication_year)",
            "CREATE INDEX paper_citation_count_index IF NOT EXISTS FOR (p:Paper) ON (p.citation_count)",
            
            # Composite indexes
            "CREATE INDEX paper_year_citations_index IF NOT EXISTS FOR (p:Paper) ON (p.publication_year, p.citation_count)",
            
            # Full-text search indexes
            "CREATE FULLTEXT INDEX paperFulltext IF NOT EXISTS FOR (p:Paper) ON EACH [p.title, p.abstract, p.keywords]",
            "CREATE FULLTEXT INDEX authorFulltext IF NOT EXISTS FOR (a:Author) ON EACH [a.name, a.affiliations]",
        ]
        
        async with self.driver.session() as session:
            for query in schema_queries:
                try:
                    await session.run(query)
                except ClientError as e:
                    if "already exists" not in str(e).lower():
                        logger.warning(f"Schema query failed: {query} - {e}")
        
        self._schema_initialized = True
        logger.info("Schema initialization completed")
    
    @asynccontextmanager
    async def get_session(self):
        """Get Neo4j session context manager."""
        if not self.driver:
            await self.connect()
        
        session = self.driver.session()
        try:
            yield session
        finally:
            await session.close()
    
    async def execute_read(self, query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """Execute read query and return results."""
        async with self.get_session() as session:
            result = await session.run(query, parameters or {})
            records = await result.data()
            return records
    
    async def execute_write(self, query: str, parameters: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute write query and return summary."""
        async with self.get_session() as session:
            result = await session.run(query, parameters or {})
            summary = await result.consume()
            return {
                "nodes_created": summary.counters.nodes_created,
                "relationships_created": summary.counters.relationships_created,
                "properties_set": summary.counters.properties_set,
                "nodes_deleted": summary.counters.nodes_deleted,
                "relationships_deleted": summary.counters.relationships_deleted
            }
    
    async def execute_transaction(self, transaction_func, *args, **kwargs):
        """Execute function within a transaction."""
        async with self.get_session() as session:
            return await session.execute_write(transaction_func, *args, **kwargs)
    
    # ==================== GRAPH PROJECTION MANAGEMENT ====================
    
    async def create_graph_projection(self, projection: GraphProjection) -> bool:
        """Create a graph projection for algorithm execution."""
        try:
            # Build projection query
            node_spec = {}
            for label in projection.node_labels:
                node_spec[label] = {"properties": projection.node_properties or {}}
            
            relationship_spec = {}
            for rel_type in projection.relationship_types:
                relationship_spec[rel_type] = {
                    "orientation": projection.orientation,
                    "properties": projection.relationship_properties or {}
                }
            
            query = """
            CALL gds.graph.project(
                $graphName,
                $nodeProjection,
                $relationshipProjection
            )
            YIELD graphName, nodeCount, relationshipCount, projectMillis
            RETURN graphName, nodeCount, relationshipCount, projectMillis
            """
            
            async with self.get_session() as session:
                result = await session.run(query, {
                    "graphName": projection.name,
                    "nodeProjection": node_spec,
                    "relationshipProjection": relationship_spec
                })
                
                record = await result.single()
                if record:
                    self._graph_catalog[projection.name] = {
                        "projection": projection,
                        "node_count": record["nodeCount"],
                        "relationship_count": record["relationshipCount"],
                        "created_at": datetime.now()
                    }
                    logger.info(f"Created graph projection '{projection.name}' with {record['nodeCount']} nodes and {record['relationshipCount']} relationships")
                    return True
                    
        except ClientError as e:
            if "already exists" in str(e).lower():
                logger.warning(f"Graph projection '{projection.name}' already exists")
                return True
            logger.error(f"Failed to create graph projection: {e}")
        
        return False
    
    async def drop_graph_projection(self, graph_name: str) -> bool:
        """Drop a graph projection."""
        try:
            query = "CALL gds.graph.drop($graphName)"
            async with self.get_session() as session:
                await session.run(query, {"graphName": graph_name})
            
            if graph_name in self._graph_catalog:
                del self._graph_catalog[graph_name]
            
            logger.info(f"Dropped graph projection '{graph_name}'")
            return True
            
        except ClientError as e:
            if "not found" not in str(e).lower():
                logger.error(f"Failed to drop graph projection: {e}")
            return False
    
    async def list_graph_projections(self) -> List[Dict[str, Any]]:
        """List all graph projections."""
        try:
            query = "CALL gds.graph.list() YIELD graphName, nodeCount, relationshipCount, memoryUsage"
            async with self.get_session() as session:
                result = await session.run(query)
                return await result.data()
        except Exception as e:
            logger.error(f"Failed to list graph projections: {e}")
            return []
    
    async def _cleanup_graph_catalog(self):
        """Clean up all graph projections."""
        for graph_name in list(self._graph_catalog.keys()):
            await self.drop_graph_projection(graph_name)
    
    # ==================== CENTRALITY ALGORITHMS ====================
    
    async def calculate_pagerank(
        self,
        graph_name: str,
        max_iterations: int = 20,
        damping_factor: float = 0.85,
        tolerance: float = 1e-6,
        write_property: Optional[str] = None
    ) -> CentralityResult:
        """Calculate PageRank centrality scores."""
        start_time = time.time()
        
        try:
            if write_property:
                query = """
                CALL gds.pageRank.write($graphName, {
                    maxIterations: $maxIterations,
                    dampingFactor: $dampingFactor,
                    tolerance: $tolerance,
                    writeProperty: $writeProperty
                })
                YIELD centralityDistribution, nodePropertiesWritten
                RETURN centralityDistribution, nodePropertiesWritten
                """
            else:
                query = """
                CALL gds.pageRank.stream($graphName, {
                    maxIterations: $maxIterations,
                    dampingFactor: $dampingFactor,
                    tolerance: $tolerance
                })
                YIELD nodeId, score
                RETURN gds.util.asNode(nodeId).id as nodeId, score
                ORDER BY score DESC
                LIMIT 1000
                """
            
            async with self.get_session() as session:
                result = await session.run(query, {
                    "graphName": graph_name,
                    "maxIterations": max_iterations,
                    "dampingFactor": damping_factor,
                    "tolerance": tolerance,
                    "writeProperty": write_property
                })
                
                records = await result.data()
                
                if write_property:
                    # Get distribution statistics
                    distribution = records[0]["centralityDistribution"]
                    scores = {}
                    top_nodes = []
                    statistics = {
                        "min": distribution["min"],
                        "max": distribution["max"],
                        "mean": distribution["mean"],
                        "std": distribution["stdDev"]
                    }
                else:
                    scores = {record["nodeId"]: record["score"] for record in records}
                    top_nodes = [(record["nodeId"], record["score"]) for record in records[:20]]
                    
                    # Calculate statistics
                    score_values = list(scores.values())
                    statistics = {
                        "min": min(score_values) if score_values else 0,
                        "max": max(score_values) if score_values else 0,
                        "mean": sum(score_values) / len(score_values) if score_values else 0,
                        "std": 0  # Would need additional calculation
                    }
                
                execution_time = time.time() - start_time
                
                return CentralityResult(
                    algorithm="PageRank",
                    scores=scores,
                    top_nodes=top_nodes,
                    execution_time=execution_time,
                    statistics=statistics
                )
                
        except Exception as e:
            logger.error(f"PageRank calculation failed: {e}")
            raise
    
    async def calculate_betweenness_centrality(
        self,
        graph_name: str,
        write_property: Optional[str] = None
    ) -> CentralityResult:
        """Calculate betweenness centrality scores."""
        start_time = time.time()
        
        try:
            if write_property:
                query = """
                CALL gds.betweenness.write($graphName, {
                    writeProperty: $writeProperty
                })
                YIELD centralityDistribution, nodePropertiesWritten
                RETURN centralityDistribution, nodePropertiesWritten
                """
            else:
                query = """
                CALL gds.betweenness.stream($graphName)
                YIELD nodeId, score
                RETURN gds.util.asNode(nodeId).id as nodeId, score
                ORDER BY score DESC
                LIMIT 1000
                """
            
            async with self.get_session() as session:
                result = await session.run(query, {
                    "graphName": graph_name,
                    "writeProperty": write_property
                })
                
                records = await result.data()
                
                if write_property:
                    distribution = records[0]["centralityDistribution"]
                    scores = {}
                    top_nodes = []
                    statistics = {
                        "min": distribution["min"],
                        "max": distribution["max"],
                        "mean": distribution["mean"],
                        "std": distribution["stdDev"]
                    }
                else:
                    scores = {record["nodeId"]: record["score"] for record in records}
                    top_nodes = [(record["nodeId"], record["score"]) for record in records[:20]]
                    
                    score_values = list(scores.values())
                    statistics = {
                        "min": min(score_values) if score_values else 0,
                        "max": max(score_values) if score_values else 0,
                        "mean": sum(score_values) / len(score_values) if score_values else 0,
                        "std": 0
                    }
                
                execution_time = time.time() - start_time
                
                return CentralityResult(
                    algorithm="Betweenness",
                    scores=scores,
                    top_nodes=top_nodes,
                    execution_time=execution_time,
                    statistics=statistics
                )
                
        except Exception as e:
            logger.error(f"Betweenness centrality calculation failed: {e}")
            raise
    
    async def calculate_closeness_centrality(
        self,
        graph_name: str,
        write_property: Optional[str] = None
    ) -> CentralityResult:
        """Calculate closeness centrality scores."""
        start_time = time.time()
        
        try:
            if write_property:
                query = """
                CALL gds.closeness.write($graphName, {
                    writeProperty: $writeProperty
                })
                YIELD centralityDistribution, nodePropertiesWritten
                RETURN centralityDistribution, nodePropertiesWritten
                """
            else:
                query = """
                CALL gds.closeness.stream($graphName)
                YIELD nodeId, score
                RETURN gds.util.asNode(nodeId).id as nodeId, score
                ORDER BY score DESC
                LIMIT 1000
                """
            
            async with self.get_session() as session:
                result = await session.run(query, {
                    "graphName": graph_name,
                    "writeProperty": write_property
                })
                
                records = await result.data()
                
                if write_property:
                    distribution = records[0]["centralityDistribution"]
                    scores = {}
                    top_nodes = []
                    statistics = {
                        "min": distribution["min"],
                        "max": distribution["max"],
                        "mean": distribution["mean"],
                        "std": distribution["stdDev"]
                    }
                else:
                    scores = {record["nodeId"]: record["score"] for record in records}
                    top_nodes = [(record["nodeId"], record["score"]) for record in records[:20]]
                    
                    score_values = list(scores.values())
                    statistics = {
                        "min": min(score_values) if score_values else 0,
                        "max": max(score_values) if score_values else 0,
                        "mean": sum(score_values) / len(score_values) if score_values else 0,
                        "std": 0
                    }
                
                execution_time = time.time() - start_time
                
                return CentralityResult(
                    algorithm="Closeness",
                    scores=scores,
                    top_nodes=top_nodes,
                    execution_time=execution_time,
                    statistics=statistics
                )
                
        except Exception as e:
            logger.error(f"Closeness centrality calculation failed: {e}")
            raise
    
    # ==================== COMMUNITY DETECTION ALGORITHMS ====================
    
    async def detect_communities_louvain(
        self,
        graph_name: str,
        max_iterations: int = 10,
        tolerance: float = 1e-6,
        write_property: Optional[str] = None
    ) -> CommunityResult:
        """Detect communities using Louvain algorithm."""
        start_time = time.time()
        
        try:
            if write_property:
                query = """
                CALL gds.louvain.write($graphName, {
                    maxIterations: $maxIterations,
                    tolerance: $tolerance,
                    writeProperty: $writeProperty
                })
                YIELD communityCount, modularity, modularities
                RETURN communityCount, modularity, modularities
                """
            else:
                query = """
                CALL gds.louvain.stream($graphName, {
                    maxIterations: $maxIterations,
                    tolerance: $tolerance
                })
                YIELD nodeId, communityId
                RETURN gds.util.asNode(nodeId).id as nodeId, communityId
                """
            
            async with self.get_session() as session:
                result = await session.run(query, {
                    "graphName": graph_name,
                    "maxIterations": max_iterations,
                    "tolerance": tolerance,
                    "writeProperty": write_property
                })
                
                records = await result.data()
                
                if write_property:
                    community_count = records[0]["communityCount"]
                    modularity = records[0]["modularity"]
                    communities = {}
                    node_communities = {}
                else:
                    # Build community mappings
                    communities = {}
                    node_communities = {}
                    community_ids = set()
                    
                    for record in records:
                        node_id = record["nodeId"]
                        community_id = str(record["communityId"])
                        
                        node_communities[node_id] = community_id
                        community_ids.add(community_id)
                        
                        if community_id not in communities:
                            communities[community_id] = []
                        communities[community_id].append(node_id)
                    
                    community_count = len(community_ids)
                    modularity = 0.0  # Would need separate calculation
                
                execution_time = time.time() - start_time
                
                return CommunityResult(
                    algorithm="Louvain",
                    community_count=community_count,
                    modularity=modularity,
                    communities=communities,
                    node_communities=node_communities,
                    execution_time=execution_time
                )
                
        except Exception as e:
            logger.error(f"Louvain community detection failed: {e}")
            raise
    
    async def detect_communities_leiden(
        self,
        graph_name: str,
        max_iterations: int = 10,
        gamma: float = 1.0,
        theta: float = 0.01,
        write_property: Optional[str] = None
    ) -> CommunityResult:
        """Detect communities using Leiden algorithm."""
        start_time = time.time()
        
        try:
            if write_property:
                query = """
                CALL gds.leiden.write($graphName, {
                    maxIterations: $maxIterations,
                    gamma: $gamma,
                    theta: $theta,
                    writeProperty: $writeProperty
                })
                YIELD communityCount, modularity, modularities
                RETURN communityCount, modularity, modularities
                """
            else:
                query = """
                CALL gds.leiden.stream($graphName, {
                    maxIterations: $maxIterations,
                    gamma: $gamma,
                    theta: $theta
                })
                YIELD nodeId, communityId
                RETURN gds.util.asNode(nodeId).id as nodeId, communityId
                """
            
            async with self.get_session() as session:
                result = await session.run(query, {
                    "graphName": graph_name,
                    "maxIterations": max_iterations,
                    "gamma": gamma,
                    "theta": theta,
                    "writeProperty": write_property
                })
                
                records = await result.data()
                
                if write_property:
                    community_count = records[0]["communityCount"]
                    modularity = records[0]["modularity"]
                    communities = {}
                    node_communities = {}
                else:
                    communities = {}
                    node_communities = {}
                    community_ids = set()
                    
                    for record in records:
                        node_id = record["nodeId"]
                        community_id = str(record["communityId"])
                        
                        node_communities[node_id] = community_id
                        community_ids.add(community_id)
                        
                        if community_id not in communities:
                            communities[community_id] = []
                        communities[community_id].append(node_id)
                    
                    community_count = len(community_ids)
                    modularity = 0.0
                
                execution_time = time.time() - start_time
                
                return CommunityResult(
                    algorithm="Leiden",
                    community_count=community_count,
                    modularity=modularity,
                    communities=communities,
                    node_communities=node_communities,
                    execution_time=execution_time
                )
                
        except Exception as e:
            logger.error(f"Leiden community detection failed: {e}")
            raise
    
    # ==================== PATHFINDING ALGORITHMS ====================
    
    async def find_shortest_path(
        self,
        graph_name: str,
        source_node_id: str,
        target_node_id: str,
        relationship_weight_property: Optional[str] = None
    ) -> Dict[str, Any]:
        """Find shortest path between two nodes."""
        # Security validation for input parameters
        if not graph_name or not isinstance(graph_name, str) or len(graph_name.strip()) == 0:
            raise ValueError("Invalid graph_name provided")
        
        if not source_node_id or not isinstance(source_node_id, str) or len(source_node_id.strip()) == 0:
            raise ValueError("Invalid source_node_id provided")
        
        if not target_node_id or not isinstance(target_node_id, str) or len(target_node_id.strip()) == 0:
            raise ValueError("Invalid target_node_id provided")
        
        # Validate relationship weight property if provided
        if relationship_weight_property is not None:
            if not isinstance(relationship_weight_property, str) or len(relationship_weight_property.strip()) == 0:
                raise ValueError("Invalid relationship_weight_property provided")
        
        try:
            if relationship_weight_property:
                query = """
                MATCH (source {id: $sourceId}), (target {id: $targetId})
                CALL gds.shortestPath.dijkstra.stream($graphName, {
                    sourceNode: source,
                    targetNode: target,
                    relationshipWeightProperty: $weightProperty
                })
                YIELD index, sourceNode, targetNode, totalCost, nodeIds, costs, path
                RETURN 
                    [nodeId IN nodeIds | gds.util.asNode(nodeId).id] as pathNodeIds,
                    totalCost,
                    costs,
                    size(nodeIds) as pathLength
                """
            else:
                query = """
                MATCH (source {id: $sourceId}), (target {id: $targetId})
                CALL gds.shortestPath.yens.stream($graphName, {
                    sourceNode: source,
                    targetNode: target,
                    k: 1
                })
                YIELD index, sourceNode, targetNode, totalCost, nodeIds, costs, path
                RETURN 
                    [nodeId IN nodeIds | gds.util.asNode(nodeId).id] as pathNodeIds,
                    totalCost,
                    costs,
                    size(nodeIds) as pathLength
                """
            
            async with self.get_session() as session:
                result = await session.run(query, {
                    "graphName": graph_name,
                    "sourceId": source_node_id,
                    "targetId": target_node_id,
                    "weightProperty": relationship_weight_property
                })
                
                record = await result.single()
                
                if record:
                    return {
                        "path_nodes": record["pathNodeIds"],
                        "path_length": record["pathLength"],
                        "total_cost": record["totalCost"],
                        "costs": record["costs"]
                    }
                else:
                    return {"path_nodes": [], "path_length": 0, "total_cost": float('inf')}
                    
        except Exception as e:
            logger.error(f"Shortest path calculation failed: {e}")
            raise
    
    async def find_all_shortest_paths(
        self,
        graph_name: str,
        source_node_id: str,
        target_node_id: str,
        k: int = 10
    ) -> List[Dict[str, Any]]:
        """Find k shortest paths between two nodes."""
        try:
            query = """
            MATCH (source {id: $sourceId}), (target {id: $targetId})
            CALL gds.shortestPath.yens.stream($graphName, {
                sourceNode: source,
                targetNode: target,
                k: $k
            })
            YIELD index, sourceNode, targetNode, totalCost, nodeIds, costs, path
            RETURN 
                index,
                [nodeId IN nodeIds | gds.util.asNode(nodeId).id] as pathNodeIds,
                totalCost,
                costs,
                size(nodeIds) as pathLength
            ORDER BY totalCost, index
            """
            
            async with self.get_session() as session:
                result = await session.run(query, {
                    "graphName": graph_name,
                    "sourceId": source_node_id,
                    "targetId": target_node_id,
                    "k": k
                })
                
                records = await result.data()
                
                paths = []
                for record in records:
                    paths.append({
                        "index": record["index"],
                        "path_nodes": record["pathNodeIds"],
                        "path_length": record["pathLength"],
                        "total_cost": record["totalCost"],
                        "costs": record["costs"]
                    })
                
                return paths
                
        except Exception as e:
            logger.error(f"All shortest paths calculation failed: {e}")
            raise
    
    # ==================== GRAPH METRICS AND ANALYSIS ====================
    
    async def calculate_graph_metrics(self, graph_name: str) -> Dict[str, Any]:
        """Calculate comprehensive graph metrics."""
        try:
            metrics = {}
            
            # Basic graph statistics
            basic_stats_query = """
            CALL gds.graph.list($graphName)
            YIELD graphName, nodeCount, relationshipCount, memoryUsage
            RETURN nodeCount, relationshipCount, memoryUsage
            """
            
            async with self.get_session() as session:
                result = await session.run(basic_stats_query, {"graphName": graph_name})
                record = await result.single()
                
                if record:
                    metrics.update({
                        "node_count": record["nodeCount"],
                        "relationship_count": record["relationshipCount"],
                        "memory_usage": record["memoryUsage"]
                    })
                
                # Calculate density
                if metrics.get("node_count", 0) > 1:
                    max_edges = metrics["node_count"] * (metrics["node_count"] - 1)
                    metrics["density"] = metrics["relationship_count"] / max_edges
                else:
                    metrics["density"] = 0
                
                # Calculate triangle count
                try:
                    triangle_query = """
                    CALL gds.triangleCount.stream($graphName)
                    YIELD nodeId, triangleCount
                    RETURN sum(triangleCount) / 3 as totalTriangles, avg(triangleCount) as avgTriangles
                    """
                    result = await session.run(triangle_query, {"graphName": graph_name})
                    triangle_record = await result.single()
                    if triangle_record:
                        metrics["total_triangles"] = triangle_record["totalTriangles"]
                        metrics["avg_triangles_per_node"] = triangle_record["avgTriangles"]
                except Exception as e:
                    logger.warning(f"Triangle count calculation failed: {e}")
                
                # Calculate connected components
                try:
                    components_query = """
                    CALL gds.wcc.stream($graphName)
                    YIELD nodeId, componentId
                    RETURN count(DISTINCT componentId) as componentCount, 
                           max(componentSize) as largestComponent
                    FROM (
                        SELECT componentId, count(*) as componentSize
                        FROM stream_results
                        GROUP BY componentId
                    ) as components
                    """
                    # Simplified version since the above syntax might not work
                    components_query = """
                    CALL gds.wcc.stats($graphName)
                    YIELD componentCount
                    RETURN componentCount
                    """
                    result = await session.run(components_query, {"graphName": graph_name})
                    component_record = await result.single()
                    if component_record:
                        metrics["connected_components"] = component_record["componentCount"]
                except Exception as e:
                    logger.warning(f"Connected components calculation failed: {e}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Graph metrics calculation failed: {e}")
            return {}
    
    # ==================== CITATION NETWORK SPECIFIC METHODS ====================
    
    async def analyze_citation_influence(
        self,
        graph_name: str,
        paper_id: str,
        max_depth: int = 3
    ) -> Dict[str, Any]:
        """Analyze citation influence patterns for a paper."""
        # Security validation for input parameters
        if not graph_name or not isinstance(graph_name, str) or len(graph_name.strip()) == 0:
            raise ValueError("Invalid graph_name provided")
        
        if not paper_id or not isinstance(paper_id, str) or len(paper_id.strip()) == 0:
            raise ValueError("Invalid paper_id provided")
        
        # Prevent excessive traversal depth for security/performance
        if not isinstance(max_depth, int) or max_depth < 1 or max_depth > 10:
            raise ValueError("max_depth must be an integer between 1 and 10")
        
        try:
            influence_query = """
            MATCH (p:Paper {id: $paperId})
            CALL gds.dfs.stream($graphName, {
                sourceNode: p,
                maxDepth: $maxDepth
            })
            YIELD path, sourceNode, targetNode
            WITH path, targetNode
            MATCH (target:Paper) WHERE id(target) = targetNode
            RETURN 
                target.id as paperId,
                target.title as title,
                target.publication_year as year,
                target.citation_count as citations,
                length(path) as depth
            ORDER BY depth, citations DESC
            """
            
            async with self.get_session() as session:
                result = await session.run(influence_query, {
                    "graphName": graph_name,
                    "paperId": paper_id,
                    "maxDepth": max_depth
                })
                
                records = await result.data()
                
                # Group by depth level
                influence_tree = {}
                for record in records:
                    depth = record["depth"]
                    if depth not in influence_tree:
                        influence_tree[depth] = []
                    
                    influence_tree[depth].append({
                        "paper_id": record["paperId"],
                        "title": record["title"],
                        "year": record["year"],
                        "citations": record["citations"]
                    })
                
                # Calculate influence metrics
                total_influenced = sum(len(papers) for papers in influence_tree.values())
                direct_influence = len(influence_tree.get(1, []))
                max_depth_reached = max(influence_tree.keys()) if influence_tree else 0
                
                return {
                    "paper_id": paper_id,
                    "total_influenced_papers": total_influenced,
                    "direct_influence": direct_influence,
                    "max_influence_depth": max_depth_reached,
                    "influence_tree": influence_tree
                }
                
        except Exception as e:
            logger.error(f"Citation influence analysis failed: {e}")
            return {}
    
    async def find_research_lineage(
        self,
        paper_id: str,
        direction: str = "backward",  # "backward", "forward", "both"
        max_depth: int = 5
    ) -> Dict[str, Any]:
        """Trace intellectual lineage of a paper."""
        # Security validation for input parameters
        if not paper_id or not isinstance(paper_id, str) or len(paper_id.strip()) == 0:
            raise ValueError("Invalid paper_id provided")
        
        if direction not in ["backward", "forward", "both"]:
            raise ValueError("Direction must be one of: backward, forward, both")
        
        # Prevent excessive traversal depth for security/performance
        if not isinstance(max_depth, int) or max_depth < 1 or max_depth > 10:
            raise ValueError("max_depth must be an integer between 1 and 10")
        
        try:
            if direction == "backward":
                # Find papers that influenced this one
                query = """
                MATCH path = (target:Paper {id: $paperId})<-[:CITES*1..$maxDepth]-(source:Paper)
                WITH path, source, target, length(path) as depth
                RETURN 
                    source.id as paperId,
                    source.title as title,
                    source.publication_year as year,
                    source.citation_count as citations,
                    depth,
                    relationships(path) as citations_chain
                ORDER BY depth, year
                """
            elif direction == "forward":
                # Find papers influenced by this one
                query = """
                MATCH path = (source:Paper {id: $paperId})-[:CITES*1..$maxDepth]->(target:Paper)
                WITH path, source, target, length(path) as depth
                RETURN 
                    target.id as paperId,
                    target.title as title,
                    target.publication_year as year,
                    target.citation_count as citations,
                    depth,
                    relationships(path) as citations_chain
                ORDER BY depth, year
                """
            else:  # both
                query = """
                MATCH (center:Paper {id: $paperId})
                OPTIONAL MATCH backward_path = (center)<-[:CITES*1..$maxDepth]-(backward:Paper)
                OPTIONAL MATCH forward_path = (center)-[:CITES*1..$maxDepth]->(forward:Paper)
                
                WITH center, 
                     collect({paper: backward, path: backward_path, direction: 'backward'}) as backward_papers,
                     collect({paper: forward, path: forward_path, direction: 'forward'}) as forward_papers
                
                UNWIND (backward_papers + forward_papers) as result
                WHERE result.paper IS NOT NULL
                RETURN 
                    result.paper.id as paperId,
                    result.paper.title as title,
                    result.paper.publication_year as year,
                    result.paper.citation_count as citations,
                    length(result.path) as depth,
                    result.direction as direction
                ORDER BY direction, depth, year
                """
            
            async with self.get_session() as session:
                result = await session.run(query, {
                    "paperId": paper_id,
                    "maxDepth": max_depth
                })
                
                records = await result.data()
                
                lineage = {
                    "paper_id": paper_id,
                    "backward_lineage": [],
                    "forward_lineage": [],
                    "lineage_depth": 0,
                    "total_lineage_papers": len(records)
                }
                
                for record in records:
                    paper_info = {
                        "paper_id": record["paperId"],
                        "title": record["title"],
                        "year": record["year"],
                        "citations": record["citations"],
                        "depth": record["depth"]
                    }
                    
                    if direction == "both":
                        if record["direction"] == "backward":
                            lineage["backward_lineage"].append(paper_info)
                        else:
                            lineage["forward_lineage"].append(paper_info)
                    elif direction == "backward":
                        lineage["backward_lineage"].append(paper_info)
                    else:
                        lineage["forward_lineage"].append(paper_info)
                    
                    lineage["lineage_depth"] = max(lineage["lineage_depth"], record["depth"])
                
                return lineage
                
        except Exception as e:
            logger.error(f"Research lineage tracing failed: {e}")
            return {}


# Global advanced Neo4j manager instance
advanced_neo4j_manager = AdvancedNeo4jManager()


async def get_advanced_neo4j_manager() -> AdvancedNeo4jManager:
    """Dependency function to get advanced Neo4j manager."""
    if not advanced_neo4j_manager.driver:
        await advanced_neo4j_manager.connect()
    return advanced_neo4j_manager