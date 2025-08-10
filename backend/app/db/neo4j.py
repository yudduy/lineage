"""
Neo4j database connection and management.
"""

from typing import Any, Dict, List, Optional
import asyncio
from contextlib import asynccontextmanager

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from neo4j.exceptions import ServiceUnavailable, AuthError

from ..core.config import get_settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


class Neo4jManager:
    """Neo4j database connection manager."""
    
    def __init__(self):
        self.settings = get_settings()
        self.driver: Optional[AsyncDriver] = None
        self._lock = asyncio.Lock()
        
    async def connect(self):
        """Establish connection to Neo4j."""
        async with self._lock:
            if self.driver is None:
                try:
                    self.driver = AsyncGraphDatabase.driver(
                        self.settings.neo4j_uri,
                        auth=(
                            self.settings.neo4j_user,
                            self.settings.neo4j_password
                        ),
                        database=self.settings.neo4j_database
                    )
                    
                    # Test connection
                    await self.driver.verify_connectivity()
                    logger.info("Successfully connected to Neo4j")
                    
                except (ServiceUnavailable, AuthError) as e:
                    logger.error(f"Failed to connect to Neo4j: {e}")
                    raise
                except Exception as e:
                    logger.error(f"Unexpected error connecting to Neo4j: {e}")
                    raise
    
    async def disconnect(self):
        """Close Neo4j connection."""
        async with self._lock:
            if self.driver:
                await self.driver.close()
                self.driver = None
                logger.info("Disconnected from Neo4j")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Neo4j connection health."""
        if not self.driver:
            return {
                "status": "unhealthy",
                "error": "No database connection"
            }
        
        try:
            start_time = asyncio.get_event_loop().time()
            
            async with self.driver.session() as session:
                result = await session.run("RETURN 1 as test")
                record = await result.single()
                
            end_time = asyncio.get_event_loop().time()
            response_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            return {
                "status": "healthy",
                "response_time_ms": round(response_time, 2),
                "test_result": record["test"] if record else None
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
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
    
    # Paper-specific database operations
    async def get_paper_by_id(self, paper_id: str) -> Optional[Dict]:
        """Get paper by ID."""
        query = """
        MATCH (p:Paper {id: $paper_id})
        RETURN p
        """
        results = await self.execute_read(query, {"paper_id": paper_id})
        return results[0]["p"] if results else None
    
    async def get_paper_by_doi(self, doi: str) -> Optional[Dict]:
        """Get paper by DOI."""
        query = """
        MATCH (p:Paper {doi: $doi})
        RETURN p
        """
        results = await self.execute_read(query, {"doi": doi})
        return results[0]["p"] if results else None
    
    async def create_paper(self, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create or update a paper node."""
        query = """
        MERGE (p:Paper {id: $id})
        SET p += $properties
        SET p.updated_at = datetime()
        RETURN p
        """
        results = await self.execute_write(query, {
            "id": paper_data["id"],
            "properties": paper_data
        })
        return results
    
    async def create_citation_edge(self, citing_id: str, cited_id: str) -> Dict[str, Any]:
        """Create a citation edge between two papers."""
        query = """
        MATCH (citing:Paper {id: $citing_id})
        MATCH (cited:Paper {id: $cited_id})
        MERGE (citing)-[r:CITES]->(cited)
        SET r.created_at = datetime()
        RETURN r
        """
        return await self.execute_write(query, {
            "citing_id": citing_id,
            "cited_id": cited_id
        })
    
    async def get_paper_citations(
        self,
        paper_id: str,
        direction: str = "both",
        limit: int = 100
    ) -> List[Dict]:
        """Get paper citations (references or cited by)."""
        if direction == "references":
            query = """
            MATCH (p:Paper {id: $paper_id})-[:CITES]->(ref:Paper)
            RETURN ref
            LIMIT $limit
            """
        elif direction == "cited_by":
            query = """
            MATCH (citing:Paper)-[:CITES]->(p:Paper {id: $paper_id})
            RETURN citing as ref
            LIMIT $limit
            """
        else:  # both
            query = """
            MATCH (p:Paper {id: $paper_id})
            OPTIONAL MATCH (p)-[:CITES]->(ref:Paper)
            OPTIONAL MATCH (citing:Paper)-[:CITES]->(p)
            RETURN COALESCE(ref, citing) as ref
            LIMIT $limit
            """
        
        results = await self.execute_read(query, {"paper_id": paper_id, "limit": limit})
        return [r["ref"] for r in results if r["ref"]]
    
    async def get_citation_network(
        self,
        paper_id: str,
        max_depth: int = 2,
        max_nodes: int = 1000
    ) -> Dict[str, Any]:
        """Get citation network around a paper."""
        query = """
        MATCH path = (p:Paper {id: $paper_id})-[:CITES*1..$max_depth]-(connected:Paper)
        WITH collect(DISTINCT connected) + [p] as nodes
        UNWIND nodes as node
        MATCH (node)-[r:CITES]-(other)
        WHERE other IN nodes
        RETURN 
            collect(DISTINCT node) as nodes,
            collect(DISTINCT {
                source: startNode(r).id,
                target: endNode(r).id,
                relationship: type(r)
            }) as edges
        LIMIT $max_nodes
        """
        
        results = await self.execute_read(query, {
            "paper_id": paper_id,
            "max_depth": max_depth,
            "max_nodes": max_nodes
        })
        
        if not results:
            return {"nodes": [], "edges": []}
        
        return {
            "nodes": results[0].get("nodes", []),
            "edges": results[0].get("edges", [])
        }
    
    async def search_papers(
        self,
        query: str,
        limit: int = 20,
        skip: int = 0
    ) -> List[Dict]:
        """Search papers by title, authors, or content."""
        cypher_query = """
        CALL db.index.fulltext.queryNodes('paperFulltext', $query)
        YIELD node as p, score
        RETURN p, score
        ORDER BY score DESC
        SKIP $skip
        LIMIT $limit
        """
        
        results = await self.execute_read(cypher_query, {
            "query": query,
            "skip": skip,
            "limit": limit
        })
        
        return [{"paper": r["p"], "score": r["score"]} for r in results]
    
    async def create_paper(self, paper_data: Dict) -> str:
        """Create new paper node."""
        query = """
        CREATE (p:Paper $properties)
        RETURN p.id as paper_id
        """
        
        # Ensure paper has an ID
        if "id" not in paper_data:
            import uuid
            paper_data["id"] = str(uuid.uuid4())
        
        results = await self.execute_read(query, {"properties": paper_data})
        return results[0]["paper_id"] if results else paper_data["id"]
    
    async def update_paper(self, paper_id: str, updates: Dict) -> bool:
        """Update paper properties."""
        query = """
        MATCH (p:Paper {id: $paper_id})
        SET p += $updates
        RETURN p.id as updated_id
        """
        
        results = await self.execute_read(query, {
            "paper_id": paper_id,
            "updates": updates
        })
        
        return len(results) > 0
    
    async def create_citation_relationship(
        self,
        citing_paper_id: str,
        cited_paper_id: str,
        context: Optional[str] = None
    ) -> bool:
        """Create citation relationship between papers."""
        query = """
        MATCH (citing:Paper {id: $citing_id})
        MATCH (cited:Paper {id: $cited_id})
        MERGE (citing)-[r:CITES]->(cited)
        SET r.context = $context
        RETURN r
        """
        
        results = await self.execute_read(query, {
            "citing_id": citing_paper_id,
            "cited_id": cited_paper_id,
            "context": context
        })
        
        return len(results) > 0
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        queries = {
            "total_papers": "MATCH (p:Paper) RETURN count(p) as count",
            "total_citations": "MATCH ()-[r:CITES]->() RETURN count(r) as count",
            "papers_with_dois": "MATCH (p:Paper) WHERE p.doi IS NOT NULL RETURN count(p) as count",
            "average_citations": """
                MATCH (p:Paper)
                OPTIONAL MATCH (p)<-[:CITES]-(citing)
                RETURN avg(count(citing)) as avg_citations
            """
        }
        
        stats = {}
        for stat_name, query in queries.items():
            try:
                results = await self.execute_read(query)
                stats[stat_name] = results[0].get("count", 0) if "count" in query else results[0].get("avg_citations", 0)
            except Exception as e:
                logger.error(f"Error getting {stat_name}: {e}")
                stats[stat_name] = 0
        
        return stats


# Global Neo4j manager instance
neo4j_manager = Neo4jManager()


async def get_neo4j_manager() -> Neo4jManager:
    """Dependency function to get Neo4j manager."""
    if not neo4j_manager.driver:
        await neo4j_manager.connect()
    return neo4j_manager