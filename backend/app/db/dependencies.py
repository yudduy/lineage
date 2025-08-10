"""
Database dependency injection for FastAPI.
"""

from typing import AsyncGenerator, Optional
from fastapi import Depends
from neo4j import AsyncSession

from .neo4j import Neo4jManager, get_neo4j_manager
from .redis import RedisManager, get_redis_manager


async def get_db_session(
    db_manager: Neo4jManager = Depends(get_neo4j_manager)
) -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency to get Neo4j database session.
    
    Usage:
        @app.get("/papers/{paper_id}")
        async def get_paper(
            paper_id: str,
            db: AsyncSession = Depends(get_db_session)
        ):
            result = await db.run("MATCH (p:Paper {id: $id}) RETURN p", id=paper_id)
            return await result.single()
    """
    async with db_manager.get_session() as session:
        yield session


async def get_cache(
    redis_manager: RedisManager = Depends(get_redis_manager)
) -> RedisManager:
    """
    Dependency to get Redis cache manager.
    
    Usage:
        @app.get("/papers/{paper_id}")
        async def get_paper(
            paper_id: str,
            cache: RedisManager = Depends(get_cache)
        ):
            # Check cache first
            cached = await cache.cache_get(f"paper:{paper_id}")
            if cached:
                return cached
            
            # ... fetch from database and cache result
            await cache.cache_set(f"paper:{paper_id}", result, expire=3600)
            return result
    """
    return redis_manager


class DatabaseService:
    """
    Higher-level database service combining Neo4j and Redis operations.
    """
    
    def __init__(self, db_manager: Neo4jManager, cache_manager: RedisManager):
        self.db = db_manager
        self.cache = cache_manager
    
    async def get_paper_with_cache(
        self,
        paper_id: str,
        cache_ttl: int = 3600
    ) -> Optional[dict]:
        """Get paper with caching."""
        cache_key = f"paper:{paper_id}"
        
        # Try cache first
        cached_paper = await self.cache.cache_get(cache_key)
        if cached_paper:
            return cached_paper
        
        # Fetch from database
        paper = await self.db.get_paper_by_id(paper_id)
        if paper:
            # Cache the result
            await self.cache.cache_set(cache_key, paper, expire=cache_ttl)
        
        return paper
    
    async def invalidate_paper_cache(self, paper_id: str):
        """Invalidate paper cache."""
        cache_keys = [
            f"paper:{paper_id}",
            f"paper_citations:{paper_id}",
            f"paper_network:{paper_id}"
        ]
        await self.cache.cache_delete(*cache_keys)
    
    async def get_citation_network_with_cache(
        self,
        paper_id: str,
        max_depth: int = 2,
        cache_ttl: int = 1800  # 30 minutes
    ) -> dict:
        """Get citation network with caching."""
        cache_key = f"paper_network:{paper_id}:{max_depth}"
        
        # Try cache first
        cached_network = await self.cache.cache_get(cache_key)
        if cached_network:
            return cached_network
        
        # Fetch from database
        network = await self.db.get_citation_network(paper_id, max_depth)
        if network:
            # Cache the result
            await self.cache.cache_set(cache_key, network, expire=cache_ttl)
        
        return network
    
    async def search_papers_with_cache(
        self,
        query: str,
        limit: int = 20,
        skip: int = 0,
        cache_ttl: int = 600  # 10 minutes
    ) -> list:
        """Search papers with caching."""
        cache_key = f"search:{hash(query)}:{limit}:{skip}"
        
        # Try cache first
        cached_results = await self.cache.cache_get(cache_key)
        if cached_results:
            return cached_results
        
        # Search in database
        results = await self.db.search_papers(query, limit, skip)
        if results:
            # Cache the results
            await self.cache.cache_set(cache_key, results, expire=cache_ttl)
        
        return results


async def get_db_service(
    db_manager: Neo4jManager = Depends(get_neo4j_manager),
    cache_manager: RedisManager = Depends(get_redis_manager)
) -> DatabaseService:
    """
    Dependency to get combined database service.
    
    Usage:
        @app.get("/papers/{paper_id}")
        async def get_paper(
            paper_id: str,
            db_service: DatabaseService = Depends(get_db_service)
        ):
            return await db_service.get_paper_with_cache(paper_id)
    """
    return DatabaseService(db_manager, cache_manager)


# Connection pooling and management utilities
class ConnectionPool:
    """Manage database connection pools."""
    
    def __init__(self):
        self.neo4j_manager: Optional[Neo4jManager] = None
        self.redis_manager: Optional[RedisManager] = None
    
    async def initialize(self):
        """Initialize all database connections."""
        self.neo4j_manager = Neo4jManager()
        self.redis_manager = RedisManager()
        
        await self.neo4j_manager.connect()
        await self.redis_manager.connect()
    
    async def close_all(self):
        """Close all database connections."""
        if self.neo4j_manager:
            await self.neo4j_manager.disconnect()
        
        if self.redis_manager:
            await self.redis_manager.disconnect()
    
    async def health_check_all(self) -> dict:
        """Check health of all database connections."""
        health_status = {}
        
        if self.neo4j_manager:
            health_status["neo4j"] = await self.neo4j_manager.health_check()
        
        if self.redis_manager:
            health_status["redis"] = await self.redis_manager.health_check()
        
        return health_status


# Global connection pool
connection_pool = ConnectionPool()


async def get_connection_pool() -> ConnectionPool:
    """Get global connection pool."""
    return connection_pool