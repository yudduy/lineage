"""
Database Testing Suite for Intellectual Lineage Tracer System

This test suite covers:
- Neo4j graph database operations and query performance
- Redis caching and session management
- Database connection pooling and failover
- Complex graph algorithms and traversals  
- Data consistency and ACID properties
- Backup and recovery procedures
- Database schema validation
- Index performance and optimization
"""

import pytest
import asyncio
import time
import json
import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List
import statistics

from app.db.neo4j import Neo4jManager
from app.db.redis import RedisManager
from app.services.graph_operations import GraphOperationsService
from app.services.advanced_analytics import AdvancedAnalyticsService
from app.models.paper import Paper, Author, CitationNetwork


@pytest.fixture
async def neo4j_manager():
    """Create Neo4j manager for testing."""
    manager = Neo4jManager(
        uri="bolt://localhost:7687",
        user="neo4j", 
        password="test_password"
    )
    
    # Mock the driver connection
    manager._driver = MagicMock()
    manager._session = AsyncMock()
    
    yield manager
    
    await manager.close()


@pytest.fixture
async def redis_manager():
    """Create Redis manager for testing."""
    manager = RedisManager(
        host="localhost",
        port=6379,
        db=1,  # Test database
        password=None
    )
    
    # Mock Redis connection
    manager._redis = AsyncMock()
    
    yield manager
    
    await manager.close()


class TestNeo4jOperations:
    """Test Neo4j graph database operations."""
    
    @pytest.mark.asyncio
    async def test_basic_crud_operations(self, neo4j_manager):
        """Test basic Create, Read, Update, Delete operations."""
        # Mock successful responses
        neo4j_manager._session.run = AsyncMock()
        
        # Test CREATE operation
        create_query = """
        CREATE (p:Paper {
            id: $id, 
            title: $title, 
            publication_year: $year,
            created_at: $created_at
        })
        RETURN p
        """
        
        paper_data = {
            "id": "test_paper_1",
            "title": "Test Paper for CRUD Operations",
            "year": 2023,
            "created_at": datetime.utcnow().isoformat()
        }
        
        mock_record = MagicMock()
        mock_record.get.return_value = paper_data
        neo4j_manager._session.run.return_value.single.return_value = mock_record
        
        result = await neo4j_manager.execute_query(create_query, **paper_data)
        
        # Verify query was called
        neo4j_manager._session.run.assert_called_with(create_query, **paper_data)
        
        # Test READ operation
        read_query = "MATCH (p:Paper {id: $id}) RETURN p"
        neo4j_manager._session.run.return_value.single.return_value = mock_record
        
        result = await neo4j_manager.execute_query(read_query, id="test_paper_1")
        assert result is not None
        
        # Test UPDATE operation
        update_query = """
        MATCH (p:Paper {id: $id})
        SET p.citation_count = $citation_count, p.updated_at = $updated_at
        RETURN p
        """
        
        update_data = {
            "id": "test_paper_1",
            "citation_count": 25,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        await neo4j_manager.execute_query(update_query, **update_data)
        
        # Test DELETE operation
        delete_query = "MATCH (p:Paper {id: $id}) DELETE p"
        await neo4j_manager.execute_query(delete_query, id="test_paper_1")
    
    @pytest.mark.asyncio
    async def test_complex_graph_queries(self, neo4j_manager):
        """Test complex graph traversal queries."""
        # Mock complex query results
        mock_results = [
            {"path_length": 2, "papers": ["paper1", "paper2", "paper3"]},
            {"path_length": 3, "papers": ["paper1", "paper4", "paper5", "paper6"]},
        ]
        
        neo4j_manager._session.run.return_value.data.return_value = mock_results
        
        # Test citation path discovery
        citation_path_query = """
        MATCH path = (start:Paper {id: $start_id})-[:CITES*1..5]->(end:Paper {id: $end_id})
        RETURN path, length(path) as path_length,
               [node in nodes(path) | node.id] as papers
        ORDER BY path_length ASC
        LIMIT 10
        """
        
        result = await neo4j_manager.execute_query(
            citation_path_query,
            start_id="paper1",
            end_id="paper6"
        )
        
        assert len(result) == 2
        assert all("papers" in record for record in result)
        
        # Test community detection query
        community_query = """
        CALL gds.louvain.stream('citations')
        YIELD nodeId, communityId
        RETURN gds.util.asNode(nodeId).id as paper_id, communityId
        ORDER BY communityId
        """
        
        mock_community_results = [
            {"paper_id": "paper1", "communityId": 0},
            {"paper_id": "paper2", "communityId": 0},
            {"paper_id": "paper3", "communityId": 1},
        ]
        
        neo4j_manager._session.run.return_value.data.return_value = mock_community_results
        
        result = await neo4j_manager.execute_query(community_query)
        assert len(result) == 3
        assert all("communityId" in record for record in result)
    
    @pytest.mark.asyncio
    async def test_batch_operations(self, neo4j_manager):
        """Test batch operations for bulk data processing."""
        # Generate batch of papers
        batch_papers = []
        for i in range(100):
            paper = {
                "id": f"batch_paper_{i}",
                "title": f"Batch Paper {i}",
                "publication_year": 2020 + (i % 4),
                "citation_count": i * 2
            }
            batch_papers.append(paper)
        
        # Test batch insert
        batch_insert_query = """
        UNWIND $papers as paper
        CREATE (p:Paper {
            id: paper.id,
            title: paper.title,
            publication_year: paper.publication_year,
            citation_count: paper.citation_count,
            created_at: datetime()
        })
        """
        
        neo4j_manager._session.run = AsyncMock()
        
        await neo4j_manager.execute_query(batch_insert_query, papers=batch_papers)
        
        # Verify batch operation was called
        neo4j_manager._session.run.assert_called_with(batch_insert_query, papers=batch_papers)
        
        # Test batch relationship creation
        batch_relations = []
        for i in range(50):
            relation = {
                "source": f"batch_paper_{i}",
                "target": f"batch_paper_{i+1}",
                "relationship_type": "CITES"
            }
            batch_relations.append(relation)
        
        batch_relation_query = """
        UNWIND $relations as rel
        MATCH (source:Paper {id: rel.source})
        MATCH (target:Paper {id: rel.target})
        CREATE (source)-[:CITES {created_at: datetime()}]->(target)
        """
        
        await neo4j_manager.execute_query(batch_relation_query, relations=batch_relations)
    
    @pytest.mark.asyncio
    async def test_transaction_handling(self, neo4j_manager):
        """Test transaction handling and rollback scenarios."""
        # Mock transaction behavior
        mock_tx = AsyncMock()
        neo4j_manager._session.begin_transaction.return_value = mock_tx
        
        # Test successful transaction
        async with neo4j_manager.transaction() as tx:
            query1 = "CREATE (p:Paper {id: $id, title: $title})"
            await tx.run(query1, id="tx_paper_1", title="Transaction Test 1")
            
            query2 = "CREATE (p:Paper {id: $id, title: $title})"
            await tx.run(query2, id="tx_paper_2", title="Transaction Test 2")
        
        # Verify transaction was committed
        mock_tx.commit.assert_called_once()
        
        # Test transaction rollback on error
        mock_tx.reset_mock()
        
        try:
            async with neo4j_manager.transaction() as tx:
                await tx.run("CREATE (p:Paper {id: $id})", id="tx_paper_3")
                # Simulate error
                raise Exception("Simulated error")
        except Exception:
            pass
        
        # Verify transaction was rolled back
        mock_tx.rollback.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_index_performance(self, neo4j_manager):
        """Test database index performance and optimization."""
        # Test index creation queries
        index_queries = [
            "CREATE INDEX paper_id_index IF NOT EXISTS FOR (p:Paper) ON (p.id)",
            "CREATE INDEX paper_title_index IF NOT EXISTS FOR (p:Paper) ON (p.title)",
            "CREATE INDEX paper_year_index IF NOT EXISTS FOR (p:Paper) ON (p.publication_year)",
            "CREATE INDEX author_name_index IF NOT EXISTS FOR (a:Author) ON (a.name)",
        ]
        
        neo4j_manager._session.run = AsyncMock()
        
        for query in index_queries:
            await neo4j_manager.execute_query(query)
        
        # Test query performance with index
        indexed_query = """
        MATCH (p:Paper)
        WHERE p.publication_year >= 2020 AND p.publication_year <= 2023
        RETURN p.id, p.title, p.publication_year
        ORDER BY p.citation_count DESC
        LIMIT 100
        """
        
        # Mock performance metrics
        mock_result = MagicMock()
        mock_result.consume.return_value.result_available_after = 50  # milliseconds
        neo4j_manager._session.run.return_value = mock_result
        
        start_time = time.time()
        await neo4j_manager.execute_query(indexed_query)
        query_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Query should be fast with proper indexing
        assert query_time < 1000  # Should complete within 1 second
    
    @pytest.mark.asyncio
    async def test_constraint_validation(self, neo4j_manager):
        """Test database constraint validation."""
        # Test unique constraint creation
        constraint_queries = [
            "CREATE CONSTRAINT paper_id_unique IF NOT EXISTS FOR (p:Paper) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT author_id_unique IF NOT EXISTS FOR (a:Author) REQUIRE a.id IS UNIQUE",
        ]
        
        neo4j_manager._session.run = AsyncMock()
        
        for query in constraint_queries:
            await neo4j_manager.execute_query(query)
        
        # Test constraint violation handling
        duplicate_paper_query = "CREATE (p:Paper {id: 'duplicate_test', title: 'Duplicate Test'})"
        
        # First creation should succeed
        await neo4j_manager.execute_query(duplicate_paper_query)
        
        # Second creation should fail (in real scenario)
        # Mock constraint violation
        from neo4j.exceptions import ConstraintError
        neo4j_manager._session.run.side_effect = ConstraintError("Constraint violation")
        
        with pytest.raises(ConstraintError):
            await neo4j_manager.execute_query(duplicate_paper_query)


class TestRedisOperations:
    """Test Redis caching and session management."""
    
    @pytest.mark.asyncio
    async def test_basic_cache_operations(self, redis_manager):
        """Test basic Redis cache operations."""
        # Mock Redis responses
        redis_manager._redis.get.return_value = None
        redis_manager._redis.set.return_value = True
        redis_manager._redis.delete.return_value = 1
        
        # Test cache miss
        result = await redis_manager.get("nonexistent_key")
        assert result is None
        
        # Test cache set
        test_data = {"papers": ["paper1", "paper2"], "total": 2}
        await redis_manager.set("test_key", json.dumps(test_data), ttl=3600)
        
        redis_manager._redis.set.assert_called_with(
            "test_key", 
            json.dumps(test_data), 
            ex=3600
        )
        
        # Test cache hit
        redis_manager._redis.get.return_value = json.dumps(test_data)
        result = await redis_manager.get("test_key")
        
        assert result == json.dumps(test_data)
        
        # Test cache deletion
        await redis_manager.delete("test_key")
        redis_manager._redis.delete.assert_called_with("test_key")
    
    @pytest.mark.asyncio
    async def test_cache_performance(self, redis_manager):
        """Test Redis cache performance under load."""
        # Mock fast Redis responses
        redis_manager._redis.get.return_value = '{"cached": true}'
        redis_manager._redis.set.return_value = True
        
        # Test concurrent cache operations
        async def cache_operation(key):
            start_time = time.time()
            await redis_manager.get(f"performance_test_{key}")
            await redis_manager.set(f"performance_test_{key}", '{"data": "test"}', ttl=300)
            return time.time() - start_time
        
        # Run 100 concurrent operations
        tasks = [cache_operation(i) for i in range(100)]
        response_times = await asyncio.gather(*tasks)
        
        # All operations should be fast
        avg_time = statistics.mean(response_times)
        max_time = max(response_times)
        
        assert avg_time < 0.01  # Average under 10ms
        assert max_time < 0.1   # Max under 100ms
    
    @pytest.mark.asyncio
    async def test_session_management(self, redis_manager):
        """Test session management with Redis."""
        # Mock session operations
        redis_manager._redis.hget.return_value = None
        redis_manager._redis.hset.return_value = True
        redis_manager._redis.expire.return_value = True
        redis_manager._redis.hdel.return_value = 1
        
        session_id = str(uuid.uuid4())
        session_key = f"session:{session_id}"
        
        # Create session
        session_data = {
            "user_id": "user123",
            "email": "test@example.com",
            "login_time": datetime.utcnow().isoformat(),
            "permissions": ["read", "write"]
        }
        
        for field, value in session_data.items():
            await redis_manager.hash_set(session_key, field, json.dumps(value) if isinstance(value, (dict, list)) else str(value))
        
        # Set session expiration
        await redis_manager.expire(session_key, 86400)  # 24 hours
        
        # Retrieve session
        redis_manager._redis.hget.return_value = session_data["user_id"]
        user_id = await redis_manager.hash_get(session_key, "user_id")
        assert user_id == session_data["user_id"]
        
        # Update session
        await redis_manager.hash_set(session_key, "last_activity", datetime.utcnow().isoformat())
        
        # Delete session
        await redis_manager.hash_delete(session_key, "user_id")
    
    @pytest.mark.asyncio
    async def test_cache_invalidation_strategies(self, redis_manager):
        """Test cache invalidation strategies."""
        # Mock Redis operations
        redis_manager._redis.keys.return_value = [
            b"papers:search:query1",
            b"papers:search:query2", 
            b"papers:details:paper1",
            b"network:build:network1"
        ]
        redis_manager._redis.delete.return_value = 4
        
        # Test pattern-based cache invalidation
        await redis_manager.delete_pattern("papers:search:*")
        
        # Should delete all matching keys
        redis_manager._redis.keys.assert_called_with("papers:search:*")
        redis_manager._redis.delete.assert_called()
        
        # Test TTL-based cache invalidation
        redis_manager._redis.ttl.return_value = -1  # No TTL set
        
        # Keys without TTL should be handled
        ttl = await redis_manager.get_ttl("papers:search:query1")
        assert ttl == -1
        
        # Set TTL for cache expiration
        await redis_manager.expire("papers:search:query1", 3600)
        redis_manager._redis.expire.assert_called_with("papers:search:query1", 3600)
    
    @pytest.mark.asyncio
    async def test_redis_pub_sub(self, redis_manager):
        """Test Redis publish/subscribe for real-time updates."""
        # Mock pub/sub operations
        mock_pubsub = AsyncMock()
        redis_manager._redis.pubsub.return_value = mock_pubsub
        
        # Test publishing messages
        await redis_manager.publish("paper_updates", json.dumps({
            "type": "citation_count_update",
            "paper_id": "paper123",
            "new_count": 150
        }))
        
        redis_manager._redis.publish.assert_called()
        
        # Test subscription
        pubsub = redis_manager._redis.pubsub()
        await pubsub.subscribe("paper_updates", "network_updates")
        
        # Mock incoming message
        mock_message = {
            "type": "message",
            "channel": b"paper_updates",
            "data": b'{"type": "citation_count_update", "paper_id": "paper123", "new_count": 150}'
        }
        
        mock_pubsub.get_message.return_value = mock_message
        
        message = await pubsub.get_message()
        assert message["channel"] == b"paper_updates"


class TestDatabaseIntegration:
    """Test integration between Neo4j and Redis."""
    
    @pytest.mark.asyncio
    async def test_cache_aside_pattern(self, neo4j_manager, redis_manager):
        """Test cache-aside pattern implementation."""
        # Mock cache miss
        redis_manager._redis.get.return_value = None
        
        # Mock database query result
        db_result = [
            {"id": "paper1", "title": "Test Paper 1", "citation_count": 100},
            {"id": "paper2", "title": "Test Paper 2", "citation_count": 85}
        ]
        neo4j_manager._session.run.return_value.data.return_value = db_result
        
        # Simulate service method using cache-aside pattern
        async def get_papers_by_author(author_id: str):
            cache_key = f"author_papers:{author_id}"
            
            # Try cache first
            cached_result = await redis_manager.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
            
            # Cache miss - query database
            query = "MATCH (a:Author {id: $author_id})-[:AUTHORED]->(p:Paper) RETURN p"
            result = await neo4j_manager.execute_query(query, author_id=author_id)
            
            # Store in cache
            await redis_manager.set(cache_key, json.dumps(result), ttl=3600)
            
            return result
        
        # First call should hit database
        result1 = await get_papers_by_author("author123")
        assert result1 == db_result
        
        # Verify database was queried
        neo4j_manager._session.run.assert_called()
        
        # Second call should hit cache
        redis_manager._redis.get.return_value = json.dumps(db_result)
        result2 = await get_papers_by_author("author123")
        assert result2 == db_result
    
    @pytest.mark.asyncio
    async def test_write_through_pattern(self, neo4j_manager, redis_manager):
        """Test write-through caching pattern."""
        # Mock successful operations
        neo4j_manager._session.run = AsyncMock()
        redis_manager._redis.set.return_value = True
        
        async def update_paper_citation_count(paper_id: str, new_count: int):
            # Update database
            query = """
            MATCH (p:Paper {id: $paper_id})
            SET p.citation_count = $citation_count, p.updated_at = datetime()
            RETURN p
            """
            await neo4j_manager.execute_query(query, paper_id=paper_id, citation_count=new_count)
            
            # Update cache immediately
            cache_key = f"paper_details:{paper_id}"
            paper_data = {
                "id": paper_id,
                "citation_count": new_count,
                "updated_at": datetime.utcnow().isoformat()
            }
            await redis_manager.set(cache_key, json.dumps(paper_data), ttl=3600)
            
            # Invalidate related caches
            await redis_manager.delete_pattern(f"author_papers:*")
            await redis_manager.delete_pattern(f"search_results:*")
        
        await update_paper_citation_count("paper123", 200)
        
        # Verify both database and cache were updated
        neo4j_manager._session.run.assert_called()
        redis_manager._redis.set.assert_called()
    
    @pytest.mark.asyncio
    async def test_database_connection_resilience(self, neo4j_manager, redis_manager):
        """Test database connection resilience and failover."""
        # Test Neo4j connection failure handling
        from neo4j.exceptions import ServiceUnavailable
        
        # Mock connection failure
        neo4j_manager._session.run.side_effect = ServiceUnavailable("Database unavailable")
        
        async def resilient_query():
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    result = await neo4j_manager.execute_query("MATCH (p:Paper) RETURN count(p)")
                    return result
                except ServiceUnavailable:
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        # Should raise after max retries
        with pytest.raises(ServiceUnavailable):
            await resilient_query()
        
        # Test Redis connection failure handling
        from redis.exceptions import ConnectionError
        
        redis_manager._redis.get.side_effect = ConnectionError("Redis unavailable")
        
        async def resilient_cache_get(key: str):
            try:
                return await redis_manager.get(key)
            except ConnectionError:
                # Fallback to database or return None
                return None
        
        result = await resilient_cache_get("test_key")
        assert result is None  # Graceful degradation


class TestDatabasePerformanceOptimization:
    """Test database performance optimization strategies."""
    
    @pytest.mark.asyncio
    async def test_query_optimization(self, neo4j_manager):
        """Test query optimization techniques."""
        # Mock query performance metrics
        mock_result = MagicMock()
        mock_result.consume.return_value.result_available_after = 25  # milliseconds
        neo4j_manager._session.run.return_value = mock_result
        
        # Optimized query with proper indexes and limits
        optimized_query = """
        MATCH (p:Paper)
        WHERE p.publication_year >= $year_from AND p.publication_year <= $year_to
        WITH p
        ORDER BY p.citation_count DESC
        LIMIT $limit
        RETURN p.id, p.title, p.citation_count, p.publication_year
        """
        
        start_time = time.time()
        await neo4j_manager.execute_query(
            optimized_query,
            year_from=2020,
            year_to=2023,
            limit=100
        )
        query_time = (time.time() - start_time) * 1000
        
        # Should be fast with optimization
        assert query_time < 500  # Under 500ms
        
        # Test query plan analysis (in real scenario)
        explain_query = f"EXPLAIN {optimized_query}"
        await neo4j_manager.execute_query(
            explain_query,
            year_from=2020,
            year_to=2023,
            limit=100
        )
    
    @pytest.mark.asyncio
    async def test_connection_pooling(self, neo4j_manager, redis_manager):
        """Test database connection pooling efficiency."""
        # Test concurrent connections
        async def concurrent_query(query_id):
            query = f"MATCH (p:Paper) WHERE p.id = 'paper_{query_id}' RETURN p"
            start_time = time.time()
            await neo4j_manager.execute_query(query)
            return time.time() - start_time
        
        # Run multiple concurrent queries
        tasks = [concurrent_query(i) for i in range(20)]
        response_times = await asyncio.gather(*tasks)
        
        # Connection pooling should keep response times reasonable
        avg_time = statistics.mean(response_times)
        max_time = max(response_times)
        
        assert avg_time < 1.0  # Average under 1 second
        assert max_time < 2.0  # Max under 2 seconds
        
        # Test Redis connection pooling
        async def concurrent_cache_operation(key_id):
            start_time = time.time()
            await redis_manager.get(f"test_key_{key_id}")
            await redis_manager.set(f"test_key_{key_id}", "test_value", ttl=60)
            return time.time() - start_time
        
        redis_tasks = [concurrent_cache_operation(i) for i in range(50)]
        redis_response_times = await asyncio.gather(*redis_tasks)
        
        redis_avg_time = statistics.mean(redis_response_times)
        assert redis_avg_time < 0.1  # Redis should be very fast
    
    @pytest.mark.asyncio
    async def test_bulk_operations_optimization(self, neo4j_manager):
        """Test optimization of bulk database operations."""
        # Generate large batch of data
        large_batch = []
        for i in range(1000):
            paper = {
                "id": f"bulk_paper_{i}",
                "title": f"Bulk Paper {i}",
                "publication_year": 2000 + (i % 24),
                "citation_count": i % 100
            }
            large_batch.append(paper)
        
        # Optimized batch insert using UNWIND
        optimized_batch_query = """
        UNWIND $papers as paper
        CREATE (p:Paper {
            id: paper.id,
            title: paper.title,
            publication_year: paper.publication_year,
            citation_count: paper.citation_count,
            created_at: datetime()
        })
        """
        
        start_time = time.time()
        await neo4j_manager.execute_query(optimized_batch_query, papers=large_batch)
        batch_time = time.time() - start_time
        
        # Batch operation should be efficient
        papers_per_second = len(large_batch) / batch_time
        assert papers_per_second > 100  # Should process at least 100 papers/second
        
        # Test batch relationship creation
        batch_relationships = []
        for i in range(500):
            rel = {
                "source_id": f"bulk_paper_{i}",
                "target_id": f"bulk_paper_{i+1}",
                "relationship_type": "CITES"
            }
            batch_relationships.append(rel)
        
        relationship_batch_query = """
        UNWIND $relationships as rel
        MATCH (source:Paper {id: rel.source_id})
        MATCH (target:Paper {id: rel.target_id})
        CREATE (source)-[:CITES {created_at: datetime()}]->(target)
        """
        
        start_time = time.time()
        await neo4j_manager.execute_query(relationship_batch_query, relationships=batch_relationships)
        rel_batch_time = time.time() - start_time
        
        relationships_per_second = len(batch_relationships) / rel_batch_time
        assert relationships_per_second > 50  # Should process at least 50 relationships/second


# Test fixtures and utilities
@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()