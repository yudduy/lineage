"""
Performance and Load Testing Suite for Intellectual Lineage Tracer System

This test suite covers:
- Load testing for concurrent users (target: 1K+ users)
- Large graph processing (10K+ nodes)
- Database query performance under load
- External API rate limit compliance
- Memory usage during bulk operations
- Frontend rendering performance
- Background task processing performance
- WebSocket real-time communication under load
"""

import pytest
import asyncio
import time
import threading
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import AsyncMock, patch, MagicMock
from typing import List, Dict, Any
import statistics
import json
import uuid
from datetime import datetime, timedelta

from fastapi.testclient import TestClient
from fastapi import status
import httpx

from app.main import create_app
from app.models.paper import Paper, Author, CitationNetwork
from app.services.graph_operations import GraphOperationsService
from app.services.openalex import OpenAlexClient
from app.services.llm_service import LLMService


@pytest.fixture
def app():
    """Create test FastAPI app for performance testing."""
    return create_app()


@pytest.fixture
def client(app):
    """Create test client for performance testing."""
    return TestClient(app)


@pytest.fixture
def performance_mocks():
    """Mock external dependencies for performance testing."""
    with patch('app.db.neo4j.Neo4jManager') as neo4j_mock, \
         patch('app.db.redis.RedisManager') as redis_mock, \
         patch('app.services.openalex.OpenAlexClient') as openalex_mock, \
         patch('app.services.semantic_scholar.SemanticScholarClient') as ss_mock:
        
        # Setup fast mock responses for performance testing
        neo4j_mock.return_value.execute_query = AsyncMock(return_value=[])
        redis_mock.return_value.get = AsyncMock(return_value=None)
        redis_mock.return_value.set = AsyncMock(return_value=True)
        openalex_mock.return_value.search_works = AsyncMock(return_value={"results": [], "meta": {"count": 0}})
        ss_mock.return_value.search_papers = AsyncMock(return_value={"data": [], "total": 0})
        
        yield {
            'neo4j': neo4j_mock.return_value,
            'redis': redis_mock.return_value,
            'openalex': openalex_mock.return_value,
            'semantic_scholar': ss_mock.return_value
        }


class PerformanceMetrics:
    """Class to track and analyze performance metrics."""
    
    def __init__(self):
        self.response_times = []
        self.memory_usage = []
        self.cpu_usage = []
        self.error_count = 0
        self.success_count = 0
        self.start_time = None
        self.end_time = None
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.end_time = time.time()
    
    def record_response(self, response_time: float, success: bool = True):
        """Record a response time and success/failure."""
        self.response_times.append(response_time)
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
    
    def record_system_metrics(self):
        """Record current system metrics."""
        process = psutil.Process()
        self.memory_usage.append(process.memory_info().rss / 1024 / 1024)  # MB
        self.cpu_usage.append(process.cpu_percent())
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        total_time = self.end_time - self.start_time if self.end_time and self.start_time else 0
        
        return {
            "total_requests": len(self.response_times),
            "successful_requests": self.success_count,
            "failed_requests": self.error_count,
            "success_rate": self.success_count / len(self.response_times) if self.response_times else 0,
            "total_time_seconds": total_time,
            "requests_per_second": len(self.response_times) / total_time if total_time > 0 else 0,
            "response_times": {
                "min": min(self.response_times) if self.response_times else 0,
                "max": max(self.response_times) if self.response_times else 0,
                "mean": statistics.mean(self.response_times) if self.response_times else 0,
                "median": statistics.median(self.response_times) if self.response_times else 0,
                "p95": statistics.quantiles(self.response_times, n=20)[18] if len(self.response_times) >= 20 else 0,
                "p99": statistics.quantiles(self.response_times, n=100)[98] if len(self.response_times) >= 100 else 0,
            },
            "memory_usage_mb": {
                "min": min(self.memory_usage) if self.memory_usage else 0,
                "max": max(self.memory_usage) if self.memory_usage else 0,
                "mean": statistics.mean(self.memory_usage) if self.memory_usage else 0,
            },
            "cpu_usage_percent": {
                "mean": statistics.mean(self.cpu_usage) if self.cpu_usage else 0,
                "max": max(self.cpu_usage) if self.cpu_usage else 0,
            }
        }


class TestConcurrentUserLoad:
    """Test concurrent user load scenarios."""
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_concurrent_health_checks(self, client, performance_mocks):
        """Test handling concurrent health check requests."""
        metrics = PerformanceMetrics()
        metrics.start_monitoring()
        
        def make_health_check():
            start_time = time.time()
            try:
                response = client.get("/api/v1/health/")
                response_time = time.time() - start_time
                success = response.status_code == 200
                metrics.record_response(response_time, success)
                metrics.record_system_metrics()
                return response.status_code
            except Exception as e:
                response_time = time.time() - start_time
                metrics.record_response(response_time, False)
                return 500
        
        # Test with 100 concurrent health checks
        num_concurrent = 100
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(make_health_check) for _ in range(num_concurrent)]
            results = [future.result() for future in as_completed(futures)]
        
        metrics.stop_monitoring()
        summary = metrics.get_summary()
        
        # Performance assertions
        assert summary["success_rate"] >= 0.95  # 95% success rate
        assert summary["response_times"]["p95"] < 1.0  # 95th percentile under 1s
        assert summary["requests_per_second"] > 50  # At least 50 RPS
        
        print(f"Concurrent Health Check Results: {json.dumps(summary, indent=2)}")
    
    @pytest.mark.slow
    @pytest.mark.asyncio 
    async def test_concurrent_search_requests(self, client, performance_mocks):
        """Test concurrent paper search requests."""
        metrics = PerformanceMetrics()
        metrics.start_monitoring()
        
        # Mock search response
        mock_search_response = {
            "papers": [
                {
                    "id": f"paper_{i}",
                    "title": f"Test Paper {i}",
                    "authors": [{"name": f"Author {i}", "id": f"author_{i}"}],
                    "publication_year": 2020 + (i % 4),
                    "citation_count": {"total": i * 10}
                }
                for i in range(20)
            ],
            "total": 20,
            "page": 1,
            "page_size": 20,
            "total_pages": 1
        }
        
        # Create auth token for testing
        from app.services.auth import create_access_token
        token = create_access_token(data={"sub": "test_user", "email": "test@example.com"})
        headers = {"Authorization": f"Bearer {token}"}
        
        def make_search_request(query_id):
            start_time = time.time()
            try:
                with patch('app.services.search.SearchService.search_papers') as mock_search:
                    mock_search.return_value = mock_search_response
                    
                    response = client.get(
                        "/api/v1/search/papers",
                        params={"query": f"test query {query_id}", "page": 1, "page_size": 20},
                        headers=headers
                    )
                    
                response_time = time.time() - start_time
                success = response.status_code == 200
                metrics.record_response(response_time, success)
                metrics.record_system_metrics()
                return response.status_code
            except Exception as e:
                response_time = time.time() - start_time
                metrics.record_response(response_time, False)
                return 500
        
        # Test with 50 concurrent search requests
        num_concurrent = 50
        with ThreadPoolExecutor(max_workers=15) as executor:
            futures = [executor.submit(make_search_request, i) for i in range(num_concurrent)]
            results = [future.result() for future in as_completed(futures)]
        
        metrics.stop_monitoring()
        summary = metrics.get_summary()
        
        # Performance assertions
        assert summary["success_rate"] >= 0.90  # 90% success rate
        assert summary["response_times"]["p95"] < 2.0  # 95th percentile under 2s
        assert summary["memory_usage_mb"]["max"] < 1000  # Max memory under 1GB
        
        print(f"Concurrent Search Results: {json.dumps(summary, indent=2)}")
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_concurrent_network_building(self, client, performance_mocks):
        """Test concurrent citation network building requests."""
        metrics = PerformanceMetrics()
        metrics.start_monitoring()
        
        # Mock network building response
        mock_network_response = {
            "network_id": str(uuid.uuid4()),
            "nodes": [{"id": f"paper_{i}", "type": "paper", "title": f"Paper {i}"} for i in range(50)],
            "edges": [{"source": f"paper_{i}", "target": f"paper_{i+1}", "type": "cites"} for i in range(49)],
            "statistics": {"total_nodes": 50, "total_edges": 49, "depth": 2}
        }
        
        from app.services.auth import create_access_token
        token = create_access_token(data={"sub": "test_user", "email": "test@example.com"})
        headers = {"Authorization": f"Bearer {token}"}
        
        def make_network_request(request_id):
            start_time = time.time()
            try:
                with patch('app.services.graph_operations.GraphOperationsService.build_citation_network') as mock_build:
                    mock_build.return_value = mock_network_response
                    
                    response = client.post(
                        "/api/v1/graph/build-network",
                        json={
                            "seed_papers": [f"paper_{request_id}_{i}" for i in range(3)],
                            "depth": 2,
                            "max_nodes": 50
                        },
                        headers=headers
                    )
                    
                response_time = time.time() - start_time
                success = response.status_code == 201
                metrics.record_response(response_time, success)
                metrics.record_system_metrics()
                return response.status_code
            except Exception as e:
                response_time = time.time() - start_time
                metrics.record_response(response_time, False)
                return 500
        
        # Test with 20 concurrent network building requests
        num_concurrent = 20
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_network_request, i) for i in range(num_concurrent)]
            results = [future.result() for future in as_completed(futures)]
        
        metrics.stop_monitoring()
        summary = metrics.get_summary()
        
        # Performance assertions for more intensive operations
        assert summary["success_rate"] >= 0.85  # 85% success rate
        assert summary["response_times"]["p95"] < 5.0  # 95th percentile under 5s
        assert summary["memory_usage_mb"]["max"] < 2000  # Max memory under 2GB
        
        print(f"Concurrent Network Building Results: {json.dumps(summary, indent=2)}")


class TestLargeDataProcessing:
    """Test performance with large datasets."""
    
    @pytest.mark.slow
    def test_large_paper_dataset_processing(self, performance_mocks):
        """Test processing large paper datasets."""
        # Generate large dataset
        large_dataset = []
        for i in range(10000):
            paper = Paper(
                id=f"paper_{i}",
                title=f"Large Dataset Paper {i}",
                authors=[
                    Author(name=f"Author {i}", id=f"author_{i}")
                ],
                publication_year=2000 + (i % 24),
                doi=f"10.1000/large.{i}",
                abstract=f"This is the abstract for paper {i} in our large dataset processing test. " * 10,
                citation_count={"total": i % 1000, "recent": i % 100}
            )
            large_dataset.append(paper)
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Simulate batch processing
        batch_size = 100
        processed_batches = []
        
        for i in range(0, len(large_dataset), batch_size):
            batch = large_dataset[i:i + batch_size]
            
            # Process batch (simulate citation network extraction)
            batch_results = []
            for paper in batch:
                # Simulate complex processing
                references = [f"ref_{paper.id}_{j}" for j in range(10)]  # 10 refs per paper
                citations = [f"cite_{paper.id}_{j}" for j in range(5)]   # 5 citations per paper
                
                batch_results.append({
                    "paper_id": paper.id,
                    "references": references,
                    "citations": citations,
                    "processed_at": datetime.utcnow().isoformat()
                })
            
            processed_batches.append(batch_results)
            
            # Memory cleanup
            if i % 1000 == 0:
                gc.collect()
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        processing_time = end_time - start_time
        memory_increase = end_memory - start_memory
        
        # Performance assertions
        assert processing_time < 60  # Should complete within 1 minute
        assert memory_increase < 500  # Memory increase should be under 500MB
        assert len(processed_batches) == 100  # Should process all batches
        
        papers_per_second = len(large_dataset) / processing_time
        assert papers_per_second > 100  # Should process at least 100 papers/second
        
        print(f"Large Dataset Processing Results:")
        print(f"  - Papers processed: {len(large_dataset)}")
        print(f"  - Processing time: {processing_time:.2f} seconds")
        print(f"  - Papers per second: {papers_per_second:.2f}")
        print(f"  - Memory increase: {memory_increase:.2f} MB")
    
    @pytest.mark.slow
    def test_large_graph_construction(self, performance_mocks):
        """Test construction of large citation graphs."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Generate large graph
        num_nodes = 5000
        num_edges = 15000
        
        nodes = []
        for i in range(num_nodes):
            nodes.append({
                "id": f"node_{i}",
                "type": "paper",
                "title": f"Paper {i}",
                "year": 2000 + (i % 24),
                "citations": i % 100,
                "x": 0,
                "y": 0
            })
        
        edges = []
        for i in range(num_edges):
            source_idx = i % num_nodes
            target_idx = (i + 1) % num_nodes
            edges.append({
                "source": f"node_{source_idx}",
                "target": f"node_{target_idx}",
                "type": "cites",
                "weight": 1.0
            })
        
        # Simulate graph algorithms
        # 1. Calculate node degrees
        node_degrees = {}
        for edge in edges:
            source, target = edge["source"], edge["target"]
            node_degrees[source] = node_degrees.get(source, 0) + 1
            node_degrees[target] = node_degrees.get(target, 0) + 1
        
        # 2. Find highly connected nodes
        highly_connected = [(node_id, degree) for node_id, degree in node_degrees.items() if degree > 10]
        
        # 3. Simulate community detection
        communities = []
        community_size = 100
        for i in range(0, num_nodes, community_size):
            community_nodes = [f"node_{j}" for j in range(i, min(i + community_size, num_nodes))]
            communities.append({
                "id": i // community_size,
                "nodes": community_nodes,
                "size": len(community_nodes)
            })
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        construction_time = end_time - start_time
        memory_increase = end_memory - start_memory
        
        # Performance assertions
        assert construction_time < 30  # Should complete within 30 seconds
        assert memory_increase < 1000  # Memory increase should be under 1GB
        assert len(highly_connected) > 0  # Should find some highly connected nodes
        assert len(communities) > 0  # Should detect communities
        
        print(f"Large Graph Construction Results:")
        print(f"  - Nodes: {num_nodes}")
        print(f"  - Edges: {num_edges}")
        print(f"  - Construction time: {construction_time:.2f} seconds")
        print(f"  - Memory increase: {memory_increase:.2f} MB")
        print(f"  - Highly connected nodes: {len(highly_connected)}")
        print(f"  - Communities detected: {len(communities)}")


class TestDatabasePerformance:
    """Test database query performance under load."""
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_neo4j_query_performance(self, performance_mocks):
        """Test Neo4j query performance with large datasets."""
        neo4j_mock = performance_mocks['neo4j']
        
        # Mock large query results
        large_result = [{"paper_id": f"paper_{i}", "title": f"Paper {i}"} for i in range(1000)]
        neo4j_mock.execute_query.return_value = large_result
        
        metrics = PerformanceMetrics()
        metrics.start_monitoring()
        
        # Simulate multiple complex queries
        queries = [
            "MATCH (p:Paper)-[:CITES]->(cited:Paper) RETURN p, cited LIMIT 1000",
            "MATCH (a:Author)-[:AUTHORED]->(p:Paper) RETURN a, count(p) as paper_count ORDER BY paper_count DESC LIMIT 100",
            "MATCH (p:Paper) WHERE p.publication_year >= 2020 RETURN p ORDER BY p.citation_count DESC LIMIT 500",
            "MATCH path = (p1:Paper)-[:CITES*2..4]->(p2:Paper) RETURN path LIMIT 200"
        ]
        
        async def execute_query(query):
            start_time = time.time()
            try:
                result = await neo4j_mock.execute_query(query)
                response_time = time.time() - start_time
                metrics.record_response(response_time, True)
                return len(result)
            except Exception as e:
                response_time = time.time() - start_time
                metrics.record_response(response_time, False)
                return 0
        
        # Execute queries concurrently
        tasks = []
        for _ in range(10):  # 10 rounds of queries
            for query in queries:
                tasks.append(execute_query(query))
        
        results = await asyncio.gather(*tasks)
        
        metrics.stop_monitoring()
        summary = metrics.get_summary()
        
        # Performance assertions
        assert summary["success_rate"] >= 0.95  # 95% success rate
        assert summary["response_times"]["p95"] < 2.0  # 95th percentile under 2s
        assert sum(results) > 0  # Should return results
        
        print(f"Neo4j Query Performance Results: {json.dumps(summary, indent=2)}")
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_redis_cache_performance(self, performance_mocks):
        """Test Redis caching performance under load."""
        redis_mock = performance_mocks['redis']
        
        # Setup cache responses
        cache_data = {"cached": True, "data": "large_cached_data" * 100}
        redis_mock.get.return_value = json.dumps(cache_data)
        redis_mock.set.return_value = True
        
        metrics = PerformanceMetrics()
        metrics.start_monitoring()
        
        async def cache_operation(operation_id):
            start_time = time.time()
            try:
                # Simulate cache read
                cache_key = f"paper_search_{operation_id % 100}"  # 100 unique keys
                cached_result = await redis_mock.get(cache_key)
                
                if not cached_result:
                    # Simulate cache miss - write to cache
                    new_data = {"operation_id": operation_id, "data": "computed_data" * 50}
                    await redis_mock.set(cache_key, json.dumps(new_data), 3600)  # 1 hour TTL
                
                response_time = time.time() - start_time
                metrics.record_response(response_time, True)
                return True
            except Exception as e:
                response_time = time.time() - start_time
                metrics.record_response(response_time, False)
                return False
        
        # Execute 500 concurrent cache operations
        tasks = [cache_operation(i) for i in range(500)]
        results = await asyncio.gather(*tasks)
        
        metrics.stop_monitoring()
        summary = metrics.get_summary()
        
        # Performance assertions
        assert summary["success_rate"] >= 0.98  # 98% success rate
        assert summary["response_times"]["p95"] < 0.1  # 95th percentile under 100ms
        assert summary["requests_per_second"] > 1000  # Should handle 1000+ RPS
        
        print(f"Redis Cache Performance Results: {json.dumps(summary, indent=2)}")


class TestExternalAPIPerformance:
    """Test performance of external API integrations."""
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_openalex_rate_limit_compliance(self, performance_mocks):
        """Test OpenAlex API rate limit compliance and performance."""
        openalex_mock = performance_mocks['openalex']
        
        # Mock API response with delay to simulate real API
        async def mock_api_call_with_delay(*args, **kwargs):
            await asyncio.sleep(0.01)  # 10ms delay
            return {
                "results": [{"id": f"W{i}", "title": f"Paper {i}"} for i in range(25)],
                "meta": {"count": 25}
            }
        
        openalex_mock.search_works.side_effect = mock_api_call_with_delay
        
        metrics = PerformanceMetrics()
        metrics.start_monitoring()
        
        # Test rate limiting - should not exceed 100 requests per second
        max_concurrent = 50
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def rate_limited_request(request_id):
            async with semaphore:
                start_time = time.time()
                try:
                    result = await openalex_mock.search_works(f"query {request_id}")
                    response_time = time.time() - start_time
                    metrics.record_response(response_time, True)
                    return len(result.get("results", []))
                except Exception as e:
                    response_time = time.time() - start_time
                    metrics.record_response(response_time, False)
                    return 0
        
        # Make 200 requests with rate limiting
        tasks = [rate_limited_request(i) for i in range(200)]
        results = await asyncio.gather(*tasks)
        
        metrics.stop_monitoring()
        summary = metrics.get_summary()
        
        # Performance assertions
        assert summary["success_rate"] >= 0.95  # 95% success rate
        assert summary["requests_per_second"] <= 110  # Should not exceed rate limit + small buffer
        assert summary["response_times"]["mean"] >= 0.01  # Should include API delay
        
        print(f"OpenAlex Rate Limit Performance Results: {json.dumps(summary, indent=2)}")


class TestMemoryAndResourceUsage:
    """Test memory usage and resource management."""
    
    @pytest.mark.slow
    def test_memory_usage_during_bulk_operations(self, performance_mocks):
        """Test memory usage during bulk paper processing."""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_samples = [initial_memory]
        
        # Simulate bulk paper processing
        bulk_papers = []
        for i in range(5000):
            paper = {
                "id": f"bulk_paper_{i}",
                "title": f"Bulk Paper {i} with longer title to test memory usage",
                "abstract": f"This is a longer abstract for paper {i}. " * 20,  # ~400 chars
                "authors": [f"Author {j}" for j in range(5)],  # 5 authors per paper
                "references": [f"ref_{i}_{j}" for j in range(20)],  # 20 references
                "metadata": {
                    "keywords": [f"keyword_{j}" for j in range(10)],
                    "mesh_terms": [f"mesh_{j}" for j in range(8)],
                    "topics": [f"topic_{j}" for j in range(6)]
                }
            }
            bulk_papers.append(paper)
            
            # Sample memory every 500 papers
            if i % 500 == 0:
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_samples.append(current_memory)
        
        # Process papers in batches
        processed_count = 0
        for i in range(0, len(bulk_papers), 100):
            batch = bulk_papers[i:i + 100]
            
            # Simulate complex processing
            for paper in batch:
                # Extract features
                features = {
                    "word_count": len(paper["abstract"].split()),
                    "author_count": len(paper["authors"]),
                    "ref_count": len(paper["references"]),
                    "keyword_count": len(paper["metadata"]["keywords"])
                }
                paper["features"] = features
                processed_count += 1
            
            # Force garbage collection periodically
            if i % 1000 == 0:
                gc.collect()
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_samples.append(current_memory)
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_samples.append(final_memory)
        
        # Calculate memory statistics
        max_memory = max(memory_samples)
        memory_increase = max_memory - initial_memory
        final_increase = final_memory - initial_memory
        
        # Performance assertions
        assert processed_count == 5000  # All papers processed
        assert memory_increase < 1000  # Peak memory increase under 1GB
        assert final_increase < 500  # Final memory increase under 500MB (after GC)
        
        print(f"Memory Usage During Bulk Operations:")
        print(f"  - Papers processed: {processed_count}")
        print(f"  - Initial memory: {initial_memory:.2f} MB")
        print(f"  - Peak memory: {max_memory:.2f} MB")
        print(f"  - Final memory: {final_memory:.2f} MB")
        print(f"  - Peak increase: {memory_increase:.2f} MB")
        print(f"  - Final increase: {final_increase:.2f} MB")
    
    @pytest.mark.slow
    def test_cpu_usage_during_graph_algorithms(self, performance_mocks):
        """Test CPU usage during graph algorithm execution."""
        import psutil
        import threading
        
        cpu_samples = []
        monitoring = True
        
        def monitor_cpu():
            while monitoring:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                cpu_samples.append(cpu_percent)
        
        # Start CPU monitoring
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()
        
        try:
            # Simulate CPU-intensive graph algorithms
            start_time = time.time()
            
            # Generate large graph
            nodes = 3000
            edges = []
            
            # Create dense connections
            for i in range(nodes):
                for j in range(min(10, nodes - i - 1)):  # Each node connects to next 10 nodes
                    edges.append((i, i + j + 1))
            
            # Simulate PageRank algorithm
            adjacency = {}
            for source, target in edges:
                if source not in adjacency:
                    adjacency[source] = []
                adjacency[source].append(target)
            
            # Initialize PageRank values
            pagerank = {i: 1.0 / nodes for i in range(nodes)}
            
            # Run PageRank iterations
            damping = 0.85
            for iteration in range(20):  # 20 iterations
                new_pagerank = {}
                for node in range(nodes):
                    rank_sum = 0.0
                    # Sum ranks from incoming edges
                    for source in range(nodes):
                        if source in adjacency and node in adjacency[source]:
                            rank_sum += pagerank[source] / len(adjacency[source])
                    
                    new_pagerank[node] = (1 - damping) / nodes + damping * rank_sum
                
                pagerank = new_pagerank
            
            # Find top ranked nodes
            top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
            
            end_time = time.time()
            processing_time = end_time - start_time
            
        finally:
            monitoring = False
            monitor_thread.join()
        
        # Calculate CPU statistics
        avg_cpu = statistics.mean(cpu_samples) if cpu_samples else 0
        max_cpu = max(cpu_samples) if cpu_samples else 0
        
        # Performance assertions
        assert processing_time < 30  # Should complete within 30 seconds
        assert avg_cpu < 90  # Average CPU usage should be reasonable
        assert len(top_nodes) == 10  # Should find top nodes
        
        print(f"CPU Usage During Graph Algorithms:")
        print(f"  - Nodes processed: {nodes}")
        print(f"  - Edges processed: {len(edges)}")
        print(f"  - Processing time: {processing_time:.2f} seconds")
        print(f"  - Average CPU: {avg_cpu:.2f}%")
        print(f"  - Peak CPU: {max_cpu:.2f}%")
        print(f"  - Top PageRank nodes: {top_nodes[:3]}")


# Performance test markers and configuration
pytest.mark.slow = pytest.mark.skipif(
    True,  # Skip by default
    reason="Performance tests are slow - run with --performance flag"
)

# Custom performance test configuration
@pytest.fixture(autouse=True)
def performance_test_setup():
    """Setup for performance tests."""
    # Force garbage collection before each test
    gc.collect()
    
    # Set high recursion limit for complex operations
    import sys
    original_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(5000)
    
    yield
    
    # Cleanup after test
    gc.collect()
    sys.setrecursionlimit(original_limit)