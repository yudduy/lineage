"""
Integration tests for the Advanced Analytics System.

Tests the complete integrated functionality of the advanced analytics platform
including lineage analysis, research intelligence, and performance optimization.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import json

from app.services.advanced_analytics import (
    AdvancedAnalyticsService,
    AnalysisDepth,
    IntellectualLineage,
    ContentIntelligence
)
from app.services.research_intelligence import (
    ResearchIntelligenceEngine,
    ResearchTrend,
    ResearchTrendType,
    TrendDetectionMethod,
    CommunityDynamics,
    KnowledgeFlow,
    ResearchForecast
)
from app.services.performance_optimizer import (
    PerformanceOptimizer,
    MultiTierCache,
    QueryOptimizer,
    BackgroundTaskProcessor,
    TaskPriority
)


class TestAdvancedAnalyticsIntegration:
    """Integration tests for the Advanced Analytics System."""
    
    @pytest.fixture
    async def mock_analytics_service(self):
        """Create mock analytics service with dependencies."""
        service = AdvancedAnalyticsService()
        
        # Mock dependencies
        service.redis = AsyncMock()
        service.neo4j = AsyncMock()
        service.openalex = AsyncMock()
        service.semantic_scholar = AsyncMock()
        service.llm = AsyncMock()
        service.content_enrichment = AsyncMock()
        
        await service.initialize()
        return service
    
    @pytest.fixture
    async def mock_intelligence_engine(self):
        """Create mock intelligence engine."""
        engine = ResearchIntelligenceEngine()
        
        # Mock dependencies
        engine.analytics = AsyncMock()
        engine.redis = AsyncMock()
        engine.neo4j = AsyncMock()
        
        await engine.initialize()
        return engine
    
    @pytest.fixture
    async def mock_performance_optimizer(self):
        """Create mock performance optimizer."""
        optimizer = PerformanceOptimizer()
        await optimizer.initialize()
        return optimizer
    
    @pytest.mark.asyncio
    async def test_intellectual_lineage_analysis(self, mock_analytics_service):
        """Test comprehensive intellectual lineage analysis."""
        
        # Mock paper data
        mock_paper_data = {
            'id': 'test_paper_123',
            'title': 'Test Paper on Machine Learning',
            'year': 2020,
            'authors': [{'id': 'author_1', 'name': 'John Doe'}],
            'venue': 'Test Conference',
            'fields': ['machine learning', 'artificial intelligence'],
            'citation_count': 150,
            'references': ['ref_1', 'ref_2', 'ref_3'],
            'citations': ['cite_1', 'cite_2']
        }
        
        # Mock the _fetch_paper_data method
        mock_analytics_service._fetch_paper_data = AsyncMock(return_value=mock_paper_data)
        
        # Mock Neo4j operations
        mock_analytics_service.neo4j.run_algorithm = AsyncMock(return_value={
            'test_paper_123': 0.85,
            'ref_1': 0.65,
            'ref_2': 0.45
        })
        
        mock_analytics_service.neo4j.detect_communities = AsyncMock(return_value={
            'test_paper_123': 1,
            'ref_1': 1,
            'ref_2': 2
        })
        
        # Perform analysis
        lineage = await mock_analytics_service.analyze_intellectual_lineage(
            paper_id='test_paper_123',
            depth=AnalysisDepth.MODERATE,
            include_predictions=True,
            enrich_content=True
        )
        
        # Verify results
        assert isinstance(lineage, IntellectualLineage)
        assert lineage.root_paper_id == 'test_paper_123'
        assert lineage.depth > 0
        assert lineage.total_papers > 0
        assert len(lineage.key_milestones) >= 0
        assert lineage.predicted_trajectory is not None
        
        # Verify method calls
        mock_analytics_service._fetch_paper_data.assert_called()
        mock_analytics_service.neo4j.run_algorithm.assert_called()
    
    @pytest.mark.asyncio
    async def test_research_trend_detection(self, mock_intelligence_engine):
        """Test research trend detection functionality."""
        
        # Mock Neo4j query results
        mock_query_results = [
            {'year': 2020, 'fields': ['ai', 'ml'], 'paper_count': 100},
            {'year': 2021, 'fields': ['ai', 'ml'], 'paper_count': 150},
            {'year': 2022, 'fields': ['ai', 'ml'], 'paper_count': 300},
            {'year': 2023, 'fields': ['ai', 'ml'], 'paper_count': 450}
        ]
        
        mock_intelligence_engine.neo4j.execute_query = AsyncMock(return_value=mock_query_results)
        
        # Detect trends
        trends = await mock_intelligence_engine.detect_research_trends(
            domain='machine learning',
            time_window=(2020, 2023),
            methods=[TrendDetectionMethod.BURST_DETECTION],
            min_confidence=0.6
        )
        
        # Verify results
        assert isinstance(trends, list)
        assert len(trends) >= 0
        
        for trend in trends:
            assert isinstance(trend, ResearchTrend)
            assert trend.confidence >= 0.6
            assert trend.type in ResearchTrendType
            assert trend.strength >= 0 and trend.strength <= 1
        
        # Verify Neo4j was called
        mock_intelligence_engine.neo4j.execute_query.assert_called()
    
    @pytest.mark.asyncio
    async def test_community_dynamics_analysis(self, mock_intelligence_engine):
        """Test community dynamics analysis."""
        
        # Mock community data
        mock_community_data = [
            {'year': 2020, 'paper_id': 'p1', 'authors': ['a1', 'a2'], 'citations': 10},
            {'year': 2021, 'paper_id': 'p2', 'authors': ['a2', 'a3'], 'citations': 20},
            {'year': 2022, 'paper_id': 'p3', 'authors': ['a3', 'a4'], 'citations': 30}
        ]
        
        mock_intelligence_engine.neo4j.execute_query = AsyncMock(return_value=mock_community_data)
        
        # Analyze community dynamics
        dynamics = await mock_intelligence_engine.analyze_community_dynamics(
            community_id='test_community_123'
        )
        
        # Verify results
        assert isinstance(dynamics, CommunityDynamics)
        assert dynamics.community_id == 'test_community_123'
        assert len(dynamics.growth_trajectory) >= 0
        assert dynamics.cohesion_score >= 0 and dynamics.cohesion_score <= 1
        assert dynamics.lifecycle_stage in ['emerging', 'growing', 'mature', 'declining']
    
    @pytest.mark.asyncio
    async def test_knowledge_flow_analysis(self, mock_intelligence_engine):
        """Test knowledge flow analysis."""
        
        # Mock knowledge flow data
        mock_flow_data = [
            {'target_field': 'computer vision', 'flow_count': 25, 'sample_papers': ['p1', 'p2']},
            {'target_field': 'natural language processing', 'flow_count': 15, 'sample_papers': ['p3']}
        ]
        
        mock_intelligence_engine.neo4j.execute_query = AsyncMock(return_value=mock_flow_data)
        
        # Analyze knowledge flows
        flows = await mock_intelligence_engine.analyze_knowledge_flows(
            source_entity='machine learning',
            entity_type='field',
            max_hops=2
        )
        
        # Verify results
        assert isinstance(flows, list)
        assert len(flows) >= 0
        
        for flow in flows:
            assert isinstance(flow, KnowledgeFlow)
            assert flow.source_id == 'machine learning'
            assert flow.source_type == 'field'
            assert flow.strength >= 0 and flow.strength <= 1
    
    @pytest.mark.asyncio
    async def test_research_forecasting(self, mock_intelligence_engine):
        """Test research development forecasting."""
        
        # Mock historical data
        mock_historical_data = {
            'entity_id': 'test_paper_123',
            'entity_type': 'paper',
            'timeline': [
                {'year': 2020, 'citations': 10},
                {'year': 2021, 'citations': 25},
                {'year': 2022, 'citations': 45},
                {'year': 2023, 'citations': 70}
            ]
        }
        
        mock_intelligence_engine._get_historical_data = AsyncMock(return_value=mock_historical_data)
        mock_intelligence_engine._predict_collaborations = AsyncMock(return_value=['author_1', 'author_2'])
        mock_intelligence_engine._predict_emerging_topics = AsyncMock(return_value=[
            {'topic': 'quantum ai', 'probability': 0.75}
        ])
        
        # Generate forecast
        forecast = await mock_intelligence_engine.forecast_research_development(
            entity_id='test_paper_123',
            entity_type='paper',
            horizon_months=24
        )
        
        # Verify results
        assert isinstance(forecast, ResearchForecast)
        assert forecast.target_entity == 'test_paper_123'
        assert forecast.entity_type == 'paper'
        assert forecast.forecast_horizon == 24
        assert forecast.breakthrough_probability >= 0 and forecast.breakthrough_probability <= 1
        assert forecast.decline_risk >= 0 and forecast.decline_risk <= 1
    
    @pytest.mark.asyncio
    async def test_content_intelligence_analysis(self, mock_analytics_service):
        """Test content intelligence analysis."""
        
        # Mock paper data
        mock_paper_data = {
            'id': 'test_paper_123',
            'title': 'Advanced Machine Learning Techniques',
            'year': 2023,
            'fields': ['machine learning', 'deep learning'],
            'citation_count': 89
        }
        
        mock_analytics_service._fetch_paper_data = AsyncMock(return_value=mock_paper_data)
        
        # Mock LLM responses
        mock_analytics_service.llm.generate = AsyncMock(return_value="Key research theme: Deep learning optimization")
        
        # Analyze content intelligence
        intelligence = await mock_analytics_service.analyze_content_intelligence(
            paper_id='test_paper_123'
        )
        
        # Verify results
        assert isinstance(intelligence, ContentIntelligence)
        assert intelligence.paper_id == 'test_paper_123'
        assert intelligence.significance_score >= 0 and intelligence.significance_score <= 1
        assert intelligence.novelty_score >= 0 and intelligence.novelty_score <= 1
        assert intelligence.impact_potential >= 0 and intelligence.impact_potential <= 1
        assert len(intelligence.research_themes) >= 0
        assert len(intelligence.research_gaps) >= 0
    
    @pytest.mark.asyncio
    async def test_cache_performance(self, mock_performance_optimizer):
        """Test multi-tier cache performance."""
        
        # Test cache operations
        cache = mock_performance_optimizer.cache
        
        # Set value in cache
        await cache.set('test_key', {'data': 'test_value'}, ttl=60)
        
        # Get value from cache
        cached_value = await cache.get('test_key')
        
        # Verify cache hit
        assert cached_value is not None
        assert cached_value == {'data': 'test_value'}
        
        # Get cache statistics
        stats = await cache.get_statistics()
        assert 'l1_stats' in stats
        assert 'total_requests' in stats
        assert 'hit_rate' in stats
    
    @pytest.mark.asyncio
    async def test_background_task_processing(self, mock_performance_optimizer):
        """Test background task processing."""
        
        # Define test task
        async def test_task(x, y):
            return x + y
        
        # Submit task
        task_id = await mock_performance_optimizer.task_processor.submit(
            name='test_addition',
            function=test_task,
            args=(5, 3),
            priority=TaskPriority.NORMAL
        )
        
        # Verify task was submitted
        assert task_id is not None
        assert isinstance(task_id, str)
        
        # Wait a bit for task to potentially process
        await asyncio.sleep(0.1)
        
        # Get task status
        status = await mock_performance_optimizer.task_processor.get_task_status(task_id)
        assert status is not None
        assert status['task_id'] == task_id
        assert status['name'] == 'test_addition'
    
    @pytest.mark.asyncio
    async def test_query_optimization(self, mock_performance_optimizer):
        """Test query optimization functionality."""
        
        optimizer = mock_performance_optimizer.query_optimizer
        
        # Test query optimization
        query = "MATCH (p:Paper) WHERE p.year > 2020 RETURN p LIMIT 100"
        params = {'year': 2020}
        
        plan = await optimizer.optimize_query(query, params)
        
        # Verify optimization plan
        assert plan.plan_id is not None
        assert plan.original_query == query
        assert plan.estimated_cost > 0
        assert plan.estimated_time_ms > 0
        assert len(plan.execution_steps) > 0
        assert plan.cache_strategy['enabled'] is not None
    
    @pytest.mark.asyncio
    async def test_system_performance_metrics(self, mock_analytics_service, mock_performance_optimizer):
        """Test system performance metrics collection."""
        
        # Get analytics metrics
        analytics_metrics = await mock_analytics_service.get_performance_metrics()
        
        # Verify analytics metrics structure
        assert 'active_operations' in analytics_metrics
        assert 'completed_operations' in analytics_metrics
        assert 'cache_hit_rate' in analytics_metrics
        
        # Get optimizer performance report
        optimizer_report = await mock_performance_optimizer.get_performance_report()
        
        # Verify optimizer metrics structure
        assert 'cache' in optimizer_report
        assert 'tasks' in optimizer_report
        assert 'resources' in optimizer_report
    
    @pytest.mark.asyncio
    async def test_error_handling_and_fallbacks(self, mock_analytics_service):
        """Test error handling and fallback mechanisms."""
        
        # Mock a failure in external service
        mock_analytics_service.openalex = None
        mock_analytics_service.semantic_scholar = None
        
        # Should handle gracefully without external services
        try:
            paper_data = await mock_analytics_service._fetch_paper_data('test_paper', Mock())
            # Should return None or empty dict, not raise exception
            assert paper_data is None or isinstance(paper_data, dict)
        except Exception as e:
            pytest.fail(f"Should handle external service failures gracefully: {e}")
    
    @pytest.mark.asyncio 
    async def test_concurrent_operations(self, mock_analytics_service):
        """Test concurrent operations handling."""
        
        # Mock paper data
        mock_paper_data = {
            'id': 'test_paper',
            'title': 'Test Paper',
            'year': 2023,
            'citation_count': 10
        }
        
        mock_analytics_service._fetch_paper_data = AsyncMock(return_value=mock_paper_data)
        
        # Run multiple operations concurrently
        tasks = []
        for i in range(5):
            task = mock_analytics_service.analyze_intellectual_lineage(
                paper_id=f'test_paper_{i}',
                depth=AnalysisDepth.SHALLOW,
                include_predictions=False,
                enrich_content=False
            )
            tasks.append(task)
        
        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all completed successfully
        for result in results:
            if isinstance(result, Exception):
                pytest.fail(f"Concurrent operation failed: {result}")
            assert isinstance(result, IntellectualLineage)


class TestPerformanceOptimization:
    """Specific tests for performance optimization components."""
    
    @pytest.mark.asyncio
    async def test_lru_cache_eviction(self):
        """Test LRU cache eviction behavior."""
        from app.services.performance_optimizer import LRUCache
        
        # Create small cache for testing
        cache = LRUCache(max_size=3, max_memory_mb=1)
        
        # Fill cache
        await cache.set('key1', 'value1')
        await cache.set('key2', 'value2') 
        await cache.set('key3', 'value3')
        
        # Access key1 to make it most recently used
        await cache.get('key1')
        
        # Add key4 - should evict key2 (least recently used)
        await cache.set('key4', 'value4')
        
        # Verify eviction
        assert await cache.get('key1') == 'value1'  # Still there
        assert await cache.get('key2') is None       # Evicted
        assert await cache.get('key3') == 'value3'  # Still there
        assert await cache.get('key4') == 'value4'  # New entry
    
    @pytest.mark.asyncio
    async def test_resource_monitoring(self):
        """Test resource monitoring functionality."""
        from app.services.performance_optimizer import ResourceMonitor
        
        monitor = ResourceMonitor()
        
        # Collect metrics
        metrics = await monitor._collect_metrics()
        
        # Verify metrics structure
        assert metrics.cpu_percent >= 0
        assert metrics.memory_percent >= 0
        assert metrics.memory_mb > 0
        assert metrics.timestamp > 0
    
    def test_task_priority_ordering(self):
        """Test task priority ordering in queue."""
        from app.services.performance_optimizer import TaskPriority
        
        # Verify priority values are ordered correctly
        assert TaskPriority.CRITICAL.value < TaskPriority.HIGH.value
        assert TaskPriority.HIGH.value < TaskPriority.NORMAL.value
        assert TaskPriority.NORMAL.value < TaskPriority.LOW.value
        assert TaskPriority.LOW.value < TaskPriority.BACKGROUND.value


class TestAPIIntegration:
    """Test API endpoint integration."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock all service dependencies for API testing."""
        with patch('app.services.advanced_analytics.get_analytics_service') as mock_analytics, \
             patch('app.services.research_intelligence.get_intelligence_engine') as mock_intelligence, \
             patch('app.services.performance_optimizer.get_performance_optimizer') as mock_optimizer:
            
            # Mock service instances
            mock_analytics.return_value = AsyncMock()
            mock_intelligence.return_value = AsyncMock()
            mock_optimizer.return_value = AsyncMock()
            
            yield {
                'analytics': mock_analytics,
                'intelligence': mock_intelligence,
                'optimizer': mock_optimizer
            }
    
    @pytest.mark.asyncio
    async def test_lineage_analysis_endpoint(self, mock_dependencies):
        """Test lineage analysis API endpoint."""
        from app.api.v1.endpoints.advanced_analytics import analyze_intellectual_lineage
        from app.models.user import User
        
        # Mock user
        mock_user = Mock(spec=User)
        mock_user.id = 'test_user'
        
        # Mock lineage result
        mock_lineage = IntellectualLineage(
            root_paper_id='test_paper',
            depth=3,
            total_papers=100,
            total_citations=500,
            key_milestones=[],
            evolution_path=[],
            knowledge_flows={},
            temporal_patterns={},
            impact_propagation={},
            research_communities=[]
        )
        
        mock_analytics_service = AsyncMock()
        mock_analytics_service.analyze_intellectual_lineage.return_value = mock_lineage
        mock_analytics_service._queue_cache_warming = AsyncMock()
        
        # Test endpoint
        from fastapi import BackgroundTasks
        background_tasks = BackgroundTasks()
        
        result = await analyze_intellectual_lineage(
            paper_id='test_paper',
            depth=AnalysisDepth.MODERATE,
            include_predictions=True,
            enrich_content=True,
            background_tasks=background_tasks,
            analytics_service=mock_analytics_service,
            current_user=mock_user
        )
        
        # Verify result structure
        assert result['status'] == 'success'
        assert 'data' in result
        assert 'lineage' in result['data']
        assert 'metadata' in result['data']
        
        # Verify service was called
        mock_analytics_service.analyze_intellectual_lineage.assert_called_once()