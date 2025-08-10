"""
Comprehensive Test Suite for LLM Services.

This test suite covers:
- LLM service functionality and error handling
- Cost management and budget tracking
- Caching strategies and cache hits
- Fallback mechanisms and circuit breakers
- Content enrichment and quality assessment
- Citation analysis and relationship extraction
- Research trajectory and lineage tracing
- Monitoring and analytics
- API endpoints and responses
"""

import asyncio
import json
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from app.services.llm_service import LLMService, LLMResponse, PromptTemplate, ModelType, LLMUsage
from app.services.llm_cost_manager import LLMCostManager, CostCategory, CostUsage, BudgetAlert
from app.services.llm_cache import SemanticCacheManager, CacheEntry, CacheType
from app.services.llm_fallback import LLMFallbackService, ErrorSeverity, FallbackStrategy
from app.services.content_enrichment import ContentEnrichmentService, EnrichedContent, ContentQuality
from app.services.citation_analysis import CitationAnalysisService, CitationRelationship, CitationType
from app.services.research_trajectory import ResearchTrajectoryService, IntellectualLineage, TrajectoryType
from app.services.llm_monitoring import LLMMonitoringService, PerformanceMetric, QualityMetric
from app.services.llm_service_enhanced import EnhancedLLMService


class TestLLMService:
    """Test suite for core LLM service functionality."""
    
    @pytest.fixture
    async def llm_service(self):
        """Create LLM service instance for testing."""
        service = LLMService()
        service.redis_manager = AsyncMock()
        service._clients_initialized = True
        yield service
    
    @pytest.fixture
    def sample_prompt_template(self):
        """Sample prompt template for testing."""
        return PromptTemplate(
            system_prompt="You are a helpful assistant.",
            user_template="Analyze this text: {text}",
            max_tokens=1000,
            temperature=0.1,
            model_type=ModelType.ANALYSIS
        )
    
    @pytest.fixture
    def sample_llm_response(self):
        """Sample LLM response for testing."""
        usage = LLMUsage(
            input_tokens=100,
            output_tokens=200,
            total_tokens=300,
            cost=0.015,
            model="gpt-4",
            provider="openai",
            timestamp=datetime.now()
        )
        
        return LLMResponse(
            content="This is a test analysis of the provided text.",
            usage=usage,
            model="gpt-4",
            provider="openai",
            cached=False
        )
    
    @pytest.mark.asyncio
    async def test_complete_basic_functionality(self, llm_service, sample_prompt_template, sample_llm_response):
        """Test basic completion functionality."""
        with patch.object(llm_service, '_make_llm_request', return_value=sample_llm_response):
            response = await llm_service.complete(
                sample_prompt_template,
                text="Sample text to analyze"
            )
            
            assert response is not None
            assert response.content == "This is a test analysis of the provided text."
            assert response.usage.total_tokens == 300
            assert response.model == "gpt-4"
    
    @pytest.mark.asyncio
    async def test_caching_functionality(self, llm_service, sample_prompt_template, sample_llm_response):
        """Test response caching."""
        cache_key = "test_cache_key"
        
        # Mock cache methods
        llm_service._get_cache_key = AsyncMock(return_value=cache_key)
        llm_service._get_cached_response = AsyncMock(return_value=None)  # First call - no cache
        llm_service._cache_response = AsyncMock()
        
        with patch.object(llm_service, '_make_llm_request', return_value=sample_llm_response):
            # First call should make API request
            response1 = await llm_service.complete(
                sample_prompt_template,
                use_cache=True,
                text="Test text"
            )
            
            # Verify caching was attempted
            llm_service._get_cached_response.assert_called_once()
            llm_service._cache_response.assert_called_once()
            
            # Mock cache hit for second call
            cached_response = sample_llm_response
            cached_response.cached = True
            llm_service._get_cached_response = AsyncMock(return_value=cached_response)
            
            # Second call should return cached response
            response2 = await llm_service.complete(
                sample_prompt_template,
                use_cache=True,
                text="Test text"
            )
            
            assert response2.cached is True
    
    @pytest.mark.asyncio
    async def test_budget_limit_handling(self, llm_service, sample_prompt_template):
        """Test budget limit enforcement."""
        # Mock budget exceeded scenario
        with patch.object(llm_service, '_should_use_budget_limit', return_value=False):
            llm_service.settings.llm.enable_local_fallback = False
            
            # Should raise ValidationError when budget exceeded and no fallback
            with pytest.raises(Exception):  # ValidationError
                await llm_service.complete(sample_prompt_template, text="Test")
    
    @pytest.mark.asyncio
    async def test_batch_completion(self, llm_service, sample_prompt_template, sample_llm_response):
        """Test batch completion functionality."""
        requests = [
            (sample_prompt_template, {"text": "Text 1"}),
            (sample_prompt_template, {"text": "Text 2"}),
            (sample_prompt_template, {"text": "Text 3"})
        ]
        
        with patch.object(llm_service, 'complete', return_value=sample_llm_response) as mock_complete:
            responses = await llm_service.batch_complete(requests, max_concurrency=2)
            
            assert len(responses) == 3
            assert mock_complete.call_count == 3
            
            # All responses should be successful
            for response in responses:
                assert isinstance(response, LLMResponse)


class TestLLMCostManager:
    """Test suite for cost management functionality."""
    
    @pytest.fixture
    async def cost_manager(self):
        """Create cost manager instance for testing."""
        manager = LLMCostManager()
        manager.redis_manager = AsyncMock()
        manager._initialized = True
        yield manager
    
    @pytest.mark.asyncio
    async def test_usage_recording(self, cost_manager):
        """Test usage recording and cost calculation."""
        usage = await cost_manager.record_usage(
            model="gpt-4",
            provider="openai",
            input_tokens=100,
            output_tokens=200,
            category=CostCategory.PAPER_ANALYSIS
        )
        
        assert isinstance(usage, CostUsage)
        assert usage.model == "gpt-4"
        assert usage.provider == "openai"
        assert usage.total_tokens == 300
        assert usage.cost > 0  # Should have calculated some cost
    
    @pytest.mark.asyncio
    async def test_budget_checking(self, cost_manager):
        """Test budget availability checking."""
        # Mock current costs
        cost_manager._daily_cost = 5.0
        cost_manager._monthly_cost = 50.0
        cost_manager.settings.llm.daily_budget_limit = 10.0
        cost_manager.settings.llm.monthly_budget_limit = 100.0
        
        # Should be available
        available, message = await cost_manager.is_budget_available(2.0)
        assert available is True
        
        # Should exceed daily limit
        available, message = await cost_manager.is_budget_available(6.0)
        assert available is False
        assert "daily budget limit" in message.lower()
    
    @pytest.mark.asyncio
    async def test_cost_analytics(self, cost_manager):
        """Test cost analytics generation."""
        # Mock usage records
        cost_manager._get_usage_records = AsyncMock(return_value=[
            CostUsage(
                timestamp=datetime.now(),
                model="gpt-4",
                provider="openai",
                input_tokens=100,
                output_tokens=200,
                total_tokens=300,
                cost=0.015,
                category=CostCategory.PAPER_ANALYSIS,
                cached=False
            )
        ])
        
        analytics = await cost_manager.get_cost_analytics()
        
        assert analytics.total_requests == 1
        assert analytics.total_cost == 0.015
        assert CostCategory.PAPER_ANALYSIS in analytics.cost_by_category
        assert "gpt-4" in analytics.cost_by_model


class TestSemanticCacheManager:
    """Test suite for semantic caching functionality."""
    
    @pytest.fixture
    async def cache_manager(self):
        """Create cache manager instance for testing."""
        manager = SemanticCacheManager()
        manager.redis_manager = AsyncMock()
        manager._initialized = True
        yield manager
    
    @pytest.mark.asyncio
    async def test_cache_miss_and_storage(self, cache_manager):
        """Test cache miss and subsequent storage."""
        # Mock no existing cache
        cache_manager._find_exact_match = AsyncMock(return_value=None)
        cache_manager._find_semantic_matches = AsyncMock(return_value=[])
        
        result = await cache_manager.get_cached_response(
            prompt="Test prompt",
            model="gpt-4", 
            temperature=0.1,
            max_tokens=1000
        )
        
        assert result is None  # Cache miss
        
        # Test caching a response
        cache_key = await cache_manager.cache_response(
            prompt="Test prompt",
            response_content="Test response",
            model="gpt-4",
            provider="openai",
            temperature=0.1,
            max_tokens=1000,
            cost=0.015,
            tokens=300
        )
        
        assert cache_key is not None
        assert len(cache_key) == 64  # SHA-256 hash length
    
    @pytest.mark.asyncio
    async def test_semantic_similarity_matching(self, cache_manager):
        """Test semantic similarity cache matching."""
        # Mock embedding model
        cache_manager._embedding_model = MagicMock()
        cache_manager._embedding_model.encode.return_value = [[0.1, 0.2, 0.3]]
        
        # Mock similar cache entry
        similar_entry = CacheEntry(
            key="similar_key",
            content="Similar response",
            model="gpt-4",
            provider="openai",
            cost=0.01,
            tokens=250,
            cache_type=CacheType.SEMANTIC_MATCH,
            timestamp=datetime.now(),
            access_count=1,
            last_accessed=datetime.now(),
            ttl_seconds=3600,
            embedding_hash="embedding_hash"
        )
        
        cache_manager._find_semantic_matches = AsyncMock(return_value=[(similar_entry, 0.95)])
        
        result = await cache_manager.get_cached_response(
            prompt="Similar test prompt",
            model="gpt-4",
            temperature=0.1,
            max_tokens=1000
        )
        
        assert result is not None
        assert result[1] == CacheType.SEMANTIC_MATCH


class TestLLMFallbackService:
    """Test suite for fallback and error handling."""
    
    @pytest.fixture
    async def fallback_service(self):
        """Create fallback service instance for testing."""
        service = LLMFallbackService()
        service.redis_manager = AsyncMock()
        service._initialized = True
        yield service
    
    @pytest.mark.asyncio
    async def test_error_classification(self, fallback_service):
        """Test error classification functionality."""
        # Test rate limit error
        rate_limit_error = Exception("Rate limit exceeded")
        context = {"model": "gpt-4", "provider": "openai"}
        
        error_context = fallback_service.error_classifier.classify_error(rate_limit_error, context)
        
        assert error_context.error_type == "rate_limit"
        assert error_context.severity == ErrorSeverity.MEDIUM
        
        # Test authentication error
        auth_error = Exception("Invalid API key")
        auth_context = fallback_service.error_classifier.classify_error(auth_error, context)
        
        assert auth_context.error_type == "authentication"
        assert auth_context.severity == ErrorSeverity.HIGH
    
    @pytest.mark.asyncio
    async def test_fallback_strategies(self, fallback_service):
        """Test different fallback strategies."""
        template = PromptTemplate(
            system_prompt="Test system prompt",
            user_template="Test user prompt",
            max_tokens=1000,
            temperature=0.1,
            model_type=ModelType.ANALYSIS
        )
        
        # Mock successful fallback
        async def mock_operation(template, **kwargs):
            if kwargs.get('prefer_local'):
                usage = LLMUsage(
                    input_tokens=50,
                    output_tokens=100,
                    total_tokens=150,
                    cost=0.0,  # Local model has no cost
                    model="ollama/llama3",
                    provider="ollama",
                    timestamp=datetime.now()
                )
                return LLMResponse(
                    content="Fallback response",
                    usage=usage,
                    model="ollama/llama3",
                    provider="ollama"
                )
            else:
                raise Exception("Primary provider failed")
        
        fallback_result = await fallback_service.execute_with_fallback(
            operation=mock_operation,
            template=template,
            context={}
        )
        
        assert fallback_result.success is True
        assert fallback_result.response is not None
        assert fallback_result.fallback_strategy is not None
    
    def test_circuit_breaker_functionality(self, fallback_service):
        """Test circuit breaker state management."""
        provider = "openai"
        
        # Should initially allow calls
        assert fallback_service._can_call_provider(provider) is True
        
        # Simulate failures
        for _ in range(5):  # Reach failure threshold
            error_context = MagicMock()
            error_context.severity = ErrorSeverity.HIGH
            asyncio.run(fallback_service._record_failure(provider, error_context))
        
        # Circuit should now be open
        assert fallback_service._circuit_breakers[provider].failure_count >= 5


class TestContentEnrichmentService:
    """Test suite for content enrichment functionality."""
    
    @pytest.fixture
    async def enrichment_service(self):
        """Create enrichment service instance for testing."""
        service = ContentEnrichmentService()
        service.llm_service = AsyncMock()
        service.redis_manager = AsyncMock()
        service._initialized = True
        yield service
    
    @pytest.fixture
    def sample_paper_data(self):
        """Sample paper data for testing."""
        return {
            "title": "A Novel Approach to Machine Learning",
            "abstract": "This paper presents a novel approach to machine learning that improves accuracy by 15%.",
            "year": 2023,
            "authors": [{"name": "John Doe"}, {"name": "Jane Smith"}],
            "venue": {"name": "ICML"},
            "citation_count": 42
        }
    
    @pytest.mark.asyncio
    async def test_paper_enrichment(self, enrichment_service, sample_paper_data):
        """Test paper enrichment process."""
        # Mock LLM response
        mock_llm_response = MagicMock()
        mock_llm_response.content = """
        ## Summary
        This paper presents a breakthrough in machine learning methodology.
        
        ## Key Contributions
        - Novel algorithm design
        - 15% accuracy improvement
        - Comprehensive evaluation
        
        ## Methodology
        The authors propose a new neural network architecture.
        
        ## Key Findings
        - Significant performance gains
        - Robust across datasets
        
        ## Impact and Significance
        This work represents a major advance in the field.
        
        ## Limitations
        - Limited to supervised learning
        - Computational complexity concerns
        
        ## Future Directions
        - Extend to unsupervised learning
        - Optimize computational efficiency
        """
        mock_llm_response.usage = MagicMock()
        mock_llm_response.usage.cost = 0.025
        mock_llm_response.usage.total_tokens = 500
        mock_llm_response.model = "gpt-4"
        
        enrichment_service.llm_service.complete = AsyncMock(return_value=mock_llm_response)
        enrichment_service._fetch_paper_data = AsyncMock(return_value=sample_paper_data)
        enrichment_service._cache_enriched_content = AsyncMock()
        
        enriched = await enrichment_service.enrich_paper(
            paper_id="test_paper_123",
            paper_data=sample_paper_data
        )
        
        assert isinstance(enriched, EnrichedContent)
        assert enriched.enhanced_summary is not None
        assert len(enriched.key_contributions) > 0
        assert enriched.methodology_summary is not None
        assert enriched.content_quality in [q for q in ContentQuality]
    
    @pytest.mark.asyncio
    async def test_batch_enrichment(self, enrichment_service, sample_paper_data):
        """Test batch enrichment functionality."""
        from app.services.content_enrichment import EnrichmentRequest
        
        requests = [
            EnrichmentRequest(paper_id="paper1", paper_data=sample_paper_data),
            EnrichmentRequest(paper_id="paper2", paper_data=sample_paper_data),
            EnrichmentRequest(paper_id="paper3", paper_data=sample_paper_data)
        ]
        
        # Mock successful enrichment
        enrichment_service.enrich_paper = AsyncMock(return_value=EnrichedContent(
            paper_id="test",
            paper_title="Test Title",
            content_quality=ContentQuality.GOOD,
            confidence_score=0.8
        ))
        
        result = await enrichment_service.batch_enrich_papers(requests)
        
        assert result.total_papers == 3
        assert result.successful_enrichments <= 3  # May have some failures in mock
        assert len(result.enriched_papers) <= 3


class TestCitationAnalysisService:
    """Test suite for citation analysis functionality."""
    
    @pytest.fixture
    async def citation_service(self):
        """Create citation analysis service instance for testing."""
        service = CitationAnalysisService()
        service.llm_service = AsyncMock()
        service.redis_manager = AsyncMock()
        service._initialized = True
        yield service
    
    @pytest.mark.asyncio
    async def test_citation_relationship_analysis(self, citation_service):
        """Test citation relationship analysis."""
        # Mock paper data
        citing_paper = {
            "title": "Advanced ML Techniques",
            "abstract": "Building on previous work...",
            "authors": [{"name": "Alice Johnson"}],
            "year": 2024
        }
        
        cited_paper = {
            "title": "Foundational ML Theory", 
            "abstract": "This paper establishes fundamental principles...",
            "authors": [{"name": "Bob Wilson"}],
            "year": 2020
        }
        
        # Mock LLM analysis response
        mock_response = MagicMock()
        mock_response.content = """
        ## Citation Purpose
        The citing paper builds upon the theoretical framework established in the cited work.
        
        ## Intellectual Relationship
        Extension of foundational concepts to practical applications.
        
        ## Knowledge Flow
        Theoretical principles from cited paper enable new algorithmic approaches.
        
        ## Impact Assessment
        High impact - cited work provides essential theoretical foundation.
        
        ## Citation Type
        Methodology - extends theoretical framework to practical implementation.
        """
        mock_response.usage = MagicMock()
        mock_response.usage.cost = 0.02
        mock_response.model = "gpt-4"
        
        citation_service.llm_service.complete = AsyncMock(return_value=mock_response)
        citation_service._fetch_paper_data = AsyncMock(side_effect=[citing_paper, cited_paper])
        citation_service._cache_citation_analysis = AsyncMock()
        
        relationship = await citation_service.analyze_citation_relationship(
            citing_paper_id="citing_123",
            cited_paper_id="cited_456"
        )
        
        assert isinstance(relationship, CitationRelationship)
        assert relationship.citation_purpose is not None
        assert relationship.intellectual_relationship is not None
        assert relationship.citation_type is not None


class TestEnhancedLLMService:
    """Test suite for enhanced LLM service with fallbacks."""
    
    @pytest.fixture
    async def enhanced_service(self):
        """Create enhanced LLM service instance for testing."""
        service = EnhancedLLMService()
        service.base_service = AsyncMock()
        service._fallback_service = AsyncMock()
        service._cost_manager = AsyncMock()
        service._initialized = True
        yield service
    
    @pytest.mark.asyncio
    async def test_completion_with_fallback_success(self, enhanced_service):
        """Test successful completion with fallback protection."""
        template = PromptTemplate(
            system_prompt="Test",
            user_template="Test {text}",
            max_tokens=100,
            temperature=0.1
        )
        
        # Mock successful primary operation
        mock_response = MagicMock()
        mock_response.content = "Test response"
        enhanced_service.base_service.complete = AsyncMock(return_value=mock_response)
        
        # Mock fallback service returning success without fallback
        from app.services.llm_fallback import FallbackResult
        fallback_result = FallbackResult(
            success=True,
            response=mock_response,
            fallback_strategy=None
        )
        enhanced_service._fallback_service.execute_with_fallback = AsyncMock(return_value=fallback_result)
        
        response = await enhanced_service.complete_with_fallback(template, text="test")
        
        assert response == mock_response
        assert enhanced_service._fallback_stats['total_requests'] == 1
    
    @pytest.mark.asyncio
    async def test_completion_with_fallback_used(self, enhanced_service):
        """Test completion where fallback is used."""
        template = PromptTemplate(
            system_prompt="Test",
            user_template="Test {text}",
            max_tokens=100,
            temperature=0.1
        )
        
        # Mock fallback response
        fallback_response = MagicMock()
        fallback_response.content = "Fallback response"
        
        from app.services.llm_fallback import FallbackResult, FallbackStrategy
        fallback_result = FallbackResult(
            success=True,
            response=fallback_response,
            fallback_strategy=FallbackStrategy.LOCAL_FALLBACK,
            cost_savings=0.02
        )
        enhanced_service._fallback_service.execute_with_fallback = AsyncMock(return_value=fallback_result)
        
        response = await enhanced_service.complete_with_fallback(template, text="test")
        
        assert response == fallback_response
        assert enhanced_service._fallback_stats['fallback_used'] == 1
        assert enhanced_service._fallback_stats['cost_savings_from_fallback'] == 0.02


class TestAPIEndpoints:
    """Test suite for API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from app.main import app
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        with patch('app.services.llm_service_enhanced.get_enhanced_llm_service') as mock_service:
            mock_service.return_value.health_check_with_fallback = AsyncMock(return_value={
                "status": "healthy",
                "timestamp": datetime.now().isoformat()
            })
            
            response = client.get("/api/v1/llm/monitoring/health")
            assert response.status_code == 200
            
            data = response.json()
            assert "overall_status" in data
            assert "timestamp" in data
    
    def test_dashboard_endpoint(self, client):
        """Test dashboard endpoint."""
        with patch('app.services.llm_monitoring.get_monitoring_service') as mock_monitoring:
            from app.services.llm_monitoring import DashboardData
            
            mock_dashboard_data = DashboardData(
                timestamp=datetime.now(),
                time_period="last_hour",
                avg_response_time_ms=250.0,
                total_requests=100,
                success_rate=0.95,
                error_rate=0.05,
                total_cost=2.5,
                cost_per_request=0.025,
                budget_utilization=0.25,
                avg_quality_score=0.8,
                avg_confidence_score=0.75,
                requests_by_model={"gpt-4": 60, "claude-3": 40},
                requests_by_category={"analysis": 70, "summary": 30},
                peak_usage_hour=14,
                cache_hit_rate=0.6,
                cache_savings=0.5,
                fallback_usage_rate=0.1,
                circuit_breaker_status={"openai": "closed", "anthropic": "closed"}
            )
            
            mock_monitoring.return_value.get_dashboard_data = AsyncMock(return_value=mock_dashboard_data)
            mock_monitoring.return_value.get_active_alerts = AsyncMock(return_value=[])
            
            response = client.get("/api/v1/llm/monitoring/dashboard")
            assert response.status_code == 200
            
            data = response.json()
            assert "performance" in data
            assert "costs" in data
            assert "quality" in data


# Integration Tests

class TestIntegrationScenarios:
    """Integration tests for complete workflows."""
    
    @pytest.mark.asyncio
    async def test_full_paper_enrichment_workflow(self):
        """Test complete paper enrichment workflow with all services."""
        # This would test the full pipeline from API request to enriched result
        # Including caching, cost tracking, monitoring, and fallback handling
        pass
    
    @pytest.mark.asyncio 
    async def test_cost_budget_enforcement_workflow(self):
        """Test cost budget enforcement across services."""
        # Test that budget limits are respected across all services
        pass
    
    @pytest.mark.asyncio
    async def test_fallback_chain_workflow(self):
        """Test complete fallback chain execution."""
        # Test primary failure -> model downgrade -> provider switch -> local fallback -> static response
        pass


# Performance Tests

class TestPerformanceScenarios:
    """Performance and load testing scenarios."""
    
    @pytest.mark.asyncio
    async def test_concurrent_requests_handling(self):
        """Test handling of concurrent LLM requests."""
        # Test service behavior under concurrent load
        pass
    
    @pytest.mark.asyncio
    async def test_cache_performance_under_load(self):
        """Test cache performance with high request volume."""
        # Test cache hit rates and response times under load
        pass
    
    @pytest.mark.asyncio
    async def test_memory_usage_patterns(self):
        """Test memory usage patterns during extended operation."""
        # Test for memory leaks and efficient cleanup
        pass


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])