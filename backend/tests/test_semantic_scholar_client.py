"""
Comprehensive tests for Semantic Scholar API client.

Tests the Semantic Scholar client with mocked responses to ensure
proper functionality without making actual API calls.
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, date
from httpx import Response, HTTPStatusError

from app.services.semantic_scholar import SemanticScholarClient
from app.models.semantic_scholar import (
    SemanticScholarPaper,
    SemanticScholarEmbedding,
    SemanticScholarSimilarityResult,
    SemanticScholarInfluentialCitation,
    CitationIntent
)
from app.utils.exceptions import APIError, RateLimitError, ValidationError


@pytest.fixture
def mock_redis_manager():
    """Mock Redis manager."""
    redis_mock = AsyncMock()
    redis_mock.cache_get.return_value = None
    redis_mock.cache_set.return_value = None
    return redis_mock


@pytest.fixture
def semantic_scholar_client(mock_redis_manager):
    """Create Semantic Scholar client with mocked dependencies."""
    return SemanticScholarClient(
        api_key="test_api_key",
        redis_manager=mock_redis_manager,
        cache_ttl=3600,
        max_retries=2
    )


@pytest.fixture
def mock_paper_response():
    """Mock paper response data."""
    return {
        "paperId": "test_paper_id",
        "corpusId": "12345678",
        "url": "https://www.semanticscholar.org/paper/test_paper_id",
        "title": "Test Paper Title",
        "abstract": "This is a test paper abstract.",
        "venue": {
            "name": "Test Journal",
            "type": "journal"
        },
        "year": 2023,
        "referenceCount": 25,
        "citationCount": 10,
        "influentialCitationCount": 3,
        "isOpenAccess": True,
        "fieldsOfStudy": ["Computer Science", "Machine Learning"],
        "s2FieldsOfStudy": [
            {"category": "Computer Science", "score": 0.95},
            {"category": "Machine Learning", "score": 0.89}
        ],
        "authors": [
            {
                "authorId": "author_1",
                "name": "John Doe"
            },
            {
                "authorId": "author_2", 
                "name": "Jane Smith"
            }
        ],
        "externalIds": {
            "DOI": "10.1000/test.doi",
            "ArXiv": "2023.12345"
        }
    }


@pytest.fixture
def mock_embedding_response():
    """Mock embedding response data."""
    return {
        "model": "specter",
        "vector": [0.1] * 768  # 768-dimensional vector
    }


@pytest.fixture
def mock_httpx_client():
    """Mock httpx client."""
    client_mock = AsyncMock()
    return client_mock


class TestSemanticScholarClient:
    """Test cases for SemanticScholarClient."""
    
    @pytest.mark.asyncio
    async def test_client_initialization(self, semantic_scholar_client):
        """Test client initialization."""
        assert semantic_scholar_client.api_key == "test_api_key"
        assert semantic_scholar_client.is_authenticated is True
        assert semantic_scholar_client.cache_ttl == 3600
        assert semantic_scholar_client.max_retries == 2
        assert semantic_scholar_client.rate_limiter.requests_per_second == 100.0
    
    @pytest.mark.asyncio
    async def test_client_initialization_without_api_key(self, mock_redis_manager):
        """Test client initialization without API key."""
        client = SemanticScholarClient(redis_manager=mock_redis_manager)
        assert client.api_key is None
        assert client.is_authenticated is False
        assert client.rate_limiter.requests_per_second == 1.0
    
    @pytest.mark.asyncio
    async def test_get_paper_by_id_success(self, semantic_scholar_client, mock_paper_response):
        """Test successful paper retrieval."""
        with patch.object(semantic_scholar_client, '_make_request') as mock_request:
            mock_request.return_value = mock_paper_response
            
            paper = await semantic_scholar_client.get_paper_by_id("test_paper_id")
            
            assert paper is not None
            assert paper.paper_id == "test_paper_id"
            assert paper.title == "Test Paper Title"
            assert paper.year == 2023
            assert paper.citation_count == 10
            assert len(paper.authors) == 2
            
            mock_request.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_paper_by_id_not_found(self, semantic_scholar_client):
        """Test paper not found scenario."""
        with patch.object(semantic_scholar_client, '_make_request') as mock_request:
            mock_request.side_effect = APIError("Resource not found", status_code=404)
            
            paper = await semantic_scholar_client.get_paper_by_id("nonexistent_id")
            
            assert paper is None
    
    @pytest.mark.asyncio
    async def test_get_papers_batch_success(self, semantic_scholar_client, mock_paper_response):
        """Test successful batch paper retrieval."""
        batch_response = [mock_paper_response, mock_paper_response.copy()]
        batch_response[1]["paperId"] = "test_paper_id_2"
        
        with patch.object(semantic_scholar_client.client, 'post') as mock_post:
            mock_response = AsyncMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = batch_response
            mock_post.return_value = mock_response
            
            papers = await semantic_scholar_client.get_papers_batch(
                ["test_paper_id", "test_paper_id_2"]
            )
            
            assert len(papers) == 2
            assert papers[0].paper_id == "test_paper_id"
            assert papers[1].paper_id == "test_paper_id_2"
    
    @pytest.mark.asyncio
    async def test_get_papers_batch_too_many_ids(self, semantic_scholar_client):
        """Test batch request with too many paper IDs."""
        paper_ids = [f"paper_{i}" for i in range(501)]  # Exceed limit
        
        with pytest.raises(ValidationError, match="Maximum 500 paper IDs allowed"):
            await semantic_scholar_client.get_papers_batch(paper_ids)
    
    @pytest.mark.asyncio
    async def test_search_papers_success(self, semantic_scholar_client, mock_paper_response):
        """Test successful paper search."""
        search_response = {
            "total": 100,
            "offset": 0,
            "next": 10,
            "data": [mock_paper_response]
        }
        
        with patch.object(semantic_scholar_client, '_make_request') as mock_request:
            mock_request.return_value = search_response
            
            results = await semantic_scholar_client.search_papers(
                query="machine learning",
                limit=10
            )
            
            assert results.total == 100
            assert len(results.data) == 1
            assert results.data[0].paper_id == "test_paper_id"
    
    @pytest.mark.asyncio
    async def test_search_papers_with_filters(self, semantic_scholar_client):
        """Test paper search with filters."""
        from app.models.semantic_scholar import SemanticScholarSearchFilters
        
        filters = SemanticScholarSearchFilters(
            year="2020-2023",
            min_citation_count=5,
            fields_of_study=["Computer Science"],
            open_access_pdf=True
        )
        
        with patch.object(semantic_scholar_client, '_make_request') as mock_request:
            mock_request.return_value = {"total": 0, "data": []}
            
            await semantic_scholar_client.search_papers(
                query="test query",
                filters=filters
            )
            
            # Verify filters were applied
            call_args = mock_request.call_args
            params = call_args[1]["params"]
            
            assert "year" in params
            assert "minCitationCount" in params
            assert "fieldsOfStudy" in params
            assert "openAccessPdf" in params
    
    @pytest.mark.asyncio
    async def test_get_paper_citations_success(self, semantic_scholar_client, mock_paper_response):
        """Test successful citation retrieval."""
        citations_response = {
            "data": [
                {
                    "citingPaper": mock_paper_response,
                    "contexts": ["This is a citation context"],
                    "intents": ["background"],
                    "isInfluential": True
                }
            ]
        }
        
        with patch.object(semantic_scholar_client, '_make_request') as mock_request:
            mock_request.return_value = citations_response
            
            citations = await semantic_scholar_client.get_paper_citations("test_paper_id")
            
            assert len(citations) == 1
            assert citations[0].paper_id == "test_paper_id"
            assert hasattr(citations[0], 'contexts')
            assert hasattr(citations[0], 'intents')
            assert hasattr(citations[0], 'is_influential')
    
    @pytest.mark.asyncio
    async def test_get_paper_references_success(self, semantic_scholar_client, mock_paper_response):
        """Test successful reference retrieval."""
        references_response = {
            "data": [
                {
                    "citedPaper": mock_paper_response,
                    "contexts": ["This is a reference context"],
                    "intents": ["method"],
                    "isInfluential": False
                }
            ]
        }
        
        with patch.object(semantic_scholar_client, '_make_request') as mock_request:
            mock_request.return_value = references_response
            
            references = await semantic_scholar_client.get_paper_references("test_paper_id")
            
            assert len(references) == 1
            assert references[0].paper_id == "test_paper_id"
    
    @pytest.mark.asyncio
    async def test_get_paper_embedding_success(self, semantic_scholar_client, mock_embedding_response):
        """Test successful embedding retrieval."""
        paper_response = {
            "paperId": "test_paper_id",
            "embedding": mock_embedding_response
        }
        
        with patch.object(semantic_scholar_client, 'get_paper_by_id') as mock_get_paper:
            mock_paper = SemanticScholarPaper(**{
                "paper_id": "test_paper_id",
                "embedding": SemanticScholarEmbedding(**mock_embedding_response)
            })
            mock_get_paper.return_value = mock_paper
            
            embedding = await semantic_scholar_client.get_paper_embedding("test_paper_id")
            
            assert embedding is not None
            assert embedding.model == "specter"
            assert len(embedding.vector) == 768
    
    @pytest.mark.asyncio
    async def test_get_paper_embedding_not_available(self, semantic_scholar_client):
        """Test embedding retrieval when not available."""
        with patch.object(semantic_scholar_client, 'get_paper_by_id') as mock_get_paper:
            mock_paper = SemanticScholarPaper(paper_id="test_paper_id")
            mock_get_paper.return_value = mock_paper
            
            embedding = await semantic_scholar_client.get_paper_embedding("test_paper_id")
            
            assert embedding is None
    
    @pytest.mark.asyncio
    async def test_compute_semantic_similarity_success(self, semantic_scholar_client, mock_embedding_response):
        """Test successful semantic similarity computation."""
        with patch.object(semantic_scholar_client, 'get_paper_embedding') as mock_get_embedding:
            embedding1 = SemanticScholarEmbedding(**mock_embedding_response)
            embedding2_data = mock_embedding_response.copy()
            embedding2_data["vector"] = [0.9] + [0.1] * 767  # Different vector
            embedding2 = SemanticScholarEmbedding(**embedding2_data)
            
            mock_get_embedding.side_effect = [embedding1, embedding2]
            
            similarity = await semantic_scholar_client.compute_semantic_similarity(
                "paper1", "paper2"
            )
            
            assert similarity is not None
            assert -1.0 <= similarity <= 1.0
    
    @pytest.mark.asyncio
    async def test_compute_semantic_similarity_missing_embeddings(self, semantic_scholar_client):
        """Test similarity computation with missing embeddings."""
        with patch.object(semantic_scholar_client, 'get_paper_embedding') as mock_get_embedding:
            mock_get_embedding.side_effect = [None, None]
            
            similarity = await semantic_scholar_client.compute_semantic_similarity(
                "paper1", "paper2"
            )
            
            assert similarity is None
    
    @pytest.mark.asyncio
    async def test_find_similar_papers_success(self, semantic_scholar_client):
        """Test successful similar papers finding."""
        with patch.object(semantic_scholar_client, 'compute_semantic_similarity') as mock_similarity:
            with patch.object(semantic_scholar_client, 'get_paper_by_id') as mock_get_paper:
                # Mock similarity scores
                mock_similarity.side_effect = [0.8, 0.6, 0.4]  # Only first two above threshold
                
                # Mock paper data
                mock_paper = SemanticScholarPaper(paper_id="ref_paper", title="Reference Paper")
                mock_candidate1 = SemanticScholarPaper(paper_id="candidate1", title="Candidate 1")
                mock_candidate2 = SemanticScholarPaper(paper_id="candidate2", title="Candidate 2")
                
                mock_get_paper.side_effect = [mock_paper, mock_candidate1, mock_candidate2]
                
                results = await semantic_scholar_client.find_similar_papers(
                    "ref_paper",
                    ["candidate1", "candidate2", "candidate3"],
                    similarity_threshold=0.5
                )
                
                assert len(results) == 2  # Only first two above threshold
                assert results[0].similarity_score == 0.8
                assert results[1].similarity_score == 0.6
                assert results[0].similarity_score > results[1].similarity_score  # Sorted
    
    @pytest.mark.asyncio
    async def test_get_influential_citations_success(self, semantic_scholar_client):
        """Test successful influential citations retrieval."""
        mock_citation = SemanticScholarPaper(
            paper_id="citing_paper",
            title="Citing Paper",
            year=2023,
            is_influential=True,
            contexts=["This is an influential citation context"],
            intents=[CitationIntent.METHOD]
        )
        
        with patch.object(semantic_scholar_client, 'get_paper_citations') as mock_get_citations:
            mock_get_citations.return_value = [mock_citation]
            
            influential_citations = await semantic_scholar_client.get_influential_citations(
                "test_paper_id"
            )
            
            assert len(influential_citations) == 1
            assert influential_citations[0].is_influential is True
            assert influential_citations[0].citing_paper_id == "citing_paper"
    
    @pytest.mark.asyncio
    async def test_rate_limiting_behavior(self, semantic_scholar_client):
        """Test rate limiting behavior."""
        # Test that rate limiter is properly initialized
        assert semantic_scholar_client.rate_limiter is not None
        
        # Test acquire permission
        can_proceed = await semantic_scholar_client.rate_limiter.acquire()
        assert can_proceed is True
        
        # Test wait_if_needed doesn't raise errors
        await semantic_scholar_client.rate_limiter.wait_if_needed()
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_behavior(self, semantic_scholar_client):
        """Test circuit breaker behavior."""
        circuit_breaker = semantic_scholar_client.circuit_breaker
        
        # Initially closed
        assert circuit_breaker.can_execute() is True
        assert circuit_breaker.is_open() is False
        
        # Record failures to trigger opening
        for _ in range(3):  # Exceed threshold
            circuit_breaker.record_failure()
        
        assert circuit_breaker.is_open() is True
        assert circuit_breaker.can_execute() is False
        
        # Record success to reset
        circuit_breaker.record_success()
        assert circuit_breaker.is_open() is False
        assert circuit_breaker.can_execute() is True
    
    @pytest.mark.asyncio
    async def test_caching_behavior(self, semantic_scholar_client, mock_paper_response):
        """Test caching behavior."""
        cache_key = "test_cache_key"
        
        # Mock cache miss then hit
        semantic_scholar_client.redis_manager.cache_get.side_effect = [None, mock_paper_response]
        
        # First call - cache miss
        with patch.object(semantic_scholar_client, '_make_request') as mock_request:
            mock_request.return_value = mock_paper_response
            
            result1 = await semantic_scholar_client._get_cached_response(cache_key)
            assert result1 is None
            assert semantic_scholar_client.cache_misses == 1
            
            # Second call - cache hit
            result2 = await semantic_scholar_client._get_cached_response(cache_key)
            assert result2 == mock_paper_response
            assert semantic_scholar_client.cache_hits == 1
    
    @pytest.mark.asyncio
    async def test_error_handling_http_errors(self, semantic_scholar_client):
        """Test HTTP error handling."""
        with patch.object(semantic_scholar_client.client, 'get') as mock_get:
            # Test 429 rate limit error
            response_429 = MagicMock()
            response_429.status_code = 429
            response_429.headers = {"Retry-After": "60"}
            
            mock_get.side_effect = HTTPStatusError(
                "Rate limited", 
                request=MagicMock(), 
                response=response_429
            )
            
            with pytest.raises(RateLimitError):
                await semantic_scholar_client._make_request("/test")
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self, semantic_scholar_client):
        """Test metrics collection."""
        initial_metrics = semantic_scholar_client.get_metrics()
        
        assert "total_requests" in initial_metrics
        assert "cache_hits" in initial_metrics
        assert "cache_misses" in initial_metrics
        assert "is_authenticated" in initial_metrics
        assert initial_metrics["is_authenticated"] is True
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, semantic_scholar_client):
        """Test successful health check."""
        with patch.object(semantic_scholar_client, '_make_request') as mock_request:
            mock_request.return_value = {"total": 1, "data": []}
            
            health = await semantic_scholar_client.health_check()
            
            assert health["status"] == "healthy"
            assert "response_time_ms" in health
            assert "rate_limit" in health
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, semantic_scholar_client):
        """Test health check with failure."""
        with patch.object(semantic_scholar_client, '_make_request') as mock_request:
            mock_request.side_effect = Exception("API unavailable")
            
            health = await semantic_scholar_client.health_check()
            
            assert health["status"] == "unhealthy"
            assert "error" in health
    
    @pytest.mark.asyncio
    async def test_client_cleanup(self, semantic_scholar_client):
        """Test client cleanup."""
        with patch.object(semantic_scholar_client.client, 'aclose') as mock_close:
            await semantic_scholar_client.close()
            mock_close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_context_manager(self, mock_redis_manager):
        """Test client as context manager."""
        async with SemanticScholarClient(redis_manager=mock_redis_manager) as client:
            assert client is not None
            
        # Client should be closed after context exit
        # This would be tested by checking if aclose was called


@pytest.mark.asyncio
async def test_client_factory_functions(mock_redis_manager):
    """Test client factory functions."""
    from app.services.semantic_scholar import get_semantic_scholar_client, close_semantic_scholar_client
    
    # Test getting client instance
    with patch('app.services.semantic_scholar.get_redis_manager', return_value=mock_redis_manager):
        client1 = await get_semantic_scholar_client(mock_redis_manager)
        client2 = await get_semantic_scholar_client(mock_redis_manager)
        
        # Should return same instance
        assert client1 is client2
    
    # Test closing client
    with patch.object(client1, 'close') as mock_close:
        await close_semantic_scholar_client()
        mock_close.assert_called_once()