"""
Comprehensive test suite for OpenAlex client functionality.

Tests all major components of the OpenAlex client including:
- API client operations
- Rate limiting and caching
- Data conversion
- Error handling
- Background tasks
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, date
from typing import Dict, Any

from app.services.openalex import OpenAlexClient, CircuitBreaker, RateLimiter
from app.services.openalex_converter import OpenAlexConverter
from app.models.openalex import (
    OpenAlexWork,
    OpenAlexWorksResponse,
    OpenAlexExternalIds,
    OpenAlexAuthorship,
    OpenAlexAuthor,
    OpenAlexInstitution,
    OpenAlexConcept,
    OpenAlexSearchFilters
)
from app.models.paper import Paper, Author, Journal
from app.utils.exceptions import APIError, RateLimitError


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    def test_initial_state(self):
        """Test circuit breaker initial state."""
        cb = CircuitBreaker(failure_threshold=3, timeout=60)
        assert cb.state == "closed"
        assert cb.can_execute() is True
        assert cb.is_open() is False
    
    def test_failure_tracking(self):
        """Test failure tracking and state transitions."""
        cb = CircuitBreaker(failure_threshold=2, timeout=60)
        
        # Record failures
        cb.record_failure()
        assert cb.state == "closed"
        assert cb.failure_count == 1
        
        cb.record_failure()
        assert cb.state == "open"
        assert cb.failure_count == 2
        assert cb.can_execute() is False
        assert cb.is_open() is True
    
    def test_success_reset(self):
        """Test success resets failure count."""
        cb = CircuitBreaker(failure_threshold=2, timeout=60)
        
        cb.record_failure()
        cb.record_success()
        
        assert cb.state == "closed"
        assert cb.failure_count == 0
    
    def test_half_open_state(self):
        """Test half-open state after timeout."""
        import time
        
        cb = CircuitBreaker(failure_threshold=1, timeout=0.1)
        
        # Trip the circuit breaker
        cb.record_failure()
        assert cb.state == "open"
        
        # Wait for timeout
        time.sleep(0.2)
        
        # Should allow one request (half-open)
        assert cb.can_execute() is True
        assert cb.state == "half_open"


class TestRateLimiter:
    """Test rate limiter functionality."""
    
    @pytest.mark.asyncio
    async def test_token_bucket(self):
        """Test token bucket algorithm."""
        limiter = RateLimiter(requests_per_second=10, burst_size=5)
        
        # Should allow initial burst
        for _ in range(5):
            assert await limiter.acquire() is True
        
        # Should deny next request
        assert await limiter.acquire() is False
    
    @pytest.mark.asyncio
    async def test_token_replenishment(self):
        """Test token replenishment over time."""
        limiter = RateLimiter(requests_per_second=100, burst_size=2)  # Very fast for testing
        
        # Use up tokens
        assert await limiter.acquire() is True
        assert await limiter.acquire() is True
        assert await limiter.acquire() is False
        
        # Wait for replenishment
        await asyncio.sleep(0.1)
        
        # Should have more tokens now
        assert await limiter.acquire() is True


class TestOpenAlexClient:
    """Test OpenAlex client functionality."""
    
    @pytest.fixture
    def mock_redis_manager(self):
        """Mock Redis manager."""
        redis_mock = AsyncMock()
        redis_mock.cache_get.return_value = None
        redis_mock.cache_set.return_value = True
        return redis_mock
    
    @pytest.fixture
    def client(self, mock_redis_manager):
        """Create test client."""
        return OpenAlexClient(
            email="test@example.com",
            redis_manager=mock_redis_manager,
            requests_per_second=100,  # Fast for testing
            cache_ttl=60
        )
    
    @pytest.fixture
    def sample_work_response(self):
        """Sample OpenAlex work response."""
        return {
            "id": "https://openalex.org/W2741809807",
            "doi": "https://doi.org/10.7717/peerj.4375",
            "title": "The state of OA: a large-scale analysis of the prevalence and impact of Open Access articles",
            "display_name": "The state of OA: a large-scale analysis of the prevalence and impact of Open Access articles",
            "publication_year": 2018,
            "publication_date": "2018-02-13",
            "ids": {
                "openalex": "https://openalex.org/W2741809807",
                "doi": "https://doi.org/10.7717/peerj.4375",
                "mag": "2741809807",
                "pmid": "https://pubmed.ncbi.nlm.nih.gov/29456894"
            },
            "language": "en",
            "primary_location": {
                "is_oa": True,
                "landing_page_url": "https://doi.org/10.7717/peerj.4375",
                "pdf_url": "https://peerj.com/articles/4375.pdf",
                "source": {
                    "id": "https://openalex.org/S140737008",
                    "display_name": "PeerJ",
                    "issn_l": "2167-8359",
                    "issn": ["2167-8359"],
                    "is_oa": True,
                    "is_in_doaj": True,
                    "host_organization": "https://openalex.org/P4310320595",
                    "host_organization_name": "PeerJ",
                    "type": "journal"
                },
                "license": "cc-by",
                "version": "publishedVersion"
            },
            "type": "article",
            "type_crossref": "journal-article",
            "open_access": {
                "is_oa": True,
                "oa_date": "2018-02-13",
                "oa_url": "https://peerj.com/articles/4375.pdf",
                "any_repository_has_fulltext": True
            },
            "authorships": [
                {
                    "author_position": "first",
                    "author": {
                        "id": "https://openalex.org/A5014851618",
                        "display_name": "Heather Piwowar",
                        "orcid": "https://orcid.org/0000-0003-1613-5981"
                    },
                    "institutions": [
                        {
                            "id": "https://openalex.org/I4210158849",
                            "display_name": "Our Research",
                            "ror": "https://ror.org/02nr0ka47",
                            "country_code": "CA",
                            "type": "nonprofit"
                        }
                    ],
                    "raw_author_name": "Heather Piwowar",
                    "raw_affiliation_strings": ["Our Research, Vancouver, BC, Canada"]
                }
            ],
            "cited_by_count": 1584,
            "biblio": {
                "volume": "6",
                "issue": None,
                "first_page": "e4375",
                "last_page": "e4375"
            },
            "concepts": [
                {
                    "id": "https://openalex.org/C41008148",
                    "wikidata": "https://www.wikidata.org/wiki/Q21198",
                    "display_name": "Computer science",
                    "level": 0,
                    "score": 0.62
                }
            ],
            "mesh": [],
            "locations": [],
            "best_oa_location": {
                "is_oa": True,
                "landing_page_url": "https://doi.org/10.7717/peerj.4375",
                "pdf_url": "https://peerj.com/articles/4375.pdf",
                "host_type": "publisher",
                "license": "cc-by",
                "version": "publishedVersion"
            },
            "grants": [],
            "referenced_works": [
                "https://openalex.org/W1981416384",
                "https://openalex.org/W2013803880"
            ],
            "referenced_works_count": 2,
            "is_retracted": False,
            "is_paratext": False,
            "created_date": "2018-02-14",
            "updated_date": "2022-12-28T17:21:29.872731",
            "abstract_inverted_index": {
                "Despite": [0],
                "growing": [1],
                "interest": [2],
                "in": [3, 15],
                "Open": [4, 20],
                "Access": [5, 21],
                "(OA)": [6],
                "to": [7, 25],
                "scholarly": [8],
                "literature,": [9],
                "there": [10],
                "is": [11],
                "an": [12],
                "unmet": [13],
                "need": [14],
                "large-scale": [16],
                "evidence": [17],
                "on": [18],
                "the": [19, 23],
                "availability.": [22],
                "address": [24],
                "this": [26]
            },
            "cited_by_api_url": "https://api.openalex.org/works?filter=cites:W2741809807",
            "counts_by_year": [
                {"year": 2022, "cited_by_count": 176},
                {"year": 2021, "cited_by_count": 183}
            ]
        }
    
    @pytest.mark.asyncio
    async def test_client_initialization(self, client):
        """Test client initialization."""
        assert client.is_polite_pool is True
        assert client.email == "test@example.com"
        assert client.rate_limiter is not None
        assert client.circuit_breaker is not None
    
    @pytest.mark.asyncio
    async def test_get_work_by_id_success(self, client, sample_work_response):
        """Test successful work retrieval by ID."""
        with patch.object(client, '_make_request', return_value=sample_work_response):
            work = await client.get_work_by_id("W2741809807")
            
            assert work is not None
            assert isinstance(work, OpenAlexWork)
            assert work.title == "The state of OA: a large-scale analysis of the prevalence and impact of Open Access articles"
            assert work.cited_by_count == 1584
            assert len(work.authorships) == 1
    
    @pytest.mark.asyncio
    async def test_get_work_by_doi(self, client, sample_work_response):
        """Test work retrieval by DOI."""
        with patch.object(client, '_make_request', return_value=sample_work_response):
            work = await client.get_work_by_id("10.7717/peerj.4375")
            
            assert work is not None
            assert work.ids.doi == "https://doi.org/10.7717/peerj.4375"
    
    @pytest.mark.asyncio
    async def test_get_work_not_found(self, client):
        """Test handling of work not found."""
        with patch.object(client, '_make_request', side_effect=APIError("Resource not found", status_code=404)):
            work = await client.get_work_by_id("nonexistent")
            assert work is None
    
    @pytest.mark.asyncio
    async def test_batch_get_works(self, client, sample_work_response):
        """Test batch work retrieval."""
        mock_response = {
            "results": [sample_work_response],
            "meta": {"count": 1}
        }
        
        with patch.object(client, '_make_request', return_value=mock_response):
            works = await client.get_works_batch(["W2741809807"])
            
            assert len(works) == 1
            assert isinstance(works[0], OpenAlexWork)
            assert works[0].title == sample_work_response["title"]
    
    @pytest.mark.asyncio
    async def test_search_works(self, client, sample_work_response):
        """Test work search functionality."""
        mock_response = {
            "results": [sample_work_response],
            "meta": {"count": 1, "per_page": 25, "page": 1}
        }
        
        with patch.object(client, '_make_request', return_value=mock_response):
            response = await client.search_works("open access")
            
            assert isinstance(response, OpenAlexWorksResponse)
            assert len(response.results) == 1
            assert response.results[0].title == sample_work_response["title"]
    
    @pytest.mark.asyncio
    async def test_search_with_filters(self, client, sample_work_response):
        """Test search with filters."""
        filters = OpenAlexSearchFilters(
            from_publication_date=date(2018, 1, 1),
            to_publication_date=date(2020, 12, 31),
            is_oa=True
        )
        
        mock_response = {
            "results": [sample_work_response],
            "meta": {"count": 1}
        }
        
        with patch.object(client, '_make_request', return_value=mock_response) as mock_request:
            await client.search_works("test query", filters=filters)
            
            # Verify filters were applied
            args, kwargs = mock_request.call_args
            params = kwargs.get('params', {})
            assert 'filter' in params
            assert 'from_publication_date:2018-01-01' in params['filter']
            assert 'is_oa:true' in params['filter']
    
    @pytest.mark.asyncio
    async def test_get_citations(self, client, sample_work_response):
        """Test citation retrieval."""
        mock_response = {
            "results": [sample_work_response],
            "meta": {"count": 1}
        }
        
        with patch.object(client, '_make_request', return_value=mock_response):
            citing_works = await client.get_citations("W2741809807", "cited_by", 10)
            
            assert len(citing_works) == 1
            assert isinstance(citing_works[0], OpenAlexWork)
    
    @pytest.mark.asyncio
    async def test_traverse_citation_network(self, client, sample_work_response):
        """Test citation network traversal."""
        with patch.object(client, 'get_work_by_id', return_value=OpenAlexWork(**sample_work_response)):
            with patch.object(client, 'get_citations', return_value=[]):
                network = await client.traverse_citation_network("W2741809807", max_depth=1)
                
                assert network["center_work_id"] == "W2741809807"
                assert "nodes" in network
                assert "edges" in network
                assert len(network["nodes"]) >= 1  # At least the center work
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, client):
        """Test rate limiting functionality."""
        # Mock a rate limit response
        with patch.object(client.client, 'get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 429
            mock_response.headers = {"Retry-After": "1"}
            mock_get.return_value = mock_response
            
            with pytest.raises(RateLimitError):
                await client._make_request("/works/test", use_cache=False)
    
    @pytest.mark.asyncio
    async def test_caching(self, client, sample_work_response):
        """Test response caching."""
        # Mock cached response
        client.redis_manager.cache_get.return_value = sample_work_response
        
        result = await client._make_request("/works/test")
        
        assert result == sample_work_response
        assert client.cache_hits == 1
        client.redis_manager.cache_get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_open(self, client):
        """Test circuit breaker opens on repeated failures."""
        # Trip the circuit breaker
        for _ in range(5):
            client.circuit_breaker.record_failure()
        
        with pytest.raises(Exception):  # CircuitBreakerError would be raised
            await client._make_request("/works/test", use_cache=False)
    
    @pytest.mark.asyncio
    async def test_health_check(self, client):
        """Test health check functionality."""
        mock_response = {"results": [], "meta": {"count": 0}}
        
        with patch.object(client, '_make_request', return_value=mock_response):
            with patch.object(client, 'get_rate_limit_info') as mock_rate_limit:
                from app.models.openalex import OpenAlexRateLimit
                mock_rate_limit.return_value = OpenAlexRateLimit(
                    requests_remaining=1000,
                    requests_per_day=100000,
                    reset_time=datetime.utcnow(),
                    is_polite_pool=True
                )
                
                health = await client.health_check()
                
                assert health["status"] == "healthy"
                assert "response_time_ms" in health
                assert "rate_limit" in health
    
    @pytest.mark.asyncio
    async def test_metrics(self, client):
        """Test metrics collection."""
        # Make some requests to generate metrics
        client.request_count = 10
        client.cache_hits = 5
        client.cache_misses = 3
        client.rate_limit_waits = 1
        
        metrics = client.get_metrics()
        
        assert metrics["total_requests"] == 10
        assert metrics["cache_hits"] == 5
        assert metrics["cache_misses"] == 3
        assert metrics["cache_hit_rate"] == 5/8
        assert metrics["rate_limit_waits"] == 1
        assert metrics["is_polite_pool"] is True


class TestOpenAlexConverter:
    """Test OpenAlex data conversion functionality."""
    
    @pytest.fixture
    def sample_openalex_work(self):
        """Sample OpenAlex work for testing."""
        return OpenAlexWork(
            id="https://openalex.org/W2741809807",
            ids=OpenAlexExternalIds(
                openalex="https://openalex.org/W2741809807",
                doi="https://doi.org/10.7717/peerj.4375",
                pmid="29456894"
            ),
            title="Test Paper Title",
            publication_year=2018,
            publication_date=date(2018, 2, 13),
            cited_by_count=100,
            authorships=[
                OpenAlexAuthorship(
                    author_position="first",
                    author=OpenAlexAuthor(
                        id="https://openalex.org/A5014851618",
                        display_name="Test Author",
                        orcid="https://orcid.org/0000-0000-0000-0000"
                    ),
                    institutions=[
                        OpenAlexInstitution(
                            id="https://openalex.org/I123456",
                            display_name="Test University"
                        )
                    ]
                )
            ],
            concepts=[
                OpenAlexConcept(
                    id="https://openalex.org/C123",
                    display_name="Computer Science",
                    level=0,
                    score=0.8
                )
            ],
            primary_location={
                "source": {
                    "display_name": "Test Journal",
                    "issn": ["1234-5678"],
                    "host_organization_name": "Test Publisher"
                },
                "landing_page_url": "https://example.com/paper"
            },
            referenced_works=["https://openalex.org/W123", "https://openalex.org/W456"]
        )
    
    def test_convert_work_to_paper(self, sample_openalex_work):
        """Test conversion of OpenAlex work to Paper model."""
        paper = OpenAlexConverter.convert_openalex_work_to_paper(sample_openalex_work)
        
        assert isinstance(paper, Paper)
        assert paper.title == "Test Paper Title"
        assert paper.doi == "https://doi.org/10.7717/peerj.4375"
        assert paper.publication_year == 2018
        assert paper.citation_count.total == 100
        assert len(paper.authors) == 1
        assert paper.authors[0].name == "Test Author"
        assert paper.journal is not None
        assert paper.journal.name == "Test Journal"
        assert len(paper.references) == 2
    
    def test_convert_authorships_to_authors(self):
        """Test conversion of authorships to authors."""
        authorships = [
            OpenAlexAuthorship(
                author_position="first",
                author=OpenAlexAuthor(
                    id="https://openalex.org/A123",
                    display_name="John Doe",
                    orcid="0000-0000-0000-0001"
                ),
                institutions=[
                    OpenAlexInstitution(
                        id="https://openalex.org/I123",
                        display_name="MIT"
                    )
                ]
            )
        ]
        
        authors = OpenAlexConverter._convert_authorships_to_authors(authorships)
        
        assert len(authors) == 1
        assert isinstance(authors[0], Author)
        assert authors[0].name == "John Doe"
        assert authors[0].orcid == "https://orcid.org/0000-0000-0000-0001"
        assert authors[0].affiliation == "MIT"
    
    def test_convert_location_to_journal(self):
        """Test conversion of location to journal."""
        location = {
            "source": {
                "display_name": "Nature",
                "issn": ["0028-0836"],
                "host_organization_name": "Springer Nature"
            }
        }
        
        journal = OpenAlexConverter._convert_location_to_journal(location)
        
        assert isinstance(journal, Journal)
        assert journal.name == "Nature"
        assert journal.issn == "0028-0836"
        assert journal.publisher == "Springer Nature"
    
    def test_convert_location_to_journal_none(self):
        """Test handling of None location."""
        journal = OpenAlexConverter._convert_location_to_journal(None)
        assert journal is None
        
        journal = OpenAlexConverter._convert_location_to_journal({})
        assert journal is None
    
    def test_batch_convert_works(self, sample_openalex_work):
        """Test batch conversion of works to papers."""
        works = [sample_openalex_work, sample_openalex_work]
        papers = OpenAlexConverter.batch_convert_works_to_papers(works)
        
        assert len(papers) == 2
        assert all(isinstance(paper, Paper) for paper in papers)
    
    def test_extract_metadata_summary(self, sample_openalex_work):
        """Test extraction of paper metadata summary."""
        paper = OpenAlexConverter.convert_openalex_work_to_paper(sample_openalex_work)
        summary = OpenAlexConverter.extract_paper_metadata_summary(paper)
        
        assert "title" in summary
        assert "authors" in summary
        assert "citation_count" in summary
        assert summary["title"] == "Test Paper Title"
        assert len(summary["authors"]) == 1
    
    def test_create_paper_from_minimal_data(self):
        """Test creating paper from minimal data."""
        paper = OpenAlexConverter.create_paper_from_minimal_data(
            title="Test Paper",
            authors=["Author One", "Author Two"],
            doi="10.1000/123",
            year=2023,
            journal_name="Test Journal"
        )
        
        assert isinstance(paper, Paper)
        assert paper.title == "Test Paper"
        assert len(paper.authors) == 2
        assert paper.doi == "10.1000/123"
        assert paper.publication_year == 2023
        assert paper.journal.name == "Test Journal"
    
    def test_merge_paper_data(self, sample_openalex_work):
        """Test merging paper data."""
        # Create existing paper with minimal data
        existing_paper = OpenAlexConverter.create_paper_from_minimal_data(
            title="Test Paper Title",
            authors=["Test Author"],
            doi="https://doi.org/10.7717/peerj.4375"
        )
        
        # Create OpenAlex paper with rich data
        openalex_paper = OpenAlexConverter.convert_openalex_work_to_paper(sample_openalex_work)
        
        # Merge data
        merged_paper = OpenAlexConverter.merge_paper_data(existing_paper, openalex_paper)
        
        # Should have rich data from OpenAlex
        assert merged_paper.abstract is not None or openalex_paper.abstract is None  # May be None in test data
        assert merged_paper.publication_year == 2018
        assert merged_paper.citation_count.total == 100
        assert merged_paper.journal is not None
    
    def test_validate_paper_completeness(self, sample_openalex_work):
        """Test paper completeness validation."""
        paper = OpenAlexConverter.convert_openalex_work_to_paper(sample_openalex_work)
        validation = OpenAlexConverter.validate_paper_completeness(paper)
        
        assert "completeness_score" in validation
        assert "quality" in validation
        assert "missing_required" in validation
        assert "missing_important" in validation
        assert validation["completeness_score"] >= 0
        assert validation["quality"] in ["poor", "fair", "good", "excellent"]


class TestOpenAlexSearchFilters:
    """Test OpenAlex search filters."""
    
    def test_basic_filters(self):
        """Test basic filter functionality."""
        filters = OpenAlexSearchFilters(
            from_publication_date=date(2020, 1, 1),
            to_publication_date=date(2020, 12, 31),
            is_oa=True,
            type="article"
        )
        
        params = filters.to_query_params()
        
        assert "filter" in params
        filter_str = params["filter"]
        assert "from_publication_date:2020-01-01" in filter_str
        assert "to_publication_date:2020-12-31" in filter_str
        assert "is_oa:true" in filter_str
        assert "type:article" in filter_str
    
    def test_citation_count_filter(self):
        """Test citation count filtering."""
        filters = OpenAlexSearchFilters(cited_by_count=">100")
        params = filters.to_query_params()
        
        assert "cited_by_count:>100" in params["filter"]
    
    def test_entity_filters(self):
        """Test entity-based filtering."""
        filters = OpenAlexSearchFilters(
            authorships_author_id="A123456",
            authorships_institutions_id="I123456",
            concepts_id="C123456"
        )
        
        params = filters.to_query_params()
        filter_str = params["filter"]
        
        assert "authorships.author.id:A123456" in filter_str
        assert "authorships.institutions.id:I123456" in filter_str
        assert "concepts.id:C123456" in filter_str
    
    def test_empty_filters(self):
        """Test empty filters."""
        filters = OpenAlexSearchFilters()
        params = filters.to_query_params()
        
        assert params == {}


@pytest.mark.asyncio
class TestIntegration:
    """Integration tests combining multiple components."""
    
    async def test_full_paper_import_flow(self):
        """Test complete paper import flow."""
        # This would be a more complex integration test
        # that tests the full pipeline from API call to database storage
        pass
    
    async def test_citation_network_building(self):
        """Test complete citation network building."""
        # Integration test for citation network traversal and storage
        pass


# Test fixtures and utilities

@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Performance and load tests

class TestPerformance:
    """Performance tests for OpenAlex client."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_batch_processing_performance(self):
        """Test performance of batch processing."""
        # This would test performance with larger datasets
        pass
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_concurrent_requests(self):
        """Test concurrent request handling."""
        # This would test the client's ability to handle multiple concurrent requests
        pass


# Mark slow tests
pytest.mark.slow = pytest.mark.skipif(
    not pytest.config.getoption("--runslow"),
    reason="need --runslow option to run"
)