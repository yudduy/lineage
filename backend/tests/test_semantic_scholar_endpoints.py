"""
Tests for Semantic Scholar API endpoints.

Tests the FastAPI endpoints for Semantic Scholar functionality including
paper retrieval, search, enrichment, and background tasks.
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient
from fastapi import FastAPI, status

from app.api.v1.api import api_router
from app.models.semantic_scholar import (
    SemanticScholarPaper,
    SemanticScholarEmbedding,
    EnrichedPaper,
    CitationIntent
)


@pytest.fixture
def app():
    """Create FastAPI app for testing."""
    app = FastAPI()
    app.include_router(api_router)
    return app


@pytest.fixture
async def client(app):
    """Create HTTP client for testing."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def mock_semantic_scholar_client():
    """Mock Semantic Scholar client."""
    return AsyncMock()


@pytest.fixture
def mock_semantic_analysis_service():
    """Mock semantic analysis service."""
    return AsyncMock()


@pytest.fixture
def mock_enrichment_pipeline():
    """Mock enrichment pipeline."""
    return AsyncMock()


@pytest.fixture
def mock_paper_response():
    """Mock paper response data."""
    return {
        "paper_id": "test_paper_id",
        "corpus_id": "12345678",
        "url": "https://www.semanticscholar.org/paper/test_paper_id",
        "title": "Test Paper Title",
        "abstract": "This is a test paper abstract.",
        "venue": {
            "name": "Test Journal",
            "type": "journal"
        },
        "year": 2023,
        "reference_count": 25,
        "citation_count": 10,
        "influential_citation_count": 3,
        "is_open_access": True,
        "fields_of_study": ["Computer Science", "Machine Learning"],
        "s2_fields_of_study": [
            {"category": "Computer Science", "score": 0.95},
            {"category": "Machine Learning", "score": 0.89}
        ],
        "authors": [
            {
                "author_id": "author_1",
                "name": "John Doe"
            },
            {
                "author_id": "author_2",
                "name": "Jane Smith"
            }
        ],
        "external_ids": {
            "doi": "10.1000/test.doi",
            "arxiv_id": "2023.12345"
        }
    }


class TestSemanticScholarEndpoints:
    """Test cases for Semantic Scholar endpoints."""
    
    @pytest.mark.asyncio
    async def test_get_paper_success(
        self, 
        client, 
        mock_semantic_scholar_client,
        mock_paper_response
    ):
        """Test successful paper retrieval."""
        mock_paper = SemanticScholarPaper(**mock_paper_response)
        mock_semantic_scholar_client.get_paper_by_id.return_value = mock_paper
        
        with patch(
            'app.api.v1.endpoints.semantic_scholar.get_semantic_scholar_client',
            return_value=mock_semantic_scholar_client
        ):
            response = await client.get("/api/v1/semantic-scholar/paper/test_paper_id")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["paper_id"] == "test_paper_id"
            assert data["title"] == "Test Paper Title"
            assert data["citation_count"] == 10
    
    @pytest.mark.asyncio
    async def test_get_paper_not_found(self, client, mock_semantic_scholar_client):
        """Test paper not found."""
        mock_semantic_scholar_client.get_paper_by_id.return_value = None
        
        with patch(
            'app.api.v1.endpoints.semantic_scholar.get_semantic_scholar_client',
            return_value=mock_semantic_scholar_client
        ):
            response = await client.get("/api/v1/semantic-scholar/paper/nonexistent_id")
            
            assert response.status_code == status.HTTP_404_NOT_FOUND
            data = response.json()
            assert "not found" in data["detail"].lower()
    
    @pytest.mark.asyncio
    async def test_get_paper_with_embeddings(
        self,
        client,
        mock_semantic_scholar_client,
        mock_paper_response
    ):
        """Test paper retrieval with embeddings."""
        mock_paper_response["embedding"] = {
            "model": "specter",
            "vector": [0.1] * 768
        }
        mock_paper = SemanticScholarPaper(**mock_paper_response)
        mock_semantic_scholar_client.get_paper_by_id.return_value = mock_paper
        
        with patch(
            'app.api.v1.endpoints.semantic_scholar.get_semantic_scholar_client',
            return_value=mock_semantic_scholar_client
        ):
            response = await client.get(
                "/api/v1/semantic-scholar/paper/test_paper_id",
                params={"include_embeddings": True}
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "embedding" in data
            assert data["embedding"]["model"] == "specter"
    
    @pytest.mark.asyncio
    async def test_get_papers_batch_success(
        self,
        client,
        mock_semantic_scholar_client,
        mock_paper_response
    ):
        """Test successful batch paper retrieval."""
        paper1 = SemanticScholarPaper(**mock_paper_response)
        paper2_data = mock_paper_response.copy()
        paper2_data["paper_id"] = "test_paper_id_2"
        paper2 = SemanticScholarPaper(**paper2_data)
        
        mock_semantic_scholar_client.get_papers_batch.return_value = [paper1, paper2]
        
        with patch(
            'app.api.v1.endpoints.semantic_scholar.get_semantic_scholar_client',
            return_value=mock_semantic_scholar_client
        ):
            response = await client.post(
                "/api/v1/semantic-scholar/papers/batch",
                json=["test_paper_id", "test_paper_id_2"]
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert len(data) == 2
            assert data[0]["paper_id"] == "test_paper_id"
            assert data[1]["paper_id"] == "test_paper_id_2"
    
    @pytest.mark.asyncio
    async def test_search_papers_success(
        self,
        client,
        mock_semantic_scholar_client,
        mock_paper_response
    ):
        """Test successful paper search."""
        from app.models.semantic_scholar import SemanticScholarPapersResponse
        
        mock_search_response = SemanticScholarPapersResponse(
            total=100,
            offset=0,
            next=10,
            data=[SemanticScholarPaper(**mock_paper_response)]
        )
        mock_semantic_scholar_client.search_papers.return_value = mock_search_response
        
        with patch(
            'app.api.v1.endpoints.semantic_scholar.get_semantic_scholar_client',
            return_value=mock_semantic_scholar_client
        ):
            response = await client.get(
                "/api/v1/semantic-scholar/search",
                params={
                    "query": "machine learning",
                    "limit": 10,
                    "year": "2020-2023",
                    "min_citation_count": 5
                }
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["total"] == 100
            assert len(data["data"]) == 1
            assert data["data"][0]["paper_id"] == "test_paper_id"
    
    @pytest.mark.asyncio
    async def test_get_paper_citations_success(
        self,
        client,
        mock_semantic_scholar_client,
        mock_paper_response
    ):
        """Test successful citation retrieval."""
        citation_paper = SemanticScholarPaper(**mock_paper_response)
        citation_paper.paper_id = "citing_paper_id"
        
        mock_semantic_scholar_client.get_paper_citations.return_value = [citation_paper]
        
        with patch(
            'app.api.v1.endpoints.semantic_scholar.get_semantic_scholar_client',
            return_value=mock_semantic_scholar_client
        ):
            response = await client.get(
                "/api/v1/semantic-scholar/paper/test_paper_id/citations",
                params={"limit": 50}
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert len(data) == 1
            assert data[0]["paper_id"] == "citing_paper_id"
    
    @pytest.mark.asyncio
    async def test_get_paper_references_success(
        self,
        client,
        mock_semantic_scholar_client,
        mock_paper_response
    ):
        """Test successful reference retrieval."""
        reference_paper = SemanticScholarPaper(**mock_paper_response)
        reference_paper.paper_id = "referenced_paper_id"
        
        mock_semantic_scholar_client.get_paper_references.return_value = [reference_paper]
        
        with patch(
            'app.api.v1.endpoints.semantic_scholar.get_semantic_scholar_client',
            return_value=mock_semantic_scholar_client
        ):
            response = await client.get(
                "/api/v1/semantic-scholar/paper/test_paper_id/references",
                params={"limit": 50}
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert len(data) == 1
            assert data[0]["paper_id"] == "referenced_paper_id"
    
    @pytest.mark.asyncio
    async def test_get_influential_citations_success(
        self,
        client,
        mock_semantic_scholar_client
    ):
        """Test successful influential citations retrieval."""
        from app.models.semantic_scholar import SemanticScholarInfluentialCitation
        
        influential_citation = SemanticScholarInfluentialCitation(
            citing_paper_id="citing_paper_id",
            cited_paper_id="test_paper_id",
            is_influential=True,
            contexts=["This is an influential citation context"],
            intents=[CitationIntent.METHOD],
            citing_paper_title="Citing Paper Title",
            citation_year=2023
        )
        
        mock_semantic_scholar_client.get_influential_citations.return_value = [influential_citation]
        
        with patch(
            'app.api.v1.endpoints.semantic_scholar.get_semantic_scholar_client',
            return_value=mock_semantic_scholar_client
        ):
            response = await client.get(
                "/api/v1/semantic-scholar/paper/test_paper_id/influential-citations"
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert len(data) == 1
            assert data[0]["is_influential"] is True
            assert data[0]["citing_paper_id"] == "citing_paper_id"
    
    @pytest.mark.asyncio
    async def test_analyze_semantic_similarity_success(
        self,
        client,
        mock_semantic_scholar_client
    ):
        """Test successful semantic similarity analysis."""
        from app.models.semantic_scholar import SemanticScholarSimilarityResult
        
        similarity_result = SemanticScholarSimilarityResult(
            paper_id_1="paper1",
            paper_id_2="paper2",
            similarity_score=0.85,
            title_1="Paper 1 Title",
            title_2="Paper 2 Title"
        )
        
        mock_semantic_scholar_client.find_similar_papers.return_value = [similarity_result]
        
        with patch(
            'app.api.v1.endpoints.semantic_scholar.get_semantic_scholar_client',
            return_value=mock_semantic_scholar_client
        ):
            response = await client.post(
                "/api/v1/semantic-scholar/similarity/analyze",
                json={
                    "paper_id": "paper1",
                    "candidate_papers": ["paper2", "paper3"],
                    "similarity_threshold": 0.5
                }
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert len(data) == 1
            assert data[0]["similarity_score"] == 0.85
            assert data[0]["paper_id_1"] == "paper1"
            assert data[0]["paper_id_2"] == "paper2"
    
    @pytest.mark.asyncio
    async def test_get_paper_embedding_success(
        self,
        client,
        mock_semantic_scholar_client
    ):
        """Test successful paper embedding retrieval."""
        embedding = SemanticScholarEmbedding(
            model="specter",
            vector=[0.1] * 768
        )
        
        mock_semantic_scholar_client.get_paper_embedding.return_value = embedding
        
        with patch(
            'app.api.v1.endpoints.semantic_scholar.get_semantic_scholar_client',
            return_value=mock_semantic_scholar_client
        ):
            response = await client.get(
                "/api/v1/semantic-scholar/paper/test_paper_id/embedding"
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["model"] == "specter"
            assert len(data["vector"]) == 768
    
    @pytest.mark.asyncio
    async def test_get_paper_embedding_not_available(
        self,
        client,
        mock_semantic_scholar_client
    ):
        """Test paper embedding not available."""
        mock_semantic_scholar_client.get_paper_embedding.return_value = None
        
        with patch(
            'app.api.v1.endpoints.semantic_scholar.get_semantic_scholar_client',
            return_value=mock_semantic_scholar_client
        ):
            response = await client.get(
                "/api/v1/semantic-scholar/paper/test_paper_id/embedding"
            )
            
            assert response.status_code == status.HTTP_404_NOT_FOUND
            data = response.json()
            assert "No embedding available" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_build_semantic_citation_network_success(
        self,
        client,
        mock_semantic_scholar_client
    ):
        """Test successful citation network building."""
        from app.models.semantic_scholar import SemanticScholarCitationNetwork
        
        mock_network = SemanticScholarCitationNetwork(
            center_paper_id="test_paper_id",
            nodes=[SemanticScholarPaper(paper_id="test_paper_id", title="Test Paper")],
            edges=[{
                "source_id": "test_paper_id",
                "target_id": "citing_paper_id",
                "relation_type": "cited_by",
                "depth": 1
            }],
            influential_citations=[],
            citation_intents={},
            similarity_scores={},
            total_nodes=1,
            total_edges=1,
            influential_edges=0
        )
        
        mock_semantic_scholar_client.build_semantic_citation_network.return_value = mock_network
        
        with patch(
            'app.api.v1.endpoints.semantic_scholar.get_semantic_scholar_client',
            return_value=mock_semantic_scholar_client
        ):
            response = await client.post(
                "/api/v1/semantic-scholar/network/build",
                json={
                    "paper_id": "test_paper_id",
                    "max_depth": 2,
                    "max_papers_per_level": 20,
                    "similarity_threshold": 0.3
                }
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["center_paper_id"] == "test_paper_id"
            assert data["total_nodes"] == 1
            assert data["total_edges"] == 1
    
    @pytest.mark.asyncio
    async def test_enrich_paper_success(
        self,
        client,
        mock_enrichment_pipeline
    ):
        """Test successful paper enrichment."""
        mock_enrichment_pipeline.enrich_paper.return_value = "task_12345"
        
        with patch(
            'app.api.v1.endpoints.semantic_scholar.get_enrichment_pipeline',
            return_value=mock_enrichment_pipeline
        ):
            response = await client.post(
                "/api/v1/semantic-scholar/enrich/paper",
                json={
                    "paper_identifier": "10.1000/test.doi",
                    "priority": "medium",
                    "include_citations": True,
                    "include_references": True,
                    "include_embeddings": True,
                    "include_semantic_analysis": True,
                    "max_citations": 100,
                    "max_references": 100
                }
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["task_id"] == "task_12345"
            assert data["status"] == "queued"
    
    @pytest.mark.asyncio
    async def test_enrich_papers_batch_success(
        self,
        client,
        mock_enrichment_pipeline
    ):
        """Test successful batch paper enrichment."""
        mock_enrichment_pipeline.enrich_papers_batch.return_value = ["task_1", "task_2"]
        
        with patch(
            'app.api.v1.endpoints.semantic_scholar.get_enrichment_pipeline',
            return_value=mock_enrichment_pipeline
        ):
            response = await client.post(
                "/api/v1/semantic-scholar/enrich/batch",
                json={
                    "paper_identifiers": ["10.1000/test1.doi", "10.1000/test2.doi"],
                    "priority": "high",
                    "include_citations": True,
                    "include_references": False,
                    "max_citations": 50
                }
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["count"] == 2
            assert len(data["task_ids"]) == 2
            assert data["status"] == "queued"
    
    @pytest.mark.asyncio
    async def test_get_enrichment_status_success(
        self,
        client,
        mock_enrichment_pipeline
    ):
        """Test successful enrichment status retrieval."""
        from app.services.enrichment_pipeline import EnrichmentTask, EnrichmentStatus
        from datetime import datetime
        
        mock_task = EnrichmentTask(
            task_id="task_12345",
            paper_identifier="10.1000/test.doi",
            status=EnrichmentStatus.COMPLETED,
            created_at=datetime.utcnow(),
            completed_at=datetime.utcnow()
        )
        
        mock_enrichment_pipeline.get_task_status.return_value = mock_task
        
        with patch(
            'app.api.v1.endpoints.semantic_scholar.get_enrichment_pipeline',
            return_value=mock_enrichment_pipeline
        ):
            response = await client.get(
                "/api/v1/semantic-scholar/enrich/status/task_12345"
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["task_id"] == "task_12345"
            assert data["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_get_enrichment_status_not_found(
        self,
        client,
        mock_enrichment_pipeline
    ):
        """Test enrichment status not found."""
        mock_enrichment_pipeline.get_task_status.return_value = None
        
        with patch(
            'app.api.v1.endpoints.semantic_scholar.get_enrichment_pipeline',
            return_value=mock_enrichment_pipeline
        ):
            response = await client.get(
                "/api/v1/semantic-scholar/enrich/status/nonexistent_task"
            )
            
            assert response.status_code == status.HTTP_404_NOT_FOUND
            data = response.json()
            assert "Task not found" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_get_enriched_paper_success(
        self,
        client,
        mock_enrichment_pipeline,
        mock_paper_response
    ):
        """Test successful enriched paper retrieval."""
        enriched_paper = EnrichedPaper(
            doi="10.1000/test.doi",
            title="Test Paper",
            semantic_scholar_id="s2_id",
            openalex_id="oa_id",
            enrichment_sources=["semantic_scholar", "openalex"],
            semantic_scholar_data=SemanticScholarPaper(**mock_paper_response)
        )
        
        mock_enrichment_pipeline.get_enriched_paper.return_value = enriched_paper
        
        with patch(
            'app.api.v1.endpoints.semantic_scholar.get_enrichment_pipeline',
            return_value=mock_enrichment_pipeline
        ):
            response = await client.get(
                "/api/v1/semantic-scholar/enrich/result/10.1000/test.doi"
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["doi"] == "10.1000/test.doi"
            assert data["title"] == "Test Paper"
            assert "semantic_scholar" in data["enrichment_sources"]
    
    @pytest.mark.asyncio
    async def test_analyze_research_trajectory_success(
        self,
        client,
        mock_semantic_analysis_service
    ):
        """Test successful research trajectory analysis."""
        mock_analysis = {
            "paper_count": 5,
            "time_span": 3,
            "semantic_evolution": {},
            "citation_trajectory": {},
            "research_focus_shifts": {},
            "collaboration_patterns": {},
            "venue_diversity": {}
        }
        
        mock_semantic_analysis_service.analyze_research_trajectory.return_value = mock_analysis
        
        with patch(
            'app.api.v1.endpoints.semantic_scholar.get_semantic_analysis_service',
            return_value=mock_semantic_analysis_service
        ):
            response = await client.post(
                "/api/v1/semantic-scholar/analysis/research-trajectory",
                json={
                    "author_papers": ["paper1", "paper2", "paper3", "paper4", "paper5"],
                    "time_window_years": 5
                }
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["paper_count"] == 5
            assert data["time_span"] == 3
    
    @pytest.mark.asyncio
    async def test_identify_emerging_trends_success(
        self,
        client,
        mock_semantic_analysis_service
    ):
        """Test successful emerging trends identification."""
        mock_trends = {
            "field": "Machine Learning",
            "analysis_period": {
                "start_date": "2023-01-01",
                "end_date": "2024-01-01",
                "papers_analyzed": 100
            },
            "publication_trends": {},
            "citation_trends": {},
            "collaboration_trends": {},
            "venue_trends": {},
            "semantic_clusters": {}
        }
        
        mock_semantic_analysis_service.identify_emerging_research_trends.return_value = mock_trends
        
        with patch(
            'app.api.v1.endpoints.semantic_scholar.get_semantic_analysis_service',
            return_value=mock_semantic_analysis_service
        ):
            response = await client.post(
                "/api/v1/semantic-scholar/analysis/emerging-trends",
                json={
                    "field_of_study": "Machine Learning",
                    "time_window_months": 12,
                    "min_papers": 50
                }
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["field"] == "Machine Learning"
            assert data["analysis_period"]["papers_analyzed"] == 100
    
    @pytest.mark.asyncio
    async def test_get_queue_status_success(
        self,
        client,
        mock_enrichment_pipeline
    ):
        """Test successful queue status retrieval."""
        mock_status = {
            "queue_size": 5,
            "active_tasks": 2,
            "max_concurrent_tasks": 10,
            "metrics": {
                "total_tasks_processed": 100,
                "successful_enrichments": 85,
                "failed_enrichments": 15,
                "success_rate": 0.85
            }
        }
        
        mock_enrichment_pipeline.get_queue_status.return_value = mock_status
        
        with patch(
            'app.api.v1.endpoints.semantic_scholar.get_enrichment_pipeline',
            return_value=mock_enrichment_pipeline
        ):
            response = await client.get(
                "/api/v1/semantic-scholar/status/queue"
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["queue_size"] == 5
            assert data["active_tasks"] == 2
            assert data["metrics"]["success_rate"] == 0.85
    
    @pytest.mark.asyncio
    async def test_health_check_success(
        self,
        client,
        mock_semantic_scholar_client,
        mock_semantic_analysis_service,
        mock_enrichment_pipeline
    ):
        """Test successful health check."""
        mock_semantic_scholar_client.health_check.return_value = {"status": "healthy"}
        mock_semantic_analysis_service.get_metrics.return_value = {"requests": 100}
        mock_enrichment_pipeline.health_check.return_value = {"status": "healthy"}
        
        with patch(
            'app.api.v1.endpoints.semantic_scholar.get_semantic_scholar_client',
            return_value=mock_semantic_scholar_client
        ):
            with patch(
                'app.api.v1.endpoints.semantic_scholar.get_semantic_analysis_service',
                return_value=mock_semantic_analysis_service
            ):
                with patch(
                    'app.api.v1.endpoints.semantic_scholar.get_enrichment_pipeline',
                    return_value=mock_enrichment_pipeline
                ):
                    response = await client.get(
                        "/api/v1/semantic-scholar/status/health"
                    )
                    
                    assert response.status_code == status.HTTP_200_OK
                    data = response.json()
                    assert data["status"] == "healthy"
                    assert "components" in data
    
    @pytest.mark.asyncio
    async def test_get_metrics_success(
        self,
        client,
        mock_semantic_scholar_client,
        mock_semantic_analysis_service,
        mock_enrichment_pipeline
    ):
        """Test successful metrics retrieval."""
        mock_semantic_scholar_client.get_metrics.return_value = {
            "total_requests": 1000,
            "cache_hit_rate": 0.75
        }
        mock_semantic_analysis_service.get_metrics.return_value = {
            "clients_initialized": True
        }
        mock_enrichment_pipeline.get_queue_status.return_value = {
            "queue_size": 0,
            "active_tasks": 0
        }
        
        with patch(
            'app.api.v1.endpoints.semantic_scholar.get_semantic_scholar_client',
            return_value=mock_semantic_scholar_client
        ):
            with patch(
                'app.api.v1.endpoints.semantic_scholar.get_semantic_analysis_service',
                return_value=mock_semantic_analysis_service
            ):
                with patch(
                    'app.api.v1.endpoints.semantic_scholar.get_enrichment_pipeline',
                    return_value=mock_enrichment_pipeline
                ):
                    response = await client.get(
                        "/api/v1/semantic-scholar/metrics"
                    )
                    
                    assert response.status_code == status.HTTP_200_OK
                    data = response.json()
                    assert "semantic_scholar_client" in data
                    assert "semantic_analysis_service" in data
                    assert "enrichment_pipeline" in data
    
    @pytest.mark.asyncio
    async def test_validation_errors(self, client, mock_semantic_scholar_client):
        """Test validation errors for various endpoints."""
        
        with patch(
            'app.api.v1.endpoints.semantic_scholar.get_semantic_scholar_client',
            return_value=mock_semantic_scholar_client
        ):
            # Test batch with too many papers (over 500)
            large_batch = [f"paper_{i}" for i in range(501)]
            
            response = await client.post(
                "/api/v1/semantic-scholar/papers/batch",
                json=large_batch
            )
            
            # This should be handled by the client validation, but if it reaches the endpoint
            # it should return a 400 error
            assert response.status_code in [status.HTTP_400_BAD_REQUEST, status.HTTP_422_UNPROCESSABLE_ENTITY]
    
    @pytest.mark.asyncio 
    async def test_search_with_invalid_params(self, client, mock_semantic_scholar_client):
        """Test search with invalid parameters."""
        
        with patch(
            'app.api.v1.endpoints.semantic_scholar.get_semantic_scholar_client',
            return_value=mock_semantic_scholar_client
        ):
            # Test with limit too high
            response = await client.get(
                "/api/v1/semantic-scholar/search",
                params={
                    "query": "test",
                    "limit": 101  # Over maximum of 100
                }
            )
            
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY