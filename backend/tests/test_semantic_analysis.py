"""
Tests for semantic analysis service.

Tests the high-level semantic analysis functionality including
paper enrichment, trajectory analysis, and trend identification.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, date

from app.services.semantic_analysis import SemanticAnalysisService
from app.models.semantic_scholar import (
    SemanticScholarPaper,
    SemanticScholarEmbedding,
    EnrichedPaper,
    CitationIntent
)
from app.models.openalex import OpenAlexWork, OpenAlexExternalIds
from app.utils.exceptions import ValidationError


@pytest.fixture
def mock_semantic_scholar_client():
    """Mock Semantic Scholar client."""
    client = AsyncMock()
    return client


@pytest.fixture
def mock_openalex_client():
    """Mock OpenAlex client."""
    client = AsyncMock()
    return client


@pytest.fixture
def mock_redis_manager():
    """Mock Redis manager."""
    redis = AsyncMock()
    redis.cache_get.return_value = None
    redis.cache_set.return_value = None
    return redis


@pytest.fixture
def semantic_analysis_service(mock_semantic_scholar_client, mock_openalex_client, mock_redis_manager):
    """Create semantic analysis service with mocked dependencies."""
    service = SemanticAnalysisService(
        semantic_scholar_client=mock_semantic_scholar_client,
        openalex_client=mock_openalex_client,
        redis_manager=mock_redis_manager
    )
    service._clients_initialized = True
    return service


@pytest.fixture
def mock_semantic_scholar_paper():
    """Mock Semantic Scholar paper."""
    return SemanticScholarPaper(
        paper_id="s2_paper_id",
        title="Test Paper",
        abstract="This is a test paper abstract.",
        year=2023,
        citation_count=50,
        influential_citation_count=8,
        external_ids={
            "DOI": "10.1000/test.doi",
            "ArXiv": "2023.12345"
        },
        authors=[
            {"authorId": "author_1", "name": "John Doe"},
            {"authorId": "author_2", "name": "Jane Smith"}
        ],
        fields_of_study=["Computer Science", "Machine Learning"],
        s2_fields_of_study=[
            {"category": "Computer Science", "score": 0.95},
            {"category": "Machine Learning", "score": 0.89}
        ]
    )


@pytest.fixture
def mock_openalex_work():
    """Mock OpenAlex work."""
    return OpenAlexWork(
        id="https://openalex.org/W12345",
        title="Test Paper",
        publication_year=2023,
        cited_by_count=55,
        ids=OpenAlexExternalIds(
            doi="10.1000/test.doi",
            openalex="https://openalex.org/W12345"
        ),
        authorships=[
            {
                "author_position": "first",
                "author": {
                    "id": "https://openalex.org/A12345",
                    "display_name": "John Doe"
                },
                "institutions": []
            }
        ]
    )


@pytest.fixture
def mock_embedding():
    """Mock SPECTER embedding."""
    return SemanticScholarEmbedding(
        model="specter",
        vector=[0.1] * 768
    )


class TestSemanticAnalysisService:
    """Test cases for SemanticAnalysisService."""
    
    @pytest.mark.asyncio
    async def test_service_initialization(self, semantic_analysis_service):
        """Test service initialization."""
        assert semantic_analysis_service.semantic_scholar_client is not None
        assert semantic_analysis_service.openalex_client is not None
        assert semantic_analysis_service.redis_manager is not None
        assert semantic_analysis_service._clients_initialized is True
    
    @pytest.mark.asyncio
    async def test_enrich_paper_with_semantic_features_success(
        self, 
        semantic_analysis_service,
        mock_semantic_scholar_paper,
        mock_openalex_work
    ):
        """Test successful paper enrichment with both sources."""
        # Mock client responses
        semantic_analysis_service.semantic_scholar_client.get_paper_by_id.return_value = mock_semantic_scholar_paper
        semantic_analysis_service.openalex_client.get_work_by_id.return_value = mock_openalex_work
        
        # Mock analysis methods
        with patch.object(semantic_analysis_service, '_analyze_citation_patterns') as mock_citation_analysis:
            with patch.object(semantic_analysis_service, '_analyze_influential_citations') as mock_influence_analysis:
                with patch.object(semantic_analysis_service, '_compute_semantic_similarities') as mock_similarity_analysis:
                    
                    enriched_paper = await semantic_analysis_service.enrich_paper_with_semantic_features(
                        "10.1000/test.doi"
                    )
                    
                    assert enriched_paper is not None
                    assert enriched_paper.doi == "10.1000/test.doi"
                    assert enriched_paper.title == "Test Paper"
                    assert "semantic_scholar" in enriched_paper.enrichment_sources
                    assert "openalex" in enriched_paper.enrichment_sources
                    assert enriched_paper.semantic_scholar_data is not None
                    assert enriched_paper.openalex_data is not None
                    
                    # Verify analysis methods were called
                    mock_citation_analysis.assert_called_once()
                    mock_influence_analysis.assert_called_once()
                    mock_similarity_analysis.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_enrich_paper_semantic_scholar_only(
        self,
        semantic_analysis_service,
        mock_semantic_scholar_paper
    ):
        """Test paper enrichment with only Semantic Scholar data."""
        # Only Semantic Scholar returns data
        semantic_analysis_service.semantic_scholar_client.get_paper_by_id.return_value = mock_semantic_scholar_paper
        semantic_analysis_service.openalex_client.get_work_by_id.return_value = None
        
        with patch.object(semantic_analysis_service, '_analyze_citation_patterns'):
            with patch.object(semantic_analysis_service, '_analyze_influential_citations'):
                with patch.object(semantic_analysis_service, '_compute_semantic_similarities'):
                    
                    enriched_paper = await semantic_analysis_service.enrich_paper_with_semantic_features(
                        "s2_paper_id"
                    )
                    
                    assert enriched_paper is not None
                    assert "semantic_scholar" in enriched_paper.enrichment_sources
                    assert "openalex" not in enriched_paper.enrichment_sources
                    assert enriched_paper.semantic_scholar_data is not None
                    assert enriched_paper.openalex_data is None
    
    @pytest.mark.asyncio
    async def test_enrich_paper_no_data_found(self, semantic_analysis_service):
        """Test paper enrichment when no data is found."""
        # Both clients return None
        semantic_analysis_service.semantic_scholar_client.get_paper_by_id.return_value = None
        semantic_analysis_service.openalex_client.get_work_by_id.return_value = None
        
        enriched_paper = await semantic_analysis_service.enrich_paper_with_semantic_features(
            "nonexistent_paper"
        )
        
        assert enriched_paper is None
    
    @pytest.mark.asyncio
    async def test_analyze_citation_patterns(
        self,
        semantic_analysis_service,
        mock_semantic_scholar_paper
    ):
        """Test citation pattern analysis."""
        # Create enriched paper with S2 data
        enriched_paper = EnrichedPaper(
            semantic_scholar_id="s2_paper_id",
            semantic_scholar_data=mock_semantic_scholar_paper
        )
        
        # Mock citations with contexts and intents
        mock_citations = [
            SemanticScholarPaper(
                paper_id="citing_paper_1",
                year=2023,
                contexts=["This paper builds on the work of..."],
                intents=[CitationIntent.BACKGROUND]
            ),
            SemanticScholarPaper(
                paper_id="citing_paper_2", 
                year=2024,
                contexts=["We use the method described in..."],
                intents=[CitationIntent.METHOD]
            )
        ]
        
        semantic_analysis_service.semantic_scholar_client.get_paper_citations.return_value = mock_citations
        
        await semantic_analysis_service._analyze_citation_patterns(enriched_paper)
        
        assert enriched_paper.citation_intent_analysis is not None
        assert enriched_paper.citation_intent_analysis["total_citations"] == 2
        assert "intent_distribution" in enriched_paper.citation_intent_analysis
        assert "context_analysis" in enriched_paper.citation_intent_analysis
    
    @pytest.mark.asyncio
    async def test_analyze_influential_citations(
        self,
        semantic_analysis_service,
        mock_semantic_scholar_paper
    ):
        """Test influential citation analysis."""
        enriched_paper = EnrichedPaper(
            semantic_scholar_id="s2_paper_id",
            semantic_scholar_data=mock_semantic_scholar_paper
        )
        
        # Mock influential citations
        mock_influential_citations = [
            {
                "citing_paper_id": "influential_paper_1",
                "cited_paper_id": "s2_paper_id",
                "is_influential": True,
                "citation_year": 2023,
                "intents": [CitationIntent.RESULT]
            },
            {
                "citing_paper_id": "influential_paper_2",
                "cited_paper_id": "s2_paper_id", 
                "is_influential": True,
                "citation_year": 2024,
                "intents": [CitationIntent.METHOD]
            }
        ]
        
        semantic_analysis_service.semantic_scholar_client.get_influential_citations.return_value = mock_influential_citations
        
        await semantic_analysis_service._analyze_influential_citations(enriched_paper)
        
        assert enriched_paper.influential_citation_analysis is not None
        assert enriched_paper.influential_citation_analysis["influential_citations"] == 2
        assert enriched_paper.influential_citation_analysis["total_citations"] == 50  # From mock paper
        assert "temporal_influence" in enriched_paper.influential_citation_analysis
    
    @pytest.mark.asyncio
    async def test_compute_semantic_similarities(
        self,
        semantic_analysis_service,
        mock_semantic_scholar_paper,
        mock_embedding
    ):
        """Test semantic similarity computation."""
        enriched_paper = EnrichedPaper(
            semantic_scholar_id="s2_paper_id",
            semantic_scholar_data=mock_semantic_scholar_paper
        )
        
        # Mock embedding and related papers
        semantic_analysis_service.semantic_scholar_client.get_paper_embedding.return_value = mock_embedding
        
        mock_citations = [SemanticScholarPaper(paper_id="citing_paper_1")]
        mock_references = [SemanticScholarPaper(paper_id="referenced_paper_1")]
        
        semantic_analysis_service.semantic_scholar_client.get_paper_citations.return_value = mock_citations
        semantic_analysis_service.semantic_scholar_client.get_paper_references.return_value = mock_references
        
        # Mock similarity results
        mock_similarity_results = [
            {
                "paper_id_1": "s2_paper_id",
                "paper_id_2": "citing_paper_1",
                "similarity_score": 0.85
            }
        ]
        
        semantic_analysis_service.semantic_scholar_client.find_similar_papers.return_value = mock_similarity_results
        
        await semantic_analysis_service._compute_semantic_similarities(enriched_paper)
        
        assert enriched_paper.semantic_similarity_scores is not None
        assert "citing_paper_1" in enriched_paper.semantic_similarity_scores
        assert enriched_paper.semantic_similarity_scores["citing_paper_1"] == 0.85
    
    @pytest.mark.asyncio
    async def test_analyze_research_trajectory_success(self, semantic_analysis_service):
        """Test successful research trajectory analysis."""
        # Mock paper data with temporal progression
        mock_papers = [
            SemanticScholarPaper(
                paper_id="paper_1",
                title="Early Work",
                year=2020,
                citation_count=10,
                authors=[{"authorId": "author_1", "name": "John Doe"}],
                fields_of_study=["Computer Science"]
            ),
            SemanticScholarPaper(
                paper_id="paper_2", 
                title="Middle Work",
                year=2021,
                citation_count=25,
                authors=[{"authorId": "author_1", "name": "John Doe"}],
                fields_of_study=["Computer Science", "Machine Learning"]
            ),
            SemanticScholarPaper(
                paper_id="paper_3",
                title="Recent Work", 
                year=2022,
                citation_count=40,
                authors=[{"authorId": "author_1", "name": "John Doe"}],
                fields_of_study=["Machine Learning", "Deep Learning"]
            )
        ]
        
        # Mock embeddings
        mock_embeddings = [
            SemanticScholarEmbedding(model="specter", vector=[0.1] * 768),
            SemanticScholarEmbedding(model="specter", vector=[0.5] * 768),
            SemanticScholarEmbedding(model="specter", vector=[0.9] * 768)
        ]
        
        semantic_analysis_service.semantic_scholar_client.get_paper_by_id.side_effect = mock_papers
        semantic_analysis_service.semantic_scholar_client.get_paper_embedding.side_effect = mock_embeddings
        
        analysis = await semantic_analysis_service.analyze_research_trajectory(
            ["paper_1", "paper_2", "paper_3"],
            time_window_years=3
        )
        
        assert analysis["paper_count"] == 3
        assert analysis["time_span"] == 2  # 2022 - 2020
        assert "semantic_evolution" in analysis
        assert "citation_trajectory" in analysis
        assert "research_focus_shifts" in analysis
        assert "collaboration_patterns" in analysis
        assert "venue_diversity" in analysis
    
    @pytest.mark.asyncio
    async def test_analyze_research_trajectory_insufficient_papers(self, semantic_analysis_service):
        """Test research trajectory analysis with insufficient papers."""
        with pytest.raises(ValidationError, match="Need at least 3 papers"):
            await semantic_analysis_service.analyze_research_trajectory(["paper_1", "paper_2"])
    
    @pytest.mark.asyncio
    async def test_identify_emerging_research_trends_success(self, semantic_analysis_service):
        """Test successful emerging trends identification."""
        # Mock search results with recent papers
        mock_papers = []
        for i in range(60):  # Above minimum threshold
            mock_papers.append(SemanticScholarPaper(
                paper_id=f"paper_{i}",
                title=f"Paper {i}",
                year=2023,
                citation_count=i * 2,
                publication_date=date(2023, 6, 1),
                venue={"name": f"Journal {i % 5}", "type": "journal"},
                authors=[{"authorId": f"author_{i}", "name": f"Author {i}"}]
            ))
        
        # Mock search response
        search_response = {
            "total": 60,
            "data": mock_papers
        }
        
        semantic_analysis_service.semantic_scholar_client.search_papers.return_value = search_response
        
        # Mock embeddings for clustering
        mock_embeddings = [SemanticScholarEmbedding(model="specter", vector=[i/100] * 768) for i in range(20)]
        semantic_analysis_service.semantic_scholar_client.get_paper_embedding.side_effect = mock_embeddings
        
        trends = await semantic_analysis_service.identify_emerging_research_trends(
            "Machine Learning",
            time_window_months=12,
            min_papers=50
        )
        
        assert trends["field"] == "Machine Learning"
        assert trends["analysis_period"]["papers_analyzed"] == 60
        assert "publication_trends" in trends
        assert "citation_trends" in trends
        assert "collaboration_trends" in trends
        assert "venue_trends" in trends
        assert "semantic_clusters" in trends
    
    @pytest.mark.asyncio
    async def test_identify_emerging_trends_insufficient_papers(self, semantic_analysis_service):
        """Test trend analysis with insufficient papers."""
        # Mock insufficient search results
        mock_papers = [SemanticScholarPaper(paper_id="paper_1", title="Paper 1")]
        search_response = {"total": 1, "data": mock_papers}
        
        semantic_analysis_service.semantic_scholar_client.search_papers.return_value = search_response
        
        with pytest.raises(ValidationError, match="Insufficient papers found"):
            await semantic_analysis_service.identify_emerging_research_trends(
                "Rare Field",
                min_papers=50
            )
    
    @pytest.mark.asyncio
    async def test_semantic_evolution_analysis(self, semantic_analysis_service):
        """Test semantic evolution analysis component."""
        # Test data for evolution analysis
        embeddings = [
            [0.1] * 768,  # Early paper
            [0.5] * 768,  # Middle paper  
            [0.9] * 768   # Recent paper
        ]
        years = [2020, 2021, 2022]
        
        analysis = semantic_analysis_service._analyze_semantic_evolution(embeddings, years)
        
        assert "avg_temporal_similarity" in analysis
        assert "similarity_variance" in analysis
        assert "potential_shifts" in analysis
        assert "research_phases" in analysis
        
        # Check that shifts are detected (low similarity between consecutive papers)
        assert len(analysis["potential_shifts"]) == 2  # Between consecutive pairs
    
    @pytest.mark.asyncio
    async def test_citation_trajectory_analysis(self, semantic_analysis_service):
        """Test citation trajectory analysis component."""
        citation_counts = [5, 15, 30, 45, 50]
        years = [2020, 2021, 2022, 2023, 2024]
        
        analysis = semantic_analysis_service._analyze_citation_trajectory(citation_counts, years)
        
        assert analysis["total_citations"] == 145
        assert analysis["avg_citations_per_paper"] == 29.0
        assert analysis["max_citations"] == 50
        assert "citation_growth_rates" in analysis
        assert "avg_growth_rate" in analysis
        assert "citation_timeline" in analysis
        
        # Growth rates should be positive (citations increasing)
        assert all(rate > 0 for rate in analysis["citation_growth_rates"])
    
    @pytest.mark.asyncio
    async def test_research_focus_shifts_identification(self, semantic_analysis_service):
        """Test research focus shifts identification."""
        # Mock papers with changing fields
        papers = [
            SemanticScholarPaper(
                paper_id="paper_1",
                year=2020,
                fields_of_study=["Computer Science", "Algorithms"],
                s2_fields_of_study=[
                    {"category": "Computer Science", "score": 0.9},
                    {"category": "Algorithms", "score": 0.8}
                ]
            ),
            SemanticScholarPaper(
                paper_id="paper_2",
                year=2021,
                fields_of_study=["Computer Science", "Machine Learning"],
                s2_fields_of_study=[
                    {"category": "Computer Science", "score": 0.9},
                    {"category": "Machine Learning", "score": 0.85}
                ]
            ),
            SemanticScholarPaper(
                paper_id="paper_3",
                year=2022,
                fields_of_study=["Machine Learning", "Deep Learning"],
                s2_fields_of_study=[
                    {"category": "Machine Learning", "score": 0.95},
                    {"category": "Deep Learning", "score": 0.9}
                ]
            )
        ]
        
        analysis = await semantic_analysis_service._identify_research_focus_shifts(papers)
        
        assert "field_timeline" in analysis
        assert "field_transitions" in analysis
        assert "field_diversity_over_time" in analysis
        
        assert len(analysis["field_timeline"]) == 3
        assert len(analysis["field_transitions"]) == 2
        
        # Check that transitions show changing fields
        transitions = analysis["field_transitions"]
        assert all("new_fields" in transition for transition in transitions)
        assert all("dropped_fields" in transition for transition in transitions)
    
    @pytest.mark.asyncio
    async def test_collaboration_patterns_analysis(self, semantic_analysis_service):
        """Test collaboration patterns analysis."""
        papers = [
            SemanticScholarPaper(
                paper_id="paper_1",
                year=2020,
                authors=[
                    {"authorId": "author_1", "name": "John Doe"}
                ]
            ),
            SemanticScholarPaper(
                paper_id="paper_2",
                year=2021,
                authors=[
                    {"authorId": "author_1", "name": "John Doe"},
                    {"authorId": "author_2", "name": "Jane Smith"}
                ]
            ),
            SemanticScholarPaper(
                paper_id="paper_3",
                year=2022,
                authors=[
                    {"authorId": "author_1", "name": "John Doe"},
                    {"authorId": "author_2", "name": "Jane Smith"},
                    {"authorId": "author_3", "name": "Bob Johnson"}
                ]
            )
        ]
        
        analysis = semantic_analysis_service._analyze_collaboration_patterns(papers)
        
        assert analysis["total_unique_collaborators"] == 2  # Excluding main author
        assert analysis["avg_collaborators_per_paper"] > 0
        assert analysis["avg_authors_per_paper"] == 2.0  # Average of 1, 2, 3
        assert "collaboration_timeline" in analysis
        assert "collaboration_growth" in analysis
        
        # Collaboration should be growing
        collaboration_counts = analysis["collaboration_growth"]
        assert collaboration_counts == [0, 1, 2]  # Excluding main author
    
    @pytest.mark.asyncio
    async def test_venue_diversity_analysis(self, semantic_analysis_service):
        """Test venue diversity analysis."""
        papers = [
            SemanticScholarPaper(
                paper_id="paper_1",
                venue={"name": "Journal A", "type": "journal"}
            ),
            SemanticScholarPaper(
                paper_id="paper_2", 
                venue={"name": "Conference B", "type": "conference"}
            ),
            SemanticScholarPaper(
                paper_id="paper_3",
                venue={"name": "Journal A", "type": "journal"}  # Repeat venue
            )
        ]
        
        analysis = semantic_analysis_service._analyze_venue_diversity(papers)
        
        assert analysis["total_venues"] == 2  # Journal A and Conference B
        assert analysis["venue_diversity_score"] == 2/3  # 2 unique venues / 3 papers
        assert "venue_distribution" in analysis
        assert "venue_type_distribution" in analysis
        
        # Check venue counts
        assert analysis["venue_distribution"]["Journal A"] == 2
        assert analysis["venue_distribution"]["Conference B"] == 1
        assert analysis["venue_type_distribution"]["journal"] == 2
        assert analysis["venue_type_distribution"]["conference"] == 1
    
    @pytest.mark.asyncio
    async def test_get_metrics(self, semantic_analysis_service):
        """Test metrics collection."""
        with patch.object(semantic_analysis_service.semantic_scholar_client, 'get_metrics') as mock_s2_metrics:
            with patch.object(semantic_analysis_service.openalex_client, 'get_metrics') as mock_oa_metrics:
                mock_s2_metrics.return_value = {"requests": 100, "cache_hits": 50}
                mock_oa_metrics.return_value = {"requests": 200, "cache_hits": 100}
                
                metrics = semantic_analysis_service.get_metrics()
                
                assert "clients_initialized" in metrics
                assert "semantic_scholar" in metrics
                assert "openalex" in metrics
                assert metrics["clients_initialized"] is True


@pytest.mark.asyncio
async def test_service_factory_function():
    """Test service factory function."""
    from app.services.semantic_analysis import get_semantic_analysis_service
    
    service1 = await get_semantic_analysis_service()
    service2 = await get_semantic_analysis_service()
    
    # Should return same instance
    assert service1 is service2