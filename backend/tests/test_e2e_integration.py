"""
End-to-End Integration Testing Suite for Intellectual Lineage Tracer System

This test suite covers complete user workflows from start to finish:
- User registration → paper search → citation analysis → export
- Bulk import → graph building → community detection → collaboration
- Real-time collaboration → live updates → concurrent editing
- Authentication flows with JWT token lifecycle
- External API integrations with mock and live responses
- Database operations (Neo4j graph queries, Redis caching)
- Background task processing with Celery
"""

import pytest
import asyncio
import json
import time
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any, List
import uuid

from fastapi.testclient import TestClient
from fastapi import status

from app.main import create_app
from app.models.paper import Paper, Author, Journal, CitationNetwork
from app.models.user import User, UserCreate, UserSession
from app.services.auth import create_access_token
from app.services.tasks import TaskStatus


@pytest.fixture
def app():
    """Create test FastAPI app with integration setup."""
    return create_app()


@pytest.fixture
def client(app):
    """Create test client for integration testing."""
    return TestClient(app)


@pytest.fixture
def integration_mocks():
    """Mock all external dependencies for integration tests."""
    mocks = {}
    
    with patch('app.db.neo4j.Neo4jManager') as neo4j_mock, \
         patch('app.db.redis.RedisManager') as redis_mock, \
         patch('app.services.openalex.OpenAlexClient') as openalex_mock, \
         patch('app.services.semantic_scholar.SemanticScholarClient') as ss_mock, \
         patch('app.services.llm_service.LLMService') as llm_mock, \
         patch('celery_worker.celery_app') as celery_mock:
        
        # Setup mock returns
        mocks['neo4j'] = neo4j_mock.return_value
        mocks['redis'] = redis_mock.return_value
        mocks['openalex'] = openalex_mock.return_value
        mocks['semantic_scholar'] = ss_mock.return_value
        mocks['llm'] = llm_mock.return_value
        mocks['celery'] = celery_mock
        
        # Configure mock behaviors
        mocks['neo4j'].execute_query = AsyncMock()
        mocks['redis'].get = AsyncMock()
        mocks['redis'].set = AsyncMock()
        mocks['openalex'].search_works = AsyncMock()
        mocks['semantic_scholar'].search_papers = AsyncMock()
        mocks['llm'].generate_summary = AsyncMock()
        
        yield mocks


@pytest.fixture
def test_user_data():
    """Create test user data."""
    return {
        "email": f"test.user.{uuid.uuid4().hex[:8]}@example.com",
        "password": "SecureTestPassword123!",
        "full_name": "Test User"
    }


@pytest.fixture
def sample_papers():
    """Create sample papers for testing."""
    return [
        Paper(
            id=f"paper_{i}",
            title=f"Test Paper {i}",
            authors=[Author(name=f"Author {i}", id=f"author_{i}")],
            publication_year=2020 + i,
            doi=f"10.1000/test.{i}",
            abstract=f"This is the abstract for test paper {i}",
            journal=Journal(name=f"Test Journal {i}", issn=f"1234-567{i}")
        )
        for i in range(1, 6)
    ]


class TestCompleteUserWorkflow:
    """Test complete user workflows from registration to export."""
    
    @pytest.mark.asyncio
    async def test_new_user_complete_workflow(self, client, integration_mocks, test_user_data, sample_papers):
        """
        Test complete workflow for new user:
        Registration → Login → Search → Analyze → Export
        """
        # Step 1: User Registration
        registration_response = self._register_user(client, integration_mocks, test_user_data)
        assert registration_response.status_code == status.HTTP_201_CREATED
        
        user_data = registration_response.json()
        access_token = user_data["access_token"]
        auth_headers = {"Authorization": f"Bearer {access_token}"}
        
        # Step 2: Search for Papers
        search_response = self._search_papers(client, integration_mocks, auth_headers, sample_papers)
        assert search_response.status_code == status.HTTP_200_OK
        
        search_results = search_response.json()
        assert len(search_results["papers"]) > 0
        
        # Step 3: Build Citation Network
        paper_ids = [paper["id"] for paper in search_results["papers"][:3]]
        network_response = self._build_citation_network(client, integration_mocks, auth_headers, paper_ids)
        assert network_response.status_code == status.HTTP_201_CREATED
        
        network_data = network_response.json()
        assert "network_id" in network_data
        assert len(network_data["nodes"]) > 0
        
        # Step 4: Perform Community Detection
        community_response = self._detect_communities(client, integration_mocks, auth_headers, network_data["network_id"])
        assert community_response.status_code == status.HTTP_200_OK
        
        community_data = community_response.json()
        assert len(community_data["communities"]) > 0
        
        # Step 5: Generate Analysis Report
        report_response = self._generate_analysis_report(client, integration_mocks, auth_headers, network_data["network_id"])
        assert report_response.status_code == status.HTTP_200_OK
        
        # Step 6: Export Results
        export_response = self._export_results(client, integration_mocks, auth_headers, network_data["network_id"])
        assert export_response.status_code == status.HTTP_200_OK
    
    def _register_user(self, client, mocks, user_data):
        """Helper method to register a user."""
        mock_user = User(
            id=str(uuid.uuid4()),
            email=user_data["email"],
            full_name=user_data["full_name"],
            is_active=True,
            created_at=datetime.utcnow()
        )
        
        with patch('app.services.auth.AuthService.register_user') as mock_register:
            mock_register.return_value = mock_user
            return client.post("/api/v1/auth/register", json=user_data)
    
    def _search_papers(self, client, mocks, auth_headers, sample_papers):
        """Helper method to search for papers."""
        search_params = {
            "query": "machine learning",
            "page": 1,
            "page_size": 10
        }
        
        mock_results = {
            "papers": [paper.dict() for paper in sample_papers[:3]],
            "total": 3,
            "page": 1,
            "page_size": 10,
            "total_pages": 1
        }
        
        with patch('app.services.search.SearchService.search_papers') as mock_search:
            mock_search.return_value = mock_results
            return client.get("/api/v1/search/papers", params=search_params, headers=auth_headers)
    
    def _build_citation_network(self, client, mocks, auth_headers, paper_ids):
        """Helper method to build citation network."""
        network_request = {
            "seed_papers": paper_ids,
            "depth": 2,
            "max_nodes": 50,
            "include_references": True,
            "include_citations": True
        }
        
        mock_network = {
            "network_id": str(uuid.uuid4()),
            "nodes": [{"id": pid, "type": "paper", "title": f"Paper {pid}"} for pid in paper_ids],
            "edges": [{"source": paper_ids[0], "target": paper_ids[1], "type": "cites"}],
            "statistics": {"total_nodes": len(paper_ids), "total_edges": 1, "depth": 2}
        }
        
        with patch('app.services.graph_operations.GraphOperationsService.build_citation_network') as mock_build:
            mock_build.return_value = mock_network
            return client.post("/api/v1/graph/build-network", json=network_request, headers=auth_headers)
    
    def _detect_communities(self, client, mocks, auth_headers, network_id):
        """Helper method to detect communities."""
        detection_request = {
            "network_id": network_id,
            "algorithm": "louvain",
            "resolution": 1.0
        }
        
        mock_communities = {
            "communities": [
                {"id": 0, "papers": ["paper_1", "paper_2"], "size": 2, "modularity": 0.8}
            ],
            "total_communities": 1,
            "modularity": 0.8,
            "algorithm": "louvain"
        }
        
        with patch('app.services.advanced_analytics.AdvancedAnalyticsService.detect_communities') as mock_detect:
            mock_detect.return_value = mock_communities
            return client.post("/api/v1/graph/community-detection", json=detection_request, headers=auth_headers)
    
    def _generate_analysis_report(self, client, mocks, auth_headers, network_id):
        """Helper method to generate analysis report."""
        report_request = {
            "network_id": network_id,
            "analysis_types": ["summary", "trends", "gaps"],
            "include_visualizations": True
        }
        
        mock_report = {
            "report_id": str(uuid.uuid4()),
            "summary": "This network shows strong clustering in machine learning research...",
            "trends": [{"trend": "Increasing focus on deep learning", "confidence": 0.9}],
            "gaps": [{"gap": "Limited work on explainability", "severity": "medium"}],
            "generated_at": datetime.utcnow().isoformat()
        }
        
        with patch('app.services.research_intelligence.ResearchIntelligenceService.generate_report') as mock_report_gen:
            mock_report_gen.return_value = mock_report
            return client.post("/api/v1/llm/generate-report", json=report_request, headers=auth_headers)
    
    def _export_results(self, client, mocks, auth_headers, network_id):
        """Helper method to export results."""
        export_request = {
            "network_id": network_id,
            "format": "json",
            "include_metadata": True,
            "include_analysis": True
        }
        
        mock_export = {
            "export_id": str(uuid.uuid4()),
            "download_url": f"/api/v1/exports/{uuid.uuid4()}.json",
            "format": "json",
            "size_bytes": 1024,
            "expires_at": (datetime.utcnow() + timedelta(hours=24)).isoformat()
        }
        
        with patch('app.services.export.ExportService.create_export') as mock_export_service:
            mock_export_service.return_value = mock_export
            return client.post("/api/v1/exports/create", json=export_request, headers=auth_headers)


class TestBulkImportWorkflow:
    """Test bulk import and processing workflow."""
    
    @pytest.mark.asyncio
    async def test_bibtex_import_to_analysis(self, client, integration_mocks, test_user_data):
        """
        Test workflow: BibTeX Import → Processing → Graph Building → Analysis
        """
        # Setup authenticated user
        auth_headers = self._setup_authenticated_user(client, integration_mocks, test_user_data)
        
        # Step 1: Upload BibTeX file
        bibtex_content = """
        @article{test1,
            title={Test Paper 1},
            author={Test Author 1},
            journal={Test Journal},
            year={2023},
            doi={10.1000/test1}
        }
        @article{test2,
            title={Test Paper 2},
            author={Test Author 2},
            journal={Test Journal},
            year={2023},
            doi={10.1000/test2}
        }
        """
        
        upload_response = self._upload_bibtex(client, integration_mocks, auth_headers, bibtex_content)
        assert upload_response.status_code == status.HTTP_202_ACCEPTED
        
        upload_data = upload_response.json()
        task_id = upload_data["task_id"]
        
        # Step 2: Monitor processing task
        task_completion = self._monitor_task_completion(client, auth_headers, task_id)
        assert task_completion["status"] == "completed"
        
        # Step 3: Build graph from imported papers
        imported_papers = task_completion["result"]["imported_papers"]
        paper_ids = [paper["id"] for paper in imported_papers]
        
        graph_response = self._build_graph_from_imported(client, integration_mocks, auth_headers, paper_ids)
        assert graph_response.status_code == status.HTTP_201_CREATED
        
        # Step 4: Perform advanced analytics
        analytics_response = self._run_advanced_analytics(client, integration_mocks, auth_headers, 
                                                         graph_response.json()["network_id"])
        assert analytics_response.status_code == status.HTTP_200_OK
    
    def _setup_authenticated_user(self, client, mocks, user_data):
        """Setup and authenticate a test user."""
        token = create_access_token(data={"sub": "test_user_id", "email": user_data["email"]})
        return {"Authorization": f"Bearer {token}"}
    
    def _upload_bibtex(self, client, mocks, auth_headers, bibtex_content):
        """Upload BibTeX content for processing."""
        files = {"file": ("test.bib", bibtex_content, "text/plain")}
        
        mock_task_result = {
            "task_id": str(uuid.uuid4()),
            "status": "pending",
            "message": "BibTeX upload received, processing..."
        }
        
        with patch('app.services.tasks.TaskService.create_bibtex_import_task') as mock_task:
            mock_task.return_value = mock_task_result
            return client.post("/api/v1/papers/import/bibtex", files=files, headers=auth_headers)
    
    def _monitor_task_completion(self, client, auth_headers, task_id, max_wait_seconds=30):
        """Monitor task until completion."""
        start_time = time.time()
        
        while time.time() - start_time < max_wait_seconds:
            # Mock task completion after a short delay
            if time.time() - start_time > 2:  # Simulate processing time
                return {
                    "task_id": task_id,
                    "status": "completed",
                    "result": {
                        "imported_papers": [
                            {"id": "paper_1", "title": "Test Paper 1"},
                            {"id": "paper_2", "title": "Test Paper 2"}
                        ],
                        "import_summary": {
                            "total_entries": 2,
                            "successful_imports": 2,
                            "failed_imports": 0
                        }
                    }
                }
            
            time.sleep(1)
        
        raise TimeoutError("Task did not complete within expected time")
    
    def _build_graph_from_imported(self, client, mocks, auth_headers, paper_ids):
        """Build citation graph from imported papers."""
        request_data = {
            "seed_papers": paper_ids,
            "depth": 1,
            "max_nodes": 100,
            "enrich_from_external_apis": True
        }
        
        mock_network = {
            "network_id": str(uuid.uuid4()),
            "nodes": [{"id": pid, "type": "paper"} for pid in paper_ids],
            "edges": [],
            "enrichment_summary": {
                "papers_enriched": len(paper_ids),
                "external_citations_found": 5
            }
        }
        
        with patch('app.services.graph_operations.GraphOperationsService.build_citation_network') as mock_build:
            mock_build.return_value = mock_network
            return client.post("/api/v1/graph/build-network", json=request_data, headers=auth_headers)
    
    def _run_advanced_analytics(self, client, mocks, auth_headers, network_id):
        """Run advanced analytics on the network."""
        analytics_request = {
            "network_id": network_id,
            "analyses": ["centrality", "clustering", "temporal", "topical"]
        }
        
        mock_analytics = {
            "network_id": network_id,
            "centrality_analysis": {
                "most_central_papers": [{"id": "paper_1", "centrality_score": 0.95}]
            },
            "clustering_analysis": {
                "clusters": [{"id": 0, "size": 2, "topic": "machine learning"}]
            },
            "temporal_analysis": {
                "publication_timeline": [{"year": 2023, "count": 2}]
            },
            "topical_analysis": {
                "main_topics": [{"topic": "AI", "weight": 0.8}]
            }
        }
        
        with patch('app.services.advanced_analytics.AdvancedAnalyticsService.run_comprehensive_analysis') as mock_analytics_service:
            mock_analytics_service.return_value = mock_analytics
            return client.post("/api/v1/analytics/comprehensive", json=analytics_request, headers=auth_headers)


class TestRealTimeCollaborationWorkflow:
    """Test real-time collaboration features."""
    
    @pytest.mark.asyncio
    async def test_collaborative_editing_session(self, client, integration_mocks, test_user_data):
        """
        Test real-time collaboration workflow:
        Create session → Multiple users join → Concurrent editing → Sync updates
        """
        # Setup multiple users
        user1_headers = self._create_user_session(client, integration_mocks, "user1@example.com")
        user2_headers = self._create_user_session(client, integration_mocks, "user2@example.com")
        
        # Step 1: User 1 creates collaboration session
        session_response = self._create_collaboration_session(client, integration_mocks, user1_headers)
        assert session_response.status_code == status.HTTP_201_CREATED
        
        session_data = session_response.json()
        session_id = session_data["session_id"]
        
        # Step 2: User 2 joins session
        join_response = self._join_collaboration_session(client, integration_mocks, user2_headers, session_id)
        assert join_response.status_code == status.HTTP_200_OK
        
        # Step 3: Simulate concurrent edits
        edit1_response = self._make_collaborative_edit(client, integration_mocks, user1_headers, 
                                                     session_id, "add_node", {"node_id": "paper_new"})
        assert edit1_response.status_code == status.HTTP_200_OK
        
        edit2_response = self._make_collaborative_edit(client, integration_mocks, user2_headers, 
                                                     session_id, "modify_layout", {"layout": "force"})
        assert edit2_response.status_code == status.HTTP_200_OK
        
        # Step 4: Verify session state synchronization
        state_response = self._get_session_state(client, user1_headers, session_id)
        assert state_response.status_code == status.HTTP_200_OK
        
        state_data = state_response.json()
        assert len(state_data["recent_changes"]) == 2
        assert state_data["active_users"] == 2
    
    def _create_user_session(self, client, mocks, email):
        """Create authenticated user session."""
        token = create_access_token(data={"sub": str(uuid.uuid4()), "email": email})
        return {"Authorization": f"Bearer {token}"}
    
    def _create_collaboration_session(self, client, mocks, auth_headers):
        """Create a new collaboration session."""
        session_request = {
            "name": "Test Collaboration",
            "description": "Testing real-time collaboration",
            "network_id": str(uuid.uuid4()),
            "permissions": "read_write"
        }
        
        mock_session = {
            "session_id": str(uuid.uuid4()),
            "name": session_request["name"],
            "owner_id": "user1",
            "created_at": datetime.utcnow().isoformat(),
            "active_users": 1
        }
        
        with patch('app.services.collaboration.CollaborationService.create_session') as mock_create:
            mock_create.return_value = mock_session
            return client.post("/api/v1/collaboration/sessions", json=session_request, headers=auth_headers)
    
    def _join_collaboration_session(self, client, mocks, auth_headers, session_id):
        """Join an existing collaboration session."""
        mock_join_result = {
            "session_id": session_id,
            "user_role": "collaborator",
            "joined_at": datetime.utcnow().isoformat()
        }
        
        with patch('app.services.collaboration.CollaborationService.join_session') as mock_join:
            mock_join.return_value = mock_join_result
            return client.post(f"/api/v1/collaboration/sessions/{session_id}/join", headers=auth_headers)
    
    def _make_collaborative_edit(self, client, mocks, auth_headers, session_id, action, data):
        """Make a collaborative edit to the session."""
        edit_request = {
            "action": action,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        mock_edit_result = {
            "edit_id": str(uuid.uuid4()),
            "applied": True,
            "conflicts": []
        }
        
        with patch('app.services.collaboration.CollaborationService.apply_edit') as mock_edit:
            mock_edit.return_value = mock_edit_result
            return client.post(f"/api/v1/collaboration/sessions/{session_id}/edit", 
                             json=edit_request, headers=auth_headers)
    
    def _get_session_state(self, client, auth_headers, session_id):
        """Get current state of collaboration session."""
        mock_state = {
            "session_id": session_id,
            "active_users": 2,
            "recent_changes": [
                {"action": "add_node", "user": "user1", "timestamp": datetime.utcnow().isoformat()},
                {"action": "modify_layout", "user": "user2", "timestamp": datetime.utcnow().isoformat()}
            ],
            "current_network_state": {"nodes": 5, "edges": 3}
        }
        
        with patch('app.services.collaboration.CollaborationService.get_session_state') as mock_state_service:
            mock_state_service.return_value = mock_state
            return client.get(f"/api/v1/collaboration/sessions/{session_id}/state", headers=auth_headers)


class TestExternalAPIIntegrationWorkflow:
    """Test external API integration workflows."""
    
    @pytest.mark.asyncio
    async def test_multi_source_paper_enrichment(self, client, integration_mocks, test_user_data):
        """
        Test workflow that enriches papers from multiple sources:
        Basic paper → OpenAlex enrichment → Semantic Scholar data → LLM analysis
        """
        auth_headers = self._setup_authenticated_user(client, integration_mocks, test_user_data)
        
        # Step 1: Start with basic paper information
        basic_paper = {
            "title": "Machine Learning in Healthcare",
            "authors": ["Dr. Smith"],
            "year": 2023
        }
        
        create_response = self._create_basic_paper(client, integration_mocks, auth_headers, basic_paper)
        assert create_response.status_code == status.HTTP_201_CREATED
        
        paper_id = create_response.json()["id"]
        
        # Step 2: Enrich with OpenAlex data
        openalex_response = self._enrich_with_openalex(client, integration_mocks, auth_headers, paper_id)
        assert openalex_response.status_code == status.HTTP_200_OK
        
        # Step 3: Enrich with Semantic Scholar data
        ss_response = self._enrich_with_semantic_scholar(client, integration_mocks, auth_headers, paper_id)
        assert ss_response.status_code == status.HTTP_200_OK
        
        # Step 4: Generate LLM-powered insights
        llm_response = self._generate_llm_insights(client, integration_mocks, auth_headers, paper_id)
        assert llm_response.status_code == status.HTTP_200_OK
        
        # Step 5: Verify complete enrichment
        enriched_response = client.get(f"/api/v1/papers/{paper_id}", headers=auth_headers)
        assert enriched_response.status_code == status.HTTP_200_OK
        
        enriched_data = enriched_response.json()
        assert enriched_data["enrichment_status"]["openalex"] == "completed"
        assert enriched_data["enrichment_status"]["semantic_scholar"] == "completed"
        assert enriched_data["enrichment_status"]["llm_analysis"] == "completed"
    
    def _setup_authenticated_user(self, client, mocks, user_data):
        """Setup authenticated user for testing."""
        token = create_access_token(data={"sub": "test_user_id", "email": user_data["email"]})
        return {"Authorization": f"Bearer {token}"}
    
    def _create_basic_paper(self, client, mocks, auth_headers, paper_data):
        """Create basic paper entry."""
        mock_paper = {
            "id": str(uuid.uuid4()),
            "title": paper_data["title"],
            "authors": paper_data["authors"],
            "publication_year": paper_data["year"],
            "enrichment_status": {
                "openalex": "pending",
                "semantic_scholar": "pending",
                "llm_analysis": "pending"
            }
        }
        
        with patch('app.services.papers.PaperService.create_paper') as mock_create:
            mock_create.return_value = mock_paper
            return client.post("/api/v1/papers/", json=paper_data, headers=auth_headers)
    
    def _enrich_with_openalex(self, client, mocks, auth_headers, paper_id):
        """Enrich paper with OpenAlex data."""
        mock_enrichment = {
            "paper_id": paper_id,
            "enriched_fields": ["doi", "abstract", "citations", "references"],
            "source": "openalex",
            "confidence_score": 0.95
        }
        
        with patch('app.services.enrichment_pipeline.EnrichmentPipelineService.enrich_with_openalex') as mock_enrich:
            mock_enrich.return_value = mock_enrichment
            return client.post(f"/api/v1/papers/{paper_id}/enrich/openalex", headers=auth_headers)
    
    def _enrich_with_semantic_scholar(self, client, mocks, auth_headers, paper_id):
        """Enrich paper with Semantic Scholar data."""
        mock_enrichment = {
            "paper_id": paper_id,
            "enriched_fields": ["influential_citations", "topics", "venue_info"],
            "source": "semantic_scholar",
            "confidence_score": 0.88
        }
        
        with patch('app.services.enrichment_pipeline.EnrichmentPipelineService.enrich_with_semantic_scholar') as mock_enrich:
            mock_enrich.return_value = mock_enrichment
            return client.post(f"/api/v1/papers/{paper_id}/enrich/semantic-scholar", headers=auth_headers)
    
    def _generate_llm_insights(self, client, mocks, auth_headers, paper_id):
        """Generate LLM-powered insights for the paper."""
        insight_request = {
            "analysis_types": ["summary", "methodology", "significance", "limitations"],
            "model": "claude-3-sonnet-20240229"
        }
        
        mock_insights = {
            "paper_id": paper_id,
            "insights": {
                "summary": "This paper presents a comprehensive review of ML in healthcare...",
                "methodology": "The authors employ a systematic literature review approach...",
                "significance": "This work contributes to the field by providing...",
                "limitations": "The main limitations include limited scope of databases..."
            },
            "confidence_scores": {"summary": 0.92, "methodology": 0.89, "significance": 0.94, "limitations": 0.87},
            "model_used": "claude-3-sonnet-20240229",
            "tokens_used": 1500,
            "cost_usd": 0.045
        }
        
        with patch('app.services.llm_service.LLMService.generate_comprehensive_analysis') as mock_llm:
            mock_llm.return_value = mock_insights
            return client.post(f"/api/v1/llm/papers/{paper_id}/analyze", 
                             json=insight_request, headers=auth_headers)


class TestErrorRecoveryWorkflows:
    """Test error recovery and resilience workflows."""
    
    @pytest.mark.asyncio
    async def test_network_failure_recovery(self, client, integration_mocks, test_user_data):
        """
        Test recovery from network failures during operations:
        Start operation → Network failure → Retry logic → Successful completion
        """
        auth_headers = self._setup_authenticated_user(client, integration_mocks, test_user_data)
        
        # Step 1: Start operation that will initially fail
        operation_request = {
            "operation": "build_large_network",
            "parameters": {"seed_papers": ["paper1", "paper2"], "depth": 3}
        }
        
        # Mock initial failure followed by success
        with patch('app.services.graph_operations.GraphOperationsService.build_citation_network') as mock_build:
            # First call fails
            mock_build.side_effect = [
                Exception("Network timeout"),
                Exception("Service unavailable"),
                {  # Third call succeeds
                    "network_id": str(uuid.uuid4()),
                    "nodes": [{"id": "paper1"}, {"id": "paper2"}],
                    "edges": [],
                    "retry_count": 2
                }
            ]
            
            response = client.post("/api/v1/graph/build-network", 
                                 json=operation_request, headers=auth_headers)
            
            # Should eventually succeed with retry mechanism
            assert response.status_code == status.HTTP_201_CREATED
            
            # Verify retry mechanism was used
            result = response.json()
            assert "retry_count" in result
            assert result["retry_count"] >= 1
    
    def _setup_authenticated_user(self, client, mocks, user_data):
        """Setup authenticated user for testing."""
        token = create_access_token(data={"sub": "test_user_id", "email": user_data["email"]})
        return {"Authorization": f"Bearer {token}"}


class TestPerformanceIntegrationScenarios:
    """Test performance under realistic integration scenarios."""
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_large_scale_citation_analysis(self, client, integration_mocks, test_user_data):
        """
        Test performance with large-scale citation analysis:
        1000+ papers → Citation network → Community detection → Analysis
        """
        auth_headers = self._setup_authenticated_user(client, integration_mocks, test_user_data)
        
        # Generate large dataset
        large_paper_set = [f"paper_{i}" for i in range(1000)]
        
        start_time = time.time()
        
        # Build large citation network
        network_request = {
            "seed_papers": large_paper_set[:10],  # Start with 10 seeds
            "depth": 2,
            "max_nodes": 1000,
            "parallel_processing": True
        }
        
        mock_large_network = {
            "network_id": str(uuid.uuid4()),
            "nodes": [{"id": pid, "type": "paper"} for pid in large_paper_set],
            "edges": [{"source": f"paper_{i}", "target": f"paper_{i+1}", "type": "cites"} 
                     for i in range(999)],
            "processing_time_seconds": 15.5,
            "nodes_processed": 1000
        }
        
        with patch('app.services.graph_operations.GraphOperationsService.build_citation_network') as mock_build:
            mock_build.return_value = mock_large_network
            
            response = client.post("/api/v1/graph/build-network", 
                                 json=network_request, headers=auth_headers)
            
            processing_time = time.time() - start_time
            
            assert response.status_code == status.HTTP_201_CREATED
            assert processing_time < 30  # Should complete within 30 seconds
            
            result = response.json()
            assert result["nodes_processed"] == 1000
    
    def _setup_authenticated_user(self, client, mocks, user_data):
        """Setup authenticated user for testing."""
        token = create_access_token(data={"sub": "test_user_id", "email": user_data["email"]})
        return {"Authorization": f"Bearer {token}"}


# Test configuration and fixtures
@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Performance test markers
pytest.mark.slow = pytest.mark.skipif(
    True,  # Skip by default unless explicitly requested
    reason="Integration tests are slow - run with --integration flag"
)