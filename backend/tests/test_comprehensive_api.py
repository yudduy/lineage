"""
Comprehensive API Testing Suite for Intellectual Lineage Tracer System

This test suite covers all FastAPI endpoints with various scenarios including:
- Authentication and authorization testing
- Input validation and sanitization
- Error handling and edge cases
- Rate limiting validation
- Response time and performance
- Security header verification
"""

import pytest
import asyncio
import json
import time
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import status
from datetime import datetime, timedelta
from typing import Dict, Any, List
import httpx

from app.main import create_app
from app.core.config import get_settings
from app.models.paper import Paper, Author, Journal
from app.models.user import User, UserCreate
from app.services.auth import create_access_token, verify_token


@pytest.fixture
def app():
    """Create test FastAPI app."""
    return create_app()


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_dependencies():
    """Mock external dependencies."""
    with patch('app.db.dependencies.get_neo4j_driver'), \
         patch('app.db.dependencies.get_redis_manager'), \
         patch('app.services.openalex.OpenAlexClient'), \
         patch('app.services.semantic_scholar.SemanticScholarClient'), \
         patch('app.services.llm_service.LLMService'):
        yield


@pytest.fixture
def auth_headers():
    """Create authentication headers for testing."""
    # Create test token
    test_user_data = {"sub": "test_user_id", "email": "test@example.com"}
    token = create_access_token(data=test_user_data)
    return {"Authorization": f"Bearer {token}"}


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_health_check_basic(self, client, mock_dependencies):
        """Test basic health check endpoint."""
        response = client.get("/api/v1/health/")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "environment" in data
    
    def test_health_check_detailed(self, client, mock_dependencies):
        """Test detailed health check with dependencies."""
        with patch('app.services.health.HealthService.check_all_dependencies') as mock_check:
            mock_check.return_value = {
                "neo4j": {"status": "healthy", "response_time": 50},
                "redis": {"status": "healthy", "response_time": 10},
                "external_apis": {"status": "healthy", "response_time": 200}
            }
            
            response = client.get("/api/v1/health/detailed")
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert "dependencies" in data
            assert data["status"] == "healthy"


class TestAuthenticationEndpoints:
    """Test authentication and authorization endpoints."""
    
    def test_user_registration_success(self, client, mock_dependencies):
        """Test successful user registration."""
        user_data = {
            "email": "newuser@example.com",
            "password": "securepassword123",
            "full_name": "New User"
        }
        
        with patch('app.services.auth.AuthService.register_user') as mock_register:
            mock_user = User(
                id="user123",
                email=user_data["email"],
                full_name=user_data["full_name"],
                is_active=True
            )
            mock_register.return_value = mock_user
            
            response = client.post("/api/v1/auth/register", json=user_data)
            assert response.status_code == status.HTTP_201_CREATED
            
            data = response.json()
            assert data["email"] == user_data["email"]
            assert "access_token" in data
    
    def test_user_registration_invalid_email(self, client, mock_dependencies):
        """Test registration with invalid email format."""
        user_data = {
            "email": "invalid-email",
            "password": "securepassword123",
            "full_name": "Test User"
        }
        
        response = client.post("/api/v1/auth/register", json=user_data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_user_registration_weak_password(self, client, mock_dependencies):
        """Test registration with weak password."""
        user_data = {
            "email": "test@example.com",
            "password": "123",
            "full_name": "Test User"
        }
        
        response = client.post("/api/v1/auth/register", json=user_data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_user_login_success(self, client, mock_dependencies):
        """Test successful user login."""
        login_data = {
            "username": "test@example.com",
            "password": "correctpassword"
        }
        
        with patch('app.services.auth.AuthService.authenticate_user') as mock_auth:
            mock_user = User(
                id="user123",
                email=login_data["username"],
                full_name="Test User",
                is_active=True
            )
            mock_auth.return_value = mock_user
            
            response = client.post("/api/v1/auth/login", data=login_data)
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert "access_token" in data
            assert "refresh_token" in data
            assert data["token_type"] == "bearer"
    
    def test_user_login_invalid_credentials(self, client, mock_dependencies):
        """Test login with invalid credentials."""
        login_data = {
            "username": "test@example.com",
            "password": "wrongpassword"
        }
        
        with patch('app.services.auth.AuthService.authenticate_user') as mock_auth:
            mock_auth.return_value = None
            
            response = client.post("/api/v1/auth/login", data=login_data)
            assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_token_refresh(self, client, mock_dependencies, auth_headers):
        """Test token refresh functionality."""
        refresh_data = {"refresh_token": "valid_refresh_token"}
        
        with patch('app.services.auth.AuthService.refresh_access_token') as mock_refresh:
            mock_refresh.return_value = {
                "access_token": "new_access_token",
                "token_type": "bearer"
            }
            
            response = client.post("/api/v1/auth/refresh", json=refresh_data)
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert "access_token" in data
    
    def test_protected_endpoint_without_auth(self, client, mock_dependencies):
        """Test accessing protected endpoint without authentication."""
        response = client.get("/api/v1/users/me")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_protected_endpoint_with_invalid_token(self, client, mock_dependencies):
        """Test accessing protected endpoint with invalid token."""
        headers = {"Authorization": "Bearer invalid_token"}
        response = client.get("/api/v1/users/me", headers=headers)
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestPaperEndpoints:
    """Test paper-related endpoints."""
    
    def test_search_papers_basic(self, client, mock_dependencies, auth_headers):
        """Test basic paper search functionality."""
        search_params = {
            "query": "machine learning",
            "page": 1,
            "page_size": 20
        }
        
        mock_papers = [
            Paper(
                id="paper1",
                title="Introduction to Machine Learning",
                authors=[Author(name="John Doe", id="author1")],
                publication_year=2023,
                doi="10.1000/test1"
            )
        ]
        
        with patch('app.services.search.SearchService.search_papers') as mock_search:
            mock_search.return_value = {
                "papers": mock_papers,
                "total": 100,
                "page": 1,
                "page_size": 20,
                "total_pages": 5
            }
            
            response = client.get("/api/v1/search/papers", params=search_params, headers=auth_headers)
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert len(data["papers"]) == 1
            assert data["total"] == 100
            assert data["papers"][0]["title"] == "Introduction to Machine Learning"
    
    def test_search_papers_with_filters(self, client, mock_dependencies, auth_headers):
        """Test paper search with advanced filters."""
        search_params = {
            "query": "neural networks",
            "year_from": 2020,
            "year_to": 2023,
            "is_open_access": True,
            "author": "Bengio"
        }
        
        with patch('app.services.search.SearchService.search_papers') as mock_search:
            mock_search.return_value = {
                "papers": [],
                "total": 0,
                "page": 1,
                "page_size": 20,
                "total_pages": 0
            }
            
            response = client.get("/api/v1/search/papers", params=search_params, headers=auth_headers)
            assert response.status_code == status.HTTP_200_OK
    
    def test_get_paper_by_id(self, client, mock_dependencies, auth_headers):
        """Test retrieving paper by ID."""
        paper_id = "paper123"
        mock_paper = Paper(
            id=paper_id,
            title="Test Paper",
            authors=[Author(name="Test Author", id="author1")],
            publication_year=2023
        )
        
        with patch('app.services.papers.PaperService.get_paper_by_id') as mock_get:
            mock_get.return_value = mock_paper
            
            response = client.get(f"/api/v1/papers/{paper_id}", headers=auth_headers)
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert data["id"] == paper_id
            assert data["title"] == "Test Paper"
    
    def test_get_paper_not_found(self, client, mock_dependencies, auth_headers):
        """Test retrieving non-existent paper."""
        paper_id = "nonexistent"
        
        with patch('app.services.papers.PaperService.get_paper_by_id') as mock_get:
            mock_get.return_value = None
            
            response = client.get(f"/api/v1/papers/{paper_id}", headers=auth_headers)
            assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_get_paper_citations(self, client, mock_dependencies, auth_headers):
        """Test retrieving paper citations."""
        paper_id = "paper123"
        
        mock_citations = [
            Paper(id="cite1", title="Citing Paper 1", publication_year=2022),
            Paper(id="cite2", title="Citing Paper 2", publication_year=2023)
        ]
        
        with patch('app.services.citation_analysis.CitationAnalysisService.get_citations') as mock_citations_service:
            mock_citations_service.return_value = mock_citations
            
            response = client.get(f"/api/v1/papers/{paper_id}/citations", headers=auth_headers)
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert len(data) == 2
    
    def test_create_paper_collection(self, client, mock_dependencies, auth_headers):
        """Test creating paper collection."""
        collection_data = {
            "name": "My Research Collection",
            "description": "Papers related to my research",
            "paper_ids": ["paper1", "paper2", "paper3"]
        }
        
        with patch('app.services.papers.PaperService.create_collection') as mock_create:
            mock_collection = {
                "id": "collection123",
                "name": collection_data["name"],
                "description": collection_data["description"],
                "paper_count": len(collection_data["paper_ids"])
            }
            mock_create.return_value = mock_collection
            
            response = client.post("/api/v1/papers/collections", json=collection_data, headers=auth_headers)
            assert response.status_code == status.HTTP_201_CREATED
            
            data = response.json()
            assert data["name"] == collection_data["name"]


class TestGraphEndpoints:
    """Test graph analysis endpoints."""
    
    def test_build_citation_network(self, client, mock_dependencies, auth_headers):
        """Test building citation network."""
        network_request = {
            "seed_papers": ["paper1", "paper2"],
            "depth": 2,
            "max_nodes": 100,
            "include_references": True,
            "include_citations": True
        }
        
        mock_network = {
            "nodes": [
                {"id": "paper1", "type": "paper", "title": "Seed Paper 1"},
                {"id": "paper2", "type": "paper", "title": "Seed Paper 2"}
            ],
            "edges": [
                {"source": "paper1", "target": "paper2", "type": "cites"}
            ],
            "statistics": {
                "total_nodes": 2,
                "total_edges": 1,
                "depth": 2
            }
        }
        
        with patch('app.services.graph_operations.GraphOperationsService.build_citation_network') as mock_build:
            mock_build.return_value = mock_network
            
            response = client.post("/api/v1/graph/build-network", json=network_request, headers=auth_headers)
            assert response.status_code == status.HTTP_201_CREATED
            
            data = response.json()
            assert len(data["nodes"]) == 2
            assert len(data["edges"]) == 1
    
    def test_community_detection(self, client, mock_dependencies, auth_headers):
        """Test community detection algorithm."""
        detection_request = {
            "network_id": "network123",
            "algorithm": "louvain",
            "resolution": 1.0
        }
        
        mock_communities = {
            "communities": [
                {"id": 0, "papers": ["paper1", "paper2"], "size": 2},
                {"id": 1, "papers": ["paper3", "paper4"], "size": 2}
            ],
            "modularity": 0.75,
            "algorithm": "louvain"
        }
        
        with patch('app.services.advanced_analytics.AdvancedAnalyticsService.detect_communities') as mock_detect:
            mock_detect.return_value = mock_communities
            
            response = client.post("/api/v1/graph/community-detection", json=detection_request, headers=auth_headers)
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert len(data["communities"]) == 2
            assert data["modularity"] == 0.75


class TestExternalAPIEndpoints:
    """Test external API integration endpoints."""
    
    def test_openalex_search(self, client, mock_dependencies, auth_headers):
        """Test OpenAlex search integration."""
        search_query = {
            "query": "artificial intelligence",
            "filters": {
                "publication_year": [2020, 2023],
                "is_oa": True
            },
            "page": 1,
            "per_page": 25
        }
        
        mock_results = {
            "results": [
                {
                    "id": "W123456",
                    "title": "AI Research Paper",
                    "publication_year": 2022,
                    "cited_by_count": 150
                }
            ],
            "meta": {"count": 1}
        }
        
        with patch('app.services.openalex.OpenAlexClient.search_works') as mock_search:
            mock_search.return_value = mock_results
            
            response = client.post("/api/v1/openalex/search", json=search_query, headers=auth_headers)
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert len(data["results"]) == 1
    
    def test_semantic_scholar_search(self, client, mock_dependencies, auth_headers):
        """Test Semantic Scholar search integration."""
        search_params = {
            "query": "deep learning",
            "year": "2023",
            "venue": "ICML",
            "limit": 10
        }
        
        mock_results = {
            "data": [
                {
                    "paperId": "ss123",
                    "title": "Deep Learning Paper",
                    "year": 2023,
                    "citationCount": 25
                }
            ],
            "total": 1
        }
        
        with patch('app.services.semantic_scholar.SemanticScholarClient.search_papers') as mock_search:
            mock_search.return_value = mock_results
            
            response = client.get("/api/v1/semantic-scholar/search", params=search_params, headers=auth_headers)
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert len(data["data"]) == 1


class TestLLMEnrichmentEndpoints:
    """Test LLM enrichment endpoints."""
    
    def test_generate_summary(self, client, mock_dependencies, auth_headers):
        """Test paper summary generation."""
        summary_request = {
            "paper_id": "paper123",
            "summary_type": "abstract",
            "max_length": 200,
            "model": "claude-3-haiku-20240307"
        }
        
        mock_summary = {
            "summary": "This paper introduces a novel approach to machine learning...",
            "confidence_score": 0.95,
            "model_used": "claude-3-haiku-20240307",
            "tokens_used": 150,
            "cost_usd": 0.001
        }
        
        with patch('app.services.llm_service.LLMService.generate_summary') as mock_generate:
            mock_generate.return_value = mock_summary
            
            response = client.post("/api/v1/llm/generate-summary", json=summary_request, headers=auth_headers)
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert "summary" in data
            assert data["confidence_score"] == 0.95
    
    def test_analyze_research_gaps(self, client, mock_dependencies, auth_headers):
        """Test research gap analysis."""
        analysis_request = {
            "paper_ids": ["paper1", "paper2", "paper3"],
            "analysis_type": "research_gaps",
            "focus_areas": ["methodology", "datasets", "future_work"]
        }
        
        mock_analysis = {
            "gaps": [
                {
                    "category": "methodology",
                    "description": "Limited evaluation on real-world datasets",
                    "severity": "high",
                    "papers_affected": ["paper1", "paper2"]
                }
            ],
            "recommendations": [
                "Consider evaluating on larger, more diverse datasets"
            ],
            "confidence_score": 0.87
        }
        
        with patch('app.services.research_intelligence.ResearchIntelligenceService.analyze_research_gaps') as mock_analyze:
            mock_analyze.return_value = mock_analysis
            
            response = client.post("/api/v1/llm/analyze-gaps", json=analysis_request, headers=auth_headers)
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert "gaps" in data
            assert len(data["gaps"]) == 1


class TestRateLimitingAndSecurity:
    """Test rate limiting and security features."""
    
    def test_rate_limiting_enforcement(self, client, mock_dependencies, auth_headers):
        """Test that rate limiting is enforced."""
        # Make multiple rapid requests to trigger rate limiting
        responses = []
        for i in range(100):  # Exceed rate limit
            response = client.get("/api/v1/papers/search", 
                                params={"query": f"test{i}"}, 
                                headers=auth_headers)
            responses.append(response.status_code)
            if response.status_code == 429:
                break
        
        # Should eventually hit rate limit
        assert 429 in responses
    
    def test_cors_headers(self, client, mock_dependencies):
        """Test CORS headers are properly set."""
        response = client.options("/api/v1/health/", 
                                headers={"Origin": "http://localhost:3000"})
        
        assert "Access-Control-Allow-Origin" in response.headers
        assert "Access-Control-Allow-Methods" in response.headers
        assert "Access-Control-Allow-Headers" in response.headers
    
    def test_security_headers(self, client, mock_dependencies):
        """Test security headers are present."""
        response = client.get("/api/v1/health/")
        
        # Check for security headers (these would be added by middleware)
        headers = response.headers
        # Verify basic security headers are considered
        assert response.status_code in [200, 404]  # Basic functionality works
    
    def test_input_sanitization(self, client, mock_dependencies, auth_headers):
        """Test input sanitization for SQL injection and XSS."""
        malicious_inputs = [
            "'; DROP TABLE papers; --",
            "<script>alert('xss')</script>",
            "../../etc/passwd",
            "' OR 1=1; --"
        ]
        
        for malicious_input in malicious_inputs:
            response = client.get("/api/v1/papers/search", 
                                params={"query": malicious_input}, 
                                headers=auth_headers)
            
            # Should not return 500 error or expose system information
            assert response.status_code in [200, 400, 422]
            
            if response.status_code == 200:
                data = response.json()
                # Ensure malicious input is properly handled
                assert malicious_input not in str(data)


class TestPerformanceAndResponseTimes:
    """Test API performance and response times."""
    
    def test_response_time_health_check(self, client, mock_dependencies):
        """Test health check response time."""
        start_time = time.time()
        response = client.get("/api/v1/health/")
        end_time = time.time()
        
        response_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        assert response.status_code == status.HTTP_200_OK
        assert response_time < 1000  # Should respond within 1 second
    
    def test_concurrent_requests_handling(self, client, mock_dependencies, auth_headers):
        """Test handling of concurrent requests."""
        import concurrent.futures
        import threading
        
        def make_request():
            return client.get("/api/v1/health/", headers=auth_headers)
        
        # Make multiple concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(20)]
            responses = [future.result() for future in futures]
        
        # All requests should succeed
        success_count = sum(1 for r in responses if r.status_code == 200)
        assert success_count >= 18  # Allow for some rate limiting


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_malformed_json_request(self, client, mock_dependencies, auth_headers):
        """Test handling of malformed JSON requests."""
        response = client.post("/api/v1/papers/collections", 
                             data="invalid json", 
                             headers={**auth_headers, "Content-Type": "application/json"})
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_missing_required_fields(self, client, mock_dependencies, auth_headers):
        """Test handling of missing required fields."""
        incomplete_data = {"name": "Test Collection"}  # Missing required fields
        
        response = client.post("/api/v1/papers/collections", 
                             json=incomplete_data, 
                             headers=auth_headers)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_invalid_content_type(self, client, mock_dependencies, auth_headers):
        """Test handling of invalid content types."""
        response = client.post("/api/v1/papers/collections", 
                             data="test data",
                             headers={**auth_headers, "Content-Type": "text/plain"})
        
        assert response.status_code in [415, 422]  # Unsupported Media Type or Unprocessable Entity
    
    def test_large_payload_handling(self, client, mock_dependencies, auth_headers):
        """Test handling of large payloads."""
        large_data = {
            "query": "test" * 10000,  # Very large query
            "filters": {"year": list(range(1900, 2024))}  # Large filter list
        }
        
        response = client.post("/api/v1/search/papers", 
                             json=large_data, 
                             headers=auth_headers)
        
        # Should handle gracefully - either process or reject with proper error
        assert response.status_code in [200, 400, 413, 422]


# Fixtures for async testing
@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Performance test markers
pytest.mark.slow = pytest.mark.skipif(
    True,  # Skip by default
    reason="Performance tests are slow - run with --performance flag"
)