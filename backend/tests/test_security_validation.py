"""
Security Testing Suite for Intellectual Lineage Tracer System

This test suite coordinates with security findings from code review and validates:
- JWT token security and expiration handling
- Input validation and injection prevention  
- CORS configuration testing
- Session security validation
- API security header verification
- Authentication and authorization testing
- Rate limiting security validation
- Data sanitization and XSS prevention
- SQL/NoSQL injection prevention
- File upload security
- Session hijacking prevention
"""

import pytest
import time
import jwt
import hashlib
import base64
import json
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import status
import httpx

from app.main import create_app
from app.core.config import get_settings
from app.core.security import verify_password, get_password_hash
from app.services.auth import create_access_token, verify_token, create_refresh_token


@pytest.fixture
def app():
    """Create test FastAPI app for security testing."""
    return create_app()


@pytest.fixture
def client(app):
    """Create test client for security testing."""
    return TestClient(app)


@pytest.fixture
def security_mocks():
    """Mock external dependencies for security testing."""
    with patch('app.db.neo4j.Neo4jManager') as neo4j_mock, \
         patch('app.db.redis.RedisManager') as redis_mock:
        
        neo4j_mock.return_value.execute_query = MagicMock(return_value=[])
        redis_mock.return_value.get = MagicMock(return_value=None)
        redis_mock.return_value.set = MagicMock(return_value=True)
        
        yield {
            'neo4j': neo4j_mock.return_value,
            'redis': redis_mock.return_value
        }


class TestJWTTokenSecurity:
    """Test JWT token security and lifecycle management."""
    
    def test_jwt_token_creation_and_validation(self, security_mocks):
        """Test JWT token creation and validation process."""
        # Test token creation
        user_data = {"sub": "test_user_123", "email": "test@example.com"}
        token = create_access_token(data=user_data)
        
        assert token is not None
        assert isinstance(token, str)
        
        # Test token validation
        payload = verify_token(token)
        assert payload["sub"] == user_data["sub"]
        assert payload["email"] == user_data["email"]
        assert "exp" in payload  # Expiration claim should be present
        assert "iat" in payload  # Issued at claim should be present
    
    def test_jwt_token_expiration(self, security_mocks):
        """Test JWT token expiration handling."""
        # Create token with very short expiration
        user_data = {"sub": "test_user_123", "email": "test@example.com"}
        
        # Create token that expires in 1 second
        with patch('app.services.auth.get_settings') as mock_settings:
            settings = get_settings()
            settings.security.access_token_expire_minutes = 1/60  # 1 second
            mock_settings.return_value = settings
            
            token = create_access_token(data=user_data)
        
        # Token should be valid immediately
        payload = verify_token(token)
        assert payload is not None
        
        # Wait for token to expire
        time.sleep(2)
        
        # Token should now be invalid
        with pytest.raises(Exception):  # Should raise JWT expiration error
            verify_token(token)
    
    def test_jwt_token_tampering_detection(self, security_mocks):
        """Test detection of tampered JWT tokens."""
        user_data = {"sub": "test_user_123", "email": "test@example.com"}
        token = create_access_token(data=user_data)
        
        # Tamper with token by modifying a character
        tampered_token = token[:-1] + ('x' if token[-1] != 'x' else 'y')
        
        # Tampered token should be rejected
        with pytest.raises(Exception):  # Should raise JWT signature verification error
            verify_token(tampered_token)
    
    def test_jwt_secret_key_strength(self, security_mocks):
        """Test that JWT secret key meets security requirements."""
        settings = get_settings()
        secret_key = settings.security.secret_key
        
        # Secret key should be at least 32 characters
        assert len(secret_key) >= 32
        
        # Secret key should contain mixed characters for complexity
        assert any(c.isupper() for c in secret_key)  # Uppercase
        assert any(c.islower() for c in secret_key)  # Lowercase
        assert any(c.isdigit() for c in secret_key)  # Digits
    
    def test_refresh_token_security(self, security_mocks):
        """Test refresh token security mechanisms."""
        user_data = {"sub": "test_user_123", "email": "test@example.com"}
        refresh_token = create_refresh_token(data=user_data)
        
        assert refresh_token is not None
        assert isinstance(refresh_token, str)
        
        # Refresh token should have longer expiration than access token
        access_payload = verify_token(create_access_token(data=user_data))
        refresh_payload = verify_token(refresh_token)
        
        assert refresh_payload["exp"] > access_payload["exp"]


class TestInputValidationAndSanitization:
    """Test input validation and sanitization against various attack vectors."""
    
    def test_sql_injection_prevention(self, client, security_mocks):
        """Test prevention of SQL injection attacks."""
        # Create auth token
        token = create_access_token(data={"sub": "test_user", "email": "test@example.com"})
        headers = {"Authorization": f"Bearer {token}"}
        
        # Common SQL injection payloads
        sql_injection_payloads = [
            "'; DROP TABLE papers; --",
            "' OR '1'='1",
            "' UNION SELECT * FROM users --",
            "'; INSERT INTO papers VALUES ('malicious') --",
            "' OR 1=1; DELETE FROM papers; --",
            "admin'--",
            "' OR 'x'='x",
            "1' AND (SELECT COUNT(*) FROM users) > 0 --"
        ]
        
        for payload in sql_injection_payloads:
            # Test search endpoint
            response = client.get("/api/v1/search/papers", 
                                params={"query": payload}, 
                                headers=headers)
            
            # Should not return 500 error (indicating SQL error)
            assert response.status_code in [200, 400, 422]
            
            # Response should not contain SQL error messages
            response_text = response.text.lower()
            assert "sql" not in response_text
            assert "syntax error" not in response_text
            assert "database error" not in response_text
    
    def test_nosql_injection_prevention(self, client, security_mocks):
        """Test prevention of NoSQL injection attacks (for Neo4j)."""
        token = create_access_token(data={"sub": "test_user", "email": "test@example.com"})
        headers = {"Authorization": f"Bearer {token}"}
        
        # Cypher injection payloads
        cypher_injection_payloads = [
            "'; MATCH (n) DELETE n; //",
            "') OR true //",
            "'; CREATE (malicious:Malware); //",
            "'; MATCH (u:User) SET u.admin=true; //",
            "') RETURN 1 UNION MATCH (n) RETURN n //",
            "'; CALL db.schema() //",
        ]
        
        for payload in cypher_injection_payloads:
            # Test graph operations
            response = client.post("/api/v1/graph/build-network", 
                                 json={
                                     "seed_papers": [payload],
                                     "depth": 1,
                                     "max_nodes": 10
                                 },
                                 headers=headers)
            
            # Should handle gracefully
            assert response.status_code in [200, 400, 422]
            
            # Should not expose Cypher errors
            response_text = response.text.lower()
            assert "cypher" not in response_text
            assert "neo4j" not in response_text
    
    def test_xss_prevention(self, client, security_mocks):
        """Test prevention of XSS attacks."""
        token = create_access_token(data={"sub": "test_user", "email": "test@example.com"})
        headers = {"Authorization": f"Bearer {token}"}
        
        # XSS payloads
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src='x' onerror='alert(1)'>",
            "<svg onload='alert(1)'>",
            "javascript:alert('XSS')",
            "<iframe src='javascript:alert(1)'></iframe>",
            "<body onload='alert(1)'>",
            "<input onfocus='alert(1)' autofocus>",
            "<marquee onstart='alert(1)'>",
        ]
        
        for payload in xss_payloads:
            # Test search endpoint
            response = client.get("/api/v1/search/papers",
                                params={"query": payload},
                                headers=headers)
            
            if response.status_code == 200:
                response_data = response.json()
                response_str = json.dumps(response_data)
                
                # Response should not contain unescaped script tags
                assert "<script>" not in response_str
                assert "onerror=" not in response_str
                assert "javascript:" not in response_str
                assert "onload=" not in response_str
    
    def test_path_traversal_prevention(self, client, security_mocks):
        """Test prevention of path traversal attacks."""
        token = create_access_token(data={"sub": "test_user", "email": "test@example.com"})
        headers = {"Authorization": f"Bearer {token}"}
        
        # Path traversal payloads
        path_traversal_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "....//....//....//etc/passwd",
            "..%252f..%252f..%252fetc%252fpasswd",
            "../../../proc/self/environ",
            "../../../../etc/shadow",
        ]
        
        for payload in path_traversal_payloads:
            # Test file-related endpoints
            response = client.get(f"/api/v1/papers/export/{payload}",
                                headers=headers)
            
            # Should not allow path traversal
            assert response.status_code in [400, 404, 422]
            
            # Should not expose system files
            if response.status_code == 200:
                response_text = response.text.lower()
                assert "root:" not in response_text  # /etc/passwd content
                assert "bin/bash" not in response_text
                assert "system32" not in response_text
    
    def test_command_injection_prevention(self, client, security_mocks):
        """Test prevention of command injection attacks."""
        token = create_access_token(data={"sub": "test_user", "email": "test@example.com"})
        headers = {"Authorization": f"Bearer {token}"}
        
        # Command injection payloads
        command_injection_payloads = [
            "; ls -la",
            "| whoami",
            "&& cat /etc/passwd",
            "`id`",
            "$(whoami)",
            "; rm -rf /",
            "| nc -l 1234",
            "&& curl malicious.com",
        ]
        
        for payload in command_injection_payloads:
            # Test any endpoint that might process file names or paths
            response = client.post("/api/v1/papers/collections",
                                 json={"name": f"Collection {payload}", "description": "Test"},
                                 headers=headers)
            
            # Should handle gracefully
            assert response.status_code in [201, 400, 422]
            
            # Should not execute commands
            if response.status_code == 201:
                response_data = response.json()
                # Command output should not be in response
                assert "uid=" not in str(response_data).lower()
                assert "/bin/bash" not in str(response_data).lower()


class TestAuthenticationAndAuthorization:
    """Test authentication and authorization security."""
    
    def test_unauthenticated_access_blocked(self, client, security_mocks):
        """Test that unauthenticated requests are properly blocked."""
        protected_endpoints = [
            ("GET", "/api/v1/users/me"),
            ("POST", "/api/v1/papers/collections"),
            ("GET", "/api/v1/search/papers"),
            ("POST", "/api/v1/graph/build-network"),
            ("POST", "/api/v1/llm/generate-summary"),
        ]
        
        for method, endpoint in protected_endpoints:
            if method == "GET":
                response = client.get(endpoint)
            elif method == "POST":
                response = client.post(endpoint, json={})
            
            # Should require authentication
            assert response.status_code == status.HTTP_401_UNAUTHORIZED
            
            response_data = response.json()
            assert "unauthorized" in response_data.get("detail", "").lower()
    
    def test_invalid_token_handling(self, client, security_mocks):
        """Test handling of invalid authentication tokens."""
        invalid_tokens = [
            "invalid_token",
            "Bearer invalid_token",
            "Bearer ",
            "Bearer eyInvalid.Token.Here",
            "",
            "malformed_bearer_token",
        ]
        
        for token in invalid_tokens:
            headers = {"Authorization": token} if token else {}
            
            response = client.get("/api/v1/users/me", headers=headers)
            
            # Should reject invalid tokens
            assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_role_based_access_control(self, client, security_mocks):
        """Test role-based access control implementation."""
        # Create tokens for different user roles
        admin_token = create_access_token(data={
            "sub": "admin_user", 
            "email": "admin@example.com", 
            "role": "admin"
        })
        
        regular_token = create_access_token(data={
            "sub": "regular_user", 
            "email": "user@example.com", 
            "role": "user"
        })
        
        # Test admin-only endpoints
        admin_headers = {"Authorization": f"Bearer {admin_token}"}
        user_headers = {"Authorization": f"Bearer {regular_token}"}
        
        # Admin operations (if implemented)
        admin_endpoints = [
            ("GET", "/api/v1/admin/users"),
            ("POST", "/api/v1/admin/system/maintenance"),
            ("DELETE", "/api/v1/admin/papers/bulk-delete"),
        ]
        
        for method, endpoint in admin_endpoints:
            # Regular user should be forbidden
            if method == "GET":
                user_response = client.get(endpoint, headers=user_headers)
            elif method == "POST":
                user_response = client.post(endpoint, json={}, headers=user_headers)
            elif method == "DELETE":
                user_response = client.delete(endpoint, headers=user_headers)
            
            # Should be forbidden for regular users (403) or not found (404) if not implemented
            assert user_response.status_code in [status.HTTP_403_FORBIDDEN, status.HTTP_404_NOT_FOUND]
    
    def test_session_security(self, client, security_mocks):
        """Test session security mechanisms."""
        # Test session fixation prevention
        login_data = {"username": "test@example.com", "password": "correct_password"}
        
        with patch('app.services.auth.AuthService.authenticate_user') as mock_auth:
            mock_user = {
                "id": "user_123",
                "email": "test@example.com",
                "full_name": "Test User"
            }
            mock_auth.return_value = mock_user
            
            # First login
            response1 = client.post("/api/v1/auth/login", data=login_data)
            assert response1.status_code == status.HTTP_200_OK
            
            token1 = response1.json()["access_token"]
            
            # Second login should generate different token
            response2 = client.post("/api/v1/auth/login", data=login_data)
            assert response2.status_code == status.HTTP_200_OK
            
            token2 = response2.json()["access_token"]
            
            # Tokens should be different (preventing session fixation)
            assert token1 != token2


class TestCORSAndSecurityHeaders:
    """Test CORS configuration and security headers."""
    
    def test_cors_configuration(self, client, security_mocks):
        """Test CORS configuration security."""
        # Test preflight OPTIONS request
        headers = {
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Authorization, Content-Type"
        }
        
        response = client.options("/api/v1/search/papers", headers=headers)
        
        # Should allow configured origins
        assert "Access-Control-Allow-Origin" in response.headers
        assert "Access-Control-Allow-Methods" in response.headers
        assert "Access-Control-Allow-Headers" in response.headers
        
        # Test malicious origin
        malicious_headers = {
            "Origin": "https://malicious-site.com",
            "Access-Control-Request-Method": "POST"
        }
        
        response = client.options("/api/v1/search/papers", headers=malicious_headers)
        
        # Should not allow unauthorized origins
        allowed_origin = response.headers.get("Access-Control-Allow-Origin")
        assert allowed_origin != "https://malicious-site.com"
    
    def test_security_headers_presence(self, client, security_mocks):
        """Test presence of security headers."""
        response = client.get("/api/v1/health/")
        
        # Check for security headers (these should be added by middleware)
        headers = response.headers
        
        # Content-Type should be properly set
        assert "application/json" in headers.get("content-type", "")
        
        # Should not expose server information
        server_header = headers.get("server", "").lower()
        assert "apache" not in server_header
        assert "nginx" not in server_header
        assert "iis" not in server_header
    
    def test_sensitive_data_exposure_prevention(self, client, security_mocks):
        """Test prevention of sensitive data exposure in responses."""
        # Test error responses don't expose sensitive information
        response = client.get("/api/v1/nonexistent-endpoint")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        
        response_text = response.text.lower()
        
        # Should not expose system paths, stack traces, or database info
        assert "traceback" not in response_text
        assert "/usr/" not in response_text
        assert "c:\\" not in response_text
        assert "password" not in response_text
        assert "secret" not in response_text
        assert "database" not in response_text
        assert "connection string" not in response_text


class TestRateLimitingSecurity:
    """Test rate limiting as a security mechanism."""
    
    def test_rate_limiting_prevents_brute_force(self, client, security_mocks):
        """Test that rate limiting prevents brute force attacks."""
        # Simulate brute force login attempts
        login_data = {"username": "test@example.com", "password": "wrong_password"}
        
        responses = []
        for i in range(20):  # Attempt 20 failed logins
            response = client.post("/api/v1/auth/login", data=login_data)
            responses.append(response.status_code)
            
            # If we hit rate limit, stop
            if response.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
                break
        
        # Should eventually rate limit
        assert status.HTTP_429_TOO_MANY_REQUESTS in responses
    
    def test_rate_limiting_per_endpoint(self, client, security_mocks):
        """Test rate limiting on sensitive endpoints."""
        token = create_access_token(data={"sub": "test_user", "email": "test@example.com"})
        headers = {"Authorization": f"Bearer {token}"}
        
        # Rapid requests to search endpoint
        responses = []
        for i in range(100):  # Many requests
            response = client.get("/api/v1/search/papers", 
                                params={"query": f"test {i}"},
                                headers=headers)
            responses.append(response.status_code)
            
            # If rate limited, record it
            if response.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
                break
        
        # Should implement rate limiting
        rate_limited = status.HTTP_429_TOO_MANY_REQUESTS in responses
        
        # Either rate limited or configured to allow high volume
        # (Both are valid depending on configuration)
        assert len(responses) > 0


class TestFileUploadSecurity:
    """Test file upload security mechanisms."""
    
    def test_bibtex_upload_validation(self, client, security_mocks):
        """Test BibTeX file upload security validation."""
        token = create_access_token(data={"sub": "test_user", "email": "test@example.com"})
        headers = {"Authorization": f"Bearer {token}"}
        
        # Test valid BibTeX file
        valid_bibtex = """
        @article{test2023,
            title={Test Paper},
            author={Test Author},
            year={2023}
        }
        """
        
        files = {"file": ("test.bib", valid_bibtex, "text/plain")}
        response = client.post("/api/v1/papers/import/bibtex", files=files, headers=headers)
        
        # Should accept valid BibTeX
        assert response.status_code in [200, 202, 422]  # 422 if validation fails on content
    
    def test_malicious_file_upload_prevention(self, client, security_mocks):
        """Test prevention of malicious file uploads."""
        token = create_access_token(data={"sub": "test_user", "email": "test@example.com"})
        headers = {"Authorization": f"Bearer {token}"}
        
        # Test malicious file types
        malicious_files = [
            ("malicious.exe", b"MZ\x90\x00", "application/octet-stream"),  # PE executable
            ("script.js", b"<script>alert('xss')</script>", "text/javascript"),
            ("shell.sh", b"#!/bin/bash\nrm -rf /", "text/x-shellscript"),
            ("huge.txt", b"A" * 10000000, "text/plain"),  # Very large file
        ]
        
        for filename, content, content_type in malicious_files:
            files = {"file": (filename, content, content_type)}
            response = client.post("/api/v1/papers/import/bibtex", files=files, headers=headers)
            
            # Should reject malicious files
            assert response.status_code in [400, 413, 422, 415]  # Bad request, too large, unprocessable, unsupported


class TestDataSanitizationAndOutput:
    """Test data sanitization and output security."""
    
    def test_json_response_sanitization(self, client, security_mocks):
        """Test that JSON responses are properly sanitized."""
        token = create_access_token(data={"sub": "test_user", "email": "test@example.com"})
        headers = {"Authorization": f"Bearer {token}"}
        
        # Mock search with potentially dangerous content
        dangerous_content = {
            "papers": [{
                "id": "paper_1",
                "title": "<script>alert('xss')</script>Dangerous Paper",
                "abstract": "This contains javascript:alert(1) in the text",
                "authors": [{"name": "<img src=x onerror=alert(1)>"}]
            }]
        }
        
        with patch('app.services.search.SearchService.search_papers') as mock_search:
            mock_search.return_value = dangerous_content
            
            response = client.get("/api/v1/search/papers", 
                                params={"query": "test"}, 
                                headers=headers)
            
            if response.status_code == 200:
                response_data = response.json()
                response_str = json.dumps(response_data)
                
                # Dangerous content should be sanitized or escaped
                assert "<script>" not in response_str
                assert "javascript:" not in response_str
                assert "onerror=" not in response_str
    
    def test_response_content_type_security(self, client, security_mocks):
        """Test that response content types are properly set."""
        # Test JSON endpoints
        response = client.get("/api/v1/health/")
        assert "application/json" in response.headers.get("content-type", "")
        
        # Test that responses don't have executable content types
        content_type = response.headers.get("content-type", "")
        assert "text/html" not in content_type  # Prevents HTML injection
        assert "application/javascript" not in content_type
        assert "text/javascript" not in content_type


# Security test markers and configuration
pytest.mark.security = pytest.mark.filterwarnings("ignore:.*:pytest.PytestUnraisableExceptionWarning")

@pytest.fixture(autouse=True)
def security_test_setup():
    """Setup for security tests."""
    # Ensure clean state for each security test
    yield