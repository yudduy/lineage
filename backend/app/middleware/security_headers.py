"""
Security headers middleware for protecting against common web vulnerabilities.
"""

from typing import Dict, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ..core.config import get_settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add security headers to all HTTP responses.
    
    Protects against:
    - XSS attacks
    - Clickjacking
    - MIME-type sniffing
    - Content injection
    - Referrer leakage
    """
    
    def __init__(self, app, settings=None):
        super().__init__(app)
        self.settings = settings or get_settings()
        
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Add security headers
        security_headers = self._get_security_headers(request)
        
        for header, value in security_headers.items():
            response.headers[header] = value
        
        return response
    
    def _get_security_headers(self, request: Request) -> Dict[str, str]:
        """Generate security headers based on environment."""
        headers = {}
        
        # X-Content-Type-Options - Prevent MIME sniffing
        headers["X-Content-Type-Options"] = "nosniff"
        
        # X-Frame-Options - Prevent clickjacking
        headers["X-Frame-Options"] = "DENY"
        
        # X-XSS-Protection - Legacy XSS protection (for older browsers)
        headers["X-XSS-Protection"] = "1; mode=block"
        
        # Referrer-Policy - Control referrer information
        headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Content-Security-Policy - Prevent XSS and injection attacks
        csp = self._build_content_security_policy()
        if csp:
            headers["Content-Security-Policy"] = csp
        
        # Permissions-Policy - Control browser features
        permissions_policy = self._build_permissions_policy()
        if permissions_policy:
            headers["Permissions-Policy"] = permissions_policy
        
        # HSTS - Force HTTPS in production
        if self.settings.is_production:
            headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
        
        # Cross-Origin-Embedder-Policy - Isolation against Spectre attacks
        headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        
        # Cross-Origin-Opener-Policy - Prevent cross-origin attacks
        headers["Cross-Origin-Opener-Policy"] = "same-origin"
        
        # Cross-Origin-Resource-Policy - Control resource access
        headers["Cross-Origin-Resource-Policy"] = "cross-origin"
        
        # Server information hiding
        headers["Server"] = "CitationNetworkAPI"
        
        # Cache control for API responses
        if request.url.path.startswith("/api/"):
            headers["Cache-Control"] = "no-store, no-cache, must-revalidate, private"
            headers["Pragma"] = "no-cache"
            headers["Expires"] = "0"
        
        return headers
    
    def _build_content_security_policy(self) -> str:
        """Build Content Security Policy based on environment."""
        # Base policy for API server
        policy_parts = [
            "default-src 'self'",
            "script-src 'self'",
            "style-src 'self' 'unsafe-inline'",  # Allow inline styles for API docs
            "img-src 'self' data: https:",
            "font-src 'self'",
            "connect-src 'self'",
            "media-src 'self'",
            "object-src 'none'",
            "base-uri 'self'",
            "form-action 'self'",
            "frame-ancestors 'none'",
            "upgrade-insecure-requests" if self.settings.is_production else ""
        ]
        
        # Add development-specific policies
        if self.settings.is_development:
            # Allow webpack dev server and hot reload
            policy_parts[4] = "connect-src 'self' ws: wss:"
            # Allow inline scripts for development tools
            policy_parts[1] = "script-src 'self' 'unsafe-inline' 'unsafe-eval'"
        
        return "; ".join(filter(None, policy_parts))
    
    def _build_permissions_policy(self) -> str:
        """Build Permissions Policy to restrict browser features."""
        # Restrict potentially dangerous features
        policies = [
            "camera=()",
            "microphone=()",
            "geolocation=()",
            "interest-cohort=()",  # Disable FLoC
            "payment=()",
            "usb=()",
            "bluetooth=()",
            "accelerometer=()",
            "gyroscope=()",
            "magnetometer=()"
        ]
        
        return ", ".join(policies)


def setup_security_headers(app):
    """Setup security headers middleware."""
    logger.info("Setting up security headers middleware")
    app.add_middleware(SecurityHeadersMiddleware)


class CSPViolationHandler:
    """Handle Content Security Policy violation reports."""
    
    @staticmethod
    async def handle_csp_report(request: Request):
        """Handle CSP violation report."""
        try:
            report = await request.json()
            logger.warning("CSP violation reported", extra={"report": report})
            
            # In production, you might want to send these to a monitoring service
            if get_settings().is_production:
                # TODO: Send to monitoring service
                pass
                
        except Exception as e:
            logger.error(f"Failed to process CSP report: {e}")
        
        return Response(status_code=204)


def add_security_response_headers(response: Response, request: Request = None) -> Response:
    """Manually add security headers to a specific response."""
    middleware = SecurityHeadersMiddleware(None)
    headers = middleware._get_security_headers(request) if request else {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
    }
    
    for header, value in headers.items():
        response.headers[header] = value
    
    return response


def validate_request_headers(request: Request) -> Dict[str, str]:
    """Validate request headers for security issues."""
    issues = {}
    
    # Check for potentially dangerous headers
    dangerous_headers = [
        "x-forwarded-host",  # Can be used for host header injection
        "x-real-ip",         # IP spoofing attempts
        "x-forwarded-for"    # IP spoofing attempts
    ]
    
    for header in dangerous_headers:
        if header in request.headers:
            value = request.headers[header]
            # Basic validation - check for suspicious values
            if any(char in value for char in ['<', '>', '"', "'"]):
                issues[header] = f"Suspicious characters in {header}"
    
    # Validate User-Agent
    user_agent = request.headers.get("user-agent", "")
    if len(user_agent) > 1000:  # Unusually long user agent
        issues["user-agent"] = "User agent too long"
    
    # Validate Content-Type for POST/PUT requests
    if request.method in ["POST", "PUT", "PATCH"]:
        content_type = request.headers.get("content-type", "")
        if content_type and "application/json" not in content_type and "multipart/form-data" not in content_type:
            # Log unexpected content types
            logger.warning(f"Unexpected content type: {content_type}")
    
    return issues