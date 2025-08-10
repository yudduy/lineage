"""
CORS middleware configuration for frontend integration.
"""

from typing import List, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..core.config import get_settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


def setup_cors(app: FastAPI) -> None:
    """
    Setup CORS middleware for the FastAPI application with security best practices.
    
    This allows the React frontend to make requests to the FastAPI backend
    from different origins with proper security controls.
    """
    settings = get_settings()
    
    # Get allowed origins from settings
    origins = settings.backend_cors_origins
    
    # Add default origins if none specified (development only)
    if not origins and settings.is_development:
        origins = [
            "http://localhost:3000",  # React dev server
            "https://localhost:3000", # React dev server with HTTPS
            "http://127.0.0.1:3000",  # Alternative localhost
            "https://127.0.0.1:3000", # Alternative localhost with HTTPS
        ]
    elif not origins:
        # Production requires explicit origins
        logger.error("No CORS origins configured for production environment")
        raise ValueError("BACKEND_CORS_ORIGINS must be configured in production")
    
    # Security validation for production
    if settings.is_production:
        for origin in origins:
            if origin == "*" or "localhost" in origin or "127.0.0.1" in origin:
                logger.error(f"Insecure CORS origin detected in production: {origin}")
                raise ValueError(f"Insecure CORS origin not allowed in production: {origin}")
    
    logger.info(f"Setting up CORS with origins: {origins}")
    
    # Configure CORS middleware with security-focused settings
    cors_settings = {
        "allow_origins": origins,
        "allow_credentials": True,
        "allow_methods": [
            "GET",
            "POST", 
            "PUT",
            "PATCH",
            "DELETE",
            "OPTIONS",
            "HEAD"
        ],
        "expose_headers": [
            "X-Correlation-ID",
            "X-Response-Time",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining", 
            "X-RateLimit-Reset",
            "X-Total-Count",
        ]
    }
    
    # Different header policies for development vs production
    if settings.is_production:
        # Restrict headers in production
        cors_settings["allow_headers"] = [
            "Accept",
            "Accept-Language", 
            "Content-Language",
            "Content-Type",
            "Authorization",
            "X-Requested-With",
            "X-Session-ID",
            "X-API-Key",
            "X-Correlation-ID"
        ]
        cors_settings["max_age"] = 86400  # Cache preflight for 24 hours in production
    else:
        # More permissive in development but still not wildcard
        cors_settings["allow_headers"] = [
            "Accept",
            "Accept-Language", 
            "Content-Language",
            "Content-Type",
            "Authorization",
            "X-Requested-With",
            "X-Session-ID",
            "X-API-Key",
            "X-Correlation-ID",
            "X-Debug-Mode"
        ]
        cors_settings["max_age"] = 600  # Cache preflight for 10 minutes
    
    app.add_middleware(CORSMiddleware, **cors_settings)
    
    logger.info("CORS middleware configured successfully")


class CORSConfig:
    """CORS configuration management."""
    
    def __init__(self):
        self.settings = get_settings()
        
    def get_allowed_origins(self) -> List[str]:
        """Get list of allowed CORS origins."""
        return self.settings.backend_cors_origins
    
    def is_origin_allowed(self, origin: str) -> bool:
        """Check if an origin is allowed."""
        allowed_origins = self.get_allowed_origins()
        
        # Never allow any origin in production for security
        if self.settings.is_production:
            # Only exact matches allowed in production
            return origin in allowed_origins
        
        # In development, still validate against allowed list for security
        # but be more permissive with localhost variations
        if self.settings.is_development:
            # Allow localhost variations in development
            localhost_patterns = [
                "http://localhost:",
                "https://localhost:",
                "http://127.0.0.1:",
                "https://127.0.0.1:"
            ]
            
            # Check if origin matches localhost patterns
            for pattern in localhost_patterns:
                if origin.startswith(pattern):
                    # Extract port and validate it's reasonable
                    try:
                        port = int(origin.split(':')[-1])
                        if 3000 <= port <= 9999:  # Common development port range
                            return True
                    except (ValueError, IndexError):
                        pass
        
        # Check exact matches
        if origin in allowed_origins:
            return True
        
        # For production and staging, no wildcard matching
        if not self.settings.is_development:
            return False
        
        # Development only: Check limited wildcard patterns
        for allowed_origin in allowed_origins:
            # Simple wildcard matching only for .localhost domains in development
            if allowed_origin.startswith("*.localhost"):
                domain = allowed_origin[2:]
                if origin.endswith(f".{domain}"):
                    return True
        
        return False
    
    def get_cors_headers(self, request_origin: Optional[str] = None) -> dict:
        """Get CORS headers for a response."""
        headers = {}
        
        if request_origin and self.is_origin_allowed(request_origin):
            headers["Access-Control-Allow-Origin"] = request_origin
            headers["Access-Control-Allow-Credentials"] = "true"
        
        return headers


def get_cors_config() -> CORSConfig:
    """Get CORS configuration instance."""
    return CORSConfig()


# Custom CORS handler for more control
async def handle_cors_preflight(request, call_next):
    """
    Custom CORS preflight handler.
    
    This can be used as middleware if you need more control over CORS handling
    than what FastAPI's CORSMiddleware provides.
    """
    from starlette.responses import Response
    from starlette.middleware.base import BaseHTTPMiddleware
    
    class CustomCORSMiddleware(BaseHTTPMiddleware):
        def __init__(self, app, cors_config: CORSConfig):
            super().__init__(app)
            self.cors_config = cors_config
        
        async def dispatch(self, request, call_next):
            origin = request.headers.get("origin")
            
            # Handle preflight requests
            if request.method == "OPTIONS":
                response = Response()
                
                if origin and self.cors_config.is_origin_allowed(origin):
                    response.headers["Access-Control-Allow-Origin"] = origin
                    response.headers["Access-Control-Allow-Credentials"] = "true"
                    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, PATCH, DELETE, OPTIONS, HEAD"
                    response.headers["Access-Control-Allow-Headers"] = request.headers.get(
                        "access-control-request-headers", "*"
                    )
                    response.headers["Access-Control-Max-Age"] = "600"
                
                return response
            
            # Process normal request
            response = await call_next(request)
            
            # Add CORS headers to response
            if origin and self.cors_config.is_origin_allowed(origin):
                cors_headers = self.cors_config.get_cors_headers(origin)
                for key, value in cors_headers.items():
                    response.headers[key] = value
            
            return response
    
    return CustomCORSMiddleware


# CORS utilities for manual handling
def add_cors_headers(response, origin: Optional[str] = None):
    """Manually add CORS headers to a response."""
    cors_config = get_cors_config()
    
    if origin and cors_config.is_origin_allowed(origin):
        cors_headers = cors_config.get_cors_headers(origin)
        for key, value in cors_headers.items():
            response.headers[key] = value
    
    return response


def validate_cors_origin(origin: str) -> bool:
    """Validate if an origin is allowed for CORS."""
    cors_config = get_cors_config()
    return cors_config.is_origin_allowed(origin)


# Development CORS configuration
def setup_development_cors(app: FastAPI) -> None:
    """
    Setup permissive CORS for development.
    
    WARNING: Only use this in development environments.
    """
    logger.warning("Setting up permissive CORS for development - DO NOT USE IN PRODUCTION")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins
        allow_credentials=True,
        allow_methods=["*"],  # Allow all methods
        allow_headers=["*"],  # Allow all headers
        expose_headers=["*"], # Expose all headers
    )


# Production CORS configuration
def setup_production_cors(app: FastAPI, allowed_origins: List[str]) -> None:
    """
    Setup strict CORS for production.
    
    Args:
        app: FastAPI application instance
        allowed_origins: List of allowed origins
    """
    logger.info(f"Setting up production CORS with origins: {allowed_origins}")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=[
            "GET",
            "POST",
            "PUT", 
            "PATCH",
            "DELETE",
            "HEAD"
            # Note: OPTIONS is handled automatically
        ],
        allow_headers=[
            "Accept",
            "Accept-Language",
            "Content-Language", 
            "Content-Type",
            "Authorization",
            "X-Requested-With",
            "X-Session-ID",
            "X-API-Key",
        ],
        expose_headers=[
            "X-Correlation-ID",
            "X-Response-Time", 
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
            "X-Total-Count",
            "Content-Range",
        ],
        max_age=86400,  # Cache preflight for 24 hours in production
    )