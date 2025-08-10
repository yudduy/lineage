"""
Logging middleware for request/response tracking.
"""

import time
import uuid
from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import structlog

from ..utils.logger import get_logger


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests and responses."""
    
    def __init__(self, app):
        super().__init__(app)
        self.logger = get_logger("http")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with logging."""
        
        # Generate correlation ID for request tracking
        correlation_id = str(uuid.uuid4())
        request.state.correlation_id = correlation_id
        
        # Add correlation ID to logging context
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(correlation_id=correlation_id)
        
        # Extract request information
        method = request.method
        url = str(request.url)
        path = request.url.path
        query_params = dict(request.query_params)
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Get user ID if available (from JWT token)
        user_id = None
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            # In a real implementation, you'd decode the JWT to get user ID
            # For now, we'll extract it from request state if available
            user_id = getattr(request.state, "user_id", None)
        
        # Log incoming request
        self.logger.info(
            "Request started",
            method=method,
            path=path,
            query_params=query_params,
            client_ip=client_ip,
            user_agent=user_agent,
            user_id=user_id,
            correlation_id=correlation_id
        )
        
        # Record start time
        start_time = time.time()
        
        # Process request and handle errors
        try:
            response = await call_next(request)
            
            # Calculate response time
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            # Add correlation ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id
            
            # Log successful response
            self.logger.info(
                "Request completed",
                method=method,
                path=path,
                status_code=response.status_code,
                response_time_ms=round(response_time_ms, 2),
                client_ip=client_ip,
                user_id=user_id,
                correlation_id=correlation_id
            )
            
            return response
            
        except Exception as exc:
            # Calculate response time for error cases
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            # Log error
            self.logger.error(
                "Request failed",
                method=method,
                path=path,
                error_type=type(exc).__name__,
                error_message=str(exc),
                response_time_ms=round(response_time_ms, 2),
                client_ip=client_ip,
                user_id=user_id,
                correlation_id=correlation_id,
                exc_info=True
            )
            
            # Re-raise the exception
            raise
        
        finally:
            # Clear correlation ID from context
            structlog.contextvars.clear_contextvars()


class ErrorLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive error logging and handling."""
    
    def __init__(self, app):
        super().__init__(app)
        self.logger = get_logger("errors")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with error handling."""
        try:
            response = await call_next(request)
            
            # Log 4xx and 5xx responses
            if response.status_code >= 400:
                self.logger.warning(
                    "HTTP error response",
                    method=request.method,
                    path=request.url.path,
                    status_code=response.status_code,
                    client_ip=request.client.host if request.client else "unknown",
                    correlation_id=getattr(request.state, "correlation_id", None)
                )
            
            return response
            
        except Exception as exc:
            # Log unhandled exceptions
            self.logger.error(
                "Unhandled exception",
                method=request.method,
                path=request.url.path,
                error_type=type(exc).__name__,
                error_message=str(exc),
                client_ip=request.client.host if request.client else "unknown",
                correlation_id=getattr(request.state, "correlation_id", None),
                exc_info=True
            )
            
            # Create error response
            from ..utils.exceptions import APIException
            from ..models.common import ErrorResponse
            from fastapi.responses import JSONResponse
            from fastapi import status
            
            if isinstance(exc, APIException):
                # Return the API exception as-is
                raise exc
            else:
                # Convert unexpected exceptions to internal server error
                error_response = ErrorResponse(
                    error="INTERNAL_SERVER_ERROR",
                    message="An unexpected error occurred",
                    details={"correlation_id": getattr(request.state, "correlation_id", None)}
                )
                
                return JSONResponse(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content=error_response.model_dump()
                )


class SecurityLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for security-related logging."""
    
    def __init__(self, app):
        super().__init__(app)
        self.logger = get_logger("security")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with security logging."""
        
        # Log authentication attempts
        if request.url.path in ["/auth/login", "/auth/token", "/auth/refresh"]:
            self.logger.info(
                "Authentication attempt",
                endpoint=request.url.path,
                client_ip=request.client.host if request.client else "unknown",
                user_agent=request.headers.get("user-agent", "unknown")
            )
        
        # Log sensitive operations
        sensitive_paths = [
            "/users/",
            "/admin/",
            "/settings/",
            "/api/v1/zotero/auth"
        ]
        
        if any(request.url.path.startswith(path) for path in sensitive_paths):
            self.logger.info(
                "Sensitive operation",
                method=request.method,
                path=request.url.path,
                client_ip=request.client.host if request.client else "unknown",
                user_id=getattr(request.state, "user_id", None)
            )
        
        response = await call_next(request)
        
        # Log failed authentication
        if request.url.path.startswith("/auth/") and response.status_code == 401:
            self.logger.warning(
                "Authentication failed",
                endpoint=request.url.path,
                client_ip=request.client.host if request.client else "unknown",
                status_code=response.status_code
            )
        
        # Log authorization failures
        if response.status_code == 403:
            self.logger.warning(
                "Authorization denied",
                method=request.method,
                path=request.url.path,
                client_ip=request.client.host if request.client else "unknown",
                user_id=getattr(request.state, "user_id", None)
            )
        
        return response


class PerformanceLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for performance monitoring and logging."""
    
    def __init__(self, app, slow_request_threshold_ms: float = 1000):
        super().__init__(app)
        self.logger = get_logger("performance")
        self.slow_request_threshold_ms = slow_request_threshold_ms
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with performance monitoring."""
        
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            # Log slow requests
            if response_time_ms > self.slow_request_threshold_ms:
                self.logger.warning(
                    "Slow request detected",
                    method=request.method,
                    path=request.url.path,
                    response_time_ms=round(response_time_ms, 2),
                    threshold_ms=self.slow_request_threshold_ms,
                    status_code=response.status_code,
                    client_ip=request.client.host if request.client else "unknown"
                )
            
            # Add performance headers
            response.headers["X-Response-Time"] = f"{response_time_ms:.2f}ms"
            
            return response
            
        except Exception as exc:
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            # Log error performance
            self.logger.error(
                "Request error performance",
                method=request.method,
                path=request.url.path,
                response_time_ms=round(response_time_ms, 2),
                error_type=type(exc).__name__,
                error_message=str(exc)
            )
            
            raise