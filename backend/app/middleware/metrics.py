"""
Metrics collection middleware for automatic request tracking.
"""

import time
from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from ..services.metrics import get_metrics_service
from ..utils.logger import get_logger

logger = get_logger(__name__)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for automatic metrics collection."""
    
    def __init__(self, app):
        super().__init__(app)
        self.metrics_service = get_metrics_service()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with metrics collection."""
        
        # Record start time
        start_time = time.time()
        
        # Extract request information
        method = request.method
        path = request.url.path
        
        # Normalize endpoint for metrics (remove IDs, etc.)
        normalized_endpoint = self._normalize_endpoint(path)
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            end_time = time.time()
            duration_seconds = end_time - start_time
            
            # Record metrics
            self.metrics_service.record_request(
                method=method,
                endpoint=normalized_endpoint,
                status_code=response.status_code,
                duration_seconds=duration_seconds
            )
            
            return response
            
        except Exception as exc:
            # Record error metrics
            end_time = time.time()
            duration_seconds = end_time - start_time
            
            # Determine status code from exception
            status_code = 500
            if hasattr(exc, 'status_code'):
                status_code = exc.status_code
            
            self.metrics_service.record_request(
                method=method,
                endpoint=normalized_endpoint,
                status_code=status_code,
                duration_seconds=duration_seconds
            )
            
            # Re-raise the exception
            raise
    
    def _normalize_endpoint(self, path: str) -> str:
        """
        Normalize endpoint path for metrics collection.
        
        Replace dynamic segments with placeholders to avoid high cardinality.
        """
        # Skip normalization for health and metrics endpoints
        if path in ["/health", "/metrics", "/ready", "/live"]:
            return path
        
        # Replace common ID patterns
        import re
        
        # Replace UUIDs
        path = re.sub(
            r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
            '/{uuid}',
            path,
            flags=re.IGNORECASE
        )
        
        # Replace numeric IDs
        path = re.sub(r'/\d+', '/{id}', path)
        
        # Replace DOIs
        path = re.sub(r'/10\.\d+/[^\s/]+', '/{doi}', path)
        
        # Replace common patterns
        patterns = [
            (r'/users/[^/]+', '/users/{user_id}'),
            (r'/papers/[^/]+', '/papers/{paper_id}'),
            (r'/collections/[^/]+', '/collections/{collection_id}'),
        ]
        
        for pattern, replacement in patterns:
            path = re.sub(pattern, replacement, path)
        
        return path


class DatabaseMetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting database-related metrics."""
    
    def __init__(self, app):
        super().__init__(app)
        self.metrics_service = get_metrics_service()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with database metrics collection."""
        
        # Get initial database connection counts
        # (In a real implementation, you'd get actual connection pool stats)
        initial_neo4j_connections = 1  # Placeholder
        initial_redis_connections = 1   # Placeholder
        
        try:
            response = await call_next(request)
            
            # Record final connection counts
            self.metrics_service.record_database_connection("neo4j", initial_neo4j_connections)
            self.metrics_service.record_database_connection("redis", initial_redis_connections)
            
            return response
            
        except Exception as exc:
            # Still record metrics on error
            self.metrics_service.record_database_connection("neo4j", initial_neo4j_connections)
            self.metrics_service.record_database_connection("redis", initial_redis_connections)
            
            raise


class CacheMetricsDecorator:
    """Decorator for cache operation metrics."""
    
    def __init__(self, metrics_service=None):
        self.metrics_service = metrics_service or get_metrics_service()
    
    def __call__(self, operation: str):
        """Decorator for cache operations."""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                try:
                    result = await func(*args, **kwargs)
                    
                    # Determine if it was a cache hit based on result
                    hit = result is not None
                    self.metrics_service.record_cache_operation(operation, hit)
                    
                    return result
                    
                except Exception as exc:
                    # Record cache miss on error
                    self.metrics_service.record_cache_operation(operation, False)
                    raise
            
            return wrapper
        return decorator


class ExternalAPIMetricsDecorator:
    """Decorator for external API call metrics."""
    
    def __init__(self, service_name: str, metrics_service=None):
        self.service_name = service_name
        self.metrics_service = metrics_service or get_metrics_service()
    
    def __call__(self, func):
        """Decorator for external API calls."""
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status_code = 0
            
            try:
                result = await func(*args, **kwargs)
                
                # Try to extract status code from result
                if hasattr(result, 'status_code'):
                    status_code = result.status_code
                else:
                    status_code = 200  # Assume success
                
                return result
                
            except Exception as exc:
                # Extract status code from exception if available
                if hasattr(exc, 'status_code'):
                    status_code = exc.status_code
                elif hasattr(exc, 'response') and hasattr(exc.response, 'status_code'):
                    status_code = exc.response.status_code
                else:
                    status_code = 500
                
                raise
                
            finally:
                end_time = time.time()
                duration_seconds = end_time - start_time
                
                self.metrics_service.record_external_api_call(
                    service=self.service_name,
                    status_code=status_code,
                    duration_seconds=duration_seconds
                )
        
        return wrapper


# Convenience decorators
def track_cache_operation(operation: str):
    """Decorator for tracking cache operations."""
    return CacheMetricsDecorator()(operation)


def track_external_api(service_name: str):
    """Decorator for tracking external API calls."""
    return ExternalAPIMetricsDecorator(service_name)


def track_search_operation(engine: str):
    """Decorator for tracking search operations."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            metrics_service = get_metrics_service()
            success = False
            
            try:
                result = await func(*args, **kwargs)
                success = True
                return result
                
            except Exception:
                raise
                
            finally:
                metrics_service.record_search_operation(engine, success)
        
        return wrapper
    return decorator


def track_paper_operation(operation: str):
    """Decorator for tracking paper operations."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            metrics_service = get_metrics_service()
            success = False
            
            try:
                result = await func(*args, **kwargs)
                success = True
                return result
                
            except Exception:
                raise
                
            finally:
                metrics_service.record_paper_operation(operation, success)
        
        return wrapper
    return decorator