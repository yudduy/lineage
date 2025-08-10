"""
Rate limiting middleware using Redis backend.
"""

import time
from typing import Optional, Callable, Dict, Any
from datetime import datetime, timedelta

from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
import redis.asyncio as redis

from ..core.config import get_settings


class RateLimitManager:
    """Redis-based rate limiting manager."""
    
    def __init__(self):
        self.settings = get_settings()
        self.redis_client: Optional[redis.Redis] = None
        
    async def setup_redis(self):
        """Setup Redis connection."""
        if not self.redis_client:
            self.redis_client = redis.from_url(
                self.settings.database.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
    
    async def close_redis(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
    
    async def is_rate_limited(
        self,
        key: str,
        limit: int,
        window_seconds: int,
        increment: bool = True
    ) -> tuple[bool, Dict[str, Any]]:
        """
        Check if key is rate limited using sliding window algorithm.
        
        Returns:
            tuple: (is_limited, {current_count, limit, remaining, reset_time})
        """
        await self.setup_redis()
        
        current_time = time.time()
        window_start = current_time - window_seconds
        
        # Redis key for this rate limit
        rate_limit_key = f"rate_limit:{key}"
        
        # Use pipeline for atomic operations
        pipe = self.redis_client.pipeline()
        
        # Remove old entries outside the window
        pipe.zremrangebyscore(rate_limit_key, 0, window_start)
        
        # Count current entries in window
        pipe.zcard(rate_limit_key)
        
        # Add current request if not just checking
        if increment:
            pipe.zadd(rate_limit_key, {str(current_time): current_time})
        
        # Set expiration on key
        pipe.expire(rate_limit_key, window_seconds)
        
        results = await pipe.execute()
        
        current_count = results[1]
        if increment:
            current_count = results[2]  # After adding current request
        
        is_limited = current_count > limit
        remaining = max(0, limit - current_count)
        reset_time = current_time + window_seconds
        
        return is_limited, {
            "current_count": current_count,
            "limit": limit,
            "remaining": remaining,
            "reset_time": reset_time,
            "window_seconds": window_seconds
        }
    
    async def get_rate_limit_info(
        self,
        key: str,
        limit: int,
        window_seconds: int
    ) -> Dict[str, Any]:
        """Get rate limit information without incrementing counter."""
        _, info = await self.is_rate_limited(key, limit, window_seconds, increment=False)
        return info
    
    async def reset_rate_limit(self, key: str) -> bool:
        """Reset rate limit for a key."""
        await self.setup_redis()
        
        rate_limit_key = f"rate_limit:{key}"
        result = await self.redis_client.delete(rate_limit_key)
        
        return result > 0


class RateLimitRule:
    """Rate limiting rule configuration."""
    
    def __init__(
        self,
        limit: int,
        window_seconds: int,
        key_func: Optional[Callable[[Request], str]] = None,
        message: str = "Rate limit exceeded",
        headers: bool = True
    ):
        self.limit = limit
        self.window_seconds = window_seconds
        self.key_func = key_func or self.default_key_func
        self.message = message
        self.headers = headers
    
    def default_key_func(self, request: Request) -> str:
        """Default key function using client IP."""
        client_ip = request.client.host
        return f"ip:{client_ip}"
    
    def user_key_func(self, request: Request) -> str:
        """Key function for authenticated users."""
        # Try to get user ID from JWT token
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            # In a real implementation, you'd decode the JWT
            # For now, we'll use a simple approach
            token = auth_header.split(" ")[1]
            return f"user:{hash(token) % 1000000}"  # Simple hash for demo
        
        # Fall back to IP-based limiting
        return self.default_key_func(request)
    
    def endpoint_key_func(self, request: Request) -> str:
        """Key function for endpoint-specific limiting."""
        client_ip = request.client.host
        endpoint = f"{request.method}:{request.url.path}"
        return f"endpoint:{client_ip}:{endpoint}"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware."""
    
    def __init__(self, app, rules: Optional[list[RateLimitRule]] = None):
        super().__init__(app)
        self.rules = rules or []
        self.rate_limit_manager = RateLimitManager()
        self.settings = get_settings()
        
        # Add default rules if none provided
        if not self.rules:
            self._add_default_rules()
    
    def _add_default_rules(self):
        """Add default rate limiting rules."""
        # General rate limiting
        self.rules.append(
            RateLimitRule(
                limit=self.settings.rate_limits.rate_limit_per_minute,
                window_seconds=60,
                key_func=lambda req: f"ip:{req.client.host}",
                message="Too many requests per minute"
            )
        )
        
        # API endpoint specific limiting
        api_rule = RateLimitRule(
            limit=self.settings.rate_limits.api_rate_limit_per_minute,
            window_seconds=60,
            message="API rate limit exceeded"
        )
        api_rule.key_func = lambda req: (
            f"api:{req.client.host}:{req.method}:{req.url.path}"
            if req.url.path.startswith("/api/")
            else f"general:{req.client.host}"
        )
        self.rules.append(api_rule)
        
        # Search endpoint specific limiting
        search_rule = RateLimitRule(
            limit=self.settings.rate_limits.search_rate_limit_per_minute,
            window_seconds=60,
            message="Search rate limit exceeded"
        )
        search_rule.key_func = lambda req: (
            f"search:{req.client.host}"
            if "/search" in req.url.path
            else f"general:{req.client.host}"
        )
        self.rules.append(search_rule)
    
    def _should_apply_rule(self, request: Request, rule: RateLimitRule) -> bool:
        """Determine if rule should be applied to request."""
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/metrics", "/ready", "/live"]:
            return False
        
        # Apply API-specific rules only to API endpoints
        if hasattr(rule, 'key_func') and 'api:' in str(rule.key_func):
            return request.url.path.startswith("/api/")
        
        # Apply search-specific rules only to search endpoints
        if hasattr(rule, 'key_func') and 'search:' in str(rule.key_func):
            return "/search" in request.url.path
        
        return True
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with rate limiting."""
        # Check each rate limiting rule
        for rule in self.rules:
            if not self._should_apply_rule(request, rule):
                continue
            
            # Get rate limit key for this request
            rate_limit_key = rule.key_func(request)
            
            # Check if request is rate limited
            is_limited, info = await self.rate_limit_manager.is_rate_limited(
                rate_limit_key,
                rule.limit,
                rule.window_seconds
            )
            
            if is_limited:
                # Create rate limit exceeded response
                headers = {}
                if rule.headers:
                    headers.update({
                        "X-RateLimit-Limit": str(rule.limit),
                        "X-RateLimit-Remaining": str(info["remaining"]),
                        "X-RateLimit-Reset": str(int(info["reset_time"])),
                        "Retry-After": str(rule.window_seconds)
                    })
                
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=rule.message,
                    headers=headers
                )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to successful responses
        if hasattr(response, 'headers'):
            # Add general rate limit info to response headers
            for rule in self.rules:
                if self._should_apply_rule(request, rule):
                    rate_limit_key = rule.key_func(request)
                    info = await self.rate_limit_manager.get_rate_limit_info(
                        rate_limit_key,
                        rule.limit,
                        rule.window_seconds
                    )
                    
                    if rule.headers:
                        response.headers["X-RateLimit-Limit"] = str(rule.limit)
                        response.headers["X-RateLimit-Remaining"] = str(info["remaining"])
                        response.headers["X-RateLimit-Reset"] = str(int(info["reset_time"]))
                    break  # Only add headers from first applicable rule
        
        return response


# Rate limiting decorator for specific endpoints
def rate_limit(
    limit: int,
    window_seconds: int = 60,
    key_func: Optional[Callable[[Request], str]] = None,
    message: str = "Rate limit exceeded"
):
    """
    Decorator for applying rate limiting to specific endpoints.
    
    Args:
        limit: Number of requests allowed
        window_seconds: Time window in seconds
        key_func: Function to generate rate limit key
        message: Error message when rate limited
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Get request from function arguments
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if not request:
                # If no request found, proceed without rate limiting
                return await func(*args, **kwargs)
            
            # Apply rate limiting
            rule = RateLimitRule(limit, window_seconds, key_func, message)
            rate_limit_manager = RateLimitManager()
            
            rate_limit_key = rule.key_func(request)
            is_limited, info = await rate_limit_manager.is_rate_limited(
                rate_limit_key,
                rule.limit,
                rule.window_seconds
            )
            
            if is_limited:
                headers = {
                    "X-RateLimit-Limit": str(rule.limit),
                    "X-RateLimit-Remaining": str(info["remaining"]),
                    "X-RateLimit-Reset": str(int(info["reset_time"])),
                    "Retry-After": str(rule.window_seconds)
                }
                
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=rule.message,
                    headers=headers
                )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


# Predefined rate limiting decorators
def api_rate_limit(func):
    """Apply API rate limiting to endpoint."""
    settings = get_settings()
    return rate_limit(
        settings.rate_limits.api_rate_limit_per_minute,
        60,
        message="API rate limit exceeded"
    )(func)


def search_rate_limit(func):
    """Apply search rate limiting to endpoint."""
    settings = get_settings()
    return rate_limit(
        settings.rate_limits.search_rate_limit_per_minute,
        60,
        message="Search rate limit exceeded"
    )(func)


def user_rate_limit(func):
    """Apply user-based rate limiting to endpoint."""
    settings = get_settings()
    rule = RateLimitRule(
        settings.rate_limits.rate_limit_per_minute,
        60,
        message="User rate limit exceeded"
    )
    rule.key_func = rule.user_key_func
    
    return rate_limit(
        rule.limit,
        rule.window_seconds,
        rule.key_func,
        rule.message
    )(func)