"""
Structured logging configuration using structlog.
"""

import sys
import logging
from typing import Any, Dict, Optional
from datetime import datetime

import structlog
from structlog.types import FilteringBoundLogger

from ..core.config import get_settings


class CustomTimestampProcessor:
    """Add custom timestamp to log records."""
    
    def __call__(self, logger: FilteringBoundLogger, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        event_dict["timestamp"] = datetime.utcnow().isoformat() + "Z"
        return event_dict


class CorrelationIdProcessor:
    """Add correlation ID to log records."""
    
    def __call__(self, logger: FilteringBoundLogger, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        # In a real implementation, you'd get correlation ID from context
        # For now, we'll use a placeholder
        if "correlation_id" not in event_dict:
            event_dict["correlation_id"] = "N/A"
        return event_dict


class ServiceContextProcessor:
    """Add service context information."""
    
    def __init__(self):
        self.settings = get_settings()
    
    def __call__(self, logger: FilteringBoundLogger, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        event_dict.update({
            "service": self.settings.app_name,
            "version": self.settings.app_version,
            "environment": self.settings.environment
        })
        return event_dict


def setup_logging():
    """Configure structured logging."""
    settings = get_settings()
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
    )
    
    # Configure processors based on format
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        CustomTimestampProcessor(),
        CorrelationIdProcessor(),
        ServiceContextProcessor(),
    ]
    
    if settings.log_format == "json":
        # JSON formatting for production
        processors.extend([
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer()
        ])
    else:
        # Human-readable formatting for development
        processors.extend([
            structlog.dev.ConsoleRenderer(colors=True)
        ])
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> FilteringBoundLogger:
    """Get a configured logger instance."""
    return structlog.get_logger(name)


class RequestLoggingMixin:
    """Mixin for adding request logging capabilities."""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
    
    def log_request(
        self,
        method: str,
        path: str,
        user_id: Optional[str] = None,
        **kwargs
    ):
        """Log incoming request."""
        self.logger.info(
            "Request received",
            method=method,
            path=path,
            user_id=user_id,
            **kwargs
        )
    
    def log_response(
        self,
        method: str,
        path: str,
        status_code: int,
        response_time_ms: float,
        user_id: Optional[str] = None,
        **kwargs
    ):
        """Log response."""
        log_level = "info"
        if status_code >= 500:
            log_level = "error"
        elif status_code >= 400:
            log_level = "warning"
        
        getattr(self.logger, log_level)(
            "Request completed",
            method=method,
            path=path,
            status_code=status_code,
            response_time_ms=response_time_ms,
            user_id=user_id,
            **kwargs
        )
    
    def log_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Log error with context."""
        self.logger.error(
            "Error occurred",
            error_type=type(error).__name__,
            error_message=str(error),
            context=context or {},
            **kwargs,
            exc_info=True
        )
    
    def log_database_operation(
        self,
        operation: str,
        table_or_collection: str,
        duration_ms: float,
        record_count: Optional[int] = None,
        **kwargs
    ):
        """Log database operation."""
        self.logger.info(
            "Database operation",
            operation=operation,
            target=table_or_collection,
            duration_ms=duration_ms,
            record_count=record_count,
            **kwargs
        )
    
    def log_external_api_call(
        self,
        service: str,
        endpoint: str,
        method: str,
        status_code: int,
        response_time_ms: float,
        **kwargs
    ):
        """Log external API call."""
        log_level = "info"
        if status_code >= 500:
            log_level = "error"
        elif status_code >= 400:
            log_level = "warning"
        
        getattr(self.logger, log_level)(
            "External API call",
            service=service,
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            response_time_ms=response_time_ms,
            **kwargs
        )


class SecurityLogger:
    """Specialized logger for security events."""
    
    def __init__(self):
        self.logger = get_logger("security")
    
    def log_authentication_attempt(
        self,
        email: str,
        success: bool,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        """Log authentication attempt."""
        self.logger.info(
            "Authentication attempt",
            email=email,
            success=success,
            ip_address=ip_address,
            user_agent=user_agent
        )
    
    def log_authorization_failure(
        self,
        user_id: str,
        resource: str,
        action: str,
        ip_address: Optional[str] = None
    ):
        """Log authorization failure."""
        self.logger.warning(
            "Authorization denied",
            user_id=user_id,
            resource=resource,
            action=action,
            ip_address=ip_address
        )
    
    def log_suspicious_activity(
        self,
        activity_type: str,
        details: Dict[str, Any],
        ip_address: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """Log suspicious activity."""
        self.logger.warning(
            "Suspicious activity detected",
            activity_type=activity_type,
            details=details,
            ip_address=ip_address,
            user_id=user_id
        )
    
    def log_rate_limit_exceeded(
        self,
        ip_address: str,
        endpoint: str,
        limit: int,
        window_seconds: int
    ):
        """Log rate limit exceeded."""
        self.logger.warning(
            "Rate limit exceeded",
            ip_address=ip_address,
            endpoint=endpoint,
            limit=limit,
            window_seconds=window_seconds
        )


class PerformanceLogger:
    """Logger for performance metrics."""
    
    def __init__(self):
        self.logger = get_logger("performance")
    
    def log_slow_query(
        self,
        query_type: str,
        duration_ms: float,
        threshold_ms: float = 1000,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log slow database query."""
        if duration_ms > threshold_ms:
            self.logger.warning(
                "Slow query detected",
                query_type=query_type,
                duration_ms=duration_ms,
                threshold_ms=threshold_ms,
                details=details or {}
            )
    
    def log_cache_performance(
        self,
        operation: str,
        hit: bool,
        duration_ms: float,
        key: Optional[str] = None
    ):
        """Log cache performance."""
        self.logger.info(
            "Cache operation",
            operation=operation,
            hit=hit,
            duration_ms=duration_ms,
            key=key
        )
    
    def log_external_api_performance(
        self,
        service: str,
        endpoint: str,
        duration_ms: float,
        status_code: int
    ):
        """Log external API performance."""
        log_level = "info"
        if duration_ms > 5000:  # 5 seconds threshold
            log_level = "warning"
        
        getattr(self.logger, log_level)(
            "External API performance",
            service=service,
            endpoint=endpoint,
            duration_ms=duration_ms,
            status_code=status_code
        )


# Global logger instances
security_logger = SecurityLogger()
performance_logger = PerformanceLogger()


def get_security_logger() -> SecurityLogger:
    """Get security logger instance."""
    return security_logger


def get_performance_logger() -> PerformanceLogger:
    """Get performance logger instance."""
    return performance_logger