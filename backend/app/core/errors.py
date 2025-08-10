"""
Global error handlers for the FastAPI application.
"""

import traceback
from typing import Any, Dict
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError, HTTPException
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import ValidationError

from ..utils.exceptions import APIException
from ..utils.logger import get_logger
from ..models.common import ErrorResponse

logger = get_logger(__name__)


async def api_exception_handler(request: Request, exc: APIException) -> JSONResponse:
    """Handle custom API exceptions."""
    
    # Log the error
    logger.error(
        "API exception occurred",
        error_type=type(exc).__name__,
        error_message=str(exc.detail),
        error_code=getattr(exc, 'error_code', None),
        context=getattr(exc, 'context', {}),
        path=request.url.path,
        method=request.method,
        client_ip=request.client.host if request.client else "unknown",
        correlation_id=getattr(request.state, "correlation_id", None)
    )
    
    # Format error response
    if isinstance(exc.detail, dict):
        error_response = ErrorResponse(
            error=exc.detail.get("error_code", "API_ERROR"),
            message=exc.detail.get("message", "An error occurred"),
            details=exc.detail.get("context", {})
        )
    else:
        error_response = ErrorResponse(
            error=getattr(exc, 'error_code', 'API_ERROR'),
            message=str(exc.detail),
            details=getattr(exc, 'context', {})
        )
    
    # Add correlation ID if available
    if hasattr(request.state, 'correlation_id'):
        error_response.details["correlation_id"] = request.state.correlation_id
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump(),
        headers=getattr(exc, 'headers', None)
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle standard HTTP exceptions."""
    
    logger.warning(
        "HTTP exception occurred",
        status_code=exc.status_code,
        error_message=str(exc.detail),
        path=request.url.path,
        method=request.method,
        client_ip=request.client.host if request.client else "unknown",
        correlation_id=getattr(request.state, "correlation_id", None)
    )
    
    error_response = ErrorResponse(
        error="HTTP_ERROR",
        message=str(exc.detail),
        code=exc.status_code,
        details={"correlation_id": getattr(request.state, "correlation_id", None)}
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump(),
        headers=getattr(exc, 'headers', None)
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
) -> JSONResponse:
    """Handle request validation errors."""
    
    logger.warning(
        "Validation error occurred",
        errors=exc.errors(),
        path=request.url.path,
        method=request.method,
        client_ip=request.client.host if request.client else "unknown",
        correlation_id=getattr(request.state, "correlation_id", None)
    )
    
    # Format validation errors
    field_errors = []
    for error in exc.errors():
        field_errors.append({
            "field": ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })
    
    error_response = ErrorResponse(
        error="VALIDATION_ERROR",
        message="Request validation failed",
        details={
            "field_errors": field_errors,
            "correlation_id": getattr(request.state, "correlation_id", None)
        }
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_response.model_dump()
    )


async def pydantic_validation_exception_handler(
    request: Request,
    exc: ValidationError
) -> JSONResponse:
    """Handle Pydantic validation errors."""
    
    logger.warning(
        "Pydantic validation error occurred",
        errors=exc.errors(),
        path=request.url.path,
        method=request.method,
        client_ip=request.client.host if request.client else "unknown",
        correlation_id=getattr(request.state, "correlation_id", None)
    )
    
    # Format validation errors
    field_errors = []
    for error in exc.errors():
        field_errors.append({
            "field": ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })
    
    error_response = ErrorResponse(
        error="VALIDATION_ERROR",
        message="Data validation failed",
        details={
            "field_errors": field_errors,
            "correlation_id": getattr(request.state, "correlation_id", None)
        }
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_response.model_dump()
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle all other unhandled exceptions."""
    
    # Log the full exception with traceback
    logger.error(
        "Unhandled exception occurred",
        error_type=type(exc).__name__,
        error_message=str(exc),
        path=request.url.path,
        method=request.method,
        client_ip=request.client.host if request.client else "unknown",
        correlation_id=getattr(request.state, "correlation_id", None),
        traceback=traceback.format_exc(),
        exc_info=True
    )
    
    # Don't expose internal error details in production
    from ..core.config import get_settings
    settings = get_settings()
    
    if settings.is_production:
        error_message = "An unexpected error occurred"
        details = {"correlation_id": getattr(request.state, "correlation_id", None)}
    else:
        error_message = f"{type(exc).__name__}: {str(exc)}"
        details = {
            "traceback": traceback.format_exc(),
            "correlation_id": getattr(request.state, "correlation_id", None)
        }
    
    error_response = ErrorResponse(
        error="INTERNAL_SERVER_ERROR",
        message=error_message,
        code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        details=details
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.model_dump()
    )


def setup_error_handlers(app: FastAPI):
    """Setup global error handlers for the FastAPI application."""
    
    # Custom API exceptions
    app.add_exception_handler(APIException, api_exception_handler)
    
    # Standard HTTP exceptions
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    
    # Validation errors
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(ValidationError, pydantic_validation_exception_handler)
    
    # Catch-all for unexpected errors
    app.add_exception_handler(Exception, generic_exception_handler)
    
    logger.info("Global error handlers configured")