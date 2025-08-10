"""
Custom exception classes for the Citation Network Explorer API.
"""

from typing import Any, Dict, Optional, List
from fastapi import HTTPException, status


class APIException(HTTPException):
    """Base API exception class."""
    
    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ):
        self.error_code = error_code
        self.context = context or {}
        
        # Build detailed error response
        error_detail = {
            "message": detail,
            "error_code": error_code,
            "context": self.context
        }
        
        super().__init__(
            status_code=status_code,
            detail=error_detail,
            headers=headers
        )


class ValidationError(APIException):
    """Validation error exception."""
    
    def __init__(
        self,
        message: str = "Validation error",
        field_errors: Optional[List[Dict[str, str]]] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        self.field_errors = field_errors or []
        
        if context is None:
            context = {}
        context["field_errors"] = self.field_errors
        
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=message,
            error_code="VALIDATION_ERROR",
            context=context
        )


class NotFoundError(APIException):
    """Resource not found exception."""
    
    def __init__(
        self,
        resource_type: str = "Resource",
        resource_id: Optional[str] = None,
        message: Optional[str] = None
    ):
        if message is None:
            if resource_id:
                message = f"{resource_type} with ID '{resource_id}' not found"
            else:
                message = f"{resource_type} not found"
        
        context = {
            "resource_type": resource_type,
            "resource_id": resource_id
        }
        
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=message,
            error_code="RESOURCE_NOT_FOUND",
            context=context
        )


class AuthenticationError(APIException):
    """Authentication error exception."""
    
    def __init__(
        self,
        message: str = "Authentication required",
        auth_type: str = "Bearer"
    ):
        headers = {"WWW-Authenticate": auth_type}
        
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=message,
            error_code="AUTHENTICATION_ERROR",
            headers=headers
        )


class AuthorizationError(APIException):
    """Authorization error exception."""
    
    def __init__(
        self,
        message: str = "Insufficient permissions",
        required_permission: Optional[str] = None,
        resource: Optional[str] = None
    ):
        context = {
            "required_permission": required_permission,
            "resource": resource
        }
        
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=message,
            error_code="AUTHORIZATION_ERROR",
            context=context
        )


class ConflictError(APIException):
    """Resource conflict exception."""
    
    def __init__(
        self,
        message: str = "Resource conflict",
        resource_type: Optional[str] = None,
        conflicting_field: Optional[str] = None
    ):
        context = {
            "resource_type": resource_type,
            "conflicting_field": conflicting_field
        }
        
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            detail=message,
            error_code="RESOURCE_CONFLICT",
            context=context
        )


class RateLimitError(APIException):
    """Rate limit exceeded exception."""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        limit: Optional[int] = None,
        window_seconds: Optional[int] = None,
        retry_after: Optional[int] = None
    ):
        context = {
            "limit": limit,
            "window_seconds": window_seconds
        }
        
        headers = {}
        if retry_after:
            headers["Retry-After"] = str(retry_after)
        
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=message,
            error_code="RATE_LIMIT_EXCEEDED",
            context=context,
            headers=headers
        )


class ExternalServiceError(APIException):
    """External service error exception."""
    
    def __init__(
        self,
        service_name: str,
        message: str = "External service error",
        service_status_code: Optional[int] = None,
        service_response: Optional[str] = None
    ):
        context = {
            "service_name": service_name,
            "service_status_code": service_status_code,
            "service_response": service_response
        }
        
        # Map service errors to appropriate HTTP status codes
        if service_status_code:
            if service_status_code >= 500:
                status_code = status.HTTP_502_BAD_GATEWAY
            elif service_status_code == 404:
                status_code = status.HTTP_404_NOT_FOUND
            elif service_status_code >= 400:
                status_code = status.HTTP_400_BAD_REQUEST
            else:
                status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        else:
            status_code = status.HTTP_502_BAD_GATEWAY
        
        super().__init__(
            status_code=status_code,
            detail=f"{service_name}: {message}",
            error_code="EXTERNAL_SERVICE_ERROR",
            context=context
        )


class DatabaseError(APIException):
    """Database operation error exception."""
    
    def __init__(
        self,
        message: str = "Database error",
        operation: Optional[str] = None,
        table_or_collection: Optional[str] = None
    ):
        context = {
            "operation": operation,
            "table_or_collection": table_or_collection
        }
        
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=message,
            error_code="DATABASE_ERROR",
            context=context
        )


class CacheError(APIException):
    """Cache operation error exception."""
    
    def __init__(
        self,
        message: str = "Cache error",
        operation: Optional[str] = None,
        key: Optional[str] = None
    ):
        context = {
            "operation": operation,
            "key": key
        }
        
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=message,
            error_code="CACHE_ERROR",
            context=context
        )


class BusinessLogicError(APIException):
    """Business logic error exception."""
    
    def __init__(
        self,
        message: str,
        error_code: str,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message,
            error_code=error_code,
            context=context
        )


# Domain-specific exceptions
class PaperNotFoundError(NotFoundError):
    """Paper not found exception."""
    
    def __init__(self, paper_id: Optional[str] = None, doi: Optional[str] = None):
        if doi:
            super().__init__(
                resource_type="Paper",
                resource_id=doi,
                message=f"Paper with DOI '{doi}' not found"
            )
        else:
            super().__init__(
                resource_type="Paper",
                resource_id=paper_id
            )


class UserNotFoundError(NotFoundError):
    """User not found exception."""
    
    def __init__(self, user_id: Optional[str] = None, email: Optional[str] = None):
        if email:
            super().__init__(
                resource_type="User",
                resource_id=email,
                message=f"User with email '{email}' not found"
            )
        else:
            super().__init__(
                resource_type="User",
                resource_id=user_id
            )


class ZoteroIntegrationError(ExternalServiceError):
    """Zotero integration error exception."""
    
    def __init__(
        self,
        message: str = "Zotero API error",
        zotero_error_code: Optional[str] = None,
        service_status_code: Optional[int] = None
    ):
        context = {"zotero_error_code": zotero_error_code}
        
        super().__init__(
            service_name="Zotero",
            message=message,
            service_status_code=service_status_code,
        )
        self.context.update(context)


class SearchError(APIException):
    """Search operation error exception."""
    
    def __init__(
        self,
        message: str = "Search error",
        search_engine: Optional[str] = None,
        query: Optional[str] = None
    ):
        context = {
            "search_engine": search_engine,
            "query": query
        }
        
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=message,
            error_code="SEARCH_ERROR",
            context=context
        )


class ExportError(APIException):
    """Export operation error exception."""
    
    def __init__(
        self,
        message: str = "Export error",
        export_format: Optional[str] = None,
        item_count: Optional[int] = None
    ):
        context = {
            "export_format": export_format,
            "item_count": item_count
        }
        
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=message,
            error_code="EXPORT_ERROR",
            context=context
        )


# Alias for backwards compatibility
APIError = ExternalServiceError

# Exception mapping for automatic error handling
EXCEPTION_STATUS_CODES = {
    ValidationError: status.HTTP_422_UNPROCESSABLE_ENTITY,
    NotFoundError: status.HTTP_404_NOT_FOUND,
    AuthenticationError: status.HTTP_401_UNAUTHORIZED,
    AuthorizationError: status.HTTP_403_FORBIDDEN,
    ConflictError: status.HTTP_409_CONFLICT,
    RateLimitError: status.HTTP_429_TOO_MANY_REQUESTS,
    ExternalServiceError: status.HTTP_502_BAD_GATEWAY,
    DatabaseError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    CacheError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    BusinessLogicError: status.HTTP_400_BAD_REQUEST,
    SearchError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    ExportError: status.HTTP_500_INTERNAL_SERVER_ERROR
}