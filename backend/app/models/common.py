"""
Common models used across the application.
"""

from typing import Any, Dict, Generic, List, Optional, TypeVar, Union
from pydantic import Field, validator
from .base import BaseModel

T = TypeVar("T")


class PaginationParams(BaseModel):
    """Pagination parameters for API endpoints."""
    
    skip: int = Field(default=0, ge=0, description="Number of items to skip")
    limit: int = Field(default=20, ge=1, le=100, description="Number of items to return")
    
    @property
    def offset(self) -> int:
        """Alias for skip for database queries."""
        return self.skip


class APIResponse(BaseModel, Generic[T]):
    """Generic API response wrapper."""
    
    success: bool = Field(default=True, description="Whether the request was successful")
    message: str = Field(default="", description="Response message")
    data: Optional[T] = Field(default=None, description="Response data")
    meta: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class ErrorResponse(BaseModel):
    """Error response model."""
    
    success: bool = Field(default=False, description="Always false for error responses")
    error: str = Field(description="Error type")
    message: str = Field(description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    code: Optional[int] = Field(default=None, description="Error code")


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated response wrapper."""
    
    items: List[T] = Field(description="List of items")
    total: int = Field(description="Total number of items")
    skip: int = Field(description="Number of items skipped")
    limit: int = Field(description="Number of items per page")
    has_more: bool = Field(description="Whether there are more items")
    
    @validator("has_more", pre=False, always=True)
    def calculate_has_more(cls, v, values):
        """Calculate if there are more items based on pagination."""
        if "total" in values and "skip" in values and "limit" in values:
            return values["skip"] + values["limit"] < values["total"]
        return v


class SearchParams(BaseModel):
    """Common search parameters with security validation."""
    
    q: str = Field(description="Search query", min_length=1, max_length=500)
    fields: Optional[List[str]] = Field(default=None, description="Fields to search in")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Additional filters")
    sort_by: Optional[str] = Field(default=None, description="Field to sort by")
    sort_order: Optional[str] = Field(default="desc", pattern="^(asc|desc)$", description="Sort order")
    
    @validator("q")
    def validate_query(cls, v):
        """Validate search query for security."""
        if not v or not v.strip():
            raise ValueError("Search query cannot be empty")
        
        # Remove excessive whitespace
        v = ' '.join(v.split())
        
        # Check for potential injection patterns
        dangerous_patterns = [
            'javascript:', 'vbscript:', 'data:', 'file:', 
            '<script', '</script>', 'onload=', 'onerror=',
            'UNION', 'DROP', 'DELETE', 'INSERT', 'UPDATE',
            'EXEC', 'EXECUTE', '--', '/*', '*/',
            'MATCH', 'CREATE', 'MERGE', 'SET'  # Cypher patterns
        ]
        
        v_lower = v.lower()
        for pattern in dangerous_patterns:
            if pattern.lower() in v_lower:
                raise ValueError(f"Search query contains potentially dangerous pattern: {pattern}")
        
        return v.strip()
    
    @validator("fields")
    def validate_fields(cls, v):
        """Validate search fields."""
        if v and len(v) > 10:
            raise ValueError("Maximum 10 search fields allowed")
        
        if v:
            # Whitelist allowed field names
            allowed_fields = [
                'title', 'abstract', 'authors', 'keywords', 'doi', 'year',
                'venue', 'citation_count', 'open_access', 'affiliations'
            ]
            for field in v:
                if field not in allowed_fields:
                    raise ValueError(f"Invalid search field: {field}")
        
        return v
    
    @validator("sort_by")
    def validate_sort_by(cls, v):
        """Validate sort field."""
        if v:
            # Whitelist allowed sort fields
            allowed_sort_fields = [
                'relevance', 'publication_year', 'citation_count', 
                'title', 'authors', 'created_at', 'updated_at'
            ]
            if v not in allowed_sort_fields:
                raise ValueError(f"Invalid sort field: {v}")
        return v


class BulkResponse(BaseModel):
    """Response for bulk operations."""
    
    total_processed: int = Field(description="Total number of items processed")
    successful: int = Field(description="Number of successful operations")
    failed: int = Field(description="Number of failed operations")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="List of errors")
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_processed == 0:
            return 0.0
        return (self.successful / self.total_processed) * 100


class ExportRequest(BaseModel):
    """Request for data export operations."""
    
    format: str = Field(description="Export format", pattern="^(json|csv|bibtex|ris)$")
    items: List[str] = Field(description="List of item IDs to export", min_items=1, max_items=1000)
    fields: Optional[List[str]] = Field(default=None, description="Specific fields to export")
    filename: Optional[str] = Field(default=None, description="Custom filename for export")
    
    @validator("filename")
    def validate_filename(cls, v):
        """Validate filename format."""
        if v:
            # Remove potentially dangerous characters
            import re
            v = re.sub(r'[<>:"/\\|?*]', '', v)
            if not v:
                raise ValueError("Filename contains only invalid characters")
        return v