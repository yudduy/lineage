"""
Pydantic models for the Citation Network Explorer API.
"""

from .base import BaseModel
from .paper import Paper, PaperResponse, PaperSearchRequest, PaperEdgesResponse
from .user import User, UserResponse, UserSession
from .zotero import ZoteroCollection, ZoteroItem, ZoteroAuthRequest, ZoteroAuthResponse
from .health import HealthCheck, HealthStatus
from .common import APIResponse, ErrorResponse, PaginationParams

__all__ = [
    "BaseModel",
    "Paper",
    "PaperResponse", 
    "PaperSearchRequest",
    "PaperEdgesResponse",
    "User",
    "UserResponse",
    "UserSession",
    "ZoteroCollection",
    "ZoteroItem", 
    "ZoteroAuthRequest",
    "ZoteroAuthResponse",
    "HealthCheck",
    "HealthStatus",
    "APIResponse",
    "ErrorResponse",
    "PaginationParams",
]