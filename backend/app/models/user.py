"""
User-related Pydantic models.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import Field, EmailStr, validator
from .base import BaseModel


class UserPreferences(BaseModel):
    """User preferences and settings."""
    
    default_search_engine: str = Field(default="openalex", description="Default search engine")
    results_per_page: int = Field(default=20, ge=5, le=100, description="Default results per page")
    citation_style: str = Field(default="apa", description="Preferred citation style")
    export_format: str = Field(default="bibtex", description="Default export format")
    email_notifications: bool = Field(default=True, description="Enable email notifications")
    
    # UI preferences
    theme: str = Field(default="light", pattern="^(light|dark|auto)$", description="UI theme")
    graph_layout: str = Field(default="force", description="Default graph layout")
    show_abstracts: bool = Field(default=True, description="Show abstracts in results")
    
    # Privacy settings
    profile_public: bool = Field(default=False, description="Make profile public")
    share_collections: bool = Field(default=False, description="Allow sharing collections")


class UserStats(BaseModel):
    """User activity statistics."""
    
    papers_saved: int = Field(default=0, description="Number of papers saved")
    searches_performed: int = Field(default=0, description="Number of searches performed")
    citations_exported: int = Field(default=0, description="Number of citations exported")
    collections_created: int = Field(default=0, description="Number of collections created")
    last_active: Optional[datetime] = Field(default=None, description="Last activity timestamp")


class User(BaseModel):
    """Complete user model."""
    
    # Core identifiers
    id: Optional[str] = Field(default=None, description="Internal user ID")
    email: EmailStr = Field(description="User email address")
    username: Optional[str] = Field(default=None, description="Username")
    
    # Profile information
    full_name: Optional[str] = Field(default=None, description="User's full name")
    affiliation: Optional[str] = Field(default=None, description="User's affiliation")
    orcid: Optional[str] = Field(default=None, description="ORCID identifier")
    bio: Optional[str] = Field(default=None, max_length=500, description="User bio")
    website: Optional[str] = Field(default=None, description="User's website")
    
    # Account status
    is_active: bool = Field(default=True, description="Account is active")
    is_verified: bool = Field(default=False, description="Email is verified")
    is_premium: bool = Field(default=False, description="Premium account")
    
    # Settings and preferences
    preferences: UserPreferences = Field(default_factory=UserPreferences, description="User preferences")
    stats: UserStats = Field(default_factory=UserStats, description="User statistics")
    
    # External integrations
    zotero_user_id: Optional[str] = Field(default=None, description="Zotero user ID")
    google_scholar_id: Optional[str] = Field(default=None, description="Google Scholar ID")
    
    # Metadata
    created_at: Optional[datetime] = Field(default=None, description="Account creation timestamp")
    updated_at: Optional[datetime] = Field(default=None, description="Last update timestamp")
    last_login: Optional[datetime] = Field(default=None, description="Last login timestamp")
    
    @validator("username")
    def validate_username(cls, v):
        """Validate username format."""
        if v:
            import re
            if not re.match("^[a-zA-Z0-9_.-]+$", v):
                raise ValueError("Username can only contain letters, numbers, dots, hyphens, and underscores")
            if len(v) < 3 or len(v) > 30:
                raise ValueError("Username must be between 3 and 30 characters")
        return v
    
    @validator("orcid")
    def validate_orcid(cls, v):
        """Validate ORCID format."""
        if v and not v.startswith("https://orcid.org/"):
            # Allow just the ID part and construct full URL
            if len(v) == 19 and v[4] == '-' and v[9] == '-' and v[14] == '-':
                return f"https://orcid.org/{v}"
        return v


class UserCreate(BaseModel):
    """Model for creating a new user."""
    
    email: EmailStr = Field(description="User email address")
    password: str = Field(description="User password", min_length=8)
    full_name: Optional[str] = Field(default=None, description="User's full name")
    username: Optional[str] = Field(default=None, description="Username")
    affiliation: Optional[str] = Field(default=None, description="User's affiliation")
    
    @validator("password")
    def validate_password(cls, v):
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        
        # Check for at least one letter and one number
        has_letter = any(c.isalpha() for c in v)
        has_number = any(c.isdigit() for c in v)
        
        if not (has_letter and has_number):
            raise ValueError("Password must contain at least one letter and one number")
        
        return v


class UserUpdate(BaseModel):
    """Model for updating user information."""
    
    full_name: Optional[str] = Field(default=None, description="User's full name")
    username: Optional[str] = Field(default=None, description="Username")
    affiliation: Optional[str] = Field(default=None, description="User's affiliation")
    bio: Optional[str] = Field(default=None, max_length=500, description="User bio")
    website: Optional[str] = Field(default=None, description="User's website")
    orcid: Optional[str] = Field(default=None, description="ORCID identifier")
    preferences: Optional[UserPreferences] = Field(default=None, description="User preferences")


class UserResponse(BaseModel):
    """Response model for user data."""
    
    id: str = Field(description="User ID")
    email: EmailStr = Field(description="User email")
    username: Optional[str] = Field(description="Username")
    full_name: Optional[str] = Field(description="User's full name")
    affiliation: Optional[str] = Field(description="User's affiliation")
    bio: Optional[str] = Field(description="User bio")
    website: Optional[str] = Field(description="User's website")
    orcid: Optional[str] = Field(description="ORCID identifier")
    is_active: bool = Field(description="Account is active")
    is_verified: bool = Field(description="Email is verified")
    is_premium: bool = Field(description="Premium account")
    preferences: UserPreferences = Field(description="User preferences")
    stats: UserStats = Field(description="User statistics")
    created_at: datetime = Field(description="Account creation timestamp")
    last_login: Optional[datetime] = Field(description="Last login timestamp")


class UserSession(BaseModel):
    """User session information."""
    
    user_id: str = Field(description="User ID")
    session_id: str = Field(description="Session ID")
    created_at: datetime = Field(description="Session creation timestamp")
    expires_at: datetime = Field(description="Session expiration timestamp")
    ip_address: Optional[str] = Field(default=None, description="Client IP address")
    user_agent: Optional[str] = Field(default=None, description="Client user agent")
    is_active: bool = Field(default=True, description="Session is active")
    
    # OAuth tokens for external services
    zotero_token: Optional[str] = Field(default=None, description="Zotero OAuth token")
    zotero_token_secret: Optional[str] = Field(default=None, description="Zotero OAuth token secret")
    
    @property
    def is_expired(self) -> bool:
        """Check if session is expired."""
        return datetime.utcnow() > self.expires_at


class TokenResponse(BaseModel):
    """JWT token response."""
    
    access_token: str = Field(description="JWT access token")
    refresh_token: str = Field(description="JWT refresh token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(description="Token expiration time in seconds")
    
    
class LoginRequest(BaseModel):
    """User login request."""
    
    email: EmailStr = Field(description="User email address")
    password: str = Field(description="User password")
    remember_me: bool = Field(default=False, description="Remember user session")


class PasswordChangeRequest(BaseModel):
    """Password change request."""
    
    current_password: str = Field(description="Current password")
    new_password: str = Field(description="New password", min_length=8)
    
    @validator("new_password")
    def validate_new_password(cls, v):
        """Validate new password strength."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        
        has_letter = any(c.isalpha() for c in v)
        has_number = any(c.isdigit() for c in v)
        
        if not (has_letter and has_number):
            raise ValueError("Password must contain at least one letter and one number")
        
        return v


class PasswordResetRequest(BaseModel):
    """Password reset request."""
    
    email: EmailStr = Field(description="User email address")


class PasswordResetConfirm(BaseModel):
    """Password reset confirmation."""
    
    token: str = Field(description="Reset token")
    new_password: str = Field(description="New password", min_length=8)
    
    @validator("new_password")
    def validate_new_password(cls, v):
        """Validate new password strength."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        
        has_letter = any(c.isalpha() for c in v)
        has_number = any(c.isdigit() for c in v)
        
        if not (has_letter and has_number):
            raise ValueError("Password must contain at least one letter and one number")
        
        return v