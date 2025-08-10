"""
Zotero integration Pydantic models.
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pydantic import Field, HttpUrl, validator
from .base import BaseModel


class ZoteroCreator(BaseModel):
    """Zotero item creator (author)."""
    
    creator_type: str = Field(description="Type of creator (author, editor, etc.)")
    first_name: Optional[str] = Field(default=None, description="Creator's first name")
    last_name: Optional[str] = Field(default=None, description="Creator's last name")
    name: Optional[str] = Field(default=None, description="Full name (for organizations)")
    
    @validator("name", pre=True, always=True)
    def build_full_name(cls, v, values):
        """Build full name if not provided."""
        if not v and values.get("first_name") and values.get("last_name"):
            return f"{values['first_name']} {values['last_name']}"
        return v


class ZoteroItemData(BaseModel):
    """Zotero item data structure."""
    
    # Core identifiers
    key: str = Field(description="Zotero item key")
    version: int = Field(description="Item version")
    item_type: str = Field(description="Type of item (book, journalArticle, etc.)")
    
    # Basic metadata
    title: Optional[str] = Field(default=None, description="Item title")
    creators: List[ZoteroCreator] = Field(default_factory=list, description="List of creators")
    abstract_note: Optional[str] = Field(default=None, description="Abstract or notes")
    
    # Publication details
    publication_title: Optional[str] = Field(default=None, description="Journal/publication title")
    publisher: Optional[str] = Field(default=None, description="Publisher")
    date: Optional[str] = Field(default=None, description="Publication date")
    pages: Optional[str] = Field(default=None, description="Page range")
    volume: Optional[str] = Field(default=None, description="Volume")
    issue: Optional[str] = Field(default=None, description="Issue")
    
    # Identifiers
    doi: Optional[str] = Field(default=None, description="DOI")
    isbn: Optional[str] = Field(default=None, description="ISBN")
    issn: Optional[str] = Field(default=None, description="ISSN")
    url: Optional[HttpUrl] = Field(default=None, description="URL")
    
    # Additional fields (Zotero has many item-type specific fields)
    extra: Optional[str] = Field(default=None, description="Extra field for additional data")
    tags: List[Dict[str, Any]] = Field(default_factory=list, description="Item tags")
    collections: List[str] = Field(default_factory=list, description="Collection keys")
    
    # Metadata
    date_added: Optional[datetime] = Field(default=None, description="Date added to Zotero")
    date_modified: Optional[datetime] = Field(default=None, description="Date last modified")


class ZoteroItem(BaseModel):
    """Complete Zotero item with metadata."""
    
    key: str = Field(description="Zotero item key")
    version: int = Field(description="Item version") 
    library: Dict[str, Any] = Field(description="Library information")
    links: Dict[str, Any] = Field(default_factory=dict, description="Related links")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Metadata")
    data: ZoteroItemData = Field(description="Item data")


class ZoteroCollection(BaseModel):
    """Zotero collection information."""
    
    key: str = Field(description="Collection key")
    version: int = Field(description="Collection version")
    library: Dict[str, Any] = Field(description="Library information")
    links: Dict[str, Any] = Field(default_factory=dict, description="Related links")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Metadata")
    data: Dict[str, Any] = Field(description="Collection data")
    
    @property
    def name(self) -> Optional[str]:
        """Get collection name from data."""
        return self.data.get("name")
    
    @property 
    def parent_collection(self) -> Optional[str]:
        """Get parent collection key."""
        return self.data.get("parentCollection")


class ZoteroAuthRequest(BaseModel):
    """Zotero OAuth authentication request."""
    
    oauth_token: Optional[str] = Field(default=None, description="OAuth token from callback")
    oauth_verifier: Optional[str] = Field(default=None, description="OAuth verifier from callback")
    
    
class ZoteroAuthResponse(BaseModel):
    """Zotero OAuth authentication response."""
    
    success: bool = Field(description="Authentication success status")
    user_id: Optional[str] = Field(default=None, description="Zotero user ID")
    username: Optional[str] = Field(default=None, description="Zotero username")
    access_token: Optional[str] = Field(default=None, description="OAuth access token")
    token_secret: Optional[str] = Field(default=None, description="OAuth token secret")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class ZoteroCollectionRequest(BaseModel):
    """Request to get Zotero collections."""
    
    user_id: Optional[str] = Field(default=None, description="Specific user ID")
    include_items: bool = Field(default=False, description="Include item counts")
    sort: Optional[str] = Field(default="title", description="Sort field")
    direction: Optional[str] = Field(default="asc", pattern="^(asc|desc)$", description="Sort direction")


class ZoteroItemsRequest(BaseModel):
    """Request to get Zotero items."""
    
    collection_key: Optional[str] = Field(default=None, description="Collection key to filter by")
    item_type: Optional[str] = Field(default=None, description="Item type filter")
    tag: Optional[str] = Field(default=None, description="Tag filter")
    q: Optional[str] = Field(default=None, description="Search query")
    qmode: Optional[str] = Field(default="titleCreatorYear", description="Search mode")
    include_trashed: bool = Field(default=False, description="Include trashed items")
    since: Optional[int] = Field(default=None, description="Modified since version")
    format: Optional[str] = Field(default="json", description="Response format")
    limit: int = Field(default=25, ge=1, le=100, description="Number of items to return")
    start: int = Field(default=0, ge=0, description="Starting index")


class ZoteroAddItemsRequest(BaseModel):
    """Request to add items to Zotero."""
    
    items: List[Dict[str, Any]] = Field(description="Items to add", min_items=1, max_items=50)
    collection_key: Optional[str] = Field(default=None, description="Collection to add items to")
    
    @validator("items")
    def validate_items(cls, v):
        """Validate items structure."""
        for item in v:
            if "itemType" not in item:
                raise ValueError("Each item must have an 'itemType' field")
        return v


class ZoteroImportRequest(BaseModel):
    """Request to import papers from Zotero."""
    
    collection_keys: Optional[List[str]] = Field(default=None, description="Collection keys to import from")
    item_keys: Optional[List[str]] = Field(default=None, description="Specific item keys to import")
    import_mode: str = Field(default="merge", pattern="^(merge|replace|skip)$", description="Import mode")
    include_attachments: bool = Field(default=False, description="Include file attachments")
    include_notes: bool = Field(default=True, description="Include notes")
    
    @validator("item_keys")
    def validate_item_keys_limit(cls, v):
        """Validate item keys list length."""
        if v and len(v) > 100:
            raise ValueError("Maximum 100 item keys allowed per import")
        return v


class ZoteroSyncStatus(BaseModel):
    """Zotero synchronization status."""
    
    last_sync: Optional[datetime] = Field(default=None, description="Last synchronization timestamp")
    sync_in_progress: bool = Field(default=False, description="Sync currently in progress")
    items_synced: int = Field(default=0, description="Number of items synchronized")
    collections_synced: int = Field(default=0, description="Number of collections synchronized")
    errors: List[str] = Field(default_factory=list, description="Sync errors")
    
    
class ZoteroExportRequest(BaseModel):
    """Request to export papers to Zotero."""
    
    paper_ids: List[str] = Field(description="Paper IDs to export", min_items=1, max_items=50)
    collection_key: Optional[str] = Field(default=None, description="Target collection")
    create_collection: Optional[str] = Field(default=None, description="Name for new collection")
    include_pdf: bool = Field(default=False, description="Include PDF attachments if available")
    update_existing: bool = Field(default=True, description="Update existing items")
    
    @validator("create_collection")
    def validate_collection_name(cls, v):
        """Validate collection name."""
        if v and (len(v.strip()) < 1 or len(v.strip()) > 255):
            raise ValueError("Collection name must be between 1 and 255 characters")
        return v.strip() if v else v


class ZoteroWebhookEvent(BaseModel):
    """Zotero webhook event data."""
    
    event_type: str = Field(description="Type of webhook event")
    user_id: str = Field(description="Zotero user ID")
    library_id: str = Field(description="Library ID")
    object_type: str = Field(description="Object type (item, collection, etc.)")
    object_key: str = Field(description="Object key")
    version: int = Field(description="Object version")
    timestamp: datetime = Field(description="Event timestamp")