"""
Paper-related Pydantic models.
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime, date
from pydantic import Field, validator, HttpUrl
from .base import BaseModel
from .common import PaginationParams


class Author(BaseModel):
    """Author information."""
    
    name: str = Field(description="Author's name")
    orcid: Optional[str] = Field(default=None, description="ORCID identifier")
    affiliation: Optional[str] = Field(default=None, description="Author's affiliation")
    email: Optional[str] = Field(default=None, description="Author's email")
    
    @validator("orcid")
    def validate_orcid(cls, v):
        """Validate ORCID format."""
        if v and not v.startswith("https://orcid.org/"):
            # Allow just the ID part and construct full URL
            if len(v) == 19 and v[4] == '-' and v[9] == '-' and v[14] == '-':
                return f"https://orcid.org/{v}"
        return v


class Journal(BaseModel):
    """Journal information."""
    
    name: str = Field(description="Journal name")
    issn: Optional[str] = Field(default=None, description="ISSN")
    publisher: Optional[str] = Field(default=None, description="Publisher")
    impact_factor: Optional[float] = Field(default=None, description="Impact factor")


class CitationCount(BaseModel):
    """Citation count from different sources."""
    
    total: int = Field(default=0, description="Total citation count")
    crossref: Optional[int] = Field(default=None, description="CrossRef citation count")
    semantic_scholar: Optional[int] = Field(default=None, description="Semantic Scholar citation count")
    openalex: Optional[int] = Field(default=None, description="OpenAlex citation count")
    last_updated: Optional[datetime] = Field(default=None, description="Last update timestamp")


class Paper(BaseModel):
    """Complete paper model."""
    
    # Core identifiers
    id: Optional[str] = Field(default=None, description="Internal paper ID")
    doi: Optional[str] = Field(default=None, description="DOI")
    pmid: Optional[str] = Field(default=None, description="PubMed ID")
    arxiv_id: Optional[str] = Field(default=None, description="arXiv ID")
    openalex_id: Optional[str] = Field(default=None, description="OpenAlex ID")
    semantic_scholar_id: Optional[str] = Field(default=None, description="Semantic Scholar ID")
    
    # Basic metadata
    title: str = Field(description="Paper title")
    abstract: Optional[str] = Field(default=None, description="Paper abstract")
    authors: List[Author] = Field(default_factory=list, description="List of authors")
    
    # Publication details
    journal: Optional[Journal] = Field(default=None, description="Journal information")
    publication_date: Optional[Union[date, str]] = Field(default=None, description="Publication date")
    publication_year: Optional[int] = Field(default=None, description="Publication year")
    volume: Optional[str] = Field(default=None, description="Journal volume")
    issue: Optional[str] = Field(default=None, description="Journal issue")
    pages: Optional[str] = Field(default=None, description="Page range")
    
    # URLs and links
    url: Optional[HttpUrl] = Field(default=None, description="Primary URL")
    pdf_url: Optional[HttpUrl] = Field(default=None, description="PDF URL")
    open_access_url: Optional[HttpUrl] = Field(default=None, description="Open access URL")
    
    # Citation information
    citation_count: CitationCount = Field(default_factory=CitationCount, description="Citation counts")
    references: List[str] = Field(default_factory=list, description="List of reference IDs")
    cited_by: List[str] = Field(default_factory=list, description="List of citing paper IDs")
    
    # Classification and topics
    subjects: List[str] = Field(default_factory=list, description="Subject areas")
    keywords: List[str] = Field(default_factory=list, description="Keywords")
    concepts: List[Dict[str, Any]] = Field(default_factory=list, description="Semantic concepts")
    
    # Metadata
    language: Optional[str] = Field(default=None, description="Paper language")
    paper_type: Optional[str] = Field(default=None, description="Type of paper")
    is_open_access: bool = Field(default=False, description="Whether paper is open access")
    
    # System metadata
    created_at: Optional[datetime] = Field(default=None, description="Record creation timestamp")
    updated_at: Optional[datetime] = Field(default=None, description="Record update timestamp")
    
    @validator("publication_year")
    def validate_publication_year(cls, v):
        """Validate publication year is reasonable."""
        if v and (v < 1000 or v > datetime.now().year + 2):
            raise ValueError("Invalid publication year")
        return v
    
    @validator("doi")
    def validate_doi(cls, v):
        """Validate DOI format."""
        if v and not v.startswith("10."):
            raise ValueError("DOI must start with '10.'")
        return v


class PaperSearchRequest(BaseModel):
    """Request model for paper search."""
    
    # Search parameters
    query: Optional[str] = Field(default=None, description="Search query")
    title: Optional[str] = Field(default=None, description="Title search")
    authors: Optional[str] = Field(default=None, description="Author search")
    journal: Optional[str] = Field(default=None, description="Journal search")
    
    # Identifiers
    doi: Optional[str] = Field(default=None, description="DOI to search for")
    dois: Optional[List[str]] = Field(default=None, description="List of DOIs to search for")
    
    # Filters
    publication_year_min: Optional[int] = Field(default=None, description="Minimum publication year")
    publication_year_max: Optional[int] = Field(default=None, description="Maximum publication year")
    citation_count_min: Optional[int] = Field(default=None, description="Minimum citation count")
    is_open_access: Optional[bool] = Field(default=None, description="Filter by open access")
    
    # Sorting and pagination
    sort_by: Optional[str] = Field(default="relevance", description="Sort field")
    sort_order: Optional[str] = Field(default="desc", pattern="^(asc|desc)$", description="Sort order")
    pagination: PaginationParams = Field(default_factory=PaginationParams)
    
    @validator("dois")
    def validate_dois_list(cls, v):
        """Validate DOI list length."""
        if v and len(v) > 50:
            raise ValueError("Maximum 50 DOIs allowed per request")
        return v


class PaperEdgesRequest(BaseModel):
    """Request model for getting paper citation edges."""
    
    paper_id: str = Field(description="Paper ID to get edges for")
    direction: str = Field(default="both", pattern="^(cited_by|references|both)$", description="Edge direction")
    max_depth: int = Field(default=2, ge=1, le=5, description="Maximum traversal depth")
    include_metadata: bool = Field(default=True, description="Include full paper metadata")
    pagination: PaginationParams = Field(default_factory=PaginationParams)


class PaperEdge(BaseModel):
    """Citation edge between papers."""
    
    source_id: str = Field(description="Source paper ID")
    target_id: str = Field(description="Target paper ID")
    edge_type: str = Field(description="Edge type (cites/cited_by)")
    weight: float = Field(default=1.0, description="Edge weight")
    context: Optional[str] = Field(default=None, description="Citation context")
    created_at: Optional[datetime] = Field(default=None, description="Edge creation timestamp")


class PaperResponse(BaseModel):
    """Response model for single paper."""
    
    paper: Paper = Field(description="Paper data")
    edges: Optional[List[PaperEdge]] = Field(default=None, description="Citation edges")
    related_papers: Optional[List[Paper]] = Field(default=None, description="Related papers")


class PaperEdgesResponse(BaseModel):
    """Response model for paper edges."""
    
    center_paper: Paper = Field(description="Central paper")
    nodes: List[Paper] = Field(description="All papers in the network")
    edges: List[PaperEdge] = Field(description="Citation relationships")
    total_nodes: int = Field(description="Total number of nodes")
    total_edges: int = Field(description="Total number of edges")
    depth: int = Field(description="Maximum depth traversed")


class PaperBulkRequest(BaseModel):
    """Request for bulk paper operations."""
    
    paper_ids: List[str] = Field(description="List of paper IDs", min_items=1, max_items=100)
    operation: str = Field(description="Bulk operation type")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Operation parameters")
    
    @validator("paper_ids")
    def validate_paper_ids_unique(cls, v):
        """Ensure paper IDs are unique."""
        if len(v) != len(set(v)):
            raise ValueError("Paper IDs must be unique")
        return v