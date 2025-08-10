"""
OpenAlex API response models.

Comprehensive Pydantic models for all OpenAlex data structures including
Works, Authors, Institutions, Venues, Concepts, and their relationships.
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime, date
from pydantic import Field, validator, HttpUrl, BaseModel as PydanticBaseModel
from .base import BaseModel


class OpenAlexLocation(BaseModel):
    """Location information for OpenAlex entities."""
    
    display_name: Optional[str] = Field(default=None, description="Location display name")
    relevance_score: Optional[float] = Field(default=None, description="Relevance score")


class OpenAlexExternalIds(BaseModel):
    """External identifiers for OpenAlex entities."""
    
    openalex: Optional[str] = Field(default=None, description="OpenAlex ID")
    doi: Optional[str] = Field(default=None, description="DOI")
    pmid: Optional[str] = Field(default=None, description="PubMed ID")
    pmcid: Optional[str] = Field(default=None, description="PMC ID")
    arxiv: Optional[str] = Field(default=None, description="arXiv ID")
    mag: Optional[str] = Field(default=None, description="Microsoft Academic Graph ID")
    
    @validator("doi")
    def validate_doi(cls, v):
        """Validate DOI format."""
        if v and not v.startswith("10."):
            raise ValueError("DOI must start with '10.'")
        return v


class OpenAlexConcept(BaseModel):
    """OpenAlex concept/topic information."""
    
    id: str = Field(description="Concept OpenAlex ID")
    wikidata: Optional[str] = Field(default=None, description="Wikidata ID")
    display_name: str = Field(description="Concept display name")
    level: int = Field(description="Concept level in hierarchy")
    score: float = Field(description="Relevance score")


class OpenAlexInstitution(BaseModel):
    """OpenAlex institution information."""
    
    id: str = Field(description="Institution OpenAlex ID")
    display_name: str = Field(description="Institution name")
    ror: Optional[str] = Field(default=None, description="ROR ID")
    country_code: Optional[str] = Field(default=None, description="Country code")
    type: Optional[str] = Field(default=None, description="Institution type")
    lineage: Optional[List[str]] = Field(default_factory=list, description="Institution hierarchy")


class OpenAlexAuthor(BaseModel):
    """OpenAlex author information."""
    
    id: str = Field(description="Author OpenAlex ID")
    display_name: str = Field(description="Author display name")
    orcid: Optional[str] = Field(default=None, description="ORCID ID")
    
    @validator("orcid")
    def validate_orcid(cls, v):
        """Validate ORCID format."""
        if v and not v.startswith("https://orcid.org/"):
            if len(v) == 19 and v[4] == '-' and v[9] == '-' and v[14] == '-':
                return f"https://orcid.org/{v}"
        return v


class OpenAlexAuthorship(BaseModel):
    """OpenAlex authorship information with affiliations."""
    
    author_position: str = Field(description="Position (first, middle, last)")
    author: OpenAlexAuthor = Field(description="Author information")
    institutions: List[OpenAlexInstitution] = Field(default_factory=list, description="Author institutions")
    raw_author_name: Optional[str] = Field(default=None, description="Raw author name from source")
    raw_affiliation_strings: Optional[List[str]] = Field(default_factory=list, description="Raw affiliation strings")


class OpenAlexVenue(BaseModel):
    """OpenAlex venue (journal/conference) information."""
    
    id: Optional[str] = Field(default=None, description="Venue OpenAlex ID")
    issn_l: Optional[str] = Field(default=None, description="ISSN-L")
    issn: Optional[List[str]] = Field(default_factory=list, description="ISSN list")
    display_name: Optional[str] = Field(default=None, description="Venue name")
    publisher: Optional[str] = Field(default=None, description="Publisher")
    is_oa: Optional[bool] = Field(default=None, description="Is open access venue")
    is_in_doaj: Optional[bool] = Field(default=None, description="Is in DOAJ")


class OpenAlexGrant(BaseModel):
    """OpenAlex funding grant information."""
    
    funder: Optional[str] = Field(default=None, description="Funder OpenAlex ID")
    funder_display_name: Optional[str] = Field(default=None, description="Funder name")
    award_id: Optional[str] = Field(default=None, description="Grant award ID")


class OpenAlexOpenAccess(BaseModel):
    """OpenAlex open access information."""
    
    is_oa: bool = Field(description="Is open access")
    oa_date: Optional[date] = Field(default=None, description="Open access date")
    oa_url: Optional[HttpUrl] = Field(default=None, description="Open access URL")
    any_repository_has_fulltext: Optional[bool] = Field(default=None, description="Repository has fulltext")


class OpenAlexAPC(BaseModel):
    """OpenAlex article processing charge information."""
    
    value: Optional[int] = Field(default=None, description="APC value")
    currency: Optional[str] = Field(default=None, description="Currency code")
    value_usd: Optional[int] = Field(default=None, description="APC value in USD")
    provenance: Optional[str] = Field(default=None, description="Data provenance")


class OpenAlexBiblio(BaseModel):
    """OpenAlex bibliographic information."""
    
    volume: Optional[str] = Field(default=None, description="Volume")
    issue: Optional[str] = Field(default=None, description="Issue")
    first_page: Optional[str] = Field(default=None, description="First page")
    last_page: Optional[str] = Field(default=None, description="Last page")


class OpenAlexMesh(BaseModel):
    """OpenAlex MeSH term information."""
    
    descriptor_ui: str = Field(description="MeSH descriptor ID")
    descriptor_name: str = Field(description="MeSH descriptor name")
    qualifier_ui: Optional[str] = Field(default=None, description="MeSH qualifier ID")
    qualifier_name: Optional[str] = Field(default=None, description="MeSH qualifier name")
    is_major_topic: bool = Field(description="Is major topic")


class OpenAlexKeyword(BaseModel):
    """OpenAlex keyword information."""
    
    id: Optional[str] = Field(default=None, description="Keyword ID")
    display_name: str = Field(description="Keyword text")
    score: Optional[float] = Field(default=None, description="Relevance score")


class OpenAlexAlternateTitles(BaseModel):
    """OpenAlex alternate titles."""
    
    original: Optional[str] = Field(default=None, description="Original title")
    translated: Optional[List[str]] = Field(default_factory=list, description="Translated titles")


class OpenAlexAbstractInvertedIndex(BaseModel):
    """OpenAlex abstract as inverted index."""
    
    # This represents the inverted index structure where keys are words
    # and values are lists of positions where the word appears
    inverted_index: Dict[str, List[int]] = Field(default_factory=dict, description="Inverted index mapping")
    
    def to_text(self) -> str:
        """Convert inverted index to readable text."""
        if not self.inverted_index:
            return ""
        
        # Create a list to hold words in order
        max_position = max([max(positions) for positions in self.inverted_index.values() if positions])
        text_array = [""] * (max_position + 1)
        
        # Place each word at its correct positions
        for word, positions in self.inverted_index.items():
            for position in positions:
                if position < len(text_array):
                    text_array[position] = word
        
        # Join words and clean up
        text = " ".join(filter(None, text_array))
        return text.strip()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create from raw OpenAlex abstract data."""
        if isinstance(data, dict):
            return cls(inverted_index=data)
        return cls(inverted_index={})


class OpenAlexCitation(BaseModel):
    """OpenAlex citation reference."""
    
    openalex_id: Optional[str] = Field(default=None, description="Cited work OpenAlex ID")
    raw_citation: Optional[str] = Field(default=None, description="Raw citation text")


class OpenAlexWork(BaseModel):
    """Complete OpenAlex work (paper) model."""
    
    # Core identifiers
    id: str = Field(description="OpenAlex work ID")
    ids: OpenAlexExternalIds = Field(description="External identifiers")
    
    # Basic metadata
    title: Optional[str] = Field(default=None, description="Work title")
    display_name: Optional[str] = Field(default=None, description="Display name (usually same as title)")
    alternate_titles: Optional[OpenAlexAlternateTitles] = Field(default=None, description="Alternate titles")
    
    # Abstract
    abstract_inverted_index: Optional[OpenAlexAbstractInvertedIndex] = Field(
        default=None, 
        description="Abstract as inverted index"
    )
    
    # Publication info
    publication_year: Optional[int] = Field(default=None, description="Publication year")
    publication_date: Optional[date] = Field(default=None, description="Publication date")
    
    # Authorship and affiliations
    authorships: List[OpenAlexAuthorship] = Field(default_factory=list, description="Author information")
    
    # Venue information
    primary_location: Optional[Dict[str, Any]] = Field(default=None, description="Primary publication location")
    best_oa_location: Optional[Dict[str, Any]] = Field(default=None, description="Best open access location")
    locations: List[Dict[str, Any]] = Field(default_factory=list, description="All publication locations")
    
    # Classification
    concepts: List[OpenAlexConcept] = Field(default_factory=list, description="Associated concepts")
    mesh: List[OpenAlexMesh] = Field(default_factory=list, description="MeSH terms")
    keywords: List[OpenAlexKeyword] = Field(default_factory=list, description="Keywords")
    
    # Citation information
    cited_by_count: int = Field(default=0, description="Number of citations")
    cited_by_api_url: Optional[str] = Field(default=None, description="API URL for citing works")
    counts_by_year: List[Dict[str, int]] = Field(default_factory=list, description="Citation counts by year")
    
    # References
    referenced_works: List[str] = Field(default_factory=list, description="Referenced work OpenAlex IDs")
    referenced_works_count: int = Field(default=0, description="Number of references")
    
    # Open access information
    open_access: Optional[OpenAlexOpenAccess] = Field(default=None, description="Open access info")
    apc_list: Optional[OpenAlexAPC] = Field(default=None, description="Article processing charges")
    apc_paid: Optional[OpenAlexAPC] = Field(default=None, description="Paid APC info")
    
    # Bibliographic details
    biblio: Optional[OpenAlexBiblio] = Field(default=None, description="Bibliographic info")
    
    # Funding
    grants: List[OpenAlexGrant] = Field(default_factory=list, description="Funding grants")
    
    # Work type and attributes
    type: Optional[str] = Field(default=None, description="Work type")
    type_crossref: Optional[str] = Field(default=None, description="CrossRef work type")
    is_paratext: bool = Field(default=False, description="Is paratext")
    is_retracted: bool = Field(default=False, description="Is retracted")
    
    # Language
    language: Optional[str] = Field(default=None, description="Language code")
    
    # System metadata
    created_date: Optional[date] = Field(default=None, description="Record creation date")
    updated_date: Optional[date] = Field(default=None, description="Last update date")
    
    # API URLs
    api_url: Optional[str] = Field(default=None, description="OpenAlex API URL")
    
    @validator("publication_year")
    def validate_publication_year(cls, v):
        """Validate publication year is reasonable."""
        if v and (v < 1000 or v > datetime.now().year + 2):
            raise ValueError("Invalid publication year")
        return v
    
    @validator("abstract_inverted_index", pre=True)
    def validate_abstract_inverted_index(cls, v):
        """Convert raw abstract data to model."""
        if v is None:
            return None
        if isinstance(v, dict):
            return OpenAlexAbstractInvertedIndex.from_dict(v)
        return v
    
    def get_abstract_text(self) -> Optional[str]:
        """Get abstract as readable text."""
        if self.abstract_inverted_index:
            return self.abstract_inverted_index.to_text()
        return None
    
    def get_author_names(self) -> List[str]:
        """Get list of author names."""
        return [auth.author.display_name for auth in self.authorships]
    
    def get_institution_names(self) -> List[str]:
        """Get unique institution names."""
        institutions = set()
        for auth in self.authorships:
            for inst in auth.institutions:
                institutions.add(inst.display_name)
        return list(institutions)
    
    def get_primary_venue_name(self) -> Optional[str]:
        """Get primary venue name."""
        if self.primary_location and isinstance(self.primary_location, dict):
            source = self.primary_location.get("source", {})
            if isinstance(source, dict):
                return source.get("display_name")
        return None
    
    def is_open_access(self) -> bool:
        """Check if work is open access."""
        return self.open_access.is_oa if self.open_access else False
    
    def get_concept_names(self) -> List[str]:
        """Get concept names sorted by relevance."""
        return [concept.display_name for concept in sorted(self.concepts, key=lambda x: x.score, reverse=True)]


class OpenAlexWorksResponse(BaseModel):
    """Response from OpenAlex works API."""
    
    results: List[OpenAlexWork] = Field(description="List of works")
    meta: Dict[str, Any] = Field(description="Response metadata")
    group_by: Optional[List[Dict[str, Any]]] = Field(default=None, description="Grouping information")


class OpenAlexAuthorResponse(BaseModel):
    """OpenAlex author entity response."""
    
    id: str = Field(description="Author OpenAlex ID")
    orcid: Optional[str] = Field(default=None, description="ORCID ID")
    display_name: str = Field(description="Author display name")
    display_name_alternatives: List[str] = Field(default_factory=list, description="Alternative names")
    works_count: int = Field(default=0, description="Number of works")
    cited_by_count: int = Field(default=0, description="Total citations")
    last_known_institutions: List[OpenAlexInstitution] = Field(
        default_factory=list, 
        description="Recent affiliations"
    )
    concepts: List[OpenAlexConcept] = Field(default_factory=list, description="Research concepts")
    counts_by_year: List[Dict[str, int]] = Field(default_factory=list, description="Works/citations by year")
    works_api_url: Optional[str] = Field(default=None, description="API URL for author's works")
    updated_date: Optional[date] = Field(default=None, description="Last update date")


class OpenAlexInstitutionResponse(BaseModel):
    """OpenAlex institution entity response."""
    
    id: str = Field(description="Institution OpenAlex ID")
    ror: Optional[str] = Field(default=None, description="ROR ID")
    display_name: str = Field(description="Institution name")
    display_name_alternatives: List[str] = Field(default_factory=list, description="Alternative names")
    display_name_acronyms: List[str] = Field(default_factory=list, description="Acronyms")
    country_code: Optional[str] = Field(default=None, description="Country code")
    type: Optional[str] = Field(default=None, description="Institution type")
    homepage_url: Optional[HttpUrl] = Field(default=None, description="Homepage URL")
    image_url: Optional[HttpUrl] = Field(default=None, description="Logo image URL")
    works_count: int = Field(default=0, description="Number of works")
    cited_by_count: int = Field(default=0, description="Total citations")
    concepts: List[OpenAlexConcept] = Field(default_factory=list, description="Research concepts")
    geo: Optional[Dict[str, Any]] = Field(default=None, description="Geographic information")
    international: Optional[Dict[str, Any]] = Field(default=None, description="International info")
    associated_institutions: List[str] = Field(default_factory=list, description="Associated institution IDs")
    counts_by_year: List[Dict[str, int]] = Field(default_factory=list, description="Works/citations by year")
    works_api_url: Optional[str] = Field(default=None, description="API URL for institution's works")
    updated_date: Optional[date] = Field(default=None, description="Last update date")


class OpenAlexVenueResponse(BaseModel):
    """OpenAlex venue (journal/source) entity response."""
    
    id: str = Field(description="Venue OpenAlex ID")
    issn_l: Optional[str] = Field(default=None, description="ISSN-L")
    issn: Optional[List[str]] = Field(default_factory=list, description="ISSN list")
    display_name: str = Field(description="Venue name")
    publisher: Optional[str] = Field(default=None, description="Publisher")
    works_count: int = Field(default=0, description="Number of works")
    cited_by_count: int = Field(default=0, description="Total citations")
    is_oa: Optional[bool] = Field(default=None, description="Is open access venue")
    is_in_doaj: Optional[bool] = Field(default=None, description="Is in DOAJ")
    homepage_url: Optional[HttpUrl] = Field(default=None, description="Homepage URL")
    apc_prices: List[OpenAlexAPC] = Field(default_factory=list, description="APC pricing")
    apc_usd: Optional[int] = Field(default=None, description="APC in USD")
    country_code: Optional[str] = Field(default=None, description="Country code")
    societies: List[str] = Field(default_factory=list, description="Associated societies")
    alternate_titles: List[str] = Field(default_factory=list, description="Alternate titles")
    abbreviated_title: Optional[str] = Field(default=None, description="Abbreviated title")
    type: Optional[str] = Field(default=None, description="Venue type")
    x_concepts: List[OpenAlexConcept] = Field(default_factory=list, description="Associated concepts")
    counts_by_year: List[Dict[str, int]] = Field(default_factory=list, description="Works/citations by year")
    works_api_url: Optional[str] = Field(default=None, description="API URL for venue's works")
    updated_date: Optional[date] = Field(default=None, description="Last update date")


class OpenAlexConceptResponse(BaseModel):
    """OpenAlex concept entity response."""
    
    id: str = Field(description="Concept OpenAlex ID")
    wikidata: Optional[str] = Field(default=None, description="Wikidata ID")
    display_name: str = Field(description="Concept name")
    level: int = Field(description="Level in concept hierarchy")
    description: Optional[str] = Field(default=None, description="Concept description")
    works_count: int = Field(default=0, description="Number of associated works")
    cited_by_count: int = Field(default=0, description="Total citations")
    image_url: Optional[HttpUrl] = Field(default=None, description="Concept image URL")
    image_thumbnail_url: Optional[HttpUrl] = Field(default=None, description="Thumbnail image URL")
    international: Optional[Dict[str, Any]] = Field(default=None, description="International names")
    ancestors: List[Dict[str, Any]] = Field(default_factory=list, description="Ancestor concepts")
    related_concepts: List[Dict[str, Any]] = Field(default_factory=list, description="Related concepts")
    counts_by_year: List[Dict[str, int]] = Field(default_factory=list, description="Works/citations by year")
    works_api_url: Optional[str] = Field(default=None, description="API URL for concept's works")
    updated_date: Optional[date] = Field(default=None, description="Last update date")


class OpenAlexSearchFilters(BaseModel):
    """Search filters for OpenAlex API queries."""
    
    # Date filters
    from_publication_date: Optional[date] = Field(default=None, description="Minimum publication date")
    to_publication_date: Optional[date] = Field(default=None, description="Maximum publication date")
    from_created_date: Optional[date] = Field(default=None, description="Minimum record creation date")
    to_created_date: Optional[date] = Field(default=None, description="Maximum record creation date")
    
    # Citation filters
    cited_by_count: Optional[str] = Field(default=None, description="Citation count filter (e.g., '>10', '5-20')")
    
    # Work type filters
    type: Optional[str] = Field(default=None, description="Work type filter")
    is_oa: Optional[bool] = Field(default=None, description="Open access filter")
    is_retracted: Optional[bool] = Field(default=None, description="Retraction filter")
    is_paratext: Optional[bool] = Field(default=None, description="Paratext filter")
    
    # Entity filters
    authorships_institutions_id: Optional[str] = Field(default=None, description="Institution filter")
    authorships_author_id: Optional[str] = Field(default=None, description="Author filter")
    locations_source_id: Optional[str] = Field(default=None, description="Venue filter")
    concepts_id: Optional[str] = Field(default=None, description="Concept filter")
    
    # Language filter
    language: Optional[str] = Field(default=None, description="Language code filter")
    
    def to_query_params(self) -> Dict[str, Any]:
        """Convert filters to OpenAlex API query parameters."""
        params = {}
        
        # Build filter string
        filters = []
        
        if self.from_publication_date:
            filters.append(f"from_publication_date:{self.from_publication_date.isoformat()}")
        if self.to_publication_date:
            filters.append(f"to_publication_date:{self.to_publication_date.isoformat()}")
        if self.from_created_date:
            filters.append(f"from_created_date:{self.from_created_date.isoformat()}")
        if self.to_created_date:
            filters.append(f"to_created_date:{self.to_created_date.isoformat()}")
        
        if self.cited_by_count:
            filters.append(f"cited_by_count:{self.cited_by_count}")
        
        if self.type:
            filters.append(f"type:{self.type}")
        if self.is_oa is not None:
            filters.append(f"is_oa:{str(self.is_oa).lower()}")
        if self.is_retracted is not None:
            filters.append(f"is_retracted:{str(self.is_retracted).lower()}")
        if self.is_paratext is not None:
            filters.append(f"is_paratext:{str(self.is_paratext).lower()}")
        
        if self.authorships_institutions_id:
            filters.append(f"authorships.institutions.id:{self.authorships_institutions_id}")
        if self.authorships_author_id:
            filters.append(f"authorships.author.id:{self.authorships_author_id}")
        if self.locations_source_id:
            filters.append(f"locations.source.id:{self.locations_source_id}")
        if self.concepts_id:
            filters.append(f"concepts.id:{self.concepts_id}")
        
        if self.language:
            filters.append(f"language:{self.language}")
        
        if filters:
            params["filter"] = ",".join(filters)
        
        return params


class OpenAlexBatchRequest(BaseModel):
    """Batch request for multiple OpenAlex operations."""
    
    ids: List[str] = Field(description="List of OpenAlex IDs", min_items=1, max_items=50)
    entity_type: str = Field(description="Entity type (works, authors, institutions, venues, concepts)")
    include_relations: bool = Field(default=False, description="Include related entities")
    
    @validator("entity_type")
    def validate_entity_type(cls, v):
        """Validate entity type."""
        valid_types = ["works", "authors", "institutions", "venues", "concepts"]
        if v not in valid_types:
            raise ValueError(f"Entity type must be one of: {valid_types}")
        return v
    
    @validator("ids")
    def validate_ids_unique(cls, v):
        """Ensure IDs are unique."""
        if len(v) != len(set(v)):
            raise ValueError("IDs must be unique")
        return v


class OpenAlexError(BaseModel):
    """OpenAlex API error response."""
    
    error: str = Field(description="Error message")
    message: Optional[str] = Field(default=None, description="Detailed error message")
    documentation_url: Optional[str] = Field(default=None, description="Documentation URL")


class OpenAlexRateLimit(BaseModel):
    """OpenAlex API rate limit information."""
    
    requests_remaining: int = Field(description="Requests remaining")
    requests_per_day: int = Field(description="Daily request limit")
    reset_time: datetime = Field(description="Rate limit reset time")
    is_polite_pool: bool = Field(default=False, description="Using polite pool")