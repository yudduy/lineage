"""
Semantic Scholar API response models.

Comprehensive Pydantic models for Semantic Scholar data structures including
Papers, Authors, citation intent classification, influential citations, and 
SPECTER embeddings for advanced semantic analysis.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, date
from pydantic import Field, validator, HttpUrl, BaseModel as PydanticBaseModel
from enum import Enum
from .base import BaseModel


class CitationIntent(str, Enum):
    """Citation intent classification from SciCite model."""
    
    BACKGROUND = "background"
    METHOD = "method"  
    RESULT = "result"


class SemanticScholarExternalIds(BaseModel):
    """External identifiers for Semantic Scholar entities."""
    
    arxiv_id: Optional[str] = Field(default=None, description="arXiv ID")
    dblp_id: Optional[str] = Field(default=None, description="DBLP ID")
    doi: Optional[str] = Field(default=None, description="DOI")
    mag_id: Optional[str] = Field(default=None, description="Microsoft Academic Graph ID")
    pubmed_id: Optional[str] = Field(default=None, description="PubMed ID")
    acl_id: Optional[str] = Field(default=None, description="ACL Anthology ID")
    corpus_id: Optional[str] = Field(default=None, description="Semantic Scholar Corpus ID")
    
    @validator("doi")
    def validate_doi(cls, v):
        """Validate DOI format."""
        if v and not v.startswith("10."):
            raise ValueError("DOI must start with '10.'")
        return v


class SemanticScholarAuthor(BaseModel):
    """Semantic Scholar author information."""
    
    author_id: str = Field(description="Semantic Scholar author ID")
    name: str = Field(description="Author name")
    url: Optional[HttpUrl] = Field(default=None, description="Author profile URL")
    aliases: Optional[List[str]] = Field(default_factory=list, description="Author name aliases")
    affiliations: Optional[List[str]] = Field(default_factory=list, description="Author affiliations")
    homepage: Optional[HttpUrl] = Field(default=None, description="Author homepage URL")
    paper_count: Optional[int] = Field(default=None, description="Number of papers")
    citation_count: Optional[int] = Field(default=None, description="Total citations")
    h_index: Optional[int] = Field(default=None, description="H-index")


class SemanticScholarVenue(BaseModel):
    """Semantic Scholar venue (journal/conference) information."""
    
    name: Optional[str] = Field(default=None, description="Venue name")
    type: Optional[str] = Field(default=None, description="Venue type")
    alternate_names: Optional[List[str]] = Field(default_factory=list, description="Alternate venue names")
    issn: Optional[str] = Field(default=None, description="ISSN")
    url: Optional[HttpUrl] = Field(default=None, description="Venue URL")


class SemanticScholarTldr(BaseModel):
    """Auto-generated TL;DR summary from Semantic Scholar."""
    
    model: str = Field(description="Model used to generate summary")
    text: str = Field(description="TL;DR summary text")


class SemanticScholarCitationContext(BaseModel):
    """Citation context information."""
    
    context: str = Field(description="Text context around citation")
    intent: Optional[CitationIntent] = Field(default=None, description="Citation intent classification")
    is_influential: Optional[bool] = Field(default=False, description="Is this an influential citation")
    confidence: Optional[float] = Field(default=None, description="Classification confidence score")


class SemanticScholarEmbedding(BaseModel):
    """SPECTER paper embedding information."""
    
    model: str = Field(description="Embedding model name (e.g., 'specter')")
    vector: List[float] = Field(description="768-dimensional embedding vector")
    
    @validator("vector")
    def validate_vector_dimension(cls, v):
        """Validate embedding vector dimension."""
        if len(v) != 768:
            raise ValueError("SPECTER embedding must be 768-dimensional")
        return v


class SemanticScholarCitation(BaseModel):
    """Semantic Scholar citation with enriched information."""
    
    paper_id: str = Field(description="Cited paper Semantic Scholar ID")
    corpus_id: Optional[str] = Field(default=None, description="Corpus ID")
    url: Optional[HttpUrl] = Field(default=None, description="Paper URL")
    title: Optional[str] = Field(default=None, description="Paper title")
    abstract: Optional[str] = Field(default=None, description="Paper abstract")
    venue: Optional[SemanticScholarVenue] = Field(default=None, description="Publication venue")
    year: Optional[int] = Field(default=None, description="Publication year")
    reference_count: Optional[int] = Field(default=None, description="Number of references")
    citation_count: Optional[int] = Field(default=None, description="Number of citations")
    influential_citation_count: Optional[int] = Field(default=None, description="Influential citations count")
    is_open_access: Optional[bool] = Field(default=None, description="Open access status")
    fields_of_study: Optional[List[str]] = Field(default_factory=list, description="Academic fields")
    s2_fields_of_study: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="S2 field categories")
    authors: Optional[List[SemanticScholarAuthor]] = Field(default_factory=list, description="Paper authors")
    
    # Citation-specific enrichments
    contexts: Optional[List[str]] = Field(default_factory=list, description="Citation contexts")
    intents: Optional[List[CitationIntent]] = Field(default_factory=list, description="Citation intents")
    is_influential: Optional[bool] = Field(default=False, description="Is influential citation")
    
    @validator("year")
    def validate_year(cls, v):
        """Validate publication year."""
        if v and (v < 1000 or v > datetime.now().year + 2):
            raise ValueError("Invalid publication year")
        return v


class SemanticScholarPaper(BaseModel):
    """Complete Semantic Scholar paper model with semantic features."""
    
    # Core identifiers
    paper_id: str = Field(description="Semantic Scholar paper ID")
    corpus_id: Optional[str] = Field(default=None, description="Corpus ID")
    url: Optional[HttpUrl] = Field(default=None, description="Paper URL")
    external_ids: Optional[SemanticScholarExternalIds] = Field(default=None, description="External IDs")
    
    # Basic metadata
    title: Optional[str] = Field(default=None, description="Paper title")
    abstract: Optional[str] = Field(default=None, description="Paper abstract")
    tldr: Optional[SemanticScholarTldr] = Field(default=None, description="Auto-generated TL;DR")
    
    # Publication info
    year: Optional[int] = Field(default=None, description="Publication year")
    publication_date: Optional[date] = Field(default=None, description="Publication date")
    venue: Optional[SemanticScholarVenue] = Field(default=None, description="Publication venue")
    publication_types: Optional[List[str]] = Field(default_factory=list, description="Publication types")
    
    # Authors
    authors: List[SemanticScholarAuthor] = Field(default_factory=list, description="Paper authors")
    
    # Citation metrics
    citation_count: Optional[int] = Field(default=0, description="Total citation count")
    influential_citation_count: Optional[int] = Field(default=0, description="Influential citation count")
    reference_count: Optional[int] = Field(default=0, description="Reference count")
    
    # Citations and references with semantic features
    citations: Optional[List[SemanticScholarCitation]] = Field(
        default_factory=list, 
        description="Papers citing this work"
    )
    references: Optional[List[SemanticScholarCitation]] = Field(
        default_factory=list, 
        description="Papers referenced by this work"
    )
    
    # Academic classification
    fields_of_study: Optional[List[str]] = Field(default_factory=list, description="Academic fields")
    s2_fields_of_study: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list, 
        description="Semantic Scholar field categories"
    )
    
    # Open access information
    is_open_access: Optional[bool] = Field(default=None, description="Open access status")
    open_access_pdf: Optional[Dict[str, str]] = Field(default=None, description="Open access PDF info")
    
    # Semantic features
    embedding: Optional[SemanticScholarEmbedding] = Field(default=None, description="SPECTER embedding")
    
    # System metadata
    created_date: Optional[datetime] = Field(default=None, description="Record creation date")
    updated_date: Optional[datetime] = Field(default=None, description="Last update date")
    
    @validator("year")
    def validate_year(cls, v):
        """Validate publication year."""
        if v and (v < 1000 or v > datetime.now().year + 2):
            raise ValueError("Invalid publication year")
        return v
    
    def get_author_names(self) -> List[str]:
        """Get list of author names."""
        return [author.name for author in self.authors]
    
    def get_primary_field(self) -> Optional[str]:
        """Get primary field of study."""
        if self.s2_fields_of_study:
            # Return field with highest score
            fields_with_scores = [
                field for field in self.s2_fields_of_study 
                if isinstance(field, dict) and "category" in field
            ]
            if fields_with_scores:
                primary_field = max(fields_with_scores, key=lambda x: x.get("score", 0))
                return primary_field.get("category")
        
        if self.fields_of_study:
            return self.fields_of_study[0]
        
        return None
    
    def get_influential_citation_ratio(self) -> float:
        """Calculate ratio of influential to total citations."""
        if not self.citation_count or self.citation_count == 0:
            return 0.0
        return (self.influential_citation_count or 0) / self.citation_count
    
    def is_highly_influential(self) -> bool:
        """Check if paper has high influential citation ratio."""
        ratio = self.get_influential_citation_ratio()
        return ratio > 0.1  # More than 10% influential citations


class SemanticScholarPapersResponse(BaseModel):
    """Response from Semantic Scholar papers API."""
    
    total: Optional[int] = Field(default=None, description="Total number of results")
    offset: Optional[int] = Field(default=None, description="Result offset")
    next: Optional[int] = Field(default=None, description="Next page offset")
    data: List[SemanticScholarPaper] = Field(description="List of papers")


class SemanticScholarAuthorResponse(BaseModel):
    """Semantic Scholar author entity response."""
    
    author_id: str = Field(description="Author Semantic Scholar ID")
    external_ids: Optional[Dict[str, str]] = Field(default=None, description="External IDs")
    url: Optional[HttpUrl] = Field(default=None, description="Author profile URL")
    name: str = Field(description="Author name")
    aliases: Optional[List[str]] = Field(default_factory=list, description="Name aliases")
    affiliations: Optional[List[str]] = Field(default_factory=list, description="Affiliations")
    homepage: Optional[HttpUrl] = Field(default=None, description="Homepage URL")
    paper_count: Optional[int] = Field(default=None, description="Number of papers")
    citation_count: Optional[int] = Field(default=None, description="Total citations")
    h_index: Optional[int] = Field(default=None, description="H-index")
    papers: Optional[List[SemanticScholarPaper]] = Field(default_factory=list, description="Author papers")


class SemanticScholarSearchFilters(BaseModel):
    """Search filters for Semantic Scholar API queries."""
    
    # Date filters
    year: Optional[str] = Field(default=None, description="Publication year filter (e.g., '2020-2023')")
    publication_date_or_year: Optional[str] = Field(default=None, description="Date/year filter")
    
    # Content filters
    min_citation_count: Optional[int] = Field(default=None, description="Minimum citation count")
    
    # Field filters
    fields_of_study: Optional[List[str]] = Field(default=None, description="Academic fields filter")
    publication_types: Optional[List[str]] = Field(default=None, description="Publication types filter")
    
    # Venue filters
    venue: Optional[str] = Field(default=None, description="Venue name filter")
    
    # Author filters  
    author: Optional[str] = Field(default=None, description="Author name filter")
    
    # Access filters
    open_access_pdf: Optional[bool] = Field(default=None, description="Open access PDF filter")
    
    def to_query_params(self) -> Dict[str, Any]:
        """Convert filters to Semantic Scholar API query parameters."""
        params = {}
        
        if self.year:
            params["year"] = self.year
        if self.publication_date_or_year:
            params["publicationDateOrYear"] = self.publication_date_or_year
        if self.min_citation_count is not None:
            params["minCitationCount"] = self.min_citation_count
        if self.fields_of_study:
            params["fieldsOfStudy"] = ",".join(self.fields_of_study)
        if self.publication_types:
            params["publicationTypes"] = ",".join(self.publication_types)
        if self.venue:
            params["venue"] = self.venue
        if self.author:
            params["author"] = self.author
        if self.open_access_pdf is not None:
            params["openAccessPdf"] = str(self.open_access_pdf).lower()
        
        return params


class SemanticScholarBatchRequest(BaseModel):
    """Batch request for multiple Semantic Scholar papers."""
    
    paper_ids: List[str] = Field(description="List of paper IDs", min_items=1, max_items=500)
    fields: Optional[List[str]] = Field(default=None, description="Fields to retrieve")
    
    @validator("paper_ids")
    def validate_paper_ids_unique(cls, v):
        """Ensure paper IDs are unique."""
        if len(v) != len(set(v)):
            raise ValueError("Paper IDs must be unique")
        return v


class SemanticScholarSimilarityResult(BaseModel):
    """Result from semantic similarity computation."""
    
    paper_id_1: str = Field(description="First paper ID")
    paper_id_2: str = Field(description="Second paper ID")
    similarity_score: float = Field(description="Cosine similarity score")
    title_1: Optional[str] = Field(default=None, description="First paper title")
    title_2: Optional[str] = Field(default=None, description="Second paper title")
    
    @validator("similarity_score")
    def validate_similarity_score(cls, v):
        """Validate similarity score is between -1 and 1."""
        if not -1.0 <= v <= 1.0:
            raise ValueError("Similarity score must be between -1 and 1")
        return v


class SemanticScholarInfluentialCitation(BaseModel):
    """Influential citation with metadata."""
    
    citing_paper_id: str = Field(description="Citing paper ID")
    cited_paper_id: str = Field(description="Cited paper ID")
    is_influential: bool = Field(description="Is influential citation")
    contexts: List[str] = Field(default_factory=list, description="Citation contexts")
    intents: List[CitationIntent] = Field(default_factory=list, description="Citation intents")
    confidence_score: Optional[float] = Field(default=None, description="Influence confidence")
    
    # Metadata for enriched display
    citing_paper_title: Optional[str] = Field(default=None, description="Citing paper title")
    cited_paper_title: Optional[str] = Field(default=None, description="Cited paper title")
    citation_year: Optional[int] = Field(default=None, description="Citation year")


class SemanticScholarCitationNetwork(BaseModel):
    """Citation network with semantic enrichments."""
    
    center_paper_id: str = Field(description="Central paper ID")
    nodes: List[SemanticScholarPaper] = Field(description="Network nodes (papers)")
    edges: List[Dict[str, Any]] = Field(description="Network edges (citations)")
    
    # Semantic analysis results
    influential_citations: List[SemanticScholarInfluentialCitation] = Field(
        default_factory=list,
        description="Influential citations in network"
    )
    citation_intents: Dict[str, List[CitationIntent]] = Field(
        default_factory=dict,
        description="Citation intents by edge"
    )
    similarity_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Semantic similarity scores between papers"
    )
    
    # Network statistics
    total_nodes: int = Field(description="Total number of nodes")
    total_edges: int = Field(description="Total number of edges")
    influential_edges: int = Field(description="Number of influential citations")
    
    def get_most_similar_papers(self, limit: int = 10) -> List[Tuple[str, str, float]]:
        """Get most similar paper pairs."""
        similarity_pairs = [
            (pair.split("|")[0], pair.split("|")[1], score)
            for pair, score in self.similarity_scores.items()
        ]
        return sorted(similarity_pairs, key=lambda x: x[2], reverse=True)[:limit]
    
    def get_citation_intent_summary(self) -> Dict[CitationIntent, int]:
        """Get summary of citation intents."""
        intent_counts = {intent: 0 for intent in CitationIntent}
        
        for intents in self.citation_intents.values():
            for intent in intents:
                intent_counts[intent] += 1
        
        return intent_counts


class SemanticScholarError(BaseModel):
    """Semantic Scholar API error response."""
    
    error: str = Field(description="Error message")
    message: Optional[str] = Field(default=None, description="Detailed error message")


class SemanticScholarRateLimit(BaseModel):
    """Semantic Scholar API rate limit information."""
    
    requests_remaining: Optional[int] = Field(default=None, description="Requests remaining")
    requests_per_second: int = Field(description="Requests per second limit")
    requests_per_minute: Optional[int] = Field(default=None, description="Requests per minute limit")
    reset_time: Optional[datetime] = Field(default=None, description="Rate limit reset time")
    is_authenticated: bool = Field(default=False, description="Using API key authentication")


# Integration models for combining OpenAlex and Semantic Scholar data

class EnrichedPaper(BaseModel):
    """Paper enriched with both OpenAlex and Semantic Scholar data."""
    
    # Core identifiers (common to both APIs)
    openalex_id: Optional[str] = Field(default=None, description="OpenAlex ID")
    semantic_scholar_id: Optional[str] = Field(default=None, description="Semantic Scholar ID")
    doi: Optional[str] = Field(default=None, description="DOI")
    title: Optional[str] = Field(default=None, description="Paper title")
    
    # OpenAlex data
    openalex_data: Optional[Dict[str, Any]] = Field(default=None, description="OpenAlex paper data")
    
    # Semantic Scholar data with enrichments
    semantic_scholar_data: Optional[SemanticScholarPaper] = Field(
        default=None, 
        description="Semantic Scholar paper data"
    )
    
    # Combined analysis
    citation_intent_analysis: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Citation intent analysis results"
    )
    influential_citation_analysis: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Influential citation analysis results"
    )
    semantic_similarity_scores: Optional[Dict[str, float]] = Field(
        default=None,
        description="Semantic similarity to other papers"
    )
    
    # Enrichment metadata
    enrichment_timestamp: Optional[datetime] = Field(default=None, description="Last enrichment time")
    enrichment_sources: List[str] = Field(default_factory=list, description="Data sources used")
    
    def has_semantic_features(self) -> bool:
        """Check if paper has semantic enrichment features."""
        return bool(
            self.semantic_scholar_data and (
                self.semantic_scholar_data.embedding or
                self.citation_intent_analysis or
                self.influential_citation_analysis
            )
        )
    
    def get_unified_citation_count(self) -> int:
        """Get citation count from best available source."""
        if self.semantic_scholar_data and self.semantic_scholar_data.citation_count:
            return self.semantic_scholar_data.citation_count
        if self.openalex_data and self.openalex_data.get("cited_by_count"):
            return self.openalex_data["cited_by_count"]
        return 0
    
    def get_unified_author_names(self) -> List[str]:
        """Get author names from best available source."""
        if self.semantic_scholar_data and self.semantic_scholar_data.authors:
            return self.semantic_scholar_data.get_author_names()
        if self.openalex_data and self.openalex_data.get("authorships"):
            return [auth.get("author", {}).get("display_name", "") 
                   for auth in self.openalex_data["authorships"]]
        return []