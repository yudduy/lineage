"""
Semantic Scholar API client with advanced semantic features.

Production-ready client for the Semantic Scholar API with rate limiting, caching,
error handling, and comprehensive semantic analysis capabilities including
citation intent classification, influential citations, and SPECTER embeddings.
"""

import asyncio
import json
import time
import numpy as np
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from datetime import datetime, timedelta
from urllib.parse import urlencode, quote

import httpx
from httpx import AsyncClient, Response, HTTPStatusError, RequestError

from ..core.config import get_settings
from ..db.redis import RedisManager, get_redis_manager
from ..models.semantic_scholar import (
    SemanticScholarPaper,
    SemanticScholarPapersResponse,
    SemanticScholarAuthorResponse,
    SemanticScholarSearchFilters,
    SemanticScholarBatchRequest,
    SemanticScholarCitationNetwork,
    SemanticScholarInfluentialCitation,
    SemanticScholarSimilarityResult,
    SemanticScholarError,
    SemanticScholarRateLimit,
    SemanticScholarEmbedding,
    CitationIntent,
    EnrichedPaper
)
from ..utils.logger import get_logger
from ..utils.exceptions import APIError, RateLimitError, ValidationError

logger = get_logger(__name__)


class SemanticScholarCircuitBreakerError(Exception):
    """Semantic Scholar circuit breaker is open."""
    pass


class SemanticScholarCircuitBreaker:
    """Circuit breaker pattern for Semantic Scholar API fault tolerance."""
    
    def __init__(self, failure_threshold: int = 3, timeout: int = 30):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
    
    def can_execute(self) -> bool:
        """Check if request can be executed."""
        if self.state == "closed":
            return True
        
        if self.state == "open":
            if time.time() - self.last_failure_time >= self.timeout:
                self.state = "half_open"
                return True
            return False
        
        if self.state == "half_open":
            return True
        
        return False
    
    def record_success(self):
        """Record successful request."""
        self.failure_count = 0
        self.state = "closed"
        self.last_failure_time = None
    
    def record_failure(self):
        """Record failed request."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
    
    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        return self.state == "open"


class SemanticScholarRateLimiter:
    """Rate limiter for Semantic Scholar API with different limits based on authentication."""
    
    def __init__(self, requests_per_second: float = 1.0, requests_per_minute: int = 100):
        self.requests_per_second = requests_per_second
        self.requests_per_minute = requests_per_minute
        self.request_times = []
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """Acquire permission to make a request."""
        async with self._lock:
            now = time.time()
            
            # Remove old requests outside the minute window
            self.request_times = [req_time for req_time in self.request_times if now - req_time < 60]
            
            # Check per-minute limit
            if len(self.request_times) >= self.requests_per_minute:
                return False
            
            # Check per-second limit
            recent_requests = [req_time for req_time in self.request_times if now - req_time < 1]
            if len(recent_requests) >= self.requests_per_second:
                return False
            
            # Record this request
            self.request_times.append(now)
            return True
    
    async def wait_if_needed(self):
        """Wait if rate limit would be exceeded."""
        while not await self.acquire():
            # Wait for next available slot
            await asyncio.sleep(1.0 / self.requests_per_second)


class SemanticScholarClient:
    """
    Comprehensive Semantic Scholar API client with advanced semantic features.
    
    Features:
    - Rate limiting (1000 RPS shared, 1 RPS dedicated)
    - Redis caching for responses and expensive computations
    - Circuit breaker for fault tolerance
    - Citation intent classification using SciCite model
    - Influential citation detection
    - SPECTER embedding support
    - Semantic similarity computation
    - Batch processing up to 500 papers
    - Integration with OpenAlex data
    """
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    # Standard fields for different request types
    PAPER_FIELDS = [
        "paperId", "corpusId", "url", "title", "abstract", "venue", "year",
        "referenceCount", "citationCount", "influentialCitationCount",
        "isOpenAccess", "openAccessPdf", "fieldsOfStudy", "s2FieldsOfStudy",
        "publicationTypes", "publicationDate", "authors", "externalIds",
        "embedding", "tldr"
    ]
    
    CITATION_FIELDS = [
        "paperId", "corpusId", "url", "title", "abstract", "venue", "year",
        "referenceCount", "citationCount", "influentialCitationCount", 
        "isOpenAccess", "fieldsOfStudy", "s2FieldsOfStudy", "authors",
        "contexts", "intents", "isInfluential"
    ]
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        redis_manager: Optional[RedisManager] = None,
        user_agent: Optional[str] = None,
        cache_ttl: int = 3600,  # 1 hour default cache
        max_retries: int = 3,
        circuit_breaker_threshold: int = 3
    ):
        self.settings = get_settings()
        self.api_key = api_key or getattr(self.settings.external_apis, 'semantic_scholar_api_key', None)
        self.redis_manager = redis_manager
        self.cache_ttl = cache_ttl
        self.max_retries = max_retries
        
        # Set up rate limiting based on authentication
        if self.api_key:
            # Authenticated users get higher rate limits
            self.rate_limiter = SemanticScholarRateLimiter(
                requests_per_second=100.0,  # Higher limit for authenticated
                requests_per_minute=10000
            )
            self.is_authenticated = True
        else:
            # Shared rate limit for unauthenticated users
            self.rate_limiter = SemanticScholarRateLimiter(
                requests_per_second=1.0,  # 1 RPS for shared pool
                requests_per_minute=100
            )
            self.is_authenticated = False
        
        # Circuit breaker for fault tolerance
        self.circuit_breaker = SemanticScholarCircuitBreaker(
            failure_threshold=circuit_breaker_threshold
        )
        
        # HTTP client setup
        headers = {
            "User-Agent": user_agent or "Citation Network Explorer/2.0 (https://github.com/example/citation-explorer)",
            "Accept": "application/json"
        }
        
        if self.api_key:
            headers["x-api-key"] = self.api_key
        
        self.client = AsyncClient(
            base_url=self.BASE_URL,
            headers=headers,
            timeout=30.0,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
        )
        
        # Metrics
        self.request_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.rate_limit_waits = 0
        self.embedding_cache_hits = 0
        
        logger.info(
            f"Semantic Scholar client initialized with {'authenticated' if self.is_authenticated else 'shared'} rate limits"
        )
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    def _get_cache_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        """Generate cache key for request."""
        param_str = urlencode(sorted(params.items()))
        return f"semantic_scholar:{endpoint}:{hash(param_str)}"
    
    async def _get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached response if available."""
        if not self.redis_manager:
            return None
        
        try:
            cached = await self.redis_manager.cache_get(cache_key)
            if cached:
                self.cache_hits += 1
                return cached
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
        
        self.cache_misses += 1
        return None
    
    async def _cache_response(self, cache_key: str, data: Dict[str, Any], ttl: Optional[int] = None):
        """Cache response data."""
        if not self.redis_manager:
            return
        
        try:
            await self.redis_manager.cache_set(cache_key, data, ttl or self.cache_ttl)
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
    
    async def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        cache_ttl: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request with rate limiting, caching, and error handling.
        """
        if not self.circuit_breaker.can_execute():
            raise SemanticScholarCircuitBreakerError("Circuit breaker is open")
        
        params = params or {}
        cache_key = self._get_cache_key(endpoint, params) if use_cache else None
        
        # Check cache first
        if cache_key:
            cached_response = await self._get_cached_response(cache_key)
            if cached_response:
                return cached_response
        
        # Apply rate limiting
        await self.rate_limiter.wait_if_needed()
        
        # Make request with retries
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                self.request_count += 1
                
                logger.debug(f"Semantic Scholar API request: {endpoint} (attempt {attempt + 1})")
                
                response = await self.client.get(endpoint, params=params)
                
                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 60))
                    self.rate_limit_waits += 1
                    
                    if attempt < self.max_retries:
                        logger.warning(f"Rate limited, waiting {retry_after} seconds")
                        await asyncio.sleep(retry_after)
                        continue
                    else:
                        self.circuit_breaker.record_failure()
                        raise RateLimitError(f"Rate limit exceeded after {self.max_retries} retries")
                
                # Handle other HTTP errors
                response.raise_for_status()
                
                # Parse response
                data = response.json()
                
                # Record success and cache response
                self.circuit_breaker.record_success()
                
                if cache_key:
                    await self._cache_response(cache_key, data, cache_ttl)
                
                return data
                
            except HTTPStatusError as e:
                last_exception = e
                
                if e.response.status_code == 404:
                    # Don't retry 404s
                    self.circuit_breaker.record_failure()
                    raise APIError(f"Resource not found: {endpoint}", status_code=404)
                
                elif e.response.status_code >= 500:
                    # Retry server errors
                    if attempt < self.max_retries:
                        wait_time = (2 ** attempt) + (0.1 * attempt)
                        logger.warning(f"Server error {e.response.status_code}, retrying in {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue
                
                self.circuit_breaker.record_failure()
                
            except RequestError as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    wait_time = (2 ** attempt) + (0.1 * attempt)
                    logger.warning(f"Request error, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                    continue
                
                self.circuit_breaker.record_failure()
        
        # All retries exhausted
        if last_exception:
            raise APIError(f"Semantic Scholar API request failed after {self.max_retries} retries: {last_exception}")
        
        raise APIError("Semantic Scholar API request failed for unknown reason")
    
    async def get_paper_by_id(
        self, 
        paper_id: str, 
        fields: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> Optional[SemanticScholarPaper]:
        """
        Get a paper by its Semantic Scholar ID, DOI, or other identifier.
        
        Args:
            paper_id: Semantic Scholar paper ID, DOI, ArXiv ID, etc.
            fields: Fields to retrieve (defaults to comprehensive set)
            use_cache: Whether to use caching
            
        Returns:
            SemanticScholarPaper object or None if not found
        """
        fields = fields or self.PAPER_FIELDS
        
        params = {
            "fields": ",".join(fields)
        }
        
        # Encode paper ID for URL
        encoded_paper_id = quote(paper_id, safe="")
        
        try:
            response_data = await self._make_request(
                f"/paper/{encoded_paper_id}", 
                params=params, 
                use_cache=use_cache
            )
            return SemanticScholarPaper(**response_data)
        
        except APIError as e:
            if "not found" in str(e).lower():
                return None
            raise
    
    async def get_papers_batch(
        self,
        paper_ids: List[str],
        fields: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> List[SemanticScholarPaper]:
        """
        Get multiple papers in a single batch request.
        
        Args:
            paper_ids: List of paper identifiers (max 500)
            fields: Fields to retrieve
            use_cache: Whether to use caching
            
        Returns:
            List of SemanticScholarPaper objects
        """
        if len(paper_ids) > 500:
            raise ValidationError("Maximum 500 paper IDs allowed per batch request")
        
        fields = fields or self.PAPER_FIELDS
        
        # Prepare batch request payload
        payload = {
            "ids": paper_ids
        }
        
        params = {
            "fields": ",".join(fields)
        }
        
        try:
            # Make POST request for batch operation
            response = await self.client.post(
                "/paper/batch",
                json=payload,
                params=params
            )
            
            response.raise_for_status()
            response_data = response.json()
            
            papers = []
            for paper_data in response_data:
                if paper_data is not None:  # API returns null for not found papers
                    papers.append(SemanticScholarPaper(**paper_data))
            
            return papers
        
        except Exception as e:
            logger.error(f"Batch papers request failed: {e}")
            return []
    
    async def search_papers(
        self,
        query: str,
        filters: Optional[SemanticScholarSearchFilters] = None,
        limit: int = 10,
        offset: int = 0,
        fields: Optional[List[str]] = None,
        use_cache: bool = True,
        cache_ttl: int = 900  # 15 minutes for search results
    ) -> SemanticScholarPapersResponse:
        """
        Search for papers using Semantic Scholar search API.
        
        Args:
            query: Search query string
            filters: Additional search filters
            limit: Maximum number of results (max 100)
            offset: Result offset for pagination
            fields: Fields to retrieve
            use_cache: Whether to use caching
            cache_ttl: Cache time-to-live in seconds
            
        Returns:
            SemanticScholarPapersResponse with results and metadata
        """
        if limit > 100:
            raise ValidationError("Maximum 100 results per request allowed")
        
        fields = fields or self.PAPER_FIELDS
        
        params = {
            "query": query,
            "limit": limit,
            "offset": offset,
            "fields": ",".join(fields)
        }
        
        # Add filters
        if filters:
            filter_params = filters.to_query_params()
            params.update(filter_params)
        
        response_data = await self._make_request(
            "/paper/search", 
            params=params, 
            use_cache=use_cache, 
            cache_ttl=cache_ttl
        )
        
        return SemanticScholarPapersResponse(**response_data)
    
    async def get_paper_citations(
        self,
        paper_id: str,
        limit: int = 100,
        offset: int = 0,
        fields: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> List[SemanticScholarPaper]:
        """
        Get papers that cite the specified paper.
        
        Args:
            paper_id: Paper identifier
            limit: Maximum number of citations to retrieve
            offset: Result offset
            fields: Fields to retrieve
            use_cache: Whether to use caching
            
        Returns:
            List of citing papers with citation context
        """
        fields = fields or self.CITATION_FIELDS
        
        params = {
            "limit": min(1000, limit),
            "offset": offset,
            "fields": ",".join(fields)
        }
        
        encoded_paper_id = quote(paper_id, safe="")
        
        try:
            response_data = await self._make_request(
                f"/paper/{encoded_paper_id}/citations",
                params=params,
                use_cache=use_cache
            )
            
            citations = []
            for citation_data in response_data.get("data", []):
                if "citingPaper" in citation_data:
                    paper_data = citation_data["citingPaper"]
                    # Add citation-specific metadata
                    paper_data["contexts"] = citation_data.get("contexts", [])
                    paper_data["intents"] = citation_data.get("intents", [])
                    paper_data["isInfluential"] = citation_data.get("isInfluential", False)
                    
                    citations.append(SemanticScholarPaper(**paper_data))
            
            return citations
        
        except Exception as e:
            logger.error(f"Error getting citations for paper {paper_id}: {e}")
            return []
    
    async def get_paper_references(
        self,
        paper_id: str,
        limit: int = 100,
        offset: int = 0,
        fields: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> List[SemanticScholarPaper]:
        """
        Get papers referenced by the specified paper.
        
        Args:
            paper_id: Paper identifier
            limit: Maximum number of references to retrieve
            offset: Result offset
            fields: Fields to retrieve
            use_cache: Whether to use caching
            
        Returns:
            List of referenced papers with citation context
        """
        fields = fields or self.CITATION_FIELDS
        
        params = {
            "limit": min(1000, limit),
            "offset": offset,
            "fields": ",".join(fields)
        }
        
        encoded_paper_id = quote(paper_id, safe="")
        
        try:
            response_data = await self._make_request(
                f"/paper/{encoded_paper_id}/references",
                params=params,
                use_cache=use_cache
            )
            
            references = []
            for reference_data in response_data.get("data", []):
                if "citedPaper" in reference_data:
                    paper_data = reference_data["citedPaper"]
                    # Add citation-specific metadata
                    paper_data["contexts"] = reference_data.get("contexts", [])
                    paper_data["intents"] = reference_data.get("intents", [])
                    paper_data["isInfluential"] = reference_data.get("isInfluential", False)
                    
                    references.append(SemanticScholarPaper(**paper_data))
            
            return references
        
        except Exception as e:
            logger.error(f"Error getting references for paper {paper_id}: {e}")
            return []
    
    async def get_paper_embedding(
        self, 
        paper_id: str, 
        use_cache: bool = True
    ) -> Optional[SemanticScholarEmbedding]:
        """
        Get SPECTER embedding for a paper.
        
        Args:
            paper_id: Paper identifier
            use_cache: Whether to use caching
            
        Returns:
            SPECTER embedding or None if not available
        """
        # Check cache first for embedding
        if use_cache and self.redis_manager:
            cache_key = f"semantic_scholar:embedding:{paper_id}"
            try:
                cached_embedding = await self.redis_manager.cache_get(cache_key)
                if cached_embedding:
                    self.embedding_cache_hits += 1
                    return SemanticScholarEmbedding(**cached_embedding)
            except Exception:
                pass
        
        # Get paper with embedding field
        paper = await self.get_paper_by_id(
            paper_id, 
            fields=["paperId", "embedding"],
            use_cache=use_cache
        )
        
        if paper and paper.embedding:
            # Cache the embedding separately for efficiency
            if use_cache and self.redis_manager:
                cache_key = f"semantic_scholar:embedding:{paper_id}"
                try:
                    await self.redis_manager.cache_set(
                        cache_key, 
                        paper.embedding.dict(),
                        ttl=86400  # Cache embeddings for 24 hours
                    )
                except Exception:
                    pass
            
            return paper.embedding
        
        return None
    
    async def compute_semantic_similarity(
        self,
        paper_id_1: str,
        paper_id_2: str,
        use_cache: bool = True
    ) -> Optional[float]:
        """
        Compute semantic similarity between two papers using SPECTER embeddings.
        
        Args:
            paper_id_1: First paper identifier
            paper_id_2: Second paper identifier
            use_cache: Whether to use caching
            
        Returns:
            Cosine similarity score between -1 and 1, or None if embeddings unavailable
        """
        # Get embeddings for both papers
        embedding_1 = await self.get_paper_embedding(paper_id_1, use_cache=use_cache)
        embedding_2 = await self.get_paper_embedding(paper_id_2, use_cache=use_cache)
        
        if not embedding_1 or not embedding_2:
            return None
        
        # Compute cosine similarity
        vec_1 = np.array(embedding_1.vector)
        vec_2 = np.array(embedding_2.vector)
        
        # Cosine similarity formula
        dot_product = np.dot(vec_1, vec_2)
        norm_1 = np.linalg.norm(vec_1)
        norm_2 = np.linalg.norm(vec_2)
        
        if norm_1 == 0 or norm_2 == 0:
            return 0.0
        
        similarity = dot_product / (norm_1 * norm_2)
        return float(similarity)
    
    async def find_similar_papers(
        self,
        paper_id: str,
        candidate_papers: List[str],
        similarity_threshold: float = 0.7,
        use_cache: bool = True
    ) -> List[SemanticScholarSimilarityResult]:
        """
        Find papers similar to the given paper from a list of candidates.
        
        Args:
            paper_id: Reference paper identifier
            candidate_papers: List of candidate paper identifiers
            similarity_threshold: Minimum similarity threshold
            use_cache: Whether to use caching
            
        Returns:
            List of similarity results sorted by similarity score
        """
        similarity_results = []
        
        # Get reference paper info for context
        ref_paper = await self.get_paper_by_id(
            paper_id, 
            fields=["paperId", "title"],
            use_cache=use_cache
        )
        
        for candidate_id in candidate_papers:
            similarity = await self.compute_semantic_similarity(
                paper_id, candidate_id, use_cache=use_cache
            )
            
            if similarity is not None and similarity >= similarity_threshold:
                # Get candidate paper info
                candidate_paper = await self.get_paper_by_id(
                    candidate_id,
                    fields=["paperId", "title"],
                    use_cache=use_cache
                )
                
                similarity_results.append(SemanticScholarSimilarityResult(
                    paper_id_1=paper_id,
                    paper_id_2=candidate_id,
                    similarity_score=similarity,
                    title_1=ref_paper.title if ref_paper else None,
                    title_2=candidate_paper.title if candidate_paper else None
                ))
        
        # Sort by similarity score descending
        return sorted(similarity_results, key=lambda x: x.similarity_score, reverse=True)
    
    async def analyze_citation_intents(
        self,
        citing_paper_id: str,
        cited_paper_id: str,
        use_cache: bool = True
    ) -> List[CitationIntent]:
        """
        Analyze citation intents between two papers.
        
        Args:
            citing_paper_id: Paper that cites
            cited_paper_id: Paper being cited
            use_cache: Whether to use caching
            
        Returns:
            List of citation intents
        """
        # Get citations with intent information
        citations = await self.get_paper_citations(
            cited_paper_id,
            limit=1000,
            fields=self.CITATION_FIELDS,
            use_cache=use_cache
        )
        
        # Find the specific citation
        for citation in citations:
            if citation.paper_id == citing_paper_id:
                return getattr(citation, 'intents', [])
        
        return []
    
    async def get_influential_citations(
        self,
        paper_id: str,
        limit: int = 50,
        use_cache: bool = True
    ) -> List[SemanticScholarInfluentialCitation]:
        """
        Get influential citations for a paper.
        
        Args:
            paper_id: Paper identifier
            limit: Maximum number of influential citations
            use_cache: Whether to use caching
            
        Returns:
            List of influential citations with metadata
        """
        citations = await self.get_paper_citations(
            paper_id,
            limit=limit * 3,  # Get more to filter for influential ones
            fields=self.CITATION_FIELDS,
            use_cache=use_cache
        )
        
        influential_citations = []
        
        for citation in citations:
            if getattr(citation, 'is_influential', False):
                influential_citation = SemanticScholarInfluentialCitation(
                    citing_paper_id=citation.paper_id,
                    cited_paper_id=paper_id,
                    is_influential=True,
                    contexts=getattr(citation, 'contexts', []),
                    intents=getattr(citation, 'intents', []),
                    citing_paper_title=citation.title,
                    cited_paper_title=None,  # Could be filled by caller
                    citation_year=citation.year
                )
                influential_citations.append(influential_citation)
                
                if len(influential_citations) >= limit:
                    break
        
        return influential_citations
    
    async def build_semantic_citation_network(
        self,
        center_paper_id: str,
        max_depth: int = 2,
        max_papers_per_level: int = 20,
        similarity_threshold: float = 0.5,
        use_cache: bool = True
    ) -> SemanticScholarCitationNetwork:
        """
        Build a semantically-enriched citation network.
        
        Args:
            center_paper_id: Central paper identifier
            max_depth: Maximum network depth
            max_papers_per_level: Maximum papers per level
            similarity_threshold: Minimum similarity for inclusion
            use_cache: Whether to use caching
            
        Returns:
            Enriched citation network with semantic features
        """
        if max_depth > 3:
            raise ValidationError("Maximum depth is 3 for semantic analysis")
        
        # Data structures
        nodes = {}  # paper_id -> SemanticScholarPaper
        edges = []
        visited = set()
        
        # Queue for BFS: (paper_id, depth, parent_id, relation)
        queue = [(center_paper_id, 0, None, None)]
        
        # Semantic analysis data
        influential_citations = []
        citation_intents = {}
        similarity_scores = {}
        
        while queue:
            current_id, depth, parent_id, relation = queue.pop(0)
            
            if depth > max_depth or current_id in visited:
                continue
            
            visited.add(current_id)
            
            try:
                # Get current paper
                paper = await self.get_paper_by_id(
                    current_id,
                    fields=self.PAPER_FIELDS,
                    use_cache=use_cache
                )
                
                if not paper:
                    continue
                
                nodes[current_id] = paper
                
                # Add edge to parent
                if parent_id and relation:
                    edge_id = f"{parent_id}|{current_id}"
                    edges.append({
                        "source_id": parent_id,
                        "target_id": current_id,
                        "relation_type": relation,
                        "depth": depth
                    })
                    
                    # Analyze citation intent and influence
                    if relation == "cites":
                        intents = await self.analyze_citation_intents(
                            parent_id, current_id, use_cache=use_cache
                        )
                        citation_intents[edge_id] = intents
                        
                        # Check if influential
                        influential_cites = await self.get_influential_citations(
                            current_id, limit=100, use_cache=use_cache
                        )
                        
                        for inf_cite in influential_cites:
                            if inf_cite.citing_paper_id == parent_id:
                                influential_citations.append(inf_cite)
                    
                    # Compute semantic similarity
                    similarity = await self.compute_semantic_similarity(
                        parent_id, current_id, use_cache=use_cache
                    )
                    
                    if similarity is not None:
                        similarity_scores[edge_id] = similarity
                
                # Continue traversal
                if depth < max_depth:
                    # Get citing papers
                    citing_papers = await self.get_paper_citations(
                        current_id,
                        limit=max_papers_per_level,
                        use_cache=use_cache
                    )
                    
                    for citing_paper in citing_papers:
                        if citing_paper.paper_id not in visited:
                            queue.append((citing_paper.paper_id, depth + 1, current_id, "cited_by"))
                    
                    # Get referenced papers  
                    referenced_papers = await self.get_paper_references(
                        current_id,
                        limit=max_papers_per_level,
                        use_cache=use_cache
                    )
                    
                    for ref_paper in referenced_papers:
                        if ref_paper.paper_id not in visited:
                            queue.append((ref_paper.paper_id, depth + 1, current_id, "references"))
            
            except Exception as e:
                logger.warning(f"Error processing paper {current_id}: {e}")
                continue
        
        # Filter edges by semantic similarity if threshold is set
        if similarity_threshold > 0:
            filtered_edges = []
            for edge in edges:
                edge_id = f"{edge['source_id']}|{edge['target_id']}"
                similarity = similarity_scores.get(edge_id, 0)
                if similarity >= similarity_threshold:
                    filtered_edges.append(edge)
            edges = filtered_edges
        
        return SemanticScholarCitationNetwork(
            center_paper_id=center_paper_id,
            nodes=list(nodes.values()),
            edges=edges,
            influential_citations=influential_citations,
            citation_intents=citation_intents,
            similarity_scores=similarity_scores,
            total_nodes=len(nodes),
            total_edges=len(edges),
            influential_edges=len(influential_citations)
        )
    
    async def get_author_by_id(
        self, 
        author_id: str, 
        use_cache: bool = True
    ) -> Optional[SemanticScholarAuthorResponse]:
        """Get author information by Semantic Scholar ID."""
        try:
            response_data = await self._make_request(
                f"/author/{author_id}",
                use_cache=use_cache
            )
            return SemanticScholarAuthorResponse(**response_data)
        except APIError as e:
            if "not found" in str(e).lower():
                return None
            raise
    
    async def get_rate_limit_info(self) -> SemanticScholarRateLimit:
        """Get current rate limit information."""
        return SemanticScholarRateLimit(
            requests_per_second=self.rate_limiter.requests_per_second,
            requests_per_minute=self.rate_limiter.requests_per_minute,
            is_authenticated=self.is_authenticated
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get client performance metrics."""
        return {
            "total_requests": self.request_count,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hits / max(self.cache_hits + self.cache_misses, 1),
            "embedding_cache_hits": self.embedding_cache_hits,
            "rate_limit_waits": self.rate_limit_waits,
            "is_authenticated": self.is_authenticated,
            "circuit_breaker_state": self.circuit_breaker.state,
            "circuit_breaker_failures": self.circuit_breaker.failure_count
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on Semantic Scholar API."""
        try:
            start_time = time.time()
            
            # Make a simple request
            response_data = await self._make_request(
                "/paper/search",
                {"query": "test", "limit": 1},
                use_cache=False
            )
            
            end_time = time.time()
            response_time = (end_time - start_time) * 1000
            
            rate_limit_info = await self.get_rate_limit_info()
            
            return {
                "status": "healthy",
                "response_time_ms": round(response_time, 2),
                "rate_limit": {
                    "requests_per_second": rate_limit_info.requests_per_second,
                    "is_authenticated": rate_limit_info.is_authenticated
                },
                "circuit_breaker_state": self.circuit_breaker.state,
                "metrics": self.get_metrics()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "circuit_breaker_state": self.circuit_breaker.state
            }


# Global client instance
_semantic_scholar_client: Optional[SemanticScholarClient] = None


async def get_semantic_scholar_client(redis_manager: Optional[RedisManager] = None) -> SemanticScholarClient:
    """Get or create Semantic Scholar client instance."""
    global _semantic_scholar_client
    
    if _semantic_scholar_client is None:
        if redis_manager is None:
            redis_manager = await get_redis_manager()
        
        _semantic_scholar_client = SemanticScholarClient(redis_manager=redis_manager)
    
    return _semantic_scholar_client


async def close_semantic_scholar_client():
    """Close the global Semantic Scholar client."""
    global _semantic_scholar_client
    
    if _semantic_scholar_client:
        await _semantic_scholar_client.close()
        _semantic_scholar_client = None