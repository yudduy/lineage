"""
OpenAlex API client with comprehensive functionality.

Production-ready client for the OpenAlex API with rate limiting, caching,
error handling, and batch processing capabilities.
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from datetime import datetime, timedelta
from urllib.parse import urlencode, quote

import httpx
from httpx import AsyncClient, Response, HTTPStatusError, RequestError

from ..core.config import get_settings
from ..db.redis import RedisManager, get_redis_manager
from ..models.openalex import (
    OpenAlexWork,
    OpenAlexWorksResponse,
    OpenAlexAuthorResponse,
    OpenAlexInstitutionResponse,
    OpenAlexVenueResponse,
    OpenAlexConceptResponse,
    OpenAlexSearchFilters,
    OpenAlexBatchRequest,
    OpenAlexError,
    OpenAlexRateLimit
)
from ..utils.logger import get_logger
from ..utils.exceptions import APIError, RateLimitError, ValidationError

logger = get_logger(__name__)


class CircuitBreakerError(Exception):
    """Circuit breaker is open."""
    pass


class CircuitBreaker:
    """Circuit breaker pattern for API fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
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


class RateLimiter:
    """Rate limiter with exponential backoff."""
    
    def __init__(self, requests_per_second: int = 10, burst_size: int = 50):
        self.requests_per_second = requests_per_second
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_update = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """Acquire permission to make a request."""
        async with self._lock:
            now = time.time()
            time_passed = now - self.last_update
            
            # Add tokens based on time passed
            tokens_to_add = time_passed * self.requests_per_second
            self.tokens = min(self.burst_size, self.tokens + tokens_to_add)
            self.last_update = now
            
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            
            return False
    
    async def wait_if_needed(self):
        """Wait if rate limit would be exceeded."""
        if not await self.acquire():
            wait_time = (1 - self.tokens) / self.requests_per_second
            await asyncio.sleep(wait_time)
            await self.acquire()


class OpenAlexClient:
    """
    Comprehensive OpenAlex API client.
    
    Features:
    - Rate limiting with polite pool support
    - Redis caching for responses
    - Circuit breaker for fault tolerance
    - Exponential backoff retry logic
    - Batch processing capabilities
    - Citation network traversal
    - Comprehensive error handling
    """
    
    BASE_URL = "https://api.openalex.org"
    
    def __init__(
        self,
        email: Optional[str] = None,
        redis_manager: Optional[RedisManager] = None,
        user_agent: Optional[str] = None,
        cache_ttl: int = 3600,  # 1 hour default cache
        requests_per_second: int = 10,  # Default rate limit
        max_retries: int = 3,
        circuit_breaker_threshold: int = 5
    ):
        self.settings = get_settings()
        self.email = email or getattr(self.settings.external_apis, 'openalex_email', None)
        self.redis_manager = redis_manager
        self.cache_ttl = cache_ttl
        self.max_retries = max_retries
        
        # Set up rate limiting - use higher limits for polite pool
        if self.email:
            self.rate_limiter = RateLimiter(requests_per_second=100, burst_size=200)  # Polite pool
            self.is_polite_pool = True
        else:
            self.rate_limiter = RateLimiter(requests_per_second=requests_per_second, burst_size=50)
            self.is_polite_pool = False
        
        # Circuit breaker for fault tolerance
        self.circuit_breaker = CircuitBreaker(failure_threshold=circuit_breaker_threshold)
        
        # HTTP client setup
        headers = {
            "User-Agent": user_agent or f"Citation Network Explorer (mailto:{self.email or 'noreply@example.com'})",
            "Accept": "application/json"
        }
        
        if self.email:
            headers["From"] = self.email
        
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
        
        logger.info(
            f"OpenAlex client initialized with {'polite pool' if self.is_polite_pool else 'standard'} rate limits"
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
        return f"openalex:{endpoint}:{hash(param_str)}"
    
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
            raise CircuitBreakerError("Circuit breaker is open")
        
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
                
                logger.debug(f"OpenAlex API request: {endpoint} (attempt {attempt + 1})")
                
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
                        wait_time = (2 ** attempt) + (0.1 * attempt)  # Exponential backoff
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
            raise APIError(f"OpenAlex API request failed after {self.max_retries} retries: {last_exception}")
        
        raise APIError("OpenAlex API request failed for unknown reason")
    
    async def get_work_by_id(self, work_id: str, use_cache: bool = True) -> Optional[OpenAlexWork]:
        """
        Get a work by its OpenAlex ID or DOI.
        
        Args:
            work_id: OpenAlex work ID or DOI
            use_cache: Whether to use caching
            
        Returns:
            OpenAlexWork object or None if not found
        """
        # Normalize ID
        if work_id.startswith("10."):
            work_id = f"https://doi.org/{work_id}"
        elif not work_id.startswith("https://openalex.org/"):
            work_id = f"https://openalex.org/{work_id}"
        
        try:
            response_data = await self._make_request(f"/works/{work_id}", use_cache=use_cache)
            return OpenAlexWork(**response_data)
        
        except APIError as e:
            if "not found" in str(e).lower():
                return None
            raise
    
    async def get_works_batch(
        self,
        work_ids: List[str],
        use_cache: bool = True
    ) -> List[OpenAlexWork]:
        """
        Get multiple works in a single request.
        
        Args:
            work_ids: List of OpenAlex work IDs or DOIs (max 50)
            use_cache: Whether to use caching
            
        Returns:
            List of OpenAlexWork objects
        """
        if len(work_ids) > 50:
            raise ValidationError("Maximum 50 work IDs allowed per batch request")
        
        # Normalize IDs
        normalized_ids = []
        for work_id in work_ids:
            if work_id.startswith("10."):
                normalized_ids.append(f"https://doi.org/{work_id}")
            elif not work_id.startswith("https://openalex.org/"):
                normalized_ids.append(f"https://openalex.org/{work_id}")
            else:
                normalized_ids.append(work_id)
        
        params = {"filter": f"openalex_id:{','.join(normalized_ids)}", "per_page": 50}
        
        try:
            response_data = await self._make_request("/works", params=params, use_cache=use_cache)
            works_response = OpenAlexWorksResponse(**response_data)
            return works_response.results
        
        except Exception as e:
            logger.error(f"Batch works request failed: {e}")
            return []
    
    async def search_works(
        self,
        query: str,
        filters: Optional[OpenAlexSearchFilters] = None,
        sort: str = "relevance_score:desc",
        page: int = 1,
        per_page: int = 25,
        use_cache: bool = True,
        cache_ttl: int = 900  # 15 minutes for search results
    ) -> OpenAlexWorksResponse:
        """
        Search for works using OpenAlex search API.
        
        Args:
            query: Search query string
            filters: Additional search filters
            sort: Sort order (relevance_score:desc, cited_by_count:desc, publication_date:desc, etc.)
            page: Page number (1-based)
            per_page: Results per page (max 200)
            use_cache: Whether to use caching
            cache_ttl: Cache time-to-live in seconds
            
        Returns:
            OpenAlexWorksResponse with results and metadata
        """
        if per_page > 200:
            raise ValidationError("Maximum 200 results per page allowed")
        
        params = {
            "search": query,
            "sort": sort,
            "page": page,
            "per_page": per_page
        }
        
        # Add filters
        if filters:
            filter_params = filters.to_query_params()
            params.update(filter_params)
        
        response_data = await self._make_request("/works", params=params, use_cache=use_cache, cache_ttl=cache_ttl)
        return OpenAlexWorksResponse(**response_data)
    
    async def get_citations(
        self,
        work_id: str,
        direction: str = "cited_by",
        max_results: int = 100,
        use_cache: bool = True
    ) -> List[OpenAlexWork]:
        """
        Get citations for a work (papers that cite this work or papers this work cites).
        
        Args:
            work_id: OpenAlex work ID or DOI
            direction: "cited_by" (papers citing this work) or "references" (papers this work cites)
            max_results: Maximum number of results to return
            use_cache: Whether to use caching
            
        Returns:
            List of citing/referenced works
        """
        if direction not in ["cited_by", "references"]:
            raise ValidationError("Direction must be 'cited_by' or 'references'")
        
        # Normalize work ID
        if work_id.startswith("10."):
            work_id = f"https://doi.org/{work_id}"
        elif not work_id.startswith("https://openalex.org/"):
            work_id = f"https://openalex.org/{work_id}"
        
        all_works = []
        page = 1
        per_page = min(200, max_results)
        
        while len(all_works) < max_results:
            if direction == "cited_by":
                params = {
                    "filter": f"cites:{work_id}",
                    "sort": "cited_by_count:desc",
                    "page": page,
                    "per_page": per_page
                }
            else:  # references
                # First get the work to access its references
                work = await self.get_work_by_id(work_id, use_cache=use_cache)
                if not work or not work.referenced_works:
                    break
                
                # Get referenced works in batches
                remaining_refs = work.referenced_works[len(all_works):]
                if not remaining_refs:
                    break
                
                batch_refs = remaining_refs[:min(50, max_results - len(all_works))]
                batch_works = await self.get_works_batch(batch_refs, use_cache=use_cache)
                all_works.extend(batch_works)
                break  # References are finite, no pagination needed
            
            if direction == "cited_by":
                try:
                    response_data = await self._make_request("/works", params=params, use_cache=use_cache)
                    works_response = OpenAlexWorksResponse(**response_data)
                    
                    if not works_response.results:
                        break
                    
                    all_works.extend(works_response.results)
                    
                    # Check if we have more results
                    total_results = works_response.meta.get("count", 0)
                    if len(all_works) >= total_results:
                        break
                    
                    page += 1
                    
                except Exception as e:
                    logger.error(f"Error getting citations page {page}: {e}")
                    break
        
        return all_works[:max_results]
    
    async def traverse_citation_network(
        self,
        start_work_id: str,
        direction: str = "both",
        max_depth: int = 2,
        max_works_per_level: int = 50,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Traverse citation network from a starting work.
        
        Args:
            start_work_id: Starting work ID
            direction: "cited_by", "references", or "both"
            max_depth: Maximum traversal depth
            max_works_per_level: Maximum works to explore per level
            use_cache: Whether to use caching
            
        Returns:
            Dictionary containing nodes, edges, and traversal metadata
        """
        if direction not in ["cited_by", "references", "both"]:
            raise ValidationError("Direction must be 'cited_by', 'references', or 'both'")
        
        if max_depth > 5:
            raise ValidationError("Maximum depth is 5")
        
        # Data structures for traversal
        nodes = {}  # work_id -> OpenAlexWork
        edges = []  # List of edge dicts
        visited = set()  # Track visited work IDs
        
        # Queue for BFS traversal: (work_id, depth, parent_id)
        queue = [(start_work_id, 0, None)]
        
        while queue:
            current_work_id, current_depth, parent_id = queue.pop(0)
            
            if current_depth > max_depth or current_work_id in visited:
                continue
            
            visited.add(current_work_id)
            
            try:
                # Get current work
                work = await self.get_work_by_id(current_work_id, use_cache=use_cache)
                if not work:
                    continue
                
                nodes[current_work_id] = work
                
                # Add edge to parent if exists
                if parent_id:
                    edge_type = "cites" if parent_id in work.referenced_works else "cited_by"
                    edges.append({
                        "source_id": parent_id,
                        "target_id": current_work_id,
                        "edge_type": edge_type,
                        "weight": 1.0,
                        "depth": current_depth
                    })
                
                # Continue traversal if not at max depth
                if current_depth < max_depth:
                    next_works = []
                    
                    # Get citing works
                    if direction in ["cited_by", "both"]:
                        citing_works = await self.get_citations(
                            current_work_id,
                            "cited_by",
                            max_works_per_level,
                            use_cache=use_cache
                        )
                        next_works.extend([(w.id, "cited_by") for w in citing_works])
                    
                    # Get referenced works
                    if direction in ["references", "both"]:
                        referenced_works = await self.get_citations(
                            current_work_id,
                            "references",
                            max_works_per_level,
                            use_cache=use_cache
                        )
                        next_works.extend([(w.id, "references") for w in referenced_works])
                    
                    # Add to queue
                    for next_work_id, _ in next_works:
                        if next_work_id not in visited:
                            queue.append((next_work_id, current_depth + 1, current_work_id))
            
            except Exception as e:
                logger.warning(f"Error traversing work {current_work_id}: {e}")
                continue
        
        return {
            "center_work_id": start_work_id,
            "nodes": list(nodes.values()),
            "edges": edges,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "max_depth_reached": max([e.get("depth", 0) for e in edges] + [0]),
            "direction": direction
        }
    
    async def get_author_by_id(self, author_id: str, use_cache: bool = True) -> Optional[OpenAlexAuthorResponse]:
        """Get author by OpenAlex ID."""
        if not author_id.startswith("https://openalex.org/"):
            author_id = f"https://openalex.org/{author_id}"
        
        try:
            response_data = await self._make_request(f"/authors/{author_id}", use_cache=use_cache)
            return OpenAlexAuthorResponse(**response_data)
        except APIError as e:
            if "not found" in str(e).lower():
                return None
            raise
    
    async def get_institution_by_id(self, institution_id: str, use_cache: bool = True) -> Optional[OpenAlexInstitutionResponse]:
        """Get institution by OpenAlex ID."""
        if not institution_id.startswith("https://openalex.org/"):
            institution_id = f"https://openalex.org/{institution_id}"
        
        try:
            response_data = await self._make_request(f"/institutions/{institution_id}", use_cache=use_cache)
            return OpenAlexInstitutionResponse(**response_data)
        except APIError as e:
            if "not found" in str(e).lower():
                return None
            raise
    
    async def get_venue_by_id(self, venue_id: str, use_cache: bool = True) -> Optional[OpenAlexVenueResponse]:
        """Get venue by OpenAlex ID."""
        if not venue_id.startswith("https://openalex.org/"):
            venue_id = f"https://openalex.org/{venue_id}"
        
        try:
            response_data = await self._make_request(f"/sources/{venue_id}", use_cache=use_cache)
            return OpenAlexVenueResponse(**response_data)
        except APIError as e:
            if "not found" in str(e).lower():
                return None
            raise
    
    async def get_concept_by_id(self, concept_id: str, use_cache: bool = True) -> Optional[OpenAlexConceptResponse]:
        """Get concept by OpenAlex ID."""
        if not concept_id.startswith("https://openalex.org/"):
            concept_id = f"https://openalex.org/{concept_id}"
        
        try:
            response_data = await self._make_request(f"/concepts/{concept_id}", use_cache=use_cache)
            return OpenAlexConceptResponse(**response_data)
        except APIError as e:
            if "not found" in str(e).lower():
                return None
            raise
    
    async def get_works_by_author(
        self,
        author_id: str,
        limit: int = 100,
        sort: str = "cited_by_count:desc",
        use_cache: bool = True
    ) -> List[OpenAlexWork]:
        """Get works by an author."""
        if not author_id.startswith("https://openalex.org/"):
            author_id = f"https://openalex.org/{author_id}"
        
        params = {
            "filter": f"authorships.author.id:{author_id}",
            "sort": sort,
            "per_page": min(200, limit)
        }
        
        works = []
        page = 1
        
        while len(works) < limit:
            params["page"] = page
            
            try:
                response_data = await self._make_request("/works", params=params, use_cache=use_cache)
                works_response = OpenAlexWorksResponse(**response_data)
                
                if not works_response.results:
                    break
                
                works.extend(works_response.results)
                
                if len(works_response.results) < params["per_page"]:
                    break
                
                page += 1
                
            except Exception as e:
                logger.error(f"Error getting works for author {author_id}: {e}")
                break
        
        return works[:limit]
    
    async def get_rate_limit_info(self) -> OpenAlexRateLimit:
        """Get current rate limit information."""
        # Make a lightweight request to get rate limit headers
        try:
            response = await self.client.get("/works", params={"per_page": 1})
            
            # Extract rate limit information from headers
            requests_remaining = int(response.headers.get("x-ratelimit-remaining", 0))
            requests_per_day = int(response.headers.get("x-ratelimit-limit", 100000))
            
            # Calculate reset time (OpenAlex resets daily at midnight UTC)
            now = datetime.utcnow()
            next_reset = datetime(now.year, now.month, now.day) + timedelta(days=1)
            
            return OpenAlexRateLimit(
                requests_remaining=requests_remaining,
                requests_per_day=requests_per_day,
                reset_time=next_reset,
                is_polite_pool=self.is_polite_pool
            )
            
        except Exception as e:
            logger.warning(f"Could not get rate limit info: {e}")
            return OpenAlexRateLimit(
                requests_remaining=0,
                requests_per_day=240000 if self.is_polite_pool else 100000,
                reset_time=datetime.utcnow() + timedelta(days=1),
                is_polite_pool=self.is_polite_pool
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get client performance metrics."""
        return {
            "total_requests": self.request_count,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hits / max(self.cache_hits + self.cache_misses, 1),
            "rate_limit_waits": self.rate_limit_waits,
            "is_polite_pool": self.is_polite_pool,
            "circuit_breaker_state": self.circuit_breaker.state,
            "circuit_breaker_failures": self.circuit_breaker.failure_count
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on OpenAlex API."""
        try:
            start_time = time.time()
            
            # Make a simple request
            response_data = await self._make_request("/works", {"per_page": 1}, use_cache=False)
            
            end_time = time.time()
            response_time = (end_time - start_time) * 1000
            
            rate_limit_info = await self.get_rate_limit_info()
            
            return {
                "status": "healthy",
                "response_time_ms": round(response_time, 2),
                "rate_limit": {
                    "requests_remaining": rate_limit_info.requests_remaining,
                    "requests_per_day": rate_limit_info.requests_per_day,
                    "is_polite_pool": rate_limit_info.is_polite_pool
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
    
    async def build_citation_network_sync(
        self,
        center_identifier: str,
        direction: str = "both",
        max_depth: int = 2,
        max_per_level: int = 20
    ) -> Dict[str, Any]:
        """
        Build citation network synchronously and persist to Neo4j.
        
        This is the minimal demo version that:
        1. Resolves the identifier (DOI, OpenAlex URL/ID, or title)
        2. Traverses citations using existing methods
        3. Persists to Neo4j
        4. Returns compact payload
        
        Args:
            center_identifier: DOI, OpenAlex URL/ID, or paper title
            direction: "backward" (references), "forward" (cited_by), or "both"
            max_depth: Maximum traversal depth (capped at 3)
            max_per_level: Maximum papers per level (capped at 50)
            
        Returns:
            Compact network data with nodes and edges
        """
        from ..db.neo4j import get_neo4j_manager
        from .openalex_converter import OpenAlexConverter
        
        # Validate and cap parameters
        max_depth = min(max_depth, 3)
        max_per_level = min(max_per_level, 50)
        
        # Map direction terms
        direction_map = {
            "backward": "references",
            "forward": "cited_by",
            "both": "both"
        }
        traverse_direction = direction_map.get(direction, direction)
        
        # Step 1: Resolve identifier
        center_work = None
        
        # Try as DOI
        if center_identifier.startswith("10."):
            center_work = await self.get_work_by_id(center_identifier)
        # Try as OpenAlex URL or ID
        elif "openalex.org" in center_identifier or center_identifier.startswith("W"):
            center_work = await self.get_work_by_id(center_identifier)
        # Try as title search
        else:
            search_results = await self.search_works(
                query=center_identifier,
                per_page=1
            )
            if search_results:
                center_work = search_results[0]
        
        if not center_work:
            raise ValidationError(f"Could not find paper with identifier: {center_identifier}")
        
        # Step 2: Traverse citation network
        network_data = await self.traverse_citation_network(
            start_work_id=center_work.id,
            direction=traverse_direction,
            max_depth=max_depth,
            max_works_per_level=max_per_level
        )
        
        # Step 3: Persist to Neo4j
        neo4j_manager = await get_neo4j_manager()
        converter = OpenAlexConverter()
        
        # Convert and persist nodes
        for work in network_data["nodes"]:
            paper = converter.convert_openalex_work_to_paper(work)
            
            # Create or update paper node
            await neo4j_manager.execute_write(
                """
                MERGE (p:Paper {id: $id})
                SET p.title = $title,
                    p.doi = $doi,
                    p.publication_year = $year,
                    p.citation_count = $citation_count,
                    p.abstract = $abstract,
                    p.url = $url,
                    p.authors = $authors,
                    p.journal = $journal,
                    p.updated_at = datetime()
                """,
                {
                    "id": paper.id,
                    "title": paper.title,
                    "doi": paper.doi,
                    "year": paper.publication_year,
                    "citation_count": paper.citation_count.total if paper.citation_count else 0,
                    "abstract": paper.abstract[:1000] if paper.abstract else None,
                    "url": paper.url,
                    "authors": [a.name for a in paper.authors] if paper.authors else [],
                    "journal": paper.journal.name if paper.journal else None
                }
            )
        
        # Create citation edges
        for edge in network_data["edges"]:
            await neo4j_manager.execute_write(
                """
                MATCH (citing:Paper {id: $citing_id})
                MATCH (cited:Paper {id: $cited_id})
                MERGE (citing)-[r:CITES]->(cited)
                SET r.created_at = datetime()
                """,
                {
                    "citing_id": edge["source_id"].split('/')[-1] if '/' in edge["source_id"] else edge["source_id"],
                    "cited_id": edge["target_id"].split('/')[-1] if '/' in edge["target_id"] else edge["target_id"]
                }
            )
        
        # Step 4: Return compact payload
        nodes_simple = []
        for work in network_data["nodes"]:
            work_id = work.id.split('/')[-1] if work.id else None
            nodes_simple.append({
                "id": work_id,
                "title": work.title[:100] if work.title else "Untitled",
                "publication_year": work.publication_year,
                "doi": work.ids.doi if work.ids else None,
                "citation_count": work.cited_by_count
            })
        
        edges_simple = []
        for edge in network_data["edges"]:
            source_id = edge["source_id"].split('/')[-1] if '/' in edge["source_id"] else edge["source_id"]
            target_id = edge["target_id"].split('/')[-1] if '/' in edge["target_id"] else edge["target_id"]
            edges_simple.append({
                "source_id": source_id,
                "target_id": target_id
            })
        
        return {
            "center_paper_id": center_work.id.split('/')[-1] if center_work.id else None,
            "total_nodes": len(nodes_simple),
            "total_edges": len(edges_simple),
            "max_depth_reached": network_data.get("max_depth_reached", 0),
            "nodes": nodes_simple,
            "edges": edges_simple
        }


# Global client instance
_openalex_client: Optional[OpenAlexClient] = None


async def get_openalex_client(redis_manager: Optional[RedisManager] = None) -> OpenAlexClient:
    """Get or create OpenAlex client instance."""
    global _openalex_client
    
    if _openalex_client is None:
        if redis_manager is None:
            redis_manager = await get_redis_manager()
        
        _openalex_client = OpenAlexClient(redis_manager=redis_manager)
    
    return _openalex_client


async def close_openalex_client():
    """Close the global OpenAlex client."""
    global _openalex_client
    
    if _openalex_client:
        await _openalex_client.close()
        _openalex_client = None