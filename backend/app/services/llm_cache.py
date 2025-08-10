"""
LLM Intelligent Caching Service - Advanced caching with semantic similarity and content deduplication.

This service provides:
- Semantic similarity-based cache matching
- Content deduplication to avoid redundant API calls
- TTL-based cache expiration with intelligent refresh
- Cache warming for frequently accessed content
- Cache analytics and optimization
"""

import asyncio
import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import logging

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from ..core.config import Settings, get_settings
from ..db.redis import RedisManager, get_redis_manager
from ..utils.logger import get_logger

logger = get_logger(__name__)


class CacheType(Enum):
    """Types of cache entries."""
    EXACT_MATCH = "exact_match"
    SEMANTIC_MATCH = "semantic_match"
    CONTENT_DEDUP = "content_dedup"
    PRECOMPUTED = "precomputed"


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    content: str
    model: str
    provider: str
    cost: float
    tokens: int
    cache_type: CacheType
    timestamp: datetime
    access_count: int
    last_accessed: datetime
    ttl_seconds: int
    embedding_hash: Optional[str] = None
    content_hash: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['cache_type'] = self.cache_type.value
        data['timestamp'] = self.timestamp.isoformat()
        data['last_accessed'] = self.last_accessed.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create from dictionary."""
        data['cache_type'] = CacheType(data['cache_type'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['last_accessed'] = datetime.fromisoformat(data['last_accessed'])
        return cls(**data)
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return datetime.now() - self.timestamp > timedelta(seconds=self.ttl_seconds)
    
    def refresh_access(self):
        """Update access metadata."""
        self.access_count += 1
        self.last_accessed = datetime.now()


@dataclass
class CacheStats:
    """Cache performance statistics."""
    total_requests: int
    cache_hits: int
    cache_misses: int
    semantic_hits: int
    exact_hits: int
    hit_rate: float
    semantic_hit_rate: float
    avg_response_time_ms: float
    cost_savings: float
    total_cache_entries: int
    cache_size_mb: float
    top_cached_models: Dict[str, int]
    cache_age_distribution: Dict[str, int]


class SemanticCacheManager:
    """
    Advanced caching system with semantic similarity matching and intelligent optimization.
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        redis_manager: Optional[RedisManager] = None
    ):
        self.settings = settings or get_settings()
        self.redis_manager = redis_manager
        self._embedding_model = None
        self._cache_stats = {
            'requests': 0,
            'hits': 0,
            'misses': 0,
            'semantic_hits': 0,
            'exact_hits': 0
        }
        self._initialized = False
        
        # Cache configuration
        self.similarity_threshold = self.settings.llm.semantic_similarity_threshold
        self.default_ttl = self.settings.llm.cache_ttl_hours * 3600
        self.max_cache_entries = 10000  # Configurable limit
        
    async def initialize(self):
        """Initialize the cache manager."""
        if self._initialized:
            return
        
        if not self.redis_manager:
            self.redis_manager = await get_redis_manager()
        
        # Initialize embedding model for semantic similarity
        if self.settings.llm.enable_semantic_caching:
            await self._initialize_embedding_model()
        
        self._initialized = True
        logger.info("Semantic Cache Manager initialized")
    
    async def _initialize_embedding_model(self):
        """Initialize the embedding model for semantic similarity."""
        try:
            # Use a fast, lightweight model for cache matching
            self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model initialized for semantic caching")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            self._embedding_model = None
    
    def _generate_content_hash(self, content: str) -> str:
        """Generate hash for content deduplication."""
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _generate_prompt_hash(self, prompt: str, model: str, temperature: float, max_tokens: int) -> str:
        """Generate hash for exact prompt matching."""
        content = f"{prompt}:{model}:{temperature}:{max_tokens}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    async def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for semantic similarity."""
        if not self._embedding_model:
            return None
        
        try:
            embedding = self._embedding_model.encode([text])[0]
            return embedding
        except Exception as e:
            logger.warning(f"Failed to generate embedding: {e}")
            return None
    
    async def _store_embedding(self, embedding_hash: str, embedding: np.ndarray):
        """Store embedding in Redis."""
        if not self.redis_manager:
            return
        
        try:
            embedding_key = f"llm_embedding:{embedding_hash}"
            embedding_bytes = embedding.tobytes()
            await self.redis_manager.setex(
                embedding_key,
                self.default_ttl,
                embedding_bytes
            )
        except Exception as e:
            logger.warning(f"Failed to store embedding: {e}")
    
    async def _get_embedding_from_cache(self, embedding_hash: str) -> Optional[np.ndarray]:
        """Retrieve embedding from Redis."""
        if not self.redis_manager:
            return None
        
        try:
            embedding_key = f"llm_embedding:{embedding_hash}"
            embedding_bytes = await self.redis_manager.get(embedding_key)
            
            if embedding_bytes:
                return np.frombuffer(embedding_bytes, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Failed to retrieve embedding: {e}")
        
        return None
    
    async def _find_exact_match(self, prompt_hash: str) -> Optional[CacheEntry]:
        """Find exact cache match."""
        if not self.redis_manager:
            return None
        
        try:
            cache_key = f"llm_cache:exact:{prompt_hash}"
            cached_data = await self.redis_manager.get(cache_key)
            
            if cached_data:
                entry_data = json.loads(cached_data)
                entry = CacheEntry.from_dict(entry_data)
                
                if not entry.is_expired():
                    entry.refresh_access()
                    await self._update_cache_entry(entry)
                    return entry
                else:
                    # Remove expired entry
                    await self.redis_manager.delete(cache_key)
        except Exception as e:
            logger.warning(f"Failed to find exact match: {e}")
        
        return None
    
    async def _find_semantic_matches(
        self,
        query_embedding: np.ndarray,
        prompt: str,
        model: str
    ) -> List[Tuple[CacheEntry, float]]:
        """Find semantically similar cached responses."""
        if not self.redis_manager or not self._embedding_model:
            return []
        
        matches = []
        
        try:
            # Get all semantic cache keys for this model
            pattern = f"llm_cache:semantic:{model}:*"
            cache_keys = await self.redis_manager.keys(pattern)
            
            # Limit search to avoid performance issues
            search_limit = min(100, len(cache_keys))
            
            for cache_key in cache_keys[:search_limit]:
                try:
                    cached_data = await self.redis_manager.get(cache_key)
                    if not cached_data:
                        continue
                    
                    entry_data = json.loads(cached_data)
                    entry = CacheEntry.from_dict(entry_data)
                    
                    if entry.is_expired():
                        await self.redis_manager.delete(cache_key)
                        continue
                    
                    # Get cached embedding
                    if entry.embedding_hash:
                        cached_embedding = await self._get_embedding_from_cache(entry.embedding_hash)
                        
                        if cached_embedding is not None:
                            # Calculate similarity
                            similarity = cosine_similarity(
                                query_embedding.reshape(1, -1),
                                cached_embedding.reshape(1, -1)
                            )[0][0]
                            
                            if similarity >= self.similarity_threshold:
                                matches.append((entry, similarity))
                
                except Exception as e:
                    logger.warning(f"Error processing cache key {cache_key}: {e}")
                    continue
            
            # Sort by similarity (highest first)
            matches.sort(key=lambda x: x[1], reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to find semantic matches: {e}")
        
        return matches
    
    async def _store_cache_entry(
        self,
        entry: CacheEntry,
        prompt: str,
        prompt_hash: str,
        embedding: Optional[np.ndarray] = None
    ):
        """Store cache entry with both exact and semantic indexing."""
        if not self.redis_manager:
            return
        
        try:
            # Store exact match cache
            exact_key = f"llm_cache:exact:{prompt_hash}"
            await self.redis_manager.setex(
                exact_key,
                entry.ttl_seconds,
                json.dumps(entry.to_dict())
            )
            
            # Store semantic cache if embedding available
            if embedding is not None and entry.embedding_hash:
                semantic_key = f"llm_cache:semantic:{entry.model}:{entry.embedding_hash}"
                await self.redis_manager.setex(
                    semantic_key,
                    entry.ttl_seconds,
                    json.dumps(entry.to_dict())
                )
                
                # Store embedding separately
                await self._store_embedding(entry.embedding_hash, embedding)
            
            # Add to cache index for cleanup
            index_key = "llm_cache:index"
            await self.redis_manager.sadd(index_key, exact_key)
            if embedding is not None:
                await self.redis_manager.sadd(index_key, f"llm_cache:semantic:{entry.model}:{entry.embedding_hash}")
            
        except Exception as e:
            logger.error(f"Failed to store cache entry: {e}")
    
    async def _update_cache_entry(self, entry: CacheEntry):
        """Update cache entry metadata."""
        if not self.redis_manager:
            return
        
        try:
            # Update exact match cache
            exact_key = f"llm_cache:exact:{entry.key}"
            await self.redis_manager.setex(
                exact_key,
                entry.ttl_seconds,
                json.dumps(entry.to_dict())
            )
            
            # Update semantic cache if exists
            if entry.embedding_hash:
                semantic_key = f"llm_cache:semantic:{entry.model}:{entry.embedding_hash}"
                await self.redis_manager.setex(
                    semantic_key,
                    entry.ttl_seconds,
                    json.dumps(entry.to_dict())
                )
        except Exception as e:
            logger.warning(f"Failed to update cache entry: {e}")
    
    async def get_cached_response(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int
    ) -> Optional[Tuple[CacheEntry, CacheType]]:
        """
        Get cached response with exact match first, then semantic similarity.
        
        Returns:
            Tuple of (CacheEntry, CacheType) if found, None otherwise
        """
        await self.initialize()
        
        self._cache_stats['requests'] += 1
        start_time = time.time()
        
        # Generate hashes
        prompt_hash = self._generate_prompt_hash(prompt, model, temperature, max_tokens)
        
        # Try exact match first
        exact_match = await self._find_exact_match(prompt_hash)
        if exact_match:
            self._cache_stats['hits'] += 1
            self._cache_stats['exact_hits'] += 1
            
            response_time = (time.time() - start_time) * 1000
            logger.debug(f"Exact cache hit: {prompt_hash[:16]}... ({response_time:.2f}ms)")
            
            return exact_match, CacheType.EXACT_MATCH
        
        # Try semantic similarity matching
        if self.settings.llm.enable_semantic_caching and self._embedding_model:
            query_embedding = await self._get_embedding(prompt)
            
            if query_embedding is not None:
                semantic_matches = await self._find_semantic_matches(query_embedding, prompt, model)
                
                if semantic_matches:
                    best_match, similarity = semantic_matches[0]
                    self._cache_stats['hits'] += 1
                    self._cache_stats['semantic_hits'] += 1
                    
                    response_time = (time.time() - start_time) * 1000
                    logger.debug(f"Semantic cache hit: similarity={similarity:.3f} ({response_time:.2f}ms)")
                    
                    # Update access metadata
                    best_match.refresh_access()
                    await self._update_cache_entry(best_match)
                    
                    return best_match, CacheType.SEMANTIC_MATCH
        
        # No cache hit
        self._cache_stats['misses'] += 1
        response_time = (time.time() - start_time) * 1000
        logger.debug(f"Cache miss for prompt hash: {prompt_hash[:16]}... ({response_time:.2f}ms)")
        
        return None
    
    async def cache_response(
        self,
        prompt: str,
        response_content: str,
        model: str,
        provider: str,
        temperature: float,
        max_tokens: int,
        cost: float,
        tokens: int,
        ttl_seconds: Optional[int] = None
    ) -> str:
        """
        Cache LLM response with both exact and semantic indexing.
        
        Returns:
            Cache key for the stored entry
        """
        await self.initialize()
        
        if ttl_seconds is None:
            ttl_seconds = self.default_ttl
        
        # Generate hashes
        prompt_hash = self._generate_prompt_hash(prompt, model, temperature, max_tokens)
        content_hash = self._generate_content_hash(response_content)
        
        # Get embedding for semantic caching
        embedding = None
        embedding_hash = None
        if self.settings.llm.enable_semantic_caching:
            embedding = await self._get_embedding(prompt)
            if embedding is not None:
                embedding_hash = hashlib.sha256(embedding.tobytes()).hexdigest()
        
        # Create cache entry
        entry = CacheEntry(
            key=prompt_hash,
            content=response_content,
            model=model,
            provider=provider,
            cost=cost,
            tokens=tokens,
            cache_type=CacheType.EXACT_MATCH,
            timestamp=datetime.now(),
            access_count=0,
            last_accessed=datetime.now(),
            ttl_seconds=ttl_seconds,
            embedding_hash=embedding_hash,
            content_hash=content_hash
        )
        
        # Store the cache entry
        await self._store_cache_entry(entry, prompt, prompt_hash, embedding)
        
        logger.debug(f"Cached response: {prompt_hash[:16]}... (ttl={ttl_seconds}s)")
        
        return prompt_hash
    
    async def warm_cache(self, common_prompts: List[Dict[str, Any]]):
        """
        Pre-warm cache with common prompts.
        
        Args:
            common_prompts: List of prompt configurations to pre-cache
        """
        await self.initialize()
        
        logger.info(f"Warming cache with {len(common_prompts)} common prompts")
        
        for prompt_config in common_prompts:
            try:
                # This would typically involve making actual LLM calls
                # For now, we'll just create placeholder entries
                prompt = prompt_config.get('prompt', '')
                model = prompt_config.get('model', self.settings.llm.default_summarization_model)
                
                # Check if already cached
                existing = await self.get_cached_response(
                    prompt, model, 0.1, 1000
                )
                
                if not existing:
                    # Would make actual LLM call here in production
                    logger.debug(f"Would warm cache for: {prompt[:50]}...")
                    
            except Exception as e:
                logger.warning(f"Failed to warm cache entry: {e}")
    
    async def cleanup_expired_entries(self):
        """Remove expired cache entries."""
        if not self.redis_manager:
            return
        
        try:
            # Get all cache keys
            index_key = "llm_cache:index"
            cache_keys = await self.redis_manager.smembers(index_key)
            
            expired_count = 0
            
            for cache_key in cache_keys:
                try:
                    cached_data = await self.redis_manager.get(cache_key)
                    
                    if cached_data:
                        entry_data = json.loads(cached_data)
                        entry = CacheEntry.from_dict(entry_data)
                        
                        if entry.is_expired():
                            await self.redis_manager.delete(cache_key)
                            await self.redis_manager.srem(index_key, cache_key)
                            expired_count += 1
                    else:
                        # Key doesn't exist, remove from index
                        await self.redis_manager.srem(index_key, cache_key)
                        expired_count += 1
                        
                except Exception as e:
                    logger.warning(f"Error cleaning up cache key {cache_key}: {e}")
            
            if expired_count > 0:
                logger.info(f"Cleaned up {expired_count} expired cache entries")
                
        except Exception as e:
            logger.error(f"Failed to cleanup expired entries: {e}")
    
    async def get_cache_stats(self) -> CacheStats:
        """Get comprehensive cache statistics."""
        await self.initialize()
        
        total_requests = self._cache_stats['requests']
        cache_hits = self._cache_stats['hits']
        cache_misses = self._cache_stats['misses']
        semantic_hits = self._cache_stats['semantic_hits']
        exact_hits = self._cache_stats['exact_hits']
        
        hit_rate = (cache_hits / total_requests) if total_requests > 0 else 0.0
        semantic_hit_rate = (semantic_hits / total_requests) if total_requests > 0 else 0.0
        
        # Get cache size information
        total_entries = 0
        cache_size_mb = 0.0
        model_distribution = {}
        
        if self.redis_manager:
            try:
                # Get cache index size
                index_key = "llm_cache:index"
                cache_keys = await self.redis_manager.smembers(index_key)
                total_entries = len(cache_keys)
                
                # Sample some entries for model distribution
                sample_size = min(100, len(cache_keys))
                for cache_key in list(cache_keys)[:sample_size]:
                    try:
                        cached_data = await self.redis_manager.get(cache_key)
                        if cached_data:
                            entry_data = json.loads(cached_data)
                            model = entry_data.get('model', 'unknown')
                            model_distribution[model] = model_distribution.get(model, 0) + 1
                            cache_size_mb += len(cached_data) / (1024 * 1024)
                    except Exception:
                        continue
                
                # Extrapolate cache size
                if sample_size > 0:
                    cache_size_mb = cache_size_mb * (total_entries / sample_size)
                    
            except Exception as e:
                logger.warning(f"Failed to get cache statistics: {e}")
        
        return CacheStats(
            total_requests=total_requests,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            semantic_hits=semantic_hits,
            exact_hits=exact_hits,
            hit_rate=hit_rate,
            semantic_hit_rate=semantic_hit_rate,
            avg_response_time_ms=50.0,  # Placeholder
            cost_savings=0.0,  # Would calculate based on cached costs
            total_cache_entries=total_entries,
            cache_size_mb=cache_size_mb,
            top_cached_models=model_distribution,
            cache_age_distribution={}  # Would implement age analysis
        )
    
    async def invalidate_cache(self, pattern: Optional[str] = None):
        """Invalidate cache entries matching pattern."""
        if not self.redis_manager:
            return
        
        try:
            if pattern:
                # Invalidate specific pattern
                keys_to_delete = await self.redis_manager.keys(pattern)
            else:
                # Invalidate all cache
                index_key = "llm_cache:index"
                keys_to_delete = await self.redis_manager.smembers(index_key)
                keys_to_delete.append(index_key)
            
            if keys_to_delete:
                await self.redis_manager.delete(*keys_to_delete)
                logger.info(f"Invalidated {len(keys_to_delete)} cache entries")
                
        except Exception as e:
            logger.error(f"Failed to invalidate cache: {e}")
    
    def reset_stats(self):
        """Reset cache statistics."""
        self._cache_stats = {
            'requests': 0,
            'hits': 0,
            'misses': 0,
            'semantic_hits': 0,
            'exact_hits': 0
        }


# Global cache manager instance
_cache_manager: Optional[SemanticCacheManager] = None


async def get_cache_manager() -> SemanticCacheManager:
    """Get or create the global cache manager instance."""
    global _cache_manager
    
    if _cache_manager is None:
        _cache_manager = SemanticCacheManager()
        await _cache_manager.initialize()
    
    return _cache_manager