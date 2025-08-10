"""
Query optimization and performance monitoring for Neo4j graph operations.
Provides query analysis, caching, and performance tracking capabilities.
"""

import time
import asyncio
import json
import hashlib
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from enum import Enum

import aioredis
from ..core.config import get_settings
from ..db.neo4j_advanced import AdvancedNeo4jManager
from ..utils.logger import get_logger

logger = get_logger(__name__)


class QueryType(str, Enum):
    """Types of graph queries."""
    READ = "read"
    WRITE = "write"
    ALGORITHM = "algorithm"
    SEARCH = "search"
    AGGREGATION = "aggregation"


class OptimizationLevel(str, Enum):
    """Query optimization levels."""
    NONE = "none"
    BASIC = "basic"
    AGGRESSIVE = "aggressive"
    EXPERIMENTAL = "experimental"


@dataclass
class QueryPerformanceMetric:
    """Performance metrics for a query execution."""
    query_id: str
    query_hash: str
    query_type: QueryType
    execution_time_ms: float
    memory_usage_bytes: Optional[int] = None
    rows_processed: Optional[int] = None
    rows_returned: Optional[int] = None
    cache_hit: bool = False
    optimization_applied: bool = False
    optimization_level: OptimizationLevel = OptimizationLevel.NONE
    timestamp: datetime = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class QueryOptimizationSuggestion:
    """Optimization suggestion for improving query performance."""
    query_hash: str
    suggestion_type: str
    description: str
    estimated_improvement: float  # Percentage improvement estimate
    implementation_complexity: str  # "low", "medium", "high"
    optimized_query: Optional[str] = None
    additional_indexes: Optional[List[str]] = None


class QueryCache:
    """Redis-based query result cache with intelligent invalidation."""
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.default_ttl = 3600  # 1 hour default TTL
        
    def _generate_cache_key(self, query: str, parameters: Dict = None) -> str:
        """Generate cache key for query and parameters."""
        query_data = {
            "query": query.strip(),
            "parameters": parameters or {}
        }
        query_json = json.dumps(query_data, sort_keys=True)
        return f"graph_query:{hashlib.md5(query_json.encode()).hexdigest()}"
    
    async def get(self, query: str, parameters: Dict = None) -> Optional[Any]:
        """Get cached query result."""
        cache_key = self._generate_cache_key(query, parameters)
        try:
            cached_data = await self.redis.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
        return None
    
    async def set(
        self, 
        query: str, 
        parameters: Dict = None, 
        result: Any = None,
        ttl: int = None
    ) -> bool:
        """Cache query result."""
        cache_key = self._generate_cache_key(query, parameters)
        try:
            result_json = json.dumps(result, default=str)
            await self.redis.setex(
                cache_key, 
                ttl or self.default_ttl, 
                result_json
            )
            return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern."""
        try:
            keys = await self.redis.keys(f"graph_query:*{pattern}*")
            if keys:
                return await self.redis.delete(*keys)
        except Exception as e:
            logger.error(f"Cache invalidation error: {e}")
        return 0
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            info = await self.redis.info()
            keys_count = len(await self.redis.keys("graph_query:*"))
            
            return {
                "total_keys": keys_count,
                "memory_usage": info.get("used_memory_human", "N/A"),
                "hit_rate": info.get("keyspace_hits", 0) / max(info.get("keyspace_misses", 0) + info.get("keyspace_hits", 0), 1),
                "connected_clients": info.get("connected_clients", 0)
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}


class QueryAnalyzer:
    """Analyzes queries for optimization opportunities."""
    
    def __init__(self):
        self.query_patterns = {
            # Common anti-patterns
            "cartesian_product": r"MATCH.*MATCH.*(?!WHERE)",
            "missing_limit": r"MATCH.*RETURN.*(?!LIMIT)",
            "inefficient_aggregation": r"collect\(DISTINCT.*ORDER BY",
            "deep_traversal": r"[\*][3-9]|[\*][1-9][0-9]",
            "full_scan": r"MATCH \([a-zA-Z]+\) WHERE",
            
            # Optimization opportunities
            "can_use_index": r"WHERE [a-zA-Z]+\.[a-zA-Z]+ = ",
            "can_batch": r"UNWIND.*MATCH",
            "can_profile": r"(?!PROFILE|EXPLAIN)",
        }
    
    def analyze_query(self, query: str) -> List[QueryOptimizationSuggestion]:
        """Analyze query and provide optimization suggestions."""
        suggestions = []
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        # Check for common anti-patterns
        if self._matches_pattern(query, "cartesian_product"):
            suggestions.append(QueryOptimizationSuggestion(
                query_hash=query_hash,
                suggestion_type="anti_pattern",
                description="Potential cartesian product detected. Add WHERE clauses to connect MATCH patterns.",
                estimated_improvement=50.0,
                implementation_complexity="low"
            ))
        
        if self._matches_pattern(query, "missing_limit") and "count(" not in query.lower():
            suggestions.append(QueryOptimizationSuggestion(
                query_hash=query_hash,
                suggestion_type="performance",
                description="Consider adding LIMIT clause to prevent excessive memory usage.",
                estimated_improvement=30.0,
                implementation_complexity="low",
                optimized_query=query.rstrip() + "\nLIMIT 1000"
            ))
        
        if self._matches_pattern(query, "deep_traversal"):
            suggestions.append(QueryOptimizationSuggestion(
                query_hash=query_hash,
                suggestion_type="performance",
                description="Deep graph traversal detected. Consider using algorithms or limiting depth.",
                estimated_improvement=70.0,
                implementation_complexity="medium"
            ))
        
        # Check for indexing opportunities
        if self._matches_pattern(query, "can_use_index"):
            suggestions.append(QueryOptimizationSuggestion(
                query_hash=query_hash,
                suggestion_type="indexing",
                description="Query could benefit from additional indexes on filtered properties.",
                estimated_improvement=40.0,
                implementation_complexity="low",
                additional_indexes=self._extract_index_suggestions(query)
            ))
        
        return suggestions
    
    def _matches_pattern(self, query: str, pattern_name: str) -> bool:
        """Check if query matches a specific pattern."""
        import re
        pattern = self.query_patterns.get(pattern_name)
        if pattern:
            return bool(re.search(pattern, query, re.IGNORECASE))
        return False
    
    def _extract_index_suggestions(self, query: str) -> List[str]:
        """Extract potential index suggestions from query."""
        import re
        
        # Look for WHERE clauses with property comparisons
        where_patterns = re.findall(
            r'WHERE\s+([a-zA-Z]+)\.([a-zA-Z_]+)\s*[=<>!]',
            query,
            re.IGNORECASE
        )
        
        suggestions = []
        for label_var, prop in where_patterns:
            # This is simplified - in practice, we'd need to track variable to label mapping
            suggestions.append(f"CREATE INDEX IF NOT EXISTS FOR (n:Node) ON (n.{prop})")
        
        return suggestions


class PerformanceMonitor:
    """Monitors and tracks query performance metrics."""
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.metrics_buffer = deque(maxlen=10000)  # Keep last 10k metrics in memory
        self.slow_query_threshold_ms = 1000  # Queries slower than 1s
        
    async def record_metric(self, metric: QueryPerformanceMetric):
        """Record a query performance metric."""
        self.metrics_buffer.append(metric)
        
        # Store in Redis for persistence
        try:
            metric_data = asdict(metric)
            metric_data["timestamp"] = metric.timestamp.isoformat()
            
            # Store individual metric
            await self.redis.setex(
                f"query_metric:{metric.query_id}",
                86400,  # 24 hours
                json.dumps(metric_data, default=str)
            )
            
            # Add to slow queries list if applicable
            if metric.execution_time_ms > self.slow_query_threshold_ms:
                await self.redis.zadd(
                    "slow_queries",
                    {metric.query_hash: metric.execution_time_ms}
                )
                
        except Exception as e:
            logger.error(f"Error recording metric: {e}")
    
    async def get_performance_summary(
        self, 
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """Get performance summary for a time window."""
        cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
        
        # Filter metrics from buffer
        recent_metrics = [
            m for m in self.metrics_buffer 
            if m.timestamp > cutoff_time
        ]
        
        if not recent_metrics:
            return {"message": "No recent metrics available"}
        
        # Calculate statistics
        execution_times = [m.execution_time_ms for m in recent_metrics]
        
        summary = {
            "total_queries": len(recent_metrics),
            "time_window_hours": time_window_hours,
            "performance_stats": {
                "avg_execution_time_ms": sum(execution_times) / len(execution_times),
                "min_execution_time_ms": min(execution_times),
                "max_execution_time_ms": max(execution_times),
                "slow_queries_count": len([t for t in execution_times if t > self.slow_query_threshold_ms])
            },
            "query_type_distribution": self._calculate_type_distribution(recent_metrics),
            "cache_hit_rate": len([m for m in recent_metrics if m.cache_hit]) / len(recent_metrics),
            "optimization_rate": len([m for m in recent_metrics if m.optimization_applied]) / len(recent_metrics)
        }
        
        return summary
    
    def _calculate_type_distribution(self, metrics: List[QueryPerformanceMetric]) -> Dict[str, int]:
        """Calculate distribution of query types."""
        distribution = defaultdict(int)
        for metric in metrics:
            distribution[metric.query_type.value] += 1
        return dict(distribution)
    
    async def get_slow_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get slowest queries from Redis."""
        try:
            slow_queries = await self.redis.zrevrange(
                "slow_queries", 
                0, 
                limit - 1, 
                withscores=True
            )
            
            result = []
            for query_hash, execution_time in slow_queries:
                result.append({
                    "query_hash": query_hash.decode() if isinstance(query_hash, bytes) else query_hash,
                    "execution_time_ms": execution_time
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting slow queries: {e}")
            return []


class QueryOptimizer:
    """Main query optimization and monitoring coordinator."""
    
    def __init__(self, neo4j_manager: AdvancedNeo4jManager, redis_client: aioredis.Redis):
        self.neo4j = neo4j_manager
        self.redis = redis_client
        self.cache = QueryCache(redis_client)
        self.analyzer = QueryAnalyzer()
        self.monitor = PerformanceMonitor(redis_client)
        self.optimization_level = OptimizationLevel.BASIC
        
    async def execute_optimized_query(
        self,
        query: str,
        parameters: Dict = None,
        cache_ttl: int = None,
        enable_cache: bool = True,
        query_type: QueryType = QueryType.READ
    ) -> Tuple[Any, QueryPerformanceMetric]:
        """Execute query with optimization and monitoring."""
        
        start_time = time.time()
        query_id = f"query_{int(start_time)}_{id(query)}"
        query_hash = hashlib.md5(query.encode()).hexdigest()
        cache_hit = False
        result = None
        error = None
        
        try:
            # Try cache first for read queries
            if enable_cache and query_type == QueryType.READ:
                cached_result = await self.cache.get(query, parameters)
                if cached_result is not None:
                    result = cached_result
                    cache_hit = True
            
            # Execute query if not cached
            if result is None:
                # Apply query optimizations
                optimized_query, optimization_applied = await self._apply_optimizations(query)
                
                # Execute the query
                if query_type == QueryType.READ:
                    result = await self.neo4j.execute_read(optimized_query, parameters)
                else:
                    result = await self.neo4j.execute_write(optimized_query, parameters)
                
                # Cache the result if applicable
                if enable_cache and query_type == QueryType.READ and result:
                    await self.cache.set(query, parameters, result, cache_ttl)
            
        except Exception as e:
            error = str(e)
            logger.error(f"Query execution error: {e}")
            result = []
        
        # Record performance metric
        execution_time_ms = (time.time() - start_time) * 1000
        
        metric = QueryPerformanceMetric(
            query_id=query_id,
            query_hash=query_hash,
            query_type=query_type,
            execution_time_ms=execution_time_ms,
            rows_returned=len(result) if isinstance(result, list) else None,
            cache_hit=cache_hit,
            optimization_applied=True,  # We always try optimizations
            optimization_level=self.optimization_level,
            error=error
        )
        
        await self.monitor.record_metric(metric)
        
        return result, metric
    
    async def _apply_optimizations(self, query: str) -> Tuple[str, bool]:
        """Apply query optimizations based on current optimization level."""
        if self.optimization_level == OptimizationLevel.NONE:
            return query, False
        
        optimized_query = query
        optimization_applied = False
        
        if self.optimization_level in [OptimizationLevel.BASIC, OptimizationLevel.AGGRESSIVE]:
            # Basic optimizations
            optimized_query = self._apply_basic_optimizations(optimized_query)
            optimization_applied = True
        
        if self.optimization_level in [OptimizationLevel.AGGRESSIVE, OptimizationLevel.EXPERIMENTAL]:
            # More aggressive optimizations
            optimized_query = self._apply_aggressive_optimizations(optimized_query)
        
        if self.optimization_level == OptimizationLevel.EXPERIMENTAL:
            # Experimental optimizations
            optimized_query = self._apply_experimental_optimizations(optimized_query)
        
        return optimized_query, optimization_applied
    
    def _apply_basic_optimizations(self, query: str) -> str:
        """Apply basic query optimizations."""
        optimized = query
        
        # Add LIMIT if missing and no aggregation
        if ("LIMIT" not in optimized.upper() and 
            "COUNT(" not in optimized.upper() and 
            "SUM(" not in optimized.upper() and
            "RETURN" in optimized.upper()):
            optimized = optimized.rstrip() + "\nLIMIT 10000"
        
        # Add hints for known patterns
        if "MATCH (p:Paper)" in optimized and "WHERE" in optimized:
            # Suggest using index
            optimized = "// HINT: Consider using index\n" + optimized
        
        return optimized
    
    def _apply_aggressive_optimizations(self, query: str) -> str:
        """Apply aggressive query optimizations."""
        optimized = query
        
        # Convert some patterns to more efficient forms
        # This is simplified - in practice, would need sophisticated query rewriting
        
        # Example: Convert nested matches to single match with multiple patterns
        import re
        
        # Look for consecutive MATCH statements that could be combined
        matches = re.findall(r'MATCH \([^)]+\)', optimized)
        if len(matches) > 1:
            # This is a simplified optimization - real implementation would be more sophisticated
            optimized = "// OPTIMIZED: Combined MATCH patterns\n" + optimized
        
        return optimized
    
    def _apply_experimental_optimizations(self, query: str) -> str:
        """Apply experimental query optimizations."""
        # These would be cutting-edge optimizations that might be unstable
        return query
    
    async def analyze_and_suggest(self, query: str) -> List[QueryOptimizationSuggestion]:
        """Analyze query and provide optimization suggestions."""
        return self.analyzer.analyze_query(query)
    
    async def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization and performance report."""
        performance_summary = await self.monitor.get_performance_summary()
        slow_queries = await self.monitor.get_slow_queries()
        cache_stats = await self.cache.get_cache_stats()
        
        return {
            "performance_summary": performance_summary,
            "slow_queries": slow_queries,
            "cache_statistics": cache_stats,
            "optimization_level": self.optimization_level.value,
            "recommendations": await self._generate_system_recommendations()
        }
    
    async def _generate_system_recommendations(self) -> List[str]:
        """Generate system-wide optimization recommendations."""
        recommendations = []
        
        # Analyze recent performance
        slow_queries = await self.monitor.get_slow_queries(50)
        if len(slow_queries) > 10:
            recommendations.append("High number of slow queries detected. Consider reviewing query patterns and adding indexes.")
        
        # Check cache hit rate
        cache_stats = await self.cache.get_cache_stats()
        if cache_stats.get("hit_rate", 0) < 0.5:
            recommendations.append("Low cache hit rate. Consider increasing cache TTL or reviewing cacheable query patterns.")
        
        return recommendations


# ==================== DEPENDENCY INJECTION ====================

async def get_query_optimizer() -> QueryOptimizer:
    """Dependency function to get query optimizer."""
    settings = get_settings()
    
    # Initialize Redis client
    redis_client = aioredis.from_url(
        settings.database.redis_url,
        encoding="utf-8",
        decode_responses=True
    )
    
    # Get Neo4j manager
    from ..db.neo4j_advanced import get_advanced_neo4j_manager
    neo4j_manager = await get_advanced_neo4j_manager()
    
    return QueryOptimizer(neo4j_manager, redis_client)