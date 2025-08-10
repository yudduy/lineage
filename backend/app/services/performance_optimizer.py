"""
Performance Optimization System for the Analytics Platform.

This module provides intelligent query planning, multi-tier caching,
background processing, and resource optimization for enterprise-grade
performance and scalability.
"""

import asyncio
import json
import time
import hashlib
import pickle
import heapq
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, OrderedDict
import psutil
import aioredis
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from ..core.config import get_settings
from ..db.redis import RedisManager, get_redis_manager
from ..utils.logger import get_logger

logger = get_logger(__name__)


class CacheTier(str, Enum):
    """Cache tier levels."""
    L1_MEMORY = "l1_memory"  # In-process memory cache
    L2_REDIS = "l2_redis"    # Redis distributed cache
    L3_DATABASE = "l3_database"  # Database query cache
    

class TaskPriority(int, Enum):
    """Task priority levels."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class QueryPlan:
    """Represents an optimized query execution plan."""
    plan_id: str
    original_query: str
    optimized_query: str
    execution_steps: List[Dict[str, Any]]
    estimated_cost: float
    estimated_time_ms: int
    cache_strategy: Dict[str, Any]
    parallelization_strategy: Dict[str, Any]
    resource_requirements: Dict[str, float]


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    key: str
    value: Any
    tier: CacheTier
    size_bytes: int
    created_at: float
    last_accessed: float
    access_count: int
    ttl: Optional[int]
    cost: float  # Computational cost to regenerate
    

@dataclass
class BackgroundTask:
    """Represents a background processing task."""
    task_id: str
    name: str
    priority: TaskPriority
    function: Callable
    args: tuple
    kwargs: dict
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    

@dataclass
class ResourceMetrics:
    """System resource metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_io_sent_mb: float
    network_io_recv_mb: float
    active_connections: int
    cache_size_mb: float
    queue_size: int


class LRUCache:
    """Thread-safe LRU cache implementation."""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 512):
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb * 1024 * 1024  # Convert to bytes
        self.cache: OrderedDict = OrderedDict()
        self.memory_usage = 0
        self._lock = asyncio.Lock()
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        async with self._lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                entry = self.cache[key]
                entry.last_accessed = time.time()
                entry.access_count += 1
                return entry.value
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None, cost: float = 1.0):
        """Set value in cache."""
        async with self._lock:
            # Calculate size
            size = len(pickle.dumps(value))
            
            # Check if we need to evict
            while (len(self.cache) >= self.max_size or 
                   self.memory_usage + size > self.max_memory_mb):
                if not self.cache:
                    break
                    
                # Evict least recently used
                evicted_key, evicted_entry = self.cache.popitem(last=False)
                self.memory_usage -= evicted_entry.size_bytes
                logger.debug(f"Evicted cache entry: {evicted_key}")
            
            # Add new entry
            entry = CacheEntry(
                key=key,
                value=value,
                tier=CacheTier.L1_MEMORY,
                size_bytes=size,
                created_at=time.time(),
                last_accessed=time.time(),
                access_count=0,
                ttl=ttl,
                cost=cost
            )
            
            self.cache[key] = entry
            self.memory_usage += size
            
    async def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        async with self._lock:
            if key in self.cache:
                entry = self.cache.pop(key)
                self.memory_usage -= entry.size_bytes
                return True
            return False
    
    async def clear(self):
        """Clear all cache entries."""
        async with self._lock:
            self.cache.clear()
            self.memory_usage = 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        async with self._lock:
            total_hits = sum(e.access_count for e in self.cache.values())
            return {
                'size': len(self.cache),
                'memory_usage_mb': self.memory_usage / (1024 * 1024),
                'total_hits': total_hits,
                'avg_access_count': total_hits / len(self.cache) if self.cache else 0
            }


class MultiTierCache:
    """Multi-tier caching system with intelligent tier management."""
    
    def __init__(
        self,
        l1_cache: Optional[LRUCache] = None,
        redis_manager: Optional[RedisManager] = None
    ):
        self.l1_cache = l1_cache or LRUCache()
        self.redis = redis_manager
        
        # Cache statistics
        self.stats = defaultdict(lambda: defaultdict(int))
        
        # Cache warming queue
        self._warm_queue: asyncio.Queue = asyncio.Queue()
        self._warm_task: Optional[asyncio.Task] = None
        
    async def initialize(self):
        """Initialize the cache system."""
        if not self.redis:
            self.redis = await get_redis_manager()
        
        # Start cache warming worker
        self._warm_task = asyncio.create_task(self._cache_warming_worker())
        
        logger.info("Multi-tier cache initialized")
    
    async def close(self):
        """Clean up resources."""
        if self._warm_task:
            self._warm_task.cancel()
        
        await self.l1_cache.clear()
    
    async def get(
        self,
        key: str,
        tier: Optional[CacheTier] = None
    ) -> Optional[Any]:
        """Get value from cache, checking tiers in order."""
        # Update stats
        self.stats[key]['requests'] += 1
        
        # Check L1 (memory) first
        if tier in [None, CacheTier.L1_MEMORY]:
            value = await self.l1_cache.get(key)
            if value is not None:
                self.stats[key]['l1_hits'] += 1
                return value
        
        # Check L2 (Redis)
        if self.redis and tier in [None, CacheTier.L2_REDIS]:
            try:
                redis_value = await self.redis.get(f"cache:{key}")
                if redis_value:
                    value = json.loads(redis_value)
                    self.stats[key]['l2_hits'] += 1
                    
                    # Promote to L1
                    await self.l1_cache.set(key, value, ttl=300)
                    
                    return value
            except Exception as e:
                logger.debug(f"Redis cache get error: {e}")
        
        # Check L3 (Database) - would be implemented with database-specific logic
        
        self.stats[key]['misses'] += 1
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tiers: Optional[List[CacheTier]] = None,
        cost: float = 1.0
    ):
        """Set value in specified cache tiers."""
        tiers = tiers or [CacheTier.L1_MEMORY, CacheTier.L2_REDIS]
        
        # Set in L1
        if CacheTier.L1_MEMORY in tiers:
            await self.l1_cache.set(key, value, ttl=ttl, cost=cost)
        
        # Set in L2
        if CacheTier.L2_REDIS in tiers and self.redis:
            try:
                await self.redis.set(
                    f"cache:{key}",
                    json.dumps(value, default=str),
                    ttl=ttl or 3600
                )
            except Exception as e:
                logger.debug(f"Redis cache set error: {e}")
    
    async def delete(self, key: str):
        """Delete from all cache tiers."""
        # Delete from L1
        await self.l1_cache.delete(key)
        
        # Delete from L2
        if self.redis:
            try:
                await self.redis.delete(f"cache:{key}")
            except Exception as e:
                logger.debug(f"Redis cache delete error: {e}")
    
    async def warm(self, key: str, generator: Callable, *args, **kwargs):
        """Queue cache warming for a key."""
        await self._warm_queue.put((key, generator, args, kwargs))
    
    async def _cache_warming_worker(self):
        """Background worker for cache warming."""
        while True:
            try:
                key, generator, args, kwargs = await self._warm_queue.get()
                
                # Generate value
                try:
                    value = await generator(*args, **kwargs)
                    
                    # Cache the value
                    await self.set(key, value, ttl=3600)
                    
                    logger.debug(f"Warmed cache for key: {key}")
                    
                except Exception as e:
                    logger.error(f"Cache warming failed for {key}: {e}")
                
                # Rate limit warming
                await asyncio.sleep(0.1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache warming worker error: {e}")
                await asyncio.sleep(1)
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        l1_stats = await self.l1_cache.get_stats()
        
        total_requests = sum(s['requests'] for s in self.stats.values())
        total_l1_hits = sum(s['l1_hits'] for s in self.stats.values())
        total_l2_hits = sum(s['l2_hits'] for s in self.stats.values())
        total_misses = sum(s['misses'] for s in self.stats.values())
        
        hit_rate = (total_l1_hits + total_l2_hits) / total_requests if total_requests > 0 else 0
        
        return {
            'l1_stats': l1_stats,
            'total_requests': total_requests,
            'l1_hits': total_l1_hits,
            'l2_hits': total_l2_hits,
            'misses': total_misses,
            'hit_rate': hit_rate,
            'warm_queue_size': self._warm_queue.qsize()
        }


class QueryOptimizer:
    """Intelligent query planning and optimization."""
    
    def __init__(self, cache: Optional[MultiTierCache] = None):
        self.cache = cache
        self.query_stats = defaultdict(lambda: {
            'count': 0,
            'total_time': 0,
            'avg_time': 0,
            'last_executed': None
        })
        
    async def optimize_query(
        self,
        query: str,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> QueryPlan:
        """Generate optimized query execution plan."""
        plan_id = hashlib.md5(f"{query}{params}".encode()).hexdigest()
        
        # Check if we have a cached plan
        if self.cache:
            cached_plan = await self.cache.get(f"query_plan:{plan_id}")
            if cached_plan:
                return QueryPlan(**cached_plan)
        
        # Analyze query
        analysis = self._analyze_query(query, params)
        
        # Generate execution steps
        steps = self._generate_execution_steps(analysis, context)
        
        # Determine cache strategy
        cache_strategy = self._determine_cache_strategy(analysis)
        
        # Determine parallelization
        parallel_strategy = self._determine_parallelization(analysis)
        
        # Estimate cost and time
        estimated_cost = self._estimate_cost(analysis, steps)
        estimated_time = self._estimate_time(analysis, steps)
        
        # Calculate resource requirements
        resources = self._calculate_resources(analysis, steps)
        
        plan = QueryPlan(
            plan_id=plan_id,
            original_query=query,
            optimized_query=self._optimize_query_text(query, analysis),
            execution_steps=steps,
            estimated_cost=estimated_cost,
            estimated_time_ms=estimated_time,
            cache_strategy=cache_strategy,
            parallelization_strategy=parallel_strategy,
            resource_requirements=resources
        )
        
        # Cache the plan
        if self.cache:
            await self.cache.set(
                f"query_plan:{plan_id}",
                plan.__dict__,
                ttl=3600
            )
        
        return plan
    
    def _analyze_query(self, query: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze query structure and characteristics."""
        analysis = {
            'type': 'unknown',
            'complexity': 'medium',
            'estimated_rows': 1000,
            'has_aggregation': False,
            'has_join': False,
            'has_subquery': False,
            'cacheable': True
        }
        
        query_lower = query.lower()
        
        # Determine query type
        if 'select' in query_lower:
            analysis['type'] = 'select'
        elif 'match' in query_lower:
            analysis['type'] = 'graph'
        elif 'insert' in query_lower or 'create' in query_lower:
            analysis['type'] = 'write'
            analysis['cacheable'] = False
        
        # Check for complexity indicators
        if 'join' in query_lower or 'match' in query_lower and '->' in query:
            analysis['has_join'] = True
            analysis['complexity'] = 'high'
        
        if any(agg in query_lower for agg in ['count', 'sum', 'avg', 'max', 'min']):
            analysis['has_aggregation'] = True
        
        if '(' in query and 'select' in query_lower and query_lower.count('select') > 1:
            analysis['has_subquery'] = True
            analysis['complexity'] = 'high'
        
        # Estimate rows based on filters
        if 'where' in query_lower or 'limit' in query_lower:
            if 'limit' in query_lower:
                # Extract limit value
                try:
                    limit_idx = query_lower.index('limit')
                    limit_str = query[limit_idx:].split()[1]
                    analysis['estimated_rows'] = min(int(limit_str), 10000)
                except:
                    pass
            else:
                analysis['estimated_rows'] = 100
        
        return analysis
    
    def _generate_execution_steps(
        self,
        analysis: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate execution steps for the query."""
        steps = []
        
        # Step 1: Check cache
        steps.append({
            'step': 1,
            'action': 'check_cache',
            'estimated_time_ms': 1
        })
        
        # Step 2: Query execution
        if analysis['type'] == 'graph':
            if analysis['has_join']:
                steps.append({
                    'step': 2,
                    'action': 'graph_traversal',
                    'parallel': True,
                    'estimated_time_ms': 50
                })
            steps.append({
                'step': 3,
                'action': 'graph_filtering',
                'estimated_time_ms': 20
            })
        else:
            steps.append({
                'step': 2,
                'action': 'execute_query',
                'estimated_time_ms': 30
            })
        
        # Step 3: Post-processing
        if analysis['has_aggregation']:
            steps.append({
                'step': len(steps) + 1,
                'action': 'aggregate_results',
                'estimated_time_ms': 10
            })
        
        # Step 4: Cache results
        if analysis['cacheable']:
            steps.append({
                'step': len(steps) + 1,
                'action': 'cache_results',
                'estimated_time_ms': 5
            })
        
        return steps
    
    def _determine_cache_strategy(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Determine optimal caching strategy."""
        strategy = {
            'enabled': analysis['cacheable'],
            'ttl': 3600,
            'tiers': [CacheTier.L1_MEMORY]
        }
        
        if not analysis['cacheable']:
            return strategy
        
        # Adjust TTL based on query type
        if analysis['type'] == 'select':
            if analysis['has_aggregation']:
                strategy['ttl'] = 1800  # 30 minutes for aggregations
            else:
                strategy['ttl'] = 600  # 10 minutes for simple selects
        
        # Use Redis for complex queries
        if analysis['complexity'] == 'high':
            strategy['tiers'].append(CacheTier.L2_REDIS)
        
        # Short TTL for frequently changing data
        if analysis['estimated_rows'] < 10:
            strategy['ttl'] = 300  # 5 minutes for small result sets
        
        return strategy
    
    def _determine_parallelization(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Determine parallelization strategy."""
        strategy = {
            'enabled': False,
            'max_workers': 1,
            'batch_size': 100
        }
        
        # Enable parallelization for complex queries
        if analysis['complexity'] == 'high':
            strategy['enabled'] = True
            strategy['max_workers'] = 4
            
            # Adjust batch size based on estimated rows
            if analysis['estimated_rows'] > 1000:
                strategy['batch_size'] = 500
            elif analysis['estimated_rows'] > 100:
                strategy['batch_size'] = 100
            else:
                strategy['batch_size'] = 10
        
        return strategy
    
    def _optimize_query_text(self, query: str, analysis: Dict[str, Any]) -> str:
        """Optimize the query text itself."""
        optimized = query
        
        # Add hints for graph queries
        if analysis['type'] == 'graph' and 'MATCH' in query:
            if 'USING INDEX' not in query:
                # Could add index hints here
                pass
        
        # Add LIMIT if missing for large result sets
        if (analysis['estimated_rows'] > 10000 and 
            'limit' not in query.lower()):
            optimized += ' LIMIT 10000'
        
        return optimized
    
    def _estimate_cost(
        self,
        analysis: Dict[str, Any],
        steps: List[Dict[str, Any]]
    ) -> float:
        """Estimate computational cost."""
        base_cost = 1.0
        
        # Adjust for complexity
        if analysis['complexity'] == 'high':
            base_cost *= 3
        elif analysis['complexity'] == 'medium':
            base_cost *= 1.5
        
        # Adjust for operations
        if analysis['has_join']:
            base_cost *= 2
        if analysis['has_aggregation']:
            base_cost *= 1.5
        if analysis['has_subquery']:
            base_cost *= 2.5
        
        # Adjust for data size
        base_cost *= (analysis['estimated_rows'] / 1000)
        
        return min(base_cost, 100.0)
    
    def _estimate_time(
        self,
        analysis: Dict[str, Any],
        steps: List[Dict[str, Any]]
    ) -> int:
        """Estimate execution time in milliseconds."""
        total_time = sum(s.get('estimated_time_ms', 0) for s in steps)
        
        # Adjust based on analysis
        if analysis['complexity'] == 'high':
            total_time *= 2
        
        # Add network latency
        total_time += 10
        
        return int(total_time)
    
    def _calculate_resources(
        self,
        analysis: Dict[str, Any],
        steps: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate resource requirements."""
        resources = {
            'cpu_cores': 1.0,
            'memory_mb': 64.0,
            'network_bandwidth_mbps': 1.0
        }
        
        # Adjust based on complexity
        if analysis['complexity'] == 'high':
            resources['cpu_cores'] = 2.0
            resources['memory_mb'] = 256.0
        
        # Adjust for parallelization
        for step in steps:
            if step.get('parallel'):
                resources['cpu_cores'] *= 2
        
        # Adjust for data size
        resources['memory_mb'] *= (analysis['estimated_rows'] / 1000)
        
        return resources
    
    async def record_execution(
        self,
        query: str,
        execution_time_ms: int,
        success: bool = True
    ):
        """Record query execution statistics."""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        stats = self.query_stats[query_hash]
        stats['count'] += 1
        stats['total_time'] += execution_time_ms
        stats['avg_time'] = stats['total_time'] / stats['count']
        stats['last_executed'] = time.time()
        
        # Learn from execution for future optimizations
        if stats['count'] > 10 and stats['avg_time'] > 1000:
            # Mark as candidate for optimization
            logger.info(f"Query candidate for optimization: {query_hash} (avg: {stats['avg_time']}ms)")


class BackgroundTaskProcessor:
    """Background task processing with priority queue."""
    
    def __init__(
        self,
        max_workers: int = 10,
        max_queue_size: int = 1000
    ):
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        
        # Priority queue (min heap)
        self._queue: List[Tuple[int, float, BackgroundTask]] = []
        self._queue_lock = asyncio.Lock()
        
        # Worker pool
        self._workers: List[asyncio.Task] = []
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Task tracking
        self._active_tasks: Dict[str, BackgroundTask] = {}
        self._completed_tasks: OrderedDict = OrderedDict()
        self._max_completed = 1000
        
        # Metrics
        self.metrics = {
            'tasks_queued': 0,
            'tasks_started': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'avg_wait_time': 0,
            'avg_execution_time': 0
        }
        
    async def initialize(self):
        """Initialize the task processor."""
        # Start worker tasks
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker_{i}"))
            self._workers.append(worker)
        
        logger.info(f"Background task processor initialized with {self.max_workers} workers")
    
    async def close(self):
        """Clean up resources."""
        # Cancel all workers
        for worker in self._workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self._workers, return_exceptions=True)
        
        # Shutdown executor
        self._executor.shutdown(wait=False)
    
    async def submit(
        self,
        name: str,
        function: Callable,
        args: tuple = (),
        kwargs: dict = None,
        priority: TaskPriority = TaskPriority.NORMAL
    ) -> str:
        """Submit a task for background processing."""
        kwargs = kwargs or {}
        
        # Check queue size
        async with self._queue_lock:
            if len(self._queue) >= self.max_queue_size:
                raise RuntimeError(f"Task queue full ({self.max_queue_size} tasks)")
        
        # Create task
        task = BackgroundTask(
            task_id=hashlib.md5(f"{name}{time.time()}".encode()).hexdigest(),
            name=name,
            priority=priority,
            function=function,
            args=args,
            kwargs=kwargs,
            created_at=time.time()
        )
        
        # Add to priority queue
        async with self._queue_lock:
            heapq.heappush(
                self._queue,
                (priority.value, task.created_at, task)
            )
            self.metrics['tasks_queued'] += 1
        
        logger.debug(f"Task {task.task_id} ({name}) queued with priority {priority.name}")
        
        return task.task_id
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a task."""
        # Check active tasks
        if task_id in self._active_tasks:
            task = self._active_tasks[task_id]
            return {
                'status': 'running',
                'task_id': task_id,
                'name': task.name,
                'started_at': task.started_at,
                'elapsed_time': time.time() - task.started_at if task.started_at else 0
            }
        
        # Check completed tasks
        if task_id in self._completed_tasks:
            task = self._completed_tasks[task_id]
            return {
                'status': 'completed' if task.error is None else 'failed',
                'task_id': task_id,
                'name': task.name,
                'completed_at': task.completed_at,
                'execution_time': task.completed_at - task.started_at if task.started_at else 0,
                'error': task.error
            }
        
        # Check queue
        async with self._queue_lock:
            for _, _, task in self._queue:
                if task.task_id == task_id:
                    return {
                        'status': 'queued',
                        'task_id': task_id,
                        'name': task.name,
                        'created_at': task.created_at,
                        'queue_position': self._get_queue_position(task_id)
                    }
        
        return None
    
    def _get_queue_position(self, task_id: str) -> int:
        """Get position of task in queue."""
        position = 0
        for _, _, task in sorted(self._queue):
            position += 1
            if task.task_id == task_id:
                return position
        return -1
    
    async def _worker(self, worker_id: str):
        """Worker coroutine for processing tasks."""
        logger.debug(f"Worker {worker_id} started")
        
        while True:
            try:
                # Get next task from priority queue
                task = None
                async with self._queue_lock:
                    if self._queue:
                        _, _, task = heapq.heappop(self._queue)
                
                if task is None:
                    # No tasks, wait a bit
                    await asyncio.sleep(0.1)
                    continue
                
                # Update metrics
                wait_time = time.time() - task.created_at
                self.metrics['avg_wait_time'] = (
                    self.metrics['avg_wait_time'] * self.metrics['tasks_started'] + wait_time
                ) / (self.metrics['tasks_started'] + 1)
                
                # Mark as active
                task.started_at = time.time()
                self._active_tasks[task.task_id] = task
                self.metrics['tasks_started'] += 1
                
                logger.debug(f"Worker {worker_id} processing task {task.task_id} ({task.name})")
                
                # Execute task
                try:
                    if asyncio.iscoroutinefunction(task.function):
                        # Async function
                        task.result = await task.function(*task.args, **task.kwargs)
                    else:
                        # Sync function - run in executor
                        loop = asyncio.get_event_loop()
                        task.result = await loop.run_in_executor(
                            self._executor,
                            task.function,
                            *task.args,
                            **task.kwargs
                        )
                    
                    task.completed_at = time.time()
                    self.metrics['tasks_completed'] += 1
                    
                    logger.debug(f"Task {task.task_id} completed successfully")
                    
                except Exception as e:
                    task.error = str(e)
                    task.completed_at = time.time()
                    self.metrics['tasks_failed'] += 1
                    
                    logger.error(f"Task {task.task_id} failed: {e}")
                    
                    # Retry logic
                    if task.retry_count < task.max_retries:
                        task.retry_count += 1
                        task.error = None
                        task.completed_at = None
                        
                        # Re-queue with lower priority
                        async with self._queue_lock:
                            heapq.heappush(
                                self._queue,
                                (task.priority.value + 1, time.time(), task)
                            )
                        
                        logger.debug(f"Task {task.task_id} queued for retry ({task.retry_count}/{task.max_retries})")
                
                finally:
                    # Move to completed
                    if task.task_id in self._active_tasks:
                        del self._active_tasks[task.task_id]
                    
                    if task.completed_at:
                        self._completed_tasks[task.task_id] = task
                        
                        # Limit completed tasks
                        if len(self._completed_tasks) > self._max_completed:
                            self._completed_tasks.popitem(last=False)
                        
                        # Update execution time metric
                        exec_time = task.completed_at - task.started_at
                        self.metrics['avg_execution_time'] = (
                            self.metrics['avg_execution_time'] * (self.metrics['tasks_completed'] - 1) + exec_time
                        ) / self.metrics['tasks_completed']
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(1)
        
        logger.debug(f"Worker {worker_id} stopped")
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get processor metrics."""
        async with self._queue_lock:
            queue_size = len(self._queue)
        
        return {
            **self.metrics,
            'queue_size': queue_size,
            'active_tasks': len(self._active_tasks),
            'completed_tasks': len(self._completed_tasks),
            'workers': self.max_workers
        }


class ResourceMonitor:
    """Monitor and optimize resource usage."""
    
    def __init__(self):
        self.metrics_history: deque = deque(maxlen=1000)
        self._monitor_task: Optional[asyncio.Task] = None
        self._monitor_interval = 10  # seconds
        
        # Thresholds
        self.cpu_threshold = 80.0
        self.memory_threshold = 80.0
        
        # Callbacks
        self._alert_callbacks: List[Callable] = []
        
    async def initialize(self):
        """Start resource monitoring."""
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Resource monitor initialized")
    
    async def close(self):
        """Stop monitoring."""
        if self._monitor_task:
            self._monitor_task.cancel()
    
    def add_alert_callback(self, callback: Callable):
        """Add callback for resource alerts."""
        self._alert_callbacks.append(callback)
    
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while True:
            try:
                metrics = await self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Check thresholds
                await self._check_thresholds(metrics)
                
                await asyncio.sleep(self._monitor_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                await asyncio.sleep(self._monitor_interval)
    
    async def _collect_metrics(self) -> ResourceMetrics:
        """Collect current resource metrics."""
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_read_mb = disk_io.read_bytes / (1024 * 1024) if disk_io else 0
        disk_write_mb = disk_io.write_bytes / (1024 * 1024) if disk_io else 0
        
        # Network I/O
        net_io = psutil.net_io_counters()
        net_sent_mb = net_io.bytes_sent / (1024 * 1024) if net_io else 0
        net_recv_mb = net_io.bytes_recv / (1024 * 1024) if net_io else 0
        
        # Connections
        connections = len(psutil.net_connections())
        
        return ResourceMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_mb=memory.used / (1024 * 1024),
            disk_io_read_mb=disk_read_mb,
            disk_io_write_mb=disk_write_mb,
            network_io_sent_mb=net_sent_mb,
            network_io_recv_mb=net_recv_mb,
            active_connections=connections,
            cache_size_mb=0,  # Would get from cache system
            queue_size=0  # Would get from task processor
        )
    
    async def _check_thresholds(self, metrics: ResourceMetrics):
        """Check if any thresholds are exceeded."""
        alerts = []
        
        if metrics.cpu_percent > self.cpu_threshold:
            alerts.append({
                'type': 'cpu',
                'value': metrics.cpu_percent,
                'threshold': self.cpu_threshold,
                'message': f"CPU usage high: {metrics.cpu_percent:.1f}%"
            })
        
        if metrics.memory_percent > self.memory_threshold:
            alerts.append({
                'type': 'memory',
                'value': metrics.memory_percent,
                'threshold': self.memory_threshold,
                'message': f"Memory usage high: {metrics.memory_percent:.1f}%"
            })
        
        # Trigger callbacks
        for alert in alerts:
            for callback in self._alert_callbacks:
                try:
                    await callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")
    
    async def get_current_metrics(self) -> Optional[ResourceMetrics]:
        """Get most recent metrics."""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
    
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of resource metrics."""
        if not self.metrics_history:
            return {}
        
        recent_metrics = list(self.metrics_history)[-100:]  # Last 100 samples
        
        return {
            'current': {
                'cpu_percent': recent_metrics[-1].cpu_percent,
                'memory_percent': recent_metrics[-1].memory_percent,
                'memory_mb': recent_metrics[-1].memory_mb,
                'connections': recent_metrics[-1].active_connections
            },
            'average': {
                'cpu_percent': sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics),
                'memory_percent': sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
            },
            'peak': {
                'cpu_percent': max(m.cpu_percent for m in recent_metrics),
                'memory_percent': max(m.memory_percent for m in recent_metrics)
            }
        }


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self):
        self.cache = MultiTierCache()
        self.query_optimizer = QueryOptimizer(cache=self.cache)
        self.task_processor = BackgroundTaskProcessor()
        self.resource_monitor = ResourceMonitor()
        
        # Auto-scaling parameters
        self.auto_scale_enabled = True
        self.min_workers = 2
        self.max_workers = 20
        
    async def initialize(self):
        """Initialize all components."""
        await self.cache.initialize()
        await self.task_processor.initialize()
        await self.resource_monitor.initialize()
        
        # Set up resource alerts
        self.resource_monitor.add_alert_callback(self._handle_resource_alert)
        
        logger.info("Performance Optimizer initialized")
    
    async def close(self):
        """Clean up all components."""
        await self.cache.close()
        await self.task_processor.close()
        await self.resource_monitor.close()
    
    async def _handle_resource_alert(self, alert: Dict[str, Any]):
        """Handle resource usage alerts."""
        logger.warning(f"Resource alert: {alert['message']}")
        
        if self.auto_scale_enabled:
            # Auto-scale based on resource usage
            if alert['type'] == 'cpu' and alert['value'] > 90:
                # Scale down workers if CPU is too high
                current_workers = self.task_processor.max_workers
                if current_workers > self.min_workers:
                    logger.info(f"Scaling down workers from {current_workers} to {current_workers - 1}")
                    # Would implement actual scaling here
            
            elif alert['type'] == 'memory' and alert['value'] > 85:
                # Clear caches if memory is high
                await self.cache.l1_cache.clear()
                logger.info("Cleared L1 cache due to high memory usage")
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        cache_stats = await self.cache.get_statistics()
        task_metrics = await self.task_processor.get_metrics()
        resource_summary = await self.resource_monitor.get_metrics_summary()
        
        return {
            'cache': cache_stats,
            'tasks': task_metrics,
            'resources': resource_summary,
            'optimization': {
                'auto_scale_enabled': self.auto_scale_enabled,
                'current_workers': self.task_processor.max_workers
            }
        }


# Singleton instance management
_performance_optimizer: Optional[PerformanceOptimizer] = None


async def get_performance_optimizer() -> PerformanceOptimizer:
    """Get or create the performance optimizer singleton."""
    global _performance_optimizer
    
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer()
        await _performance_optimizer.initialize()
    
    return _performance_optimizer