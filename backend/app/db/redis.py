"""
Redis cache and session management.
"""

import json
import pickle
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta

import redis.asyncio as redis

from ..core.config import get_settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


class RedisManager:
    """Redis connection and cache management."""
    
    def __init__(self):
        self.settings = get_settings()
        self.client: Optional[redis.Redis] = None
        
    async def connect(self):
        """Establish connection to Redis if configured."""
        if self.client is None:
            # Check if Redis is configured
            if not self.settings.redis_url:
                logger.info("Redis URL not configured, skipping Redis connection")
                return False
                
            try:
                self.client = redis.from_url(
                    self.settings.redis_url,
                    encoding="utf-8",
                    decode_responses=True
                )
                
                # Test connection
                await self.client.ping()
                logger.info("Successfully connected to Redis")
                return True
                
            except redis.ConnectionError as e:
                logger.warning(f"Failed to connect to Redis, will continue without caching: {e}")
                self.client = None
                return False
            except Exception as e:
                logger.warning(f"Redis connection error, will continue without caching: {e}")
                self.client = None
                return False
    
    async def disconnect(self):
        """Close Redis connection."""
        if self.client:
            await self.client.close()
            self.client = None
            logger.info("Disconnected from Redis")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Redis connection health."""
        if not self.client:
            return {
                "status": "unhealthy",
                "error": "No Redis connection"
            }
        
        try:
            start_time = datetime.utcnow()
            
            # Test basic operations
            test_key = "health_check_test"
            await self.client.set(test_key, "test_value", ex=5)
            value = await self.client.get(test_key)
            await self.client.delete(test_key)
            
            end_time = datetime.utcnow()
            response_time = (end_time - start_time).total_seconds() * 1000
            
            # Get Redis info
            info = await self.client.info()
            
            return {
                "status": "healthy",
                "response_time_ms": round(response_time, 2),
                "test_result": value == "test_value",
                "used_memory": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "uptime_seconds": info.get("uptime_in_seconds", 0)
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    # Basic Redis operations
    async def get(self, key: str) -> Optional[str]:
        """Get value by key."""
        if not self.client:
            await self.connect()
        return await self.client.get(key)
    
    async def set(
        self,
        key: str,
        value: str,
        expire: Optional[int] = None
    ) -> bool:
        """Set key-value pair with optional expiration."""
        if not self.client:
            await self.connect()
        return await self.client.set(key, value, ex=expire)
    
    async def delete(self, *keys: str) -> int:
        """Delete one or more keys."""
        if not self.client:
            await self.connect()
        return await self.client.delete(*keys)
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        if not self.client:
            await self.connect()
        return await self.client.exists(key) > 0
    
    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiration on key."""
        if not self.client:
            await self.connect()
        return await self.client.expire(key, seconds)
    
    # Hash operations
    async def hget(self, name: str, key: str) -> Optional[str]:
        """Get hash field value."""
        if not self.client:
            await self.connect()
        return await self.client.hget(name, key)
    
    async def hgetall(self, name: str) -> Dict[str, str]:
        """Get all hash fields."""
        if not self.client:
            await self.connect()
        return await self.client.hgetall(name)
    
    async def hset(self, name: str, mapping: Dict[str, Any]) -> int:
        """Set hash fields."""
        if not self.client:
            await self.connect()
        return await self.client.hset(name, mapping=mapping)
    
    async def hdel(self, name: str, *keys: str) -> int:
        """Delete hash fields."""
        if not self.client:
            await self.connect()
        return await self.client.hdel(name, *keys)
    
    # List operations
    async def lpush(self, name: str, *values: str) -> int:
        """Push values to left of list."""
        if not self.client:
            await self.connect()
        return await self.client.lpush(name, *values)
    
    async def rpush(self, name: str, *values: str) -> int:
        """Push values to right of list."""
        if not self.client:
            await self.connect()
        return await self.client.rpush(name, *values)
    
    async def lrange(self, name: str, start: int, end: int) -> List[str]:
        """Get list range."""
        if not self.client:
            await self.connect()
        return await self.client.lrange(name, start, end)
    
    async def llen(self, name: str) -> int:
        """Get list length."""
        if not self.client:
            await self.connect()
        return await self.client.llen(name)
    
    # Cache-specific operations
    async def cache_get(self, key: str) -> Optional[Any]:
        """Get cached value with JSON deserialization."""
        value = await self.get(key)
        if value is None:
            return None
        
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            # Return raw value if not JSON
            return value
    
    async def cache_set(
        self,
        key: str,
        value: Any,
        expire: Optional[int] = 3600  # 1 hour default
    ) -> bool:
        """Set cached value with JSON serialization."""
        try:
            if isinstance(value, (dict, list, tuple)):
                serialized = json.dumps(value)
            elif isinstance(value, str):
                serialized = value
            else:
                serialized = str(value)
            
            return await self.set(key, serialized, expire)
        except (TypeError, ValueError) as e:
            logger.error(f"Error caching value for key {key}: {e}")
            return False
    
    async def cache_delete(self, *keys: str) -> int:
        """Delete cached values."""
        return await self.delete(*keys)
    
    # Session management
    async def create_session(
        self,
        session_id: str,
        session_data: Dict[str, Any],
        expire_minutes: int = 60
    ) -> bool:
        """Create user session."""
        session_key = f"session:{session_id}"
        
        # Add timestamps
        session_data.update({
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat()
        })
        
        success = await self.hset(session_key, session_data)
        if success:
            await self.expire(session_key, expire_minutes * 60)
        
        return success > 0
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data."""
        session_key = f"session:{session_id}"
        session_data = await self.hgetall(session_key)
        
        if not session_data:
            return None
        
        # Update last activity
        await self.hset(session_key, {"last_activity": datetime.utcnow().isoformat()})
        
        return dict(session_data)
    
    async def update_session(
        self,
        session_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update session data."""
        session_key = f"session:{session_id}"
        
        if not await self.exists(session_key):
            return False
        
        updates["last_activity"] = datetime.utcnow().isoformat()
        success = await self.hset(session_key, updates)
        
        return success > 0
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete session."""
        session_key = f"session:{session_id}"
        return await self.delete(session_key) > 0
    
    # Rate limiting support
    async def incr_rate_limit(
        self,
        key: str,
        expire: Optional[int] = None
    ) -> int:
        """Increment rate limit counter."""
        if not self.client:
            await self.connect()
        
        # Increment counter
        count = await self.client.incr(key)
        
        # Set expiration if this is the first increment
        if count == 1 and expire:
            await self.client.expire(key, expire)
        
        return count
    
    async def get_rate_limit(self, key: str) -> int:
        """Get current rate limit count."""
        if not self.client:
            await self.connect()
        
        value = await self.client.get(key)
        return int(value) if value else 0
    
    # Background task queue support
    async def enqueue_task(
        self,
        queue_name: str,
        task_data: Dict[str, Any]
    ) -> bool:
        """Enqueue background task."""
        task_json = json.dumps({
            **task_data,
            "enqueued_at": datetime.utcnow().isoformat()
        })
        
        result = await self.lpush(f"queue:{queue_name}", task_json)
        return result > 0
    
    async def dequeue_task(self, queue_name: str) -> Optional[Dict[str, Any]]:
        """Dequeue background task."""
        if not self.client:
            await self.connect()
        
        result = await self.client.brpop(f"queue:{queue_name}", timeout=1)
        if not result:
            return None
        
        _, task_json = result
        try:
            return json.loads(task_json)
        except json.JSONDecodeError:
            logger.error(f"Invalid task JSON in queue {queue_name}: {task_json}")
            return None
    
    async def get_queue_length(self, queue_name: str) -> int:
        """Get queue length."""
        return await self.llen(f"queue:{queue_name}")
    
    # Metrics and monitoring
    async def get_stats(self) -> Dict[str, Any]:
        """Get Redis statistics."""
        if not self.client:
            await self.connect()
        
        info = await self.client.info()
        
        return {
            "connected_clients": info.get("connected_clients", 0),
            "used_memory": info.get("used_memory", 0),
            "used_memory_human": info.get("used_memory_human", "0B"),
            "total_commands_processed": info.get("total_commands_processed", 0),
            "keyspace_hits": info.get("keyspace_hits", 0),
            "keyspace_misses": info.get("keyspace_misses", 0),
            "uptime_seconds": info.get("uptime_in_seconds", 0),
            "redis_version": info.get("redis_version", "unknown")
        }


# Global Redis manager instance
redis_manager = RedisManager()


async def get_redis_manager() -> RedisManager:
    """Dependency function to get Redis manager."""
    if not redis_manager.client:
        await redis_manager.connect()
    return redis_manager