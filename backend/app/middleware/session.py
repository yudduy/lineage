"""
Session management middleware.
"""

from typing import Dict, Optional, Any
from datetime import datetime, timedelta
from fastapi import Request, Depends
import redis.asyncio as redis
import json

from ..core.config import get_settings
from ..core.security import get_security_manager


class SessionManager:
    """Session management using Redis backend."""
    
    def __init__(self):
        self.settings = get_settings()
        self.security_manager = get_security_manager()
        self.redis_client: Optional[redis.Redis] = None
        
    async def setup_redis(self):
        """Setup Redis connection."""
        if not self.redis_client:
            self.redis_client = redis.from_url(
                self.settings.database.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
    
    async def close_redis(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
    
    async def create_session(
        self,
        user_id: str,
        data: Optional[Dict[str, Any]] = None,
        expires_in_minutes: Optional[int] = None
    ) -> str:
        """Create new session."""
        await self.setup_redis()
        
        session_id = self.security_manager.generate_session_id()
        session_data = {
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),
            **(data or {})
        }
        
        session_key = f"session:{session_id}"
        await self.redis_client.hset(session_key, mapping=session_data)
        
        # Set expiration
        expire_minutes = expires_in_minutes or self.settings.security.session_expire_minutes
        await self.redis_client.expire(session_key, expire_minutes * 60)
        
        return session_id
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data."""
        await self.setup_redis()
        
        session_key = f"session:{session_id}"
        session_data = await self.redis_client.hgetall(session_key)
        
        if not session_data:
            return None
        
        # Update last activity
        await self.redis_client.hset(
            session_key,
            "last_activity",
            datetime.utcnow().isoformat()
        )
        
        return session_data
    
    async def update_session(
        self,
        session_id: str,
        data: Dict[str, Any]
    ) -> bool:
        """Update session data."""
        await self.setup_redis()
        
        session_key = f"session:{session_id}"
        
        # Check if session exists
        if not await self.redis_client.exists(session_key):
            return False
        
        # Update data
        data["last_activity"] = datetime.utcnow().isoformat()
        await self.redis_client.hset(session_key, mapping=data)
        
        return True
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete session."""
        await self.setup_redis()
        
        session_key = f"session:{session_id}"
        result = await self.redis_client.delete(session_key)
        
        return result > 0
    
    async def extend_session(
        self,
        session_id: str,
        additional_minutes: int = None
    ) -> bool:
        """Extend session expiration."""
        await self.setup_redis()
        
        session_key = f"session:{session_id}"
        
        if not await self.redis_client.exists(session_key):
            return False
        
        extend_by = additional_minutes or self.settings.security.session_expire_minutes
        await self.redis_client.expire(session_key, extend_by * 60)
        
        return True
    
    async def get_all_user_sessions(self, user_id: str) -> list[Dict[str, Any]]:
        """Get all sessions for a user."""
        await self.setup_redis()
        
        # Scan for all session keys
        sessions = []
        async for key in self.redis_client.scan_iter(match="session:*"):
            session_data = await self.redis_client.hgetall(key)
            if session_data.get("user_id") == user_id:
                session_data["session_id"] = key.replace("session:", "")
                sessions.append(session_data)
        
        return sessions
    
    async def invalidate_all_user_sessions(self, user_id: str) -> int:
        """Invalidate all sessions for a user."""
        await self.setup_redis()
        
        sessions = await self.get_all_user_sessions(user_id)
        deleted_count = 0
        
        for session in sessions:
            session_key = f"session:{session['session_id']}"
            if await self.redis_client.delete(session_key):
                deleted_count += 1
        
        return deleted_count


class SessionMiddleware:
    """Middleware for session management."""
    
    def __init__(self):
        self.session_manager = SessionManager()
    
    async def __call__(self, request: Request, call_next):
        """Process request with session handling."""
        # Get session ID from cookie or header
        session_id = None
        
        # Check for session cookie
        if "session_id" in request.cookies:
            session_id = request.cookies["session_id"]
        
        # Check for session header (for API clients)
        elif "X-Session-ID" in request.headers:
            session_id = request.headers["X-Session-ID"]
        
        # Get session data if session ID exists
        session_data = None
        if session_id:
            session_data = await self.session_manager.get_session(session_id)
        
        # Add session to request state
        request.state.session_id = session_id
        request.state.session_data = session_data
        
        response = await call_next(request)
        return response


# Global session manager instance
session_manager = SessionManager()


async def get_session_manager() -> SessionManager:
    """Dependency to get session manager."""
    return session_manager


async def get_session(request: Request) -> Optional[Dict[str, Any]]:
    """Dependency to get current session data."""
    return getattr(request.state, "session_data", None)


async def get_session_id(request: Request) -> Optional[str]:
    """Dependency to get current session ID."""
    return getattr(request.state, "session_id", None)


async def require_session(request: Request) -> Dict[str, Any]:
    """Dependency that requires a valid session."""
    from fastapi import HTTPException, status
    
    session_data = await get_session(request)
    
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Valid session required"
        )
    
    return session_data