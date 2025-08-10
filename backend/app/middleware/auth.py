"""
Authentication middleware and dependencies.
"""

from typing import Optional
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import redis.asyncio as redis

from ..core.config import get_settings
from ..core.security import get_security_manager, SecurityManager
from ..models.user import User, UserSession


class AuthenticationMiddleware:
    """Authentication middleware for validating JWT tokens."""
    
    def __init__(self):
        self.settings = get_settings()
        self.security_manager = get_security_manager()
        self.redis_client: Optional[redis.Redis] = None
        
    async def setup_redis(self):
        """Setup Redis connection for session management."""
        self.redis_client = redis.from_url(
            self.settings.database.redis_url,
            encoding="utf-8",
            decode_responses=True
        )
    
    async def close_redis(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
    
    async def get_user_from_token(self, token: str) -> Optional[User]:
        """Get user from JWT token."""
        payload = self.security_manager.verify_token(token)
        if not payload:
            return None
        
        user_id = payload.get("sub")
        if not user_id:
            return None
        
        # In a real implementation, you would fetch user from database
        # For now, we'll create a mock user based on token data
        return User(
            id=user_id,
            email=payload.get("email", ""),
            full_name=payload.get("full_name"),
            is_active=True,
            is_verified=True
        )
    
    async def get_session(self, session_id: str) -> Optional[UserSession]:
        """Get session from Redis."""
        if not self.redis_client:
            await self.setup_redis()
        
        try:
            session_data = await self.redis_client.hgetall(f"session:{session_id}")
            if not session_data:
                return None
            
            return UserSession(**session_data)
        except Exception:
            return None
    
    async def create_session(self, user: User, request: Request) -> UserSession:
        """Create new user session."""
        if not self.redis_client:
            await self.setup_redis()
        
        session = UserSession(
            user_id=user.id,
            session_id=self.security_manager.generate_session_id(),
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(
                minutes=self.settings.security.session_expire_minutes
            ),
            ip_address=request.client.host,
            user_agent=request.headers.get("user-agent")
        )
        
        # Store session in Redis
        session_key = f"session:{session.session_id}"
        await self.redis_client.hset(
            session_key,
            mapping=session.dict()
        )
        await self.redis_client.expire(
            session_key,
            self.settings.security.session_expire_minutes * 60
        )
        
        return session
    
    async def update_session_activity(self, session_id: str):
        """Update session last activity timestamp."""
        if not self.redis_client:
            await self.setup_redis()
        
        session_key = f"session:{session_id}"
        await self.redis_client.hset(
            session_key,
            "last_activity",
            datetime.utcnow().isoformat()
        )
    
    async def invalidate_session(self, session_id: str):
        """Invalidate user session."""
        if not self.redis_client:
            await self.setup_redis()
        
        await self.redis_client.delete(f"session:{session_id}")


# Global authentication middleware instance
auth_middleware = AuthenticationMiddleware()

# HTTP Bearer token scheme
security_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
    auth: AuthenticationMiddleware = Depends(lambda: auth_middleware)
) -> Optional[User]:
    """
    Dependency to get current authenticated user from JWT token.
    Returns None if no valid token is provided (for optional authentication).
    """
    if not credentials:
        return None
    
    user = await auth.get_user_from_token(credentials.credentials)
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Dependency to get current authenticated and active user.
    Raises HTTP 401 if user is not authenticated or not active.
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    
    return current_user


async def get_current_verified_user(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """
    Dependency to get current authenticated, active, and verified user.
    Raises HTTP 403 if user is not verified.
    """
    if not current_user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email not verified"
        )
    
    return current_user


async def get_current_premium_user(
    current_user: User = Depends(get_current_verified_user)
) -> User:
    """
    Dependency to get current premium user.
    Raises HTTP 403 if user is not premium.
    """
    if not current_user.is_premium:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Premium subscription required"
        )
    
    return current_user


async def require_scope(required_scope: str):
    """
    Dependency factory for scope-based authorization.
    """
    async def _require_scope(
        current_user: User = Depends(get_current_active_user)
    ) -> User:
        # In a real implementation, you would check user scopes/permissions
        # For now, we'll just return the user if they're authenticated
        return current_user
    
    return _require_scope