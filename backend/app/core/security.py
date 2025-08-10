"""
Security utilities for authentication and authorization.
"""

import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union

import bcrypt
from jose import JWTError, jwt
from jose.exceptions import ExpiredSignatureError
from passlib.context import CryptContext

from .config import get_settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


class SecurityManager:
    """Centralized security management."""
    
    def __init__(self):
        self.settings = get_settings()
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
    def create_access_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT access token with secure defaults."""
        to_encode = data.copy()
        now = datetime.utcnow()
        
        if expires_delta:
            expire = now + expires_delta
        else:
            expire = now + timedelta(
                minutes=self.settings.security.access_token_expire_minutes
            )
        
        # Add standard JWT claims
        to_encode.update({
            "exp": expire.timestamp() if hasattr(expire, 'timestamp') else expire,
            "iat": now.timestamp() if hasattr(now, 'timestamp') else now,
            "type": "access"
        })
        
        # Ensure required claims are present
        if "sub" not in to_encode:
            raise ValueError("Token must include 'sub' (subject) claim")
        
        encoded_jwt = jwt.encode(
            to_encode,
            self.settings.security.secret_key,
            algorithm=self.settings.security.algorithm
        )
        
        return encoded_jwt
    
    def create_refresh_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT refresh token with secure defaults."""
        to_encode = data.copy()
        now = datetime.utcnow()
        
        if expires_delta:
            expire = now + expires_delta
        else:
            expire = now + timedelta(
                days=self.settings.security.refresh_token_expire_days
            )
        
        # Add standard JWT claims
        to_encode.update({
            "exp": expire.timestamp() if hasattr(expire, 'timestamp') else expire,
            "iat": now.timestamp() if hasattr(now, 'timestamp') else now,
            "type": "refresh"
        })
        
        # Ensure required claims are present
        if "sub" not in to_encode:
            raise ValueError("Token must include 'sub' (subject) claim")
        
        encoded_jwt = jwt.encode(
            to_encode,
            self.settings.security.secret_key,
            algorithm=self.settings.security.algorithm
        )
        
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token with comprehensive validation."""
        try:
            # Decode token with proper validation
            payload = jwt.decode(
                token,
                self.settings.security.secret_key,
                algorithms=[self.settings.security.algorithm],
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_aud": False,
                    "verify_iss": False,
                    "require_exp": True,
                    "require_iat": True
                }
            )
            
            # Validate required claims
            if not payload.get("sub"):
                logger.warning("Token missing subject claim")
                return None
            
            # Validate token type for access tokens
            token_type = payload.get("type", "access")
            if token_type not in ["access", "refresh"]:
                logger.warning(f"Invalid token type: {token_type}")
                return None
            
            # Additional expiration check (jose library already checks this, but double-check for security)
            exp = payload.get("exp")
            if exp and datetime.utcnow().timestamp() >= exp:
                logger.warning("Token has expired")
                return None
            
            # Check issued at time is not in the future
            iat = payload.get("iat")
            if iat and datetime.utcnow().timestamp() < iat:
                logger.warning("Token issued in the future")
                return None
            
            return payload
            
        except ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except JWTError as e:
            logger.warning(f"Invalid token: {e}")
            return None
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            return None
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def generate_session_id(self) -> str:
        """Generate secure session ID."""
        return secrets.token_urlsafe(32)
    
    def generate_api_key(self) -> str:
        """Generate API key."""
        return secrets.token_urlsafe(64)
    
    def generate_reset_token(self) -> str:
        """Generate password reset token."""
        return secrets.token_urlsafe(32)


# Global security manager instance
security_manager = SecurityManager()


def get_security_manager() -> SecurityManager:
    """Dependency function to get security manager."""
    return security_manager