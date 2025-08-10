"""
Authentication endpoints.
"""

from datetime import datetime, timedelta
from typing import Any, Dict
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from ....models.user import (
    User,
    UserCreate,
    UserResponse,
    TokenResponse,
    LoginRequest,
    PasswordChangeRequest,
    PasswordResetRequest,
    PasswordResetConfirm
)
from ....models.common import APIResponse
from ....core.config import get_settings
from ....core.security import get_security_manager, SecurityManager
from ....middleware.auth import get_current_active_user
from ....middleware.rate_limit import api_rate_limit
from ....utils.exceptions import AuthenticationError, ValidationError, NotFoundError
from ....utils.logger import get_security_logger

security_logger = get_security_logger()
router = APIRouter()


@router.post("/register", response_model=APIResponse[UserResponse])
@api_rate_limit
async def register_user(
    user_create: UserCreate,
    security_manager: SecurityManager = Depends(get_security_manager)
):
    """
    Register a new user account.
    
    Creates a new user with the provided information.
    Requires email verification before account activation.
    """
    security_logger.log_authentication_attempt(
        email=user_create.email,
        success=False  # Will be updated on success
    )
    
    # TODO: Check if user already exists
    # In a real implementation, you'd check against your user database
    
    # Hash the password
    hashed_password = security_manager.hash_password(user_create.password)
    
    # Create user object
    user = User(
        id=security_manager.generate_session_id()[:8],  # Simple ID for demo
        email=user_create.email,
        username=user_create.username,
        full_name=user_create.full_name,
        affiliation=user_create.affiliation,
        is_active=True,  # In production, might require email verification
        is_verified=False,  # Requires email verification
        is_premium=False
    )
    
    # TODO: Save user to database
    # await db_service.create_user(user, hashed_password)
    
    # TODO: Send verification email
    # await email_service.send_verification_email(user)
    
    security_logger.log_authentication_attempt(
        email=user_create.email,
        success=True
    )
    
    return APIResponse(
        success=True,
        message="User registered successfully. Please check your email for verification.",
        data=UserResponse(
            id=user.id,
            email=user.email,
            username=user.username,
            full_name=user.full_name,
            affiliation=user.affiliation,
            is_active=user.is_active,
            is_verified=user.is_verified,
            is_premium=user.is_premium,
            preferences=user.preferences,
            stats=user.stats,
            created_at=user.created_at or user.updated_at
        )
    )


@router.post("/login", response_model=APIResponse[TokenResponse])
@api_rate_limit
async def login_user(
    login_request: LoginRequest,
    security_manager: SecurityManager = Depends(get_security_manager)
):
    """
    Authenticate user and return JWT tokens.
    
    Returns access and refresh tokens for API authentication.
    """
    security_logger.log_authentication_attempt(
        email=login_request.email,
        success=False
    )
    
    # TODO: Get user from database and verify password
    # user = await db_service.get_user_by_email(login_request.email)
    # if not user or not security_manager.verify_password(login_request.password, user.password_hash):
    #     raise AuthenticationError("Invalid email or password")
    
    # Mock user for demonstration
    if login_request.email == "demo@example.com" and login_request.password == "password123":
        user = User(
            id="demo_user_123",
            email=login_request.email,
            full_name="Demo User",
            is_active=True,
            is_verified=True
        )
    else:
        security_logger.log_authentication_attempt(
            email=login_request.email,
            success=False
        )
        raise AuthenticationError("Invalid email or password")
    
    if not user.is_active:
        raise AuthenticationError("Account is deactivated")
    
    # Create JWT tokens
    token_data = {
        "sub": user.id,
        "email": user.email,
        "full_name": user.full_name,
        "is_verified": user.is_verified,
        "is_premium": user.is_premium
    }
    
    access_token_expires = timedelta(
        minutes=get_settings().security.access_token_expire_minutes
    )
    
    access_token = security_manager.create_access_token(
        data=token_data,
        expires_delta=access_token_expires
    )
    
    refresh_token = security_manager.create_refresh_token(
        data={"sub": user.id}
    )
    
    security_logger.log_authentication_attempt(
        email=login_request.email,
        success=True
    )
    
    return APIResponse(
        success=True,
        message="Login successful",
        data=TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=get_settings().security.access_token_expire_minutes * 60
        )
    )


@router.post("/token", response_model=TokenResponse)
@api_rate_limit
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    security_manager: SecurityManager = Depends(get_security_manager)
):
    """
    OAuth2 compatible token endpoint.
    
    This endpoint is compatible with OAuth2 password flow
    and can be used with FastAPI's OAuth2PasswordBearer.
    """
    # Convert form data to LoginRequest
    login_request = LoginRequest(
        email=form_data.username,  # OAuth2 uses 'username' field
        password=form_data.password,
        remember_me=False
    )
    
    # Use the login_user logic
    result = await login_user(login_request, security_manager)
    return result.data


@router.post("/refresh", response_model=APIResponse[TokenResponse])
@api_rate_limit
async def refresh_token(
    refresh_token: str,
    security_manager: SecurityManager = Depends(get_security_manager)
):
    """
    Refresh access token using refresh token.
    
    Returns a new access token if the refresh token is valid.
    """
    # Verify refresh token
    payload = security_manager.verify_token(refresh_token)
    if not payload or payload.get("type") != "refresh":
        raise AuthenticationError("Invalid refresh token")
    
    user_id = payload.get("sub")
    if not user_id:
        raise AuthenticationError("Invalid refresh token")
    
    # TODO: Get user from database
    # user = await db_service.get_user_by_id(user_id)
    # if not user or not user.is_active:
    #     raise AuthenticationError("User not found or inactive")
    
    # Mock user for demonstration
    user = User(
        id=user_id,
        email="demo@example.com",
        full_name="Demo User",
        is_active=True,
        is_verified=True
    )
    
    # Create new access token
    token_data = {
        "sub": user.id,
        "email": user.email,
        "full_name": user.full_name,
        "is_verified": user.is_verified,
        "is_premium": user.is_premium
    }
    
    access_token_expires = timedelta(
        minutes=get_settings().security.access_token_expire_minutes
    )
    
    access_token = security_manager.create_access_token(
        data=token_data,
        expires_delta=access_token_expires
    )
    
    return APIResponse(
        success=True,
        message="Token refreshed successfully",
        data=TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,  # Keep the same refresh token
            token_type="bearer",
            expires_in=get_settings().security.access_token_expire_minutes * 60
        )
    )


@router.post("/logout", response_model=APIResponse[Dict[str, str]])
async def logout_user(
    current_user: User = Depends(get_current_active_user)
):
    """
    Logout current user.
    
    Invalidates the user's tokens and sessions.
    """
    # TODO: Invalidate tokens in token blacklist
    # TODO: Clear user sessions
    
    return APIResponse(
        success=True,
        message="Logout successful",
        data={"message": "You have been logged out successfully"}
    )


@router.post("/change-password", response_model=APIResponse[Dict[str, str]])
@api_rate_limit
async def change_password(
    password_change: PasswordChangeRequest,
    current_user: User = Depends(get_current_active_user),
    security_manager: SecurityManager = Depends(get_security_manager)
):
    """
    Change user password.
    
    Requires current password for verification.
    """
    # TODO: Verify current password
    # user_with_password = await db_service.get_user_with_password(current_user.id)
    # if not security_manager.verify_password(password_change.current_password, user_with_password.password_hash):
    #     raise AuthenticationError("Current password is incorrect")
    
    # For demo, accept any current password
    if password_change.current_password != "current_password":
        raise AuthenticationError("Current password is incorrect")
    
    # Hash new password
    new_password_hash = security_manager.hash_password(password_change.new_password)
    
    # TODO: Update password in database
    # await db_service.update_user_password(current_user.id, new_password_hash)
    
    # TODO: Invalidate all existing sessions/tokens
    
    security_logger.log_authentication_attempt(
        email=current_user.email,
        success=True
    )
    
    return APIResponse(
        success=True,
        message="Password changed successfully",
        data={"message": "Your password has been updated"}
    )


@router.post("/reset-password", response_model=APIResponse[Dict[str, str]])
@api_rate_limit
async def request_password_reset(
    reset_request: PasswordResetRequest
):
    """
    Request password reset email.
    
    Sends password reset instructions to the user's email.
    """
    # TODO: Check if user exists
    # user = await db_service.get_user_by_email(reset_request.email)
    
    # Always return success to prevent email enumeration
    # In a real implementation, only send email if user exists
    
    # TODO: Generate reset token and send email
    # reset_token = security_manager.generate_reset_token()
    # await email_service.send_password_reset_email(reset_request.email, reset_token)
    
    return APIResponse(
        success=True,
        message="If an account with this email exists, password reset instructions have been sent",
        data={"message": "Check your email for password reset instructions"}
    )


@router.post("/reset-password/confirm", response_model=APIResponse[Dict[str, str]])
@api_rate_limit
async def confirm_password_reset(
    reset_confirm: PasswordResetConfirm,
    security_manager: SecurityManager = Depends(get_security_manager)
):
    """
    Confirm password reset with token.
    
    Resets the user's password using the provided token.
    """
    # TODO: Verify reset token
    # token_data = await db_service.verify_reset_token(reset_confirm.token)
    # if not token_data:
    #     raise AuthenticationError("Invalid or expired reset token")
    
    # For demo, accept any token
    if reset_confirm.token != "valid_reset_token":
        raise AuthenticationError("Invalid or expired reset token")
    
    # Hash new password
    new_password_hash = security_manager.hash_password(reset_confirm.new_password)
    
    # TODO: Update password in database
    # await db_service.update_user_password(token_data.user_id, new_password_hash)
    
    # TODO: Invalidate reset token
    # await db_service.invalidate_reset_token(reset_confirm.token)
    
    return APIResponse(
        success=True,
        message="Password reset successful",
        data={"message": "Your password has been reset successfully"}
    )


@router.get("/me", response_model=APIResponse[UserResponse])
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user)
):
    """
    Get current authenticated user information.
    
    Returns the current user's profile and settings.
    """
    return APIResponse(
        success=True,
        message="User information retrieved successfully",
        data=UserResponse(
            id=current_user.id,
            email=current_user.email,
            username=current_user.username,
            full_name=current_user.full_name,
            affiliation=current_user.affiliation,
            bio=current_user.bio,
            website=current_user.website,
            orcid=current_user.orcid,
            is_active=current_user.is_active,
            is_verified=current_user.is_verified,
            is_premium=current_user.is_premium,
            preferences=current_user.preferences,
            stats=current_user.stats,
            created_at=current_user.created_at,
            last_login=current_user.last_login
        )
    )