"""
User management endpoints.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status

from ....models.user import User, UserUpdate, UserResponse, UserPreferences
from ....models.common import APIResponse
from ....middleware.auth import get_current_active_user, get_current_verified_user
from ....middleware.rate_limit import api_rate_limit
from ....utils.exceptions import NotFoundError, ValidationError
from ....utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/me", response_model=APIResponse[UserResponse])
async def get_current_user_profile(
    current_user: User = Depends(get_current_active_user)
):
    """
    Get current user's profile information.
    
    Returns detailed information about the authenticated user.
    """
    return APIResponse(
        success=True,
        message="User profile retrieved successfully",
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


@router.patch("/me", response_model=APIResponse[UserResponse])
@api_rate_limit
async def update_current_user_profile(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_active_user)
):
    """
    Update current user's profile information.
    
    Updates the authenticated user's profile with provided data.
    Only non-null fields will be updated.
    """
    logger.info(
        "Updating user profile",
        user_id=current_user.id,
        fields_to_update=list(user_update.model_dump(exclude_unset=True).keys())
    )
    
    # TODO: Update user in database
    # updated_user = await db_service.update_user(current_user.id, user_update)
    
    # For demo, update the current user object
    update_data = user_update.model_dump(exclude_unset=True)
    
    if "full_name" in update_data:
        current_user.full_name = update_data["full_name"]
    if "username" in update_data:
        current_user.username = update_data["username"]
    if "affiliation" in update_data:
        current_user.affiliation = update_data["affiliation"]
    if "bio" in update_data:
        current_user.bio = update_data["bio"]
    if "website" in update_data:
        current_user.website = update_data["website"]
    if "orcid" in update_data:
        current_user.orcid = update_data["orcid"]
    if "preferences" in update_data:
        current_user.preferences = UserPreferences(**update_data["preferences"])
    
    return APIResponse(
        success=True,
        message="Profile updated successfully",
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


@router.get("/me/preferences", response_model=APIResponse[UserPreferences])
async def get_user_preferences(
    current_user: User = Depends(get_current_active_user)
):
    """
    Get current user's preferences and settings.
    """
    return APIResponse(
        success=True,
        message="User preferences retrieved successfully",
        data=current_user.preferences
    )


@router.patch("/me/preferences", response_model=APIResponse[UserPreferences])
@api_rate_limit
async def update_user_preferences(
    preferences_update: UserPreferences,
    current_user: User = Depends(get_current_active_user)
):
    """
    Update current user's preferences and settings.
    """
    logger.info(
        "Updating user preferences",
        user_id=current_user.id,
        preferences=preferences_update.model_dump()
    )
    
    # TODO: Update preferences in database
    # await db_service.update_user_preferences(current_user.id, preferences_update)
    
    # For demo, update the current user
    current_user.preferences = preferences_update
    
    return APIResponse(
        success=True,
        message="Preferences updated successfully",
        data=current_user.preferences
    )


@router.delete("/me", response_model=APIResponse[dict])
@api_rate_limit
async def delete_current_user_account(
    current_user: User = Depends(get_current_verified_user)
):
    """
    Delete current user's account.
    
    This action is irreversible and requires email verification.
    All user data including papers, collections, and preferences will be deleted.
    """
    logger.warning(
        "User account deletion requested",
        user_id=current_user.id,
        email=current_user.email
    )
    
    # TODO: Implement account deletion
    # - Delete user data from database
    # - Invalidate all sessions and tokens
    # - Send confirmation email
    # - Comply with data protection regulations (GDPR, etc.)
    
    # await db_service.delete_user_account(current_user.id)
    
    return APIResponse(
        success=True,
        message="Account deletion initiated. You will receive a confirmation email.",
        data={"message": "Your account has been scheduled for deletion"}
    )


@router.get("/{user_id}", response_model=APIResponse[UserResponse])
async def get_user_public_profile(
    user_id: str,
    current_user: Optional[User] = Depends(get_current_active_user)
):
    """
    Get public profile of a user by ID.
    
    Returns only publicly visible information based on user's privacy settings.
    """
    # TODO: Get user from database
    # user = await db_service.get_user_by_id(user_id)
    # if not user:
    #     raise NotFoundError(resource_type="User", resource_id=user_id)
    
    # For demo, return a mock user
    if user_id != "demo_user_123":
        raise NotFoundError(resource_type="User", resource_id=user_id)
    
    user = User(
        id=user_id,
        email="demo@example.com",
        username="demo_user",
        full_name="Demo User",
        affiliation="Demo University",
        bio="This is a demo user profile",
        is_active=True,
        is_verified=True,
        is_premium=False
    )
    
    # Check privacy settings - only show public information
    if not user.preferences.profile_public and (not current_user or current_user.id != user_id):
        return APIResponse(
            success=True,
            message="User profile is private",
            data=UserResponse(
                id=user.id,
                email="",  # Hide email
                username=user.username,
                full_name=user.full_name,
                affiliation="",  # Hide affiliation
                bio="",  # Hide bio
                is_active=user.is_active,
                is_verified=user.is_verified,
                is_premium=user.is_premium,
                preferences=UserPreferences(),  # Default preferences
                stats=user.stats,
                created_at=user.created_at,
                last_login=None
            )
        )
    
    return APIResponse(
        success=True,
        message="User profile retrieved successfully",
        data=UserResponse(
            id=user.id,
            email=user.email if current_user and current_user.id == user_id else "",
            username=user.username,
            full_name=user.full_name,
            affiliation=user.affiliation,
            bio=user.bio,
            website=user.website,
            orcid=user.orcid,
            is_active=user.is_active,
            is_verified=user.is_verified,
            is_premium=user.is_premium,
            preferences=user.preferences,
            stats=user.stats,
            created_at=user.created_at,
            last_login=user.last_login if current_user and current_user.id == user_id else None
        )
    )