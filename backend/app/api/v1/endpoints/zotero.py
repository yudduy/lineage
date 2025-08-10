"""
Zotero integration endpoints.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, Query, HTTPException, status, Request

from ....models.user import User
from ....models.zotero import (
    ZoteroCollection,
    ZoteroItem,
    ZoteroAuthRequest,
    ZoteroAuthResponse,
    ZoteroCollectionRequest,
    ZoteroItemsRequest,
    ZoteroAddItemsRequest,
    ZoteroImportRequest,
    ZoteroExportRequest,
    ZoteroSyncStatus
)
from ....models.common import APIResponse, PaginatedResponse
from ....middleware.auth import get_current_active_user, require_session, get_session
from ....middleware.rate_limit import api_rate_limit
from ....utils.exceptions import AuthenticationError, ExternalServiceError, ZoteroIntegrationError
from ....utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/auth/login")
async def zotero_auth_login(
    request: Request,
    current_user: User = Depends(get_current_active_user)
):
    """
    Initiate Zotero OAuth authentication.
    
    Redirects user to Zotero authorization page.
    After authorization, user will be redirected back to verify endpoint.
    """
    logger.info(
        "Initiating Zotero OAuth",
        user_id=current_user.id
    )
    
    # TODO: Implement Zotero OAuth flow
    # 1. Create request token
    # 2. Store token in session
    # 3. Redirect to Zotero authorization URL
    
    # For demo, return mock authorization URL
    mock_auth_url = f"https://www.zotero.org/oauth/authorize?oauth_token=mock_token&library_access=1&write_access=1"
    
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url=mock_auth_url)


@router.get("/auth/verify", response_model=APIResponse[ZoteroAuthResponse])
async def zotero_auth_verify(
    request: Request,
    auth_request: ZoteroAuthRequest = Depends(),
    session_data: dict = Depends(require_session)
):
    """
    Verify Zotero OAuth callback.
    
    Exchanges OAuth verifier for access token and stores user credentials.
    """
    logger.info(
        "Verifying Zotero OAuth callback",
        oauth_token=auth_request.oauth_token,
        user_id=session_data.get("user_id")
    )
    
    try:
        # TODO: Implement OAuth token exchange
        # 1. Exchange verifier for access token
        # 2. Get user info from Zotero API
        # 3. Store tokens in user session/database
        
        # Mock successful authentication
        auth_response = ZoteroAuthResponse(
            success=True,
            user_id="12345",
            username="demo_user",
            access_token="mock_access_token",
            token_secret="mock_token_secret"
        )
        
        # TODO: Store tokens in session
        # await session_manager.update_session(
        #     session_data["session_id"],
        #     {
        #         "zotero_token": auth_response.access_token,
        #         "zotero_token_secret": auth_response.token_secret,
        #         "zotero_user_id": auth_response.user_id
        #     }
        # )
        
        return APIResponse(
            success=True,
            message="Zotero authentication successful",
            data=auth_response
        )
        
    except Exception as e:
        logger.error(f"Zotero OAuth verification failed: {e}")
        
        return APIResponse(
            success=False,
            message="Zotero authentication failed",
            data=ZoteroAuthResponse(
                success=False,
                error=str(e)
            )
        )


@router.get("/collections", response_model=APIResponse[List[ZoteroCollection]])
@api_rate_limit
async def get_zotero_collections(
    current_user: User = Depends(get_current_active_user),
    session_data: dict = Depends(require_session)
):
    """
    Get user's Zotero collections.
    
    Returns list of collections available to the authenticated user.
    """
    logger.info(
        "Getting Zotero collections",
        user_id=current_user.id
    )
    
    # Check if user has Zotero authentication
    zotero_token = session_data.get("zotero_token")
    if not zotero_token:
        raise AuthenticationError("Zotero authentication required")
    
    try:
        # TODO: Make API call to Zotero
        # collections = await zotero_client.get_collections(zotero_token)
        
        # Mock collections data
        mock_collections = [
            ZoteroCollection(
                key="ABCD1234",
                version=1,
                library={"type": "user", "id": 12345},
                data={
                    "key": "ABCD1234",
                    "name": "My Research Papers",
                    "parentCollection": None
                }
            ),
            ZoteroCollection(
                key="EFGH5678",
                version=1,
                library={"type": "user", "id": 12345},
                data={
                    "key": "EFGH5678",
                    "name": "Citation Network Papers",
                    "parentCollection": None
                }
            )
        ]
        
        return APIResponse(
            success=True,
            message=f"Retrieved {len(mock_collections)} collections",
            data=mock_collections
        )
        
    except Exception as e:
        logger.error(f"Failed to get Zotero collections: {e}")
        raise ZoteroIntegrationError(f"Failed to retrieve collections: {e}")


@router.get("/collections/{collection_key}/items", response_model=APIResponse[PaginatedResponse[ZoteroItem]])
@api_rate_limit
async def get_zotero_collection_items(
    collection_key: str,
    limit: int = Query(25, ge=1, le=100, description="Number of items to return"),
    start: int = Query(0, ge=0, description="Starting index"),
    current_user: User = Depends(get_current_active_user),
    session_data: dict = Depends(require_session)
):
    """
    Get items from a specific Zotero collection.
    """
    logger.info(
        "Getting Zotero collection items",
        user_id=current_user.id,
        collection_key=collection_key,
        limit=limit,
        start=start
    )
    
    zotero_token = session_data.get("zotero_token")
    if not zotero_token:
        raise AuthenticationError("Zotero authentication required")
    
    try:
        # TODO: Make API call to Zotero
        # items = await zotero_client.get_collection_items(collection_key, limit, start)
        
        # Mock items data
        mock_items = [
            ZoteroItem(
                key="ITEM1234",
                version=1,
                library={"type": "user", "id": 12345},
                data={
                    "key": "ITEM1234",
                    "version": 1,
                    "item_type": "journalArticle",
                    "title": "A Survey of Citation Networks",
                    "creators": [
                        {"creator_type": "author", "first_name": "John", "last_name": "Doe"}
                    ],
                    "publication_title": "Journal of Information Science",
                    "date": "2023",
                    "doi": "10.1000/123456"
                }
            )
        ]
        
        return APIResponse(
            success=True,
            message=f"Retrieved {len(mock_items)} items",
            data=PaginatedResponse(
                items=mock_items,
                total=len(mock_items),
                skip=start,
                limit=limit,
                has_more=False
            )
        )
        
    except Exception as e:
        logger.error(f"Failed to get collection items: {e}")
        raise ZoteroIntegrationError(f"Failed to retrieve collection items: {e}")


@router.post("/items", response_model=APIResponse[dict])
@api_rate_limit
async def add_items_to_zotero(
    add_request: ZoteroAddItemsRequest,
    current_user: User = Depends(get_current_active_user),
    session_data: dict = Depends(require_session)
):
    """
    Add items to Zotero library.
    
    Creates new items in the user's Zotero library,
    optionally in a specific collection.
    """
    logger.info(
        "Adding items to Zotero",
        user_id=current_user.id,
        item_count=len(add_request.items),
        collection_key=add_request.collection_key
    )
    
    zotero_token = session_data.get("zotero_token")
    if not zotero_token:
        raise AuthenticationError("Zotero authentication required")
    
    try:
        # TODO: Make API call to Zotero
        # result = await zotero_client.create_items(add_request.items, add_request.collection_key)
        
        # Mock successful creation
        return APIResponse(
            success=True,
            message=f"Successfully added {len(add_request.items)} items to Zotero",
            data={
                "created": len(add_request.items),
                "failed": 0,
                "items_created": [f"ITEM{i:04d}" for i in range(len(add_request.items))]
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to add items to Zotero: {e}")
        raise ZoteroIntegrationError(f"Failed to add items: {e}")


@router.post("/import", response_model=APIResponse[dict])
@api_rate_limit
async def import_from_zotero(
    import_request: ZoteroImportRequest,
    current_user: User = Depends(get_current_active_user),
    session_data: dict = Depends(require_session)
):
    """
    Import papers from Zotero into Citation Network Explorer.
    
    Imports selected collections or items from Zotero
    and adds them to the user's paper library.
    """
    logger.info(
        "Importing from Zotero",
        user_id=current_user.id,
        collection_keys=import_request.collection_keys,
        item_keys=import_request.item_keys,
        import_mode=import_request.import_mode
    )
    
    zotero_token = session_data.get("zotero_token")
    if not zotero_token:
        raise AuthenticationError("Zotero authentication required")
    
    try:
        # TODO: Implement import logic
        # 1. Fetch items from Zotero
        # 2. Convert to internal paper format
        # 3. Save to database
        # 4. Handle duplicates based on import_mode
        
        imported_count = 5  # Mock count
        
        return APIResponse(
            success=True,
            message=f"Successfully imported {imported_count} papers from Zotero",
            data={
                "imported": imported_count,
                "skipped": 0,
                "updated": 0,
                "errors": []
            }
        )
        
    except Exception as e:
        logger.error(f"Zotero import failed: {e}")
        raise ZoteroIntegrationError(f"Import failed: {e}")


@router.post("/export", response_model=APIResponse[dict])
@api_rate_limit
async def export_to_zotero(
    export_request: ZoteroExportRequest,
    current_user: User = Depends(get_current_active_user),
    session_data: dict = Depends(require_session)
):
    """
    Export papers to Zotero.
    
    Exports selected papers from Citation Network Explorer
    to the user's Zotero library.
    """
    logger.info(
        "Exporting to Zotero",
        user_id=current_user.id,
        paper_count=len(export_request.paper_ids),
        collection_key=export_request.collection_key,
        create_collection=export_request.create_collection
    )
    
    zotero_token = session_data.get("zotero_token")
    if not zotero_token:
        raise AuthenticationError("Zotero authentication required")
    
    try:
        # TODO: Implement export logic
        # 1. Get papers from database
        # 2. Convert to Zotero item format
        # 3. Create collection if requested
        # 4. Add items to Zotero
        
        exported_count = len(export_request.paper_ids)
        
        return APIResponse(
            success=True,
            message=f"Successfully exported {exported_count} papers to Zotero",
            data={
                "exported": exported_count,
                "failed": 0,
                "collection_created": bool(export_request.create_collection),
                "collection_key": export_request.collection_key or "NEW_COLLECTION"
            }
        )
        
    except Exception as e:
        logger.error(f"Zotero export failed: {e}")
        raise ZoteroIntegrationError(f"Export failed: {e}")


@router.get("/sync/status", response_model=APIResponse[ZoteroSyncStatus])
async def get_zotero_sync_status(
    current_user: User = Depends(get_current_active_user),
    session_data: dict = Depends(require_session)
):
    """
    Get Zotero synchronization status.
    
    Returns information about the last sync operation
    and current sync status.
    """
    zotero_token = session_data.get("zotero_token")
    if not zotero_token:
        raise AuthenticationError("Zotero authentication required")
    
    # TODO: Get actual sync status from database
    sync_status = ZoteroSyncStatus(
        sync_in_progress=False,
        items_synced=150,
        collections_synced=5,
        errors=[]
    )
    
    return APIResponse(
        success=True,
        message="Sync status retrieved successfully",
        data=sync_status
    )


@router.post("/disconnect", response_model=APIResponse[dict])
async def disconnect_zotero(
    current_user: User = Depends(get_current_active_user),
    session_data: dict = Depends(require_session)
):
    """
    Disconnect Zotero account.
    
    Removes Zotero authentication tokens and stops synchronization.
    """
    logger.info(
        "Disconnecting Zotero account",
        user_id=current_user.id
    )
    
    # TODO: Remove tokens from session and database
    # await session_manager.update_session(
    #     session_data["session_id"],
    #     {
    #         "zotero_token": None,
    #         "zotero_token_secret": None,
    #         "zotero_user_id": None
    #     }
    # )
    
    return APIResponse(
        success=True,
        message="Zotero account disconnected successfully",
        data={"message": "Your Zotero account has been disconnected"}
    )