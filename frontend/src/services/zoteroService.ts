import { api, handleApiError } from './api';
import { ZoteroCollection, ZoteroItem } from '@types/api';
import { Paper } from '@types/paper';

export interface ZoteroAuthRequest {
  user_id: string;
  api_key: string;
}

export interface ZoteroAuthResponse {
  success: boolean;
  user_info: {
    user_id: string;
    username: string;
    display_name: string;
  };
}

export class ZoteroService {
  // Authenticate with Zotero
  static async authenticate(credentials: ZoteroAuthRequest): Promise<ZoteroAuthResponse> {
    try {
      const response = await api.post<ZoteroAuthResponse>('/zotero/auth', credentials);
      return response;
    } catch (error) {
      throw handleApiError(error);
    }
  }

  // Get Zotero authentication URL (OAuth flow)
  static async getAuthUrl(): Promise<{ auth_url: string; request_token: string }> {
    try {
      const response = await api.get('/zotero/auth/url');
      return response;
    } catch (error) {
      throw handleApiError(error);
    }
  }

  // Complete OAuth authentication
  static async completeAuth(requestToken: string, verifier: string): Promise<ZoteroAuthResponse> {
    try {
      const response = await api.post<ZoteroAuthResponse>('/zotero/auth/complete', {
        request_token: requestToken,
        verifier,
      });
      return response;
    } catch (error) {
      throw handleApiError(error);
    }
  }

  // Disconnect Zotero
  static async disconnect(): Promise<void> {
    try {
      await api.delete('/zotero/auth');
    } catch (error) {
      throw handleApiError(error);
    }
  }

  // Get user's Zotero collections
  static async getCollections(): Promise<ZoteroCollection[]> {
    try {
      const response = await api.get<ZoteroCollection[]>('/zotero/collections');
      return response;
    } catch (error) {
      throw handleApiError(error);
    }
  }

  // Get items in a collection
  static async getItemsInCollection(
    collectionId: string,
    limit = 50,
    start = 0
  ): Promise<{
    items: ZoteroItem[];
    total: number;
    has_more: boolean;
  }> {
    try {
      const response = await api.get(`/zotero/collections/${collectionId}/items`, {
        params: { limit, start },
      });
      return response;
    } catch (error) {
      throw handleApiError(error);
    }
  }

  // Get all items from user's library
  static async getAllItems(
    limit = 100,
    start = 0,
    tag?: string
  ): Promise<{
    items: ZoteroItem[];
    total: number;
    has_more: boolean;
  }> {
    try {
      const response = await api.get('/zotero/items', {
        params: { limit, start, tag },
      });
      return response;
    } catch (error) {
      throw handleApiError(error);
    }
  }

  // Search Zotero library
  static async searchItems(query: string, limit = 50): Promise<ZoteroItem[]> {
    try {
      const response = await api.get<ZoteroItem[]>('/zotero/search', {
        params: { q: query, limit },
      });
      return response;
    } catch (error) {
      throw handleApiError(error);
    }
  }

  // Convert Zotero items to Papers
  static async convertItemsToPapers(itemIds: string[]): Promise<Paper[]> {
    try {
      const response = await api.post<Paper[]>('/zotero/convert', {
        item_ids: itemIds,
      });
      return response;
    } catch (error) {
      throw handleApiError(error);
    }
  }

  // Import papers from Zotero collection
  static async importFromCollection(collectionId: string): Promise<{
    papers: Paper[];
    imported_count: number;
    skipped_count: number;
    errors: string[];
  }> {
    try {
      const response = await api.post(`/zotero/collections/${collectionId}/import`);
      return response;
    } catch (error) {
      throw handleApiError(error);
    }
  }

  // Add papers to Zotero collection
  static async addPapersToCollection(
    collectionId: string,
    papers: Paper[]
  ): Promise<{
    added_count: number;
    skipped_count: number;
    errors: string[];
  }> {
    try {
      const response = await api.post(`/zotero/collections/${collectionId}/add`, {
        papers,
      });
      return response;
    } catch (error) {
      throw handleApiError(error);
    }
  }

  // Create new Zotero collection
  static async createCollection(name: string, parentCollection?: string): Promise<ZoteroCollection> {
    try {
      const response = await api.post<ZoteroCollection>('/zotero/collections', {
        name,
        parent_collection: parentCollection,
      });
      return response;
    } catch (error) {
      throw handleApiError(error);
    }
  }

  // Sync with Zotero (get latest changes)
  static async sync(): Promise<{
    new_items: number;
    updated_items: number;
    deleted_items: number;
    last_sync: string;
  }> {
    try {
      const response = await api.post('/zotero/sync');
      return response;
    } catch (error) {
      throw handleApiError(error);
    }
  }

  // Get sync status
  static async getSyncStatus(): Promise<{
    last_sync: string | null;
    is_syncing: boolean;
    sync_progress?: number;
    error?: string;
  }> {
    try {
      const response = await api.get('/zotero/sync/status');
      return response;
    } catch (error) {
      throw handleApiError(error);
    }
  }

  // Get Zotero connection status
  static async getConnectionStatus(): Promise<{
    is_connected: boolean;
    user_id?: string;
    username?: string;
    last_sync?: string;
    quota?: {
      used: number;
      total: number;
      percentage: number;
    };
  }> {
    try {
      const response = await api.get('/zotero/status');
      return response;
    } catch (error) {
      throw handleApiError(error);
    }
  }

  // Export papers to Zotero
  static async exportPapersToZotero(
    papers: Paper[],
    collectionId?: string
  ): Promise<{
    exported_count: number;
    skipped_count: number;
    errors: string[];
    created_items: string[];
  }> {
    try {
      const response = await api.post('/zotero/export', {
        papers,
        collection_id: collectionId,
      });
      return response;
    } catch (error) {
      throw handleApiError(error);
    }
  }

  // Get Zotero item details
  static async getItemDetails(itemId: string): Promise<ZoteroItem> {
    try {
      const response = await api.get<ZoteroItem>(`/zotero/items/${itemId}`);
      return response;
    } catch (error) {
      throw handleApiError(error);
    }
  }

  // Get user's Zotero tags
  static async getTags(): Promise<Array<{
    tag: string;
    count: number;
  }>> {
    try {
      const response = await api.get('/zotero/tags');
      return response;
    } catch (error) {
      throw handleApiError(error);
    }
  }

  // Batch operations
  static async batchImportItems(itemIds: string[]): Promise<{
    papers: Paper[];
    success_count: number;
    error_count: number;
    errors: Array<{ item_id: string; error: string }>;
  }> {
    try {
      const response = await api.post('/zotero/batch-import', {
        item_ids: itemIds,
      });
      return response;
    } catch (error) {
      throw handleApiError(error);
    }
  }
}