import { api, handleApiError, queueRequest } from './api';
import {
  Paper,
  PaperEdge,
  PaperSearchRequest,
  PaperEdgesRequest,
  PaperResponse,
  PaperEdgesResponse,
  PaperBulkRequest,
  PaginatedResponse,
} from '@types/paper';

export class PaperService {
  // Search papers
  static async searchPapers(params: PaperSearchRequest): Promise<PaginatedResponse<Paper>> {
    try {
      const response = await api.get<PaginatedResponse<Paper>>('/papers/search', {
        params: {
          ...params,
          limit: params.pagination?.limit,
          offset: params.pagination?.offset,
        },
      });
      
      return response;
    } catch (error) {
      throw handleApiError(error);
    }
  }

  // Get paper by ID
  static async getPaper(paperId: string, includeEdges = false): Promise<PaperResponse> {
    try {
      const response = await api.get<PaperResponse>(`/papers/${paperId}`, {
        params: { include_edges: includeEdges },
      });
      
      return response;
    } catch (error) {
      throw handleApiError(error);
    }
  }

  // Get paper by DOI
  static async getPaperByDoi(doi: string): Promise<Paper> {
    try {
      const response = await api.get<Paper>('/papers/by-doi', {
        params: { doi },
      });
      
      return response;
    } catch (error) {
      throw handleApiError(error);
    }
  }

  // Get multiple papers by DOIs
  static async getPapersByDois(dois: string[]): Promise<Paper[]> {
    try {
      const response = await api.post<Paper[]>('/papers/by-dois', {
        dois: dois.slice(0, 50), // Limit to prevent overloading
      });
      
      return response;
    } catch (error) {
      throw handleApiError(error);
    }
  }

  // Get paper citation network
  static async getPaperEdges(params: PaperEdgesRequest): Promise<PaperEdgesResponse> {
    try {
      const response = await api.get<PaperEdgesResponse>(`/papers/${params.paper_id}/edges`, {
        params: {
          direction: params.direction,
          max_depth: params.max_depth,
          include_metadata: params.include_metadata,
          limit: params.pagination?.limit,
          offset: params.pagination?.offset,
        },
      });
      
      return response;
    } catch (error) {
      throw handleApiError(error);
    }
  }

  // Add papers
  static async addPapers(papers: Partial<Paper>[]): Promise<Paper[]> {
    try {
      const response = await api.post<Paper[]>('/papers', { papers });
      return response;
    } catch (error) {
      throw handleApiError(error);
    }
  }

  // Update paper
  static async updatePaper(paperId: string, updates: Partial<Paper>): Promise<Paper> {
    try {
      const response = await api.patch<Paper>(`/papers/${paperId}`, updates);
      return response;
    } catch (error) {
      throw handleApiError(error);
    }
  }

  // Delete paper
  static async deletePaper(paperId: string): Promise<void> {
    try {
      await api.delete(`/papers/${paperId}`);
    } catch (error) {
      throw handleApiError(error);
    }
  }

  // Bulk operations
  static async bulkOperation(request: PaperBulkRequest): Promise<any> {
    try {
      const response = await api.post('/papers/bulk', request);
      return response;
    } catch (error) {
      throw handleApiError(error);
    }
  }

  // Search by title (fuzzy search)
  static async searchByTitle(title: string, limit = 10): Promise<Paper[]> {
    try {
      const response = await api.get<Paper[]>('/papers/search-title', {
        params: { title, limit },
      });
      
      return response;
    } catch (error) {
      throw handleApiError(error);
    }
  }

  // Get paper suggestions based on seed papers
  static async getSuggestions(seedPaperIds: string[], limit = 20): Promise<Paper[]> {
    try {
      const response = await api.post<Paper[]>('/papers/suggestions', {
        seed_paper_ids: seedPaperIds,
        limit,
      });
      
      return response;
    } catch (error) {
      throw handleApiError(error);
    }
  }

  // Import from BibTeX
  static async importFromBibtex(bibtexContent: string): Promise<Paper[]> {
    try {
      const response = await api.post<Paper[]>('/papers/import/bibtex', {
        content: bibtexContent,
      });
      
      return response;
    } catch (error) {
      throw handleApiError(error);
    }
  }

  // Export to various formats
  static async exportPapers(
    paperIds: string[],
    format: 'json' | 'csv' | 'bibtex' | 'ris' | 'graphml'
  ): Promise<string> {
    try {
      const response = await api.post<{ download_url: string }>('/papers/export', {
        paper_ids: paperIds,
        format,
      });
      
      return response.download_url;
    } catch (error) {
      throw handleApiError(error);
    }
  }

  // Get citation metrics
  static async getCitationMetrics(paperId: string): Promise<{
    total_citations: number;
    h_index: number;
    citation_distribution: Array<{ year: number; count: number }>;
  }> {
    try {
      const response = await api.get(`/papers/${paperId}/metrics`);
      return response;
    } catch (error) {
      throw handleApiError(error);
    }
  }

  // Get paper timeline data
  static async getTimelineData(paperIds: string[]): Promise<Array<{
    paper_id: string;
    year: number;
    citation_count: number;
    references_count: number;
  }>> {
    try {
      const response = await api.post('/papers/timeline', {
        paper_ids: paperIds,
      });
      
      return response;
    } catch (error) {
      throw handleApiError(error);
    }
  }

  // Batch fetch papers with rate limiting
  static async batchFetchPapers(paperIds: string[]): Promise<Paper[]> {
    const BATCH_SIZE = 10;
    const batches: string[][] = [];
    
    for (let i = 0; i < paperIds.length; i += BATCH_SIZE) {
      batches.push(paperIds.slice(i, i + BATCH_SIZE));
    }
    
    try {
      const results = await Promise.all(
        batches.map(batch =>
          queueRequest(() =>
            api.post<Paper[]>('/papers/batch', { paper_ids: batch })
          )
        )
      );
      
      return results.flat();
    } catch (error) {
      throw handleApiError(error);
    }
  }

  // Get related papers using ML/similarity
  static async getRelatedPapers(
    paperId: string,
    method: 'content' | 'citation' | 'hybrid' = 'hybrid',
    limit = 10
  ): Promise<Paper[]> {
    try {
      const response = await api.get<Paper[]>(`/papers/${paperId}/related`, {
        params: { method, limit },
      });
      
      return response;
    } catch (error) {
      throw handleApiError(error);
    }
  }

  // Advanced search with filters
  static async advancedSearch(filters: {
    query?: string;
    authors?: string[];
    journals?: string[];
    year_range?: [number, number];
    citation_range?: [number, number];
    subjects?: string[];
    open_access?: boolean;
    language?: string;
    paper_type?: string;
  }, pagination = { limit: 20, offset: 0 }): Promise<PaginatedResponse<Paper>> {
    try {
      const response = await api.post<PaginatedResponse<Paper>>('/papers/advanced-search', {
        filters,
        ...pagination,
      });
      
      return response;
    } catch (error) {
      throw handleApiError(error);
    }
  }
}