// Export all services
export { api, apiClient, uploadFile, downloadFile, ApiError, handleApiError, createCancellableRequest, queueRequest } from './api';
export { PaperService } from './paperService';
export { AuthService } from './authService';
export { ZoteroService } from './zoteroService';

// Additional service utilities
import { api } from './api';

export class HealthService {
  static async checkHealth() {
    try {
      const response = await api.get('/health');
      return response;
    } catch (error) {
      throw error;
    }
  }
}

export class SearchService {
  static async searchCrossref(query: string, limit = 10) {
    try {
      const response = await api.get('/search/crossref', {
        params: { query, limit },
      });
      return response;
    } catch (error) {
      throw error;
    }
  }

  static async searchSemanticScholar(query: string, limit = 10) {
    try {
      const response = await api.get('/search/semantic-scholar', {
        params: { query, limit },
      });
      return response;
    } catch (error) {
      throw error;
    }
  }

  static async searchOpenAlex(query: string, limit = 10) {
    try {
      const response = await api.get('/search/openalex', {
        params: { query, limit },
      });
      return response;
    } catch (error) {
      throw error;
    }
  }
}

export class MetricsService {
  static async getSystemMetrics() {
    try {
      const response = await api.get('/metrics');
      return response;
    } catch (error) {
      throw error;
    }
  }

  static async getUserMetrics() {
    try {
      const response = await api.get('/metrics/user');
      return response;
    } catch (error) {
      throw error;
    }
  }
}

export class TaskService {
  static async getTasks() {
    try {
      const response = await api.get('/tasks');
      return response;
    } catch (error) {
      throw error;
    }
  }

  static async getTask(taskId: string) {
    try {
      const response = await api.get(`/tasks/${taskId}`);
      return response;
    } catch (error) {
      throw error;
    }
  }

  static async cancelTask(taskId: string) {
    try {
      await api.delete(`/tasks/${taskId}`);
    } catch (error) {
      throw error;
    }
  }
}