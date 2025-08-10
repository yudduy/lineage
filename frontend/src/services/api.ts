import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';
import { useAuthStore } from '@store/authStore';
import { useUiStore } from '@store/uiStore';

// API configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

// Create axios instance
const createApiClient = (): AxiosInstance => {
  const client = axios.create({
    baseURL: API_BASE_URL,
    timeout: 30000,
    headers: {
      'Content-Type': 'application/json',
    },
  });

  // Request interceptor for auth
  client.interceptors.request.use(
    (config) => {
      const { token, isTokenValid, refreshAuthToken } = useAuthStore.getState();
      
      if (token) {
        // Check if token is still valid
        if (!isTokenValid()) {
          // Attempt to refresh token
          refreshAuthToken().catch(() => {
            // If refresh fails, redirect to login
            useAuthStore.getState().logout();
          });
        }
        
        config.headers.Authorization = `Bearer ${token}`;
      }
      
      return config;
    },
    (error) => {
      return Promise.reject(error);
    }
  );

  // Response interceptor for error handling
  client.interceptors.response.use(
    (response: AxiosResponse) => response,
    async (error) => {
      const { addToast } = useUiStore.getState();
      
      if (error.response?.status === 401) {
        // Unauthorized - try to refresh token
        const { refreshAuthToken, logout } = useAuthStore.getState();
        
        try {
          const refreshed = await refreshAuthToken();
          if (refreshed) {
            // Retry the original request
            return client.request(error.config);
          }
        } catch (refreshError) {
          console.error('Token refresh failed:', refreshError);
        }
        
        // If refresh fails, logout and redirect
        logout();
        addToast({
          type: 'error',
          title: 'Session Expired',
          message: 'Please log in again to continue.',
        });
      } else if (error.response?.status >= 500) {
        // Server error
        addToast({
          type: 'error',
          title: 'Server Error',
          message: 'An internal server error occurred. Please try again later.',
        });
      } else if (error.code === 'NETWORK_ERROR') {
        // Network error
        addToast({
          type: 'error',
          title: 'Network Error',
          message: 'Unable to connect to the server. Please check your connection.',
        });
      }
      
      return Promise.reject(error);
    }
  );

  return client;
};

// API client instance
export const apiClient = createApiClient();

// Generic API methods
export const api = {
  get: <T = any>(url: string, config?: AxiosRequestConfig): Promise<T> =>
    apiClient.get(url, config).then(response => response.data),
    
  post: <T = any>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> =>
    apiClient.post(url, data, config).then(response => response.data),
    
  put: <T = any>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> =>
    apiClient.put(url, data, config).then(response => response.data),
    
  patch: <T = any>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> =>
    apiClient.patch(url, data, config).then(response => response.data),
    
  delete: <T = any>(url: string, config?: AxiosRequestConfig): Promise<T> =>
    apiClient.delete(url, config).then(response => response.data),
};

// Upload helper for file uploads
export const uploadFile = async (
  url: string,
  file: File,
  onProgress?: (progress: number) => void
): Promise<any> => {
  const formData = new FormData();
  formData.append('file', file);

  return apiClient.post(url, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    onUploadProgress: (progressEvent) => {
      if (onProgress && progressEvent.total) {
        const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        onProgress(progress);
      }
    },
  }).then(response => response.data);
};

// Download helper for file downloads
export const downloadFile = async (
  url: string,
  filename?: string
): Promise<void> => {
  const response = await apiClient.get(url, {
    responseType: 'blob',
  });

  const blob = new Blob([response.data]);
  const downloadUrl = window.URL.createObjectURL(blob);
  
  const link = document.createElement('a');
  link.href = downloadUrl;
  link.download = filename || 'download';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  
  window.URL.revokeObjectURL(downloadUrl);
};

// Error handling utilities
export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public code?: string,
    public details?: any
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

export const handleApiError = (error: any): ApiError => {
  if (error.response) {
    // Server responded with error status
    const { status, data } = error.response;
    return new ApiError(
      data?.message || error.message || 'An error occurred',
      status,
      data?.code,
      data?.details
    );
  } else if (error.request) {
    // Network error
    return new ApiError('Network error - please check your connection', 0);
  } else {
    // Other error
    return new ApiError(error.message || 'An unexpected error occurred', 0);
  }
};

// Request cancellation helper
export const createCancellableRequest = () => {
  const source = axios.CancelToken.source();
  
  return {
    cancelToken: source.token,
    cancel: (message?: string) => source.cancel(message),
  };
};

// Rate limiting helper
let requestQueue: Array<() => Promise<any>> = [];
let isProcessingQueue = false;
const MAX_CONCURRENT_REQUESTS = 5;

export const queueRequest = async <T>(requestFn: () => Promise<T>): Promise<T> => {
  return new Promise((resolve, reject) => {
    requestQueue.push(async () => {
      try {
        const result = await requestFn();
        resolve(result);
      } catch (error) {
        reject(error);
      }
    });
    
    processQueue();
  });
};

const processQueue = async () => {
  if (isProcessingQueue || requestQueue.length === 0) {
    return;
  }
  
  isProcessingQueue = true;
  
  while (requestQueue.length > 0) {
    const batch = requestQueue.splice(0, MAX_CONCURRENT_REQUESTS);
    await Promise.all(batch.map(request => request()));
  }
  
  isProcessingQueue = false;
};