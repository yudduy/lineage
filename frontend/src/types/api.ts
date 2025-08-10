// API Response types
export interface ApiResponse<T> {
  success: boolean;
  data: T;
  message?: string;
  errors?: string[];
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  per_page: number;
  total_pages: number;
  has_next: boolean;
  has_prev: boolean;
}

export interface ApiError {
  message: string;
  code: string;
  details?: { [key: string]: any };
  status: number;
}

// Health check types
export interface HealthCheck {
  status: 'ok' | 'error';
  timestamp: string;
  version: string;
  services: {
    database: 'ok' | 'error';
    cache: 'ok' | 'error';
    external_apis: 'ok' | 'error';
  };
}

// Authentication types
export interface User {
  id: string;
  email?: string;
  name?: string;
  created_at: string;
  updated_at: string;
  preferences?: UserPreferences;
}

export interface UserPreferences {
  theme: 'light' | 'dark' | 'auto';
  defaultView: 'network' | 'list' | 'table';
  graphLayout: '2d' | '3d';
  notifications: boolean;
  autoSave: boolean;
}

export interface AuthInfo {
  isAuthenticated: boolean;
  user?: User;
  token?: string;
  expiresAt?: string;
}

// Search and filtering types
export interface SearchFilters {
  query?: string;
  yearRange?: [number, number];
  citationRange?: [number, number];
  openAccess?: boolean;
  subjects?: string[];
  authors?: string[];
  journals?: string[];
}

export interface SortOptions {
  field: 'title' | 'year' | 'citations' | 'relevance' | 'created_at';
  order: 'asc' | 'desc';
}

// Export types
export interface ExportOptions {
  format: 'json' | 'csv' | 'bibtex' | 'ris' | 'graphml' | 'png' | 'svg';
  includeMetadata: boolean;
  includeAbstracts: boolean;
  paperIds?: string[];
}

export interface ExportResult {
  downloadUrl: string;
  filename: string;
  format: string;
  size: number;
  expiresAt: string;
}

// Task/job types for async operations
export interface Task {
  id: string;
  type: 'search' | 'export' | 'import' | 'analysis';
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  message?: string;
  result?: any;
  error?: string;
  created_at: string;
  updated_at: string;
}

// Metrics and analytics
export interface SystemMetrics {
  papers_count: number;
  edges_count: number;
  users_count: number;
  searches_today: number;
  api_calls_today: number;
  storage_used: number;
  last_updated: string;
}

// Integration types
export interface ZoteroConfig {
  isConnected: boolean;
  userId?: string;
  apiKey?: string;
  lastSync?: string;
}

export interface ZoteroCollection {
  id: string;
  name: string;
  description?: string;
  itemCount: number;
  parentCollection?: string;
}

export interface ZoteroItem {
  id: string;
  title: string;
  creators: Array<{
    firstName?: string;
    lastName: string;
    creatorType: string;
  }>;
  date?: string;
  doi?: string;
  url?: string;
  collections: string[];
}

// WebSocket message types
export interface WebSocketMessage {
  type: 'task_update' | 'new_paper' | 'system_notification';
  data: any;
  timestamp: string;
}

export interface TaskUpdate {
  taskId: string;
  status: Task['status'];
  progress: number;
  message?: string;
  result?: any;
  error?: string;
}

// Configuration types
export interface AppConfig {
  apiUrl: string;
  wsUrl?: string;
  enableAnalytics: boolean;
  enableWebSockets: boolean;
  maxFileSize: number;
  supportedFormats: string[];
  rateLimit: {
    requests: number;
    window: number;
  };
}