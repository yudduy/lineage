// Export all stores
export { usePaperStore } from './paperStore';
export { useUiStore } from './uiStore';
export { useAuthStore } from './authStore';

// Store types
export type { PaperState } from './paperStore';
export type { UiState } from './uiStore';
export type { AuthState } from './authStore';

// Combined store hook for when you need multiple stores
import { usePaperStore } from './paperStore';
import { useUiStore } from './uiStore';
import { useAuthStore } from './authStore';

export const useStores = () => ({
  paper: usePaperStore(),
  ui: useUiStore(),
  auth: useAuthStore(),
});

// Store selectors for common operations
export const paperSelectors = {
  // Basic data
  papers: (state: any) => state.papers,
  seedPapers: (state: any) => state.seedPapers,
  edges: (state: any) => state.edges,
  
  // Graph data
  graphNodes: (state: any) => state.graphNodes,
  graphLinks: (state: any) => state.graphLinks,
  graphConfig: (state: any) => state.graphConfig,
  
  // UI state
  selectedPapers: (state: any) => state.selectedPapers,
  highlightedPaper: (state: any) => state.highlightedPaper,
  
  // Computed values
  paperCount: (state: any) => state.papers.length,
  seedCount: (state: any) => state.seedPapers.length,
  edgeCount: (state: any) => state.edges.length,
  
  // Filters
  filteredPapers: (state: any, searchQuery?: string) => {
    if (!searchQuery) return state.papers;
    const query = searchQuery.toLowerCase();
    return state.papers.filter((paper: any) =>
      paper.title.toLowerCase().includes(query) ||
      paper.authors.some((author: any) => author.name.toLowerCase().includes(query)) ||
      paper.abstract?.toLowerCase().includes(query)
    );
  },
};

export const uiSelectors = {
  // View state
  currentView: (state: any) => state.currentView,
  isSidebarOpen: (state: any) => state.isSidebarOpen,
  isNetworkView3D: (state: any) => state.isNetworkView3D,
  
  // Modal state
  modals: (state: any) => state.modals,
  isAnyModalOpen: (state: any) => Object.values(state.modals).some(Boolean),
  
  // Loading state
  loadingStates: (state: any) => state.loadingStates,
  isLoading: (state: any, key?: string) => key ? state.loadingStates[key] : Object.values(state.loadingStates).some(Boolean),
  
  // Notifications
  toasts: (state: any) => state.toasts,
  hasToasts: (state: any) => state.toasts.length > 0,
  
  // Preferences
  preferences: (state: any) => state.preferences,
  theme: (state: any) => state.preferences.theme,
  
  // Search and filters
  searchQuery: (state: any) => state.searchQuery,
  activeFilters: (state: any) => state.activeFilters,
  hasActiveFilters: (state: any) => Object.keys(state.activeFilters).length > 0,
  
  // Sort
  sortConfig: (state: any) => state.sortConfig,
};

export const authSelectors = {
  // Auth state
  isAuthenticated: (state: any) => state.isAuthenticated,
  user: (state: any) => state.user,
  token: (state: any) => state.token,
  
  // Token validity
  isTokenValid: (state: any) => state.isTokenValid(),
  
  // Zotero
  zotero: (state: any) => state.zotero,
  isZoteroConnected: (state: any) => state.zotero.isConnected,
  
  // User preferences
  userPreferences: (state: any) => state.user?.preferences,
};