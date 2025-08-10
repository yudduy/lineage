import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { ViewMode, ToastMessage } from '@types/index';

interface UiState {
  // View management
  currentView: ViewMode['type'];
  isSidebarOpen: boolean;
  isNetworkView3D: boolean;
  
  // Modal state
  modals: {
    addPaper: boolean;
    doiInput: boolean;
    titleSearch: boolean;
    bibtexUpload: boolean;
    zoteroImport: boolean;
    help: boolean;
    settings: boolean;
    export: boolean;
    paperDetails: boolean;
    researchIntelligence: boolean;
    communityAnalysis: boolean;
    citationFlow: boolean;
  };
  
  // Loading states
  loadingStates: {
    papers: boolean;
    search: boolean;
    export: boolean;
    import: boolean;
    general: boolean;
  };
  
  // Notifications
  toasts: ToastMessage[];
  
  // UI preferences (persisted)
  preferences: {
    theme: 'light' | 'dark' | 'auto';
    showWelcome: boolean;
    autoSave: boolean;
    compactMode: boolean;
    animations: boolean;
  };
  
  // Search and filters
  searchQuery: string;
  activeFilters: {
    yearRange?: [number, number];
    citationRange?: [number, number];
    openAccess?: boolean;
    subjects?: string[];
    journals?: string[];
  };
  
  // Table/List state
  sortConfig: {
    field: string;
    order: 'asc' | 'desc';
  };
  
  // Actions
  setCurrentView: (view: ViewMode['type']) => void;
  toggleSidebar: () => void;
  setSidebarOpen: (open: boolean) => void;
  toggle3DView: () => void;
  
  // Modal actions
  openModal: (modal: keyof UiState['modals']) => void;
  closeModal: (modal: keyof UiState['modals']) => void;
  closeAllModals: () => void;
  
  // Loading actions
  setLoading: (key: keyof UiState['loadingStates'], loading: boolean) => void;
  
  // Toast actions
  addToast: (toast: Omit<ToastMessage, 'id'>) => void;
  removeToast: (id: string) => void;
  clearToasts: () => void;
  
  // Preference actions
  setPreference: <K extends keyof UiState['preferences']>(
    key: K,
    value: UiState['preferences'][K]
  ) => void;
  
  // Search and filter actions
  setSearchQuery: (query: string) => void;
  setFilter: <K extends keyof UiState['activeFilters']>(
    key: K,
    value: UiState['activeFilters'][K]
  ) => void;
  clearFilters: () => void;
  
  // Sort actions
  setSortConfig: (config: UiState['sortConfig']) => void;
}

export const useUiStore = create<UiState>()(
  devtools(
    persist(
      (set, get) => ({
        // Initial state
        currentView: 'network',
        isSidebarOpen: true,
        isNetworkView3D: false,
        
        modals: {
          addPaper: false,
          doiInput: false,
          titleSearch: false,
          bibtexUpload: false,
          zoteroImport: false,
          help: false,
          settings: false,
          export: false,
          paperDetails: false,
          researchIntelligence: false,
          communityAnalysis: false,
          citationFlow: false,
        },
        
        loadingStates: {
          papers: false,
          search: false,
          export: false,
          import: false,
          general: false,
        },
        
        toasts: [],
        
        preferences: {
          theme: 'auto',
          showWelcome: true,
          autoSave: true,
          compactMode: false,
          animations: true,
        },
        
        searchQuery: '',
        activeFilters: {},
        
        sortConfig: {
          field: 'title',
          order: 'asc',
        },
        
        // Actions
        setCurrentView: (view: ViewMode['type']) => {
          set({ currentView: view });
        },
        
        toggleSidebar: () => {
          set(state => ({ isSidebarOpen: !state.isSidebarOpen }));
        },
        
        setSidebarOpen: (open: boolean) => {
          set({ isSidebarOpen: open });
        },
        
        toggle3DView: () => {
          set(state => ({ isNetworkView3D: !state.isNetworkView3D }));
        },
        
        // Modal actions
        openModal: (modal: keyof UiState['modals']) => {
          set(state => ({
            modals: {
              ...state.modals,
              [modal]: true,
            },
          }));
        },
        
        closeModal: (modal: keyof UiState['modals']) => {
          set(state => ({
            modals: {
              ...state.modals,
              [modal]: false,
            },
          }));
        },
        
        closeAllModals: () => {
          set(state => ({
            modals: Object.keys(state.modals).reduce((acc, key) => ({
              ...acc,
              [key]: false,
            }), {} as UiState['modals']),
          }));
        },
        
        // Loading actions
        setLoading: (key: keyof UiState['loadingStates'], loading: boolean) => {
          set(state => ({
            loadingStates: {
              ...state.loadingStates,
              [key]: loading,
            },
          }));
        },
        
        // Toast actions
        addToast: (toast: Omit<ToastMessage, 'id'>) => {
          const id = `toast_${Date.now()}_${Math.random()}`;
          const newToast: ToastMessage = {
            ...toast,
            id,
            duration: toast.duration || 5000,
          };
          
          set(state => ({
            toasts: [...state.toasts, newToast],
          }));
          
          // Auto-remove toast after duration
          if (newToast.duration > 0) {
            setTimeout(() => {
              get().removeToast(id);
            }, newToast.duration);
          }
        },
        
        removeToast: (id: string) => {
          set(state => ({
            toasts: state.toasts.filter(toast => toast.id !== id),
          }));
        },
        
        clearToasts: () => {
          set({ toasts: [] });
        },
        
        // Preference actions
        setPreference: <K extends keyof UiState['preferences']>(
          key: K,
          value: UiState['preferences'][K]
        ) => {
          set(state => ({
            preferences: {
              ...state.preferences,
              [key]: value,
            },
          }));
        },
        
        // Search and filter actions
        setSearchQuery: (query: string) => {
          set({ searchQuery: query });
        },
        
        setFilter: <K extends keyof UiState['activeFilters']>(
          key: K,
          value: UiState['activeFilters'][K]
        ) => {
          set(state => ({
            activeFilters: {
              ...state.activeFilters,
              [key]: value,
            },
          }));
        },
        
        clearFilters: () => {
          set({ activeFilters: {} });
        },
        
        setSortConfig: (config: UiState['sortConfig']) => {
          set({ sortConfig: config });
        },
      }),
      {
        name: 'ui-store',
        // Only persist preferences
        partialize: (state) => ({
          preferences: state.preferences,
          isSidebarOpen: state.isSidebarOpen,
          currentView: state.currentView,
          isNetworkView3D: state.isNetworkView3D,
        }),
      }
    ),
    {
      name: 'ui-store',
    }
  )
);