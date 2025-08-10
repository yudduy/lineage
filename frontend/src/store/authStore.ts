import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { User, AuthInfo, ZoteroConfig } from '@types/api';

interface AuthState {
  // Authentication
  isAuthenticated: boolean;
  user: User | null;
  token: string | null;
  refreshToken: string | null;
  expiresAt: string | null;
  
  // Integrations
  zotero: ZoteroConfig;
  
  // Actions
  login: (authInfo: AuthInfo) => void;
  logout: () => void;
  updateUser: (updates: Partial<User>) => void;
  refreshAuthToken: () => Promise<boolean>;
  
  // Zotero integration
  connectZotero: (config: Omit<ZoteroConfig, 'isConnected'>) => void;
  disconnectZotero: () => void;
  updateZoteroSync: () => void;
  
  // Token management
  setToken: (token: string, refreshToken?: string, expiresAt?: string) => void;
  clearTokens: () => void;
  isTokenValid: () => boolean;
}

export const useAuthStore = create<AuthState>()(
  devtools(
    persist(
      (set, get) => ({
        // Initial state
        isAuthenticated: false,
        user: null,
        token: null,
        refreshToken: null,
        expiresAt: null,
        
        zotero: {
          isConnected: false,
          userId: undefined,
          apiKey: undefined,
          lastSync: undefined,
        },
        
        // Actions
        login: (authInfo: AuthInfo) => {
          set({
            isAuthenticated: authInfo.isAuthenticated,
            user: authInfo.user || null,
            token: authInfo.token || null,
            expiresAt: authInfo.expiresAt || null,
          });
        },
        
        logout: () => {
          set({
            isAuthenticated: false,
            user: null,
            token: null,
            refreshToken: null,
            expiresAt: null,
          });
          
          // Also disconnect integrations
          get().disconnectZotero();
        },
        
        updateUser: (updates: Partial<User>) => {
          set(state => ({
            user: state.user ? { ...state.user, ...updates } : null,
          }));
        },
        
        refreshAuthToken: async (): Promise<boolean> => {
          const { refreshToken } = get();
          
          if (!refreshToken) {
            get().logout();
            return false;
          }
          
          try {
            // This would typically call the API to refresh the token
            // For now, we'll implement the logic structure
            const response = await fetch('/api/v1/auth/refresh', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({ refresh_token: refreshToken }),
            });
            
            if (response.ok) {
              const data = await response.json();
              get().setToken(data.access_token, data.refresh_token, data.expires_at);
              return true;
            } else {
              get().logout();
              return false;
            }
          } catch (error) {
            console.error('Token refresh failed:', error);
            get().logout();
            return false;
          }
        },
        
        // Zotero integration
        connectZotero: (config: Omit<ZoteroConfig, 'isConnected'>) => {
          set({
            zotero: {
              ...config,
              isConnected: true,
            },
          });
        },
        
        disconnectZotero: () => {
          set({
            zotero: {
              isConnected: false,
              userId: undefined,
              apiKey: undefined,
              lastSync: undefined,
            },
          });
        },
        
        updateZoteroSync: () => {
          set(state => ({
            zotero: {
              ...state.zotero,
              lastSync: new Date().toISOString(),
            },
          }));
        },
        
        // Token management
        setToken: (token: string, refreshToken?: string, expiresAt?: string) => {
          set({
            token,
            refreshToken: refreshToken || get().refreshToken,
            expiresAt: expiresAt || get().expiresAt,
            isAuthenticated: true,
          });
        },
        
        clearTokens: () => {
          set({
            token: null,
            refreshToken: null,
            expiresAt: null,
            isAuthenticated: false,
          });
        },
        
        isTokenValid: (): boolean => {
          const { token, expiresAt } = get();
          
          if (!token || !expiresAt) {
            return false;
          }
          
          const now = new Date().getTime();
          const expiry = new Date(expiresAt).getTime();
          
          // Consider token invalid if it expires within 5 minutes
          return expiry > now + (5 * 60 * 1000);
        },
      }),
      {
        name: 'auth-store',
        // Exclude sensitive data from persistence in production
        partialize: (state) => ({
          isAuthenticated: state.isAuthenticated,
          user: state.user,
          zotero: state.zotero,
          // In production, you might want to exclude tokens
          // and handle them more securely
          token: state.token,
          refreshToken: state.refreshToken,
          expiresAt: state.expiresAt,
        }),
      }
    ),
    {
      name: 'auth-store',
    }
  )
);