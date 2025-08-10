import { api, handleApiError } from './api';
import { User, AuthInfo, UserPreferences } from '@types/api';

export interface LoginRequest {
  email: string;
  password: string;
}

export interface RegisterRequest {
  email: string;
  password: string;
  name: string;
}

export interface AuthResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
  user: User;
}

export class AuthService {
  // Login
  static async login(credentials: LoginRequest): Promise<AuthResponse> {
    try {
      const response = await api.post<AuthResponse>('/auth/login', credentials);
      return response;
    } catch (error) {
      throw handleApiError(error);
    }
  }

  // Register
  static async register(userData: RegisterRequest): Promise<AuthResponse> {
    try {
      const response = await api.post<AuthResponse>('/auth/register', userData);
      return response;
    } catch (error) {
      throw handleApiError(error);
    }
  }

  // Refresh token
  static async refreshToken(refreshToken: string): Promise<AuthResponse> {
    try {
      const response = await api.post<AuthResponse>('/auth/refresh', {
        refresh_token: refreshToken,
      });
      return response;
    } catch (error) {
      throw handleApiError(error);
    }
  }

  // Logout
  static async logout(): Promise<void> {
    try {
      await api.post('/auth/logout');
    } catch (error) {
      // Don't throw error on logout failure, just log it
      console.warn('Logout request failed:', error);
    }
  }

  // Get current user info
  static async getCurrentUser(): Promise<User> {
    try {
      const response = await api.get<User>('/auth/me');
      return response;
    } catch (error) {
      throw handleApiError(error);
    }
  }

  // Update user profile
  static async updateProfile(updates: Partial<User>): Promise<User> {
    try {
      const response = await api.patch<User>('/auth/profile', updates);
      return response;
    } catch (error) {
      throw handleApiError(error);
    }
  }

  // Update user preferences
  static async updatePreferences(preferences: Partial<UserPreferences>): Promise<UserPreferences> {
    try {
      const response = await api.patch<UserPreferences>('/auth/preferences', preferences);
      return response;
    } catch (error) {
      throw handleApiError(error);
    }
  }

  // Change password
  static async changePassword(currentPassword: string, newPassword: string): Promise<void> {
    try {
      await api.post('/auth/change-password', {
        current_password: currentPassword,
        new_password: newPassword,
      });
    } catch (error) {
      throw handleApiError(error);
    }
  }

  // Request password reset
  static async requestPasswordReset(email: string): Promise<void> {
    try {
      await api.post('/auth/forgot-password', { email });
    } catch (error) {
      throw handleApiError(error);
    }
  }

  // Reset password with token
  static async resetPassword(token: string, newPassword: string): Promise<void> {
    try {
      await api.post('/auth/reset-password', {
        token,
        new_password: newPassword,
      });
    } catch (error) {
      throw handleApiError(error);
    }
  }

  // Verify email
  static async verifyEmail(token: string): Promise<void> {
    try {
      await api.post('/auth/verify-email', { token });
    } catch (error) {
      throw handleApiError(error);
    }
  }

  // Resend email verification
  static async resendEmailVerification(): Promise<void> {
    try {
      await api.post('/auth/resend-verification');
    } catch (error) {
      throw handleApiError(error);
    }
  }

  // Get authentication info (for legacy compatibility)
  static async getAuthInfo(): Promise<AuthInfo> {
    try {
      const user = await this.getCurrentUser();
      return {
        isAuthenticated: true,
        user,
      };
    } catch (error) {
      return {
        isAuthenticated: false,
      };
    }
  }

  // Delete account
  static async deleteAccount(password: string): Promise<void> {
    try {
      await api.delete('/auth/account', {
        data: { password },
      });
    } catch (error) {
      throw handleApiError(error);
    }
  }

  // Get account activity/sessions
  static async getActiveSessions(): Promise<Array<{
    id: string;
    device: string;
    location: string;
    last_active: string;
    is_current: boolean;
  }>> {
    try {
      const response = await api.get('/auth/sessions');
      return response;
    } catch (error) {
      throw handleApiError(error);
    }
  }

  // Revoke session
  static async revokeSession(sessionId: string): Promise<void> {
    try {
      await api.delete(`/auth/sessions/${sessionId}`);
    } catch (error) {
      throw handleApiError(error);
    }
  }

  // Revoke all sessions except current
  static async revokeAllSessions(): Promise<void> {
    try {
      await api.delete('/auth/sessions/all');
    } catch (error) {
      throw handleApiError(error);
    }
  }
}