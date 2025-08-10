import { io, Socket } from 'socket.io-client';
import { WebSocketMessage } from '@types/paper';

export interface WebSocketConfig {
  url: string;
  autoConnect?: boolean;
  reconnect?: boolean;
  maxReconnectAttempts?: number;
}

export interface WebSocketEventHandlers {
  onConnect?: () => void;
  onDisconnect?: () => void;
  onError?: (error: Error) => void;
  onMessage?: (message: WebSocketMessage) => void;
  onProgress?: (data: any) => void;
  onUpdate?: (data: any) => void;
  onNotification?: (data: any) => void;
  onCollaboration?: (data: any) => void;
}

class WebSocketService {
  private socket: Socket | null = null;
  private config: WebSocketConfig;
  private handlers: WebSocketEventHandlers = {};
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private isConnecting = false;

  constructor(config: WebSocketConfig) {
    this.config = {
      autoConnect: true,
      reconnect: true,
      maxReconnectAttempts: 5,
      ...config,
    };
    this.maxReconnectAttempts = this.config.maxReconnectAttempts!;

    if (this.config.autoConnect) {
      this.connect();
    }
  }

  connect(): Promise<void> {
    if (this.socket?.connected || this.isConnecting) {
      return Promise.resolve();
    }

    this.isConnecting = true;

    return new Promise((resolve, reject) => {
      try {
        this.socket = io(this.config.url, {
          transports: ['websocket', 'polling'],
          upgrade: true,
          reconnection: this.config.reconnect,
          reconnectionAttempts: this.maxReconnectAttempts,
          reconnectionDelay: 1000,
          reconnectionDelayMax: 5000,
          timeout: 10000,
        });

        this.setupEventListeners();

        this.socket.on('connect', () => {
          this.isConnecting = false;
          this.reconnectAttempts = 0;
          console.log('WebSocket connected successfully');
          this.handlers.onConnect?.();
          resolve();
        });

        this.socket.on('connect_error', (error) => {
          this.isConnecting = false;
          console.error('WebSocket connection error:', error);
          this.handlers.onError?.(error);
          reject(error);
        });

      } catch (error) {
        this.isConnecting = false;
        console.error('Failed to initialize WebSocket:', error);
        reject(error);
      }
    });
  }

  private setupEventListeners(): void {
    if (!this.socket) return;

    this.socket.on('disconnect', (reason) => {
      console.log('WebSocket disconnected:', reason);
      this.handlers.onDisconnect?.();

      // Auto-reconnect logic
      if (this.config.reconnect && this.reconnectAttempts < this.maxReconnectAttempts) {
        this.reconnectAttempts++;
        console.log(`Attempting to reconnect... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
        
        setTimeout(() => {
          this.connect().catch(console.error);
        }, Math.min(1000 * Math.pow(2, this.reconnectAttempts), 10000));
      }
    });

    this.socket.on('error', (error) => {
      console.error('WebSocket error:', error);
      this.handlers.onError?.(new Error(error));
    });

    // Handle different message types
    this.socket.on('message', (data: WebSocketMessage) => {
      this.handlers.onMessage?.(data);
      
      switch (data.type) {
        case 'progress':
          this.handlers.onProgress?.(data.data);
          break;
        case 'update':
          this.handlers.onUpdate?.(data.data);
          break;
        case 'notification':
          this.handlers.onNotification?.(data.data);
          break;
        case 'collaboration':
          this.handlers.onCollaboration?.(data.data);
          break;
      }
    });

    // Specific event handlers
    this.socket.on('progress_update', (data) => {
      this.handlers.onProgress?.(data);
    });

    this.socket.on('data_update', (data) => {
      this.handlers.onUpdate?.(data);
    });

    this.socket.on('notification', (data) => {
      this.handlers.onNotification?.(data);
    });

    this.socket.on('collaboration_event', (data) => {
      this.handlers.onCollaboration?.(data);
    });
  }

  setHandlers(handlers: WebSocketEventHandlers): void {
    this.handlers = { ...this.handlers, ...handlers };
  }

  emit(event: string, data?: any): void {
    if (this.socket?.connected) {
      this.socket.emit(event, data);
    } else {
      console.warn('WebSocket not connected, cannot emit event:', event);
    }
  }

  // Specific methods for common operations
  subscribeToTask(taskId: string): void {
    this.emit('subscribe_task', { taskId });
  }

  unsubscribeFromTask(taskId: string): void {
    this.emit('unsubscribe_task', { taskId });
  }

  subscribeToCollaboration(sessionId: string): void {
    this.emit('join_collaboration', { sessionId });
  }

  leaveCollaboration(sessionId: string): void {
    this.emit('leave_collaboration', { sessionId });
  }

  sendCollaborationUpdate(data: any): void {
    this.emit('collaboration_update', data);
  }

  requestLiveData(endpoint: string, params?: any): void {
    this.emit('request_live_data', { endpoint, params });
  }

  disconnect(): void {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }

  isConnected(): boolean {
    return this.socket?.connected || false;
  }

  getConnectionState(): 'connected' | 'connecting' | 'disconnected' | 'error' {
    if (this.isConnecting) return 'connecting';
    if (this.socket?.connected) return 'connected';
    if (this.socket?.disconnected) return 'disconnected';
    return 'error';
  }
}

// Singleton instance
let wsService: WebSocketService | null = null;

export const getWebSocketService = (config?: WebSocketConfig): WebSocketService => {
  if (!wsService && config) {
    wsService = new WebSocketService(config);
  } else if (!wsService) {
    // Default config - should be provided by environment or app config
    const defaultConfig: WebSocketConfig = {
      url: process.env.VITE_WS_URL || 'ws://localhost:8000',
      autoConnect: false, // Don't auto-connect without proper config
    };
    wsService = new WebSocketService(defaultConfig);
  }
  return wsService;
};

export const initializeWebSocket = (config: WebSocketConfig): WebSocketService => {
  if (wsService) {
    wsService.disconnect();
  }
  wsService = new WebSocketService(config);
  return wsService;
};

export default WebSocketService;