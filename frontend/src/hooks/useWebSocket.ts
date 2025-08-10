import { useEffect, useRef, useState } from 'react';
import { getWebSocketService, WebSocketEventHandlers, WebSocketConfig } from '@services/websocketService';
import { WebSocketMessage } from '@types/paper';
import { useUiStore } from '@store/index';
import toast from 'react-hot-toast';

interface UseWebSocketOptions extends Partial<WebSocketEventHandlers> {
  enabled?: boolean;
  reconnect?: boolean;
  config?: Partial<WebSocketConfig>;
}

interface UseWebSocketReturn {
  isConnected: boolean;
  isConnecting: boolean;
  connectionState: 'connected' | 'connecting' | 'disconnected' | 'error';
  send: (event: string, data?: any) => void;
  subscribeToTask: (taskId: string) => void;
  unsubscribeFromTask: (taskId: string) => void;
  subscribeToCollaboration: (sessionId: string) => void;
  leaveCollaboration: (sessionId: string) => void;
  requestLiveData: (endpoint: string, params?: any) => void;
}

export const useWebSocket = (options: UseWebSocketOptions = {}): UseWebSocketReturn => {
  const {
    enabled = true,
    reconnect = true,
    config = {},
    onConnect,
    onDisconnect,
    onError,
    onMessage,
    onProgress,
    onUpdate,
    onNotification,
    onCollaboration,
  } = options;

  const wsServiceRef = useRef<ReturnType<typeof getWebSocketService> | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [connectionState, setConnectionState] = useState<UseWebSocketReturn['connectionState']>('disconnected');

  const { addToast } = useUiStore();

  useEffect(() => {
    if (!enabled) return;

    const wsConfig: WebSocketConfig = {
      url: import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws',
      autoConnect: true,
      reconnect,
      maxReconnectAttempts: 5,
      ...config,
    };

    try {
      wsServiceRef.current = getWebSocketService(wsConfig);

      const handlers: WebSocketEventHandlers = {
        onConnect: () => {
          setIsConnected(true);
          setIsConnecting(false);
          setConnectionState('connected');
          console.log('WebSocket connected');
          
          addToast({
            type: 'success',
            message: 'Real-time updates enabled',
            duration: 3000,
          });

          onConnect?.();
        },

        onDisconnect: () => {
          setIsConnected(false);
          setIsConnecting(false);
          setConnectionState('disconnected');
          console.log('WebSocket disconnected');
          
          addToast({
            type: 'warning',
            message: 'Real-time updates disconnected',
            duration: 3000,
          });

          onDisconnect?.();
        },

        onError: (error: Error) => {
          setIsConnecting(false);
          setConnectionState('error');
          console.error('WebSocket error:', error);
          
          addToast({
            type: 'error',
            message: 'Connection error - retrying...',
            duration: 5000,
          });

          onError?.(error);
        },

        onMessage: (message: WebSocketMessage) => {
          console.log('WebSocket message:', message);
          onMessage?.(message);
        },

        onProgress: (data: any) => {
          console.log('Progress update:', data);
          
          // Show progress notifications for long-running tasks
          if (data.taskId && data.progress !== undefined) {
            addToast({
              type: 'info',
              message: `${data.taskName || 'Task'}: ${Math.round(data.progress)}% complete`,
              duration: 2000,
            });
          }

          onProgress?.(data);
        },

        onUpdate: (data: any) => {
          console.log('Data update:', data);
          
          // Handle different types of updates
          switch (data.type) {
            case 'paper_added':
              addToast({
                type: 'success',
                message: `New paper added: ${data.paper?.title?.substring(0, 50)}...`,
                duration: 4000,
              });
              break;
            
            case 'graph_updated':
              addToast({
                type: 'info',
                message: 'Network visualization updated',
                duration: 2000,
              });
              break;
            
            case 'analysis_complete':
              addToast({
                type: 'success',
                message: 'Analysis complete - new insights available',
                duration: 5000,
              });
              break;
          }

          onUpdate?.(data);
        },

        onNotification: (data: any) => {
          console.log('Notification:', data);
          
          // Display user notifications
          if (data.message) {
            addToast({
              type: data.level || 'info',
              message: data.message,
              duration: data.duration || 5000,
            });
          }

          onNotification?.(data);
        },

        onCollaboration: (data: any) => {
          console.log('Collaboration event:', data);
          
          // Handle collaboration events
          switch (data.event) {
            case 'user_joined':
              addToast({
                type: 'info',
                message: `${data.user?.name || 'Someone'} joined the collaboration`,
                duration: 3000,
              });
              break;
            
            case 'user_left':
              addToast({
                type: 'info',
                message: `${data.user?.name || 'Someone'} left the collaboration`,
                duration: 3000,
              });
              break;
            
            case 'selection_changed':
              // Handle collaborative selection changes
              break;
          }

          onCollaboration?.(data);
        },
      };

      wsServiceRef.current.setHandlers(handlers);

      // Attempt initial connection
      setIsConnecting(true);
      setConnectionState('connecting');

      // Cleanup function
      return () => {
        if (wsServiceRef.current) {
          wsServiceRef.current.disconnect();
          wsServiceRef.current = null;
        }
        setIsConnected(false);
        setIsConnecting(false);
        setConnectionState('disconnected');
      };

    } catch (error) {
      console.error('Failed to initialize WebSocket:', error);
      setConnectionState('error');
      
      addToast({
        type: 'error',
        message: 'Failed to initialize real-time connection',
        duration: 5000,
      });
    }
  }, [enabled, reconnect, addToast]);

  // Update connection state periodically
  useEffect(() => {
    if (!wsServiceRef.current) return;

    const interval = setInterval(() => {
      const newState = wsServiceRef.current!.getConnectionState();
      setConnectionState(newState);
      setIsConnected(newState === 'connected');
      setIsConnecting(newState === 'connecting');
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  const send = (event: string, data?: any) => {
    if (wsServiceRef.current) {
      wsServiceRef.current.emit(event, data);
    } else {
      console.warn('WebSocket not connected, cannot send event:', event);
    }
  };

  const subscribeToTask = (taskId: string) => {
    if (wsServiceRef.current) {
      wsServiceRef.current.subscribeToTask(taskId);
    }
  };

  const unsubscribeFromTask = (taskId: string) => {
    if (wsServiceRef.current) {
      wsServiceRef.current.unsubscribeFromTask(taskId);
    }
  };

  const subscribeToCollaboration = (sessionId: string) => {
    if (wsServiceRef.current) {
      wsServiceRef.current.subscribeToCollaboration(sessionId);
    }
  };

  const leaveCollaboration = (sessionId: string) => {
    if (wsServiceRef.current) {
      wsServiceRef.current.leaveCollaboration(sessionId);
    }
  };

  const requestLiveData = (endpoint: string, params?: any) => {
    if (wsServiceRef.current) {
      wsServiceRef.current.requestLiveData(endpoint, params);
    }
  };

  return {
    isConnected,
    isConnecting,
    connectionState,
    send,
    subscribeToTask,
    unsubscribeFromTask,
    subscribeToCollaboration,
    leaveCollaboration,
    requestLiveData,
  };
};

export default useWebSocket;