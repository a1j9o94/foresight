import { useRef, useState, useCallback, useEffect } from 'react';
import type {
  WebSocketMessage,
  TextChunkData,
  PredictionProgressData,
  PredictionCompleteData,
  ErrorData,
} from '@/types';

interface UseWebSocketOptions {
  url: string;
  onTextChunk?: (data: TextChunkData) => void;
  onPredictionStart?: (predictionId: string) => void;
  onPredictionProgress?: (data: PredictionProgressData) => void;
  onPredictionComplete?: (data: PredictionCompleteData) => void;
  onError?: (data: ErrorData) => void;
  onOpen?: () => void;
  onClose?: () => void;
  autoReconnect?: boolean;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
}

interface UseWebSocketReturn {
  isConnected: boolean;
  isConnecting: boolean;
  connect: () => void;
  disconnect: () => void;
  send: (message: object) => void;
  lastError: string | null;
}

export function useWebSocket({
  url,
  onTextChunk,
  onPredictionStart,
  onPredictionProgress,
  onPredictionComplete,
  onError,
  onOpen,
  onClose,
  autoReconnect = true,
  reconnectInterval = 3000,
  maxReconnectAttempts = 5,
}: UseWebSocketOptions): UseWebSocketReturn {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimeoutRef = useRef<number | null>(null);

  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [lastError, setLastError] = useState<string | null>(null);

  const handleMessage = useCallback(
    (event: MessageEvent) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);

        switch (message.type) {
          case 'text_chunk':
            onTextChunk?.(message.data as TextChunkData);
            break;
          case 'prediction_start':
            onPredictionStart?.((message.data as { predictionId: string }).predictionId);
            break;
          case 'prediction_progress':
            onPredictionProgress?.(message.data as PredictionProgressData);
            break;
          case 'prediction_complete':
            onPredictionComplete?.(message.data as PredictionCompleteData);
            break;
          case 'error':
            const errorData = message.data as ErrorData;
            setLastError(errorData.message);
            onError?.(errorData);
            break;
          default:
            console.warn('Unknown WebSocket message type:', message.type);
        }
      } catch (err) {
        console.error('Failed to parse WebSocket message:', err);
      }
    },
    [onTextChunk, onPredictionStart, onPredictionProgress, onPredictionComplete, onError]
  );

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    setIsConnecting(true);
    setLastError(null);

    try {
      // Build full WebSocket URL
      const wsUrl = url.startsWith('ws')
        ? url
        : `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}${url}`;

      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        setIsConnected(true);
        setIsConnecting(false);
        reconnectAttemptsRef.current = 0;
        onOpen?.();
      };

      ws.onclose = () => {
        setIsConnected(false);
        setIsConnecting(false);
        wsRef.current = null;
        onClose?.();

        // Auto-reconnect logic
        if (autoReconnect && reconnectAttemptsRef.current < maxReconnectAttempts) {
          reconnectAttemptsRef.current += 1;
          console.log(
            `WebSocket closed. Reconnecting attempt ${reconnectAttemptsRef.current}/${maxReconnectAttempts}...`
          );
          reconnectTimeoutRef.current = window.setTimeout(() => {
            connect();
          }, reconnectInterval);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setLastError('Connection error');
        setIsConnecting(false);
      };

      ws.onmessage = handleMessage;

      wsRef.current = ws;
    } catch (err) {
      setIsConnecting(false);
      setLastError(err instanceof Error ? err.message : 'Failed to connect');
    }
  }, [url, handleMessage, autoReconnect, reconnectInterval, maxReconnectAttempts, onOpen, onClose]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    reconnectAttemptsRef.current = maxReconnectAttempts; // Prevent auto-reconnect

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setIsConnected(false);
    setIsConnecting(false);
  }, [maxReconnectAttempts]);

  const send = useCallback((message: object) => {
    if (wsRef.current?.readyState !== WebSocket.OPEN) {
      console.error('WebSocket is not connected');
      return;
    }

    wsRef.current.send(JSON.stringify(message));
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  return {
    isConnected,
    isConnecting,
    connect,
    disconnect,
    send,
    lastError,
  };
}
