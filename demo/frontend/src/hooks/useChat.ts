import { useState, useCallback, useEffect, useRef } from 'react';
import type {
  ChatMessage,
  Prediction,
  TextChunkData,
  PredictionProgressData,
  PredictionCompleteData,
  ErrorData,
  PredictionFrame,
} from '@/types';
import { useWebSocket } from './useWebSocket';

interface UseChatOptions {
  wsUrl?: string;
  onError?: (message: string) => void;
}

interface UseChatReturn {
  messages: ChatMessage[];
  currentPrediction: Prediction | undefined;
  predictionHistory: Prediction[];
  isLoading: boolean;
  isStreaming: boolean;
  streamingMessageId: string | undefined;
  isConnected: boolean;
  sendMessage: (content: string, image?: File) => Promise<void>;
  clearChat: () => void;
  selectPrediction: (prediction: Prediction) => void;
  connect: () => void;
  disconnect: () => void;
}

function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

async function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      // Remove data URL prefix (e.g., "data:image/png;base64,")
      const base64 = result.split(',')[1];
      resolve(base64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function getWebSocketUrl(path: string): string {
  const apiUrl = import.meta.env.VITE_API_URL;
  if (!apiUrl) {
    // Fall back to relative path (for Vite proxy in dev)
    return path;
  }
  // Convert HTTP URL to WebSocket URL
  const wsProtocol = apiUrl.startsWith('https') ? 'wss' : 'ws';
  const wsHost = apiUrl.replace(/^https?:\/\//, '');
  return `${wsProtocol}://${wsHost}${path}`;
}

export function useChat({
  wsUrl,
  onError,
}: UseChatOptions = {}): UseChatReturn {
  const effectiveWsUrl = wsUrl ?? getWebSocketUrl('/ws/chat');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [currentPredictionId, setCurrentPredictionId] = useState<string | undefined>();
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingMessageId, setStreamingMessageId] = useState<string | undefined>();

  const pendingMessageRef = useRef<{ id: string; content: string } | null>(null);
  // Store all images in the conversation for multi-image context
  const imagesBase64Ref = useRef<string[]>([]);

  // Handlers for WebSocket messages
  const handleTextChunk = useCallback((data: TextChunkData) => {
    if (data.done) {
      setIsStreaming(false);
      setStreamingMessageId(undefined);
      pendingMessageRef.current = null;
      // Also clear loading if there's no pending prediction
      // (i.e., for text-only responses without images)
      setIsLoading(false);
      return;
    }

    // Append text chunk to the current assistant message
    setMessages((prev) => {
      const lastMessage = prev[prev.length - 1];
      if (lastMessage?.id === data.messageId && lastMessage.role === 'assistant') {
        return [
          ...prev.slice(0, -1),
          { ...lastMessage, content: lastMessage.content + data.chunk },
        ];
      }
      return prev;
    });
  }, []);

  const handlePredictionStart = useCallback((predictionId: string) => {
    const newPrediction: Prediction = {
      id: predictionId,
      frames: [],
      status: 'generating',
      createdAt: new Date(),
    };

    setPredictions((prev) => [...prev, newPrediction]);
    setCurrentPredictionId(predictionId);

    // Link prediction to the current assistant message
    setMessages((prev) => {
      const lastMessage = prev[prev.length - 1];
      if (lastMessage?.role === 'assistant') {
        return [
          ...prev.slice(0, -1),
          { ...lastMessage, predictionId },
        ];
      }
      return prev;
    });
  }, []);

  const handlePredictionProgress = useCallback((data: PredictionProgressData) => {
    setPredictions((prev) =>
      prev.map((p) =>
        p.id === data.predictionId
          ? {
              ...p,
              status: 'generating',
              frames:
                data.currentFrame !== undefined
                  ? [
                      ...p.frames,
                      {
                        index: data.currentFrame,
                        timestamp: data.currentFrame / 15, // Assuming 15 FPS
                        imageUrl: '', // Will be filled by complete
                      } as PredictionFrame,
                    ]
                  : p.frames,
            }
          : p
      )
    );
  }, []);

  const handlePredictionComplete = useCallback((data: PredictionCompleteData) => {
    setPredictions((prev) =>
      prev.map((p) =>
        p.id === data.predictionId
          ? {
              ...p,
              status: 'completed',
              videoUrl: data.videoUrl,
              thumbnailUrl: data.thumbnailUrl,
              frames: data.frames,
              metrics: data.metrics,
            }
          : p
      )
    );
    setIsLoading(false);
  }, []);

  const handleError = useCallback(
    (data: ErrorData) => {
      setIsLoading(false);
      setIsStreaming(false);
      onError?.(data.message);

      // Mark current prediction as error
      if (currentPredictionId) {
        setPredictions((prev) =>
          prev.map((p) =>
            p.id === currentPredictionId
              ? { ...p, status: 'error', errorMessage: data.message }
              : p
          )
        );
      }
    },
    [currentPredictionId, onError]
  );

  const handleOpen = useCallback(() => {
    console.log('WebSocket connected');
  }, []);

  const handleClose = useCallback(() => {
    console.log('WebSocket disconnected');
    setIsLoading(false);
    setIsStreaming(false);
  }, []);

  const { isConnected, connect, disconnect, send } = useWebSocket({
    url: effectiveWsUrl,
    onTextChunk: handleTextChunk,
    onPredictionStart: handlePredictionStart,
    onPredictionProgress: handlePredictionProgress,
    onPredictionComplete: handlePredictionComplete,
    onError: handleError,
    onOpen: handleOpen,
    onClose: handleClose,
  });

  // Auto-connect on mount - using refs to avoid reconnection cycles
  // when callback dependencies change
  const connectRef = useRef(connect);
  const disconnectRef = useRef(disconnect);

  useEffect(() => {
    connectRef.current = connect;
    disconnectRef.current = disconnect;
  });

  useEffect(() => {
    connectRef.current();
    return () => disconnectRef.current();
  }, []);

  const sendMessage = useCallback(
    async (content: string, image?: File) => {
      if (!isConnected) {
        onError?.('Not connected to server');
        return;
      }

      // Add user message
      const userMessage: ChatMessage = {
        id: generateId(),
        role: 'user',
        content,
        imageUrl: image ? URL.createObjectURL(image) : undefined,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, userMessage]);
      setIsLoading(true);

      // Create placeholder for assistant response
      const assistantMessageId = generateId();
      const assistantMessage: ChatMessage = {
        id: assistantMessageId,
        role: 'assistant',
        content: '',
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
      setIsStreaming(true);
      setStreamingMessageId(assistantMessageId);
      pendingMessageRef.current = { id: assistantMessageId, content: '' };

      // Send via WebSocket
      const payload: { message: string; messageId: string; imageBase64?: string; imagesBase64?: string[] } = {
        message: content,
        messageId: assistantMessageId,
      };

      if (image) {
        // New image provided - add to accumulated images
        const imageBase64 = await fileToBase64(image);
        imagesBase64Ref.current = [...imagesBase64Ref.current, imageBase64];
      }

      // Send all accumulated images for multi-image context
      if (imagesBase64Ref.current.length > 0) {
        // For backwards compatibility, send most recent as imageBase64
        payload.imageBase64 = imagesBase64Ref.current[imagesBase64Ref.current.length - 1];
        // Send all images for multi-image inference
        if (imagesBase64Ref.current.length > 1) {
          payload.imagesBase64 = imagesBase64Ref.current;
        }
      }

      send(payload);
    },
    [isConnected, send, onError]
  );

  const clearChat = useCallback(() => {
    setMessages([]);
    setPredictions([]);
    setCurrentPredictionId(undefined);
    setIsLoading(false);
    setIsStreaming(false);
    setStreamingMessageId(undefined);
    imagesBase64Ref.current = [];
    pendingMessageRef.current = null;
  }, []);

  const selectPrediction = useCallback((prediction: Prediction) => {
    setCurrentPredictionId(prediction.id);
  }, []);

  // Derived state
  const currentPrediction = predictions.find((p) => p.id === currentPredictionId);
  const predictionHistory = predictions.filter((p) => p.status === 'completed');

  return {
    messages,
    currentPrediction,
    predictionHistory,
    isLoading,
    isStreaming,
    streamingMessageId,
    isConnected,
    sendMessage,
    clearChat,
    selectPrediction,
    connect,
    disconnect,
  };
}
