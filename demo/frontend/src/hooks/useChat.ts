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

export function useChat({
  wsUrl = '/ws/chat',
  onError,
}: UseChatOptions = {}): UseChatReturn {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [currentPredictionId, setCurrentPredictionId] = useState<string | undefined>();
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingMessageId, setStreamingMessageId] = useState<string | undefined>();

  const pendingMessageRef = useRef<{ id: string; content: string } | null>(null);

  // Handlers for WebSocket messages
  const handleTextChunk = useCallback((data: TextChunkData) => {
    if (data.done) {
      setIsStreaming(false);
      setStreamingMessageId(undefined);
      pendingMessageRef.current = null;
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
    url: wsUrl,
    onTextChunk: handleTextChunk,
    onPredictionStart: handlePredictionStart,
    onPredictionProgress: handlePredictionProgress,
    onPredictionComplete: handlePredictionComplete,
    onError: handleError,
    onOpen: handleOpen,
    onClose: handleClose,
  });

  // Auto-connect on mount
  useEffect(() => {
    connect();
    return () => disconnect();
  }, [connect, disconnect]);

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
      const payload: { message: string; messageId: string; imageBase64?: string } = {
        message: content,
        messageId: assistantMessageId,
      };

      if (image) {
        payload.imageBase64 = await fileToBase64(image);
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
