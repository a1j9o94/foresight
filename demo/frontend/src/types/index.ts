/** Chat message from user or assistant */
export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  imageUrl?: string;
  timestamp: Date;
  predictionId?: string;
}

/** Prediction metrics returned by the model */
export interface PredictionMetrics {
  lpips: number;
  confidence: number;
  latencyMs: number;
  spatialIou?: number;
}

/** A single video prediction */
export interface Prediction {
  id: string;
  videoUrl?: string;
  thumbnailUrl?: string;
  frames: PredictionFrame[];
  metrics?: PredictionMetrics;
  status: 'pending' | 'generating' | 'completed' | 'error';
  errorMessage?: string;
  createdAt: Date;
}

/** Single frame in a prediction timeline */
export interface PredictionFrame {
  index: number;
  timestamp: number;
  imageUrl: string;
}

/** WebSocket message types */
export type WebSocketMessageType =
  | 'text_chunk'
  | 'prediction_start'
  | 'prediction_progress'
  | 'prediction_complete'
  | 'error';

/** WebSocket message payload */
export interface WebSocketMessage {
  type: WebSocketMessageType;
  data: TextChunkData | PredictionProgressData | PredictionCompleteData | ErrorData;
}

export interface TextChunkData {
  messageId: string;
  chunk: string;
  done: boolean;
}

export interface PredictionProgressData {
  predictionId: string;
  progress: number;
  currentFrame?: number;
  totalFrames?: number;
}

export interface PredictionCompleteData {
  predictionId: string;
  videoUrl: string;
  thumbnailUrl: string;
  frames: PredictionFrame[];
  metrics: PredictionMetrics;
}

export interface ErrorData {
  message: string;
  code?: string;
}

/** Chat request to backend */
export interface ChatRequest {
  message: string;
  imageBase64?: string;
  sessionId?: string;
}

/** Chat response from backend (non-streaming) */
export interface ChatResponse {
  messageId: string;
  content: string;
  predictionId?: string;
}

/** System status */
export interface SystemStatus {
  ready: boolean;
  modelLoaded: boolean;
  vramUsageGb?: number;
  vramTotalGb?: number;
  currentModel?: string;
}

/** Demo configuration */
export interface DemoConfig {
  numFrames: number;
  fps: number;
  resolution: [number, number];
  showMetrics: boolean;
  autoPlayPredictions: boolean;
}
