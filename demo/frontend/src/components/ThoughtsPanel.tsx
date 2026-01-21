import { useState } from 'react';
import { clsx } from 'clsx';
import type { Prediction, PredictionFrame } from '@/types';
import { VideoPlayer } from './VideoPlayer';
import { MetricsCard } from './MetricsDisplay';
import { Timeline } from './Timeline';
import { Eye, Layers, History, ChevronDown, ChevronUp, Loader2 } from 'lucide-react';

interface ThoughtsPanelProps {
  currentPrediction?: Prediction;
  predictionHistory?: Prediction[];
  onSelectPrediction?: (prediction: Prediction) => void;
  className?: string;
}

export function ThoughtsPanel({
  currentPrediction,
  predictionHistory = [],
  onSelectPrediction,
  className,
}: ThoughtsPanelProps) {
  const [selectedFrameIndex, setSelectedFrameIndex] = useState(0);
  const [showHistory, setShowHistory] = useState(false);

  const isGenerating = currentPrediction?.status === 'generating' || currentPrediction?.status === 'pending';
  const hasError = currentPrediction?.status === 'error';

  return (
    <div className={clsx('panel bg-surface-50', className)}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-surface-200 bg-white">
        <div className="flex items-center gap-2">
          <Eye className="w-5 h-5 text-primary-600" />
          <div>
            <h2 className="text-lg font-semibold text-surface-900">Thoughts</h2>
            <p className="text-xs text-surface-500">Visual predictions</p>
          </div>
        </div>
        {predictionHistory.length > 0 && (
          <button
            onClick={() => setShowHistory(!showHistory)}
            className="btn btn-ghost gap-1 text-sm"
          >
            <History className="w-4 h-4" />
            History ({predictionHistory.length})
            {showHistory ? (
              <ChevronUp className="w-4 h-4" />
            ) : (
              <ChevronDown className="w-4 h-4" />
            )}
          </button>
        )}
      </div>

      {/* History dropdown */}
      {showHistory && predictionHistory.length > 0 && (
        <div className="border-b border-surface-200 bg-white p-3 max-h-40 overflow-y-auto">
          <div className="space-y-2">
            {predictionHistory.map((prediction, index) => (
              <button
                key={prediction.id}
                onClick={() => onSelectPrediction?.(prediction)}
                className={clsx(
                  'w-full flex items-center gap-3 p-2 rounded-lg text-left transition-colors',
                  currentPrediction?.id === prediction.id
                    ? 'bg-primary-50 border border-primary-200'
                    : 'hover:bg-surface-100'
                )}
              >
                {prediction.thumbnailUrl && (
                  <img
                    src={prediction.thumbnailUrl}
                    alt={`Prediction ${index + 1}`}
                    className="w-12 h-8 object-cover rounded"
                  />
                )}
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-surface-700 truncate">
                    Prediction #{predictionHistory.length - index}
                  </p>
                  <p className="text-xs text-surface-500">
                    {prediction.createdAt.toLocaleTimeString()}
                  </p>
                </div>
                {prediction.metrics && (
                  <span className="text-xs text-surface-500">
                    LPIPS: {prediction.metrics.lpips.toFixed(2)}
                  </span>
                )}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Main content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {!currentPrediction ? (
          <EmptyThoughtsState />
        ) : hasError ? (
          <ErrorState message={currentPrediction.errorMessage} />
        ) : (
          <>
            {/* Main video display */}
            <div className="card overflow-hidden">
              <div className="p-2 bg-surface-50 border-b border-surface-100 flex items-center gap-2">
                <Layers className="w-4 h-4 text-surface-500" />
                <span className="text-xs font-medium text-surface-600">Current Prediction</span>
                {isGenerating && (
                  <span className="flex items-center gap-1 text-xs text-primary-600">
                    <Loader2 className="w-3 h-3 animate-spin" />
                    Generating...
                  </span>
                )}
              </div>
              <VideoPlayer
                src={currentPrediction.videoUrl}
                poster={currentPrediction.thumbnailUrl}
                autoPlay={!isGenerating}
              />
              {isGenerating && (
                <GeneratingOverlay progress={getProgressPercent(currentPrediction)} />
              )}
            </div>

            {/* Metrics */}
            <MetricsCard
              metrics={currentPrediction.metrics}
              isLoading={isGenerating}
            />

            {/* Timeline */}
            {currentPrediction.frames.length > 0 && (
              <div className="card p-4">
                <Timeline
                  frames={currentPrediction.frames}
                  currentIndex={selectedFrameIndex}
                  onFrameSelect={setSelectedFrameIndex}
                />
              </div>
            )}

            {/* Before/After comparison */}
            {currentPrediction.frames.length > 0 && (
              <BeforeAfterComparison
                frames={currentPrediction.frames}
                currentIndex={selectedFrameIndex}
              />
            )}
          </>
        )}
      </div>
    </div>
  );
}

function EmptyThoughtsState() {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center px-6">
      <div className="w-20 h-20 rounded-full bg-surface-100 flex items-center justify-center mb-4">
        <Eye className="w-10 h-10 text-surface-300" />
      </div>
      <h3 className="text-lg font-medium text-surface-600 mb-2">
        No predictions yet
      </h3>
      <p className="text-sm text-surface-400 max-w-xs">
        Start a conversation to see the AI's visual predictions appear here.
      </p>
    </div>
  );
}

function ErrorState({ message }: { message?: string }) {
  return (
    <div className="flex flex-col items-center justify-center h-64 text-center px-6">
      <div className="w-16 h-16 rounded-full bg-red-50 flex items-center justify-center mb-4">
        <svg
          className="w-8 h-8 text-red-500"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
          />
        </svg>
      </div>
      <h3 className="text-lg font-medium text-red-700 mb-2">
        Prediction failed
      </h3>
      <p className="text-sm text-surface-500 max-w-xs">
        {message || 'An error occurred while generating the prediction. Please try again.'}
      </p>
    </div>
  );
}

function GeneratingOverlay({ progress }: { progress: number }) {
  return (
    <div className="absolute inset-0 bg-black/50 flex flex-col items-center justify-center">
      <Loader2 className="w-8 h-8 text-white animate-spin mb-3" />
      <p className="text-white text-sm font-medium">Generating prediction...</p>
      <div className="w-48 h-1.5 bg-white/30 rounded-full mt-3 overflow-hidden">
        <div
          className="h-full bg-primary-400 rounded-full transition-all duration-300"
          style={{ width: `${progress}%` }}
        />
      </div>
      <p className="text-white/70 text-xs mt-2">{progress}%</p>
    </div>
  );
}

interface BeforeAfterComparisonProps {
  frames: PredictionFrame[];
  currentIndex: number;
}

function BeforeAfterComparison({ frames, currentIndex }: BeforeAfterComparisonProps) {
  if (frames.length < 2) return null;

  const firstFrame = frames[0];
  const currentFrame = frames[currentIndex] || frames[frames.length - 1];

  return (
    <div className="card p-4">
      <h3 className="text-sm font-medium text-surface-700 mb-3">Before / After</h3>
      <div className="grid grid-cols-2 gap-3">
        <div>
          <p className="text-xs text-surface-500 mb-2">Input (t=0)</p>
          <div className="aspect-video rounded-lg overflow-hidden border border-surface-200">
            <img
              src={firstFrame.imageUrl}
              alt="Input frame"
              className="w-full h-full object-cover"
            />
          </div>
        </div>
        <div>
          <p className="text-xs text-surface-500 mb-2">
            Predicted (t={currentFrame.timestamp.toFixed(1)}s)
          </p>
          <div className="aspect-video rounded-lg overflow-hidden border border-surface-200">
            <img
              src={currentFrame.imageUrl}
              alt="Predicted frame"
              className="w-full h-full object-cover"
            />
          </div>
        </div>
      </div>
    </div>
  );
}

function getProgressPercent(prediction: Prediction): number {
  if (prediction.status === 'completed') return 100;
  if (prediction.status === 'pending') return 0;
  // Estimate based on frames if available
  if (prediction.frames.length > 0) {
    return Math.min(95, (prediction.frames.length / 30) * 100);
  }
  return 30; // Default progress for generating state
}
