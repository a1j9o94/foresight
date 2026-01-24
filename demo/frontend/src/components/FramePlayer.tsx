import { useState, useEffect, useRef, useCallback } from 'react';
import { clsx } from 'clsx';
import { Play, Pause } from 'lucide-react';
import type { PredictionFrame } from '@/types';

interface FramePlayerProps {
  frames: PredictionFrame[];
  fps?: number;
  autoPlay?: boolean;
  loop?: boolean;
  className?: string;
  onFrameChange?: (index: number) => void;
}

export function FramePlayer({
  frames,
  fps = 15,
  autoPlay = true,
  loop = true,
  className,
  onFrameChange,
}: FramePlayerProps) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(autoPlay);
  const intervalRef = useRef<number | null>(null);

  const frameInterval = 1000 / fps;

  const startPlayback = useCallback(() => {
    if (intervalRef.current) return;
    if (frames.length === 0) return;

    intervalRef.current = window.setInterval(() => {
      setCurrentIndex((prev) => {
        const next = prev + 1;
        if (next >= frames.length) {
          if (loop) {
            return 0;
          } else {
            // Stop at last frame
            if (intervalRef.current) {
              clearInterval(intervalRef.current);
              intervalRef.current = null;
            }
            setIsPlaying(false);
            return prev;
          }
        }
        return next;
      });
    }, frameInterval);
  }, [frames.length, frameInterval, loop]);

  const stopPlayback = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  // Start/stop based on isPlaying state
  useEffect(() => {
    if (isPlaying && frames.length > 0) {
      startPlayback();
    } else {
      stopPlayback();
    }

    return stopPlayback;
  }, [isPlaying, frames.length, startPlayback, stopPlayback]);

  // Auto-start when frames are available
  useEffect(() => {
    if (autoPlay && frames.length > 0 && !isPlaying) {
      setIsPlaying(true);
    }
  }, [autoPlay, frames.length, isPlaying]);

  // Reset when frames change
  useEffect(() => {
    setCurrentIndex(0);
  }, [frames]);

  // Notify parent of frame changes
  useEffect(() => {
    onFrameChange?.(currentIndex);
  }, [currentIndex, onFrameChange]);

  const togglePlay = () => {
    setIsPlaying(!isPlaying);
  };

  const currentFrame = frames[currentIndex];

  if (frames.length === 0) {
    return (
      <div className={clsx('video-container flex items-center justify-center', className)}>
        <div className="text-surface-400 text-sm">No prediction yet</div>
      </div>
    );
  }

  return (
    <div className={clsx('video-container group relative', className)}>
      {/* Current frame */}
      <img
        src={currentFrame?.imageUrl}
        alt={`Frame ${currentIndex + 1}`}
        className="w-full h-full object-contain"
      />

      {/* Play/Pause overlay */}
      <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
        <button
          onClick={togglePlay}
          className="p-3 rounded-full bg-black/50 hover:bg-black/70 transition-colors"
        >
          {isPlaying ? (
            <Pause className="w-6 h-6 text-white" />
          ) : (
            <Play className="w-6 h-6 text-white" />
          )}
        </button>
      </div>

      {/* Progress bar at bottom */}
      <div className="absolute bottom-0 left-0 right-0 h-1 bg-black/30">
        <div
          className="h-full bg-primary-500 transition-all duration-75"
          style={{ width: `${((currentIndex + 1) / frames.length) * 100}%` }}
        />
      </div>

      {/* Frame counter */}
      <div className="absolute bottom-2 right-2 px-2 py-1 rounded bg-black/50 text-white text-xs">
        {currentIndex + 1} / {frames.length}
      </div>
    </div>
  );
}
