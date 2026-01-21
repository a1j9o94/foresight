import { useState, useRef, useEffect } from 'react';
import { clsx } from 'clsx';
import type { PredictionFrame } from '@/types';
import { ChevronLeft, ChevronRight, Play, Pause } from 'lucide-react';

interface TimelineProps {
  frames: PredictionFrame[];
  currentIndex?: number;
  onFrameSelect?: (index: number) => void;
  className?: string;
}

export function Timeline({
  frames,
  currentIndex = 0,
  onFrameSelect,
  className,
}: TimelineProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isAutoPlaying, setIsAutoPlaying] = useState(false);
  const [internalIndex, setInternalIndex] = useState(currentIndex);

  const activeIndex = currentIndex ?? internalIndex;

  // Auto-play functionality
  useEffect(() => {
    if (!isAutoPlaying || frames.length === 0) return;

    const interval = setInterval(() => {
      setInternalIndex((prev) => {
        const next = (prev + 1) % frames.length;
        onFrameSelect?.(next);
        return next;
      });
    }, 200);

    return () => clearInterval(interval);
  }, [isAutoPlaying, frames.length, onFrameSelect]);

  // Scroll to active frame
  useEffect(() => {
    if (!containerRef.current) return;
    const activeElement = containerRef.current.children[activeIndex] as HTMLElement;
    if (activeElement) {
      activeElement.scrollIntoView({
        behavior: 'smooth',
        block: 'nearest',
        inline: 'center',
      });
    }
  }, [activeIndex]);

  const handleFrameClick = (index: number) => {
    setInternalIndex(index);
    onFrameSelect?.(index);
  };

  const handleScroll = (direction: 'left' | 'right') => {
    if (!containerRef.current) return;
    const scrollAmount = direction === 'left' ? -200 : 200;
    containerRef.current.scrollBy({ left: scrollAmount, behavior: 'smooth' });
  };

  if (frames.length === 0) {
    return (
      <div className={clsx('flex items-center justify-center h-20 text-surface-400 text-sm', className)}>
        No frames available
      </div>
    );
  }

  return (
    <div className={clsx('space-y-2', className)}>
      {/* Header with controls */}
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-surface-600">
          Timeline ({frames.length} frames)
        </span>
        <div className="flex items-center gap-1">
          <button
            onClick={() => setIsAutoPlaying(!isAutoPlaying)}
            className="btn btn-ghost p-1"
            title={isAutoPlaying ? 'Pause' : 'Play'}
          >
            {isAutoPlaying ? (
              <Pause className="w-4 h-4" />
            ) : (
              <Play className="w-4 h-4" />
            )}
          </button>
          <button
            onClick={() => handleScroll('left')}
            className="btn btn-ghost p-1"
            title="Scroll left"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>
          <button
            onClick={() => handleScroll('right')}
            className="btn btn-ghost p-1"
            title="Scroll right"
          >
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Timeline frames */}
      <div
        ref={containerRef}
        className="timeline-container"
      >
        {frames.map((frame, index) => (
          <button
            key={frame.index}
            onClick={() => handleFrameClick(index)}
            className={clsx(
              'timeline-frame',
              index === activeIndex && 'active'
            )}
            title={`t=${frame.timestamp.toFixed(2)}s`}
          >
            <img
              src={frame.imageUrl}
              alt={`Frame ${index + 1}`}
              className="w-full h-full object-cover"
            />
          </button>
        ))}
      </div>

      {/* Current frame indicator */}
      <div className="flex items-center justify-between text-xs text-surface-500">
        <span>Frame {activeIndex + 1} / {frames.length}</span>
        <span>t = {frames[activeIndex]?.timestamp.toFixed(2)}s</span>
      </div>
    </div>
  );
}

interface TimelineMiniProps {
  frames: PredictionFrame[];
  className?: string;
}

export function TimelineMini({ frames, className }: TimelineMiniProps) {
  if (frames.length === 0) return null;

  // Show first, middle, and last frames
  const indices = [
    0,
    Math.floor(frames.length / 2),
    frames.length - 1,
  ].filter((v, i, a) => a.indexOf(v) === i);

  return (
    <div className={clsx('flex gap-1', className)}>
      {indices.map((index) => (
        <div
          key={index}
          className="w-10 h-10 rounded overflow-hidden border border-surface-200"
        >
          <img
            src={frames[index].imageUrl}
            alt={`Frame ${index + 1}`}
            className="w-full h-full object-cover"
          />
        </div>
      ))}
    </div>
  );
}
