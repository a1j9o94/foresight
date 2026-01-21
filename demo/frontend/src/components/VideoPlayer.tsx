import { useRef, useEffect, useState } from 'react';
import { clsx } from 'clsx';
import { Play, Pause, Volume2, VolumeX, Maximize2 } from 'lucide-react';

interface VideoPlayerProps {
  src?: string;
  poster?: string;
  autoPlay?: boolean;
  loop?: boolean;
  className?: string;
  onEnded?: () => void;
}

export function VideoPlayer({
  src,
  poster,
  autoPlay = true,
  loop = true,
  className,
  onEnded,
}: VideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isPlaying, setIsPlaying] = useState(autoPlay);
  const [isMuted, setIsMuted] = useState(true);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handleTimeUpdate = () => {
      if (video.duration) {
        setProgress((video.currentTime / video.duration) * 100);
      }
    };

    const handleEnded = () => {
      setIsPlaying(false);
      onEnded?.();
    };

    video.addEventListener('timeupdate', handleTimeUpdate);
    video.addEventListener('ended', handleEnded);

    return () => {
      video.removeEventListener('timeupdate', handleTimeUpdate);
      video.removeEventListener('ended', handleEnded);
    };
  }, [onEnded]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !src) return;

    if (autoPlay) {
      video.play().catch(() => {
        // Autoplay failed, likely due to browser policy
        setIsPlaying(false);
      });
    }
  }, [src, autoPlay]);

  const togglePlay = () => {
    const video = videoRef.current;
    if (!video) return;

    if (isPlaying) {
      video.pause();
    } else {
      video.play();
    }
    setIsPlaying(!isPlaying);
  };

  const toggleMute = () => {
    const video = videoRef.current;
    if (!video) return;

    video.muted = !isMuted;
    setIsMuted(!isMuted);
  };

  const handleSeek = (e: React.MouseEvent<HTMLDivElement>) => {
    const video = videoRef.current;
    if (!video) return;

    const rect = e.currentTarget.getBoundingClientRect();
    const percent = (e.clientX - rect.left) / rect.width;
    video.currentTime = percent * video.duration;
  };

  const handleFullscreen = () => {
    const video = videoRef.current;
    if (!video) return;

    if (document.fullscreenElement) {
      document.exitFullscreen();
    } else {
      video.requestFullscreen();
    }
  };

  if (!src) {
    return (
      <div className={clsx('video-container flex items-center justify-center', className)}>
        <div className="text-surface-400 text-sm">No prediction yet</div>
      </div>
    );
  }

  return (
    <div className={clsx('video-container group', className)}>
      <video
        ref={videoRef}
        src={src}
        poster={poster}
        muted={isMuted}
        loop={loop}
        playsInline
        className="w-full h-full object-contain"
      />

      {/* Controls overlay */}
      <div className="absolute inset-0 flex flex-col justify-end opacity-0 group-hover:opacity-100 transition-opacity bg-gradient-to-t from-black/50 to-transparent">
        {/* Progress bar */}
        <div
          className="h-1 bg-white/30 cursor-pointer mx-3 mb-2 rounded-full overflow-hidden"
          onClick={handleSeek}
        >
          <div
            className="h-full bg-primary-500 transition-all"
            style={{ width: `${progress}%` }}
          />
        </div>

        {/* Control buttons */}
        <div className="flex items-center gap-2 px-3 pb-3">
          <button
            onClick={togglePlay}
            className="p-1.5 rounded-full hover:bg-white/20 transition-colors"
          >
            {isPlaying ? (
              <Pause className="w-4 h-4 text-white" />
            ) : (
              <Play className="w-4 h-4 text-white" />
            )}
          </button>

          <button
            onClick={toggleMute}
            className="p-1.5 rounded-full hover:bg-white/20 transition-colors"
          >
            {isMuted ? (
              <VolumeX className="w-4 h-4 text-white" />
            ) : (
              <Volume2 className="w-4 h-4 text-white" />
            )}
          </button>

          <div className="flex-1" />

          <button
            onClick={handleFullscreen}
            className="p-1.5 rounded-full hover:bg-white/20 transition-colors"
          >
            <Maximize2 className="w-4 h-4 text-white" />
          </button>
        </div>
      </div>
    </div>
  );
}
