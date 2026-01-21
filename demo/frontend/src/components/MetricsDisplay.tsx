import { clsx } from 'clsx';
import type { PredictionMetrics } from '@/types';
import { Activity, Target, Clock, Grid } from 'lucide-react';

interface MetricsDisplayProps {
  metrics?: PredictionMetrics;
  isLoading?: boolean;
  className?: string;
}

function getMetricStatus(
  value: number,
  thresholds: { good: number; warning: number },
  higherIsBetter: boolean = false
): 'good' | 'warning' | 'bad' {
  if (higherIsBetter) {
    if (value >= thresholds.good) return 'good';
    if (value >= thresholds.warning) return 'warning';
    return 'bad';
  } else {
    if (value <= thresholds.good) return 'good';
    if (value <= thresholds.warning) return 'warning';
    return 'bad';
  }
}

export function MetricsDisplay({ metrics, isLoading = false, className }: MetricsDisplayProps) {
  if (isLoading) {
    return (
      <div className={clsx('flex flex-wrap gap-2', className)}>
        {[1, 2, 3].map((i) => (
          <div
            key={i}
            className="metric-badge neutral animate-pulse"
            style={{ width: '80px', height: '24px' }}
          />
        ))}
      </div>
    );
  }

  if (!metrics) {
    return (
      <div className={clsx('text-sm text-surface-400', className)}>
        No metrics available
      </div>
    );
  }

  const lpipsStatus = getMetricStatus(metrics.lpips, { good: 0.25, warning: 0.35 });
  const confidenceStatus = getMetricStatus(metrics.confidence, { good: 0.8, warning: 0.6 }, true);
  const latencyStatus = getMetricStatus(metrics.latencyMs, { good: 2000, warning: 3000 });

  return (
    <div className={clsx('flex flex-wrap gap-2', className)}>
      {/* LPIPS Score */}
      <div className={clsx('metric-badge', lpipsStatus)} title="LPIPS: Lower is better (perceptual similarity)">
        <Target className="w-3 h-3" />
        <span>LPIPS: {metrics.lpips.toFixed(3)}</span>
      </div>

      {/* Confidence */}
      <div className={clsx('metric-badge', confidenceStatus)} title="Model confidence">
        <Activity className="w-3 h-3" />
        <span>Conf: {(metrics.confidence * 100).toFixed(0)}%</span>
      </div>

      {/* Latency */}
      <div className={clsx('metric-badge', latencyStatus)} title="Generation time">
        <Clock className="w-3 h-3" />
        <span>{(metrics.latencyMs / 1000).toFixed(1)}s</span>
      </div>

      {/* Spatial IoU (if available) */}
      {metrics.spatialIou !== undefined && (
        <div
          className={clsx(
            'metric-badge',
            getMetricStatus(metrics.spatialIou, { good: 0.7, warning: 0.5 }, true)
          )}
          title="Spatial IoU: Higher is better"
        >
          <Grid className="w-3 h-3" />
          <span>IoU: {(metrics.spatialIou * 100).toFixed(0)}%</span>
        </div>
      )}
    </div>
  );
}

interface MetricsCardProps {
  metrics?: PredictionMetrics;
  isLoading?: boolean;
}

export function MetricsCard({ metrics, isLoading }: MetricsCardProps) {
  return (
    <div className="card p-4 space-y-3">
      <h3 className="text-sm font-medium text-surface-700">Prediction Quality</h3>
      <MetricsDisplay metrics={metrics} isLoading={isLoading} />

      {metrics && (
        <div className="pt-2 border-t border-surface-100">
          <div className="grid grid-cols-2 gap-3 text-xs">
            <div>
              <span className="text-surface-500">Perceptual Quality</span>
              <div className="mt-1 h-1.5 bg-surface-100 rounded-full overflow-hidden">
                <div
                  className={clsx(
                    'h-full rounded-full transition-all',
                    metrics.lpips <= 0.25 ? 'bg-green-500' :
                    metrics.lpips <= 0.35 ? 'bg-yellow-500' : 'bg-red-500'
                  )}
                  style={{ width: `${Math.max(0, (1 - metrics.lpips) * 100)}%` }}
                />
              </div>
            </div>
            <div>
              <span className="text-surface-500">Confidence</span>
              <div className="mt-1 h-1.5 bg-surface-100 rounded-full overflow-hidden">
                <div
                  className={clsx(
                    'h-full rounded-full transition-all',
                    metrics.confidence >= 0.8 ? 'bg-green-500' :
                    metrics.confidence >= 0.6 ? 'bg-yellow-500' : 'bg-red-500'
                  )}
                  style={{ width: `${metrics.confidence * 100}%` }}
                />
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
