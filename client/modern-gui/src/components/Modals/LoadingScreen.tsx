import React from 'react';
import { t } from '../../locales';
import { LoadingProgress } from '../../scripts/useModelLoadingProgress';

interface LoadingScreenProps {
  message?: string;
  progress?: LoadingProgress | null;
}

const STEP_LABELS: Record<number, string> = {
  1: 'Voice model',
  2: 'Embedder',
  3: 'Pitch extractor',
  4: 'FAISS index',
};

const LoadingScreen: React.FC<LoadingScreenProps> = ({ message, progress }) => {
  const hasProgress = progress != null && progress.total > 0;
  const pct = hasProgress ? Math.round((progress.step / progress.total) * 100) : 0;

  return (
    <div className="fixed inset-0 bg-scrim/40 backdrop-blur-sm flex justify-center items-center z-50 p-4 animate-fadeIn">
      <div className="bg-surface-container-high p-6 rounded-3xl border border-outline-variant shadow-elevation-3 flex flex-col items-center justify-center max-w-xs w-full gap-4">

        {/* Spinner — shrinks when progress is active */}
        <div
          className={`border-4 border-primary border-t-transparent rounded-full animate-spin flex-shrink-0 transition-all duration-300`}
          style={{ width: hasProgress ? 32 : 48, height: hasProgress ? 32 : 48 }}
        />

        {/* Title */}
        <p className="text-on-surface text-sm font-bold text-center">
          {t('loadingLabel')}
        </p>

        {/* Optional subtitle message (e.g. model name) */}
        {message && (
          <p className="text-on-surface-variant text-xs text-center leading-relaxed -mt-2">
            {message}
          </p>
        )}

        {/* MD3 Linear Progress Indicator */}
        {hasProgress && (
          <div className="w-full flex flex-col gap-2">
            {/* Track */}
            <div
              className="w-full overflow-hidden"
              style={{
                height: 4,
                borderRadius: 'var(--md-sys-shape-corner-full)',
                backgroundColor: 'var(--md-sys-color-surface-container-highest)',
              }}
              role="progressbar"
              aria-valuenow={pct}
              aria-valuemin={0}
              aria-valuemax={100}
            >
              {/* Indicator */}
              <div
                style={{
                  height: '100%',
                  width: `${pct}%`,
                  borderRadius: 'var(--md-sys-shape-corner-full)',
                  backgroundColor: 'var(--md-sys-color-primary)',
                  transition: `width var(--md-sys-motion-duration-long2, 350ms) var(--md-sys-motion-easing-emphasized, cubic-bezier(0.05, 0.7, 0.1, 1))`,
                }}
              />
            </div>

            {/* Step info row */}
            <div className="flex justify-between items-center">
              <span className="text-xs text-on-surface-variant leading-none">
                {STEP_LABELS[progress.step] ?? progress.label}
              </span>
              <span className="text-xs font-semibold text-primary leading-none tabular-nums">
                {progress.step} / {progress.total}
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default LoadingScreen;