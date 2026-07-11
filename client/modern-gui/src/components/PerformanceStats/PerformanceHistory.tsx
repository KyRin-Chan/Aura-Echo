import React from 'react';

interface PerformanceHistoryProps {
  maxDataPoints: number;
  setMaxDataPoints: (maxDataPoints: number) => void;
}

function PerformanceHistory({ maxDataPoints, setMaxDataPoints }: PerformanceHistoryProps) {
  return (
    <div className="flex items-center space-x-2">
      <span className="text-[10px] font-bold text-on-surface-variant uppercase tracking-wider">History:</span>
      <div className="inline-flex rounded-full border border-outline overflow-hidden bg-surface-container-low">
        {[20, 50, 100].map((num) => (
          <button
            key={num}
            onClick={() => setMaxDataPoints(num)}
            className={`px-3 py-1 text-[10px] font-semibold transition-all ${
              maxDataPoints === num
                ? 'bg-secondary-container text-on-secondary-container font-bold shadow-elevation-1'
                : 'bg-transparent text-on-surface hover:bg-surface-variant/20'
            }`}
          >
            {num}
          </button>
        ))}
      </div>
    </div>
  );
}

export default PerformanceHistory;