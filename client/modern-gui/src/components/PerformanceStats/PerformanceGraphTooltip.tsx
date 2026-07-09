import { PerfStatus } from './PerformanceStatsCard';

const STATUS_LABELS: Record<PerfStatus, { text: string; color: string; bg: string }> = {
  good: { text: 'Good', color: '#10b981', bg: 'rgba(16, 185, 129, 0.15)' },
  warning: { text: 'Warning', color: '#fbbf24', bg: 'rgba(251, 191, 36, 0.15)' },
  critical: { text: 'Critical', color: '#f43f5e', bg: 'rgba(244, 63, 94, 0.15)' },
};

function CustomTooltip({ active, payload, label }: any) {
  if (active && payload && payload.length) {
    const mainProcessData = payload.find((entry: any) => entry.dataKey === 'perfTimeValue');
    const chunkData = payload.find((entry: any) => entry.dataKey === 'chunkTime');

    if (mainProcessData) {
      const dataPoint = mainProcessData.payload;
      const status: PerfStatus = dataPoint.status || 'good';
      const statusConfig = STATUS_LABELS[status];

      return (
        <div className="bg-[var(--bg-secondary)] border border-[var(--border-primary)] rounded-2xl p-3.5 shadow-xl backdrop-blur-md flex flex-col gap-1.5 min-w-[200px]">
          <div className="flex justify-between items-center pb-1.5 border-b border-[var(--border-primary)] mb-1">
            <span className="text-[var(--text-secondary)] text-xs font-medium">
              {new Date(label).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
            </span>
            <span 
              className="text-xs px-2.5 py-0.5 rounded-full font-semibold"
              style={{ color: statusConfig.color, backgroundColor: statusConfig.bg }}
            >
              {statusConfig.text}
            </span>
          </div>
          <div className="flex justify-between text-xs">
            <span className="text-[var(--text-secondary)]">Process Time:</span>
            <span className="text-[var(--text-primary)] font-semibold">{Math.round(mainProcessData.value)} ms</span>
          </div>
          {chunkData && (
            <div className="flex justify-between text-xs">
              <span className="text-[var(--text-secondary)]">Chunk Limit:</span>
              <span className="text-[var(--text-tertiary)] font-medium">{Math.round(chunkData.value)} ms</span>
            </div>
          )}
        </div>
      );
    }
  }
  return null;
}

export default CustomTooltip;
