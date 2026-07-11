import { PerfStatus } from './PerformanceStatsCard';
import { t } from '../../locales';

const STATUS_LABELS: Record<PerfStatus, { text: string; color: string; bg: string }> = {
  good: { text: t('statusGood'), color: '#10b981', bg: 'rgba(16, 185, 129, 0.15)' },
  warning: { text: t('statusWarning'), color: '#fbbf24', bg: 'rgba(251, 191, 36, 0.15)' },
  critical: { text: t('statusCritical'), color: '#f43f5e', bg: 'rgba(244, 63, 94, 0.15)' },
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
        <div className="bg-[var(--md-sys-color-surface-container-high)] border border-[var(--md-sys-color-outline-variant)] rounded-2xl p-3.5 shadow-xl backdrop-blur-md flex flex-col gap-1.5 min-w-[200px]">
          <div className="flex justify-between items-center pb-1.5 border-b border-[var(--md-sys-color-outline-variant)] mb-1">
            <span className="text-[var(--md-sys-color-on-surface-variant)] text-xs font-medium">
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
            <span className="text-[var(--md-sys-color-on-surface-variant)]">{t('processTimeLabel')}</span>
            <span className="text-[var(--md-sys-color-on-surface)] font-semibold">{Math.round(mainProcessData.value)} ms</span>
          </div>
          {chunkData && (
            <div className="flex justify-between text-xs">
              <span className="text-[var(--md-sys-color-on-surface-variant)]">{t('chunkLimitLabel')}</span>
              <span className="text-[var(--md-sys-color-outline)] font-medium">{Math.round(chunkData.value)} ms</span>
            </div>
          )}
        </div>
      );
    }
  }
  return null;
}

export default CustomTooltip;
