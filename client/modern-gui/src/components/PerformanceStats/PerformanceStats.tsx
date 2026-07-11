import { CalculatedMetricValues, PerfStatus } from "./PerformanceStatsCard";
import { t } from '../../locales';

interface PerformanceStatsProps {
  calculatedMetrics: CalculatedMetricValues;
}

const PERF_TEXT_CLASSES: Record<PerfStatus, string> = {
  good: 'text-primary font-bold',
  warning: 'text-yellow-500 font-bold',
  critical: 'text-error font-bold'
};

function PerformanceStats({ calculatedMetrics }: PerformanceStatsProps) {
  // ---------------- States ----------------
  const performanceMetricKeys: string[] = [t('volLabel'), t('pingLabel'), t('totalLabel'), t('perfLabel')];

  // Calculate display values from metrics
  const displayValues: Record<string, { value: string | number; unit?: string; className?: string }> = {
    [t('volLabel')]: { value: calculatedMetrics.volumeDb, unit: ' dB' },
    [t('pingLabel')]: { value: Math.round(calculatedMetrics.ping), unit: ' ms' },
    [t('totalLabel')]: { value: Math.round(calculatedMetrics.totalLatencyTime), unit: ' ms' },
    [t('perfLabel')]: {
      value: `${Math.round(calculatedMetrics.perfTime)}ms / ${Math.round(calculatedMetrics.chunkTime)}ms`,
      className: PERF_TEXT_CLASSES[calculatedMetrics.perfStatus]
    }
  };

  // ---------------- Render ----------------

  return (
    <div className="flex justify-around w-full mb-3.5 bg-surface-container-low p-2 rounded-lg border border-outline-variant/30">
      {performanceMetricKeys.map((metricKey) => {
        const metricInfo = displayValues[metricKey];
        return (
          <span key={metricKey} className="text-[10px] font-semibold text-on-surface-variant uppercase tracking-wider">
            {metricKey}:{' '}
            <span className={`text-on-surface ${metricInfo.className || ''}`}>
              {metricInfo.value}
              {metricInfo.unit || ''}
            </span>
          </span>
        );
      })}
    </div>
  );
}

export default PerformanceStats;