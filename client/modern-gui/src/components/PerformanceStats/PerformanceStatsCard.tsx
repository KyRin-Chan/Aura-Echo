import { JSX, useState, useEffect, useMemo, useRef } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faChevronUp, faChevronDown } from '@fortawesome/free-solid-svg-icons';
import DragHandle from './../Helpers/DragHandle';
import { useAppState, usePerformance } from '../../context/AppContext';
import PerformanceRecording from './PerformanceRecording';
import PerformanceStats from './PerformanceStats';
import PerformanceHistory from './PerformanceHistory';
import PerformanceGraph from './PerformanceGraph';
import { CSS_CLASSES } from '../../styles/constants';
import { t } from '../../locales';

// Define types for the performance data
interface PerformanceMetrics {
  vol: number;
  responseTime: number;
  mainprocessTime: number;
}

interface PerformanceStatsCardProps {
  dndAttributes?: Record<string, any>;
  dndListeners?: Record<string, any>;
}

export interface ChartDataPoint {
  timestamp: number;
  perfTimeValue: number;
  chunkTime: number;
  status?: PerfStatus;
  greenTime?: number;
  yellowTime?: number;
  redTime?: number;
}

export type PerfStatus = 'good' | 'warning' | 'critical';

export interface CalculatedMetricValues {
  volumeDb: number;
  ping: number;
  totalLatencyTime: number;
  perfTime: number;
  chunkTime: number;
  perfStatus: PerfStatus;
}

interface RecordedDataEntry extends CalculatedMetricValues {
  timestamp: number;
  perfValueString: string;
}

// Default max data points for the chart
const DEFAULT_MAX_CHART_DATA_POINTS = 50;

function PerformanceStatsCard({ dndAttributes, dndListeners }: PerformanceStatsCardProps): JSX.Element {
  // ---------------- States ----------------
  const appState = useAppState();
  const performance = usePerformance();

  const [isCollapsed, setIsCollapsed] = useState(false);

  const [chartData, setChartData] = useState<ChartDataPoint[]>([]);
  const [maxDataPoints, setMaxDataPoints] = useState(DEFAULT_MAX_CHART_DATA_POINTS);
  const [isRecording, setIsRecording] = useState(false);
  const [recordedData, setRecordedData] = useState<RecordedDataEntry[]>([]);

  // ---------------- Hooks ----------------

  //Calculate the next DataPoint using performance values
  const calculatedMetrics = useMemo((): CalculatedMetricValues => {
    const serverSetting = appState.serverSetting.serverSetting;
    const volumeDb = Math.max(Math.round(20 * Math.log10(performance.vol || 0.00001)), -90);
    const chunkTime = (serverSetting.serverReadChunkSize * 128 * 1000) / 48000; // Assuming 48kHz sample rate, 128 samples per frame for chunk size unit
    const ping = Math.max(0, performance.responseTime || 0);
    const crossfadeTime = Math.max(0, (serverSetting.crossFadeOverlapSize ?? 0) * 1000);
    const lookaheadTime = Math.max(0, (serverSetting.lookaheadSize ?? 0) * 1000);
    const totalLatencyTime = Math.ceil(chunkTime + ping + crossfadeTime + lookaheadTime);

    let perfStatus: PerfStatus = 'good';
    const eightyPercentChunkTime = 0.8 * chunkTime;
    if (performance.mainprocessTime > chunkTime) {
      perfStatus = 'critical';
    } else if (performance.mainprocessTime > eightyPercentChunkTime) { // totalLatencyTime <= chunkTime is implicit
      perfStatus = 'warning';
    }

    return {
      volumeDb,
      ping,
      totalLatencyTime,
      perfTime: performance.mainprocessTime,
      chunkTime,
      perfStatus,
    };
  }, [
    performance,
    appState.serverSetting.serverSetting.serverReadChunkSize,
    appState.serverSetting.serverSetting.crossFadeOverlapSize,
    appState.serverSetting.serverSetting.lookaheadSize,
  ]);

  const lastChartUpdateTimeRef = useRef<number>(0);
  const CHART_UPDATE_INTERVAL_MS = 1000; // Throttle updates to once per second

  //Format DataPoint into ChartDataPoint and add to ChartData
  useEffect(() => {
    if (isCollapsed && !isRecording) {
      return;
    }

    const now = Date.now();
    if (now - lastChartUpdateTimeRef.current < CHART_UPDATE_INTERVAL_MS) {
      return;
    }
    lastChartUpdateTimeRef.current = now;

    const { perfStatus, perfTime } = calculatedMetrics;

    const roundedPerfTime = Math.round(perfTime);
    let gt = 0, yt = 0, rt = 0;

    if (perfStatus === 'good') gt = roundedPerfTime;
    else if (perfStatus === 'warning') yt = roundedPerfTime;
    else if (perfStatus === 'critical') rt = roundedPerfTime;

    const newDataPoint: ChartDataPoint = {
      timestamp: now,
      perfTimeValue: roundedPerfTime,
      chunkTime: Math.round(calculatedMetrics.chunkTime),
      status: perfStatus,
      greenTime: gt, yellowTime: yt, redTime: rt,
    };

    //Add DataPoint to ChartData only when graph is expanded
    if (!isCollapsed) {
      setChartData(prevData => {
        const newChartData = [...prevData, newDataPoint];
        return newChartData.length > maxDataPoints ? newChartData.slice(newChartData.length - maxDataPoints) : newChartData;
      });
    }

    //Add DataPoint to RecordedData
    if (isRecording) {
      setRecordedData(prev => [...prev, {
        timestamp: newDataPoint.timestamp,
        ...calculatedMetrics,
        perfValueString: `${Math.round(calculatedMetrics.totalLatencyTime)}ms of ${Math.round(calculatedMetrics.chunkTime)}ms`
      }]);
    }
  }, [calculatedMetrics, maxDataPoints, isRecording, isCollapsed]);

  // ---------------- Render ----------------

  return (
    <div
      className={`${CSS_CLASSES.card} flex-1 min-h-0 flex flex-col ${isCollapsed ? 'h-auto' : ''}`}
    >
      <div className={CSS_CLASSES.cardHeader}>
        <h5 className={CSS_CLASSES.heading}>{t('performanceStatsTitle')}</h5>
        <div className="flex space-x-1 items-center">
          <button onClick={() => setIsCollapsed(!isCollapsed)} className={CSS_CLASSES.iconButton} title={isCollapsed ? t('expandLabel') : t('collapseLabel')}>
            <FontAwesomeIcon icon={isCollapsed ? faChevronDown : faChevronUp} className="h-5 w-5" />
          </button>
          <DragHandle attributes={dndAttributes} listeners={dndListeners} title={t('dragLabel')} />
        </div>
      </div>
      {!isCollapsed && (
        <div className="flex-grow flex flex-col items-center">
          <PerformanceStats
            calculatedMetrics={calculatedMetrics}
          />

          <div className="flex justify-between items-center mb-2 w-full px-1">
            <PerformanceRecording
              isRecording={isRecording}
              setIsRecording={setIsRecording}
              recordedData={recordedData}
              setRecordedData={setRecordedData}
            />

            <PerformanceHistory
              maxDataPoints={maxDataPoints}
              setMaxDataPoints={setMaxDataPoints}
            />
          </div>

          <div className="w-full h-48 rounded-md flex-grow relative overflow-hidden bg-surface-container border border-outline-variant">
            <PerformanceGraph
              chartData={chartData}
            />
          </div>
        </div>
      )}
    </div>
  );
}

export default PerformanceStatsCard;