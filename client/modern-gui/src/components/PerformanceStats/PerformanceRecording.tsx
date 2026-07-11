import { useCallback } from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faPlayCircle, faStopCircle } from "@fortawesome/free-solid-svg-icons";
import { t } from '../../locales';

interface PerformanceRecordingProps {
  isRecording: boolean;
  setIsRecording: (isRecording: boolean) => void;
  recordedData: any[];
  setRecordedData: (recordedData: any[]) => void;
}

function PerformanceRecording({
  isRecording,
  setIsRecording,
  recordedData,
  setRecordedData
}: PerformanceRecordingProps) {
  // ---------------- Functions ----------------

  // Download log file
  const downloadLogFile = useCallback(() => {
    if (recordedData.length === 0) return;

    const header = "Timestamp,DateTime,Volume_dB,Ping_ms,TotalLatency_ms,PerfValue_ms,PerfChunk_ms,PerfStatus\n";
    const logContent = recordedData
      .map(
        (entry) =>
          `${entry.timestamp},${new Date(entry.timestamp).toISOString()},${entry.volumeDb},${Math.round(
            entry.ping
          )},${Math.round(entry.totalLatencyTime)},${Math.round(entry.perfTime)},${Math.round(
            entry.chunkTime
          )},${entry.perfStatus}`
      )
      .join("\n");

    const blob = new Blob([header + logContent], { type: 'text/csv;charset=utf-8' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `performance_log_${new Date().toISOString().replace(/[:.]/g, '-')}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(link.href);
    setRecordedData([]); // Clear data after download
  }, [recordedData, setRecordedData]);

  // Handle record toggle
  const handleRecordToggle = () => {
    if (isRecording) {
      downloadLogFile();
    }
    setIsRecording(!isRecording);
  };

  // ---------------- Render ----------------
  return (
    <button
      onClick={handleRecordToggle}
      className={`px-3 py-1 rounded-full text-[10px] font-semibold flex items-center space-x-1.5 transition-all duration-150 active:scale-97 hover:shadow-elevation-1 ${
        isRecording ? 'bg-error text-on-error' : 'bg-primary text-on-primary'
      }`}
      title={isRecording ? t('stopPerfRecordingTooltip') : t('startPerfRecordingTooltip')}
    >
      <FontAwesomeIcon icon={isRecording ? faStopCircle : faPlayCircle} className="h-3 w-3" />
      <span>{isRecording ? t('btnStop') : t('btnRecord')}</span>
    </button>
  );
}

export default PerformanceRecording;