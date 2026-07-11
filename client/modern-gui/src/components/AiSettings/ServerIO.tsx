import { JSX, useState, useRef, useEffect } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faPlay,
  faStop,
  faMicrophone,
  faVolumeUp,
  faChevronUp,
  faChevronDown
} from '@fortawesome/free-solid-svg-icons';
import { AUDIO_KEYS, CSS_CLASSES } from '../../styles/constants';
import { ClientState } from "@dannadori/voice-changer-client-js";
import AudioPlayer from '../Helpers/AudioPlayer';
import { useUIContext } from '../../context/UIContext';
import MD3Select from '../Helpers/MD3Select';
import { t } from '../../locales';

interface ServerIOProps {
  appState: ClientState;
}

function ServerIO({ appState }: ServerIOProps): JSX.Element {
  // ---------------- States ----------------
  const [isRecording, setIsRecording] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [isCollapsed, setIsCollapsed] = useState(false);
  const {
    audioOutputForAnalyzer,
    setAudioOutputForAnalyzer,
    outputAudioDeviceInfo,
    isConverting
  } = useUIContext();
  const [selectedOutputDevice, setSelectedOutputDevice] = useState<string>('');
  const recordingStartTimeRef = useRef<number | null>(null);
  const durationIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // ---------------- Hooks ----------------

  // Auto-stop recording when isConverting becomes false
  useEffect(() => {
    if (!isConverting && isRecording) {
      onServerIORecordStop();
    }
  }, [isConverting, isRecording]);

  // Recording duration timer
  useEffect(() => {
    if (isRecording) {
      recordingStartTimeRef.current = Date.now();
      durationIntervalRef.current = setInterval(() => {
        if (recordingStartTimeRef.current) {
          const elapsed = Math.floor((Date.now() - recordingStartTimeRef.current) / 1000);
          setRecordingDuration(elapsed);
        }
      }, 1000);
    } else {
      if (durationIntervalRef.current) {
        clearInterval(durationIntervalRef.current);
        durationIntervalRef.current = null;
      }
      setRecordingDuration(0);
      recordingStartTimeRef.current = null;
    }

    return () => {
      if (durationIntervalRef.current) {
        clearInterval(durationIntervalRef.current);
      }
    };
  }, [isRecording]);

  // Set selected output device
  useEffect(() => {
    if (audioOutputForAnalyzer) {
      setSelectedOutputDevice(audioOutputForAnalyzer);
    }
  }, [audioOutputForAnalyzer]);

  // ---------------- Functions ----------------

  // Record start
  const onServerIORecordStart = async () => {
    appState.startOutputRecording();
    setIsRecording(true);
  };

  // Record stop
  const onServerIORecordStop = async () => {
    const outputWav = await appState.stopOutputRecording();
    setIsRecording(false);
    const wavUrl = URL.createObjectURL(new Blob([outputWav], { type: "audio/wav" }));
    const audioInput = document.getElementById(AUDIO_KEYS.AUDIO_ELEMENT_FOR_SAMPLING_INPUT) as HTMLAudioElement;
    if (audioInput) {
      audioInput.src = wavUrl;
    }
  };

  // Format duration
  const formatDuration = (seconds: number): string => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  // ---------------- Handlers ----------------

  // Handle recording toggle
  const handleRecordingToggle = async () => {
    if (isRecording) {
      onServerIORecordStop();
    } else {
      onServerIORecordStart();
    }
  };

  // Handle output device change
  const handleOutputDeviceChange = (deviceId: string) => {
    setSelectedOutputDevice(deviceId);
    setAudioOutputForAnalyzer(deviceId);
  };

  // ---------------- Render ----------------

  const deviceOptions = outputAudioDeviceInfo.map((device) => ({
    value: device.deviceId,
    label: device.label || `Output Device ${device.deviceId.slice(0, 8)}`
  }));

  return (
    <div className="mt-4 pt-4 border-t border-outline-variant">
      <div className="flex justify-between items-center mb-3">
        <h5 className="text-md font-semibold text-on-surface">
          <FontAwesomeIcon icon={faMicrophone} className="mr-2 text-primary" />
          {t('serverIoAnalyzerLabel')}
        </h5>
        <button
          onClick={() => setIsCollapsed(!isCollapsed)}
          className={CSS_CLASSES.iconButton}
          title={isCollapsed ? 'Expand' : 'Collapse'}
        >
          <FontAwesomeIcon icon={isCollapsed ? faChevronDown : faChevronUp} className="h-4 w-4" />
        </button>
      </div>

      {!isCollapsed && (
        <div className="space-y-4 bg-surface-container-low p-3 rounded-md border border-outline-variant">
          {/* Recording Controls */}
          <div className="flex items-center space-x-3">
            <button
              onClick={handleRecordingToggle}
              disabled={!isConverting}
              className={`flex items-center px-4 py-2 text-sm font-semibold rounded-full transition-all duration-150 ${
                !isConverting
                  ? 'bg-surface-container-highest text-on-surface/30 cursor-not-allowed border border-outline-variant/30'
                  : isRecording
                  ? 'bg-error text-on-error hover:shadow-elevation-2 active:scale-97'
                  : 'bg-primary text-on-primary hover:shadow-elevation-2 active:scale-97'
              }`}
            >
              <FontAwesomeIcon icon={isRecording ? faStop : faPlay} className="mr-1.5 text-xs" />
              {isRecording ? 'Stop Recording' : 'Start Recording'}
            </button>

            {isRecording && (
              <div className="flex items-center text-error">
                <div className="w-2.5 h-2.5 bg-error rounded-full animate-ping mr-2"></div>
                <span className="text-xs font-semibold">
                  Recording... {formatDuration(recordingDuration)}
                </span>
              </div>
            )}
          </div>

          {/* Output Device Selection */}
          <div className="max-w-md pt-1">
            <MD3Select
              id="serverIOOutputDevice"
              label={t('outputDeviceLabel')}
              value={selectedOutputDevice}
              onChange={(e) => handleOutputDeviceChange(e.target.value)}
              options={deviceOptions}
            />
          </div>

          {/* Audio Players */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-1">
            {/* Input Audio */}
            <div>
              <label className={CSS_CLASSES.label}>Input Audio:</label>
              <AudioPlayer
                src="/tmp/in.wav"
                title="Input Audio"
                id={AUDIO_KEYS.AUDIO_ELEMENT_FOR_SAMPLING_INPUT}
                outputDeviceId={audioOutputForAnalyzer}
                modelName={
                  appState.serverSetting.serverSetting.modelSlotIndex !== undefined
                    ? appState.serverSetting.serverSetting.modelSlots[
                        appState.serverSetting.serverSetting.modelSlotIndex
                      ]?.name || 'Unknown'
                    : 'Unknown'
                }
                audioType="Input"
              />
            </div>

            {/* Output Audio */}
            <div>
              <label className={CSS_CLASSES.label}>Output Audio:</label>
              <AudioPlayer
                src="/tmp/out.wav"
                title="Output Audio"
                id={AUDIO_KEYS.AUDIO_ELEMENT_FOR_SAMPLING_OUTPUT}
                outputDeviceId={audioOutputForAnalyzer}
                modelName={
                  appState.serverSetting.serverSetting.modelSlotIndex !== undefined
                    ? appState.serverSetting.serverSetting.modelSlots[
                        appState.serverSetting.serverSetting.modelSlotIndex
                      ]?.name || 'Unknown'
                    : 'Unknown'
                }
                audioType="Output"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default ServerIO;