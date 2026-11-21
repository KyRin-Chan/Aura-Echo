import React, { createContext, ReactNode, useContext, useState, useRef, useCallback, useEffect } from 'react';
import LoadingScreen from '../components/Modals/LoadingScreen';
import ErrorNotifications from '../components/Modals/ErrorNotifications';
import { useAppState } from './AppContext';
import { useModelLoadingProgress, LoadingProgress } from '../scripts/useModelLoadingProgress';

type ErrorType = 'Error' | 'Warning' | 'Confirm';

interface UIError {
  id: number;
  message: string;
  type: ErrorType;
}

export interface UIContextType {
  startLoading: (message?: string) => void;
  stopLoading: () => void;
  showError: (message: string, type?: ErrorType | string) => void;
  setIsConverting: (isConverting: boolean) => void;
  isConverting: boolean;

  loadingProgress: LoadingProgress | null;
  assetVersion: number;
  bumpAssetVersion: () => void;

  reloadDeviceInfo: () => Promise<void>;
  inputAudioDeviceInfo: MediaDeviceInfo[];
  outputAudioDeviceInfo: MediaDeviceInfo[];
  audioInputForGUI: string;
  audioOutputForGUI: string;
  audioMonitorForGUI: string;
  audioOutputForAnalyzer: string;
  setAudioInputForGUI: (audioInputForGUI: string) => void;
  setAudioOutputForGUI: (audioOutputForGUI: string) => void;
  setAudioMonitorForGUI: (audioMonitorForGUI: string) => void;
  setAudioOutputForAnalyzer: (audioOutputForAnalyzer: string) => void;
}

// Create context
const UIContext = createContext<UIContextType | undefined>(undefined);

// Create hook
export const useUIContext = (): UIContextType => {
  const context = useContext(UIContext);
  if (!context) {
    throw new Error('useUIContext must be used within a UIContextProvider');
  }
  return context;
};

// Create provider
export const UIContextProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  // ---------------- States ----------------
  const appState = useAppState();

  const [isLoading, setIsLoading] = useState(false);
  const [isConverting, setIsConverting] = useState<boolean>(false);
  const [loadingProgress, setLoadingProgress] = useState<LoadingProgress | null>(null);
  const [assetVersion, setAssetVersion] = useState<number>(() => Date.now());
  const [inputAudioDeviceInfo, setInputAudioDeviceInfo] = useState<MediaDeviceInfo[]>([]);
  const [outputAudioDeviceInfo, setOutputAudioDeviceInfo] = useState<MediaDeviceInfo[]>([]);
  const [audioOutputForAnalyzer, setAudioOutputForAnalyzer] = useState<string>("none");
  const [audioInputForGUI, setAudioInputForGUI] = useState<string>("none");
  const [audioOutputForGUI, setAudioOutputForGUI] = useState<string>("none");
  const [audioMonitorForGUI, setAudioMonitorForGUI] = useState<string>("none");

  const [loadingMessage, setLoadingMessage] = useState<string>();
  const isLoadingRef = useRef(false);
  const [errors, setErrors] = useState<UIError[]>([]);
  const idRef = useRef(0);

  // ---------------- Functions ----------------

  // Start loading screen
  const startLoading = useCallback((message?: string) => {
    isLoadingRef.current = true;
    setLoadingMessage(message);
    setLoadingProgress(null);
    setIsLoading(true);
  }, []);

  // Stop loading screen
  const stopLoading = useCallback(() => {
    isLoadingRef.current = false;
    setIsLoading(false);
    setLoadingMessage(undefined);
    setLoadingProgress(null);
  }, []);

  const bumpAssetVersion = useCallback(() => {
    setAssetVersion(Date.now());
  }, []);

  // Normalize notification type from legacy or localized strings
  const normalizeErrorType = (rawType?: ErrorType | string): ErrorType => {
    if (!rawType) return 'Error';
    const lower = rawType.trim().toLowerCase();
    if (lower === 'confirm' || rawType === '确认' || rawType === '成功' || rawType === '確認') {
      return 'Confirm';
    }
    if (lower === 'warning' || rawType === '警告' || rawType === '提示') {
      return 'Warning';
    }
    return 'Error';
  };

  // Show error toast (top right)
  const showError = useCallback((message: string, type: ErrorType | string = 'Error') => {
    const normalizedType = normalizeErrorType(type);
    idRef.current += 1;
    const id = idRef.current;
    setErrors(prev => [...prev, { id, message, type: normalizedType }]);
    setTimeout(() => {
      setErrors(prev => prev.filter(err => err.id !== id));
    }, 5000);
  }, []);

  // Remove error toast
  const removeError = useCallback((id: number) => {
    setErrors(prev => prev.filter(err => err.id !== id));
  }, []);

  // Check if device is available
  const checkDeviceAvailable = useRef<boolean>(false);

  // Reload device list
  const _reloadDeviceInfo = async () => {
    // Check if device is available
    if (checkDeviceAvailable.current == false) {
      try {
        const ms = await navigator.mediaDevices.getUserMedia({ video: false, audio: true });
        ms.getTracks().forEach((x) => {
          x.stop();
        });
        checkDeviceAvailable.current = true;
      } catch (e) {
        console.warn("Enumerate device error:", e);
      }
    }

    // Enumerate devices
    const mediaDeviceInfos = await navigator.mediaDevices.enumerateDevices();

    // Get audio input devices
    const audioInputs = mediaDeviceInfos.filter((x) => {
      return x.kind == "audioinput";
    });

    // Get audio output devices
    const audioOutputs = mediaDeviceInfos.filter((x) => {
      return x.kind == "audiooutput";
    });

    return [audioInputs, audioOutputs];
  };

  // Reload device list and update state
  const reloadDeviceInfo = async () => {
    const audioInfo = await _reloadDeviceInfo();
    setInputAudioDeviceInfo(audioInfo[0]);
    setOutputAudioDeviceInfo(audioInfo[1]);
  };

  // ---------------- Hooks ----------------

  // Subscribe to model loading progress events
  const protocol = (appState.setting?.workletNodeSetting?.protocol ?? 'sio') as 'sio' | 'ws' | 'rest';
  const serverUrl = appState.setting?.workletNodeSetting?.serverUrl ?? '';
  useModelLoadingProgress({
    protocol,
    serverUrl,
    onProgress: (progress) => {
      if (isLoadingRef.current) {
        setLoadingProgress(progress);
      }
    },
  });

  // Check if server is running (needed if page is reloaded and servermode still enabled)
  useEffect(() => {
    if (appState.serverSetting.serverSetting.enableServerAudio == 1) {
      setIsConverting(appState.serverSetting.serverSetting.serverAudioStated == 1);
    }
  }, [
    appState.serverSetting.serverSetting.enableServerAudio,
    appState.serverSetting.serverSetting.serverAudioStated,
  ]);

  // Enumerate devices on mount and listen to native devicechange events
  useEffect(() => {
    let isMounted = true;

    const refreshDevices = async () => {
      if (!navigator.mediaDevices?.enumerateDevices) return;
      try {
        const audioInfo = await _reloadDeviceInfo();
        if (!isMounted) return;

        setInputAudioDeviceInfo((prev) => {
          const prevIds = prev.map((d) => `${d.deviceId}:${d.label}`).join('|');
          const nextIds = audioInfo[0].map((d) => `${d.deviceId}:${d.label}`).join('|');
          return prevIds === nextIds ? prev : audioInfo[0];
        });

        setOutputAudioDeviceInfo((prev) => {
          const prevIds = prev.map((d) => `${d.deviceId}:${d.label}`).join('|');
          const nextIds = audioInfo[1].map((d) => `${d.deviceId}:${d.label}`).join('|');
          return prevIds === nextIds ? prev : audioInfo[1];
        });
      } catch (err) {
        console.error("An error occurred during enumeration of devices:", err);
      }
    };

    refreshDevices();

    const mediaDevices = navigator.mediaDevices;
    if (mediaDevices?.addEventListener) {
      mediaDevices.addEventListener('devicechange', refreshDevices);
    }

    return () => {
      isMounted = false;
      if (mediaDevices?.removeEventListener) {
        mediaDevices.removeEventListener('devicechange', refreshDevices);
      }
    };
  }, []);

  // ---------------- Render ----------------

  return (
    <UIContext.Provider value={{
      startLoading, stopLoading, showError, setIsConverting, reloadDeviceInfo,
      isConverting, loadingProgress, assetVersion, bumpAssetVersion, setAudioInputForGUI, setAudioOutputForGUI, setAudioMonitorForGUI,
      inputAudioDeviceInfo, outputAudioDeviceInfo, audioInputForGUI, audioOutputForGUI, audioMonitorForGUI, audioOutputForAnalyzer, setAudioOutputForAnalyzer
    }}>
      {children}
      {isLoading && <LoadingScreen message={loadingMessage} progress={loadingProgress} />}
      <ErrorNotifications errors={errors} removeError={removeError} />
    </UIContext.Provider>
  );
};
