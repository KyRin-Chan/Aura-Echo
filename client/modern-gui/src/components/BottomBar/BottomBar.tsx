import { JSX, useState, useEffect } from 'react';
import { useThemeContext } from '../../context/ThemeContext';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faSun, faMoon, faPlay, faStop, faVolumeUp, faVolumeMute } from '@fortawesome/free-solid-svg-icons';
import { AppContextValue, useAppState } from '../../context/AppContext';
import { useUIContext } from '../../context/UIContext';
import MergeLabModal from './Modals/Merge/MergeLabModal';
import VoiceAnalyzerModal from './Modals/VoiceAnalyzer/VoiceAnalyzerModal';
import AdvancedSettingsModal from './Modals/AdvancedSettings/AdvancedSettingsModal';
import ClientInfoModal from './Modals/ClientInfoModal';
import ServerInfoModal from './Modals/ServerInfoModal';
import { CSS_CLASSES } from '../../styles/constants';
import PassthroughConfirmModal from './Modals/PassthroughConfirmModal';

function BottomBar(): JSX.Element {
  // ---------------- States ----------------
  const { theme, toggleTheme } = useThemeContext();
  const appState = useAppState() as AppContextValue;
  const uiContext = useUIContext();

  const [showMerge, setShowMerge] = useState<boolean>(false);
  const [showVoiceAnalyzer, setShowVoiceAnalyzer] = useState<boolean>(false);
  const [showSettings, setShowSettings] = useState<boolean>(false);
  const [showPassthroughConfirm, setShowPassthroughConfirm] = useState<boolean>(false);
  const [showClientInfo, setShowClientInfo] = useState<boolean>(false);
  const [showServerInfo, setShowServerInfo] = useState<boolean>(false);

  const [startWithAudioContextCreate, setStartWithAudioContextCreate] = useState<boolean>(false);

  // ---------------- Hooks ----------------

  // Set start with audio context create
  useEffect(() => {
    if (!startWithAudioContextCreate) {
      return;
    }
    uiContext.setIsConverting(true);
    appState.start();
  }, [startWithAudioContextCreate]);

  // ---------------- Handlers ----------------

  // Handle toggle audio convert
  const handleToggleClientActivity = async () => {
    if (uiContext.isConverting) {
      handleStop();
    } else {
      handleStart();
    }
  }

  // Handle start convert
  const handleStart = async () => {
    // Check if model is selected
    if (appState.serverSetting.serverSetting.modelSlotIndex === -1) {
      uiContext.showError('Select a voice model first.', "Warning")
      return
    }

    // Client Mode
    if (appState.serverSetting.serverSetting.enableServerAudio == 0) {
      // Check if audio input is selected
      if (!appState.setting.voiceChangerClientSetting.audioInput || appState.setting.voiceChangerClientSetting.audioInput == 'none') {
        uiContext.showError('Select an audio input device.', "Warning")
        return
      }

      // Check if audio output is selected
      if (uiContext.audioOutputForGUI == 'none') {
        uiContext.showError('Select an audio output device.', "Warning")
        return
      }

      // Check if audio context is initialized
      if (!appState.initializedRef.current) {
        while (true) {
          await new Promise<void>((resolve) => {
            setTimeout(resolve, 500);
          });
          if (appState.initializedRef.current) {
            break;
          }
        }
        setStartWithAudioContextCreate(true);
      }
      // Start client convert
      else {
        uiContext.setIsConverting(true);
        await appState.start();
      }
    }
    // Server Mode
    else {
      // Check if server audio input is selected
      if (!appState.serverSetting.serverSetting.serverAudioInputDevices.find(device => device.index === appState.serverSetting.serverSetting.serverInputDeviceId)) {
        uiContext.showError('Select an audio input device.', "Warning")
        return
      }

      // Check if server audio output is selected
      if (!appState.serverSetting.serverSetting.serverAudioOutputDevices.find(device => device.index === appState.serverSetting.serverSetting.serverOutputDeviceId)) {
        uiContext.showError('Select an audio output device.', "Warning")
        return
      }

      // Start server convert
      appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, serverAudioStated: 1 });
      uiContext.setIsConverting(true);
    }
  };

  // Handle stop convert
  const handleStop = async () => {
    if (appState.serverSetting.serverSetting.enableServerAudio == 0) {
      uiContext.setIsConverting(false);
      await appState.stop();
    } else {
      uiContext.setIsConverting(false);
      appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, serverAudioStated: 0 });
    }
  };

  // Handle disable pass through
  const disablePassThrough = () => {
    appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, passThrough: false });
  }

  // ---------------- Render ----------------

  return (
    <>
      <MergeLabModal
        appState={appState}
        guiState={uiContext}
        showMerge={showMerge}
        setShowMerge={setShowMerge}
      />

      <VoiceAnalyzerModal
        appState={appState}
        guiState={uiContext}
        showVoiceAnalyzer={showVoiceAnalyzer}
        setShowVoiceAnalyzer={setShowVoiceAnalyzer}
      />

      <AdvancedSettingsModal
        showAdvancedSettings={showSettings}
        setShowAdvancedSettings={setShowSettings}
      />

      <PassthroughConfirmModal
        appState={appState}
        showPassthrough={showPassthroughConfirm}
        setShowPassthrough={setShowPassthroughConfirm}
      />

      <ClientInfoModal
        showClientInfo={showClientInfo}
        setShowClientInfo={setShowClientInfo}
      />

      <ServerInfoModal
        showServerInfo={showServerInfo}
        setShowServerInfo={setShowServerInfo}
      />

      <div
        className="h-16 min-h-[56px] flex items-center justify-between px-6 py-2 flex-shrink-0 transition-colors duration-300"
        style={{
          backgroundColor: 'var(--bg-secondary)',
          borderTop: '1px solid var(--border-primary)',
        }}
      >
        {/* Left group: tools */}
        <div className="flex space-x-2">
          <button onClick={() => setShowMerge(true)} className={CSS_CLASSES.modalSecondaryButton}>Merge Lab</button>
          <button onClick={() => setShowVoiceAnalyzer(true)} className={CSS_CLASSES.modalSecondaryButton}>Voice Analyzer</button>
          <button onClick={() => setShowSettings(true)} className={CSS_CLASSES.modalSecondaryButton}>Advanced Settings</button>
        </div>

        {/* Center group: primary controls */}
        <div className="flex items-center space-x-3">
          <button
            onClick={handleToggleClientActivity}
            className={`px-5 py-2 rounded-full text-sm font-semibold transition-all duration-200 flex items-center space-x-2 shadow-sm hover:shadow-md
            ${uiContext.isConverting
                ? 'text-white'
                : 'text-white'}
            focus:outline-none focus:ring-2 focus:ring-offset-2`}
            style={{
              backgroundColor: uiContext.isConverting ? 'var(--macaron-coral)' : 'var(--macaron-mint)',
              color: '#3a3530',
            }}
          >
            <FontAwesomeIcon icon={uiContext.isConverting ? faStop : faPlay} />
            <span>{uiContext.isConverting ? 'Stop' : 'Start'}</span>
          </button>
          <button
            onClick={appState.serverSetting.serverSetting.passThrough ? disablePassThrough : () => setShowPassthroughConfirm(true)}
            className={`px-4 py-2 rounded-full text-sm font-medium transition-all duration-200 flex items-center space-x-2
            focus:outline-none focus:ring-2 focus:ring-offset-2`}
            style={{
              backgroundColor: appState.serverSetting?.serverSetting?.passThrough ? 'var(--macaron-yellow)' : 'var(--bg-tertiary)',
              color: 'var(--text-primary)',
              border: `1px solid ${appState.serverSetting?.serverSetting?.passThrough ? 'var(--macaron-yellow)' : 'var(--border-primary)'}`,
            }}
          >
            <FontAwesomeIcon icon={appState.serverSetting?.serverSetting?.passThrough ? faVolumeUp : faVolumeMute} />
            <span>{appState.serverSetting?.serverSetting?.passThrough ? 'Passthrough ON' : 'Passthrough OFF'}</span>
          </button>
        </div>

        {/* Right group: info & theme */}
        <div className="flex items-center space-x-2">
          <button onClick={() => setShowServerInfo(true)} className={CSS_CLASSES.modalSecondaryButton}>Server Info</button>
          <button onClick={() => setShowClientInfo(true)} className={CSS_CLASSES.modalSecondaryButton}>Client Info</button>
          <button
            onClick={toggleTheme}
            className={CSS_CLASSES.iconButton}
            aria-label={theme === 'light' ? "Switch to dark mode" : "Switch to light mode"}
          >
            <FontAwesomeIcon icon={theme === 'light' ? faMoon : faSun} className="h-5 w-5" />
          </button>
        </div>
      </div>
    </>
  );
}

export default BottomBar;