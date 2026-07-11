import { JSX, useState, useEffect } from 'react';
import { useThemeContext } from '../../context/ThemeContext';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faSun,
  faMoon,
  faPlay,
  faStop,
  faVolumeUp,
  faVolumeMute,
  faPalette
} from '@fortawesome/free-solid-svg-icons';
import { AppContextValue, useAppState } from '../../context/AppContext';
import { useUIContext } from '../../context/UIContext';
import { t } from '../../locales';
import MergeLabModal from './Modals/Merge/MergeLabModal';
import VoiceAnalyzerModal from './Modals/VoiceAnalyzer/VoiceAnalyzerModal';
import AdvancedSettingsModal from './Modals/AdvancedSettings/AdvancedSettingsModal';
import ClientInfoModal from './Modals/ClientInfoModal';
import ServerInfoModal from './Modals/ServerInfoModal';
import { CSS_CLASSES } from '../../styles/constants';
import PassthroughConfirmModal from './Modals/PassthroughConfirmModal';
import { PRESET_SEED_COLORS } from '../../styles/md3ColorEngine';

function BottomBar(): JSX.Element {
  // ---------------- States ----------------
  const { theme, toggleTheme, seedColor, setSeedColor } = useThemeContext();
  const appState = useAppState() as AppContextValue;
  const uiContext = useUIContext();

  const [showMerge, setShowMerge] = useState<boolean>(false);
  const [showVoiceAnalyzer, setShowVoiceAnalyzer] = useState<boolean>(false);
  const [showSettings, setShowSettings] = useState<boolean>(false);
  const [showPassthroughConfirm, setShowPassthroughConfirm] = useState<boolean>(false);
  const [showClientInfo, setShowClientInfo] = useState<boolean>(false);
  const [showServerInfo, setShowServerInfo] = useState<boolean>(false);
  const [showColorPicker, setShowColorPicker] = useState<boolean>(false);

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
  };

  // Handle start convert
  const handleStart = async () => {
    // Check if model is selected
    if (appState.serverSetting.serverSetting.modelSlotIndex === -1) {
      uiContext.showError(t('selectModelFirst'), t('voiceAnalyzerUploadWarning'));
      return;
    }

    // Client Mode
    if (appState.serverSetting.serverSetting.enableServerAudio == 0) {
      // Check if audio input is selected
      if (
        !appState.setting.voiceChangerClientSetting.audioInput ||
        appState.setting.voiceChangerClientSetting.audioInput == 'none'
      ) {
        uiContext.showError(t('selectAudioInputFirst'), t('voiceAnalyzerUploadWarning'));
        return;
      }

      // Check if audio output is selected
      if (uiContext.audioOutputForGUI == 'none') {
        uiContext.showError(t('selectAudioOutputFirst'), t('voiceAnalyzerUploadWarning'));
        return;
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
      if (
        !appState.serverSetting.serverSetting.serverAudioInputDevices.find(
          (device) => device.index === appState.serverSetting.serverSetting.serverInputDeviceId
        )
      ) {
        uiContext.showError(t('selectAudioInputFirst'), t('voiceAnalyzerUploadWarning'));
        return;
      }

      // Check if server audio output is selected
      if (
        !appState.serverSetting.serverSetting.serverAudioOutputDevices.find(
          (device) => device.index === appState.serverSetting.serverSetting.serverOutputDeviceId
        )
      ) {
        uiContext.showError(t('selectAudioOutputFirst'), t('voiceAnalyzerUploadWarning'));
        return;
      }

      // Start server convert
      appState.serverSetting.updateServerSettings({
        ...appState.serverSetting.serverSetting,
        serverAudioStated: 1
      });
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
      appState.serverSetting.updateServerSettings({
        ...appState.serverSetting.serverSetting,
        serverAudioStated: 0
      });
    }
  };

  // Handle disable pass through
  const disablePassThrough = () => {
    appState.serverSetting.updateServerSettings({
      ...appState.serverSetting.serverSetting,
      passThrough: false
    });
  };

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

      <div className="h-20 min-h-[80px] flex items-center justify-between px-6 py-3 flex-shrink-0 bg-surface-container border-t border-outline-variant transition-colors duration-300">
        {/* Left group: tools */}
        <div className="flex space-x-2">
          <button onClick={() => setShowMerge(true)} className={CSS_CLASSES.modalSecondaryButton}>
            {t('mergeLabTitle')}
          </button>
          <button onClick={() => setShowVoiceAnalyzer(true)} className={CSS_CLASSES.modalSecondaryButton}>
            {t('voiceAnalyzerTitle')}
          </button>
          <button onClick={() => setShowSettings(true)} className={CSS_CLASSES.modalSecondaryButton}>
            {t('advancedSettingsTitle')}
          </button>
        </div>

        {/* Center group: primary controls */}
        <div className="flex items-center space-x-3">
          <button
            onClick={handleToggleClientActivity}
            className={`px-6 py-2.5 rounded-full text-sm font-semibold transition-all duration-200 flex items-center space-x-2 shadow-elevation-2 hover:shadow-elevation-3 focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 ${
              uiContext.isConverting ? 'bg-error text-on-error' : 'bg-primary-container text-on-primary-container'
            }`}
          >
            <FontAwesomeIcon icon={uiContext.isConverting ? faStop : faPlay} />
            <span>{uiContext.isConverting ? t('btnStop') : t('btnStart')}</span>
          </button>
          <button
            onClick={
              appState.serverSetting.serverSetting.passThrough
                ? disablePassThrough
                : () => setShowPassthroughConfirm(true)
            }
            className={`px-5 py-2.5 rounded-full text-sm font-medium transition-all duration-200 flex items-center space-x-2 focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 ${
              appState.serverSetting?.serverSetting?.passThrough
                ? 'bg-secondary-container text-on-secondary-container'
                : 'bg-surface-container-high text-on-surface border border-outline'
            }`}
          >
            <FontAwesomeIcon
              icon={appState.serverSetting?.serverSetting?.passThrough ? faVolumeUp : faVolumeMute}
            />
            <span>
              {appState.serverSetting?.serverSetting?.passThrough
                ? t('passthroughOn')
                : t('passthroughOff')}
            </span>
          </button>
        </div>

        {/* Right group: info & theme */}
        <div className="flex items-center space-x-2 relative">
          <button onClick={() => setShowServerInfo(true)} className={CSS_CLASSES.modalSecondaryButton}>
            {t('serverInfoTitle')}
          </button>
          <button onClick={() => setShowClientInfo(true)} className={CSS_CLASSES.modalSecondaryButton}>
            {t('clientInfoTitle')}
          </button>

          {/* Seed Color Picker */}
          <div className="relative">
            <button
              onClick={() => setShowColorPicker(!showColorPicker)}
              className={CSS_CLASSES.iconButton}
              title={t('changeSeedColorTooltip')}
              aria-label={t('changeSeedColorTooltip')}
            >
              <FontAwesomeIcon icon={faPalette} className="h-5 w-5" style={{ color: seedColor }} />
            </button>

            {showColorPicker && (
              <div className="absolute right-0 bottom-14 z-50 p-3 bg-surface-container-highest border border-outline-variant rounded-md shadow-elevation-3 flex flex-col space-y-2 min-w-[150px]">
                <span className="text-xs font-semibold text-on-surface-variant mb-1">{t('themeColorsLabel')}</span>
                <div className="flex flex-col space-y-1.5">
                  {PRESET_SEED_COLORS.map((c) => (
                    <button
                      key={c.hex}
                      onClick={() => {
                        setSeedColor(c.hex);
                        setShowColorPicker(false);
                      }}
                      className="flex items-center space-x-2.5 px-2 py-1 rounded-sm hover:bg-surface-variant/20 transition-all text-xs font-medium text-on-surface text-left"
                    >
                      <span
                        className="w-3.5 h-3.5 rounded-full border border-outline-variant/30 flex-shrink-0"
                        style={{ backgroundColor: c.hex }}
                      />
                      <span>{t(c.langKey)}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          <button
            onClick={toggleTheme}
            className={CSS_CLASSES.iconButton}
            aria-label={theme === 'light' ? t('switchToDarkMode') : t('switchToLightMode')}
          >
            <FontAwesomeIcon icon={theme === 'light' ? faMoon : faSun} className="h-5 w-5" />
          </button>
        </div>
      </div>
    </>
  );
}

export default BottomBar;