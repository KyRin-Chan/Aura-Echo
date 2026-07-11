import { JSX, useState, useEffect } from 'react';
import { useAppState } from '../../../../context/AppContext';
import { useUIContext } from '../../../../context/UIContext';
import MD3Slider from '../../../Helpers/MD3Slider';
import MD3Select from '../../../Helpers/MD3Select';
import MD3Switch from '../../../Helpers/MD3Switch';
import MD3Checkbox from '../../../Helpers/MD3Checkbox';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faExclamationTriangle } from '@fortawesome/free-solid-svg-icons';
import { Protocol, useIndexedDB } from '@dannadori/voice-changer-client-js';
import { CSS_CLASSES } from '../../../../styles/constants';
import { t, currentLang, setLang, LangType } from '../../../../locales';

function SettingsView(): JSX.Element {
  // ---------------- States ----------------
  const appState = useAppState();
  const uiState = useUIContext();

  const [localCrossFadeOverlapSize, setLocalCrossFadeOverlapSize] = useState<number>(
    appState.serverSetting?.serverSetting?.crossFadeOverlapSize ?? 0.02
  );
  const [localProtect, setLocalProtect] = useState<number>(
    appState.serverSetting?.serverSetting?.protect ?? 0
  );
  const [localLang, setLocalLang] = useState<LangType>(currentLang);
  const { removeItem } = useIndexedDB({ clientType: null });

  // ---------------- Hooks ----------------

  // Update local state when server settings change
  useEffect(() => {
    const crossFade = appState.serverSetting?.serverSetting?.crossFadeOverlapSize;
    if (crossFade != null) setLocalCrossFadeOverlapSize(crossFade);

    const protect = appState.serverSetting?.serverSetting?.protect;
    if (protect != null) setLocalProtect(protect);
  }, [appState.serverSetting?.serverSetting]);

  // ---------------- Handlers ----------------

  // Handle cross fade overlap size change
  const handleCrossFadeOverlapSizeChange = async (val: number) => {
    await appState.serverSetting.updateServerSettings({
      ...appState.serverSetting.serverSetting,
      crossFadeOverlapSize: val
    });
    setLocalCrossFadeOverlapSize(val);
  };

  // Handle silence front change
  const handleSilenceFrontChange = async (val: boolean) => {
    const value = val ? 1 : 0;
    uiState.startLoading(`${value === 1 ? "Enabling" : "Disabling"} Silence Front`);
    await appState.serverSetting.updateServerSettings({
      ...appState.serverSetting.serverSetting,
      silenceFront: value
    });
    uiState.stopLoading();
  };

  // Handle force fp32 change
  const handleForceFp32Change = async (val: boolean) => {
    const value = val ? 1 : 0;
    uiState.startLoading(`${value === 1 ? "Enabling" : "Disabling"} Force FP32 Mode`);
    await appState.serverSetting.updateServerSettings({
      ...appState.serverSetting.serverSetting,
      forceFp32: value
    });
    uiState.stopLoading();
  };

  // Handle disable jit change
  const handleDisableJitChange = async (val: boolean) => {
    const value = val ? 1 : 0;
    uiState.startLoading(`${value === 1 ? "Disabling" : "Enabling"} JIT Compilation`);
    await appState.serverSetting.updateServerSettings({
      ...appState.serverSetting.serverSetting,
      disableJit: value
    });
    uiState.stopLoading();
  };

  // Handle use onnx change
  const handleUseONNXChange = async (val: boolean) => {
    const value = val ? 1 : 0;
    uiState.startLoading(`${value === 1 ? "Enabling" : "Disabling"} Convert to ONNX`);
    await appState.serverSetting.updateServerSettings({
      ...appState.serverSetting.serverSetting,
      useONNX: value
    });
    uiState.stopLoading();
  };

  // Handle protect change
  const handleProtectChange = async (val: number) => {
    await appState.serverSetting.updateServerSettings({
      ...appState.serverSetting.serverSetting,
      protect: val
    });
    setLocalProtect(val);
  };

  const handlePassThroughConfirmationSkipChange = async (val: boolean) => {
    appState.setVoiceChangerClientSetting({
      ...appState.setting.voiceChangerClientSetting,
      passThroughConfirmationSkip: val
    });
  };

  const handleAGCChange = async (val: boolean) => {
    appState.setVoiceChangerClientSetting({
      ...appState.setting.voiceChangerClientSetting,
      agcEnabled: val
    });
  };

  const handleLangChange = (lang: LangType) => {
    setLang(lang);
    setLocalLang(lang);
    window.location.reload();
  };

  const handleResetSettings = async () => {
    await appState.clearSetting();
    await removeItem('INDEXEDDB_KEY_AUDIO_INPUT');
    await removeItem('INDEXEDDB_KEY_AUDIO_OUTPUT');
    await removeItem('INDEXEDDB_KEY_AUDIO_MONITOR');
    localStorage.removeItem('app_language');
    window.location.reload();
  };

  // ---------------- Render ----------------

  const protocolOptions = [
    { value: 'ws', label: 'Raw WebSocket (High Performance)' },
    { value: 'sio', label: 'SIO (Socket.IO)' },
    { value: 'rest', label: 'REST (HTTP)' }
  ];

  const languageOptions = [
    { value: 'zh', label: '简体中文 (Chinese)' },
    { value: 'en', label: 'English' },
    { value: 'ja', label: '日本語 (Japanese)' }
  ];

  return (
    <div className="space-y-4 py-2 bg-surface-container-low p-4 rounded-md border border-outline-variant max-h-[500px] overflow-y-auto pr-1.5">
      <div>
        <MD3Select
          id="language"
          label="Language / 语言"
          value={localLang}
          onChange={(e) => handleLangChange(e.target.value as LangType)}
          options={languageOptions}
        />
      </div>

      <div>
        <MD3Select
          id="protocol"
          label="Protocol"
          value={appState.setting.workletNodeSetting.protocol}
          onChange={(e) =>
            appState.setWorkletNodeSetting({
              ...appState.setting.workletNodeSetting,
              protocol: e.target.value as Protocol
            })
          }
          options={protocolOptions}
        />
      </div>

      <div>
        <label htmlFor="crossfade" className={CSS_CLASSES.label}>
          Crossfade Overlap
        </label>
        <MD3Slider
          id="crossfade"
          min={0.05}
          max={0.2}
          step={0.01}
          value={localCrossFadeOverlapSize}
          onImmediateChange={setLocalCrossFadeOverlapSize}
          onChange={async (val) => {
            await handleCrossFadeOverlapSizeChange(val);
          }}
          showValue={true}
          valueFormatter={(val) => `${val.toFixed(2)} s`}
        />
      </div>

      <div className="space-y-3.5 pt-1.5 pb-2">
        <MD3Switch
          checked={appState.serverSetting.serverSetting.silenceFront === 1}
          onChange={handleSilenceFrontChange}
          label="Silence Front"
        />

        <MD3Switch
          checked={appState.serverSetting.serverSetting.forceFp32 === 1}
          onChange={handleForceFp32Change}
          label="Force FP32 Mode"
        />

        <MD3Switch
          checked={appState.serverSetting.serverSetting.disableJit === 1}
          onChange={handleDisableJitChange}
          label="Disable JIT Compilation"
        />

        <MD3Switch
          checked={appState.serverSetting.serverSetting.useONNX === 1}
          onChange={handleUseONNXChange}
          label="Convert to ONNX"
        />

        <MD3Switch
          checked={appState.setting.voiceChangerClientSetting.agcEnabled ?? false}
          onChange={handleAGCChange}
          label="Auto Gain Control (Limiter)"
        />
      </div>

      <div>
        <label htmlFor="protect" className={CSS_CLASSES.label}>
          Protect
        </label>
        <MD3Slider
          id="protect"
          min={0}
          max={0.5}
          step={0.01}
          value={localProtect}
          onImmediateChange={setLocalProtect}
          onChange={async (val) => {
            await handleProtectChange(val);
          }}
          showValue={true}
          valueFormatter={(val) => val.toFixed(2)}
        />
      </div>

      <div className="border border-error/50 p-4 rounded-lg bg-error/5 space-y-2 mt-2">
        <div className="flex items-center text-error mb-2 text-sm font-semibold">
          <FontAwesomeIcon icon={faExclamationTriangle} className="mr-2" />
          <span>Danger Zone</span>
        </div>
        <MD3Checkbox
          checked={appState.setting.voiceChangerClientSetting.passThroughConfirmationSkip ?? false}
          onChange={handlePassThroughConfirmationSkipChange}
          label="Skip Pass through confirmation"
        />
        <button
          onClick={handleResetSettings}
          className="w-full mt-3 px-4 py-2 bg-error hover:bg-error/90 text-on-error font-semibold rounded-full shadow-elevation-1 transition-all duration-150 text-xs text-center"
        >
          {t('resetSettings')}
        </button>
      </div>
    </div>
  );
}

export default SettingsView;
