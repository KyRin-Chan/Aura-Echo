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
import { clearLocalFormantCache } from '../../../../utils/formantStorage';

function SettingsView(): JSX.Element {
  // ---------------- States ----------------
  const appState = useAppState();
  const uiState = useUIContext();

  const [localCrossFadeOverlapSize, setLocalCrossFadeOverlapSize] = useState<number>(
    appState.serverSetting?.serverSetting?.crossFadeOverlapSize ?? 0.02
  );
  const [localProtect, setLocalProtect] = useState<number>(
    appState.serverSetting?.serverSetting?.protect ?? 0.33
  );
  const [localLang, setLocalLang] = useState<LangType>(currentLang);
  const { removeItem } = useIndexedDB({ clientType: null });

  // ---------------- Hooks ----------------

  // Update local state when individual server settings change
  useEffect(() => {
    const crossFade = appState.serverSetting?.serverSetting?.crossFadeOverlapSize;
    if (crossFade != null) setLocalCrossFadeOverlapSize(crossFade);
  }, [appState.serverSetting?.serverSetting?.crossFadeOverlapSize]);

  useEffect(() => {
    const protect = appState.serverSetting?.serverSetting?.protect;
    if (protect != null) setLocalProtect(protect);
  }, [appState.serverSetting?.serverSetting?.protect]);

  // ---------------- Handlers ----------------

  // Handle cross fade overlap size change
  const handleCrossFadeOverlapSizeChange = async (val: number) => {
    const hasModel = (appState.serverSetting?.serverSetting?.modelSlotIndex ?? -1) !== -1;
    if (hasModel) uiState.startLoading(t('warmingUpPipeline'));
    try {
      await appState.serverSetting.updateServerSettings({
        ...appState.serverSetting.serverSetting,
        crossFadeOverlapSize: val
      });
      setLocalCrossFadeOverlapSize(val);
    } finally {
      if (hasModel) uiState.stopLoading();
    }
  };

  // Handle silence front change
  const handleSilenceFrontChange = async (val: boolean) => {
    const value = val ? 1 : 0;
    uiState.startLoading(value === 1 ? t('enablingSilenceFront') : t('disablingSilenceFront'));
    try {
      await appState.serverSetting.updateServerSettings({
        ...appState.serverSetting.serverSetting,
        silenceFront: value
      });
    } finally {
      uiState.stopLoading();
    }
  };

  // Handle force fp32 change
  const handleForceFp32Change = async (val: boolean) => {
    const value = val ? 1 : 0;
    uiState.startLoading(value === 1 ? t('enablingForceFp32') : t('disablingForceFp32'));
    try {
      await appState.serverSetting.updateServerSettings({
        ...appState.serverSetting.serverSetting,
        forceFp32: value
      });
    } finally {
      uiState.stopLoading();
    }
  };

  // Handle disable jit change
  const handleDisableJitChange = async (val: boolean) => {
    const value = val ? 1 : 0;
    uiState.startLoading(value === 1 ? t('disablingJit') : t('enablingJit'));
    try {
      await appState.serverSetting.updateServerSettings({
        ...appState.serverSetting.serverSetting,
        disableJit: value
      });
    } finally {
      uiState.stopLoading();
    }
  };

  // Handle use onnx change
  const handleUseONNXChange = async (val: boolean) => {
    const value = val ? 1 : 0;
    uiState.startLoading(value === 1 ? t('enablingUseOnnx') : t('disablingUseOnnx'));
    try {
      await appState.serverSetting.updateServerSettings({
        ...appState.serverSetting.serverSetting,
        useONNX: value
      });
    } finally {
      uiState.stopLoading();
    }
  };

  // Handle use TensorRT change
  const handleUseTensorRTChange = async (val: boolean) => {
    const value = val ? 1 : 0;
    uiState.startLoading(value === 1 ? t('enablingTensorRt') : t('disablingTensorRt'));
    try {
      await appState.serverSetting.updateServerSettings({
        ...appState.serverSetting.serverSetting,
        useTensorRT: value
      });
    } finally {
      uiState.stopLoading();
    }
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
  };

  const handleResetSettings = async () => {
    await appState.clearSetting();
    await removeItem('INDEXEDDB_KEY_AUDIO_INPUT');
    await removeItem('INDEXEDDB_KEY_AUDIO_OUTPUT');
    await removeItem('INDEXEDDB_KEY_AUDIO_MONITOR');
    localStorage.removeItem('app_language');
    localStorage.removeItem('aura_card_order');
    clearLocalFormantCache();
    sessionStorage.clear();
    window.location.reload();
  };

  // ---------------- Render ----------------

  const protocolOptions = [
    { value: 'ws', label: t('protocolWsOption') },
    { value: 'sio', label: t('protocolSioOption') },
    { value: 'rest', label: t('protocolRestOption') }
  ];

  const languageOptions = [
    { value: 'zh', label: '简体中文' },
    { value: 'en', label: 'English' },
    { value: 'ja', label: '日本語' }
  ];

  return (
    <div className="space-y-4 py-2 bg-surface-container-low p-4 rounded-md border border-outline-variant max-h-[500px] overflow-y-auto pr-1.5">
      <div>
        <MD3Select
          id="language"
          label={t('languageLabel')}
          value={localLang}
          onChange={(e) => handleLangChange(e.target.value as LangType)}
          options={languageOptions}
        />
      </div>

      <div>
        <MD3Select
          id="protocol"
          label={t('websocketProtocolLabel')}
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
          {t('crossfadeOverlapLabel')}
        </label>
        <MD3Slider
          id="crossfade"
          min={0.02}
          max={0.2}
          step={0.01}
          value={localCrossFadeOverlapSize}
          onImmediateChange={setLocalCrossFadeOverlapSize}
          onChange={async (val) => {
            await handleCrossFadeOverlapSizeChange(val);
          }}
          flushOnUnmount={true}
          showValue={true}
          valueFormatter={(val) => `${val.toFixed(2)} s`}
        />
      </div>

      <div className="space-y-3.5 pt-1.5 pb-2">
        <MD3Switch
          checked={appState.serverSetting.serverSetting.silenceFront === 1}
          onChange={handleSilenceFrontChange}
          label={t('silenceFrontLabel')}
        />

        <MD3Switch
          checked={appState.serverSetting.serverSetting.forceFp32 === 1}
          onChange={handleForceFp32Change}
          label={t('forceFp32Label')}
        />

        <MD3Switch
          checked={appState.serverSetting.serverSetting.disableJit === 1}
          onChange={handleDisableJitChange}
          label={t('disableJitLabel')}
        />

        <MD3Switch
          checked={appState.serverSetting.serverSetting.useONNX === 1}
          onChange={handleUseONNXChange}
          label={t('convertToOnnxLabel')}
        />

        <MD3Switch
          checked={appState.serverSetting.serverSetting.useTensorRT === 1}
          onChange={handleUseTensorRTChange}
          label={t('useTensorRtLabel')}
        />

        <MD3Switch
          checked={appState.setting.voiceChangerClientSetting.agcEnabled ?? false}
          onChange={handleAGCChange}
          label={t('agcLabel')}
        />
      </div>

      <div>
        <label htmlFor="protect" className={CSS_CLASSES.label}>
          {t('protectLabel')}
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
          flushOnUnmount={true}
          showValue={true}
          valueFormatter={(val) => val.toFixed(2)}
        />
      </div>

      <div className="border border-error/30 p-4 rounded-xl bg-error/5 space-y-4 mt-4">
        {/* Title */}
        <div className="flex items-center text-error text-xs font-bold uppercase tracking-wider pb-2 border-b border-error/10">
          <FontAwesomeIcon icon={faExclamationTriangle} className="mr-2 text-sm" />
          <span>{t('dangerZoneLabel')}</span>
        </div>

        {/* Row 1: Skip Passthrough Confirmation */}
        <div className="flex items-center justify-between py-1">
          <div className="flex flex-col space-y-0.5 max-w-[70%]">
            <span className="text-sm font-semibold text-on-surface">
              {t('skipPassthroughConfirmLabel')}
            </span>
            <span className="text-[11px] text-on-surface-variant/70">
              {t('skipPassthroughConfirmDesc')}
            </span>
          </div>
          <MD3Switch
            checked={appState.setting.voiceChangerClientSetting.passThroughConfirmationSkip ?? false}
            onChange={handlePassThroughConfirmationSkipChange}
          />
        </div>

        {/* Row 2: Reset Settings */}
        <div className="flex items-center justify-between py-2 border-t border-outline-variant/30">
          <div className="flex flex-col space-y-0.5 max-w-[70%]">
            <span className="text-sm font-semibold text-on-surface">
              {t('resetSettingsLabel')}
            </span>
            <span className="text-[11px] text-on-surface-variant/70">
              {t('resetSettingsDesc')}
            </span>
          </div>
          <button
            onClick={handleResetSettings}
            className="px-4 py-2 bg-error hover:bg-error/90 text-on-error text-xs font-semibold rounded-full shadow-elevation-1 hover:shadow-elevation-2 active:scale-97 transition-all duration-150 whitespace-nowrap"
          >
            {t('resetSettings')}
          </button>
        </div>
      </div>
    </div>
  );
}

export default SettingsView;
