import { ClientState } from "@dannadori/voice-changer-client-js";
import { CSS_CLASSES } from "../../styles/constants";
import { t } from "../../locales";
import MD3Switch from "../Helpers/MD3Switch";

interface NoiseReductionProps {
  appState: ClientState;
}

function NoiseReduction({ appState }: NoiseReductionProps) {
  const updateSyncedToggle = (key: 'echoCancel' | 'noiseSuppression', value: boolean) => {
    appState.setVoiceChangerClientSetting({
      ...appState.setting.voiceChangerClientSetting,
      [key]: value
    });
    appState.serverSetting.updateServerSettings({
      ...appState.serverSetting.serverSetting,
      [key]: value
    });
  };

  // Handle Noise Reduction 2 Change
  const handleChangeNoiseSuppression2 = (value: boolean) => {
    appState.setVoiceChangerClientSetting({
      ...appState.setting.voiceChangerClientSetting,
      noiseSuppression2: value
    });
  };

  const isServerAudio = appState.serverSetting.serverSetting.enableServerAudio === 1;
  const echoCancelChecked = (isServerAudio
    ? appState.serverSetting.serverSetting.echoCancel
    : appState.setting.voiceChangerClientSetting.echoCancel) ?? false;
  const noiseSuppressionChecked = (isServerAudio
    ? appState.serverSetting.serverSetting.noiseSuppression
    : appState.setting.voiceChangerClientSetting.noiseSuppression) ?? false;

  return (
    <div className="flex flex-col space-y-2">
      <label className={CSS_CLASSES.label}>{t('noiseReductionLabel')}</label>
      <div className="space-y-2.5 mt-1 bg-surface-container-low p-3 rounded-md border border-outline-variant">
        <MD3Switch
          checked={echoCancelChecked}
          onChange={(val) => updateSyncedToggle('echoCancel', val)}
          label={t('echoCancellationLabel')}
        />
        <MD3Switch
          checked={noiseSuppressionChecked}
          onChange={(val) => updateSyncedToggle('noiseSuppression', val)}
          label={t('noiseSuppressionLabel')}
        />
        <MD3Switch
          checked={appState.setting.voiceChangerClientSetting.noiseSuppression2 ?? false}
          onChange={handleChangeNoiseSuppression2}
          disabled={appState.serverSetting.serverSetting.enableServerAudio === 1}
          label={t('noiseSuppression2Label')}
        />
      </div>
    </div>
  );
}

export default NoiseReduction;