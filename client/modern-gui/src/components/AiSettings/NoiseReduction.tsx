import { ClientState } from "@dannadori/voice-changer-client-js";
import { CSS_CLASSES } from "../../styles/constants";
import { t } from "../../locales";
import MD3Switch from "../Helpers/MD3Switch";

interface NoiseReductionProps {
  appState: ClientState;
}

function NoiseReduction({ appState }: NoiseReductionProps) {
  const isServerAudio = appState.serverSetting.serverSetting.enableServerAudio === 1;

  const handleToggle = (key: 'echoCancel' | 'noiseSuppression', value: boolean) => {
    if (isServerAudio) {
      appState.serverSetting.updateServerSettings({ [key]: value });
    } else {
      appState.setVoiceChangerClientSetting((prev) => ({
        ...prev,
        [key]: value
      }));
    }
  };

  // Handle Noise Reduction 2 Change
  const handleChangeNoiseSuppression2 = (value: boolean) => {
    appState.setVoiceChangerClientSetting((prev) => ({
      ...prev,
      noiseSuppression2: value
    }));
  };

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
          onChange={(val) => handleToggle('echoCancel', val)}
          label={t('echoCancellationLabel')}
        />
        <MD3Switch
          checked={noiseSuppressionChecked}
          onChange={(val) => handleToggle('noiseSuppression', val)}
          label={t('noiseSuppressionLabel')}
        />
        <MD3Switch
          checked={appState.setting.voiceChangerClientSetting.noiseSuppression2 ?? false}
          onChange={handleChangeNoiseSuppression2}
          disabled={isServerAudio}
          label={t('noiseSuppression2Label')}
        />
      </div>
    </div>
  );
}

export default NoiseReduction;