import { ClientState } from "@dannadori/voice-changer-client-js";
import { CSS_CLASSES } from "../../styles/constants";
import { t } from "../../locales";
import MD3Switch from "../Helpers/MD3Switch";

interface NoiseReductionProps {
  appState: ClientState;
}

function NoiseReduction({ appState }: NoiseReductionProps) {
  const isServerAudio = appState.serverSetting.serverSetting.enableServerAudio === 1;

  const handleToggle = (key: 'echoCancel' | 'noiseSuppression' | 'noiseSuppression2', value: boolean) => {
    if (isServerAudio) {
      appState.serverSetting.updateServerSettings({ [key]: value });
    } else {
      appState.setVoiceChangerClientSetting((prev) => ({
        ...prev,
        [key]: value
      }));
    }
  };

  const echoCancelChecked = (isServerAudio
    ? appState.serverSetting.serverSetting.echoCancel
    : appState.setting.voiceChangerClientSetting.echoCancel) ?? false;
  const noiseSuppressionChecked = (isServerAudio
    ? appState.serverSetting.serverSetting.noiseSuppression
    : appState.setting.voiceChangerClientSetting.noiseSuppression) ?? false;
  const noiseSuppression2Checked = (isServerAudio
    ? appState.serverSetting.serverSetting.noiseSuppression2
    : appState.setting.voiceChangerClientSetting.noiseSuppression2) ?? false;

  return (
    <div className="flex flex-col space-y-2">
      <label className={CSS_CLASSES.label}>{t('noiseReductionLabel')}</label>
      <div className="space-y-2.5 mt-1 bg-surface-container/60 p-3 rounded-lg border border-outline-variant/50">
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
          checked={noiseSuppression2Checked}
          onChange={(val) => handleToggle('noiseSuppression2', val)}
          label={t('noiseSuppression2Label')}
        />
      </div>
    </div>
  );
}

export default NoiseReduction;