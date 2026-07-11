import { ClientState } from "@dannadori/voice-changer-client-js";
import { CSS_CLASSES } from "../../styles/constants";
import { t } from "../../locales";
import MD3Switch from "../Helpers/MD3Switch";

interface NoiseReductionProps {
  appState: ClientState;
}

function NoiseReduction({ appState }: NoiseReductionProps) {
  // Handle Noise Reduction Change
  const handleChangeNoiseSuppression = (value: boolean) => {
    appState.setVoiceChangerClientSetting({
      ...appState.setting.voiceChangerClientSetting,
      noiseSuppression: value
    });
  };

  // Handle Noise Reduction 2 Change
  const handleChangeNoiseSuppression2 = (value: boolean) => {
    appState.setVoiceChangerClientSetting({
      ...appState.setting.voiceChangerClientSetting,
      noiseSuppression2: value
    });
  };

  // Handle Echo Cancel Change
  const handleChangeEchoCancel = (value: boolean) => {
    appState.setVoiceChangerClientSetting({
      ...appState.setting.voiceChangerClientSetting,
      echoCancel: value
    });
  };

  return (
    <div className="flex flex-col space-y-2">
      <label className={CSS_CLASSES.label}>{t('noiseReductionLabel')}</label>
      <div className="space-y-2.5 mt-1 bg-surface-container-low p-3 rounded-md border border-outline-variant">
        <MD3Switch
          checked={appState.setting.voiceChangerClientSetting.echoCancel ?? false}
          onChange={handleChangeEchoCancel}
          disabled={appState.serverSetting.serverSetting.enableServerAudio === 1}
          label={t('echoCancellationLabel')}
        />
        <MD3Switch
          checked={appState.setting.voiceChangerClientSetting.noiseSuppression ?? false}
          onChange={handleChangeNoiseSuppression}
          disabled={appState.serverSetting.serverSetting.enableServerAudio === 1}
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