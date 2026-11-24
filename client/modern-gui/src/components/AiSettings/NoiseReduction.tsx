import { ClientState } from "@dannadori/voice-changer-client-js";
import { CSS_CLASSES } from "../../styles/constants";
import { t } from "../../locales";
import MD3Select from "../Helpers/MD3Select";
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
  const noiseSuppression2Strength = appState.serverSetting.serverSetting.noiseSuppression2Strength || 'medium';

  const strengthOptions = [
    { value: 'strong', label: t('noiseSuppression2StrengthStrong') },
    { value: 'medium', label: t('noiseSuppression2StrengthMedium') },
    { value: 'weak', label: t('noiseSuppression2StrengthWeak') },
  ];

  return (
    <div className="bg-surface-container/60 p-3 rounded-lg border border-outline-variant/50 space-y-2.5">
      <div className="text-xs font-semibold text-on-surface-variant pb-1.5 border-b border-outline-variant/30">
        {t('noiseReductionLabel')}
      </div>
      <MD3Switch
        checked={echoCancelChecked}
        onChange={(val) => handleToggle('echoCancel', val)}
        label={t('echoCancellationLabel')}
        size="compact"
        labelPosition="left"
      />
      <MD3Switch
        checked={noiseSuppressionChecked}
        onChange={(val) => handleToggle('noiseSuppression', val)}
        label={t('noiseSuppressionLabel')}
        size="compact"
        labelPosition="left"
      />
      <MD3Switch
        checked={noiseSuppression2Checked}
        onChange={(val) => handleToggle('noiseSuppression2', val)}
        label={isServerAudio ? t('noiseSuppression2ServerLabel') : t('noiseSuppression2ClientLabel')}
        size="compact"
        labelPosition="left"
      />
      {isServerAudio && noiseSuppression2Checked && (
        <div className="pt-1">
          <MD3Select
            id="dfn3-strength-select"
            label={t('noiseSuppression2StrengthLabel')}
            value={noiseSuppression2Strength}
            options={strengthOptions}
            onChange={(e) => {
              appState.serverSetting.updateServerSettings({
                noiseSuppression2Strength: e.target.value as 'strong' | 'medium' | 'weak'
              });
            }}
          />
        </div>
      )}
    </div>
  );
}

export default NoiseReduction;