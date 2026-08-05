import { UIContextType } from "../../context/UIContext";
import MD3Select from "../Helpers/MD3Select";
import { ClientState } from "@dannadori/voice-changer-client-js";
import { t } from "../../locales";

interface VocoderSelectionProps {
  appState: ClientState;
  uiState: UIContextType;
}

function VocoderSelection({ appState, uiState }: VocoderSelectionProps) {
  // Handle Vocoder Change
  const handleChangeVocoder = async (value: string) => {
    uiState.startLoading(`${t('changingVocoder')}: ${value}`);
    await appState.serverSetting.updateServerSettings({
      ...appState.serverSetting?.serverSetting,
      vocoderType: value
    });
    uiState.stopLoading();
  };

  // Generate Vocoder Options
  const getVocoderOptions = () => {
    const vocoders = appState.serverSetting.serverSetting.vocoders || {};
    const options = [
      { value: "embedded", label: t("defaultEmbeddedVocoder") }
    ];

    Object.entries(vocoders).forEach(([key, vocoder]) => {
      if (vocoder.downloaded) {
        options.push({
          value: key,
          label: vocoder.name
        });
      }
    });

    return options;
  };

  return (
    <div className="bg-surface-container-low p-3 rounded-md border border-outline-variant">
      <MD3Select
        id="vocoderType"
        label={t('vocoderLabel')}
        value={appState.serverSetting?.serverSetting?.vocoderType ?? 'embedded'}
        disabled={uiState.isConverting}
        onChange={async (e) => {
          await handleChangeVocoder(e.target.value);
        }}
        options={getVocoderOptions()}
      />
    </div>
  );
}

export default VocoderSelection;
