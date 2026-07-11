import { UIContextType } from "../../context/UIContext";
import { AppGuiSettingState } from "../../scripts/useAppGuiSetting";
import MD3Select from "../Helpers/MD3Select";
import { ClientState } from "@dannadori/voice-changer-client-js";
import { F0Detector } from "@dannadori/voice-changer-client-js";

interface F0ExtractionProps {
  appState: ClientState;
  uiState: UIContextType;
  appGuiSettingState: AppGuiSettingState;
}

function F0Extraction({ appState, uiState, appGuiSettingState }: F0ExtractionProps) {
  // ---------------- Handlers ----------------

  // Handle F0 Detector Change
  const handleChangeF0Detector = async (value: string) => {
    uiState.startLoading(`Changing F0 Detector to ${value}`);
    await appState.serverSetting.updateServerSettings({
      ...appState.serverSetting?.serverSetting,
      f0Detector: value as F0Detector
    });
    uiState.stopLoading();
  };

  // ---------------- Functions ----------------

  // Generate F0 Detectors Options for Select
  const getF0DetOptions = () => {
    const pitchExtractors = appState.serverSetting.serverSetting.pitchExtractors || {};

    // Get all available extractors and filter for downloaded ones
    let extractors = Object.entries(pitchExtractors).filter(
      ([_, extractor]) => extractor.downloaded === true
    );

    // Filter for DirectML - only include ONNX models
    if (appGuiSettingState.serverInfo.edition.indexOf("DirectML") >= 0) {
      extractors = extractors.filter(([key]) => key.includes('_onnx'));
    }

    // If no downloaded extractors are available
    if (extractors.length === 0) {
      return [{ value: "", label: "No downloaded extractors" }];
    }

    // Map to options
    return extractors.map(([key, extractor]) => ({
      value: key,
      label: extractor.name
    }));
  };

  // ---------------- Render ----------------

  return (
    <div className="bg-surface-container-low p-3 rounded-md border border-outline-variant">
      <MD3Select
        id="f0Detector"
        label="Pitch Extraction Algorithm"
        value={appState.serverSetting?.serverSetting?.f0Detector ?? ''}
        disabled={uiState.isConverting}
        onChange={async (e) => {
          await handleChangeF0Detector(e.target.value);
        }}
        options={getF0DetOptions()}
      />
    </div>
  );
}

export default F0Extraction;