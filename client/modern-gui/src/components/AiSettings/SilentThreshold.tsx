import DebouncedSlider from "../Helpers/DebouncedSlider";
import { ClientState } from "@dannadori/voice-changer-client-js";
import { CSS_CLASSES } from "../../styles/constants";
import { useEffect, useState } from "react";
import { UIContextType } from "../../context/UIContext";

interface SilenceThresholdProps {
  appState: ClientState;
  uiState: UIContextType;
}

function SilentThreshold({ appState, uiState }: SilenceThresholdProps) {
  // ---------------- States ----------------
  const [localSilentThreshold, setLocalSilentThreshold] = useState<number>(
    appState.serverSetting?.serverSetting?.silentThreshold ?? -75
  );

  // ---------------- Hooks ----------------

  // Set local silent threshold
  useEffect(() => {
    const st = appState.serverSetting?.serverSetting?.silentThreshold;
    setLocalSilentThreshold(st);
  }, [appState.serverSetting?.serverSetting?.silentThreshold]);

  // ---------------- Handlers ----------------

  // Handle silent threshold change
  const handleChangeSilentThreshold = (value: number) => {
    setLocalSilentThreshold(value);
    appState.serverSetting.updateServerSettings({
      ...appState.serverSetting?.serverSetting,
      silentThreshold: value
    });
  };

  // Handle power saving mode change
  const handlePowerSavingChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const enabled = e.target.checked ? 1 : 0;
    appState.serverSetting.updateServerSettings({
      ...appState.serverSetting?.serverSetting,
      powerSavingMode: enabled
    });
  };

  // ---------------- Render ----------------

  return (
    <div>
      <label htmlFor="inSens" className={CSS_CLASSES.label}>Input Sensitivity (In. Sens):</label>
      <DebouncedSlider
        id="inSens"
        name="inSens"
        min={-90}
        max={-60}
        step={1}
        value={localSilentThreshold}
        className={CSS_CLASSES.range}
        onImmediateChange={setLocalSilentThreshold}
        onChange={handleChangeSilentThreshold}
      />
      <p className={CSS_CLASSES.sliderValue}>{localSilentThreshold} dB</p>
      
      <div className="mt-3 flex items-center">
        <input
          id="powerSavingMode"
          type="checkbox"
          checked={appState.serverSetting?.serverSetting?.powerSavingMode === 1}
          onChange={handlePowerSavingChange}
          className={CSS_CLASSES.checkbox}
        />
        <label htmlFor="powerSavingMode" className={CSS_CLASSES.checkboxLabel}>
          Power Saving Mode (Skip inference on silence)
        </label>
      </div>
    </div>
  );
}

export default SilentThreshold;