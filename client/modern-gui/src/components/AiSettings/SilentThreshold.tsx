import MD3Slider from "../Helpers/MD3Slider";
import MD3Switch from "../Helpers/MD3Switch";
import { ClientState } from "@dannadori/voice-changer-client-js";
import { CSS_CLASSES } from "../../styles/constants";
import { useEffect, useState } from "react";
import { UIContextType } from "../../context/UIContext";
import { t } from "../../locales";

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
    if (st !== undefined) {
      setLocalSilentThreshold(st);
    }
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
  const handlePowerSavingChange = (checked: boolean) => {
    const enabled = checked ? 1 : 0;
    appState.serverSetting.updateServerSettings({
      ...appState.serverSetting?.serverSetting,
      powerSavingMode: enabled
    });
  };

  // ---------------- Render ----------------

  return (
    <div className="flex flex-col space-y-3.5 bg-surface-container-low p-3 rounded-md border border-outline-variant">
      <div>
        <label htmlFor="inSens" className={CSS_CLASSES.label}>
          {t('silentThresholdDesc')}
        </label>
        <MD3Slider
          id="inSens"
          min={-90}
          max={-60}
          step={1}
          value={localSilentThreshold}
          onImmediateChange={setLocalSilentThreshold}
          onChange={handleChangeSilentThreshold}
          showValue={true}
          valueFormatter={(val) => `${val} dB`}
        />
      </div>

      <div className="pt-1">
        <MD3Switch
          id="powerSavingMode"
          checked={appState.serverSetting?.serverSetting?.powerSavingMode === 1}
          onChange={handlePowerSavingChange}
          label={t('powerSavingModeLabel')}
        />
      </div>
    </div>
  );
}

export default SilentThreshold;