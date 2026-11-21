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

  // Calculate real-time input volume in dB (-90 to 0)
  const volumeDb = Math.max(Math.round(20 * Math.log10(appState.performance.vol || 0.00001)), -90);
  const volumePercent = Math.min(100, Math.max(0, ((volumeDb + 90) / 90) * 100));
  const thresholdPercent = Math.min(100, Math.max(0, ((localSilentThreshold + 90) / 90) * 100));
  const isAboveThreshold = volumeDb >= localSilentThreshold;

  // ---------------- Render ----------------

  return (
    <div className="flex flex-col space-y-3.5 bg-surface-container-low p-3 rounded-md border border-outline-variant">
      <div>
        <div className="flex items-center justify-between mb-1">
          <label htmlFor="inSens" className={`${CSS_CLASSES.label} mb-0`}>
            {t('silentThresholdDesc')}
          </label>
          <div className="flex items-center space-x-1.5">
            <span
              className={`inline-block w-2 h-2 rounded-full transition-colors ${
                isAboveThreshold ? 'bg-primary' : 'bg-outline-variant'
              }`}
            />
            <span className="text-[11px] font-mono tabular-nums text-on-surface-variant">
              {volumeDb} dB
            </span>
          </div>
        </div>
        <div className="relative h-1.5 w-full rounded-full bg-surface-container-highest overflow-hidden mb-2.5">
          <div
            className={`h-full transition-all duration-75 rounded-full ${
              isAboveThreshold ? 'bg-primary' : 'bg-outline'
            }`}
            style={{ width: `${volumePercent}%` }}
          />
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-tertiary"
            style={{ left: `${thresholdPercent}%` }}
          />
        </div>
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