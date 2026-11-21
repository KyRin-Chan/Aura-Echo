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
  const [isSignalFresh, setIsSignalFresh] = useState<boolean>(false);

  // ---------------- Hooks ----------------

  // Set local silent threshold
  useEffect(() => {
    const st = appState.serverSetting?.serverSetting?.silentThreshold;
    if (st !== undefined) {
      setLocalSilentThreshold(st);
    }
  }, [appState.serverSetting?.serverSetting?.silentThreshold]);

  // Track whether performance stats are actively arriving while converting
  useEffect(() => {
    if (!uiState.isConverting) {
      setIsSignalFresh(false);
      return;
    }

    setIsSignalFresh(true);
    const timer = setTimeout(() => {
      setIsSignalFresh(false);
    }, 1000);

    return () => {
      clearTimeout(timer);
    };
  }, [appState.performance, uiState.isConverting]);

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
  const isActive = uiState.isConverting && isSignalFresh;
  const rawVol = isActive ? (appState.performance.vol || 0) : 0;
  const exactVolumeDb = rawVol > 1e-5 ? 20 * Math.log10(rawVol) : -90;
  const volumeDb = Math.max(Math.round(exactVolumeDb), -90);
  const volumePercent = Math.min(100, Math.max(0, ((exactVolumeDb + 90) / 90) * 100));
  const thresholdPercent = Math.min(100, Math.max(0, ((localSilentThreshold + 90) / 90) * 100));
  const isAboveThreshold = isActive && rawVol > 0 && exactVolumeDb >= localSilentThreshold;

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
        <div className="relative h-2 w-full mb-1 flex items-center">
          <div className="w-full h-2 rounded-full bg-surface-container-highest overflow-hidden">
            <div
              className={`h-full w-full origin-left rounded-full transition-transform duration-75 ${
                isAboveThreshold ? 'bg-primary' : 'bg-outline'
              }`}
              style={{ transform: `scaleX(${volumePercent / 100})` }}
            />
          </div>
          <div
            className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 h-3.5 w-0.5 rounded-full bg-tertiary pointer-events-none"
            style={{ left: `${thresholdPercent}%` }}
          />
        </div>
        <div className="text-[9px] font-mono text-on-surface-variant/60 flex justify-between mb-2 select-none">
          <span>-90</span>
          <span>-60</span>
          <span>-30</span>
          <span>0 dB</span>
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