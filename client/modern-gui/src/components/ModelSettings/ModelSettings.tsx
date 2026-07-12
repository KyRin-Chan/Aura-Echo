import { useState, useEffect } from "react";
import { RVCModelSlot, ClientState } from "@dannadori/voice-changer-client-js";
import { CSS_CLASSES } from "../../styles/constants";
import MD3Slider from "../Helpers/MD3Slider";
import MD3Select from "../Helpers/MD3Select";
import { useAppState } from "../../context/AppContext";
import { t } from '../../locales';

interface ModelSettingsProps {
  model: RVCModelSlot;
  handlePitchChange: (val: number) => void;
  handleIndexRatioChange: (val: number) => void;
  handleSpeakerChange: (val: number) => void;
  setModel: (model: RVCModelSlot) => void;
}

function ModelSettings({
  model,
  handlePitchChange,
  handleIndexRatioChange,
  handleSpeakerChange,
  setModel
}: ModelSettingsProps) {
  const appState = useAppState() as ClientState;

  // ---------------- State ----------------
  const [immediatePitch, setImmediatePitch] = useState<number>(
    appState.serverSetting?.serverSetting?.tran ?? 0
  );
  const [immediateIndexRatio, setImmediateIndexRatio] = useState<number>(
    appState.serverSetting?.serverSetting?.indexRatio ?? 0.5
  );
  const [immediateStrength, setImmediateStrength] = useState<number>(
    appState.serverSetting?.serverSetting?.formantProfileStrength ?? 0.35
  );

  useEffect(() => {
    if (appState.serverSetting?.serverSetting) {
      setImmediatePitch(appState.serverSetting.serverSetting.tran);
      setImmediateIndexRatio(appState.serverSetting.serverSetting.indexRatio);
      setImmediateStrength(appState.serverSetting.serverSetting.formantProfileStrength ?? 0.35);
    }
  }, [
    appState.serverSetting?.serverSetting?.tran,
    appState.serverSetting?.serverSetting?.indexRatio,
    appState.serverSetting?.serverSetting?.formantProfileStrength
  ]);

  const handleWarpFilterActiveChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const active = e.target.checked;
    appState.serverSetting.updateServerSettings({
      ...appState.serverSetting.serverSetting,
      formantProfileActive: active
    });
  };

  const handleWarpFilterStrengthChange = (val: number) => {
    appState.serverSetting.updateServerSettings({
      ...appState.serverSetting.serverSetting,
      formantProfileStrength: val
    });
  };

  const speakerOptions =
    model && model.speakers && Object.keys(model.speakers).length > 0
      ? Object.entries(model.speakers).map(([id, name]) => ({
          value: Number(id),
          label: name as string
        }))
      : [{ value: 0, label: t('noSpeakersLabel') }];

  // ---------------- Render ----------------

  return (
    <div className={`space-y-4 bg-surface-container-low p-3 rounded-md border border-outline-variant ${!model ? 'opacity-50 pointer-events-none' : ''}`}>
      <div>
        <label htmlFor="pitch" className={CSS_CLASSES.label}>
          {t('pitchLabel')}
        </label>
        <MD3Slider
          id="pitch"
          min={-50}
          max={50}
          step={0.5}
          value={appState.serverSetting?.serverSetting?.tran ?? 0}
          onChange={handlePitchChange}
          onImmediateChange={setImmediatePitch}
          disabled={!model}
          showValue={true}
          valueFormatter={(val) => `${val > 0 ? '+' : ''}${val}`}
        />
      </div>
      <div className="flex items-center space-x-3 py-1">
        <label className="flex items-center space-x-2 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={appState.serverSetting?.serverSetting?.formantProfileActive ?? false}
            onChange={handleWarpFilterActiveChange}
            className="w-4 h-4 rounded text-primary focus:ring-primary border-outline-variant bg-surface-container"
          />
          <span className="text-xs font-semibold text-on-surface">
            {t('timbreWarpFilterLabel')}
          </span>
        </label>
      </div>
      {appState.serverSetting?.serverSetting?.formantProfileActive && (
        <div className="animate-fadeIn">
          <label htmlFor="formantProfileStrength" className={CSS_CLASSES.label}>
            {t('timbreWarpStrengthLabel')}
          </label>
          <MD3Slider
            id="formantProfileStrength"
            min={0.0}
            max={1.0}
            step={0.05}
            value={appState.serverSetting?.serverSetting?.formantProfileStrength ?? 0.35}
            onChange={handleWarpFilterStrengthChange}
            onImmediateChange={setImmediateStrength}
            disabled={!model}
            showValue={true}
            valueFormatter={(val) => val.toFixed(2)}
          />
        </div>
      )}
      {model.indexFile !== '' && (
        <div>
          <label htmlFor="indexRatio" className={CSS_CLASSES.label}>
            {t('indexRatioLabel')}
          </label>
          <MD3Slider
            id="indexRatio"
            min={0}
            max={1}
            step={0.01}
            value={appState.serverSetting?.serverSetting?.indexRatio ?? 0.5}
            onChange={handleIndexRatioChange}
            onImmediateChange={setImmediateIndexRatio}
            disabled={!model}
            showValue={true}
            valueFormatter={(val) => val.toFixed(2)}
          />
        </div>
      )}
      {model.speakers && Object.keys(model.speakers).length > 1 && (
        <div className="pt-1">
          <MD3Select
            id="speaker"
            label={t('speakerLabel')}
            disabled={!model || !model.speakers || Object.keys(model.speakers).length === 0}
            value={appState.serverSetting?.serverSetting?.dstId ?? 0}
            onChange={(e) => handleSpeakerChange(Number(e.target.value))}
            options={speakerOptions}
          />
        </div>
      )}
    </div>
  );
}

export default ModelSettings;
