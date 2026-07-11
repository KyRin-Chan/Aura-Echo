import { useState, useEffect } from "react";
import { RVCModelSlot, ClientState } from "@dannadori/voice-changer-client-js";
import { CSS_CLASSES } from "../../styles/constants";
import MD3Slider from "../Helpers/MD3Slider";
import MD3Select from "../Helpers/MD3Select";
import { useAppState } from "../../context/AppContext";

interface ModelSettingsProps {
  model: RVCModelSlot;
  handlePitchChange: (val: number) => void;
  handleFormatShiftChange: (val: number) => void;
  handleIndexRatioChange: (val: number) => void;
  handleSpeakerChange: (val: number) => void;
  setModel: (model: RVCModelSlot) => void;
}

function ModelSettings({
  model,
  handlePitchChange,
  handleFormatShiftChange,
  handleIndexRatioChange,
  handleSpeakerChange,
  setModel
}: ModelSettingsProps) {
  const appState = useAppState() as ClientState;

  // ---------------- State ----------------
  const [immediatePitch, setImmediatePitch] = useState<number>(
    appState.serverSetting?.serverSetting?.tran ?? 0
  );
  const [immediateFormant, setImmediateFormant] = useState<number>(
    appState.serverSetting?.serverSetting?.formantShift ?? 0
  );
  const [immediateIndexRatio, setImmediateIndexRatio] = useState<number>(
    appState.serverSetting?.serverSetting?.indexRatio ?? 0.5
  );

  useEffect(() => {
    if (appState.serverSetting?.serverSetting) {
      setImmediatePitch(appState.serverSetting.serverSetting.tran);
      setImmediateFormant(appState.serverSetting.serverSetting.formantShift);
      setImmediateIndexRatio(appState.serverSetting.serverSetting.indexRatio);
    }
  }, [
    appState.serverSetting?.serverSetting?.tran,
    appState.serverSetting?.serverSetting?.formantShift,
    appState.serverSetting?.serverSetting?.indexRatio
  ]);

  const speakerOptions =
    model && model.speakers && Object.keys(model.speakers).length > 0
      ? Object.entries(model.speakers).map(([id, name]) => ({
          value: Number(id),
          label: name as string
        }))
      : [{ value: 0, label: "No speakers" }];

  // ---------------- Render ----------------

  return (
    <div className={`space-y-4 bg-surface-container-low p-3 rounded-md border border-outline-variant ${!model ? 'opacity-50 pointer-events-none' : ''}`}>
      <div>
        <label htmlFor="pitch" className={CSS_CLASSES.label}>
          Pitch:
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
      <div>
        <label htmlFor="formatShift" className={CSS_CLASSES.label}>
          Formant Shift:
        </label>
        <MD3Slider
          id="formatShift"
          min={-2.0}
          max={2.0}
          step={0.01}
          value={appState.serverSetting?.serverSetting?.formantShift ?? 0}
          onChange={handleFormatShiftChange}
          onImmediateChange={setImmediateFormant}
          disabled={!model}
          showValue={true}
          valueFormatter={(val) => val.toFixed(2)}
        />
      </div>
      {model.indexFile !== '' && (
        <div>
          <label htmlFor="indexRatio" className={CSS_CLASSES.label}>
            Index Ratio:
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
            label="Speaker"
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
