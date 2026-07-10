import { useState, useEffect } from "react";
import { RVCModelSlot, ClientState } from "@dannadori/voice-changer-client-js";
import { CSS_CLASSES } from "../../styles/constants"
import DebouncedSlider from "../Helpers/DebouncedSlider"
import { useAppState } from "../../context/AppContext"

interface ModelSettingsProps {
  model: RVCModelSlot;
  handlePitchChange: (val: number) => void;
  handleFormatShiftChange: (val: number) => void;
  handleIndexRatioChange: (val: number) => void;
  handleSpeakerChange: (val: number) => void;
  setModel: (model: RVCModelSlot) => void;
}

function ModelSettings({ model, handlePitchChange, handleFormatShiftChange, handleIndexRatioChange, handleSpeakerChange, setModel }: ModelSettingsProps) {
  const appState = useAppState() as ClientState;
  
  // ---------------- State ----------------
  const [immediatePitch, setImmediatePitch] = useState<number>(appState.serverSetting?.serverSetting?.tran ?? 0);
  const [immediateFormant, setImmediateFormant] = useState<number>(appState.serverSetting?.serverSetting?.formantShift ?? 0);
  const [immediateIndexRatio, setImmediateIndexRatio] = useState<number>(appState.serverSetting?.serverSetting?.indexRatio ?? 0.5);

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

  let speakerOptions: JSX.Element[] = [];
  if (model && model.speakers && Object.keys(model.speakers).length > 0) {
    speakerOptions = Object.entries(model.speakers).map(([id, name]) => (
      <option key={id} value={id}>{name as string}</option>
    ));
  } else {
    speakerOptions = [<option key="no-speakers" value={0} disabled>No speakers</option>];
  }

  // ---------------- Render ----------------

  return (
    <div className={`space-y-4 ${!model ? 'opacity-50 pointer-events-none' : ''}`}>
      <div>
        <label htmlFor="pitch" className={CSS_CLASSES.label}>Pitch:</label>
        <DebouncedSlider
          id="pitch"
          name="pitch"
          min={-50}
          max={50}
          step={0.5}
          value={appState.serverSetting?.serverSetting?.tran ?? 0}
          onChange={handlePitchChange}
          onImmediateChange={setImmediatePitch}
          className={CSS_CLASSES.range}
          disabled={!model}
        />
        <p className={CSS_CLASSES.sliderValue}>{immediatePitch}</p>
      </div>
      <div>
        <label htmlFor="formatShift" className={CSS_CLASSES.label}>Formant Shift:</label>
        <DebouncedSlider
          id="formatShift"
          name="formatShift"
          min={-2.0}
          max={2.0}
          step={0.01}
          value={appState.serverSetting?.serverSetting?.formantShift ?? 0}
          onChange={handleFormatShiftChange}
          onImmediateChange={setImmediateFormant}
          className={CSS_CLASSES.range}
          disabled={!model}
        />
        <p className={CSS_CLASSES.sliderValue}>{immediateFormant.toFixed(2)}</p>
      </div>
      {model.indexFile !== "" && (
        <div>
          <label htmlFor="indexRatio" className={CSS_CLASSES.label}>Index Ratio:</label>
          <DebouncedSlider
            id="indexRatio"
            name="indexRatio"
            min={0}
            max={1}
            step={0.01}
            value={appState.serverSetting?.serverSetting?.indexRatio ?? 0.5}
            onChange={handleIndexRatioChange}
            onImmediateChange={setImmediateIndexRatio}
            className={CSS_CLASSES.range}
            disabled={!model}
          />
          <p className={CSS_CLASSES.sliderValue}>{immediateIndexRatio.toFixed(2)}</p>
        </div>
      )}
      {
        // Only show speaker selection if there is more than one speaker
        model.speakers && Object.keys(model.speakers).length > 1 && (
          <div className="flex items-center space-x-2">
            <label htmlFor="speaker" className={CSS_CLASSES.label}>Speaker:</label>
            <select
              id="speaker"
              name="speaker"
              className={CSS_CLASSES.select}
              disabled={!model || !model.speakers || Object.keys(model.speakers).length === 0}
              value={appState.serverSetting?.serverSetting?.dstId ?? 0}
              onChange={(e) => handleSpeakerChange(Number(e.target.value))}
            >
              {speakerOptions}
            </select>
          </div>
        )
      }
    </div>
  )
}

export default ModelSettings;
