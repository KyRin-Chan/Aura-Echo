import { useState, useEffect } from "react";
import { RVCModelSlot, ClientState } from "@dannadori/voice-changer-client-js";
import { CSS_CLASSES } from "../../styles/constants";
import MD3Slider from "../Helpers/MD3Slider";
import MD3Select from "../Helpers/MD3Select";
import MD3Switch from "../Helpers/MD3Switch";
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

  const [profiles, setProfiles] = useState<any[]>([]);
  const [bindings, setBindings] = useState<Record<number, any>>({});

  useEffect(() => {
    const loadProfilesAndBindings = () => {
      try {
        const storedProfilesStr = localStorage.getItem('formant_profiles') || '[]';
        setProfiles(JSON.parse(storedProfilesStr));

        const storedBindingsStr = localStorage.getItem('model_formant_bindings') || '{}';
        setBindings(JSON.parse(storedBindingsStr));
      } catch (e) {
        console.error('Error loading profiles or bindings:', e);
      }
    };
    loadProfilesAndBindings();
  }, [
    appState.serverSetting?.serverSetting?.modelSlotIndex,
    appState.serverSetting?.serverSetting?.formantProfileActive
  ]);

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

  const handleWarpSwitchChange = (checked: boolean) => {
    const activeSlot = appState.serverSetting?.serverSetting?.modelSlotIndex;
    let extraSettings: any = {};
    if (checked && activeSlot !== undefined && activeSlot !== -1) {
      const currentBinding = bindings[activeSlot];
      if (currentBinding) {
        const inputProfile = profiles.find((p) => p.id === currentBinding.inputProfileId);
        const targetProfile = profiles.find((p) => p.id === currentBinding.targetProfileId);
        if (inputProfile && targetProfile) {
          extraSettings = {
            formantProfileTargetEnvelope: JSON.stringify(targetProfile.envelope),
            formantProfileTargetSr: targetProfile.sr,
            formantProfileInputEnvelope: JSON.stringify(inputProfile.envelope),
            formantProfileInputSr: inputProfile.sr
          };
        }
      }
    }

    appState.serverSetting.updateServerSettings({
      ...appState.serverSetting.serverSetting,
      formantProfileActive: checked,
      ...extraSettings
    });
  };

  const handleBindInputProfileChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const inputProfileId = e.target.value;
    const activeSlot = appState.serverSetting?.serverSetting?.modelSlotIndex;
    if (activeSlot === undefined || activeSlot === -1) return;

    const currentBinding = bindings[activeSlot] || {};
    const newBinding = {
      ...currentBinding,
      inputProfileId
    };
    const updatedBindings = { ...bindings, [activeSlot]: newBinding };
    localStorage.setItem('model_formant_bindings', JSON.stringify(updatedBindings));
    setBindings(updatedBindings);

    if (inputProfileId === "") {
      appState.serverSetting.updateServerSettings({
        ...appState.serverSetting.serverSetting,
        formantProfileInputEnvelope: "[]",
        formantProfileInputSr: 0
      });
    } else {
      const inputProfile = profiles.find((p) => p.id === inputProfileId);
      if (inputProfile) {
        appState.serverSetting.updateServerSettings({
          ...appState.serverSetting.serverSetting,
          formantProfileInputEnvelope: JSON.stringify(inputProfile.envelope),
          formantProfileInputSr: inputProfile.sr
        });
      }
    }
  };

  const handleBindTargetProfileChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const targetProfileId = e.target.value;
    const activeSlot = appState.serverSetting?.serverSetting?.modelSlotIndex;
    if (activeSlot === undefined || activeSlot === -1) return;

    const currentBinding = bindings[activeSlot] || {};
    const newBinding = {
      ...currentBinding,
      targetProfileId
    };
    const updatedBindings = { ...bindings, [activeSlot]: newBinding };
    localStorage.setItem('model_formant_bindings', JSON.stringify(updatedBindings));
    setBindings(updatedBindings);

    if (targetProfileId === "") {
      appState.serverSetting.updateServerSettings({
        ...appState.serverSetting.serverSetting,
        formantProfileTargetEnvelope: "[]",
        formantProfileTargetSr: 0
      });
    } else {
      const targetProfile = profiles.find((p) => p.id === targetProfileId);
      if (targetProfile) {
        appState.serverSetting.updateServerSettings({
          ...appState.serverSetting.serverSetting,
          formantProfileTargetEnvelope: JSON.stringify(targetProfile.envelope),
          formantProfileTargetSr: targetProfile.sr
        });
      }
    }
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
        <MD3Switch
          checked={appState.serverSetting?.serverSetting?.formantProfileActive ?? false}
          onChange={handleWarpSwitchChange}
          label={t('timbreWarpFilterLabel')}
        />
      </div>
      {appState.serverSetting?.serverSetting?.formantProfileActive && (
        <div className="space-y-3 animate-fadeIn border-l-2 border-primary/20 pl-3">
          {/* Input Profile Dropdown */}
          <div>
            <label className={CSS_CLASSES.label}>
              {t('voiceAnalyzerInputProfileDropdown')}
            </label>
            <MD3Select
              options={[
                { value: "", label: t('voiceAnalyzerNoneOption') },
                ...profiles.filter(p => p.type === 'input').map((p) => ({
                  value: p.id,
                  label: p.name
                }))
              ]}
              value={bindings[appState.serverSetting?.serverSetting?.modelSlotIndex]?.inputProfileId || ""}
              onChange={handleBindInputProfileChange}
            />
          </div>

          {/* Target Profile Dropdown */}
          <div>
            <label className={CSS_CLASSES.label}>
              {t('voiceAnalyzerTargetProfileDropdown')}
            </label>
            <MD3Select
              options={[
                { value: "", label: t('voiceAnalyzerNoneOption') },
                ...profiles.filter(p => p.type === 'target').map((p) => ({
                  value: p.id,
                  label: p.name
                }))
              ]}
              value={bindings[appState.serverSetting?.serverSetting?.modelSlotIndex]?.targetProfileId || ""}
              onChange={handleBindTargetProfileChange}
            />
          </div>

          {/* Correction Strength Slider */}
          <div>
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
