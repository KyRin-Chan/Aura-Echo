import { useState, useEffect } from "react";
import { RVCModelSlot, ClientState } from "@dannadori/voice-changer-client-js";
import { CSS_CLASSES } from "../../styles/constants";
import MD3Slider from "../Helpers/MD3Slider";
import MD3Select from "../Helpers/MD3Select";
import { useAppState } from "../../context/AppContext";
import { t } from '../../locales';
import {
  FormantProfile,
  ModelFormantBinding,
  getAllFormantProfiles,
  getAllFormantBindings,
  saveFormantBinding,
  syncProfilesFromServer,
  syncBindingsFromServer,
  serializePitchStats,
  serializeProfileEnvelope
} from "../../utils/formantStorage";

interface ModelSettingsProps {
  model: RVCModelSlot;
  handlePitchChange: (val: number) => void;
  handleFormantShiftChange: (val: number) => void;
  handleIndexRatioChange: (val: number) => void;
  handleSpeakerChange: (val: number) => void;
  setModel: (model: RVCModelSlot) => void;
}

function ModelSettings({
  model,
  handlePitchChange,
  handleFormantShiftChange,
  handleIndexRatioChange,
  handleSpeakerChange,
  setModel
}: ModelSettingsProps) {
  const appState = useAppState() as ClientState;

  // ---------------- State ----------------
  const [immediatePitch, setImmediatePitch] = useState<number>(
    appState.serverSetting?.serverSetting?.tran ?? 0
  );
  const [immediateFormantShift, setImmediateFormantShift] = useState<number>(
    appState.serverSetting?.serverSetting?.formantShift ?? 0
  );
  const [immediateIndexRatio, setImmediateIndexRatio] = useState<number>(
    appState.serverSetting?.serverSetting?.indexRatio ?? 0.5
  );
  const [immediateStrength, setImmediateStrength] = useState<number>(
    appState.serverSetting?.serverSetting?.formantProfileStrength ?? 0.35
  );
  const [immediatePitchScale, setImmediatePitchScale] = useState<number>(
    appState.serverSetting?.serverSetting?.formantProfilePitchScale ?? 1.0
  );
  const [immediatePitchOffset, setImmediatePitchOffset] = useState<number>(
    appState.serverSetting?.serverSetting?.formantProfilePitchOffset ?? 0.0
  );
  const [immediatePitchMode, setImmediatePitchMode] = useState<'speech' | 'singing'>(
    appState.serverSetting?.serverSetting?.formantProfilePitchMode === 'singing' ? 'singing' : 'speech'
  );

  const [profiles, setProfiles] = useState<FormantProfile[]>([]);
  const [bindings, setBindings] = useState<Record<string | number, ModelFormantBinding>>({});

  useEffect(() => {
    const loadProfilesAndBindings = async () => {
      try {
        setProfiles(getAllFormantProfiles());
        setBindings(getAllFormantBindings());
        const [latestProfiles, latestBindings] = await Promise.all([
          syncProfilesFromServer(),
          syncBindingsFromServer()
        ]);
        setProfiles(latestProfiles);
        setBindings(latestBindings);
      } catch (e) {
        console.error('Error loading profiles or bindings:', e);
      }
    };
    loadProfilesAndBindings();
  }, [
    appState.serverSetting?.serverSetting?.modelSlotIndex,
    appState.serverSetting?.serverSetting?.formantProfileActive,
    appState.serverSetting?.serverSetting?.formantProfileInputEnvelope,
    appState.serverSetting?.serverSetting?.formantProfileTargetEnvelope,
    appState.serverSetting?.serverSetting?.formantProfilePitchStats
  ]);

  useEffect(() => {
    if (appState.serverSetting?.serverSetting) {
      setImmediatePitch(appState.serverSetting.serverSetting.tran);
      setImmediateFormantShift(appState.serverSetting.serverSetting.formantShift ?? 0);
      setImmediateIndexRatio(appState.serverSetting.serverSetting.indexRatio);
      setImmediateStrength(appState.serverSetting.serverSetting.formantProfileStrength ?? 0.35);
      setImmediatePitchScale(appState.serverSetting.serverSetting.formantProfilePitchScale ?? 1.0);
      setImmediatePitchOffset(appState.serverSetting.serverSetting.formantProfilePitchOffset ?? 0.0);
      setImmediatePitchMode(
        appState.serverSetting.serverSetting.formantProfilePitchMode === 'singing' ? 'singing' : 'speech'
      );
    }
  }, [
    appState.serverSetting?.serverSetting?.tran,
    appState.serverSetting?.serverSetting?.formantShift,
    appState.serverSetting?.serverSetting?.indexRatio,
    appState.serverSetting?.serverSetting?.formantProfileStrength,
    appState.serverSetting?.serverSetting?.formantProfilePitchScale,
    appState.serverSetting?.serverSetting?.formantProfilePitchOffset,
    appState.serverSetting?.serverSetting?.formantProfilePitchMode
  ]);

  const handleModeChange = (isCalibration: boolean) => {
    const activeSlot = appState.serverSetting?.serverSetting?.modelSlotIndex;
    let extraSettings: any = {};
    if (activeSlot !== undefined && activeSlot !== -1) {
      saveFormantBinding(activeSlot, { active: isCalibration });
      setBindings(getAllFormantBindings());

      if (isCalibration) {
        const currentBinding = bindings[activeSlot];
        if (currentBinding) {
          const inputProfile = profiles.find((p) => p.id === currentBinding.inputProfileId);
          const targetProfile = profiles.find((p) => p.id === currentBinding.targetProfileId);
          if (inputProfile && targetProfile) {
            extraSettings = {
              formantProfileTargetEnvelope: serializeProfileEnvelope(targetProfile),
              formantProfileTargetSr: targetProfile.sr,
              formantProfileInputEnvelope: serializeProfileEnvelope(inputProfile),
              formantProfileInputSr: inputProfile.sr,
              formantProfilePitchStats: serializePitchStats(inputProfile, targetProfile),
              formantProfilePitchScale: currentBinding.pitchScale ?? appState.serverSetting?.serverSetting?.formantProfilePitchScale ?? 1.0,
              formantProfilePitchOffset: currentBinding.pitchOffset ?? appState.serverSetting?.serverSetting?.formantProfilePitchOffset ?? 0.0,
              formantProfilePitchMode: currentBinding.pitchMode ?? appState.serverSetting?.serverSetting?.formantProfilePitchMode ?? 'speech'
            };
          }
        }
      }
    }

    appState.serverSetting.updateServerSettings({
      ...appState.serverSetting.serverSetting,
      formantProfileActive: isCalibration,
      ...extraSettings
    });
  };

  const handleBindInputProfileChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const inputProfileId = e.target.value;
    const activeSlot = appState.serverSetting?.serverSetting?.modelSlotIndex;
    if (activeSlot === undefined || activeSlot === -1) return;

    const currentBinding = bindings[activeSlot] || {};
    saveFormantBinding(activeSlot, {
      inputProfileId,
      active: currentBinding.active !== undefined ? currentBinding.active : true
    });
    setBindings(getAllFormantBindings());

    const pairedTargetProfile = currentBinding.targetProfileId
      ? profiles.find((p) => p.id === currentBinding.targetProfileId)
      : undefined;

    if (inputProfileId === "") {
      appState.serverSetting.updateServerSettings({
        ...appState.serverSetting.serverSetting,
        formantProfileInputEnvelope: "[]",
        formantProfileInputSr: 0,
        formantProfilePitchStats: "{}"
      });
    } else {
      const inputProfile = profiles.find((p) => p.id === inputProfileId);
      if (inputProfile) {
        appState.serverSetting.updateServerSettings({
          ...appState.serverSetting.serverSetting,
          formantProfileInputEnvelope: serializeProfileEnvelope(inputProfile),
          formantProfileInputSr: inputProfile.sr,
          formantProfilePitchStats: serializePitchStats(inputProfile, pairedTargetProfile)
        });
      }
    }
  };

  const handleBindTargetProfileChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const targetProfileId = e.target.value;
    const activeSlot = appState.serverSetting?.serverSetting?.modelSlotIndex;
    if (activeSlot === undefined || activeSlot === -1) return;

    const currentBinding = bindings[activeSlot] || {};
    saveFormantBinding(activeSlot, {
      targetProfileId,
      active: currentBinding.active !== undefined ? currentBinding.active : true
    });
    setBindings(getAllFormantBindings());

    const pairedInputProfile = currentBinding.inputProfileId
      ? profiles.find((p) => p.id === currentBinding.inputProfileId)
      : undefined;

    if (targetProfileId === "") {
      appState.serverSetting.updateServerSettings({
        ...appState.serverSetting.serverSetting,
        formantProfileTargetEnvelope: "[]",
        formantProfileTargetSr: 0,
        formantProfilePitchStats: "{}"
      });
    } else {
      const targetProfile = profiles.find((p) => p.id === targetProfileId);
      if (targetProfile) {
        appState.serverSetting.updateServerSettings({
          ...appState.serverSetting.serverSetting,
          formantProfileTargetEnvelope: serializeProfileEnvelope(targetProfile),
          formantProfileTargetSr: targetProfile.sr,
          formantProfilePitchStats: serializePitchStats(pairedInputProfile, targetProfile)
        });
      }
    }
  };

  const handleWarpFilterStrengthChange = (val: number) => {
    const activeSlot = appState.serverSetting?.serverSetting?.modelSlotIndex;
    if (activeSlot !== undefined && activeSlot !== -1) {
      saveFormantBinding(activeSlot, { strength: val });
      setBindings(getAllFormantBindings());
    }

    appState.serverSetting.updateServerSettings({
      ...appState.serverSetting.serverSetting,
      formantProfileStrength: val
    });
  };

  const handlePitchScaleChange = (val: number) => {
    const activeSlot = appState.serverSetting?.serverSetting?.modelSlotIndex;
    if (activeSlot !== undefined && activeSlot !== -1) {
      saveFormantBinding(activeSlot, { pitchScale: val });
      setBindings(getAllFormantBindings());
    }

    appState.serverSetting.updateServerSettings({
      ...appState.serverSetting.serverSetting,
      formantProfilePitchScale: val
    });
  };

  const handlePitchOffsetChange = (val: number) => {
    const activeSlot = appState.serverSetting?.serverSetting?.modelSlotIndex;
    if (activeSlot !== undefined && activeSlot !== -1) {
      saveFormantBinding(activeSlot, { pitchOffset: val });
      setBindings(getAllFormantBindings());
    }

    appState.serverSetting.updateServerSettings({
      ...appState.serverSetting.serverSetting,
      formantProfilePitchOffset: val
    });
  };

  const handlePitchModeChange = (mode: 'speech' | 'singing') => {
    setImmediatePitchMode(mode);
    const activeSlot = appState.serverSetting?.serverSetting?.modelSlotIndex;
    if (activeSlot !== undefined && activeSlot !== -1) {
      saveFormantBinding(activeSlot, { pitchMode: mode });
      setBindings(getAllFormantBindings());
    }

    appState.serverSetting.updateServerSettings({
      ...appState.serverSetting.serverSetting,
      formantProfilePitchMode: mode
    });
  };

  const isCalibrationMode = Boolean(appState.serverSetting?.serverSetting?.formantProfileActive);
  const activeSlotIndex = appState.serverSetting?.serverSetting?.modelSlotIndex;
  const activeBinding = activeSlotIndex !== undefined && activeSlotIndex !== -1 ? bindings[activeSlotIndex] : undefined;
  const boundInputProfile = activeBinding?.inputProfileId
    ? profiles.find((p) => p.id === activeBinding.inputProfileId)
    : undefined;
  const boundTargetProfile = activeBinding?.targetProfileId
    ? profiles.find((p) => p.id === activeBinding.targetProfileId)
    : undefined;
  const hasBoundPair = Boolean(boundInputProfile && boundTargetProfile);
  const hasPitchMappingStats = serializePitchStats(boundInputProfile, boundTargetProfile) !== "{}";

  const speakerOptions =
    model && model.speakers && Object.keys(model.speakers).length > 0
      ? Object.entries(model.speakers).map(([id, name]) => ({
          value: Number(id),
          label: name as string
        }))
      : [{ value: 0, label: t('noSpeakersLabel') }];

  // ---------------- Render ----------------

  return (
    <div className={`space-y-4 bg-surface-container/60 p-3 rounded-lg border border-outline-variant/50 ${!model ? 'opacity-50 pointer-events-none' : ''}`}>
      {/* Two-way exclusive mode toggle: Manual Tuning vs. Timbre Calibration */}
      <div>
        <label className={CSS_CLASSES.label}>
          {t('pitchControlModeLabel')}
        </label>
        <div className="inline-flex w-full rounded-full border border-outline overflow-hidden mt-1">
          <button
            type="button"
            onClick={() => handleModeChange(false)}
            disabled={!model}
            className={`flex-1 px-4 py-1.5 text-xs font-semibold transition-all duration-150 ${
              !isCalibrationMode
                ? "bg-secondary-container text-on-secondary-container"
                : "bg-transparent text-on-surface hover:bg-surface-variant/20"
            }`}
          >
            {t('manualTuningModeLabel')}
          </button>
          <div className="w-[1px] bg-outline" />
          <button
            type="button"
            onClick={() => handleModeChange(true)}
            disabled={!model}
            className={`flex-1 px-4 py-1.5 text-xs font-semibold transition-all duration-150 ${
              isCalibrationMode
                ? "bg-secondary-container text-on-secondary-container"
                : "bg-transparent text-on-surface hover:bg-surface-variant/20"
            }`}
          >
            {t('calibrationModeLabel')}
          </button>
        </div>
      </div>

      {!isCalibrationMode ? (
        <div className="space-y-4 animate-fadeIn">
          <div>
            <label htmlFor="pitch" className={CSS_CLASSES.label}>
              {t('pitchLabel')}
            </label>
            <MD3Slider
              id="pitch"
              min={-50}
              max={50}
              step={0.5}
              value={immediatePitch}
              onChange={handlePitchChange}
              onImmediateChange={setImmediatePitch}
              disabled={!model}
              showValue={true}
              valueFormatter={(val) => `${val > 0 ? '+' : ''}${val}`}
            />
          </div>
          <div>
            <label htmlFor="formantShift" className={CSS_CLASSES.label}>
              {t('formantShiftLabel')}
            </label>
            <MD3Slider
              id="formantShift"
              min={-3}
              max={3}
              step={0.05}
              value={immediateFormantShift}
              onChange={handleFormantShiftChange}
              onImmediateChange={setImmediateFormantShift}
              disabled={!model}
              showValue={true}
              valueFormatter={(val) => `${val > 0 ? '+' : ''}${val.toFixed(2)}`}
            />
          </div>
        </div>
      ) : (
        <div className="space-y-3 animate-fadeIn border-l-2 border-primary/20 pl-3">
          {/* Baseline (Model Output) Profile Dropdown */}
          <div>
            <label className={CSS_CLASSES.label}>
              {t('voiceAnalyzerInputProfileDropdown')}
            </label>
            <MD3Select
              options={[
                { value: "", label: t('voiceAnalyzerNoneOption') },
                ...profiles.filter(p => p.type === 'input').map((p) => ({
                  value: p.id,
                  label: !p.mode || p.mode === 'legacy_input' ? `${p.name} (${t('matchEqLegacyTag')})` : p.name
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

          {hasBoundPair && !hasPitchMappingStats && (
            <p className="text-[11px] text-on-surface-variant bg-surface-container-high/60 px-2.5 py-1.5 rounded-md border border-outline-variant/40">
              {t('legacyProfileNoPitchStatsHint')}
            </p>
          )}

          {/* Pitch Mapping Mode Toggle: Speech vs Singing */}
          <div>
            <label className={CSS_CLASSES.label}>
              {t('pitchMappingModeLabel')}
            </label>
            <div className="inline-flex w-full rounded-full border border-outline overflow-hidden mt-1">
              <button
                type="button"
                onClick={() => handlePitchModeChange('speech')}
                disabled={!model}
                className={`flex-1 px-3 py-1 text-xs font-semibold transition-all duration-150 ${
                  immediatePitchMode === 'speech'
                    ? "bg-secondary-container text-on-secondary-container"
                    : "bg-transparent text-on-surface hover:bg-surface-variant/20"
                }`}
              >
                {t('pitchModeSpeechLabel')}
              </button>
              <div className="w-[1px] bg-outline" />
              <button
                type="button"
                onClick={() => handlePitchModeChange('singing')}
                disabled={!model}
                className={`flex-1 px-3 py-1 text-xs font-semibold transition-all duration-150 ${
                  immediatePitchMode === 'singing'
                    ? "bg-secondary-container text-on-secondary-container"
                    : "bg-transparent text-on-surface hover:bg-surface-variant/20"
                }`}
              >
                {t('pitchModeSingingLabel')}
              </button>
            </div>
            {immediatePitchMode === 'singing' && (
              <p className="mt-1.5 text-[11px] text-on-surface-variant">
                {t('pitchModeSingingHint')}
              </p>
            )}
          </div>

          {/* Pitch Dynamic Range Scale Slider */}
          <div className={immediatePitchMode === 'singing' ? 'opacity-50' : ''}>
            <label htmlFor="formantProfilePitchScale" className={CSS_CLASSES.label}>
              {t('pitchDynamicScaleLabel')}
            </label>
            <MD3Slider
              id="formantProfilePitchScale"
              min={0.5}
              max={1.5}
              step={0.05}
              value={immediatePitchMode === 'singing' ? 1.0 : immediatePitchScale}
              onChange={handlePitchScaleChange}
              onImmediateChange={setImmediatePitchScale}
              disabled={!model || immediatePitchMode === 'singing'}
              showValue={true}
              valueFormatter={(val) => `${Math.round(val * 100)}%`}
            />
          </div>

          {/* Pitch Fine-Tune Offset Slider */}
          <div>
            <label htmlFor="formantProfilePitchOffset" className={CSS_CLASSES.label}>
              {t('pitchFineTuneLabel')}
            </label>
            <MD3Slider
              id="formantProfilePitchOffset"
              min={-6}
              max={6}
              step={0.5}
              value={immediatePitchOffset}
              onChange={handlePitchOffsetChange}
              onImmediateChange={setImmediatePitchOffset}
              disabled={!model}
              showValue={true}
              valueFormatter={(val) => `${val > 0 ? '+' : ''}${val.toFixed(1)}`}
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
              value={immediateStrength}
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
