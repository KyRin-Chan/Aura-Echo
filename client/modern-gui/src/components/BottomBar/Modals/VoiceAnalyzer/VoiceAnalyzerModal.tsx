import { useState, JSX, useEffect } from 'react';
import { ClientState } from '@dannadori/voice-changer-client-js';
import { CSS_CLASSES } from '../../../../styles/constants';
import GenericModal from '../../../Modals/GenericModal';
import { UIContextType } from '../../../../context/UIContext';
import { t } from '../../../../locales';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faSync, faMicrophone, faCheck, faMusic } from '@fortawesome/free-solid-svg-icons';

interface VoiceAnalyzerModalProps {
  appState: ClientState;
  guiState: UIContextType;
  showVoiceAnalyzer: boolean;
  setShowVoiceAnalyzer: (showVoiceAnalyzer: boolean) => void;
}

interface AnalysisResult {
  success: boolean;
  target_f0: number;
  input_f0: number;
  target_centroid: number;
  input_centroid: number;
  recommended_pitch: number;
  recommended_formant_shift: number;
  target_envelope: number[];
  target_sr: number;
  input_envelope: number[];
  input_sr: number;
}

function VoiceAnalyzerModal({
  appState,
  guiState,
  showVoiceAnalyzer,
  setShowVoiceAnalyzer
}: VoiceAnalyzerModalProps): JSX.Element {
  // ---------------- States ----------------
  const [targetFile, setTargetFile] = useState<File | null>(null);
  const [inputFile, setInputFile] = useState<File | null>(null);
  const [targetPreviewUrl, setTargetPreviewUrl] = useState<string>('');
  const [inputPreviewUrl, setInputPreviewUrl] = useState<string>('');

  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);

  // Clean up object URLs when unmounting or changing files
  useEffect(() => {
    return () => {
      if (targetPreviewUrl) URL.revokeObjectURL(targetPreviewUrl);
      if (inputPreviewUrl) URL.revokeObjectURL(inputPreviewUrl);
    };
  }, [targetPreviewUrl, inputPreviewUrl]);

  // ---------------- Helpers ----------------
  const hzToNote = (hz: number): string => {
    if (!hz || hz <= 0) return 'N/A';
    const notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
    const midi = Math.round(12 * Math.log2(hz / 440) + 69);
    const noteIndex = ((midi % 12) + 12) % 12;
    const octave = Math.floor(midi / 12) - 1;
    return `${notes[noteIndex]}${octave} (${Math.round(hz)} Hz)`;
  };

  // ---------------- Handlers ----------------
  const handleTargetChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setTargetFile(file);
      if (targetPreviewUrl) URL.revokeObjectURL(targetPreviewUrl);
      setTargetPreviewUrl(URL.createObjectURL(file));
      setResult(null);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setInputFile(file);
      if (inputPreviewUrl) URL.revokeObjectURL(inputPreviewUrl);
      setInputPreviewUrl(URL.createObjectURL(file));
      setResult(null);
    }
  };

  const handleAnalyze = async () => {
    if (!targetFile || !inputFile) {
      guiState.showError(t('voiceAnalyzerUploadBoth'), t('voiceAnalyzerUploadWarning'));
      return;
    }

    setIsAnalyzing(true);
    setResult(null);

    const formData = new FormData();
    formData.append('target_file', targetFile);
    formData.append('input_file', inputFile);

    try {
      const response = await fetch('/analyze_voice', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: AnalysisResult = await response.json();
      if (data.success) {
        setResult(data);
        
        try {
          const activeSlotIndex = appState.serverSetting.serverSetting.modelSlotIndex;
          const activeModelName = appState.serverSetting.serverSetting.modelSlots[activeSlotIndex]?.name || `Model ${activeSlotIndex}`;
          
          const targetProfileId = `target-slot-${activeSlotIndex}`;
          const inputProfileId = `input-default`;
          
          const targetProfile = {
            id: targetProfileId,
            name: `Target: ${activeModelName}`,
            type: 'target',
            envelope: data.target_envelope,
            sr: data.target_sr,
            f0: data.target_f0,
            centroid: data.target_centroid
          };
          
          const inputProfile = {
            id: inputProfileId,
            name: `Input: My Voice (Auto-Generated)`,
            type: 'input',
            envelope: data.input_envelope,
            sr: data.input_sr,
            f0: data.input_f0,
            centroid: data.input_centroid
          };
          
          const storedProfilesStr = localStorage.getItem('formant_profiles') || '[]';
          let storedProfiles: any[] = [];
          try {
            storedProfiles = JSON.parse(storedProfilesStr);
          } catch (e) {
            storedProfiles = [];
          }
          
          storedProfiles = storedProfiles.filter(p => p.id !== targetProfileId && p.id !== inputProfileId);
          storedProfiles.push(targetProfile);
          storedProfiles.push(inputProfile);
          localStorage.setItem('formant_profiles', JSON.stringify(storedProfiles));
          
          const storedBindingsStr = localStorage.getItem('model_formant_bindings') || '{}';
          let storedBindings: any = {};
          try {
            storedBindings = JSON.parse(storedBindingsStr);
          } catch (e) {
            storedBindings = {};
          }
          storedBindings[activeSlotIndex] = {
            targetProfileId,
            inputProfileId,
            recommendedPitch: data.recommended_pitch,
            recommendedFormantShift: data.recommended_formant_shift
          };
          localStorage.setItem('model_formant_bindings', JSON.stringify(storedBindings));
          
          appState.serverSetting.updateServerSettings({
            ...appState.serverSetting.serverSetting,
            tran: data.recommended_pitch,
            formantShift: data.recommended_formant_shift,
            formantProfileActive: true,
            formantProfileTargetEnvelope: JSON.stringify(data.target_envelope),
            formantProfileTargetSr: data.target_sr,
            formantProfileInputEnvelope: JSON.stringify(data.input_envelope),
            formantProfileInputSr: data.input_sr,
            formantProfileStrength: 0.35
          });
        } catch (autoErr) {
          console.error('Error auto-generating/applying formant profile:', autoErr);
        }
      } else {
        throw new Error(t('voiceAnalysisUnsuccessful'));
      }
    } catch (error) {
      console.error('Voice analysis error:', error);
      guiState.showError(
        `${t('voiceAnalyzerFailedAnalysis')}${error instanceof Error ? error.message : String(error)}`,
        t('errorTitle')
      );
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleApplyPitch = () => {
    if (!result) return;
    appState.serverSetting.updateServerSettings({
      ...appState.serverSetting.serverSetting,
      tran: result.recommended_pitch
    });
    guiState.showError(
      `${t('voiceAnalyzerAppliedPitchSuccess')}${result.recommended_pitch >= 0 ? '+' : ''}${
        result.recommended_pitch
      }`,
      t('confirmTitle')
    );
  };

  const handleApplyFormant = () => {
    if (!result) return;
    appState.serverSetting.updateServerSettings({
      ...appState.serverSetting.serverSetting,
      formantShift: result.recommended_formant_shift,
      formantProfileActive: true,
      formantProfileTargetEnvelope: JSON.stringify(result.target_envelope),
      formantProfileTargetSr: result.target_sr,
      formantProfileInputEnvelope: JSON.stringify(result.input_envelope),
      formantProfileInputSr: result.input_sr,
      formantProfileStrength: 0.35
    });
    guiState.showError(
      `${t('voiceAnalyzerAppliedFormantSuccess')}${result.recommended_formant_shift >= 0 ? '+' : ''}${
        result.recommended_formant_shift
      }`,
      t('confirmTitle')
    );
  };

  const handleApplyAll = () => {
    if (!result) return;
    appState.serverSetting.updateServerSettings({
      ...appState.serverSetting.serverSetting,
      tran: result.recommended_pitch,
      formantShift: result.recommended_formant_shift,
      formantProfileActive: true,
      formantProfileTargetEnvelope: JSON.stringify(result.target_envelope),
      formantProfileTargetSr: result.target_sr,
      formantProfileInputEnvelope: JSON.stringify(result.input_envelope),
      formantProfileInputSr: result.input_sr,
      formantProfileStrength: 0.35
    });
    guiState.showError(
      `${t('voiceAnalyzerAppliedBothSuccess')}${result.recommended_pitch >= 0 ? '+' : ''}${
        result.recommended_pitch
      }${t('voiceAnalyzerFormantLabel')}${result.recommended_formant_shift >= 0 ? '+' : ''}${result.recommended_formant_shift}`,
      t('confirmTitle')
    );
  };

  const handleClose = () => {
    setShowVoiceAnalyzer(false);
    setTargetFile(null);
    setInputFile(null);
    setResult(null);
    if (targetPreviewUrl) URL.revokeObjectURL(targetPreviewUrl);
    if (inputPreviewUrl) URL.revokeObjectURL(inputPreviewUrl);
    setTargetPreviewUrl('');
    setInputPreviewUrl('');
  };

  return (
    <GenericModal isOpen={showVoiceAnalyzer} onClose={handleClose} title={t('voiceAnalyzerTitle')} size="medium">
      <div className="space-y-6 text-on-surface p-2 max-h-[70vh] overflow-y-auto pr-1.5">
        <p className="text-xs text-on-surface-variant font-medium leading-relaxed">
          {t('voiceAnalyzerDesc')}
        </p>

        {/* Upload Panels */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Target / Reference Panel */}
          <div className="p-5 rounded-2xl border border-outline-variant bg-surface-container-low flex flex-col justify-between transition-all">
            <div>
              <span className="flex items-center space-x-2 mb-3 font-semibold text-primary">
                <FontAwesomeIcon icon={faMusic} />
                <label className={CSS_CLASSES.label}>{t('voiceAnalyzerStandardTarget')}</label>
              </span>
              <label className="flex flex-col items-center justify-center border border-dashed border-outline-variant hover:border-primary/50 bg-surface-container/40 hover:bg-primary/5 p-4 rounded-xl cursor-pointer transition-all duration-200 text-center">
                <FontAwesomeIcon icon={faMusic} className="text-primary/70 text-lg mb-1.5" />
                <span className="text-xs text-on-surface-variant font-medium select-none truncate max-w-full px-2">
                  {targetFile ? targetFile.name : t('clickToChooseFile')}
                </span>
                <input type="file" accept="audio/*" onChange={handleTargetChange} className="hidden" />
              </label>
            </div>
            {targetPreviewUrl && (
              <div className="mt-4">
                <p className="text-[10px] text-on-surface-variant font-bold mb-1">{t('voiceAnalyzerPreviewAudio')}</p>
                <audio src={targetPreviewUrl} controls className="w-full h-8" />
              </div>
            )}
          </div>

          {/* User Input Panel */}
          <div className="p-5 rounded-2xl border border-outline-variant bg-surface-container-low flex flex-col justify-between transition-all">
            <div>
              <span className="flex items-center space-x-2 mb-3 font-semibold text-secondary">
                <FontAwesomeIcon icon={faMicrophone} />
                <label className={CSS_CLASSES.label}>{t('voiceAnalyzerYourInput')}</label>
              </span>
              <label className="flex flex-col items-center justify-center border border-dashed border-outline-variant hover:border-secondary/50 bg-surface-container/40 hover:bg-secondary/5 p-4 rounded-xl cursor-pointer transition-all duration-200 text-center">
                <FontAwesomeIcon icon={faMicrophone} className="text-secondary/70 text-lg mb-1.5" />
                <span className="text-xs text-on-surface-variant font-medium select-none truncate max-w-full px-2">
                  {inputFile ? inputFile.name : t('clickToChooseFile')}
                </span>
                <input type="file" accept="audio/*" onChange={handleInputChange} className="hidden" />
              </label>
            </div>
            {inputPreviewUrl && (
              <div className="mt-4">
                <p className="text-[10px] text-on-surface-variant font-bold mb-1">{t('voiceAnalyzerPreviewAudio')}</p>
                <audio src={inputPreviewUrl} controls className="w-full h-8" />
              </div>
            )}
          </div>
        </div>

        {/* Action Button */}
        <div className="flex justify-center pt-2">
          <button
            onClick={handleAnalyze}
            disabled={isAnalyzing || !targetFile || !inputFile}
            className={`flex items-center space-x-2 px-8 py-2.5 rounded-full text-sm font-semibold transition-all duration-150 ${
              isAnalyzing || !targetFile || !inputFile
                ? 'bg-surface-container-highest text-on-surface-variant/30 cursor-not-allowed'
                : 'bg-primary text-on-primary hover:shadow-elevation-1 hover:scale-103 active:scale-97'
            }`}
          >
            <FontAwesomeIcon icon={faSync} className={isAnalyzing ? 'animate-spin' : ''} />
            <span>{isAnalyzing ? t('voiceAnalyzerComparing') : t('voiceAnalyzerStartAnalysis')}</span>
          </button>
        </div>

        {/* Results Section */}
        {result && (
          <div className="p-6 rounded-2xl border border-outline-variant bg-surface-container-low space-y-6 animate-fadeIn">
            <h4 className="text-base font-bold text-center border-b border-outline-variant/30 pb-2.5 mb-4 text-on-surface uppercase tracking-wider">
              {t('voiceAnalyzerDiagnostics')}
            </h4>

            {/* Diagnostic Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-xs">
              <div className="p-4 rounded-xl bg-surface-container border border-outline-variant space-y-2">
                <p className="font-bold text-primary">{t('voiceAnalyzerPitchComparison')}</p>
                <div className="flex justify-between">
                  <span className="text-on-surface-variant/70">{t('voiceAnalyzerStandardTargetLabel')}</span>
                  <span className="font-semibold text-on-surface">{hzToNote(result.target_f0)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-on-surface-variant/70">{t('voiceAnalyzerYourInputLabel')}</span>
                  <span className="font-semibold text-on-surface">{hzToNote(result.input_f0)}</span>
                </div>
              </div>

              <div className="p-4 rounded-xl bg-surface-container border border-outline-variant space-y-2">
                <p className="font-bold text-secondary">{t('voiceAnalyzerFormantResonanceLabel')}</p>
                <div className="flex justify-between">
                  <span className="text-on-surface-variant/70">{t('voiceAnalyzerStandardTargetLabel')}</span>
                  <span className="font-semibold text-on-surface">{Math.round(result.target_centroid)} Hz</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-on-surface-variant/70">{t('voiceAnalyzerYourInputLabel')}</span>
                  <span className="font-semibold text-on-surface">{Math.round(result.input_centroid)} Hz</span>
                </div>
              </div>
            </div>

            {/* Recommendations */}
            <div className="bg-surface-container border border-outline-variant rounded-xl p-5 space-y-4">
              <p className="font-bold text-center text-sm text-on-surface-variant">
                {t('voiceAnalyzerRecommendedParams')}
              </p>

              <div className="flex flex-col md:flex-row md:items-center justify-between p-3 rounded-lg hover:bg-primary/4 transition-colors">
                <div className="mb-2 md:mb-0 flex items-baseline">
                  <span className="font-semibold text-xs text-on-surface">{t('voiceAnalyzerRecommendedPitch')}</span>
                  <span className="ml-2 text-xl font-extrabold text-primary">
                    {result.recommended_pitch >= 0 ? '+' : ''}
                    {result.recommended_pitch}
                  </span>
                  <span className="ml-1 text-[10px] text-on-surface-variant font-medium">{t('voiceAnalyzerSemitones')}</span>
                </div>
                <button
                  onClick={handleApplyPitch}
                  className="flex items-center space-x-1.5 px-4 py-1.5 bg-primary text-on-primary text-xs font-semibold rounded-full hover:shadow-elevation-1 active:scale-97 transition-all"
                >
                  <FontAwesomeIcon icon={faCheck} />
                  <span>{t('voiceAnalyzerApplyPitch')}</span>
                </button>
              </div>

              <div className="flex flex-col md:flex-row md:items-center justify-between p-3 rounded-lg hover:bg-primary/4 transition-colors">
                <div className="mb-2 md:mb-0 flex items-baseline">
                  <span className="font-semibold text-xs text-on-surface">{t('voiceAnalyzerRecommendedFormant')}</span>
                  <span className="ml-2 text-xl font-extrabold text-secondary">
                    {result.recommended_formant_shift >= 0 ? '+' : ''}
                    {result.recommended_formant_shift}
                  </span>
                  <span className="ml-1 text-[10px] text-on-surface-variant font-medium">{t('voiceAnalyzerSemitones')}</span>
                </div>
                <button
                  onClick={handleApplyFormant}
                  className="flex items-center space-x-1.5 px-4 py-1.5 bg-secondary text-on-secondary text-xs font-semibold rounded-full hover:shadow-elevation-1 active:scale-97 transition-all"
                >
                  <FontAwesomeIcon icon={faCheck} />
                  <span>{t('voiceAnalyzerApplyFormant')}</span>
                </button>
              </div>

              <div className="flex justify-center pt-2 border-t border-outline-variant/30">
                <button
                  onClick={handleApplyAll}
                  className="flex items-center space-x-2 px-6 py-2.5 bg-primary text-on-primary text-xs font-bold rounded-full hover:shadow-elevation-2 hover:scale-103 active:scale-97 transition-all"
                >
                  <FontAwesomeIcon icon={faCheck} />
                  <span>{t('voiceAnalyzerApplyAll')}</span>
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </GenericModal>
  );
}

export default VoiceAnalyzerModal;
