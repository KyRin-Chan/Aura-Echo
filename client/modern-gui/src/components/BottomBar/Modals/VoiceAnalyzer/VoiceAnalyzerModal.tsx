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
    if (!hz || hz <= 0) return "N/A";
    const notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
    const midi = Math.round(12 * Math.log2(hz / 440) + 69);
    const noteIndex = (midi % 12 + 12) % 12;
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
      guiState.showError("Please upload both Reference and Input samples.", "Warning");
      return;
    }

    setIsAnalyzing(true);
    setResult(null);

    const formData = new FormData();
    formData.append("target_file", targetFile);
    formData.append("input_file", inputFile);

    try {
      const response = await fetch("/analyze_voice", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: AnalysisResult = await response.json();
      if (data.success) {
        setResult(data);
      } else {
        throw new Error("Voice analysis returned unsuccessful status.");
      }
    } catch (error) {
      console.error("Voice analysis error:", error);
      guiState.showError(
        `Failed to analyze audio: ${error instanceof Error ? error.message : String(error)}`,
        t("errorTitle")
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
    guiState.showError(`${t("voiceAnalyzerAppliedPitchSuccess")}${result.recommended_pitch >= 0 ? '+' : ''}${result.recommended_pitch}`, t("confirmTitle"));
  };

  const handleApplyFormant = () => {
    if (!result) return;
    appState.serverSetting.updateServerSettings({
      ...appState.serverSetting.serverSetting,
      formantShift: result.recommended_formant_shift
    });
    guiState.showError(`Applied recommended Formant Shift: ${result.recommended_formant_shift >= 0 ? '+' : ''}${result.recommended_formant_shift}`, t("confirmTitle"));
  };

  const handleApplyAll = () => {
    if (!result) return;
    appState.serverSetting.updateServerSettings({
      ...appState.serverSetting.serverSetting,
      tran: result.recommended_pitch,
      formantShift: result.recommended_formant_shift
    });
    guiState.showError(
      `${t("voiceAnalyzerAppliedBothSuccess")}${result.recommended_pitch >= 0 ? '+' : ''}${result.recommended_pitch}, Formant: ${result.recommended_formant_shift >= 0 ? '+' : ''}${result.recommended_formant_shift}`,
      t("confirmTitle")
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
    <GenericModal
      isOpen={showVoiceAnalyzer}
      onClose={handleClose}
      title={t("voiceAnalyzerTitle")}
      size="medium"
    >
      <div className="space-y-6 text-slate-800 dark:text-gray-100 p-2">
        <p className="text-sm text-slate-600 dark:text-gray-400">
          {t("voiceAnalyzerDesc")}
        </p>

        {/* Upload Panels */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Target / Reference Panel */}
          <div
            className="p-5 rounded-2xl border flex flex-col justify-between transition-all"
            style={{
              backgroundColor: 'var(--bg-tertiary)',
              borderColor: 'var(--border-primary)'
            }}
          >
            <div>
              <span className="flex items-center space-x-2 mb-2 font-medium text-[var(--macaron-mint)]">
                <FontAwesomeIcon icon={faMusic} />
                <label className={CSS_CLASSES.label}>{t("voiceAnalyzerStandardTarget")}</label>
              </span>
              <input
                type="file"
                accept="audio/*"
                onChange={handleTargetChange}
                className={CSS_CLASSES.fileInput}
              />
            </div>
            {targetPreviewUrl && (
              <div className="mt-4">
                <p className="text-xs text-slate-500 mb-1">Preview Audio:</p>
                <audio src={targetPreviewUrl} controls className="w-full h-8" />
              </div>
            )}
          </div>

          {/* User Input Panel */}
          <div
            className="p-5 rounded-2xl border flex flex-col justify-between transition-all"
            style={{
              backgroundColor: 'var(--bg-tertiary)',
              borderColor: 'var(--border-primary)'
            }}
          >
            <div>
              <span className="flex items-center space-x-2 mb-2 font-medium text-[var(--macaron-blue)]">
                <FontAwesomeIcon icon={faMicrophone} />
                <label className={CSS_CLASSES.label}>{t("voiceAnalyzerYourInput")}</label>
              </span>
              <input
                type="file"
                accept="audio/*"
                onChange={handleInputChange}
                className={CSS_CLASSES.fileInput}
              />
            </div>
            {inputPreviewUrl && (
              <div className="mt-4">
                <p className="text-xs text-slate-500 mb-1">Preview Audio:</p>
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
            className={`flex items-center space-x-2 px-8 py-3 rounded-full text-base font-semibold shadow-md transition-all
              ${(isAnalyzing || !targetFile || !inputFile)
                ? 'opacity-50 cursor-not-allowed bg-slate-300 dark:bg-slate-700 text-slate-500'
                : 'bg-[var(--macaron-mint)] hover:bg-[var(--macaron-mint-hover)] text-[#3a3530] hover:scale-105 active:scale-95'
              }`}
          >
            <FontAwesomeIcon icon={faSync} className={isAnalyzing ? 'animate-spin' : ''} />
            <span>{isAnalyzing ? t("voiceAnalyzerComparing") : t("voiceAnalyzerStartAnalysis")}</span>
          </button>
        </div>

        {/* Results Section */}
        {result && (
          <div
            className="p-6 rounded-2xl border space-y-6 animate-fadeIn"
            style={{
              backgroundColor: 'var(--bg-tertiary)',
              borderColor: 'var(--border-primary)'
            }}
          >
            <h4 className="text-lg font-bold text-center border-b border-[var(--border-primary)] pb-2 mb-4">
              {t("voiceAnalyzerDiagnostics")}
            </h4>

            {/* Diagnostic Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm">
              <div className="p-4 rounded-xl bg-[var(--bg-secondary)] border border-[var(--border-primary)] space-y-2">
                <p className="font-bold text-[var(--macaron-mint)]">{t("voiceAnalyzerPitchComparison")}</p>
                <div className="flex justify-between">
                  <span className="text-slate-500">Standard Target:</span>
                  <span className="font-semibold">{hzToNote(result.target_f0)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Your Input:</span>
                  <span className="font-semibold">{hzToNote(result.input_f0)}</span>
                </div>
              </div>

              <div className="p-4 rounded-xl bg-[var(--bg-secondary)] border border-[var(--border-primary)] space-y-2">
                <p className="font-bold text-[var(--macaron-blue)]">Formant Resonance (Spectral Centroid)</p>
                <div className="flex justify-between">
                  <span className="text-slate-500">Standard Target:</span>
                  <span className="font-semibold">{Math.round(result.target_centroid)} Hz</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Your Input:</span>
                  <span className="font-semibold">{Math.round(result.input_centroid)} Hz</span>
                </div>
              </div>
            </div>

            {/* Recommendations */}
            <div className="bg-[var(--bg-secondary)] border border-[var(--border-primary)] rounded-xl p-5 space-y-4">
              <p className="font-bold text-center text-slate-700 dark:text-slate-300">{t("voiceAnalyzerRecommendedParams")}</p>
              
              <div className="flex flex-col md:flex-row md:items-center justify-between p-3 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors">
                <div className="mb-2 md:mb-0">
                  <span className="font-semibold">{t("voiceAnalyzerRecommendedPitch")}</span>
                  <span className="ml-2 text-lg font-black text-[var(--macaron-mint)]">
                    {result.recommended_pitch >= 0 ? '+' : ''}{result.recommended_pitch}
                  </span>
                  <span className="ml-1 text-xs text-slate-400">semitones</span>
                </div>
                <button
                  onClick={handleApplyPitch}
                  className="flex items-center space-x-1 px-4 py-1.5 bg-[var(--macaron-mint)] text-[#3a3530] text-xs font-semibold rounded-full hover:opacity-90 active:scale-95 transition-all"
                >
                  <FontAwesomeIcon icon={faCheck} />
                  <span>{t("voiceAnalyzerApplyPitch")}</span>
                </button>
              </div>

              <div className="flex flex-col md:flex-row md:items-center justify-between p-3 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors">
                <div className="mb-2 md:mb-0">
                  <span className="font-semibold">{t("voiceAnalyzerRecommendedFormant")}</span>
                  <span className="ml-2 text-lg font-black text-[var(--macaron-blue)]">
                    {result.recommended_formant_shift >= 0 ? '+' : ''}{result.recommended_formant_shift}
                  </span>
                  <span className="ml-1 text-xs text-slate-400">semitones</span>
                </div>
                <button
                  onClick={handleApplyFormant}
                  className="flex items-center space-x-1 px-4 py-1.5 bg-[var(--macaron-blue)] text-[#3a3530] text-xs font-semibold rounded-full hover:opacity-90 active:scale-95 transition-all"
                >
                  <FontAwesomeIcon icon={faCheck} />
                  <span>{t("voiceAnalyzerApplyFormant")}</span>
                </button>
              </div>

              <div className="flex justify-center pt-2">
                <button
                  onClick={handleApplyAll}
                  className="flex items-center space-x-2 px-6 py-2.5 bg-gradient-to-r from-[var(--macaron-mint)] to-[var(--macaron-blue)] text-[#3a3530] text-sm font-bold rounded-full hover:opacity-90 hover:scale-103 active:scale-97 transition-all shadow-sm"
                >
                  <FontAwesomeIcon icon={faCheck} />
                  <span>{t("voiceAnalyzerApplyAll")}</span>
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
