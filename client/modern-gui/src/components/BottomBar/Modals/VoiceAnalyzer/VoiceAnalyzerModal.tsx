import { useState, JSX, useEffect, useRef } from 'react';
import { ClientState } from '@dannadori/voice-changer-client-js';
import { CSS_CLASSES } from '../../../../styles/constants';
import GenericModal from '../../../Modals/GenericModal';
import { UIContextType } from '../../../../context/UIContext';
import { t } from '../../../../locales';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faSync, faMicrophone, faCheck, faMusic, faStop, faTrash, faEdit, faSave, faTimes } from '@fortawesome/free-solid-svg-icons';

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

  // New States for Mic Recording
  const [inputTab, setInputTab] = useState<'upload' | 'record'>('upload');
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [recordingDuration, setRecordingDuration] = useState<number>(0);
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);

  const recordingTimerRef = useRef<NodeJS.Timeout | null>(null);

  // New States for Profile Manager UI
  const [activeModalTab, setActiveModalTab] = useState<'analyzer' | 'manager'>('analyzer');
  const [profiles, setProfiles] = useState<any[]>([]);
  const [bindings, setBindings] = useState<Record<number, any>>({});
  const [editingProfileId, setEditingProfileId] = useState<string | null>(null);
  const [editingName, setEditingName] = useState<string>('');

  // Load profiles and bindings from localStorage
  const loadProfilesAndBindings = () => {
    try {
      const storedProfilesStr = localStorage.getItem('formant_profiles') || '[]';
      const storedProfiles = JSON.parse(storedProfilesStr);
      setProfiles(storedProfiles);

      const storedBindingsStr = localStorage.getItem('model_formant_bindings') || '{}';
      const storedBindings = JSON.parse(storedBindingsStr);
      setBindings(storedBindings);
    } catch (e) {
      console.error('Error loading profiles or bindings:', e);
    }
  };

  useEffect(() => {
    if (showVoiceAnalyzer) {
      loadProfilesAndBindings();
    }
  }, [showVoiceAnalyzer, activeModalTab]);

  // Clean up object URLs and recording timers when unmounting or changing files
  useEffect(() => {
    return () => {
      if (targetPreviewUrl) URL.revokeObjectURL(targetPreviewUrl);
      if (inputPreviewUrl) URL.revokeObjectURL(inputPreviewUrl);
      if (recordingTimerRef.current) clearInterval(recordingTimerRef.current);
    };
  }, [targetPreviewUrl, inputPreviewUrl]);

  // ---------------- Helpers ----------------
  const hzToNote = (hz: number): string => {
    if (!hz || hz <= 0) return 'N/A';
    const notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
    const midi = Math.round(12 * Math.log2(hz / 440) + 69);
    const noteIndex = ((midi % 12) + 12) % 12;
    const octave = Math.floor(midi / 12) - 1;
    return `${notes[noteIndex]}${notes[noteIndex] ? octave : ''} (${Math.round(hz)} Hz)`;
  };

  // Profile manager handlers
  const handleRenameProfileStart = (profileId: string, currentName: string) => {
    setEditingProfileId(profileId);
    setEditingName(currentName);
  };

  const handleRenameProfileSave = () => {
    if (!editingProfileId || !editingName.trim()) return;
    try {
      const updatedProfiles = profiles.map((p) => {
        if (p.id === editingProfileId) {
          return { ...p, name: editingName.trim() };
        }
        return p;
      });
      localStorage.setItem('formant_profiles', JSON.stringify(updatedProfiles));
      setProfiles(updatedProfiles);
      setEditingProfileId(null);
      setEditingName('');
    } catch (e) {
      console.error('Error saving profile name:', e);
    }
  };

  const handleRenameProfileCancel = () => {
    setEditingProfileId(null);
    setEditingName('');
  };

  const handleDeleteProfile = (profileId: string) => {
    try {
      const updatedProfiles = profiles.filter((p) => p.id !== profileId);
      localStorage.setItem('formant_profiles', JSON.stringify(updatedProfiles));
      setProfiles(updatedProfiles);

      let bindingsChanged = false;
      const updatedBindings = { ...bindings };
      Object.keys(updatedBindings).forEach((key) => {
        const slotIndex = Number(key);
        const bind = updatedBindings[slotIndex];
        if (bind && (bind.targetProfileId === profileId || bind.inputProfileId === profileId)) {
          delete updatedBindings[slotIndex];
          bindingsChanged = true;

          if (slotIndex === appState.serverSetting.serverSetting.modelSlotIndex) {
            appState.serverSetting.updateServerSettings({
              ...appState.serverSetting.serverSetting,
              formantProfileActive: false
            });
          }
        }
      });

      if (bindingsChanged) {
        localStorage.setItem('model_formant_bindings', JSON.stringify(updatedBindings));
        setBindings(updatedBindings);
      }
    } catch (e) {
      console.error('Error deleting profile:', e);
    }
  };

  const handleBindProfiles = (inputProfileId: string | null, targetProfileId: string | null) => {
    const activeSlot = appState.serverSetting.serverSetting.modelSlotIndex;
    if (activeSlot === undefined || activeSlot === -1) return;

    try {
      const updatedBindings = { ...bindings };
      
      if (!inputProfileId && !targetProfileId) {
        delete updatedBindings[activeSlot];
        localStorage.setItem('model_formant_bindings', JSON.stringify(updatedBindings));
        setBindings(updatedBindings);

        appState.serverSetting.updateServerSettings({
          ...appState.serverSetting.serverSetting,
          formantProfileActive: false
        });
        return;
      }

      const inputProfile = profiles.find((p) => p.id === inputProfileId);
      const targetProfile = profiles.find((p) => p.id === targetProfileId);

      if (!inputProfile || !targetProfile) {
        console.warn('Cannot bind: one or both profiles not found');
        return;
      }

      const currentBinding = updatedBindings[activeSlot] || {};
      updatedBindings[activeSlot] = {
        ...currentBinding,
        inputProfileId,
        targetProfileId
      };
      localStorage.setItem('model_formant_bindings', JSON.stringify(updatedBindings));
      setBindings(updatedBindings);

      appState.serverSetting.updateServerSettings({
        ...appState.serverSetting.serverSetting,
        formantProfileActive: true,
        formantProfileTargetEnvelope: JSON.stringify(targetProfile.envelope),
        formantProfileTargetSr: targetProfile.sr,
        formantProfileInputEnvelope: JSON.stringify(inputProfile.envelope),
        formantProfileInputSr: inputProfile.sr
      });
    } catch (e) {
      console.error('Error binding profiles:', e);
    }
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

  // Recording functions
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      const chunks: Blob[] = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunks.push(e.data);
        }
      };

      recorder.onstop = () => {
        const blob = new Blob(chunks, { type: 'audio/wav' });
        const file = new File([blob], 'mic_input.wav', { type: 'audio/wav' });
        setInputFile(file);
        
        if (inputPreviewUrl) URL.revokeObjectURL(inputPreviewUrl);
        const previewUrl = URL.createObjectURL(file);
        setInputPreviewUrl(previewUrl);
        setResult(null);

        // Auto-analyze if target reference file is present
        if (targetFile) {
          triggerAnalysis(targetFile, file);
        }
      };

      setMediaRecorder(recorder);
      recorder.start();
      setIsRecording(true);
      setRecordingDuration(0);

      if (recordingTimerRef.current) clearInterval(recordingTimerRef.current);
      recordingTimerRef.current = setInterval(() => {
        setRecordingDuration((prev) => prev + 1);
      }, 1000);
    } catch (err) {
      console.error('Failed to start recording:', err);
      guiState.showError('Could not access microphone. Please check permissions.', t('errorTitle'));
    }
  };

  const stopRecording = () => {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      mediaRecorder.stop();
      mediaRecorder.stream.getTracks().forEach((track) => track.stop());
    }
    setIsRecording(false);
    if (recordingTimerRef.current) {
      clearInterval(recordingTimerRef.current);
      recordingTimerRef.current = null;
    }
  };

  const triggerAnalysis = async (tgt: File, inp: File) => {
    setIsAnalyzing(true);
    setResult(null);

    const formData = new FormData();
    formData.append('target_file', tgt);
    formData.append('input_file', inp);

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

  const handleAnalyze = () => {
    if (!targetFile || !inputFile) {
      guiState.showError(t('voiceAnalyzerUploadBoth'), t('voiceAnalyzerUploadWarning'));
      return;
    }
    triggerAnalysis(targetFile, inputFile);
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

        {/* Tab Selection Header */}
        <div className="flex border-b border-outline-variant/30 mb-4 text-xs font-semibold">
          <button
            onClick={() => setActiveModalTab('analyzer')}
            className={`pb-2 px-4 border-b-2 transition-all ${
              activeModalTab === 'analyzer'
                ? 'border-primary text-primary'
                : 'border-transparent text-on-surface-variant hover:text-on-surface'
            }`}
          >
            {t('voiceAnalyzerTabDiagnostics') || '变声诊断与分析'}
          </button>
          <button
            onClick={() => setActiveModalTab('manager')}
            className={`pb-2 px-4 border-b-2 transition-all ${
              activeModalTab === 'manager'
                ? 'border-primary text-primary'
                : 'border-transparent text-on-surface-variant hover:text-on-surface'
            }`}
          >
            {t('voiceAnalyzerTabManager') || '音色档案管理'}
          </button>
        </div>

        {activeModalTab === 'analyzer' ? (
          <>
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
                  <div className="flex items-center justify-between mb-3">
                    <span className="flex items-center space-x-2 font-semibold text-secondary">
                      <FontAwesomeIcon icon={faMicrophone} />
                      <label className={CSS_CLASSES.label}>{t('voiceAnalyzerYourInput')}</label>
                    </span>
                    {/* Tab Switcher */}
                    <div className="flex bg-surface-container rounded-full p-0.5 border border-outline-variant/30 text-[10px]">
                      <button
                        onClick={() => { setInputTab('upload'); setResult(null); }}
                        disabled={isRecording}
                        className={`px-3 py-1 rounded-full font-medium transition-all ${
                          inputTab === 'upload'
                            ? 'bg-secondary text-on-secondary shadow-elevation-1'
                            : 'text-on-surface-variant hover:text-on-surface disabled:opacity-50'
                        }`}
                      >
                        {t('voiceAnalyzerTabUpload') || 'Upload'}
                      </button>
                      <button
                        onClick={() => { setInputTab('record'); setResult(null); }}
                        className={`px-3 py-1 rounded-full font-medium transition-all ${
                          inputTab === 'record'
                            ? 'bg-secondary text-on-secondary shadow-elevation-1'
                            : 'text-on-surface-variant hover:text-on-surface'
                        }`}
                      >
                        {t('voiceAnalyzerTabRecord') || 'Live Record'}
                      </button>
                    </div>
                  </div>

                  {inputTab === 'upload' ? (
                    <label className="flex flex-col items-center justify-center border border-dashed border-outline-variant hover:border-secondary/50 bg-surface-container/40 hover:bg-secondary/5 p-4 rounded-xl cursor-pointer transition-all duration-200 text-center">
                      <FontAwesomeIcon icon={faMicrophone} className="text-secondary/70 text-lg mb-1.5" />
                      <span className="text-xs text-on-surface-variant font-medium select-none truncate max-w-full px-2">
                        {inputFile ? inputFile.name : t('clickToChooseFile')}
                      </span>
                      <input type="file" accept="audio/*" onChange={handleInputChange} className="hidden" />
                    </label>
                  ) : (
                    /* Live Recording Mode */
                    <div className="flex flex-col space-y-3">
                      {/* Reading prompt guidance */}
                      <div className="p-3 bg-surface-container rounded-xl border border-outline-variant/40 text-[11px] leading-relaxed">
                        <p className="text-secondary font-bold mb-1 text-[9px] uppercase tracking-wider">
                          {t('voiceAnalyzerScriptGuidance') || 'Read-along Script Guidance:'}
                        </p>
                        <p className="text-on-surface-variant italic font-medium px-1 bg-surface-container-high rounded p-1.5 leading-normal">
                          {t('voiceAnalyzerScriptText') || '「阿尔卑斯山的雪景非常壮美，白色的雾气在山谷中漂浮，红色的枫叶随风摇曳。」'}
                        </p>
                      </div>

                      {/* Controls */}
                      <div className="flex items-center justify-center space-x-4 py-1">
                        <button
                          onClick={isRecording ? stopRecording : startRecording}
                          disabled={isAnalyzing}
                          className={`relative flex items-center justify-center w-12 h-12 rounded-full transition-all duration-300 ${
                            isRecording
                              ? 'bg-error text-on-error hover:scale-105 active:scale-95 animate-pulse'
                              : 'bg-secondary text-on-secondary hover:scale-105 active:scale-95 hover:shadow-elevation-1'
                          } disabled:opacity-50 disabled:cursor-not-allowed`}
                        >
                          <FontAwesomeIcon icon={isRecording ? faStop : faMicrophone} className="text-lg" />
                        </button>
                        <div className="flex flex-col text-left">
                          <span className="text-[10px] text-on-surface-variant font-bold uppercase tracking-wider">
                            {isRecording ? (t('voiceAnalyzerRecordingStatus') || 'Recording...') : (t('voiceAnalyzerRecordStartStatus') || 'Click to Record')}
                          </span>
                          <span className="text-xs font-mono font-bold text-on-surface mt-0.5">
                            {recordingDuration}s / 10s
                          </span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
                {inputPreviewUrl && !isRecording && (
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
                      <span className="font-semibold text-xs text-on-surface">{t('voiceAnalyzerRecommendedFormant') || 'Recommended Formant Correction'}</span>
                      <span className="ml-2 text-xl font-extrabold text-secondary font-mono">✓ Active</span>
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
          </>
        ) : (
          /* Profile Manager Tab */
          <div className="space-y-6 animate-fadeIn">
            {/* Slot Binding Settings */}
            <div className="p-5 rounded-2xl border border-outline-variant bg-surface-container-low space-y-4">
              <h5 className="text-xs font-bold text-secondary uppercase tracking-wider mb-2">
                {t('voiceAnalyzerModelBindings') || '当前模型插槽绑定'}
              </h5>
              
              {(() => {
                const activeSlotIndex = appState.serverSetting.serverSetting.modelSlotIndex;
                const activeModelName = appState.serverSetting.serverSetting.modelSlots[activeSlotIndex]?.name || `Model ${activeSlotIndex}`;
                const currentBinding = bindings[activeSlotIndex];
                
                const inputProfiles = profiles.filter(p => p.type === 'input');
                const targetProfiles = profiles.filter(p => p.type === 'target');
                
                return (
                  <div className="space-y-4 text-xs">
                    <p className="font-semibold text-on-surface">
                      {t('voiceAnalyzerActiveSlot') || '当前活动插槽:'} <span className="text-primary font-bold">{activeModelName} (Slot #{activeSlotIndex})</span>
                    </p>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {/* Input Profile Selection */}
                      <div className="flex flex-col space-y-1.5">
                        <label className="text-[10px] text-on-surface-variant font-bold uppercase tracking-wider">
                          {t('voiceAnalyzerInputProfileDropdown') || '输入音色档案 (Your Voice)'}
                        </label>
                        <select
                          value={currentBinding?.inputProfileId || ''}
                          onChange={(e) => handleBindProfiles(e.target.value || null, currentBinding?.targetProfileId || null)}
                          className="px-3 py-2 rounded-lg bg-surface-container border border-outline-variant text-on-surface font-medium text-xs focus:ring-1 focus:ring-primary focus:border-primary outline-none"
                        >
                          <option value="">{t('voiceAnalyzerNoneOption') || '-- 未绑定 (无) --'}</option>
                          {inputProfiles.map((p) => (
                            <option key={p.id} value={p.id}>{p.name}</option>
                          ))}
                        </select>
                      </div>

                      {/* Target Profile Selection */}
                      <div className="flex flex-col space-y-1.5">
                        <label className="text-[10px] text-on-surface-variant font-bold uppercase tracking-wider">
                          {t('voiceAnalyzerTargetProfileDropdown') || '目标音色档案 (Model Voice)'}
                        </label>
                        <select
                          value={currentBinding?.targetProfileId || ''}
                          onChange={(e) => handleBindProfiles(currentBinding?.inputProfileId || null, e.target.value || null)}
                          className="px-3 py-2 rounded-lg bg-surface-container border border-outline-variant text-on-surface font-medium text-xs focus:ring-1 focus:ring-primary focus:border-primary outline-none"
                        >
                          <option value="">{t('voiceAnalyzerNoneOption') || '-- 未绑定 (无) --'}</option>
                          {targetProfiles.map((p) => (
                            <option key={p.id} value={p.id}>{p.name}</option>
                          ))}
                        </select>
                      </div>
                    </div>
                  </div>
                );
              })()}
            </div>

            {/* Stored Profiles Table */}
            <div className="p-5 rounded-2xl border border-outline-variant bg-surface-container-low space-y-4">
              <h5 className="text-xs font-bold text-primary uppercase tracking-wider mb-2">
                {t('voiceAnalyzerStoredProfiles') || '已保存的音色包络档案'}
              </h5>

              {profiles.length === 0 ? (
                <div className="text-center py-6 text-on-surface-variant/60 italic text-xs">
                  {t('voiceAnalyzerNoProfiles') || '暂无已保存的音色档案，请先进行变声诊断分析。'}
                </div>
              ) : (
                <div className="overflow-x-auto border border-outline-variant/30 rounded-xl bg-surface-container/30">
                  <table className="w-full text-left text-xs border-collapse">
                    <thead>
                      <tr className="bg-surface-container border-b border-outline-variant/30 font-bold text-on-surface-variant/80">
                        <th className="p-3 text-left">{t('voiceAnalyzerColName') || '档案名称'}</th>
                        <th className="p-3 text-left w-20">{t('voiceAnalyzerColType') || '类型'}</th>
                        <th className="p-3 text-left w-20">{t('voiceAnalyzerColSR') || '采样率'}</th>
                        <th className="p-3 text-left w-24">{t('voiceAnalyzerColCentroid') || '声学重心'}</th>
                        <th className="p-3 text-center w-28">{t('voiceAnalyzerColActions') || '操作'}</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-outline-variant/20">
                      {profiles.map((p) => (
                        <tr key={p.id} className="hover:bg-surface-container-high/40 transition-colors">
                          <td className="p-3 font-semibold text-on-surface">
                            {editingProfileId === p.id ? (
                              <div className="flex items-center space-x-2">
                                <input
                                  type="text"
                                  value={editingName}
                                  onChange={(e) => setEditingName(e.target.value)}
                                  className="px-2 py-1 border border-outline-variant rounded bg-surface-container-high text-on-surface focus:outline-none focus:ring-1 focus:ring-primary w-full"
                                />
                              </div>
                            ) : (
                              p.name
                            )}
                          </td>
                          <td className="p-3">
                            <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                              p.type === 'input' 
                                ? 'bg-secondary/10 text-secondary border border-secondary/20' 
                                : 'bg-primary/10 text-primary border border-primary/20'
                            }`}>
                              {p.type === 'input' ? (t('voiceAnalyzerTypeInput') || '输入') : (t('voiceAnalyzerTypeTarget') || '目标')}
                            </span>
                          </td>
                          <td className="p-3 font-mono text-on-surface-variant">{p.sr}Hz</td>
                          <td className="p-3 font-mono text-on-surface-variant">{p.centroid ? `${Math.round(p.centroid)}Hz` : 'N/A'}</td>
                          <td className="p-3 text-center">
                            {editingProfileId === p.id ? (
                              <div className="flex items-center justify-center space-x-2">
                                <button
                                  onClick={handleRenameProfileSave}
                                  className="p-1 px-2 text-primary hover:bg-primary/10 rounded transition-colors"
                                  title={t('saveLabel') || '保存'}
                                >
                                  <FontAwesomeIcon icon={faSave} />
                                </button>
                                <button
                                  onClick={handleRenameProfileCancel}
                                  className="p-1 px-2 text-on-surface-variant hover:bg-surface-container-highest rounded transition-colors"
                                  title={t('cancelLabel') || '取消'}
                                >
                                  <FontAwesomeIcon icon={faTimes} />
                                </button>
                              </div>
                            ) : (
                              <div className="flex items-center justify-center space-x-2">
                                <button
                                  onClick={() => handleRenameProfileStart(p.id, p.name)}
                                  className="p-1 px-2 text-on-surface-variant hover:text-primary hover:bg-primary/10 rounded transition-colors"
                                  title={t('renameLabel') || '重命名'}
                                >
                                  <FontAwesomeIcon icon={faEdit} />
                                </button>
                                <button
                                  onClick={() => handleDeleteProfile(p.id)}
                                  className="p-1 px-2 text-error hover:bg-error/10 rounded transition-colors"
                                  title={t('deleteLabel') || '删除'}
                                >
                                  <FontAwesomeIcon icon={faTrash} />
                                </button>
                              </div>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </GenericModal>
  );
}

export default VoiceAnalyzerModal;
