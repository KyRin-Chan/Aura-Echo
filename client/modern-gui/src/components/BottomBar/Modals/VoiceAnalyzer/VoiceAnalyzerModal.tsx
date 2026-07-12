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
  const [activeModalTab, setActiveModalTab] = useState<'pitch' | 'formant'>('pitch');

  // Pitch comparison states
  const [targetFile, setTargetFile] = useState<File | null>(null);
  const [inputFile, setInputFile] = useState<File | null>(null);
  const [targetPreviewUrl, setTargetPreviewUrl] = useState<string>('');
  const [inputPreviewUrl, setInputPreviewUrl] = useState<string>('');
  const [inputTab, setInputTab] = useState<'upload' | 'record'>('upload');
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [recordingDuration, setRecordingDuration] = useState<number>(0);
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
  const recordingTimerRef = useRef<NodeJS.Timeout | null>(null);

  const [isComparing, setIsComparing] = useState<boolean>(false);
  const [compareResult, setCompareResult] = useState<AnalysisResult | null>(null);

  // Formant manager states
  const [profiles, setProfiles] = useState<any[]>([]);
  const [bindings, setBindings] = useState<Record<number, any>>({});
  const [editingProfileId, setEditingProfileId] = useState<string | null>(null);
  const [editingName, setEditingName] = useState<string>('');

  // Formant generation states
  const [newProfileName, setNewProfileName] = useState<string>('');
  const [newProfileType, setNewProfileType] = useState<'input' | 'target'>('input');
  const [genAudioTab, setGenAudioTab] = useState<'upload' | 'record'>('upload');
  const [genFile, setGenFile] = useState<File | null>(null);
  const [genPreviewUrl, setGenPreviewUrl] = useState<string>('');
  
  const [isGenRecording, setIsGenRecording] = useState<boolean>(false);
  const [genRecordingDuration, setGenRecordingDuration] = useState<number>(0);
  const [genMediaRecorder, setGenMediaRecorder] = useState<MediaRecorder | null>(null);
  const genRecordingTimerRef = useRef<NodeJS.Timeout | null>(null);
  const [isGenerating, setIsGenerating] = useState<boolean>(false);

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

  // Clean up object URLs and recording timers
  useEffect(() => {
    return () => {
      if (targetPreviewUrl) URL.revokeObjectURL(targetPreviewUrl);
      if (inputPreviewUrl) URL.revokeObjectURL(inputPreviewUrl);
      if (genPreviewUrl) URL.revokeObjectURL(genPreviewUrl);
      if (recordingTimerRef.current) clearInterval(recordingTimerRef.current);
      if (genRecordingTimerRef.current) clearInterval(genRecordingTimerRef.current);
    };
  }, [targetPreviewUrl, inputPreviewUrl, genPreviewUrl]);

  // Tab change cleanup
  useEffect(() => {
    stopRecording();
    stopGenRecording();
    setTargetFile(null);
    setInputFile(null);
    setGenFile(null);
    setCompareResult(null);
    setNewProfileName('');
    setNewProfileType('input');
    if (targetPreviewUrl) URL.revokeObjectURL(targetPreviewUrl);
    if (inputPreviewUrl) URL.revokeObjectURL(inputPreviewUrl);
    if (genPreviewUrl) URL.revokeObjectURL(genPreviewUrl);
    setTargetPreviewUrl('');
    setInputPreviewUrl('');
    setGenPreviewUrl('');
  }, [activeModalTab]);

  // ---------------- Helpers ----------------
  const hzToNote = (hz: number): string => {
    if (!hz || hz <= 0) return 'N/A';
    const notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
    const midi = Math.round(12 * Math.log2(hz / 440) + 69);
    const noteIndex = ((midi % 12) + 12) % 12;
    const octave = Math.floor(midi / 12) - 1;
    return `${notes[noteIndex]}${notes[noteIndex] ? octave : ''} (${Math.round(hz)} Hz)`;
  };

  // Recording for Pitch Comparison Input
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      const chunks: Blob[] = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
      };

      recorder.onstop = () => {
        const blob = new Blob(chunks, { type: 'audio/wav' });
        const file = new File([blob], 'mic_input.wav', { type: 'audio/wav' });
        setInputFile(file);
        
        if (inputPreviewUrl) URL.revokeObjectURL(inputPreviewUrl);
        const previewUrl = URL.createObjectURL(file);
        setInputPreviewUrl(previewUrl);
        setCompareResult(null);
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

  // Recording for Profile Generation
  const startGenRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      const chunks: Blob[] = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
      };

      recorder.onstop = () => {
        const blob = new Blob(chunks, { type: 'audio/wav' });
        const file = new File([blob], 'profile_mic.wav', { type: 'audio/wav' });
        setGenFile(file);
        
        if (genPreviewUrl) URL.revokeObjectURL(genPreviewUrl);
        const previewUrl = URL.createObjectURL(file);
        setGenPreviewUrl(previewUrl);
      };

      setGenMediaRecorder(recorder);
      recorder.start();
      setIsGenRecording(true);
      setGenRecordingDuration(0);

      if (genRecordingTimerRef.current) clearInterval(genRecordingTimerRef.current);
      genRecordingTimerRef.current = setInterval(() => {
        setGenRecordingDuration((prev) => prev + 1);
      }, 1000);
    } catch (err) {
      console.error('Failed to start recording:', err);
      guiState.showError('Could not access microphone. Please check permissions.', t('errorTitle'));
    }
  };

  const stopGenRecording = () => {
    if (genMediaRecorder && genMediaRecorder.state !== 'inactive') {
      genMediaRecorder.stop();
      genMediaRecorder.stream.getTracks().forEach((track) => track.stop());
    }
    setIsGenRecording(false);
    if (genRecordingTimerRef.current) {
      clearInterval(genRecordingTimerRef.current);
      genRecordingTimerRef.current = null;
    }
  };

  // Handlers
  const handleTargetChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setTargetFile(file);
      if (targetPreviewUrl) URL.revokeObjectURL(targetPreviewUrl);
      setTargetPreviewUrl(URL.createObjectURL(file));
      setCompareResult(null);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setInputFile(file);
      if (inputPreviewUrl) URL.revokeObjectURL(inputPreviewUrl);
      setInputPreviewUrl(URL.createObjectURL(file));
      setCompareResult(null);
    }
  };

  const handleGenFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setGenFile(file);
      if (genPreviewUrl) URL.revokeObjectURL(genPreviewUrl);
      setGenPreviewUrl(URL.createObjectURL(file));
    }
  };

  const handleCompare = async () => {
    if (!targetFile || !inputFile) {
      guiState.showError(t('voiceAnalyzerUploadBoth'), t('voiceAnalyzerUploadWarning'));
      return;
    }

    setIsComparing(true);
    setCompareResult(null);

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
        setCompareResult(data);
      } else {
        throw new Error(t('voiceAnalysisUnsuccessful'));
      }
    } catch (error) {
      console.error('Voice comparison error:', error);
      guiState.showError(
        `比对失败: ${error instanceof Error ? error.message : String(error)}`,
        t('errorTitle')
      );
    } finally {
      setIsComparing(false);
    }
  };

  const handleApplyPitch = () => {
    if (!compareResult) return;
    appState.serverSetting.updateServerSettings({
      ...appState.serverSetting.serverSetting,
      tran: compareResult.recommended_pitch
    });
    guiState.showError(
      `${t('voiceAnalyzerAppliedPitchSuccess')}${compareResult.recommended_pitch >= 0 ? '+' : ''}${
        compareResult.recommended_pitch
      }`,
      t('confirmTitle')
    );
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

  const handleGenerateProfile = async () => {
    const fileToUpload = genFile;
    if (!fileToUpload) {
      guiState.showError('请先上传音频或录制声音', t('errorTitle'));
      return;
    }
    if (!newProfileName.trim()) {
      guiState.showError('请先输入档案名称', t('errorTitle'));
      return;
    }

    setIsGenerating(true);
    const formData = new FormData();
    formData.append('target_file', fileToUpload);

    try {
      const response = await fetch('/analyze_voice', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      if (data.success) {
        const newProfile = {
          id: `profile-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
          name: newProfileName.trim(),
          type: newProfileType,
          envelope: data.target_envelope,
          sr: data.target_sr,
          f0: data.target_f0,
          centroid: data.target_centroid
        };

        const storedProfilesStr = localStorage.getItem('formant_profiles') || '[]';
        let storedProfiles: any[] = [];
        try {
          storedProfiles = JSON.parse(storedProfilesStr);
        } catch (e) {
          storedProfiles = [];
        }

        storedProfiles.push(newProfile);
        localStorage.setItem('formant_profiles', JSON.stringify(storedProfiles));
        setProfiles(storedProfiles);

        // Reset fields
        setNewProfileName('');
        setGenFile(null);
        if (genPreviewUrl) URL.revokeObjectURL(genPreviewUrl);
        setGenPreviewUrl('');

        guiState.showError('音色档案生成并保存成功！可在主界面的音色跃迁修正下拉框中直接选用。', t('confirmTitle'));
      } else {
        throw new Error('Analysis failed');
      }
    } catch (e) {
      console.error('Error generating profile:', e);
      guiState.showError(`生成档案失败: ${e instanceof Error ? e.message : String(e)}`, t('errorTitle'));
    } finally {
      setIsGenerating(false);
    }
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
    setGenFile(null);
    if (genPreviewUrl) URL.revokeObjectURL(genPreviewUrl);
    setGenPreviewUrl('');
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
            onClick={() => setActiveModalTab('pitch')}
            className={`pb-2 px-4 border-b-2 transition-all ${
              activeModalTab === 'pitch'
                ? 'border-primary text-primary'
                : 'border-transparent text-on-surface-variant hover:text-on-surface'
            }`}
          >
            {t('voiceAnalyzerTabPitchCompare')}
          </button>
          <button
            onClick={() => setActiveModalTab('formant')}
            className={`pb-2 px-4 border-b-2 transition-all ${
              activeModalTab === 'formant'
                ? 'border-primary text-primary'
                : 'border-transparent text-on-surface-variant hover:text-on-surface'
            }`}
          >
            {t('voiceAnalyzerTabFormantManager')}
          </button>
        </div>

        {activeModalTab === 'pitch' ? (
          <>
            {/* Upload Panels for Pitch Compare */}
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
                        onClick={() => { setInputTab('upload'); setCompareResult(null); }}
                        disabled={isRecording}
                        className={`px-3 py-1 rounded-full font-medium transition-all ${
                          inputTab === 'upload'
                            ? 'bg-secondary text-on-secondary shadow-elevation-1'
                            : 'text-on-surface-variant hover:text-on-surface disabled:opacity-50'
                        }`}
                      >
                        {t('voiceAnalyzerTabUpload')}
                      </button>
                      <button
                        onClick={() => { setInputTab('record'); setCompareResult(null); }}
                        className={`px-3 py-1 rounded-full font-medium transition-all ${
                          inputTab === 'record'
                            ? 'bg-secondary text-on-secondary shadow-elevation-1'
                            : 'text-on-surface-variant hover:text-on-surface'
                        }`}
                      >
                        {t('voiceAnalyzerTabRecord')}
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
                          {t('voiceAnalyzerScriptGuidance')}
                        </p>
                        <p className="text-on-surface-variant italic font-medium px-1 bg-surface-container-high rounded p-1.5 leading-normal">
                          {t('voiceAnalyzerScriptText')}
                        </p>
                      </div>

                      {/* Controls */}
                      <div className="flex items-center justify-center space-x-4 py-1">
                        <button
                          onClick={isRecording ? stopRecording : startRecording}
                          disabled={isComparing}
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

            {/* Compare Action Button */}
            <div className="flex justify-center pt-2">
              <button
                onClick={handleCompare}
                disabled={isComparing || !targetFile || !inputFile}
                className={`flex items-center space-x-2 px-8 py-2.5 rounded-full text-sm font-semibold transition-all duration-150 ${
                  isComparing || !targetFile || !inputFile
                    ? 'bg-surface-container-highest text-on-surface-variant/30 cursor-not-allowed'
                    : 'bg-primary text-on-primary hover:shadow-elevation-1 hover:scale-103 active:scale-97'
                }`}
              >
                <FontAwesomeIcon icon={faSync} className={isComparing ? 'animate-spin' : ''} />
                <span>{isComparing ? '正在比对音高...' : '开始音高比对'}</span>
              </button>
            </div>

            {/* Compare Results Section */}
            {compareResult && (
              <div className="p-6 rounded-2xl border border-outline-variant bg-surface-container-low space-y-6 animate-fadeIn">
                <h4 className="text-base font-bold text-center border-b border-outline-variant/30 pb-2.5 mb-4 text-on-surface uppercase tracking-wider">
                  音高诊断报告
                </h4>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-xs">
                  <div className="p-4 rounded-xl bg-surface-container border border-outline-variant space-y-2">
                    <p className="font-bold text-primary">{t('voiceAnalyzerPitchComparison')}</p>
                    <div className="flex justify-between">
                      <span className="text-on-surface-variant/70">{t('voiceAnalyzerStandardTargetLabel')}</span>
                      <span className="font-semibold text-on-surface">{hzToNote(compareResult.target_f0)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-on-surface-variant/70">{t('voiceAnalyzerYourInputLabel')}</span>
                      <span className="font-semibold text-on-surface">{hzToNote(compareResult.input_f0)}</span>
                    </div>
                  </div>

                  <div className="p-4 rounded-xl bg-surface-container border border-outline-variant space-y-2">
                    <p className="font-bold text-secondary">声道特征音色比对 (重心)</p>
                    <div className="flex justify-between">
                      <span className="text-on-surface-variant/70">{t('voiceAnalyzerStandardTargetLabel')}</span>
                      <span className="font-semibold text-on-surface">{Math.round(compareResult.target_centroid)} Hz</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-on-surface-variant/70">{t('voiceAnalyzerYourInputLabel')}</span>
                      <span className="font-semibold text-on-surface">{Math.round(compareResult.input_centroid)} Hz</span>
                    </div>
                  </div>
                </div>

                <div className="bg-surface-container border border-outline-variant rounded-xl p-5 space-y-4">
                  <p className="font-bold text-center text-sm text-on-surface-variant">
                    推荐音高变调值
                  </p>

                  <div className="flex flex-col md:flex-row md:items-center justify-between p-3 rounded-lg hover:bg-primary/4 transition-colors">
                    <div className="mb-2 md:mb-0 flex items-baseline">
                      <span className="font-semibold text-xs text-on-surface">{t('voiceAnalyzerRecommendedPitch')}</span>
                      <span className="ml-2 text-xl font-extrabold text-primary">
                        {compareResult.recommended_pitch >= 0 ? '+' : ''}
                        {compareResult.recommended_pitch}
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
                </div>
              </div>
            )}
          </>
        ) : (
          /* Profile Manager & Single Profile Generator Tab */
          <div className="space-y-6 animate-fadeIn">
            {/* Generate Single Profile Section */}
            <div className="p-5 rounded-2xl border border-outline-variant bg-surface-container-low space-y-4">
              <h5 className="text-xs font-bold text-primary uppercase tracking-wider mb-2">
                新建共振峰音色档案
              </h5>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                {/* Profile Name Input */}
                <div className="flex flex-col space-y-1.5">
                  <label className="text-[10px] text-on-surface-variant font-bold uppercase tracking-wider">
                    档案自定义名称
                  </label>
                  <input
                    type="text"
                    value={newProfileName}
                    onChange={(e) => setNewProfileName(e.target.value)}
                    placeholder="例如: 我的说话声、KyRin 原音干声..."
                    className="px-3 py-2 rounded-lg bg-surface-container border border-outline-variant text-on-surface font-medium text-xs focus:ring-1 focus:ring-primary focus:border-primary outline-none"
                  />
                </div>

                {/* Profile Type Select */}
                <div className="flex flex-col space-y-1.5">
                  <label className="text-[10px] text-on-surface-variant font-bold uppercase tracking-wider">
                    音色角色类型
                  </label>
                  <select
                    value={newProfileType}
                    onChange={(e) => setNewProfileType(e.target.value as 'input' | 'target')}
                    className="px-3 py-2 rounded-lg bg-surface-container border border-outline-variant text-on-surface font-medium text-xs focus:ring-1 focus:ring-primary focus:border-primary outline-none"
                  >
                    <option value="input">输入音色档案 (Your Voice / 原配)</option>
                    <option value="target">目标音色档案 (Model Voice / 目标角色)</option>
                  </select>
                </div>
              </div>

              {/* Audio Source Input Selector for generation */}
              <div className="border-t border-outline-variant/30 pt-4">
                <div className="flex items-center justify-between mb-3">
                  <span className="text-xs font-semibold text-secondary">
                    音频输入样本 (仅需上传/录制单个声音角色段)
                  </span>
                  
                  {/* Tab Selector */}
                  <div className="flex bg-surface-container rounded-full p-0.5 border border-outline-variant/30 text-[10px]">
                    <button
                      onClick={() => setGenAudioTab('upload')}
                      disabled={isGenRecording}
                      className={`px-3 py-1 rounded-full font-medium transition-all ${
                        genAudioTab === 'upload'
                          ? 'bg-secondary text-on-secondary shadow-elevation-1'
                          : 'text-on-surface-variant hover:text-on-surface disabled:opacity-50'
                      }`}
                    >
                      {t('voiceAnalyzerTabUpload')}
                    </button>
                    <button
                      onClick={() => setGenAudioTab('record')}
                      className={`px-3 py-1 rounded-full font-medium transition-all ${
                        genAudioTab === 'record'
                          ? 'bg-secondary text-on-secondary shadow-elevation-1'
                          : 'text-on-surface-variant hover:text-on-surface'
                      }`}
                    >
                      {t('voiceAnalyzerTabRecord')}
                    </button>
                  </div>
                </div>

                {genAudioTab === 'upload' ? (
                  <label className="flex flex-col items-center justify-center border border-dashed border-outline-variant hover:border-secondary/50 bg-surface-container/40 hover:bg-secondary/5 p-4 rounded-xl cursor-pointer transition-all duration-200 text-center">
                    <FontAwesomeIcon icon={faMusic} className="text-secondary/70 text-lg mb-1.5" />
                    <span className="text-xs text-on-surface-variant font-medium select-none truncate max-w-full px-2">
                      {genFile ? genFile.name : '点击选择要提取的单角色干声音频文件 (.wav/.mp3)'}
                    </span>
                    <input type="file" accept="audio/*" onChange={handleGenFileChange} className="hidden" />
                  </label>
                ) : (
                  /* Gen Recording Mode */
                  <div className="flex flex-col space-y-3">
                    <div className="p-3 bg-surface-container rounded-xl border border-outline-variant/40 text-[11px] leading-relaxed">
                      <p className="text-secondary font-bold mb-1 text-[9px] uppercase tracking-wider">
                        {t('voiceAnalyzerScriptGuidance')}
                      </p>
                      <p className="text-on-surface-variant italic font-medium px-1 bg-surface-container-high rounded p-1.5 leading-normal">
                        {t('voiceAnalyzerScriptText')}
                      </p>
                    </div>

                    <div className="flex items-center justify-center space-x-4 py-1">
                      <button
                        onClick={isGenRecording ? stopGenRecording : startGenRecording}
                        disabled={isGenerating}
                        className={`relative flex items-center justify-center w-12 h-12 rounded-full transition-all duration-300 ${
                          isGenRecording
                            ? 'bg-error text-on-error hover:scale-105 active:scale-95 animate-pulse'
                            : 'bg-secondary text-on-secondary hover:scale-105 active:scale-95 hover:shadow-elevation-1'
                        } disabled:opacity-50 disabled:cursor-not-allowed`}
                      >
                        <FontAwesomeIcon icon={isGenRecording ? faStop : faMicrophone} className="text-lg" />
                      </button>
                      <div className="flex flex-col text-left">
                        <span className="text-[10px] text-on-surface-variant font-bold uppercase tracking-wider">
                          {isGenRecording ? '正在录音...' : '点击录制声音样品'}
                        </span>
                        <span className="text-xs font-mono font-bold text-on-surface mt-0.5">
                          {genRecordingDuration}s / 10s
                        </span>
                      </div>
                    </div>
                  </div>
                )}

                {genPreviewUrl && !isGenRecording && (
                  <div className="mt-4">
                    <p className="text-[10px] text-on-surface-variant font-bold mb-1">{t('voiceAnalyzerPreviewAudio')}</p>
                    <audio src={genPreviewUrl} controls className="w-full h-8" />
                  </div>
                )}
              </div>

              {/* Generate Button */}
              <div className="flex justify-center pt-4 border-t border-outline-variant/20">
                <button
                  onClick={handleGenerateProfile}
                  disabled={isGenerating || !genFile || !newProfileName.trim()}
                  className={`flex items-center space-x-2 px-8 py-2 rounded-full text-xs font-semibold transition-all duration-150 ${
                    isGenerating || !genFile || !newProfileName.trim()
                      ? 'bg-surface-container-highest text-on-surface-variant/30 cursor-not-allowed'
                      : 'bg-primary text-on-primary hover:shadow-elevation-1 hover:scale-103 active:scale-97'
                  }`}
                >
                  <FontAwesomeIcon icon={faSync} className={isGenerating ? 'animate-spin' : ''} />
                  <span>{isGenerating ? '正在分析并生成共振峰档案...' : '生成并保存音色档案'}</span>
                </button>
              </div>
            </div>

            {/* Stored Profiles Table */}
            <div className="p-5 rounded-2xl border border-outline-variant bg-surface-container-low space-y-4">
              <h5 className="text-xs font-bold text-primary uppercase tracking-wider mb-2">
                {t('voiceAnalyzerStoredProfiles')}
              </h5>

              {profiles.length === 0 ? (
                <div className="text-center py-6 text-on-surface-variant/60 italic text-xs">
                  {t('voiceAnalyzerNoProfiles')}
                </div>
              ) : (
                <div className="overflow-x-auto border border-outline-variant/30 rounded-xl bg-surface-container/30">
                  <table className="w-full text-left text-xs border-collapse">
                    <thead>
                      <tr className="bg-surface-container border-b border-outline-variant/30 font-bold text-on-surface-variant/80">
                        <th className="p-3 text-left">{t('voiceAnalyzerColName')}</th>
                        <th className="p-3 text-left w-20">{t('voiceAnalyzerColType')}</th>
                        <th className="p-3 text-left w-20">{t('voiceAnalyzerColSR')}</th>
                        <th className="p-3 text-left w-24">{t('voiceAnalyzerColCentroid')}</th>
                        <th className="p-3 text-center w-28">{t('voiceAnalyzerColActions')}</th>
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
                              {p.type === 'input' ? '输入' : '目标'}
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
                                  title={t('saveLabel')}
                                >
                                  <FontAwesomeIcon icon={faSave} />
                                </button>
                                <button
                                  onClick={handleRenameProfileCancel}
                                  className="p-1 px-2 text-on-surface-variant hover:bg-surface-container-highest rounded transition-colors"
                                  title={t('cancelLabel')}
                                >
                                  <FontAwesomeIcon icon={faTimes} />
                                </button>
                              </div>
                            ) : (
                              <div className="flex items-center justify-center space-x-2">
                                <button
                                  onClick={() => handleRenameProfileStart(p.id, p.name)}
                                  className="p-1 px-2 text-on-surface-variant hover:text-primary hover:bg-primary/10 rounded transition-colors"
                                  title={t('renameLabel')}
                                >
                                  <FontAwesomeIcon icon={faEdit} />
                                </button>
                                <button
                                  onClick={() => handleDeleteProfile(p.id)}
                                  className="p-1 px-2 text-error hover:bg-error/10 rounded transition-colors"
                                  title={t('deleteLabel')}
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
