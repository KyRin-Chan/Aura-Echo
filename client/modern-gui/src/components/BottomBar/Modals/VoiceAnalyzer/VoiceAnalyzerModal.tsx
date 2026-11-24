import { useState, JSX, useEffect, useRef } from 'react';
import { ClientState } from '@dannadori/voice-changer-client-js';
import { CSS_CLASSES } from '../../../../styles/constants';
import GenericModal from '../../../Modals/GenericModal';
import { UIContextType } from '../../../../context/UIContext';
import { t } from '../../../../locales';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faSync,
  faMicrophone,
  faCheck,
  faMusic,
  faStop,
  faTrash,
  faEdit,
  faSave,
  faTimes
} from '@fortawesome/free-solid-svg-icons';
import {
  FormantProfile,
  ModelFormantBinding,
  getAllFormantProfiles,
  getAllFormantBindings,
  saveFormantProfile,
  saveFormantBinding,
  deleteFormantProfile,
  deleteFormantBinding,
  syncProfilesFromServer,
  syncBindingsFromServer,
  serializePitchStats,
  serializeProfileEnvelope
} from '../../../../utils/formantStorage';

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

interface CalibrationResult {
  success: boolean;
  calibrated_with_rvc: boolean;
  model_slot_index: number;
  calibrated_tran: number;
  calibrated_formant_shift: number;
  target_f0: number;
  target_f0_log2_mean?: number;
  target_f0_log2_std?: number;
  target_f0_p10?: number;
  target_f0_p90?: number;
  target_centroid: number;
  target_envelope: number[];
  target_sr: number;
  input_f0_log2_mean?: number;
  input_f0_log2_std?: number;
  input_f0_p10?: number;
  input_f0_p90?: number;
  baseline_f0: number;
  baseline_centroid: number;
  baseline_envelope: number[];
  baseline_sr: number;
  rms_residual_db: number;
  max_correction_db: number;
}

const inferExtFromMime = (mimeType: string): string => {
  const lower = (mimeType || '').toLowerCase();
  if (lower.includes('webm')) return '.webm';
  if (lower.includes('ogg')) return '.ogg';
  if (lower.includes('mp4') || lower.includes('aac') || lower.includes('m4a')) return '.m4a';
  return '.wav';
};

const encodeAudioBufferToWav = (audioBuffer: AudioBuffer): ArrayBuffer => {
  const numChannels = Math.max(1, audioBuffer.numberOfChannels);
  const length = audioBuffer.length;
  const sampleRate = audioBuffer.sampleRate;

  const mono = new Float32Array(length);
  for (let ch = 0; ch < numChannels; ch++) {
    const chData = audioBuffer.getChannelData(ch);
    for (let i = 0; i < length; i++) {
      mono[i] += chData[i] / numChannels;
    }
  }

  const dataByteLength = length * 2;
  const buffer = new ArrayBuffer(44 + dataByteLength);
  const view = new DataView(buffer);

  const writeStr = (offset: number, str: string) => {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  };

  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + dataByteLength, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true); // PCM chunk size
  view.setUint16(20, 1, true); // PCM format
  view.setUint16(22, 1, true); // Mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // Byte rate
  view.setUint16(32, 2, true); // Block align
  view.setUint16(34, 16, true); // Bits per sample
  writeStr(36, 'data');
  view.setUint32(40, dataByteLength, true);

  let offset = 44;
  for (let i = 0; i < length; i++) {
    const s = Math.max(-1, Math.min(1, mono[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }
  return buffer;
};

const encodeBlobToWavFile = async (blob: Blob, filename: string): Promise<File> => {
  const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (AudioCtx) {
    const audioCtx = new AudioCtx();
    try {
      const arrBuf = await blob.arrayBuffer();
      const decoded = await audioCtx.decodeAudioData(arrBuf);
      const wavBuf = encodeAudioBufferToWav(decoded);
      const wavName = filename.replace(/\.[^.]+$/, '') + '.wav';
      return new File([wavBuf], wavName, { type: 'audio/wav' });
    } catch (err) {
      console.warn('WAV decoding failed, falling back to native MediaRecorder format:', err);
    } finally {
      try {
        await audioCtx.close();
      } catch {
        // Ignore close errors
      }
    }
  }
  const fallbackExt = inferExtFromMime(blob.type);
  const fallbackName = filename.replace(/\.[^.]+$/, '') + fallbackExt;
  return new File([blob], fallbackName, { type: blob.type || 'audio/wav' });
};

const VoiceAnalyzerModal = ({
  appState,
  guiState,
  showVoiceAnalyzer,
  setShowVoiceAnalyzer
}: VoiceAnalyzerModalProps): JSX.Element | null => {
  // ---------------- State ----------------
  const [activeModalTab, setActiveModalTab] = useState<'pitch' | 'formant'>('pitch');

  // Tab 1: Pitch & Formant Comparison State
  const [inputTab, setInputTab] = useState<'upload' | 'record'>('upload');
  const [targetFile, setTargetFile] = useState<File | null>(null);
  const [inputFile, setInputFile] = useState<File | null>(null);
  const [isComparing, setIsComparing] = useState<boolean>(false);
  const [compareResult, setCompareResult] = useState<AnalysisResult | null>(null);
  const [targetPreviewUrl, setTargetPreviewUrl] = useState<string>('');
  const [inputPreviewUrl, setInputPreviewUrl] = useState<string>('');

  // Live recording for Tab 1
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [recordingSeconds, setRecordingSeconds] = useState<number>(0);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordingTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Tab 2: Match EQ Calibration & Profile Manager State
  const [profiles, setProfiles] = useState<FormantProfile[]>([]);
  const [bindings, setBindings] = useState<Record<string | number, ModelFormantBinding>>({});
  const [editingProfileId, setEditingProfileId] = useState<string | null>(null);
  const [editingName, setEditingName] = useState<string>('');

  const [calProfileName, setCalProfileName] = useState<string>('');
  const [calTargetFile, setCalTargetFile] = useState<File | null>(null);
  const [calTargetPreviewUrl, setCalTargetPreviewUrl] = useState<string>('');
  const [calInputTab, setCalInputTab] = useState<'upload' | 'record'>('upload');
  const [calInputFile, setCalInputFile] = useState<File | null>(null);
  const [calInputPreviewUrl, setCalInputPreviewUrl] = useState<string>('');
  const [runRvcConversion, setRunRvcConversion] = useState<boolean>(true);
  const [isCalibrating, setIsCalibrating] = useState<boolean>(false);
  const [calResult, setCalResult] = useState<CalibrationResult | null>(null);

  // Live recording for Tab 2 Calibration Input
  const [isCalRecording, setIsCalRecording] = useState<boolean>(false);
  const [calRecordingSeconds, setCalRecordingSeconds] = useState<number>(0);
  const calMediaRecorderRef = useRef<MediaRecorder | null>(null);
  const calRecordingTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const activeSlotIndex = appState.serverSetting?.serverSetting?.modelSlotIndex ?? -1;
  const activeSlot =
    activeSlotIndex >= 0
      ? appState.serverSetting?.serverSetting?.modelSlots?.[activeSlotIndex]
      : undefined;
  const hasActiveModel = Boolean(activeSlot && activeSlot.voiceChangerType);

  // Load profiles and bindings from storage & server
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
      if (calTargetPreviewUrl) URL.revokeObjectURL(calTargetPreviewUrl);
      if (calInputPreviewUrl) URL.revokeObjectURL(calInputPreviewUrl);
      if (recordingTimerRef.current) clearInterval(recordingTimerRef.current);
      if (calRecordingTimerRef.current) clearInterval(calRecordingTimerRef.current);
    };
  }, [targetPreviewUrl, inputPreviewUrl, calTargetPreviewUrl, calInputPreviewUrl]);

  // Tab change cleanup
  useEffect(() => {
    stopRecording();
    stopCalRecording();
    setTargetFile(null);
    setInputFile(null);
    setCompareResult(null);
    setCalProfileName('');
    setCalTargetFile(null);
    setCalInputFile(null);
    setCalResult(null);
    if (targetPreviewUrl) URL.revokeObjectURL(targetPreviewUrl);
    if (inputPreviewUrl) URL.revokeObjectURL(inputPreviewUrl);
    if (calTargetPreviewUrl) URL.revokeObjectURL(calTargetPreviewUrl);
    if (calInputPreviewUrl) URL.revokeObjectURL(calInputPreviewUrl);
    setTargetPreviewUrl('');
    setInputPreviewUrl('');
    setCalTargetPreviewUrl('');
    setCalInputPreviewUrl('');
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

  // Recording for Tab 1 Comparison Input
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      const chunks: Blob[] = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
      };

      recorder.onstop = async () => {
        const blob = new Blob(chunks, { type: recorder.mimeType || 'audio/webm' });
        const file = await encodeBlobToWavFile(blob, 'mic_input.wav');
        setInputFile(file);

        if (inputPreviewUrl) URL.revokeObjectURL(inputPreviewUrl);
        setInputPreviewUrl(URL.createObjectURL(file));
        setCompareResult(null);
      };

      mediaRecorderRef.current = recorder;
      recorder.start();
      setIsRecording(true);
      setRecordingSeconds(0);

      if (recordingTimerRef.current) clearInterval(recordingTimerRef.current);
      let elapsed = 0;
      recordingTimerRef.current = setInterval(() => {
        elapsed += 1;
        setRecordingSeconds(elapsed);
        if (elapsed >= 10) {
          stopRecording();
        }
      }, 1000);
    } catch (err) {
      console.error('Failed to start recording:', err);
      guiState.showError('Could not access microphone. Please check permissions.', 'Error');
    }
  };

  const stopRecording = () => {
    const rec = mediaRecorderRef.current;
    if (rec && rec.state !== 'inactive') {
      rec.stop();
      rec.stream.getTracks().forEach((track) => track.stop());
    }
    setIsRecording(false);
    if (recordingTimerRef.current) {
      clearInterval(recordingTimerRef.current);
      recordingTimerRef.current = null;
    }
  };

  // Recording for Tab 2 Calibration Input
  const startCalRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      const chunks: Blob[] = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
      };

      recorder.onstop = async () => {
        const blob = new Blob(chunks, { type: recorder.mimeType || 'audio/webm' });
        const file = await encodeBlobToWavFile(blob, 'calibration_mic.wav');
        setCalInputFile(file);

        if (calInputPreviewUrl) URL.revokeObjectURL(calInputPreviewUrl);
        setCalInputPreviewUrl(URL.createObjectURL(file));
        setCalResult(null);
      };

      calMediaRecorderRef.current = recorder;
      recorder.start();
      setIsCalRecording(true);
      setCalRecordingSeconds(0);

      if (calRecordingTimerRef.current) clearInterval(calRecordingTimerRef.current);
      let elapsed = 0;
      calRecordingTimerRef.current = setInterval(() => {
        elapsed += 1;
        setCalRecordingSeconds(elapsed);
        if (elapsed >= 10) {
          stopCalRecording();
        }
      }, 1000);
    } catch (err) {
      console.error('Failed to start recording:', err);
      guiState.showError('Could not access microphone. Please check permissions.', 'Error');
    }
  };

  const stopCalRecording = () => {
    const rec = calMediaRecorderRef.current;
    if (rec && rec.state !== 'inactive') {
      rec.stop();
      rec.stream.getTracks().forEach((track) => track.stop());
    }
    setIsCalRecording(false);
    if (calRecordingTimerRef.current) {
      clearInterval(calRecordingTimerRef.current);
      calRecordingTimerRef.current = null;
    }
  };

  // File Handlers
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

  const handleCalTargetChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setCalTargetFile(file);
      if (calTargetPreviewUrl) URL.revokeObjectURL(calTargetPreviewUrl);
      setCalTargetPreviewUrl(URL.createObjectURL(file));
      setCalResult(null);
    }
  };

  const handleCalInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setCalInputFile(file);
      if (calInputPreviewUrl) URL.revokeObjectURL(calInputPreviewUrl);
      setCalInputPreviewUrl(URL.createObjectURL(file));
      setCalResult(null);
    }
  };

  // Tab 1 Actions
  const handleCompare = async () => {
    if (!targetFile || !inputFile) {
      guiState.showError(t('voiceAnalyzerUploadBoth'), 'Warning');
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
        `${t('voiceAnalyzerCompareFailed')}${error instanceof Error ? error.message : String(error)}`,
        'Error'
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
      'Confirm'
    );
  };

  const handleApplyFormantShift = () => {
    if (!compareResult) return;
    appState.serverSetting.updateServerSettings({
      ...appState.serverSetting.serverSetting,
      formantShift: compareResult.recommended_formant_shift
    });
    guiState.showError(
      `${t('voiceAnalyzerAppliedFormantSuccess')}${compareResult.recommended_formant_shift >= 0 ? '+' : ''}${
        compareResult.recommended_formant_shift.toFixed(2)
      }`,
      'Confirm'
    );
  };

  const handleApplyAll = () => {
    if (!compareResult) return;
    appState.serverSetting.updateServerSettings({
      ...appState.serverSetting.serverSetting,
      tran: compareResult.recommended_pitch,
      formantShift: compareResult.recommended_formant_shift
    });
    guiState.showError(
      `${t('voiceAnalyzerAppliedBothSuccess')}${compareResult.recommended_pitch >= 0 ? '+' : ''}${
        compareResult.recommended_pitch
      }, ${t('formantShiftLabel')} ${compareResult.recommended_formant_shift >= 0 ? '+' : ''}${
        compareResult.recommended_formant_shift.toFixed(2)
      }`,
      'Confirm'
    );
  };

  // Profile Manager Handlers
  const handleRenameProfileStart = (profileId: string, currentName: string) => {
    setEditingProfileId(profileId);
    setEditingName(currentName);
  };

  const handleRenameProfileSave = () => {
    if (!editingProfileId || !editingName.trim()) return;
    try {
      const target = profiles.find((p) => p.id === editingProfileId);
      if (target) {
        saveFormantProfile({ ...target, name: editingName.trim() });
        setProfiles(getAllFormantProfiles());
      }
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
      deleteFormantProfile(profileId);
      const allBindings = getAllFormantBindings();
      Object.entries(allBindings).forEach(([slotStr, bind]) => {
        const slotIndex = Number(slotStr);
        if (bind && (bind.targetProfileId === profileId || bind.inputProfileId === profileId)) {
          deleteFormantBinding(slotIndex);

          if (slotIndex === appState.serverSetting.serverSetting.modelSlotIndex) {
            appState.serverSetting.updateServerSettings({
              ...appState.serverSetting.serverSetting,
              formantProfileActive: false
            });
          }
        }
      });

      setProfiles(getAllFormantProfiles());
      setBindings(getAllFormantBindings());
    } catch (e) {
      console.error('Error deleting profile:', e);
    }
  };

  // Tab 2: Calibrate Match EQ Handler
  const handleCalibrateMatchEq = async () => {
    if (!calTargetFile || !calInputFile) {
      guiState.showError(t('voiceAnalyzerUploadBoth'), 'Warning');
      return;
    }
    if (runRvcConversion && !hasActiveModel) {
      guiState.showError(t('selectModelFirst'), 'Warning');
      return;
    }

    setIsCalibrating(true);
    setCalResult(null);

    const formData = new FormData();
    formData.append('target_file', calTargetFile);
    formData.append('input_file', calInputFile);
    formData.append('run_rvc', String(runRvcConversion));

    try {
      const response = await fetch('/calibrate_match_eq', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();
      if (!response.ok || !data.success) {
        throw new Error(data.error || `HTTP error! status: ${response.status}`);
      }

      const calData: CalibrationResult = data;
      const baseName =
        calProfileName.trim() ||
        (activeSlot?.name ? `${activeSlot.name}` : 'MatchEQ');
      const stamp = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

      const targetProfile: FormantProfile = {
        id: `profile-tgt-${stamp}`,
        name: `${baseName} [${t('matchEqTargetSuffix')}]`,
        type: 'target',
        mode: 'target',
        envelope: calData.target_envelope,
        sr: calData.target_sr,
        f0: calData.target_f0,
        f0Log2Mean: calData.target_f0_log2_mean,
        f0Log2Std: calData.target_f0_log2_std,
        f0P10: calData.target_f0_p10,
        f0P90: calData.target_f0_p90,
        centroid: calData.target_centroid
      };

      const baselineProfile: FormantProfile = {
        id: `profile-base-${stamp}`,
        name: `${baseName} [${t('matchEqBaselineSuffix')}]`,
        type: 'input',
        mode: 'rvc_output',
        envelope: calData.baseline_envelope,
        sr: calData.baseline_sr,
        f0: calData.baseline_f0,
        f0Log2Mean: calData.input_f0_log2_mean,
        f0Log2Std: calData.input_f0_log2_std,
        f0P10: calData.input_f0_p10,
        f0P90: calData.input_f0_p90,
        centroid: calData.baseline_centroid,
        modelSlotIndex: calData.model_slot_index,
        calibratedTran: calData.calibrated_tran,
        calibratedFormantShift: calData.calibrated_formant_shift,
        rmsResidualDb: calData.rms_residual_db
      };

      saveFormantProfile(targetProfile);
      saveFormantProfile(baselineProfile);

      const defaultStrength = 0.5;
      const defaultPitchScale = 1.0;
      const defaultPitchOffset = 0.0;
      const serializedPitchStats = serializePitchStats(baselineProfile, targetProfile);
      if (activeSlotIndex !== undefined && activeSlotIndex !== -1) {
        saveFormantBinding(activeSlotIndex, {
          inputProfileId: baselineProfile.id,
          targetProfileId: targetProfile.id,
          active: true,
          strength: defaultStrength,
          pitchScale: defaultPitchScale,
          pitchOffset: defaultPitchOffset
        });

        appState.serverSetting.updateServerSettings({
          ...appState.serverSetting.serverSetting,
          formantProfileActive: true,
          formantProfileTargetEnvelope: serializeProfileEnvelope(targetProfile),
          formantProfileTargetSr: targetProfile.sr,
          formantProfileInputEnvelope: serializeProfileEnvelope(baselineProfile),
          formantProfileInputSr: baselineProfile.sr,
          formantProfileStrength: defaultStrength,
          formantProfilePitchStats: serializedPitchStats,
          formantProfilePitchScale: defaultPitchScale,
          formantProfilePitchOffset: defaultPitchOffset
        });
      }

      setProfiles(getAllFormantProfiles());
      setBindings(getAllFormantBindings());
      setCalResult(calData);
      setCalProfileName('');

      guiState.showError(t('voiceAnalyzerProfileSavedSuccess'), 'Confirm');
    } catch (e) {
      console.error('Error calibrating Match EQ:', e);
      guiState.showError(
        `${t('voiceAnalyzerProfileGenFailed')}${e instanceof Error ? e.message : String(e)}`,
        'Error'
      );
    } finally {
      setIsCalibrating(false);
    }
  };

  const handleClose = () => {
    stopRecording();
    stopCalRecording();
    setShowVoiceAnalyzer(false);
    setTargetFile(null);
    setInputFile(null);
    setCompareResult(null);
    setCalTargetFile(null);
    setCalInputFile(null);
    setCalResult(null);
    if (targetPreviewUrl) URL.revokeObjectURL(targetPreviewUrl);
    if (inputPreviewUrl) URL.revokeObjectURL(inputPreviewUrl);
    if (calTargetPreviewUrl) URL.revokeObjectURL(calTargetPreviewUrl);
    if (calInputPreviewUrl) URL.revokeObjectURL(calInputPreviewUrl);
    setTargetPreviewUrl('');
    setInputPreviewUrl('');
    setCalTargetPreviewUrl('');
    setCalInputPreviewUrl('');
  };

  const activeBinding = activeSlotIndex >= 0 ? bindings[activeSlotIndex] : undefined;
  const activeBoundBase = activeBinding?.inputProfileId
    ? profiles.find((p) => p.id === activeBinding.inputProfileId)
    : undefined;
  const activeBoundTarget = activeBinding?.targetProfileId
    ? profiles.find((p) => p.id === activeBinding.targetProfileId)
    : undefined;

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
            {/* Upload Panels for Pitch & Formant Shift Compare */}
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
                    <div className="flex bg-surface-container rounded-full p-0.5 border border-outline-variant/30 text-[10px]">
                      <button
                        onClick={() => {
                          setInputTab('upload');
                          setCompareResult(null);
                        }}
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
                        onClick={() => {
                          setInputTab('record');
                          setCompareResult(null);
                        }}
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
                            {isRecording
                              ? t('voiceAnalyzerRecordingStatus') || 'Recording...'
                              : t('voiceAnalyzerRecordStartStatus') || 'Click to Record'}
                          </span>
                          <span className="text-xs font-mono font-bold text-on-surface mt-0.5">
                            {recordingSeconds}s / 10s
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
                <span>{isComparing ? t('voiceAnalyzerComparing') : t('voiceAnalyzerStartAnalysis')}</span>
              </button>
            </div>

            {/* Compare Results Section */}
            {compareResult && (
              <div className="p-6 rounded-2xl border border-outline-variant bg-surface-container-low space-y-6 animate-fadeIn">
                <h4 className="text-base font-bold text-center border-b border-outline-variant/30 pb-2.5 mb-4 text-on-surface uppercase tracking-wider">
                  {t('voiceAnalyzerDiagnostics')}
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
                    <p className="font-bold text-secondary">{t('voiceAnalyzerFormantResonanceLabel')}</p>
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
                    {t('voiceAnalyzerRecommendedParams')}
                  </p>

                  <div className="flex flex-col md:flex-row md:items-center justify-between p-3 rounded-lg hover:bg-primary/4 transition-colors">
                    <div className="mb-2 md:mb-0 flex items-baseline">
                      <span className="font-semibold text-xs text-on-surface">{t('voiceAnalyzerRecommendedPitch')}</span>
                      <span className="ml-2 text-xl font-extrabold text-primary">
                        {compareResult.recommended_pitch >= 0 ? '+' : ''}
                        {compareResult.recommended_pitch}
                      </span>
                      <span className="ml-1 text-[10px] text-on-surface-variant font-medium">
                        {t('voiceAnalyzerSemitones')}
                      </span>
                    </div>
                    <button
                      onClick={handleApplyPitch}
                      className="flex items-center space-x-1.5 px-4 py-1.5 bg-primary text-on-primary text-xs font-semibold rounded-full hover:shadow-elevation-1 active:scale-97 transition-all"
                    >
                      <FontAwesomeIcon icon={faCheck} />
                      <span>{t('voiceAnalyzerApplyPitch')}</span>
                    </button>
                  </div>

                  <div className="flex flex-col md:flex-row md:items-center justify-between p-3 rounded-lg hover:bg-secondary/4 transition-colors border-t border-outline-variant/20">
                    <div className="mb-2 md:mb-0 flex items-baseline">
                      <span className="font-semibold text-xs text-on-surface">{t('voiceAnalyzerRecommendedFormant')}</span>
                      <span className="ml-2 text-xl font-extrabold text-secondary">
                        {compareResult.recommended_formant_shift >= 0 ? '+' : ''}
                        {compareResult.recommended_formant_shift.toFixed(2)}
                      </span>
                      <span className="ml-1 text-[10px] text-on-surface-variant font-medium">
                        {t('voiceAnalyzerSemitones')}
                      </span>
                    </div>
                    <button
                      onClick={handleApplyFormantShift}
                      className="flex items-center space-x-1.5 px-4 py-1.5 bg-secondary text-on-secondary text-xs font-semibold rounded-full hover:shadow-elevation-1 active:scale-97 transition-all"
                    >
                      <FontAwesomeIcon icon={faCheck} />
                      <span>{t('voiceAnalyzerApplyFormant')}</span>
                    </button>
                  </div>

                  <div className="flex justify-center pt-2 border-t border-outline-variant/20">
                    <button
                      onClick={handleApplyAll}
                      className="flex items-center space-x-2 px-6 py-2 bg-primary text-on-primary text-xs font-bold rounded-full hover:shadow-elevation-1 active:scale-97 transition-all"
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
          /* Tab 2: Output Match EQ Calibration & Profile Manager */
          <div className="space-y-6 animate-fadeIn">
            <div className="p-5 rounded-2xl border border-outline-variant bg-surface-container-low space-y-4">
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-2">
                <h5 className="text-xs font-bold text-primary uppercase tracking-wider">
                  {t('matchEqCalibrateTitle')}
                </h5>
                <span className="text-[11px] font-medium text-on-surface-variant">
                  {t('matchEqActiveModelLabel')}{' '}
                  <span className="font-bold text-on-surface">
                    {hasActiveModel
                      ? `${activeSlot?.name || `Slot ${activeSlotIndex}`}`
                      : t('selectModelFirst')}
                  </span>
                </span>
              </div>

              <p className="text-[11px] text-on-surface-variant leading-relaxed">
                {t('matchEqCalibrateDesc')}
              </p>

              {activeBinding && (activeBoundTarget || activeBoundBase) && (
                <div className="p-2.5 rounded-xl bg-surface-container border border-outline-variant/40 text-[11px] flex flex-wrap items-center gap-2">
                  <span className="font-bold text-primary">{t('matchEqActiveBindingLabel')}</span>
                  <span className="text-on-surface">
                    {activeBoundBase?.name || t('voiceAnalyzerNoneOption')} →{' '}
                    {activeBoundTarget?.name || t('voiceAnalyzerNoneOption')}
                  </span>
                </div>
              )}

              {/* Calibration Name */}
              <div className="flex flex-col space-y-1.5">
                <label className="text-[10px] text-on-surface-variant font-bold uppercase tracking-wider">
                  {t('voiceAnalyzerProfileNameLabel')}
                </label>
                <input
                  type="text"
                  value={calProfileName}
                  onChange={(e) => setCalProfileName(e.target.value)}
                  placeholder={t('matchEqNamePlaceholder')}
                  className="h-[44px] px-3 py-2 rounded-xs bg-surface-container border border-outline text-on-surface font-medium text-sm focus:ring-1 focus:ring-primary focus:border-primary outline-none"
                />
              </div>

              {/* Dual Audio Selectors for Calibration */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Target Dry Reference */}
                <div className="p-4 rounded-xl border border-outline-variant/60 bg-surface-container/40 flex flex-col justify-between">
                  <div>
                    <span className="flex items-center space-x-2 mb-2.5 font-semibold text-xs text-primary">
                      <FontAwesomeIcon icon={faMusic} />
                      <span>{t('matchEqTargetSampleLabel')}</span>
                    </span>
                    <label className="flex flex-col items-center justify-center border border-dashed border-outline-variant hover:border-primary/50 bg-surface-container/50 hover:bg-primary/5 p-3.5 rounded-xl cursor-pointer transition-all text-center">
                      <FontAwesomeIcon icon={faMusic} className="text-primary/70 text-base mb-1" />
                      <span className="text-xs text-on-surface-variant font-medium select-none truncate max-w-full px-2">
                        {calTargetFile ? calTargetFile.name : t('clickToChooseFile')}
                      </span>
                      <input type="file" accept="audio/*" onChange={handleCalTargetChange} className="hidden" />
                    </label>
                  </div>
                  {calTargetPreviewUrl && (
                    <div className="mt-3">
                      <audio src={calTargetPreviewUrl} controls className="w-full h-8" />
                    </div>
                  )}
                </div>

                {/* Input Sample (Upload or Record) */}
                <div className="p-4 rounded-xl border border-outline-variant/60 bg-surface-container/40 flex flex-col justify-between">
                  <div>
                    <div className="flex items-center justify-between mb-2.5">
                      <span className="flex items-center space-x-2 font-semibold text-xs text-secondary truncate pr-2">
                        <FontAwesomeIcon icon={faMicrophone} />
                        <span className="truncate">{t('matchEqInputSampleLabel')}</span>
                      </span>
                      <div className="flex shrink-0 bg-surface-container rounded-full p-0.5 border border-outline-variant/30 text-[10px]">
                        <button
                          onClick={() => {
                            setCalInputTab('upload');
                            setCalResult(null);
                          }}
                          disabled={isCalRecording}
                          className={`px-2.5 py-0.5 rounded-full font-medium transition-all ${
                            calInputTab === 'upload'
                              ? 'bg-secondary text-on-secondary shadow-elevation-1'
                              : 'text-on-surface-variant hover:text-on-surface disabled:opacity-50'
                          }`}
                        >
                          {t('voiceAnalyzerTabUpload')}
                        </button>
                        <button
                          onClick={() => {
                            setCalInputTab('record');
                            setCalResult(null);
                          }}
                          className={`px-2.5 py-0.5 rounded-full font-medium transition-all ${
                            calInputTab === 'record'
                              ? 'bg-secondary text-on-secondary shadow-elevation-1'
                              : 'text-on-surface-variant hover:text-on-surface'
                          }`}
                        >
                          {t('voiceAnalyzerTabRecord')}
                        </button>
                      </div>
                    </div>

                    {calInputTab === 'upload' ? (
                      <label className="flex flex-col items-center justify-center border border-dashed border-outline-variant hover:border-secondary/50 bg-surface-container/50 hover:bg-secondary/5 p-3.5 rounded-xl cursor-pointer transition-all text-center">
                        <FontAwesomeIcon icon={faMicrophone} className="text-secondary/70 text-base mb-1" />
                        <span className="text-xs text-on-surface-variant font-medium select-none truncate max-w-full px-2">
                          {calInputFile ? calInputFile.name : t('clickToChooseFile')}
                        </span>
                        <input type="file" accept="audio/*" onChange={handleCalInputChange} className="hidden" />
                      </label>
                    ) : (
                      <div className="flex flex-col space-y-2.5">
                        <div className="p-2.5 bg-surface-container rounded-lg border border-outline-variant/40 text-[10px] leading-relaxed">
                          <p className="text-secondary font-bold mb-0.5 uppercase tracking-wider">
                            {t('voiceAnalyzerScriptGuidance')}
                          </p>
                          <p className="text-on-surface-variant italic font-medium">
                            {t('voiceAnalyzerScriptText')}
                          </p>
                        </div>
                        <div className="flex items-center justify-center space-x-3 py-1">
                          <button
                            onClick={isCalRecording ? stopCalRecording : startCalRecording}
                            disabled={isCalibrating}
                            className={`relative flex items-center justify-center w-10 h-10 rounded-full transition-all duration-300 ${
                              isCalRecording
                                ? 'bg-error text-on-error hover:scale-105 active:scale-95 animate-pulse'
                                : 'bg-secondary text-on-secondary hover:scale-105 active:scale-95'
                            } disabled:opacity-50 disabled:cursor-not-allowed`}
                          >
                            <FontAwesomeIcon icon={isCalRecording ? faStop : faMicrophone} />
                          </button>
                          <div className="flex flex-col text-left">
                            <span className="text-[10px] text-on-surface-variant font-bold uppercase tracking-wider">
                              {isCalRecording
                                ? t('voiceAnalyzerRecordingStatus')
                                : t('voiceAnalyzerRecordStartStatus')}
                            </span>
                            <span className="text-xs font-mono font-bold text-on-surface">
                              {calRecordingSeconds}s / 10s
                            </span>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                  {calInputPreviewUrl && !isCalRecording && (
                    <div className="mt-3">
                      <audio src={calInputPreviewUrl} controls className="w-full h-8" />
                    </div>
                  )}
                </div>
              </div>

              {/* Run RVC Toggle */}
              <label className="flex items-center space-x-2.5 text-xs text-on-surface cursor-pointer select-none pt-1">
                <input
                  type="checkbox"
                  checked={runRvcConversion}
                  onChange={(e) => setRunRvcConversion(e.target.checked)}
                  className="rounded border-outline text-primary focus:ring-primary"
                />
                <span className="font-medium">{t('matchEqRunRvcLabel')}</span>
              </label>

              {/* Calibrate Button */}
              <div className="flex justify-center pt-2 border-t border-outline-variant/20">
                <button
                  onClick={handleCalibrateMatchEq}
                  disabled={
                    isCalibrating ||
                    !calTargetFile ||
                    !calInputFile ||
                    (runRvcConversion && !hasActiveModel)
                  }
                  className={`flex items-center space-x-2 px-8 py-2.5 rounded-full text-xs font-semibold transition-all duration-150 ${
                    isCalibrating ||
                    !calTargetFile ||
                    !calInputFile ||
                    (runRvcConversion && !hasActiveModel)
                      ? 'bg-surface-container-highest text-on-surface-variant/30 cursor-not-allowed'
                      : 'bg-primary text-on-primary hover:shadow-elevation-1 hover:scale-103 active:scale-97'
                  }`}
                >
                  <FontAwesomeIcon icon={faSync} className={isCalibrating ? 'animate-spin' : ''} />
                  <span>
                    {isCalibrating ? t('matchEqCalibratingBtn') : t('matchEqCalibrateBtn')}
                  </span>
                </button>
              </div>

              {/* Calibration Residual Diagnostics */}
              {calResult && (
                <div className="p-4 rounded-xl bg-surface-container border border-outline-variant space-y-2 text-xs animate-fadeIn">
                  <p className="font-bold text-primary uppercase tracking-wider text-[11px]">
                    {t('matchEqDiagTitle')}
                  </p>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                    <div className="flex justify-between">
                      <span className="text-on-surface-variant">{t('matchEqRmsResidualLabel')}</span>
                      <span className="font-mono font-bold text-on-surface">
                        {calResult.rms_residual_db.toFixed(2)} dB
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-on-surface-variant">{t('matchEqMaxCorrectionLabel')}</span>
                      <span className="font-mono font-bold text-on-surface">
                        ±{calResult.max_correction_db.toFixed(2)} dB
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-on-surface-variant">{t('voiceAnalyzerStandardTargetLabel')}</span>
                      <span className="font-mono font-semibold text-on-surface">
                        {Math.round(calResult.target_centroid)} Hz
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-on-surface-variant">{t('matchEqBaselineSuffix')}</span>
                      <span className="font-mono font-semibold text-on-surface">
                        {Math.round(calResult.baseline_centroid)} Hz
                      </span>
                    </div>
                    {Boolean(calResult.input_f0_p90 && calResult.target_f0_p90) && (
                      <div className="flex justify-between md:col-span-2 pt-1 border-t border-outline-variant/20">
                        <span className="text-on-surface-variant">{t('matchEqPitchRangeLabel')}</span>
                        <span className="font-mono font-semibold text-on-surface">
                          {Math.round(calResult.input_f0_p10 || 0)}–{Math.round(calResult.input_f0_p90 || 0)} Hz →{' '}
                          {Math.round(calResult.target_f0_p10 || 0)}–{Math.round(calResult.target_f0_p90 || 0)} Hz
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              )}
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
                        <th className="p-3 text-left w-28">{t('voiceAnalyzerColType')}</th>
                        <th className="p-3 text-left w-20">{t('voiceAnalyzerColSR')}</th>
                        <th className="p-3 text-left w-24">{t('voiceAnalyzerColCentroid')}</th>
                        <th className="p-3 text-center w-24">{t('voiceAnalyzerColActions')}</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-outline-variant/20">
                      {profiles.map((p) => {
                        const isTarget = p.type === 'target';
                        const isBaseline = !isTarget && p.mode === 'rvc_output';
                        const badgeClass = isTarget
                          ? 'bg-primary/10 text-primary border border-primary/20'
                          : isBaseline
                          ? 'bg-secondary/10 text-secondary border border-secondary/20'
                          : 'bg-surface-container-highest text-on-surface-variant border border-outline-variant/40';
                        const badgeLabel = isTarget
                          ? t('voiceAnalyzerTypeTarget')
                          : isBaseline
                          ? t('matchEqTypeBaseline')
                          : t('matchEqTypeLegacyInput');

                        return (
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
                              <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold whitespace-nowrap ${badgeClass}`}>
                                {badgeLabel}
                              </span>
                            </td>
                            <td className="p-3 font-mono text-on-surface-variant">{p.sr}Hz</td>
                            <td className="p-3 font-mono text-on-surface-variant">
                              {p.centroid ? `${Math.round(p.centroid)}Hz` : 'N/A'}
                            </td>
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
                        );
                      })}
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
};

export default VoiceAnalyzerModal;
