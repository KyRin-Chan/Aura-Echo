import { useCallback, useEffect, useRef, useState } from "react";
import { CSS_CLASSES } from "../../styles/constants";
import MD3Slider from "../Helpers/MD3Slider";
import { ClientState } from "@dannadori/voice-changer-client-js";
import { UIContextType } from "../../context/UIContext";
import { t } from "../../locales";

interface ChunkConfigProps {
  appState: ClientState;
  uiState: UIContextType;
}

const WARMUP_DEBOUNCE_MS = 800;

function ChunkConfig({ appState, uiState }: ChunkConfigProps) {
  // ---------------- States ----------------
  const [localChunkSize, setLocalChunkSize] = useState<number>(
    appState.serverSetting?.serverSetting?.serverReadChunkSize ?? 180
  );
  const [localExtraSize, setLocalExtraSize] = useState<number>(
    appState.serverSetting?.serverSetting?.extraConvertSize ?? 0.5
  );
  const [localCushion, setLocalCushion] = useState<number>(
    appState.setting.workletNodeSetting?.playBufferCushion ?? 2
  );

  // ---------------- Refs ----------------
  const commitTimerRef = useRef<number | undefined>(undefined);
  const pendingUpdatesRef = useRef<{
    serverReadChunkSize?: number;
    extraConvertSize?: number;
  }>({});
  const latestAppStateRef = useRef(appState);
  const latestUiStateRef = useRef(uiState);
  latestAppStateRef.current = appState;
  latestUiStateRef.current = uiState;

  // ---------------- Hooks ----------------

  // Set local chunk size (skip if an uncommitted drag update is pending)
  useEffect(() => {
    if (pendingUpdatesRef.current.serverReadChunkSize !== undefined) return;
    const cs = appState.serverSetting?.serverSetting?.serverReadChunkSize;
    if (cs !== undefined) {
      setLocalChunkSize(cs);
    }
  }, [appState.serverSetting?.serverSetting?.serverReadChunkSize]);

  // Set local extra size (skip if an uncommitted drag update is pending)
  useEffect(() => {
    if (pendingUpdatesRef.current.extraConvertSize !== undefined) return;
    const ex = appState.serverSetting?.serverSetting?.extraConvertSize;
    if (ex !== undefined) {
      setLocalExtraSize(ex);
    }
  }, [appState.serverSetting?.serverSetting?.extraConvertSize]);

  // Set local safety cushion size
  useEffect(() => {
    const pc = appState.setting.workletNodeSetting?.playBufferCushion;
    if (pc !== undefined) {
      setLocalCushion(pc);
    }
  }, [appState.setting.workletNodeSetting?.playBufferCushion]);

  const cancelPendingCommit = useCallback(() => {
    if (commitTimerRef.current !== undefined) {
      window.clearTimeout(commitTimerRef.current);
      commitTimerRef.current = undefined;
    }
  }, []);

  useEffect(() => {
    return () => {
      cancelPendingCommit();
    };
  }, [cancelPendingCommit]);

  // ---------------- Handlers ----------------

  const flushPendingUpdates = useCallback(async () => {
    commitTimerRef.current = undefined;
    const updates = { ...pendingUpdatesRef.current };
    pendingUpdatesRef.current = {};

    const currentAppState = latestAppStateRef.current;
    const currentUiState = latestUiStateRef.current;
    const currentServerSetting = currentAppState.serverSetting?.serverSetting;

    const nextChunk = updates.serverReadChunkSize;
    const chunkChanged =
      nextChunk !== undefined && nextChunk !== currentServerSetting?.serverReadChunkSize;

    const nextExtra = updates.extraConvertSize;
    const extraChanged =
      nextExtra !== undefined &&
      Math.abs(nextExtra - (currentServerSetting?.extraConvertSize ?? -1)) > 1e-6;

    if (!chunkChanged && !extraChanged) {
      return;
    }

    const payload = {
      ...currentServerSetting,
      ...(chunkChanged ? { serverReadChunkSize: nextChunk } : {}),
      ...(extraChanged ? { extraConvertSize: nextExtra } : {}),
    };

    if (chunkChanged && nextChunk !== undefined) {
      currentAppState.setWorkletNodeSetting({
        ...currentAppState.setting.workletNodeSetting,
        inputChunkNum: Number(nextChunk),
      });
      currentAppState.trancateBuffer();
    }

    const hasModel = (currentServerSetting?.modelSlotIndex ?? -1) !== -1;
    if (hasModel) currentUiState.startLoading(t('warmingUpPipeline'));
    try {
      await currentAppState.serverSetting.updateServerSettings(payload);
    } finally {
      if (hasModel) currentUiState.stopLoading();
    }
  }, []);

  const scheduleCommit = useCallback(
    (field: 'serverReadChunkSize' | 'extraConvertSize', value: number) => {
      pendingUpdatesRef.current[field] = value;
      cancelPendingCommit();
      commitTimerRef.current = window.setTimeout(() => {
        void flushPendingUpdates();
      }, WARMUP_DEBOUNCE_MS);
    },
    [cancelPendingCommit, flushPendingUpdates]
  );

  // Handle chunk size change (post-drag debounced)
  const handleChangeChunkSize = (value: number) => {
    setLocalChunkSize(value);
    scheduleCommit('serverReadChunkSize', value);
  };

  // Handle extra size change (post-drag debounced)
  const handleChangeExtraSize = (value: number) => {
    setLocalExtraSize(value);
    scheduleCommit('extraConvertSize', value);
  };

  // Handle safety cushion change
  const handleChangeCushion = (value: number) => {
    setLocalCushion(value);
    appState.setWorkletNodeSetting({
      ...appState.setting.workletNodeSetting,
      playBufferCushion: Number(value)
    });
  };

  // ---------------- Render ----------------

  return (
    <div className="flex flex-col space-y-4 bg-surface-container/60 p-3 rounded-lg border border-outline-variant/50">
      <div>
        <label htmlFor="chunk" className={CSS_CLASSES.label}>
          {t('chunkConfigLabel')}:
        </label>
        <MD3Slider
          id="chunk"
          min={15}
          max={360}
          step={15}
          value={localChunkSize}
          commitOnRelease={true}
          debounceMs={0}
          onDragStart={cancelPendingCommit}
          onImmediateChange={(val) => {
            cancelPendingCommit();
            setLocalChunkSize(val);
          }}
          onChange={handleChangeChunkSize}
          disabled={uiState.isConverting}
          showValue={true}
          valueFormatter={(val) => `${((val * 128 * 1000) / 48000).toFixed(1)}ms`}
        />
      </div>
      <div>
        <label htmlFor="cushion" className={CSS_CLASSES.label}>
          {t('safetyCushionLabel')}:
        </label>
        <MD3Slider
          id="cushion"
          min={1}
          max={5}
          step={1}
          value={localCushion}
          onImmediateChange={setLocalCushion}
          onChange={handleChangeCushion}
          disabled={uiState.isConverting}
          showValue={true}
          valueFormatter={(val) => `${val} Chunk`}
        />
      </div>
      <div>
        <label htmlFor="extra" className={CSS_CLASSES.label}>
          {t('extraProcessingTimeLabel')}
        </label>
        <MD3Slider
          id="extra"
          min={0.1}
          max={2.4}
          step={0.1}
          value={localExtraSize}
          commitOnRelease={true}
          debounceMs={0}
          onDragStart={cancelPendingCommit}
          onImmediateChange={(val) => {
            cancelPendingCommit();
            setLocalExtraSize(val);
          }}
          onChange={handleChangeExtraSize}
          disabled={uiState.isConverting}
          showValue={true}
          valueFormatter={(val) => `${val.toFixed(1)} s`}
        />
      </div>
    </div>
  );
}

export default ChunkConfig;