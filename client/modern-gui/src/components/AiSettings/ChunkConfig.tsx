import { useEffect, useState } from "react";
import { CSS_CLASSES } from "../../styles/constants";
import MD3Slider from "../Helpers/MD3Slider";
import { ClientState } from "@dannadori/voice-changer-client-js";
import { UIContextType } from "../../context/UIContext";

interface ChunkConfigProps {
  appState: ClientState;
  uiState: UIContextType;
}

function ChunkConfig({ appState, uiState }: ChunkConfigProps) {
  // ---------------- States ----------------
  const [localChunkSize, setLocalChunkSize] = useState<number>(
    appState.serverSetting?.serverSetting?.serverReadChunkSize
      ? appState.serverSetting.serverSetting.serverReadChunkSize
      : 5
  );
  const [localExtraSize, setLocalExtraSize] = useState<number>(
    appState.serverSetting?.serverSetting?.extraConvertSize ?? 1
  );

  // ---------------- Hooks ----------------

  // Set local chunk size
  useEffect(() => {
    const cs = appState.serverSetting?.serverSetting?.serverReadChunkSize;
    if (cs !== undefined) {
      setLocalChunkSize(cs);
    }
  }, [appState.serverSetting?.serverSetting?.serverReadChunkSize]);

  // Set local extra size
  useEffect(() => {
    const ex = appState.serverSetting?.serverSetting?.extraConvertSize;
    if (ex !== undefined) {
      setLocalExtraSize(ex);
    }
  }, [appState.serverSetting?.serverSetting?.extraConvertSize]);

  // ---------------- Handlers ----------------

  // Handle chunk size change
  const handleChangeChunkSize = (value: number) => {
    setLocalChunkSize(value);
    appState.setWorkletNodeSetting({ ...appState.setting.workletNodeSetting, inputChunkNum: Number(value) });
    appState.trancateBuffer();
    appState.serverSetting.updateServerSettings({
      ...appState.serverSetting?.serverSetting,
      serverReadChunkSize: value
    });
  };

  // Handle extra size change
  const handleChangeExtraSize = (value: number) => {
    setLocalExtraSize(value);
    appState.serverSetting.updateServerSettings({
      ...appState.serverSetting?.serverSetting,
      extraConvertSize: value
    });
  };

  // ---------------- Render ----------------

  return (
    <div className="flex flex-col space-y-4 bg-surface-container-low p-3 rounded-md border border-outline-variant">
      <div>
        <label htmlFor="chunk" className={CSS_CLASSES.label}>
          Chunk Size:
        </label>
        <MD3Slider
          id="chunk"
          min={1}
          max={1024}
          step={1}
          value={localChunkSize}
          onImmediateChange={setLocalChunkSize}
          onChange={handleChangeChunkSize}
          disabled={uiState.isConverting}
          showValue={true}
          valueFormatter={(val) => `${((val * 128 * 1000) / 48000).toFixed(1)}ms`}
        />
      </div>
      <div>
        <label htmlFor="extra" className={CSS_CLASSES.label}>
          Extra Processing Time (Extra):
        </label>
        <MD3Slider
          id="extra"
          min={0}
          max={5}
          step={0.1}
          value={localExtraSize}
          onImmediateChange={setLocalExtraSize}
          onChange={handleChangeExtraSize}
          disabled={uiState.isConverting}
          showValue={true}
          valueFormatter={(val) => `${val} s`}
        />
      </div>
    </div>
  );
}

export default ChunkConfig;