import { ClientState } from "@dannadori/voice-changer-client-js";
import { UIContextType } from "../../context/UIContext";
import MD3Select from "../Helpers/MD3Select";

interface GpuInfo {
  id: number;
  name: string;
  backend?: string;
  memory?: number;
}

interface GPUConfigProps {
  appState: ClientState;
  uiState: UIContextType;
}

function GPUConfig({ appState, uiState }: GPUConfigProps) {
  // ---------------- Handlers ----------------

  // Handle GPU Change
  const handleChangeGpu = async (gpuId: number) => {
    const gpuName = appState.serverSetting?.serverSetting?.gpus?.find((gpu) => gpu.id === gpuId)?.name;
    uiState.startLoading(`Changing to Processing Unit: ${gpuName}`);
    await appState.serverSetting.updateServerSettings({
      ...appState.serverSetting?.serverSetting,
      gpu: gpuId
    });
    uiState.stopLoading();
  };

  // ---------------- Render ----------------

  const gpus = appState.serverSetting?.serverSetting?.gpus || [];
  const options = gpus.length > 0
    ? gpus.map((gpu: GpuInfo) => ({
        value: gpu.id,
        label: `${gpu.name} ${gpu.memory ? `(${(gpu.memory / 1024 / 1024 / 1024).toFixed(0)} GB)` : ""}`
      }))
    : [{ value: -1, label: "No GPUs available" }];

  return (
    <div className="bg-surface-container-low p-3 rounded-md border border-outline-variant">
      <MD3Select
        id="gpu"
        label="Processing Unit (GPU)"
        value={appState.serverSetting?.serverSetting?.gpu ?? -1}
        onChange={async (e) => {
          await handleChangeGpu(parseInt(e.target.value));
        }}
        options={options}
        disabled={gpus.length === 0}
      />
    </div>
  );
}

export default GPUConfig;
