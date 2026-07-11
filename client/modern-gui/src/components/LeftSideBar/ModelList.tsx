import ModelSlot from "./ModelSlot";
import { ClientState, RVCModelSlot } from "@dannadori/voice-changer-client-js";

interface ModelListProps {
  filteredAndSortedModels: RVCModelSlot[];
  handleSelectModel: (slot: RVCModelSlot) => Promise<void>;
  confirmedSelectedSlotIndex: number | null;
  appState: ClientState;
}

function ModelList({
  filteredAndSortedModels,
  handleSelectModel,
  confirmedSelectedSlotIndex,
  appState
}: ModelListProps) {
  // ---------------- Render ----------------

  return (
    <ul className="space-y-0.5 flex-grow overflow-y-auto min-h-[100px] pr-1">
      {filteredAndSortedModels.length > 0 ? (
        // Render individual model slots with selection and action handlers
        filteredAndSortedModels.map((model) => (
          <ModelSlot
            key={model.slotIndex}
            model={model}
            handleSelectModel={handleSelectModel}
            modelDir={appState.serverSetting.serverSetting.voiceChangerParams.model_dir}
            selected={model.slotIndex === confirmedSelectedSlotIndex}
          />
        ))
      ) : (
        // Empty state: Show when no models match current filter criteria
        <p className="text-center text-xs text-on-surface-variant/60 italic py-6">
          No models found
        </p>
      )}
    </ul>
  );
}

export default ModelList;
