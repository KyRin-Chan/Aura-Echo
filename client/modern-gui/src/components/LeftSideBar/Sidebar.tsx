import { JSX, useState } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faPlus } from '@fortawesome/free-solid-svg-icons';
import { RVCModelSlot } from '@dannadori/voice-changer-client-js';
import { useAppState } from '../../context/AppContext';
import { useUIContext } from '../../context/UIContext';
import ModelList from './ModelList';
import ModelFilter from './ModelFilter';
import UploadModelModal from './Modals/UploadModelModal';

function LeftSidebar(): JSX.Element | null {
  // ---------------- State ----------------
  const appState = useAppState();
  const guiState = useUIContext();

  const [showUpload, setShowUpload] = useState<boolean>(false);
  const [filteredAndSortedModels, setFilteredAndSortedModels] = useState<RVCModelSlot[]>([]);

  // Currently active model slot index from server state
  const confirmedSelectedSlotIndex = appState.serverSetting?.serverSetting?.modelSlotIndex ?? null;

  // ---------------- Handlers ----------------

  // Handles model selection and server state update
  const handleSelectModel = async (slot: RVCModelSlot) => {
    guiState.startLoading(`Swapping to model: ${slot.name}`);
    await appState.serverSetting.updateServerSettings({ ...appState.serverSetting.serverSetting, modelSlotIndex: slot.slotIndex });
    guiState.stopLoading();
  };

  // ---------------- Render ----------------

  return (
    <div className="w-72 p-4 space-y-4 flex flex-col z-20 bg-surface-container text-on-surface border-r border-outline-variant transition-colors duration-300">
      {/* Header: Text Logo */}
      <div className="flex items-center justify-center py-2">
        <span className="text-2xl font-black tracking-wider bg-gradient-to-r from-primary to-tertiary bg-clip-text text-transparent select-none">
          AuraEcho
        </span>
      </div>

      <hr className="my-3 border-outline-variant" />

      <h3 className="text-xl font-semibold text-on-surface">Model Selector</h3>

      <UploadModelModal
        appState={appState}
        guiState={guiState}
        showUpload={showUpload}
        setShowUpload={setShowUpload}
      />

      {/* Model count display and upload button */}
      <div className="flex justify-between items-center mb-2">
        <span className="text-sm font-medium text-on-surface-variant">Available Models ({filteredAndSortedModels.length})</span>
        <button
          onClick={() => setShowUpload(true)}
          className="p-1 text-primary hover:opacity-80 transition-opacity"
          title="Upload New Model"
        >
          <FontAwesomeIcon icon={faPlus} size="lg" />
        </button>
      </div>

      <ModelFilter
        appState={appState}
        setFilteredAndSortedModels={setFilteredAndSortedModels}
      />

      <hr className="my-2 border-outline-variant" />

      <ModelList
        filteredAndSortedModels={filteredAndSortedModels}
        handleSelectModel={handleSelectModel}
        confirmedSelectedSlotIndex={confirmedSelectedSlotIndex}
        appState={appState}
      />
    </div>
  );
}

export default LeftSidebar;