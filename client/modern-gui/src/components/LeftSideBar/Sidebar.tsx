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
    <div
      className="w-72 p-4 space-y-4 flex flex-col z-20 transition-colors duration-300"
      style={{
        backgroundColor: 'var(--bg-secondary)',
        color: 'var(--text-primary)',
        borderRight: '1px solid var(--border-primary)',
      }}
    >
      {/* Header: Logo */}
      <div className="flex items-center justify-center relative">
        <img
          src={'logo.png'}
          alt="Logo"
          className="h-10"
        />
      </div>

      <hr style={{ borderColor: 'var(--border-primary)' }} className="my-3" />

      <h3 className="text-xl font-semibold" style={{ color: 'var(--text-primary)' }}>Model Selector</h3>

      <UploadModelModal
        appState={appState}
        guiState={guiState}
        showUpload={showUpload}
        setShowUpload={setShowUpload}
      />

      {/* Model count display and upload button */}
      <div className="flex justify-between items-center mb-2">
        <span className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>Available Models ({filteredAndSortedModels.length})</span>
        <button
          onClick={() => setShowUpload(true)}
          className="p-1 hover:opacity-80 transition-opacity"
          style={{ color: 'var(--macaron-mint)' }}
          title="Upload New Model"
        >
          <FontAwesomeIcon icon={faPlus} size="lg" />
        </button>
      </div>

      <ModelFilter
        appState={appState}
        setFilteredAndSortedModels={setFilteredAndSortedModels}
      />

      <hr style={{ borderColor: 'var(--border-primary)' }} className="my-2" />

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