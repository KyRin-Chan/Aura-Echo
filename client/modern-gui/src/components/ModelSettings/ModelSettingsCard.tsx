import { JSX, useState, useEffect } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faChevronUp, faChevronDown } from '@fortawesome/free-solid-svg-icons';
import { RVCModelSlot, ClientState } from '@dannadori/voice-changer-client-js';
import { useAppState } from '../../context/AppContext';
import DragHandle from '../Helpers/DragHandle';
import { CSS_CLASSES } from '../../styles/constants';
import { useInitialPlaceholder } from '../../scripts/usePlaceholder';
import ModelInfo from './ModelInfo';
import ModelSettings from './ModelSettings';

interface ModelSettingsCardProps {
  dndAttributes?: Record<string, any>;
  dndListeners?: Record<string, any>;
}

function ModelSettingsCard({ dndAttributes, dndListeners }: ModelSettingsCardProps): JSX.Element {
  // ---------------- State ----------------
  const appState = useAppState() as ClientState;
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [model, setModel] = useState<RVCModelSlot>();

  const modelDir = appState.serverSetting.serverSetting.voiceChangerParams.model_dir;
  const icon = (model?.iconFile && model?.iconFile.length > 0) ? "/" + modelDir + "/" + model.slotIndex + "/" + model.iconFile.split(/[\/\\]/).pop() : "";
  const placeholder = useInitialPlaceholder(model?.name || "");

  // ---------------- Hooks ----------------

  // Update model state when modelSlotIndex or model_dir changes
  useEffect(() => {
    setModel(appState.serverSetting.serverSetting.modelSlots[appState.serverSetting.serverSetting.modelSlotIndex]);
  }, [appState.serverSetting?.serverSetting?.modelSlotIndex, appState.serverSetting?.serverSetting?.modelSlots]);

  // ---------------- Handler ----------------

  // Handle pitch change
  const handlePitchChange = (val: number) => {
    appState.serverSetting.updateServerSettings({
      ...appState.serverSetting.serverSetting,
      tran: val
    });
  };

  // Handle format shift change
  const handleFormatShiftChange = (val: number) => {
    appState.serverSetting.updateServerSettings({
      ...appState.serverSetting.serverSetting,
      formantShift: val
    });
  };

  // Handle index ratio change
  const handleIndexRatioChange = (val: number) => {
    appState.serverSetting.updateServerSettings({
      ...appState.serverSetting.serverSetting,
      indexRatio: val
    });
  };

  // Handle speaker change
  const handleSpeakerChange = (val: number) => {
    appState.serverSetting.updateServerSettings({
      ...appState.serverSetting.serverSetting,
      dstId: val
    });
  };

  // Handle save settings
  const handleSaveSettings = () => {
    appState.serverSetting.updateModelDefault();
  };

  // ---------------- Render ----------------

  return (
    <div
      className={`p-4 rounded-2xl shadow-sm transition-all duration-300 flex-1 min-h-0 flex flex-col ${isCollapsed ? 'h-auto' : ''}`}
      style={{
        backgroundColor: 'var(--bg-secondary)',
        border: '1px solid var(--border-primary)',
      }}
    >
      <div className="flex justify-between items-center mb-3 pb-2" style={{ borderBottom: '1px solid var(--border-primary)' }}>
        <h4 className={CSS_CLASSES.heading}>Model Settings</h4>
        <div className="flex space-x-1 items-center">
          <button onClick={() => setIsCollapsed(!isCollapsed)} className={CSS_CLASSES.iconButton} title={isCollapsed ? "Expand" : "Collapse"}>
            <FontAwesomeIcon icon={isCollapsed ? faChevronDown : faChevronUp} className="h-5 w-5" />
          </button>
          <DragHandle attributes={dndAttributes} listeners={dndListeners} title="Drag" />
        </div>
      </div>
      {!isCollapsed && (
        <>
          {model ? (
            <ModelInfo
              model={model}
              icon={icon || placeholder}
            />
          ) : (
            <div
              className="flex items-center justify-center mb-6 p-8 rounded-xl min-h-[160px]"
              style={{ backgroundColor: 'var(--bg-tertiary)' }}
            >
              <p style={{ color: 'var(--text-tertiary)' }} className="italic text-center">Select a model from the list <br /> to see its settings.</p>
            </div>
          )}
          {
            model && (
              <ModelSettings
                model={model}
                handlePitchChange={handlePitchChange}
                handleFormatShiftChange={handleFormatShiftChange}
                handleIndexRatioChange={handleIndexRatioChange}
                handleSpeakerChange={handleSpeakerChange}
                setModel={setModel}
              />
            )
          }
        </>
      )}
      <div className="flex justify-end mt-4">
        <button
          onClick={handleSaveSettings}
          className={CSS_CLASSES.modalSecondaryButton}
        >
          Save Settings
        </button>
      </div>
    </div>
  );
}

export default ModelSettingsCard;