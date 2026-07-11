import React, { JSX } from 'react';
import GenericModal from '../../Modals/GenericModal';
import { ModelUploadSetting, RVCModelSlot } from '@dannadori/voice-changer-client-js';
import { useAppState } from '../../../context/AppContext';
import { useUIContext } from '../../../context/UIContext';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faTrash } from '@fortawesome/free-solid-svg-icons';
import { useInitialPlaceholder } from '../../../scripts/usePlaceholder';
import { t } from '../../../locales';

interface DeleteModelModalProps {
  model: RVCModelSlot;
  showModal: boolean;
  setShowDelete: (show: boolean) => void;
  modelDir: string;
}

function DeleteModelModal({ model, showModal, setShowDelete, modelDir }: DeleteModelModalProps): JSX.Element {
  // ---------------- State ----------------
  const appState = useAppState();
  const guiState = useUIContext();

  const icon =
    model.iconFile.length > 0
      ? '/model_dir/' + model.slotIndex + '/' + model.iconFile.split(/[\/\\]/).pop()
      : '';
  const placeholder = useInitialPlaceholder(model.name);

  // ---------------- Handlers ----------------

  // Handle confirm button click
  const handleConfirm = async () => {
    const settings: ModelUploadSetting & { embedder: string } = {
      voiceChangerType: 'RVC',
      slot: model.slotIndex,
      files: [],
      params: {},
      embedder: 'hubert_base'
    };
    appState.serverSetting.deleteModel(model.slotIndex);

    if (appState.serverSetting.serverSetting.modelSlotIndex === model.slotIndex) {
      guiState.startLoading();
      await appState.serverSetting.updateServerSettings({
        ...appState.serverSetting.serverSetting,
        modelSlotIndex: 0
      });
      guiState.stopLoading();
    }

    guiState.showError(t('modelDeleteSuccess'), t('confirmTitle'));
    setShowDelete(false);
  };

  // Handle cancel button click
  const handleCancel = () => {
    setShowDelete(false);
  };

  // ---------------- Render ----------------

  return (
    <GenericModal
      isOpen={showModal}
      onClose={handleCancel}
      title={t('deleteModelTitle')}
      size="small"
      primaryButton={{
        text: t('btnDelete'),
        onClick: handleConfirm,
        className: 'bg-error text-on-error hover:shadow-elevation-1 rounded-full px-6 py-2.5 font-semibold text-xs active:scale-97 transition-all'
      }}
      secondaryButton={{
        text: t('cancelLabel'),
        onClick: handleCancel,
        className: 'border border-outline text-primary hover:bg-primary/8 rounded-full px-6 py-2.5 font-semibold text-xs active:scale-97 transition-all'
      }}
    >
      <div className="space-y-6 pt-1">
        <div className="flex items-start space-x-4">
          <div className="flex-shrink-0 w-12 h-12 bg-error/10 rounded-full flex items-center justify-center">
            <FontAwesomeIcon icon={faTrash} className="w-6 h-6 text-error" />
          </div>
          <div className="flex-1">
            <h3 className="text-base font-semibold text-on-surface mb-1.5">{t('permanentlyDeleteModelTitle')}</h3>
            <p className="text-xs text-on-surface-variant">
              {t('deleteModelConfirmDesc')}
            </p>
          </div>
        </div>

        <div className="bg-surface-container-low border border-outline-variant rounded-xl p-4">
          <div className="flex items-center space-x-3">
            <img
              src={icon.length > 0 ? icon : placeholder}
              alt={model.name}
              className="w-10 h-10 rounded-full object-cover flex-shrink-0 border border-outline-variant/30"
            />
            <div>
              <p className="text-sm font-semibold text-on-surface">{model.name}</p>
              <p className="text-[10px] text-on-surface-variant font-medium mt-0.5">
                Slot {model.slotIndex} •{' '}
                {model.embedder === 'hubert_base'
                  ? 'ContentVec / Hubert'
                  : model.embedder === 'spin_base'
                  ? 'SPIN'
                  : model.embedder === 'spin_v2'
                  ? 'SPIN V2'
                  : model.embedder || 'Unknown'}{' '}
                • {model.samplingRate || 'Unknown'} Hz • {model.voiceChangerType || 'RVC'}
                {model.version || '1'}
              </p>
            </div>
          </div>
        </div>
      </div>
    </GenericModal>
  );
}

export default DeleteModelModal;