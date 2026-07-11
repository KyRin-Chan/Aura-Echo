import { RVCModelSlot } from "@dannadori/voice-changer-client-js";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faPen } from "@fortawesome/free-solid-svg-icons";
import { useState } from "react";
import EditModelModal from "../LeftSideBar/Modals/EditModelModal";
import { t } from '../../locales';

interface ModelInfoProps {
  model: RVCModelSlot;
  icon: string;
}
function ModelInfo({ model, icon }: ModelInfoProps) {
  // ---------------- State ----------------
  const [showEdit, setShowEdit] = useState<boolean>(false);

  const isONNX = model?.isONNX ?? false;
  const modelTypeDisplay = isONNX
    ? model?.modelTypeOnnx || model?.modelType
    : model?.modelType;

  // ---------------- Render ----------------

  return (
    <>
      {model ? (
        <>
          <EditModelModal
            model={model}
            showModal={showEdit}
            setShowEdit={setShowEdit}
            icon={icon}
          />
          <div
            className="flex flex-col items-center text-center mb-6 p-4 rounded-xl"
            style={{ backgroundColor: 'var(--bg-tertiary)' }}
          >
            <div className="flex w-full items-start">
              <img
                src={icon}
                alt={model.name}
                className="w-28 h-28 rounded-full object-cover shadow-lg mb-3 mr-4 flex-shrink-0"
                style={{ border: '4px solid var(--border-primary)' }}
              />
              <div className="flex-grow text-left">
                <div className="flex items-center mb-1">
                  <h3 className="text-xl font-bold break-words mr-2" style={{ color: 'var(--text-primary)' }}>{model.name}</h3>
                  <button
                    onClick={() => setShowEdit(true)}
                    className="p-1 rounded-full focus:outline-none focus:ring-2 transition-opacity hover:opacity-80"
                    style={{ color: 'var(--macaron-blue)' }}
                    title={t('editModelTooltip')}
                  >
                    <FontAwesomeIcon icon={faPen} className="h-4 w-4" />
                  </button>
                </div>
                <div className="space-y-1">
                  <div className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                    <span className="font-semibold">{t('embedderLabel')}</span> {model.embedder === 'hubert_base' ? 'ContentVec / Hubert' : (model.embedder === 'spin_base' ? 'SPIN' : (model.embedder === 'spin_v2' ? 'SPIN V2' : (model.embedder || 'N/A')))}
                  </div>
                  <div className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                    <span className="font-semibold">{t('modelTypeLabel')}</span> {modelTypeDisplay || 'N/A'}
                  </div>
                  <div className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                    <span className="font-semibold">{t('sampleRateLabel')}</span> {model.samplingRate ? `${model.samplingRate / 1000} kHz` : 'N/A'}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </>
      ) : (
        <div
          className="flex items-center justify-center mb-6 p-8 rounded-xl min-h-[160px]"
          style={{ backgroundColor: 'var(--bg-tertiary)' }}
        >
          <p className="italic text-center" style={{ color: 'var(--text-tertiary)' }}>{t('selectModelToSeeSettings')}</p>
        </div>
      )}
    </>
  )
}

export default ModelInfo;
