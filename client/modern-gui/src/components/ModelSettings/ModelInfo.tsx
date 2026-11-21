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
            className="flex flex-col items-center text-center mb-6 p-4 rounded-xl bg-surface-container border border-outline-variant/40"
          >
            <div className="flex w-full items-start">
              <img
                src={icon}
                alt={model.name}
                className="w-28 h-28 rounded-full object-cover shadow-elevation-2 mb-3 mr-4 flex-shrink-0 border-4 border-outline-variant"
              />
              <div className="flex-grow text-left">
                <div className="flex items-center mb-1">
                  <h3 className="text-xl font-bold break-words mr-2 text-on-surface">{model.name}</h3>
                  <button
                    onClick={() => setShowEdit(true)}
                    className="p-1.5 rounded-full focus:outline-none focus:ring-2 focus:ring-primary text-primary hover:bg-primary/10 transition-all"
                    title={t('editModelTooltip')}
                  >
                    <FontAwesomeIcon icon={faPen} className="h-4 w-4" />
                  </button>
                </div>
                <div className="space-y-1">
                  <div className="text-xs text-on-surface-variant">
                    <span className="font-semibold">{t('embedderLabel')}:</span> {model.embedder === 'hubert_base' ? 'ContentVec / Hubert' : (model.embedder === 'spin_base' ? 'SPIN' : (model.embedder === 'spin_v2' ? 'SPIN V2' : (model.embedder || 'N/A')))}
                  </div>
                  <div className="text-xs text-on-surface-variant">
                    <span className="font-semibold">{t('modelTypeLabel')}:</span> {modelTypeDisplay || 'N/A'}
                  </div>
                  <div className="text-xs text-on-surface-variant">
                    <span className="font-semibold">{t('sampleRateLabel')}:</span> {model.samplingRate ? `${model.samplingRate / 1000} kHz` : 'N/A'}
                  </div>
                  <div className="text-xs text-on-surface-variant">
                    <span className="font-semibold">{t('vocoderLabel')}:</span>{' '}
                    {model.vocoder === 'refinegan' ? 'RefineGAN' : 'NSF-HiFiGAN'}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </>
      ) : (
        <div
          className="flex items-center justify-center mb-6 p-8 rounded-xl min-h-[160px] bg-surface-container border border-outline-variant/40"
        >
          <p className="italic text-center text-on-surface-variant">{t('selectModelToSeeSettings')}</p>
        </div>
      )}
    </>
  )
}

export default ModelInfo;
