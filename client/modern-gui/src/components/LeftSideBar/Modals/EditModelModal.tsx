import { JSX, useEffect, useState, ChangeEvent } from 'react';
import GenericModal from '../../Modals/GenericModal';
import { CSS_CLASSES } from '../../../styles/constants';
import { RVCModelSlot } from '@dannadori/voice-changer-client-js';
import { useAppState } from '../../../context/AppContext';
import { useUIContext } from '../../../context/UIContext';
import MD3Select from '../../Helpers/MD3Select';

type EditFormState = {
  modelName: string;
  thumbnailFile: File | null;
  embedder: string;
};

interface EditModelModalProps {
  model: RVCModelSlot;
  showModal: boolean;
  setShowEdit: (show: boolean) => void;
  modelDir?: string;
  icon?: string;
}

function EditModelModal({ model, showModal, setShowEdit, modelDir, icon }: EditModelModalProps): JSX.Element {
  // ---------------- State ----------------
  const appState = useAppState();
  const guiState = useUIContext();

  const [form, setForm] = useState<EditFormState>({
    modelName: model.name || '',
    thumbnailFile: null,
    embedder: (model as any).embedder || (Object.values(appState.serverSetting.serverSetting.embedders || {})[0]?.name || '')
  });

  const [thumbnailPreview, setThumbnailPreview] = useState<string | null>(null);
  const [isThumbnailExpanded, setIsThumbnailExpanded] = useState(false);
  const [previewMode, setPreviewMode] = useState<'settings' | 'list'>('settings');

  // Auto-expand thumbnail on select
  useEffect(() => {
    if (thumbnailPreview) setIsThumbnailExpanded(true);
  }, [thumbnailPreview]);

  // Prefill thumbnail preview from existing icon if available
  useEffect(() => {
    if (!showModal) return;
    if (form.thumbnailFile) return; // user selected new file
    let currentIcon = icon || '';
    if (!currentIcon && modelDir && model.iconFile && model.iconFile.length > 0) {
      const last = model.iconFile.split(/[\\/\\]/).pop() as string;
      currentIcon = `/${modelDir}/${model.slotIndex}/${last}`;
    }
    setThumbnailPreview(currentIcon || null);
  }, [showModal, modelDir, icon, model.iconFile, model.slotIndex]);

  // ---------------- Handlers ----------------
  const handleCancel = () => {
    if (!appState.serverSetting.isUploading) {
      setShowEdit(false);
      setForm({ modelName: model.name || '', thumbnailFile: null, embedder: (model as any).embedder || 'hubert_base' });
      setThumbnailPreview(null);
    }
  };

  const handleThumbnailFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      const file = event.target.files[0];
      setForm({ ...form, thumbnailFile: file });
      const reader = new FileReader();
      reader.onloadend = () => {
        setThumbnailPreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    } else {
      setForm({ ...form, thumbnailFile: null });
      setThumbnailPreview(null);
    }
  };

  const handleSave = async () => {
    const trimmedName = form.modelName.trim();
    if (!trimmedName) {
      guiState.showError('Please enter a model name.', 'Error');
      return;
    }
    try {
      if (form.thumbnailFile) {
        const thumb = form.thumbnailFile;
        const dotPos = thumb.name.lastIndexOf('.');
        const extOnly = dotPos >= 0 ? thumb.name.substring(dotPos + 1) : '';
        const thumbName = extOnly ? `thumbnail.${extOnly}` : 'thumbnail';
        const renamedThumb = new File([thumb], thumbName, { type: thumb.type, lastModified: thumb.lastModified });
        await appState.serverSetting.uploadAssets(model.slotIndex, 'iconFile', renamedThumb);
      }
    } catch (e) {
      console.warn('Thumbnail upload failed (continuing):', e);
    }

    try {
      await appState.serverSetting.updateModelInfo(model.slotIndex, 'name', trimmedName);
    } catch (e) {
      console.warn('Name update failed (continuing):', e);
    }
    try {
      await appState.serverSetting.updateModelInfo(model.slotIndex, 'embedder', form.embedder);
    } catch (e) {
      console.warn('Embedder update failed (continuing):', e);
    }

    await appState.serverSetting.reloadServerInfo();
    guiState.showError('Model updated successfully.', 'Confirm');
    setShowEdit(false);
  };

  // ---------------- Render ----------------

  const embedders = appState.serverSetting.serverSetting.embedders || {};
  const downloadedEmbedders = Object.entries(embedders).filter(([_, embedder]) => embedder.downloaded === true);
  const embedderOptions =
    downloadedEmbedders.length === 0
      ? [{ value: '', label: 'No downloaded embedders available' }]
      : downloadedEmbedders.map(([key, embedder]) => ({
          value: key,
          label: embedder.name
        }));

  return (
    <GenericModal
      isOpen={showModal}
      onClose={handleCancel}
      title={`Edit Model - Slot ${model.slotIndex}`}
      closeOnOutsideClick={false}
      primaryButton={{
        text: `${
          appState.serverSetting.isUploading
            ? `Saving... (${appState.serverSetting.uploadProgress.toFixed(1)}%)`
            : 'Save'
        }`,
        onClick: handleSave,
        className:
          'bg-primary text-on-primary hover:shadow-elevation-1 rounded-full px-6 py-2.5 font-semibold text-xs active:scale-97 transition-all',
        disabled: appState.serverSetting.isUploading
      }}
      secondaryButton={{
        text: 'Cancel',
        onClick: handleCancel,
        className:
          'border border-outline text-primary hover:bg-primary/8 rounded-full px-6 py-2.5 font-semibold text-xs active:scale-97 transition-all',
        disabled: appState.serverSetting.isUploading
      }}
    >
      <div className="space-y-4 pt-1 max-h-[70vh] overflow-y-auto pr-1.5">
        {/* Model name and embedder */}
        <div className="space-y-4 pl-3 border-l-2 border-outline-variant">
          <div className="space-y-1">
            <label htmlFor="editModelName" className={CSS_CLASSES.label}>
              Model Name:
            </label>
            <input
              type="text"
              id="editModelName"
              value={form.modelName}
              onChange={(e) => setForm({ ...form, modelName: e.target.value })}
              className={CSS_CLASSES.input}
              placeholder="Enter model name"
              disabled={appState.serverSetting.isUploading}
            />
          </div>

          <div>
            <MD3Select
              id="editEmbedder"
              label="Embedder"
              value={form.embedder}
              onChange={(e) => setForm({ ...form, embedder: e.target.value })}
              options={embedderOptions}
              disabled={appState.serverSetting.isUploading}
            />
          </div>
        </div>

        {/* Thumbnail file */}
        <div>
          <label htmlFor="editThumbnailFile" className={CSS_CLASSES.label}>
            Thumbnail Image (Optional):
          </label>
          <input
            type="file"
            id="editThumbnailFile"
            accept="image/*"
            onChange={handleThumbnailFileChange}
            className={CSS_CLASSES.fileInput}
            disabled={appState.serverSetting.isUploading}
          />
        </div>

        {/* Thumbnail preview */}
        {thumbnailPreview && (
          <div className="space-y-3 pt-1">
            <button
              type="button"
              onClick={() => setIsThumbnailExpanded(!isThumbnailExpanded)}
              className="flex items-center justify-between w-full text-xs font-semibold text-on-surface hover:text-primary transition-colors disabled:opacity-50"
              disabled={appState.serverSetting.isUploading}
            >
              <span>Preview Thumbnail</span>
              <svg
                className={`ml-2 h-4 w-4 transition-transform duration-200 ${
                  isThumbnailExpanded ? 'rotate-180' : ''
                }`}
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>
            {isThumbnailExpanded && (
              <div className="space-y-4 p-4 bg-surface-container-low rounded-lg border border-outline-variant animate-fadeIn">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-on-surface-variant uppercase tracking-wider">
                    Preview Shape:
                  </span>
                  <div className="flex space-x-2">
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        setPreviewMode('settings');
                      }}
                      className={`px-3 py-1 text-[10px] font-semibold rounded-full transition-all ${
                        previewMode === 'settings'
                          ? 'bg-secondary-container text-on-secondary-container shadow-elevation-1'
                          : 'text-on-surface-variant hover:bg-surface-variant/20'
                      }`}
                      disabled={appState.serverSetting.isUploading}
                    >
                      Circular
                    </button>
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        setPreviewMode('list');
                      }}
                      className={`px-3 py-1 text-[10px] font-semibold rounded-full transition-all ${
                        previewMode === 'list'
                          ? 'bg-secondary-container text-on-secondary-container shadow-elevation-1'
                          : 'text-on-surface-variant hover:bg-surface-variant/20'
                      }`}
                      disabled={appState.serverSetting.isUploading}
                    >
                      Rounded
                    </button>
                  </div>
                </div>
                <div className="flex items-center justify-center p-2">
                  <div
                    className={`transition-all duration-200 ${
                      previewMode === 'settings'
                        ? 'w-24 h-24 rounded-full p-1 border border-outline-variant'
                        : 'w-24 h-24 rounded-xl p-1 border border-outline-variant'
                    } bg-surface-container-high overflow-hidden shadow-inner`}
                  >
                    <img
                      src={thumbnailPreview}
                      alt="Thumbnail preview"
                      className={`w-full h-full object-cover ${
                        previewMode === 'settings' ? 'rounded-full' : 'rounded-lg'
                      }`}
                    />
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </GenericModal>
  );
}

export default EditModelModal;
