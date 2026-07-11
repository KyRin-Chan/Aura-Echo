import { useState, ChangeEvent, useEffect } from 'react';
import {
  ClientState,
  ModelFileKind,
  ModelUploadSetting,
  RVCModelSlot
} from '@dannadori/voice-changer-client-js';
import { CSS_CLASSES } from '../../../styles/constants';
import GenericModal from '../../Modals/GenericModal';
import { UIContextType } from '../../../context/UIContext';
import { t } from '../../../locales';
import MD3Select from '../../Helpers/MD3Select';
import MD3Switch from '../../Helpers/MD3Switch';

export interface UploadFinalForm {
  modelName: string;
  thumbnailFile: File | null;
  voiceChangerType: string;
  slot: number;
  files: { kind: ModelFileKind; file: File; dir: string }[];
  params: any;
  embedder: string;
}

interface UploadModelModalProps {
  appState: ClientState;
  guiState: UIContextType;
  showUpload: boolean;
  setShowUpload: (showUpload: boolean) => void;
}

function UploadModelModal({ appState, guiState, showUpload, setShowUpload }: UploadModelModalProps) {
  // ---------------- Component State ----------------
  const [uploadSettings, setUploadSettings] = useState<UploadFinalForm>({
    modelName: '',
    thumbnailFile: null,
    voiceChangerType: 'RVC',
    slot: 0,
    files: [],
    params: {},
    embedder: (Object.values(appState.serverSetting?.serverSetting?.embedders || {})[0] as any)?.name || ''
  });
  const [autoSelectModel, setAutoSelectModel] = useState<boolean>(false);

  const [thumbnailPreview, setThumbnailPreview] = useState<string | null>(null);
  const [isThumbnailExpanded, setIsThumbnailExpanded] = useState(false);
  const [previewMode, setPreviewMode] = useState<'settings' | 'list'>('settings');

  // ---------------- Side Effects ----------------

  // Auto-populate model name from selected file and handle thumbnail preview
  useEffect(() => {
    const model = uploadSettings.files.find((x) => x.kind === 'rvcModel');
    if (model) {
      const baseName = model.file.name.substring(0, model.file.name.lastIndexOf('.'));
      setUploadSettings({ ...uploadSettings, modelName: baseName });
    }
    if (thumbnailPreview) {
      setIsThumbnailExpanded(true);
    }
  }, [uploadSettings.files, thumbnailPreview]);

  // ---------------- File Upload Handlers ----------------

  const handleModelFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      const file = event.target.files[0];
      const isZip = file.name.toLowerCase().endsWith('.zip');

      const newFile = { kind: 'rvcModel' as ModelFileKind, file: file, dir: '' };

      const updatedFiles = uploadSettings.files.filter(
        (f) => f.kind !== 'rvcModel' && (!isZip || f.kind !== 'rvcIndex')
      );
      updatedFiles.push(newFile);

      setUploadSettings({
        ...uploadSettings,
        files: updatedFiles,
        modelName: file.name.replace(/\.[^/.]+$/, '')
      });
    } else {
      const updatedFiles = uploadSettings.files.filter((f) => f.kind !== 'rvcModel');

      setUploadSettings({
        ...uploadSettings,
        files: updatedFiles,
        modelName: ''
      });
    }
  };

  const handleIndexFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      const newFile = { kind: 'rvcIndex' as ModelFileKind, file: event.target.files[0], dir: '' };

      const updatedFiles = uploadSettings.files.filter((f) => f.kind !== 'rvcIndex');
      updatedFiles.push(newFile);

      setUploadSettings({
        ...uploadSettings,
        files: updatedFiles
      });
    } else {
      const updatedFiles = uploadSettings.files.filter((f) => f.kind !== 'rvcIndex');

      setUploadSettings({
        ...uploadSettings,
        files: updatedFiles
      });
    }
  };

  const handleThumbnailFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      const file = event.target.files[0];
      setUploadSettings({ ...uploadSettings, thumbnailFile: file });
      const reader = new FileReader();
      reader.onloadend = () => {
        setThumbnailPreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    } else {
      setUploadSettings({ ...uploadSettings, thumbnailFile: null });
      setThumbnailPreview(null);
    }
  };

  // ---------------- Modal Actions ----------------

  const handleUploadCloseModal = () => {
    if (!appState.serverSetting.isUploading) {
      setShowUpload(false);
      setUploadSettings({
        modelName: '',
        thumbnailFile: null,
        voiceChangerType: 'RVC',
        slot: 0,
        files: [],
        params: {},
        embedder: (Object.values(appState.serverSetting?.serverSetting?.embedders || {})[0] as any)?.name || ''
      });
      setAutoSelectModel(false);
      setThumbnailPreview(null);
      setIsThumbnailExpanded(false);
    }
  };

  const handleUploadModal = async () => {
    if (!uploadSettings.files) {
      guiState.showError(t('modelUploadSelectFile'), t('errorTitle'));
      return;
    }
    const trimmedModelName = uploadSettings.modelName.trim();
    if (!trimmedModelName) {
      guiState.showError(t('modelUploadEnterName'), t('errorTitle'));
      return;
    }

    try {
      let emptySlotIndex = -1;
      const currentModelSlots = appState.serverSetting.serverSetting.modelSlots;

      if (currentModelSlots && currentModelSlots.length > 0) {
        emptySlotIndex = currentModelSlots.findIndex(
          (slot: RVCModelSlot) => !slot.name || slot.name.length === 0
        );
      }

      if (emptySlotIndex === -1) {
        guiState.showError(t('modelUploadNoSlot'), t('errorTitle'));
        return;
      }

      const sanitizeBaseName = (name: string) => name.replace(/[\\/:*?"<>|]+/g, '_').trim();
      const baseName = sanitizeBaseName(trimmedModelName);
      const renameWithExt = (f: File, bn: string) => {
        const dotPos = f.name.lastIndexOf('.');
        const ext = dotPos >= 0 ? f.name.substring(dotPos) : '';
        const newName = `${bn}${ext}`;
        return new File([f], newName, { type: f.type, lastModified: f.lastModified });
      };

      const modelEntry = uploadSettings.files.find((f) => f.kind === 'rvcModel');
      const indexEntry = uploadSettings.files.find((f) => f.kind === 'rvcIndex');

      const filesForUpload: { kind: ModelFileKind; file: File; dir: string }[] = [];
      if (modelEntry) {
        filesForUpload.push({
          kind: 'rvcModel' as ModelFileKind,
          file: renameWithExt(modelEntry.file, baseName),
          dir: ''
        });
      }
      if (indexEntry) {
        filesForUpload.push({
          kind: 'rvcIndex' as ModelFileKind,
          file: renameWithExt(indexEntry.file, 'added_' + baseName),
          dir: ''
        });
      }

      const uploadSettingsData: ModelUploadSetting = {
        voiceChangerType: 'RVC',
        slot: emptySlotIndex,
        files: filesForUpload,
        params: {},
        embedder: uploadSettings.embedder
      };

      console.log('Uploading model with settings:', uploadSettingsData);
      const serverInfo = await appState.serverSetting.uploadModel(uploadSettingsData);

      const uploadedModel = serverInfo?.modelSlots?.[emptySlotIndex];
      const hasModelFile = uploadedModel && 'modelFile' in uploadedModel && uploadedModel.modelFile;

      if (hasModelFile) {
        console.log('Model uploaded successfully.');

        if (uploadSettings.thumbnailFile) {
          console.log(`Uploading icon to slot ${emptySlotIndex}...`);
          const thumb = uploadSettings.thumbnailFile;
          const dotPos = thumb.name.lastIndexOf('.');
          const extOnly = dotPos >= 0 ? thumb.name.substring(dotPos + 1) : '';
          const thumbName = extOnly ? `thumbnail.${extOnly}` : 'thumbnail';
          const renamedThumb = new File([thumb], thumbName, { type: thumb.type, lastModified: thumb.lastModified });
          await appState.serverSetting.uploadAssets(emptySlotIndex, 'iconFile', renamedThumb);
          console.log('Icon uploaded.');
        }

        guiState.showError(t('modelUploadSuccess'), t('confirmTitle'));

        if (autoSelectModel) {
          guiState.startLoading(t('loadingSwappingModel') + uploadSettings.modelName);
          await appState.serverSetting.updateServerSettings({
            ...appState.serverSetting.serverSetting,
            modelSlotIndex: emptySlotIndex
          });
          guiState.stopLoading();
        }
      } else {
        console.error('Model upload failed - no model file found in slot after upload');
        guiState.showError(t('modelUploadFailed'), t('errorTitle'));
        return;
      }

      handleUploadCloseModal();
    } catch (error) {
      console.error('Error uploading model:', error);
      guiState.showError(
        `Error uploading model: ${error instanceof Error ? error.message : String(error)}`,
        'Error'
      );
    }
  };

  // ---------------- Component Render ----------------

  const embedders = appState.serverSetting?.serverSetting?.embedders || {};
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
      isOpen={showUpload}
      onClose={handleUploadCloseModal}
      title="Upload Model"
      closeOnOutsideClick={false}
      primaryButton={{
        text: `${
          appState.serverSetting.isUploading
            ? `Uploading... (${appState.serverSetting.uploadProgress.toFixed(1)}%)`
            : 'Upload'
        }`,
        onClick: handleUploadModal,
        className:
          'bg-primary text-on-primary hover:shadow-elevation-1 rounded-full px-6 py-2.5 font-semibold text-xs active:scale-97 transition-all',
        disabled: appState.serverSetting.isUploading
      }}
      secondaryButton={{
        text: 'Cancel',
        onClick: handleUploadCloseModal,
        className:
          'border border-outline text-primary hover:bg-primary/8 rounded-full px-6 py-2.5 font-semibold text-xs active:scale-97 transition-all',
        disabled: appState.serverSetting.isUploading
      }}
    >
      <div className="space-y-4 pt-1 max-h-[70vh] overflow-y-auto pr-1.5">
        {/* Main model file input */}
        <div>
          <label htmlFor="modelFile" className={CSS_CLASSES.label}>
            Model File (.pth, .safetensors, .onnx, .zip):
          </label>
          <input
            type="file"
            id="modelFile"
            accept=".pth,.safetensors,.onnx,.zip"
            onChange={handleModelFileChange}
            className={CSS_CLASSES.fileInput}
            disabled={appState.serverSetting.isUploading}
          />
        </div>

        {/* Model configuration options */}
        {uploadSettings.files.find((x) => x.kind === 'rvcModel') && (
          <div className="space-y-4 pl-3 border-l-2 border-outline-variant animate-fadeIn">
            {/* Model name input */}
            <div className="space-y-1">
              <label htmlFor="modelName" className={CSS_CLASSES.label}>
                Model Name:
              </label>
              <input
                type="text"
                id="modelName"
                value={uploadSettings.modelName}
                onChange={(e) => setUploadSettings({ ...uploadSettings, modelName: e.target.value })}
                className={CSS_CLASSES.input}
                placeholder="Enter model name"
                disabled={appState.serverSetting.isUploading}
              />
            </div>

            {/* Embedder selection */}
            <div>
              <MD3Select
                id="embedderType"
                label="Embedder Type"
                value={uploadSettings.embedder}
                onChange={(e) => setUploadSettings({ ...uploadSettings, embedder: e.target.value })}
                options={embedderOptions}
                disabled={appState.serverSetting.isUploading}
              />
            </div>
          </div>
        )}

        {/* Optional index file */}
        <div>
          <label htmlFor="indexFile" className={CSS_CLASSES.label}>
            Index File (.index) (Optional):
          </label>
          <input
            type="file"
            id="indexFile"
            accept=".index"
            onChange={handleIndexFileChange}
            className={CSS_CLASSES.fileInput}
            disabled={appState.serverSetting.isUploading}
          />
        </div>

        {/* Optional thumbnail image */}
        <div>
          <label htmlFor="thumbnailFile" className={CSS_CLASSES.label}>
            Thumbnail Image (Optional):
          </label>
          <input
            type="file"
            id="thumbnailFile"
            accept="image/*"
            onChange={handleThumbnailFileChange}
            className={CSS_CLASSES.fileInput}
            disabled={appState.serverSetting.isUploading}
          />
        </div>

        {/* Thumbnail preview section */}
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

        {/* Auto-select model after upload */}
        <div className="pt-2">
          <MD3Switch
            id="auto-select"
            checked={autoSelectModel}
            onChange={setAutoSelectModel}
            label="Select model after upload"
          />
        </div>
      </div>
    </GenericModal>
  );
}

export default UploadModelModal;
