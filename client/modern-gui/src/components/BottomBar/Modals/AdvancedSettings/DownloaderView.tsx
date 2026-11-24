import { JSX, useState, useEffect, useCallback } from 'react';
import { useAppState } from '../../../../context/AppContext';
import { useUIContext } from '../../../../context/UIContext';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faDownload, faTrash, faSpinner, faCheck, faTimes } from '@fortawesome/free-solid-svg-icons';
import { ModelInfoDict } from '@dannadori/voice-changer-client-js';
import { t } from '../../../../locales';
import { useTensorRT, formatBytes, getTensorRTPhaseText } from '../../../../scripts/useTensorRT';

interface DownloaderViewProps {
  onDownloadStateChange?: (isDownloading: boolean) => void;
}

const DownloaderView = (props: DownloaderViewProps) => {
  const appState = useAppState();
  const uiState = useUIContext();
  const { onDownloadStateChange } = props;

  const [loadingItems, setLoadingItems] = useState<Record<string, 'download' | 'delete' | null>>({});
  const [isAnyDownloading, setIsAnyDownloading] = useState(false);
  const [embedders, setEmbedders] = useState<ModelInfoDict>(appState.serverSetting.serverSetting.embedders || {});
  const [pitchExtractors, setPitchExtractors] = useState<ModelInfoDict>(
    appState.serverSetting.serverSetting.pitchExtractors || {}
  );
  const {
    status: trtStatus,
    isActive: isTrtActive,
    isSubmitting: isTrtSubmitting,
    isDeleting: isTrtDeleting,
    startDownload: startTrtDownload,
    cancelDownload: cancelTrtDownload,
    deleteRuntime: deleteTrtRuntime
  } = useTensorRT();

  // Update local state when server settings change
  useEffect(() => {
    const serverSetting = appState.serverSetting.serverSetting;
    setEmbedders(serverSetting.embedders || {});
    setPitchExtractors(serverSetting.pitchExtractors || {});
  }, [appState.serverSetting.serverSetting]);

  // Track download state changes and notify parent
  useEffect(() => {
    if (onDownloadStateChange) {
      onDownloadStateChange(isAnyDownloading);
    }
  }, [isAnyDownloading, onDownloadStateChange]);

  const updateDownloadingState = useCallback((downloading: boolean) => {
    setIsAnyDownloading(downloading);
  }, []);

  const handleModelAction = async (
    type: 'embedder' | 'pitchExtractor',
    action: 'download' | 'delete',
    id: string,
    info: ModelInfoDict[string]
  ) => {
    // Only prevent new downloads if another download is in progress
    if (action === 'download' && isAnyDownloading && loadingItems[id] !== 'download') return;
    // Prevent deletion of mandatory or in-use items
    if (
      action === 'delete' &&
      (info.mandatory ||
        (type === 'embedder' && isEmbedderInUse(id)) ||
        (type === 'pitchExtractor' && isPitchExtractorInUse(id)))
    ) {
      return;
    }

    try {
      if (action === 'download') {
        updateDownloadingState(true);
      }
      setLoadingItems((prev) => ({ ...prev, [id]: action }));

      // The id is already the model key from the dictionary
      if (action === 'download') {
        await appState.serverSetting.downloadPretrained(id);
      } else {
        await appState.serverSetting.deletePretrained(id);
      }

      // Refresh server info to update the installed status
      await appState.serverSetting.reloadServerInfo();

      // Show success message with model name
      const actionText = action === 'download' ? t('downloadedLabel') : t('deletedLabel');
      const modelType = type === 'embedder' ? t('embedderLabel') : t('pitchExtractorLabel');
      const successMsg = t('modelActionSuccess')
        .replace('{type}', modelType)
        .replace('{name}', info.name || id)
        .replace('{action}', actionText);
      uiState.showError(successMsg, 'Confirm');
    } catch (error) {
      console.error(`Error ${action}ing ${type}:`, error);
      const modelType = type === 'embedder' ? t('embedderLabel') : t('pitchExtractorLabel');
      const errorMsg = t('modelActionFailed')
        .replace('{action}', action)
        .replace('{type}', modelType)
        .replace('{error}', error instanceof Error ? error.message : String(error));
      uiState.showError(
        errorMsg,
        'Error'
      );
    } finally {
      setLoadingItems((prev) => {
        const newState = { ...prev, [id]: null };
        // Check if there are any downloads still in progress
        const anyDownloadsLeft = Object.values(newState).some((v) => v === 'download');
        if (!anyDownloadsLeft) {
          updateDownloadingState(false);
        }
        return newState;
      });
    }
  };

  // Check if an embedder is in use by any model
  const isEmbedderInUse = (embedderId: string): boolean => {
    try {
      const modelSlots = (appState.serverSetting.serverSetting as any).modelSlots || [];
      return modelSlots.some(
        (slot: any) => slot.embedder === embedderId || (slot.embFile && slot.embFile.includes(embedderId))
      );
    } catch (e) {
      console.error('Error checking embedder usage:', e);
      return false;
    }
  };

  // Check if a pitch extractor is currently in use
  const isPitchExtractorInUse = (pitchExtractorId: string): boolean => {
    try {
      const f0Detector = (appState.serverSetting.serverSetting as any).f0Detector;
      if (f0Detector === 'rmvpe_fcpe' && (pitchExtractorId === 'rmvpe' || pitchExtractorId === 'fcpe')) {
        return true;
      }
      return f0Detector === pitchExtractorId;
    } catch (e) {
      console.error('Error checking pitch extractor usage:', e);
      return false;
    }
  };

  const renderItem = (id: string, info: ModelInfoDict[string], type: 'embedder' | 'pitchExtractor') => {
    if (!info) return null;

    const isLoading = loadingItems[id];
    const isDownloading = isLoading === 'download';
    const isDeleting = isLoading === 'delete';
    const name = info.name || id;
    const isInUse =
      (type === 'embedder' && !info.mandatory && isEmbedderInUse(id)) ||
      (type === 'pitchExtractor' && !info.mandatory && isPitchExtractorInUse(id));

    return (
      <div
        key={id}
        className="flex items-center justify-between p-4 hover:bg-primary/4 transition-colors border-b border-outline-variant/30 last:border-0"
      >
        <div className="flex-1 min-w-0 pr-4">
          <span className="font-semibold text-on-surface text-sm break-words">{name}</span>
          {isInUse ? (
            <span className="ml-2.5 px-2.5 py-0.5 text-[10px] font-semibold bg-secondary-container text-on-secondary-container rounded-full">
              {t('inUseLabel')}
            </span>
          ) : info.mandatory ? (
            <span className="ml-2.5 px-2.5 py-0.5 text-[10px] font-semibold bg-tertiary-container text-on-tertiary-container rounded-full">
              {t('requiredLabel')}
            </span>
          ) : null}
        </div>
        <div className="flex items-center space-x-2.5">
          {info.downloaded ? (
            <>
                <span className="inline-flex items-center px-2.5 py-0.5 text-[10px] font-semibold text-on-primary-container bg-primary-container rounded-full">
                  <FontAwesomeIcon icon={faCheck} className="mr-1" />
                  {t('installedLabel')}
                </span>
              {!uiState.isConverting && !info.mandatory && !isInUse && (
                <button
                  onClick={() => handleModelAction(type, 'delete', id, info)}
                  disabled={isDeleting}
                  className="px-4 py-1.5 text-xs font-semibold rounded-full transition-all duration-150 flex items-center bg-error/10 text-error hover:bg-error/20 disabled:opacity-50"
                >
                  {isDeleting ? (
                    <FontAwesomeIcon icon={faSpinner} className="animate-spin mr-1" />
                  ) : (
                    <FontAwesomeIcon icon={faTrash} className="mr-1" />
                  )}
                  {t('deleteTooltip')}
                </button>
              )}
            </>
          ) : (
            <button
              onClick={() => handleModelAction(type, 'download', id, info)}
              disabled={isAnyDownloading && !isDownloading}
              className={`px-4 py-1.5 text-xs font-semibold rounded-full transition-all duration-150 flex items-center ${
                isAnyDownloading && !isDownloading
                  ? 'bg-surface-container-highest text-on-surface-variant/30 cursor-not-allowed'
                  : isDownloading
                  ? 'bg-primary text-on-primary cursor-wait'
                  : 'bg-primary text-on-primary hover:shadow-elevation-1 active:scale-97'
              }`}
            >
              {isDownloading ? (
                <>
                  <FontAwesomeIcon icon={faSpinner} className="animate-spin mr-1" />
                  {t('downloadingLabel')}
                </>
              ) : (
                <>
                  <FontAwesomeIcon icon={faDownload} className="mr-1" />
                  {t('downloadLabel')}
                </>
              )}
            </button>
          )}
        </div>
      </div>
    );
  };

  // Sort items with mandatory ones first, then by name
  const sortItems = (items: ModelInfoDict) => {
    return Object.entries(items).sort(([idA, a], [idB, b]) => {
      // Mandatory items first
      if (a.mandatory && !b.mandatory) return -1;
      if (!a.mandatory && b.mandatory) return 1;
      // Then sort by name
      return (a.name || idA).localeCompare(b.name || idB);
    });
  };

  const sortedEmbedders = sortItems(embedders);
  const sortedPitchExtractors = sortItems(pitchExtractors);
  const isTrtInUse = appState.serverSetting.serverSetting.useTensorRT === 1;

  return (
    <div className="space-y-6 max-h-[500px] overflow-y-auto pr-1.5">
      <div>
        <h3 className="text-sm font-bold uppercase tracking-wider text-primary mb-3 pl-1">
          {t('runtimeAcceleratorsTitle')}
        </h3>
        <div className="bg-surface-container-low rounded-lg border border-outline-variant overflow-hidden">
          <div className="p-4 space-y-2.5 hover:bg-primary/4 transition-colors">
            <div className="flex items-center justify-between">
              <div className="flex-1 min-w-0 pr-4">
                <div className="flex items-center flex-wrap gap-2">
                  <span className="font-semibold text-on-surface text-sm">{t('tensorRtRuntimeName')}</span>
                  {isTrtInUse && (
                    <span className="px-2.5 py-0.5 text-[10px] font-semibold bg-secondary-container text-on-secondary-container rounded-full">
                      {t('inUseLabel')}
                    </span>
                  )}
                  {trtStatus?.pendingDelete && (
                    <span className="px-2.5 py-0.5 text-[10px] font-semibold bg-error/10 text-error rounded-full">
                      {t('tensorRtPendingDeleteLabel')}
                    </span>
                  )}
                </div>
                {trtStatus && !trtStatus.installed && trtStatus.supported && !isTrtActive && !trtStatus.pendingDelete && (trtStatus.state === 'idle' || trtStatus.state === 'cancelled') && (
                  <p className="text-[11px] text-on-surface-variant/75 mt-1">{t('tensorRtNotInstalledHint')}</p>
                )}
                {trtStatus?.state === 'error' && trtStatus.error && (
                  <p className="text-[11px] text-error mt-1 break-words">{trtStatus.error}</p>
                )}
              </div>

              <div className="flex items-center space-x-2.5 shrink-0">
                {trtStatus?.pendingDelete ? null : isTrtActive ? (
                  <button
                    onClick={cancelTrtDownload}
                    disabled={isTrtSubmitting}
                    className="px-4 py-1.5 text-xs font-semibold rounded-full transition-all duration-150 flex items-center bg-surface-container-highest text-on-surface hover:bg-error/15 hover:text-error disabled:opacity-50"
                  >
                    <FontAwesomeIcon icon={faTimes} className="mr-1.5" />
                    {t('tensorRtCancelBtn')}
                  </button>
                ) : trtStatus?.installed ? (
                  <>
                    <span className="inline-flex items-center px-2.5 py-0.5 text-[10px] font-semibold text-on-primary-container bg-primary-container rounded-full">
                      <FontAwesomeIcon icon={faCheck} className="mr-1" />
                      {trtStatus.source === 'system' ? t('tensorRtSystemLabel') : t('installedLabel')}
                      {trtStatus.sizeBytes > 0 ? ` · ${formatBytes(trtStatus.sizeBytes)}` : ''}
                    </span>
                    {trtStatus.managed && !uiState.isConverting && !isTrtInUse && (
                      <button
                        onClick={deleteTrtRuntime}
                        disabled={isTrtDeleting}
                        className="px-4 py-1.5 text-xs font-semibold rounded-full transition-all duration-150 flex items-center bg-error/10 text-error hover:bg-error/20 disabled:opacity-50"
                      >
                        <FontAwesomeIcon icon={isTrtDeleting ? faSpinner : faTrash} className={isTrtDeleting ? 'animate-spin mr-1' : 'mr-1'} />
                        {t('deleteTooltip')}
                      </button>
                    )}
                  </>
                ) : trtStatus && !trtStatus.supported ? (
                  <span className="inline-flex items-center px-2.5 py-0.5 text-[10px] font-semibold bg-surface-container-highest text-on-surface-variant rounded-full">
                    {t('tensorRtUnsupportedLabel')}
                  </span>
                ) : trtStatus && trtStatus.supported ? (
                  <button
                    onClick={startTrtDownload}
                    disabled={isTrtSubmitting}
                    className="px-4 py-1.5 text-xs font-semibold rounded-full transition-all duration-150 flex items-center bg-primary text-on-primary hover:shadow-elevation-1 active:scale-97 disabled:opacity-50"
                  >
                    <FontAwesomeIcon icon={isTrtSubmitting ? faSpinner : faDownload} className={isTrtSubmitting ? 'animate-spin mr-1' : 'mr-1'} />
                    {trtStatus.state === 'error' ? t('tensorRtRetryBtn') : t('tensorRtDownloadBtn')}
                  </button>
                ) : null}
              </div>
            </div>

            {isTrtActive && trtStatus && (
              <div className="space-y-1.5 pt-1">
                <div
                  className="w-full h-1 rounded-full bg-surface-container-highest overflow-hidden"
                  role="progressbar"
                  aria-valuenow={Math.round(trtStatus.progress)}
                  aria-valuemin={0}
                  aria-valuemax={100}
                >
                  <div
                    className="h-full rounded-full bg-primary transition-all duration-300"
                    style={{ width: `${Math.min(100, Math.max(0, trtStatus.progress))}%` }}
                  />
                </div>
                <div className="flex items-center justify-between text-[11px] text-on-surface-variant">
                  <span>{getTensorRTPhaseText(trtStatus)}</span>
                  <span className="font-semibold text-primary tabular-nums">{Math.round(trtStatus.progress)}%</span>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      <div>
        <h3 className="text-sm font-bold uppercase tracking-wider text-primary mb-3 pl-1">{t('embeddersTitle')}</h3>
        <div className="space-y-3">
          {sortedEmbedders.length > 0 ? (
            <div className="bg-surface-container-low rounded-lg border border-outline-variant overflow-hidden">
              {sortedEmbedders.map(([id, info]) => renderItem(id, info, 'embedder'))}
            </div>
          ) : (
            <div className="text-center py-8 text-on-surface-variant/60 italic bg-surface-container-low rounded-lg border border-outline-variant text-sm">
              {t('noEmbeddersAvailable')}
            </div>
          )}
        </div>
      </div>

      <div>
        <h3 className="text-sm font-bold uppercase tracking-wider text-primary mb-3 pl-1">
          {t('pitchExtractorsTitle')}
        </h3>
        <div className="space-y-3">
          {sortedPitchExtractors.length > 0 ? (
            <div className="bg-surface-container-low rounded-lg border border-outline-variant overflow-hidden">
              {sortedPitchExtractors.map(([id, info]) => renderItem(id, info, 'pitchExtractor'))}
            </div>
          ) : (
            <div className="text-center py-8 text-on-surface-variant/60 italic bg-surface-container-low rounded-lg border border-outline-variant text-sm">
              {t('noPitchExtractorsAvailable')}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DownloaderView;
