import { JSX } from 'react';
import { useAppState } from '../../../context/AppContext';
import GenericModal from '../../Modals/GenericModal';
import { CSS_CLASSES } from '../../../styles/constants';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faExternalLinkAlt } from '@fortawesome/free-solid-svg-icons';
import { faGithub } from '@fortawesome/free-brands-svg-icons';
import { useAppRoot } from '../../../context/AppRootProvider';
import { t } from '../../../locales';

interface ClientInfoModalProps {
  showClientInfo: boolean;
  setShowClientInfo: (showClientInfo: boolean) => void;
}

function ClientInfoModal({ showClientInfo, setShowClientInfo }: ClientInfoModalProps): JSX.Element {
  // ---------------- States ----------------
  const appState = useAppState();
  const { appGuiSettingState } = useAppRoot();
  const clientJson = JSON.stringify(appState.setting, null, 2);

  // ---------------- Handlers ----------------

  // Handle close modal
  const handleClose = () => {
    setShowClientInfo(false);
  };

  // ---------------- Render ----------------

  return (
    <GenericModal
      isOpen={showClientInfo}
      onClose={handleClose}
      title={t('clientInfoTitle')}
      secondaryButton={{
        text: t('closeLabel'),
        onClick: handleClose,
        className: CSS_CLASSES.modalSecondaryButton,
      }}
    >
      <div className="space-y-4">
        <div className="bg-gradient-to-r from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <div className="flex items-center mb-2">
                <h4 className="text-base font-semibold text-blue-900 dark:text-blue-100 mr-2">{t('clientLabel')}</h4>
                <span className="px-2 py-1 text-xs font-medium bg-blue-200 dark:bg-blue-800 text-blue-800 dark:text-blue-200 rounded-full">
                  Aura-Echo
                </span>
              </div>

              <div className="grid grid-cols-2 gap-3 text-sm">
                <div className="bg-white/50 dark:bg-black/20 rounded-md p-2">
                  <div className="text-xs font-medium text-blue-700 dark:text-blue-300 mb-1">{t('editionLabel')}</div>
                  <div className="text-blue-900 dark:text-blue-100 font-mono">
                    {appGuiSettingState.appGuiSetting.edition || 'N/A'}
                  </div>
                </div>
                <div className="bg-white/50 dark:bg-black/20 rounded-md p-2">
                  <div className="text-xs font-medium text-blue-700 dark:text-blue-300 mb-1">{t('versionLabel')}</div>
                  <div className="text-blue-900 dark:text-blue-100 font-mono">
                    {appGuiSettingState.appGuiSetting.version || 'N/A'}
                  </div>
                </div>
              </div>
            </div>

            <a
              href="https://github.com/KyRin-Chan/Aura-Echo"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center px-3 py-2 text-xs text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-200 bg-white/50 dark:bg-black/20 hover:bg-white/80 dark:hover:bg-black/40 rounded-md transition-all duration-200"
            >
              <FontAwesomeIcon icon={faGithub} className="mr-1.5" size="sm" />
              <span className="font-medium">GitHub</span>
              <FontAwesomeIcon icon={faExternalLinkAlt} className="ml-1" size="xs" />
            </a>
          </div>
        </div>

        {/* Audio Diagnostics Section */}
        <div className="border border-outline-variant rounded-lg p-4 bg-surface-container-low">
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-sm font-semibold text-on-surface">{t('audioDiagnosticsTitle')}</h4>
            {appState.popLogs && appState.popLogs.length > 0 && (
              <button
                onClick={appState.clearPopLogs}
                className="px-2 py-1 text-xs font-semibold text-primary hover:text-primary/80 transition-colors"
              >
                {t('clearLogsLabel')}
              </button>
            )}
          </div>

          {(!appState.popLogs || appState.popLogs.length === 0) ? (
            <div className="text-xs text-on-surface-variant italic p-2 bg-surface-container-highest/20 rounded border border-dashed border-outline-variant text-center">
              {t('noAudioGlitchesDesc')}
            </div>
          ) : (
            <div className="space-y-1.5 max-h-[150px] overflow-y-auto custom-scrollbar text-xs">
              {appState.popLogs.slice().reverse().map((log, idx) => {
                let badgeColor = "bg-amber-100 text-amber-800 dark:bg-amber-950/40 dark:text-amber-300";
                if (log.type === "clipping") badgeColor = "bg-red-100 text-red-800 dark:bg-red-950/40 dark:text-red-300";
                if (log.type === "underflow") badgeColor = "bg-orange-100 text-orange-800 dark:bg-orange-950/40 dark:text-orange-300";

                return (
                  <div key={idx} className="flex items-center justify-between p-2 rounded bg-surface-container border border-outline-variant/40">
                    <div className="flex items-center space-x-2">
                      <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold uppercase ${badgeColor}`}>
                        {log.type}
                      </span>
                      <span className="text-on-surface-variant font-mono">
                        {log.type === "underflow" ? t('bufferDryDesc') : `Count: ${log.count}`}
                      </span>
                    </div>
                    <span className="text-on-surface-variant/70 text-[10px]">{log.timestamp}</span>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        <div className="bg-slate-100 dark:bg-slate-800 p-3 rounded-lg shadow-inner">
          <pre className="text-xs text-slate-700 dark:text-gray-300 overflow-auto max-h-[40vh] custom-scrollbar p-2 rounded-md bg-white dark:bg-slate-900">
            <code>{clientJson}</code>
          </pre>
        </div>
      </div>
    </GenericModal>
  );
}

export default ClientInfoModal;