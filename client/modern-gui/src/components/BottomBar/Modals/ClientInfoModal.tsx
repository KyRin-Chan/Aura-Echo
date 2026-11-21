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
        <div className="bg-surface-container-low border border-outline-variant rounded-xl p-4">
          <div className="flex items-start justify-between gap-3">
            <div className="flex-1">
              <div className="flex items-center mb-3">
                <h4 className="text-base font-semibold text-on-surface mr-2">{t('clientLabel')}</h4>
                <span className="px-2.5 py-0.5 text-xs font-medium bg-primary-container text-on-primary-container rounded-full">
                  Aura-Echo
                </span>
              </div>

              <div className="grid grid-cols-2 gap-3 text-sm">
                <div className="bg-surface-container border border-outline-variant/40 rounded-lg p-2.5">
                  <div className="text-xs font-medium text-on-surface-variant mb-1">{t('editionLabel')}</div>
                  <div className="text-on-surface font-mono">
                    {appGuiSettingState.appGuiSetting.edition || 'N/A'}
                  </div>
                </div>
                <div className="bg-surface-container border border-outline-variant/40 rounded-lg p-2.5">
                  <div className="text-xs font-medium text-on-surface-variant mb-1">{t('versionLabel')}</div>
                  <div className="text-on-surface font-mono">
                    {appGuiSettingState.appGuiSetting.version || 'N/A'}
                  </div>
                </div>
              </div>
            </div>

            <a
              href="https://github.com/KyRin-Chan/Aura-Echo"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center px-3 py-2 text-xs text-primary hover:bg-primary/10 bg-surface-container border border-outline-variant/40 rounded-lg transition-all duration-200"
            >
              <FontAwesomeIcon icon={faGithub} className="mr-1.5" size="sm" />
              <span className="font-medium">GitHub</span>
              <FontAwesomeIcon icon={faExternalLinkAlt} className="ml-1" size="xs" />
            </a>
          </div>
        </div>

        {/* Audio Diagnostics Section */}
        <div className="border border-outline-variant rounded-xl p-4 bg-surface-container-low">
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
                let badgeColor = "bg-tertiary-container text-on-tertiary-container";
                if (log.type === "clipping") badgeColor = "bg-error-container text-on-error-container";
                if (log.type === "underflow") badgeColor = "bg-secondary-container text-on-secondary-container";

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

        <div className="bg-surface-container-low border border-outline-variant p-3 rounded-xl">
          <pre className="text-xs text-on-surface overflow-auto max-h-[40vh] custom-scrollbar p-3 rounded-lg bg-surface-container">
            <code>{clientJson}</code>
          </pre>
        </div>
      </div>
    </GenericModal>
  );
}

export default ClientInfoModal;