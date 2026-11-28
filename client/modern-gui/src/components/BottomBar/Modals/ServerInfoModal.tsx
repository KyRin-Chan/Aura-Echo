import { JSX } from 'react';
import { useAppState } from '../../../context/AppContext';
import GenericModal from '../../Modals/GenericModal';
import { CSS_CLASSES } from '../../../styles/constants';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faExternalLinkAlt } from '@fortawesome/free-solid-svg-icons';
import { faGithub } from '@fortawesome/free-brands-svg-icons';
import { useAppRoot } from '../../../context/AppRootProvider';
import { t } from '../../../locales';

interface ServerInfoModalProps {
  showServerInfo: boolean;
  setShowServerInfo: (showServerInfo: boolean) => void;
}

function ServerInfoModal({ showServerInfo, setShowServerInfo }: ServerInfoModalProps): JSX.Element {
  // ---------------- States ----------------
  const appState = useAppState();
  const { appGuiSettingState } = useAppRoot();
  const serverJson = JSON.stringify(appState.serverSetting.serverSetting, null, 2);

  // ---------------- Handlers ----------------

  // Handle close modal
  const handleClose = () => {
    setShowServerInfo(false);
  };

  // ---------------- Render ----------------

  return (
    <GenericModal
      isOpen={showServerInfo}
      onClose={handleClose}
      title={t('serverInfoTitle')}
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
                <h4 className="text-base font-semibold text-on-surface mr-2">{t('serverLabel')}</h4>
                <span className="px-2.5 py-0.5 text-xs font-medium bg-primary-container text-on-primary-container rounded-full">
                  Aura-Echo
                </span>
              </div>

              <div className="grid grid-cols-2 gap-3 text-sm">
                <div className="bg-surface-container border border-outline-variant/40 rounded-lg p-2.5">
                  <div className="text-xs font-medium text-on-surface-variant mb-1">{t('editionLabel')}</div>
                  <div className="text-on-surface font-mono">
                    {appGuiSettingState.serverInfo.edition || 'N/A'}
                  </div>
                </div>
                <div className="bg-surface-container border border-outline-variant/40 rounded-lg p-2.5">
                  <div className="text-xs font-medium text-on-surface-variant mb-1">{t('versionLabel')}</div>
                  <div className="text-on-surface font-mono">
                    {appGuiSettingState.serverInfo.version || 'N/A'}
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
          <div className="mt-3 pt-3 border-t border-outline-variant/40 text-xs text-on-surface-variant leading-relaxed">
            {t('nonCommercialNotice')}
          </div>
        </div>

        <div className="bg-surface-container-low border border-outline-variant p-3 rounded-xl">
          <pre className="text-xs text-on-surface overflow-auto max-h-[40vh] custom-scrollbar p-3 rounded-lg bg-surface-container">
            <code>{serverJson}</code>
          </pre>
        </div>
      </div>
    </GenericModal>
  );
}

export default ServerInfoModal;