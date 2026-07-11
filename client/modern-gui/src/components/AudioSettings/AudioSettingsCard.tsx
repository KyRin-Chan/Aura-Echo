import { JSX, useEffect, useState } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faChevronUp, faChevronDown } from '@fortawesome/free-solid-svg-icons';
import DragHandle from '../Helpers/DragHandle';
import { AUDIO_KEYS, CSS_CLASSES } from '../../styles/constants';
import { t } from '../../locales';
import AudioMode from './AudioMode';
import AudioDevicesServer from './AudioDevicesServer';
import AudioVolume from './AudioVolume';
import AudioDevicesClient from './AudioDevicesClient';
import { useAppState } from '../../context/AppContext';

interface AudioSettingsCardProps {
  dndAttributes?: Record<string, any>;
  dndListeners?: Record<string, any>;
}

function AudioSettingsCard({ dndAttributes, dndListeners }: AudioSettingsCardProps): JSX.Element {
  // ---------------- States ----------------
  const appState = useAppState();
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [audioState, setAudioState] = useState<"client" | "server">("client")

  // ---------------- Hooks ----------------

  // Set audio state based on server audio setting
  useEffect(() => {
    if (appState.serverSetting.serverSetting.enableServerAudio == 1) {
      setAudioState("server")
    } else {
      setAudioState("client")
    }
  }, [appState.serverSetting.serverSetting.enableServerAudio]);

  // ---------------- Render ----------------

  return (
    <div
      className={`${CSS_CLASSES.card} flex-1 min-h-0 flex flex-col ${isCollapsed ? 'h-auto' : 'overflow-y-auto'}`}
    >
      <div className={CSS_CLASSES.cardHeader}>
        <div className="flex items-center">
          <h4 className={CSS_CLASSES.heading}>{t('audioSettingsTitle')}</h4>
        </div>
        <div className="flex space-x-1 items-center">
          <button
            onClick={() => setIsCollapsed(!isCollapsed)}
            className={CSS_CLASSES.iconButton}
            title={isCollapsed ? "Expand" : "Collapse"}
          >
            <FontAwesomeIcon icon={isCollapsed ? faChevronDown : faChevronUp} className="h-5 w-5" />
          </button>
          <DragHandle attributes={dndAttributes} listeners={dndListeners} title="Drag" />
        </div>
      </div>
      {!isCollapsed && (
        <>
          <AudioMode audioState={audioState} setAudioState={setAudioState} />
          {audioState === "client" ? <AudioDevicesClient /> : <AudioDevicesServer />}
          <div className="mt-6">
            <AudioVolume />
          </div>
        </>
      )}
      <audio hidden id={AUDIO_KEYS.AUDIO_ELEMENT_FOR_PLAY_RESULT}></audio>
      <audio hidden id={AUDIO_KEYS.AUDIO_ELEMENT_FOR_PLAY_MONITOR}></audio>
    </div>
  );
}

export default AudioSettingsCard;