import { JSX, useState, useEffect, useCallback } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faChevronUp, faChevronDown } from '@fortawesome/free-solid-svg-icons';
import DragHandle from '../Helpers/DragHandle';
import { CSS_CLASSES } from '../../styles/constants';
import { AudioEffect, AudioChannel, AudioEffectsConfiguration, BackgroundSoundsUploadSetting, BackgroundTrack } from '@dannadori/voice-changer-client-js';
import { createEffectFromServerSchema } from './serverEffectsUtils';
import EffectsList, { AudioEffectWithIndex as CEffect } from './EffectsList';
import EffectConfig from './EffectConfig';
import { useAppState } from '../../context/AppContext';
import BackgroundConfig from './BackgroundConfig';
import BackgroundList from './BackgroundList';
import { useUIContext } from '../../context/UIContext';
import { t } from '../../locales';

// UI type with index for client-side management
type AudioEffectWithIndex = AudioEffect & { index: number };

interface AudioEffectsCardProps {
  dndAttributes?: Record<string, any>;
  dndListeners?: Record<string, any>;
}

function AudioEffectsCard({ dndAttributes, dndListeners }: AudioEffectsCardProps): JSX.Element {
  // ---------------- App State ----------------
  const appState = useAppState();
  const guiState = useUIContext();
  const { serverSetting } = appState;
  
  // ---------------- States ----------------
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [effects, setEffects] = useState<AudioEffectWithIndex[]>([]);
  const [selectedEffectIndex, setSelectedEffectIndex] = useState<number | null>(null);
  // Background state
  const [bgTracks, setBgTracks] = useState<BackgroundTrack[]>([]);
  const [selectedBgId, setSelectedBgId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'input' | 'output' | 'background'>('output');



  // ---------------- Server Sync Functions ----------------
  
  const syncWithServer = useCallback(async () => {
    if (!serverSetting?.serverSetting?.audioEffects) return;
    
    const serverEffects = serverSetting.serverSetting.audioEffects;
    const effectsArray = serverEffects.map((effect, index) => ({
      ...effect,
      index
    }));
    setEffects(effectsArray);

    // Sync background tracks
    const serverTracks = serverSetting.serverSetting.audioBackgrounds || [];
    setBgTracks(serverTracks as any as BackgroundTrack[]);

  }, [serverSetting?.serverSetting?.audioEffects, serverSetting?.serverSetting?.audioBackgrounds]);

  const updateServerEffects = useCallback(async (newEffects: AudioEffectWithIndex[]) => {
    try {
      const effectsConfig: AudioEffectsConfiguration = newEffects.map(effect => {
        const { index, ...effectData } = effect;
        return effectData;
      });

      await appState.serverSetting.updateServerSettings({
        ...(appState.serverSetting.serverSetting || {}),
        audioEffects: effectsConfig,
      });
    } catch (error) {
      console.error('Failed to update server effects:', error);
      await syncWithServer();
    }
  }, [appState, syncWithServer]);

  // ---------------- Effects ----------------
  
  useEffect(() => {
    syncWithServer();
  }, [syncWithServer]);

  // ---------------- SoundEffects ----------------

  const handleEffectAdd = async (effectType: string, channel: AudioChannel) => {
    const newEffect = createEffectFromServerSchema(effectType, channel, serverSetting?.serverSetting?.audioEffectsSchema);
    const newIndex = effects.length;
    const newEffectWithIndex = { ...newEffect, index: newIndex };

    const updatedEffects = [...effects, newEffectWithIndex];
    setEffects(updatedEffects);
    setSelectedEffectIndex(newIndex);
    
    await updateServerEffects(updatedEffects);
  };

  const handleEffectDelete = async (effectIndex: number) => {
    const updatedEffects = effects
      .filter((_, index) => index !== effectIndex)
      .map((effect, newIndex) => ({ ...effect, index: newIndex }));
    
    setEffects(updatedEffects);

    if (selectedEffectIndex === effectIndex) {
      setSelectedEffectIndex(null);
    } else if (selectedEffectIndex !== null && selectedEffectIndex > effectIndex) {
      setSelectedEffectIndex(selectedEffectIndex - 1);
    }
    
    await updateServerEffects(updatedEffects);
  };

  const handleEffectSelect = (effectIndex: number) => {
    setSelectedEffectIndex(effectIndex);
  };

  const handleEffectToggle = async (effectIndex: number) => {
    const updatedEffects = effects.map((effect, index) => 
      index === effectIndex 
        ? { ...effect, enabled: !effect.enabled }
        : effect
    );
    
    setEffects(updatedEffects);
    await updateServerEffects(updatedEffects);
  };

  const handleParameterChange = async (effectIndex: number, parameterKey: string, value: number | boolean | string) => {
    const updatedEffects = effects.map((effect, index) => 
      index === effectIndex 
        ? {
            ...effect,
            parameters: {
              ...effect.parameters,
              [parameterKey]: value
            }
          }
        : effect
    );
    
    setEffects(updatedEffects);
    await updateServerEffects(updatedEffects);
  };

  //-------------- Background Sounds --------------
  const uploadBackgroundSound = async (file: File) => {
    const uploadSettingsData: BackgroundSoundsUploadSetting = {
      file: { file: file, dir: "" },
      params: {},
    };

    // Upload main model files (model + optional index file)
    console.log('Uploading background sound with settings:', uploadSettingsData);
    await appState.serverSetting.uploadBackgroundSound(uploadSettingsData);
    console.log('Background sound uploaded successfully.');

    // Notify user of successful upload and refresh server state
    guiState.showError(t('bgSoundUploadSuccess'), t('confirmTitle'));
    await appState.serverSetting.reloadServerInfo();
  }

  const updateSoundInfo = async (id: string, key: string, value: any) => {
    const newList = bgTracks.map((t) =>
      t.id === id ? { ...t, [key]: value } : t
    );
    setBgTracks(newList);
    const valueToSend = typeof value === 'object' ? JSON.stringify(value) : String(value);
    serverSetting?.updateSoundInfo(id, key, valueToSend);
  }

  const enableTrack = async (id: string) => {
    const track = bgTracks.find(t => t.id === id);
    if (!track) return;
    const updated = bgTracks.map(t => (t.id === id ? { ...t, enabled: !t.enabled } : t));
    setBgTracks(updated);
    serverSetting.updateSoundInfo(id, 'enabled', String(!track.enabled));
  }

  const deleteTrack = async (id: string) => {
    const filtered = bgTracks.filter(t => t.id !== id).map((t, i) => ({ ...t, order: i }));
    setBgTracks(filtered);
    if (selectedBgId === id) setSelectedBgId(null);
    serverSetting.deleteSound(id);
  }
    

  const selectedEffect = selectedEffectIndex !== null ? effects[selectedEffectIndex] || null : null;

  // Count effects by channel and enabled status
  const inputEffects = effects.filter(e => e.channel === 'input');
  const outputEffects = effects.filter(e => e.channel === 'output');
  const enabledInputEffects = inputEffects.filter(e => e.enabled).length;
  const enabledOutputEffects = outputEffects.filter(e => e.enabled).length;
  const totalActiveEffects = enabledInputEffects + enabledOutputEffects;
  const totalActiveBackground = bgTracks.filter(t => t.enabled).length;

  // ---------------- Render ----------------

  return (
    <div
      className={`${CSS_CLASSES.card} flex-1 min-h-0 flex flex-col ${isCollapsed ? 'h-auto' : 'overflow-hidden'}`}
    >
      <div className={CSS_CLASSES.cardHeader}>
        <div className="flex items-center space-x-3">
          <h4 className={CSS_CLASSES.heading}>{t('audioEffectsTitle')}</h4>
          <div className="flex items-center space-x-2">
            <span className="px-2.5 py-0.5 text-xs rounded-full bg-primary-container text-on-primary-container font-medium">
              {totalActiveEffects} {t('effectsLabel')}
            </span>
            <span className="px-2.5 py-0.5 text-xs rounded-full bg-secondary-container text-on-secondary-container font-medium">
              {totalActiveBackground} {t('backgroundTracksLabel')}
            </span>
          </div>
        </div>
        <div className="flex space-x-1 items-center">
          <button
            onClick={() => setIsCollapsed(!isCollapsed)}
            className={CSS_CLASSES.iconButton}
            title={isCollapsed ? t('expandLabel') : t('collapseLabel')}
          >
            <FontAwesomeIcon icon={isCollapsed ? faChevronDown : faChevronUp} className="h-5 w-5" />
          </button>
          <DragHandle attributes={dndAttributes} listeners={dndListeners} title={t('dragLabel')} />
        </div>
      </div>

      {!isCollapsed && (
        <div className="flex-1 min-h-0 flex">
          {/* Left Panel - Effects or Background List */}
          <div className="w-1/2 pr-3 border-r border-outline-variant">
            {/* Local Tabs above lists only */}
            <div className="flex mb-3 rounded-full p-1 bg-surface-container-high">
              {(['input', 'output', 'background'] as const).map(tab => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`flex-1 px-3 py-1.5 text-sm font-medium rounded-full transition-all duration-150 ${
                    activeTab === tab
                      ? 'bg-primary text-on-primary shadow-elevation-1'
                      : 'bg-transparent text-on-surface-variant hover:bg-surface-variant/20 hover:text-on-surface'
                  }`}
                >
                  <div className="flex items-center justify-center space-x-1">
                    <span>
                      {tab === 'input'
                        ? t('tabInput')
                        : tab === 'output'
                        ? t('tabOutput')
                        : t('tabBackground')}
                    </span>
                  </div>
                </button>
              ))}
            </div>            {activeTab === 'background' ? (
              <BackgroundList
                tracks={bgTracks}
                selectedId={selectedBgId}
                onSelect={setSelectedBgId}
                onAddFiles={(files) => { if (files.length > 0) uploadBackgroundSound(files[0]); }}
                onDelete={deleteTrack}
                onToggle={enableTrack}
              />
            ) : (
              <EffectsList
                channel={activeTab}
                effects={(activeTab === 'input' ? inputEffects : outputEffects) as unknown as CEffect[]}
                selectedEffectIndex={selectedEffectIndex}
                onEffectSelect={handleEffectSelect}
                onEffectAdd={handleEffectAdd}
                onEffectDelete={handleEffectDelete}
                onEffectToggle={handleEffectToggle}
                onEffectReorder={(reorderedChannel) => {
                  const other = effects.filter(e => e.channel !== activeTab);
                  const all = [...other, ...reorderedChannel];
                  const reindexed = all.map((e, i) => ({ ...e, index: i }));
                  setEffects(reindexed);
                  updateServerEffects(reindexed);
                }}
                serverSchema={serverSetting?.serverSetting?.audioEffectsSchema}
                providersInfo={serverSetting?.serverSetting?.audioEffectsProviders}
              />
            )}
          </div>
          
          {/* Right Panel - Effect Configuration */}
          <div className="w-1/2 pl-3">
            {activeTab !== 'background' ? (
              <EffectConfig
                effect={(selectedEffect && (effects.find(e => e.index === selectedEffectIndex && e.channel === activeTab) || null)) as any}
                onParameterChange={handleParameterChange}
                serverSchema={serverSetting?.serverSetting?.audioEffectsSchema}
              />
            ) : (
              <BackgroundConfig
                track={bgTracks.find(t => t.id === selectedBgId) || null}
                onChange={updateSoundInfo}
              />
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default AudioEffectsCard;
