import React, { JSX, useEffect } from 'react';
import { AUDIO_KEYS, INDEXEDDB_KEYS } from '../../styles/constants';
import { useAppState } from '../../context/AppContext';
import { useIndexedDB } from '@dannadori/voice-changer-client-js';
import { useUIContext } from '../../context/UIContext';
import MD3Select from '../Helpers/MD3Select';
import { t } from '../../locales';

function AudioDevicesClient(): JSX.Element {
  // ---------------- States ----------------
  const appState = useAppState();
  const uiState = useUIContext();
  const { getItem, setItem } = useIndexedDB({ clientType: null });

  // ---------------- Hooks ----------------

  // Configure Audio Sinks for Output in Client Mode
  useEffect(() => {
    const setAudioOutput = async () => {
      const mediaDeviceInfos = await navigator.mediaDevices.enumerateDevices();

      [
        AUDIO_KEYS.AUDIO_ELEMENT_FOR_PLAY_RESULT,
        AUDIO_KEYS.AUDIO_ELEMENT_FOR_TEST_CONVERTED_ECHOBACK
      ].forEach((x) => {
        const audio = document.getElementById(x) as HTMLAudioElement;
        if (audio) {
          if (appState.serverSetting.serverSetting.enableServerAudio == 1) {
            audio.volume = 0;
          } else if (uiState.audioOutputForGUI == 'none') {
            try {
              audio.setSinkId('');
              audio.volume = 0;
            } catch (e) {
              console.error('catch:' + e);
            }
          } else {
            const audioOutputs = mediaDeviceInfos.filter((x) => {
              return x.kind == 'audiooutput';
            });
            const found = audioOutputs.some((x) => {
              return x.deviceId == uiState.audioOutputForGUI;
            });
            if (found) {
              try {
                audio.setSinkId(uiState.audioOutputForGUI);
                audio.volume = 1;
              } catch (e) {
                console.error('catch:' + e);
              }
            } else {
              console.warn('No audio output device. use default');
            }
          }
        }
      });
    };
    setAudioOutput();
  }, [uiState.audioOutputForGUI, appState.serverSetting.serverSetting.enableServerAudio]);

  // Configure Audio Sinks for Monitor in Client Mode
  useEffect(() => {
    const setAudioMonitor = async () => {
      const mediaDeviceInfos = await navigator.mediaDevices.enumerateDevices();

      [AUDIO_KEYS.AUDIO_ELEMENT_FOR_PLAY_MONITOR].forEach((x) => {
        const audio = document.getElementById(x) as HTMLAudioElement;
        if (audio) {
          if (appState.serverSetting.serverSetting.enableServerAudio == 1) {
            audio.volume = 0;
          } else if (uiState.audioMonitorForGUI == 'none') {
            try {
              audio.setSinkId('');
              audio.volume = 0;
            } catch (e) {
              console.error('catch:' + e);
            }
          } else {
            const audioOutputs = mediaDeviceInfos.filter((x) => {
              return x.kind == 'audiooutput';
            });
            const found = audioOutputs.some((x) => {
              return x.deviceId == uiState.audioMonitorForGUI;
            });
            if (found) {
              try {
                audio.setSinkId(uiState.audioMonitorForGUI);
                audio.volume = 1;
              } catch (e) {
                console.error('catch:' + e);
              }
            } else {
              console.warn('No audio output device. use default');
            }
          }
        }
      });
    };
    setAudioMonitor();
  }, [uiState.audioMonitorForGUI, appState.serverSetting.serverSetting.enableServerAudio]);

  // Load Default Devices from IndexDB
  useEffect(() => {
    // Wait for initialization of the client
    if (!appState.initializedRef.current) {
      return;
    }

    // Set audio output element id
    getItem(INDEXEDDB_KEYS.INDEXEDDB_KEY_AUDIO_INPUT).then((input) => {
      if (input) {
        uiState.setAudioInputForGUI(input as string);
        appState.setVoiceChangerClientSetting({
          ...appState.setting.voiceChangerClientSetting,
          audioInput: input as string
        });
      }
    });

    // Set audio output element id
    appState.setAudioOutputElementId(AUDIO_KEYS.AUDIO_ELEMENT_FOR_PLAY_RESULT);
    getItem(INDEXEDDB_KEYS.INDEXEDDB_KEY_AUDIO_OUTPUT).then((output) => {
      if (output) {
        uiState.setAudioOutputForGUI(output as string);
      }
    });

    // Set audio monitor element id
    appState.setAudioMonitorElementId(AUDIO_KEYS.AUDIO_ELEMENT_FOR_PLAY_MONITOR);
    getItem(INDEXEDDB_KEYS.INDEXEDDB_KEY_AUDIO_MONITOR).then((monitor) => {
      if (monitor) {
        uiState.setAudioMonitorForGUI(monitor as string);
      }
    });
  }, [appState.initializedRef.current]);

  // ---------------- Handlers ----------------

  // Handle input device change
  const handleInputDeviceChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    appState.setVoiceChangerClientSetting({
      ...appState.setting.voiceChangerClientSetting,
      audioInput: event.target.value
    });
    setItem(INDEXEDDB_KEYS.INDEXEDDB_KEY_AUDIO_INPUT, event.target.value);
    uiState.setAudioInputForGUI(event.target.value);
  };

  // Handle output device change
  const handleOutputDeviceChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setItem(INDEXEDDB_KEYS.INDEXEDDB_KEY_AUDIO_OUTPUT, event.target.value);
    uiState.setAudioOutputForGUI(event.target.value);
  };

  // Handle monitor device change
  const handleMonitorDeviceChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setItem(INDEXEDDB_KEYS.INDEXEDDB_KEY_AUDIO_MONITOR, event.target.value);
    uiState.setAudioMonitorForGUI(event.target.value);
  };

  // ---------------- Render ----------------

  const inputOptions =
    uiState.inputAudioDeviceInfo.length === 0
      ? [{ value: '', label: t('noInputDevices') }]
      : uiState.inputAudioDeviceInfo.map((device) => ({
          value: device.deviceId,
          label: device.label
        }));

  const outputOptions =
    uiState.outputAudioDeviceInfo.length === 0
      ? [{ value: '', label: t('noOutputDevices') }]
      : uiState.outputAudioDeviceInfo.map((device) => ({
          value: device.deviceId,
          label: device.label
        }));

  const monitorOptions =
    uiState.outputAudioDeviceInfo.length === 0
      ? [{ value: '', label: t('noOutputDevices') }]
      : [
          { value: 'none', label: t('noDeviceSelected') },
          ...uiState.outputAudioDeviceInfo.map((device) => ({
            value: device.deviceId,
            label: device.label
          }))
        ];

  return (
    <div className="flex flex-col space-y-4 bg-surface-container-low p-3 rounded-md border border-outline-variant">
      {/* Input Device */}
      <MD3Select
        id="inputCh"
        label={t('inputDeviceLabel')}
        value={uiState.audioInputForGUI}
        onChange={handleInputDeviceChange}
        options={inputOptions}
      />

      {/* Output Device */}
      <MD3Select
        id="outputCh"
        label={t('outputDeviceLabel')}
        value={uiState.audioOutputForGUI}
        onChange={handleOutputDeviceChange}
        options={outputOptions}
      />

      {/* Monitor Device */}
      <MD3Select
        id="monCh"
        label={t('monitorDeviceLabel')}
        value={uiState.audioMonitorForGUI}
        onChange={handleMonitorDeviceChange}
        options={monitorOptions}
      />
    </div>
  );
}

export default AudioDevicesClient;
