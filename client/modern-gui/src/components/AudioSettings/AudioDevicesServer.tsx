import React, { useEffect, useMemo } from 'react';
import { useAppState } from '../../context/AppContext';
import { useState } from 'react';
import { ServerAudioDevice } from '@dannadori/voice-changer-client-js/const';
import { useUIContext } from '../../context/UIContext';
import MD3Select from '../Helpers/MD3Select';
import { t } from '../../locales';


function AudioDevicesServer() {
  // ---------------- States ----------------
  const appState = useAppState();
  const uiState = useUIContext();

  const sampleRates = [16000, 32000, 44100, 48000, 96000, 192000];

  const [availableAudioDrivers, setAvailableAudioDrivers] = useState<string[]>([]);
  const [selectedAudioDriver, setSelectedAudioDriver] = useState<string>('');
  const [selectedMonitorAudioDriver, setSelectedMonitorAudioDriver] = useState<string>('');

  // ---------------- Hooks ----------------

  // Get Server Input Devices based on selected Audio Driver
  const serverInputDevices = useMemo(() => {
    return appState.serverSetting?.serverSetting?.serverAudioInputDevices.filter(
      (device) => device.hostAPI === selectedAudioDriver
    ) || [];
  }, [appState.serverSetting, selectedAudioDriver]);

  // Get Server Output Devices based on selected Audio Driver
  const serverOutputDevices = useMemo(() => {
    return appState.serverSetting?.serverSetting?.serverAudioOutputDevices.filter(
      (device) => device.hostAPI === selectedAudioDriver
    ) || [];
  }, [appState.serverSetting, selectedAudioDriver]);

  // Get Server Monitor Devices based on selected Monitor Audio Driver
  const serverMonitorDevices = useMemo(() => {
    if (!selectedMonitorAudioDriver) return [];
    return appState.serverSetting?.serverSetting?.serverAudioOutputDevices.filter(
      (device) => device.hostAPI === selectedMonitorAudioDriver
    ) || [];
  }, [appState.serverSetting, selectedMonitorAudioDriver]);

  // Set selected monitor audio driver based on selected audio driver
  useEffect(() => {
    if (availableAudioDrivers.length > 0) {
      const monitor = appState.serverSetting.serverSetting.serverAudioOutputDevices.find(
        (x) => x.index === appState.serverSetting.serverSetting.serverMonitorDeviceId
      );
      setSelectedMonitorAudioDriver(
        availableAudioDrivers.find((x) => x === monitor?.hostAPI) || availableAudioDrivers[0]
      );
    }
  }, [availableAudioDrivers]);

  // Fetch server devices
  useEffect(() => {
    fetchServerDevices();
  }, [appState.serverSetting.serverSetting.enableServerAudio]);

  // ---------------- Handlers ----------------

  // Handle Sample Rate Change
  const handleSampleRateChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    appState.serverSetting.updateServerSettings({
      ...appState.serverSetting.serverSetting,
      serverInputAudioSampleRate: parseInt(event.target.value),
      serverOutputAudioSampleRate: parseInt(event.target.value),
      serverMonitorAudioSampleRate: parseInt(event.target.value),
      serverAudioSampleRate: parseInt(event.target.value)
    });
  };

  // Handle Audio Driver Change
  const handleAudioDriverChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    if (appState.serverSetting.serverSetting.serverAudioStated === 1) {
      uiState.setIsConverting(false);
      appState.serverSetting.updateServerSettings({
        ...appState.serverSetting.serverSetting,
        serverAudioStated: 0
      });
    }
    setSelectedAudioDriver(event.target.value);
  };

  // Handle Input Device Change
  const handleInputDeviceChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    appState.serverSetting.updateServerSettings({
      ...appState.serverSetting.serverSetting,
      serverInputDeviceId: parseInt(event.target.value)
    });
  };

  // Handle Output Device Change
  const handleOutputDeviceChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    appState.serverSetting.updateServerSettings({
      ...appState.serverSetting.serverSetting,
      serverOutputDeviceId: parseInt(event.target.value)
    });
  };

  // Handle Monitor Device Change
  const handleMonitorDeviceChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    appState.serverSetting.updateServerSettings({
      ...appState.serverSetting.serverSetting,
      serverMonitorDeviceId: parseInt(event.target.value)
    });
  };

  // Handle Input Channel Change
  const handleInputChannelChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    appState.serverSetting.updateServerSettings({
      ...appState.serverSetting.serverSetting,
      asioInputChannel: parseInt(event.target.value)
    });
  };

  // Handle Output Channel Change
  const handleOutputChannelChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    appState.serverSetting.updateServerSettings({
      ...appState.serverSetting.serverSetting,
      asioOutputChannel: parseInt(event.target.value)
    });
  };

  // Handle Monitor Audio Driver Change
  const handleMonitorAudioDriverChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const newDriver = event.target.value;
    setSelectedMonitorAudioDriver(newDriver);
  };

  // ---------------- Methods ----------------

  // Method to fetch server devices
  const fetchServerDevices = async () => {
    try {
      const serverSettings = appState.serverSetting.serverSetting;
      const inputs: ServerAudioDevice[] = serverSettings.serverAudioInputDevices || [];
      const outputs: ServerAudioDevice[] = serverSettings.serverAudioOutputDevices || [];

      // Extract unique hostAPIs for Audio Driver dropdown
      const hostApis = new Set<string>();
      inputs.forEach((device) => device.hostAPI && hostApis.add(device.hostAPI));
      outputs.forEach((device) => device.hostAPI && hostApis.add(device.hostAPI));
      const uniqueHostApis = Array.from(hostApis);
      setAvailableAudioDrivers(uniqueHostApis);

      // Set selected audio driver based on server input device
      if (uniqueHostApis.length > 0) {
        const input = appState.serverSetting.serverSetting.serverAudioInputDevices.find(
          (x) => x.index === appState.serverSetting.serverSetting.serverInputDeviceId
        );
        setSelectedAudioDriver(uniqueHostApis.find((x) => x === input?.hostAPI) || uniqueHostApis[0]);
      }
    } catch (err) {
      console.error('Error fetching server devices:', err);
    }
  };

  // ---------------- Render ----------------

  const sampleRateOptions = sampleRates.map((rate) => ({
    value: rate,
    label: `${rate} Hz`
  }));

  const audioDriverOptions =
    availableAudioDrivers.length === 0
      ? [{ value: '', label: 'No drivers available' }]
      : availableAudioDrivers.map((driver) => ({
          value: driver,
          label: driver
        }));

  const inputOptions =
    serverInputDevices.length === 0
      ? [{ value: -1, label: 'No input devices found' }]
      : [
          ...(!serverInputDevices.find(
            (device) => device.index === appState.serverSetting.serverSetting.serverInputDeviceId
          )
            ? [{ value: -1, label: 'No device selected' }]
            : []),
          ...serverInputDevices.map((device) => ({
            value: device.index,
            label: `[${device.hostAPI}] ${device.name}`
          }))
        ];

  const selectedInputDeviceObj = serverInputDevices.find(
    (device) => device.index === appState.serverSetting.serverSetting.serverInputDeviceId
  );
  const inputChannelCount = selectedInputDeviceObj?.maxInputChannels || 0;
  const inputChannelOptions = [
    { value: -1, label: 'Default' },
    ...Array.from({ length: inputChannelCount }, (_, index) => ({
      value: index,
      label: String(index)
    }))
  ];

  const outputOptions =
    serverOutputDevices.length === 0
      ? [{ value: -1, label: 'No output devices found' }]
      : [
          ...(!serverOutputDevices.find(
            (device) => device.index === appState.serverSetting.serverSetting.serverOutputDeviceId
          )
            ? [{ value: -1, label: 'No device selected' }]
            : []),
          ...serverOutputDevices.map((device) => ({
            value: device.index,
            label: `[${device.hostAPI}] ${device.name}`
          }))
        ];

  const selectedOutputDeviceObj = serverOutputDevices.find(
    (device) => device.index === appState.serverSetting.serverSetting.serverOutputDeviceId
  );
  const outputChannelCount = selectedOutputDeviceObj?.maxOutputChannels || 0;
  const outputChannelOptions = [
    { value: -1, label: 'Default' },
    ...Array.from({ length: outputChannelCount }, (_, index) => ({
      value: index,
      label: String(index)
    }))
  ];

  const monitorDriverOptions =
    availableAudioDrivers.length === 0
      ? [{ value: '', label: 'No drivers available' }]
      : availableAudioDrivers.map((driver) => ({
          value: driver,
          label: driver
        }));

  const monitorOptions =
    serverMonitorDevices.length === 0
      ? [{ value: -1, label: 'No devices for driver' }]
      : [
          { value: -1, label: 'No device selected' },
          ...serverMonitorDevices.map((device) => ({
            value: device.index,
            label: `[${device.hostAPI}] ${device.name}`
          }))
        ];

  return (
    <div className="flex flex-col space-y-4 bg-surface-container-low p-3 rounded-md border border-outline-variant">
      <MD3Select
        id="sampleRate"
        label="Sample Rate"
        value={appState.serverSetting?.serverSetting?.serverAudioSampleRate}
        onChange={handleSampleRateChange}
        options={sampleRateOptions}
      />

      <MD3Select
        id="audioDriver"
        label="Audio Driver"
        value={selectedAudioDriver}
        onChange={handleAudioDriverChange}
        options={audioDriverOptions}
      />

      <div className="flex items-end gap-3">
        <div className={selectedAudioDriver === 'ASIO' ? 'w-[70%]' : 'w-full'}>
          <MD3Select
            id="inputCh"
            label={t('inputDeviceLabel')}
            value={appState.serverSetting.serverSetting.serverInputDeviceId}
            onChange={handleInputDeviceChange}
            options={inputOptions}
          />
        </div>

        {selectedAudioDriver === 'ASIO' && selectedInputDeviceObj && (
          <div className="w-[30%]">
            <MD3Select
              id="inputChannel"
              label="Channel"
              value={appState.serverSetting.serverSetting.asioInputChannel}
              onChange={handleInputChannelChange}
              options={inputChannelOptions}
            />
          </div>
        )}
      </div>

      <div className="flex items-end gap-3">
        <div className={selectedAudioDriver === 'ASIO' ? 'w-[70%]' : 'w-full'}>
          <MD3Select
            id="outputCh"
            label={t('outputDeviceLabel')}
            value={appState.serverSetting.serverSetting.serverOutputDeviceId}
            onChange={handleOutputDeviceChange}
            options={outputOptions}
          />
        </div>

        {selectedAudioDriver === 'ASIO' && selectedOutputDeviceObj && (
          <div className="w-[30%]">
            <MD3Select
              id="outputChannel"
              label="Channel"
              value={appState.serverSetting.serverSetting.asioOutputChannel}
              onChange={handleOutputChannelChange}
              options={outputChannelOptions}
            />
          </div>
        )}
      </div>

      <div className="flex items-end gap-3">
        <div className="w-[30%]">
          <MD3Select
            id="monitorAudioDriver"
            label="Monitor Driver"
            value={selectedMonitorAudioDriver}
            onChange={handleMonitorAudioDriverChange}
            options={monitorDriverOptions}
          />
        </div>

        <div className="w-[70%]">
          <MD3Select
            id="monCh"
            label={t('monitorDeviceLabel')}
            value={appState.serverSetting.serverSetting.serverMonitorDeviceId}
            onChange={handleMonitorDeviceChange}
            options={monitorOptions}
            disabled={!selectedMonitorAudioDriver || serverMonitorDevices.length === 0}
          />
        </div>
      </div>
    </div>
  );
}

export default AudioDevicesServer;