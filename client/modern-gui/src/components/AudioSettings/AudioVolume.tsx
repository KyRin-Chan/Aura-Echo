import { JSX, useEffect, useState } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faMicrophone, faVolumeUp, faHeadphones } from '@fortawesome/free-solid-svg-icons';
import { useAppState } from '../../context/AppContext';
import { CSS_CLASSES } from '../../styles/constants';
import MD3Slider from '../Helpers/MD3Slider';

function AudioVolume(): JSX.Element {
  // ---------------- States ----------------
  const appState = useAppState();
  const [inputGain, setInputGain] = useState(1);
  const [outputGain, setOutputGain] = useState(1);
  const [monitorGain, setMonitorGain] = useState(1);

  // ---------------- Hooks ----------------

  // Set Audio Gains
  useEffect(() => {
    setInputGain(appState.serverSetting.serverSetting.serverInputAudioGain);
    setOutputGain(appState.serverSetting.serverSetting.serverOutputAudioGain);
    setMonitorGain(appState.serverSetting.serverSetting.serverMonitorAudioGain);
  }, [
    appState.serverSetting.serverSetting.serverInputAudioGain,
    appState.serverSetting.serverSetting.serverOutputAudioGain,
    appState.serverSetting.serverSetting.serverMonitorAudioGain
  ]);

  // ---------------- Handlers ----------------

  // Handle Input Gain Change
  const handleInputGainChange = (value: number) => {
    const gain = value / 100;
    appState.serverSetting.updateServerSettings({
      ...appState.serverSetting.serverSetting,
      serverInputAudioGain: gain
    });

    appState.setVoiceChangerClientSetting({
      ...appState.setting.voiceChangerClientSetting,
      inputGain: gain
    });
  };

  // Handle Output Gain Change
  const handleOutputGainChange = (value: number) => {
    const gain = value / 100;
    appState.serverSetting.updateServerSettings({
      ...appState.serverSetting.serverSetting,
      serverOutputAudioGain: gain
    });

    appState.setVoiceChangerClientSetting({
      ...appState.setting.voiceChangerClientSetting,
      outputGain: gain
    });
  };

  // Handle Monitor Gain Change
  const handleMonitorGainChange = (value: number) => {
    const gain = value / 100;
    appState.serverSetting.updateServerSettings({
      ...appState.serverSetting.serverSetting,
      serverMonitorAudioGain: gain
    });

    appState.setVoiceChangerClientSetting({
      ...appState.setting.voiceChangerClientSetting,
      monitorGain: gain
    });
  };

  // ---------------- Render ----------------

  return (
    <div className="flex flex-col space-y-4 bg-surface-container-low p-3 rounded-md border border-outline-variant">
      <div>
        <label htmlFor="inputGain" className={CSS_CLASSES.label}>
          <FontAwesomeIcon icon={faMicrophone} className="mr-2 text-primary" />
          Input Volume
        </label>
        <MD3Slider
          id="inputGain"
          min={10}
          max={250}
          step={1}
          value={Math.round(inputGain * 100)}
          onChange={handleInputGainChange}
          onImmediateChange={(value) => setInputGain(value / 100)}
          showValue={true}
          valueFormatter={(val) => `${val}%`}
        />
      </div>
      <div>
        <label htmlFor="outputGain" className={CSS_CLASSES.label}>
          <FontAwesomeIcon icon={faVolumeUp} className="mr-2 text-primary" />
          Output Volume
        </label>
        <MD3Slider
          id="outputGain"
          min={10}
          max={400}
          step={1}
          value={Math.round(outputGain * 100)}
          onChange={handleOutputGainChange}
          onImmediateChange={(value) => setOutputGain(value / 100)}
          showValue={true}
          valueFormatter={(val) => `${val}%`}
        />
      </div>
      <div>
        <label htmlFor="monitorGain" className={CSS_CLASSES.label}>
          <FontAwesomeIcon icon={faHeadphones} className="mr-2 text-primary" />
          Monitor Volume
        </label>
        <MD3Slider
          id="monitorGain"
          min={0}
          max={400}
          step={1}
          value={Math.round(monitorGain * 100)}
          onChange={handleMonitorGainChange}
          onImmediateChange={(value) => setMonitorGain(value / 100)}
          showValue={true}
          valueFormatter={(val) => `${val}%`}
        />
      </div>
    </div>
  );
}

export default AudioVolume;
