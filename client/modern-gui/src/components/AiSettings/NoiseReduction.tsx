import { ClientState } from "@dannadori/voice-changer-client-js";
import { CSS_CLASSES } from "../../styles/constants";
import { useEffect } from "react";
import { INDEXEDDB_KEYS } from "../../styles/constants";
import MD3Switch from "../Helpers/MD3Switch";

interface NoiseReductionProps {
  appState: ClientState;
  getItem: (key: string) => Promise<any>;
  setItem: (key: string, value: any) => Promise<void>;
}

function NoiseReduction({ appState, getItem, setItem }: NoiseReductionProps) {
  // Load Noise Reduction from Cache
  useEffect(() => {
    const loadCache = async () => {
      const echo = await getItem(INDEXEDDB_KEYS.INDEXEDDB_KEY_ECHO);
      if (echo !== null) {
        appState.setVoiceChangerClientSetting({
          ...appState.setting.voiceChangerClientSetting,
          echoCancel: echo as boolean
        });
      }
      const noise1 = await getItem(INDEXEDDB_KEYS.INDEXEDDB_KEY_NOISE1);
      if (noise1 !== null) {
        appState.setVoiceChangerClientSetting({
          ...appState.setting.voiceChangerClientSetting,
          noiseSuppression: noise1 as boolean
        });
      }
      const noise2 = await getItem(INDEXEDDB_KEYS.INDEXEDDB_KEY_NOISE2);
      if (noise2 !== null) {
        appState.setVoiceChangerClientSetting({
          ...appState.setting.voiceChangerClientSetting,
          noiseSuppression2: noise2 as boolean
        });
      }
    };
    loadCache();
  }, []);

  // Handle Noise Reduction Change
  const handleChangeNoiseSuppression = (value: boolean) => {
    appState.setVoiceChangerClientSetting({
      ...appState.setting.voiceChangerClientSetting,
      noiseSuppression: value
    });
    setItem(INDEXEDDB_KEYS.INDEXEDDB_KEY_NOISE1, value);
  };

  // Handle Noise Reduction 2 Change
  const handleChangeNoiseSuppression2 = (value: boolean) => {
    appState.setVoiceChangerClientSetting({
      ...appState.setting.voiceChangerClientSetting,
      noiseSuppression2: value
    });
    setItem(INDEXEDDB_KEYS.INDEXEDDB_KEY_NOISE2, value);
  };

  // Handle Echo Cancel Change
  const handleChangeEchoCancel = (value: boolean) => {
    appState.setVoiceChangerClientSetting({
      ...appState.setting.voiceChangerClientSetting,
      echoCancel: value
    });
    setItem(INDEXEDDB_KEYS.INDEXEDDB_KEY_ECHO, value);
  };

  return (
    <div className="flex flex-col space-y-2">
      <label className={CSS_CLASSES.label}>Noise Reduction</label>
      <div className="space-y-2.5 mt-1 bg-surface-container-low p-3 rounded-md border border-outline-variant">
        <MD3Switch
          checked={appState.setting.voiceChangerClientSetting.echoCancel ?? false}
          onChange={handleChangeEchoCancel}
          disabled={appState.serverSetting.serverSetting.enableServerAudio === 1}
          label="Echo Cancellation"
        />
        <MD3Switch
          checked={appState.setting.voiceChangerClientSetting.noiseSuppression ?? false}
          onChange={handleChangeNoiseSuppression}
          disabled={appState.serverSetting.serverSetting.enableServerAudio === 1}
          label="Noise Suppression"
        />
        <MD3Switch
          checked={appState.setting.voiceChangerClientSetting.noiseSuppression2 ?? false}
          onChange={handleChangeNoiseSuppression2}
          disabled={appState.serverSetting.serverSetting.enableServerAudio === 1}
          label="Noise Suppression 2"
        />
      </div>
    </div>
  );
}

export default NoiseReduction;