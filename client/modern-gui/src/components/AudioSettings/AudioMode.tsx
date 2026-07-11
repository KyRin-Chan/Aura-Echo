import { Dispatch, SetStateAction, useEffect, useState } from "react";
import { useAppState } from "../../context/AppContext";
import { useUIContext } from "../../context/UIContext";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faExclamationTriangle } from "@fortawesome/free-solid-svg-icons";
import { t } from "../../locales";

interface AudioModeProps {
  audioState: "client" | "server";
  setAudioState: Dispatch<SetStateAction<"client" | "server">>;
}

function AudioMode({ audioState, setAudioState }: AudioModeProps): JSX.Element {
  // ---------------- States ----------------
  const appState = useAppState();
  const uiContext = useUIContext();
  const [warningMessage, setWarningMessage] = useState<string | null>(null);

  // Check if there are client audio devices that are not empty
  const isClientAudioAvailable =
    uiContext?.inputAudioDeviceInfo?.length > 0 &&
    uiContext?.outputAudioDeviceInfo?.length > 0 &&
    uiContext?.inputAudioDeviceInfo[0].deviceId !== "" &&
    uiContext?.outputAudioDeviceInfo[0].deviceId !== "";

  // Check if there are server audio devices
  const serverInputDevices = appState.serverSetting?.serverSetting?.serverAudioInputDevices;
  const serverOutputDevices = appState.serverSetting?.serverSetting?.serverAudioOutputDevices;
  const isServerAudioAvailable =
    Array.isArray(serverInputDevices) && serverInputDevices.length > 0 &&
    Array.isArray(serverOutputDevices) && serverOutputDevices.length > 0;

  // ---------------- Hooks ----------------

  // Set the audio state based on the availability of client and server audio devices
  useEffect(() => {
    let newAudioState: "client" | "server" | null = null;
    if (!isClientAudioAvailable && isServerAudioAvailable) {
      newAudioState = "server";
    } else if (isClientAudioAvailable && !isServerAudioAvailable) {
      newAudioState = "client";
    }

    // Set Audio Mode if only one type is available
    if (newAudioState && newAudioState !== audioState) {
      setAudioState(newAudioState);
      if (newAudioState === "server") {
        appState.serverSetting.updateServerSettings({
          ...appState.serverSetting.serverSetting,
          enableServerAudio: 1
        });
      } else {
        appState.serverSetting.updateServerSettings({
          ...appState.serverSetting.serverSetting,
          enableServerAudio: 0
        });
      }
    }

    // Update warning message
    const messages = [];
    if (!isClientAudioAvailable) {
      messages.push("Client audio not available");
    }
    if (!isServerAudioAvailable) {
      messages.push("Server audio not available");
    }
    setWarningMessage(messages.length > 0 ? messages.join(" and ") + "." : null);
  }, [isClientAudioAvailable, isServerAudioAvailable, appState.serverSetting, setAudioState, audioState]);

  // ---------------- Handlers ----------------

  // Handle Client Select
  const selectClientMode = () => {
    if (!isClientAudioAvailable || uiContext.isConverting) return;
    appState.serverSetting.updateServerSettings({
      ...appState.serverSetting.serverSetting,
      enableServerAudio: 0
    });
    setAudioState("client");
  };

  // Handle Server Select
  const selectServerMode = () => {
    if (!isServerAudioAvailable || uiContext.isConverting) return;
    appState.serverSetting.updateServerSettings({
      ...appState.serverSetting.serverSetting,
      enableServerAudio: 1
    });
    setAudioState("server");
  };

  // ---------------- Render ----------------

  return (
    <div className="space-y-4">
      <div className="pb-3 border-b border-outline-variant">
        <div className="flex items-center mb-2.5">
          <label className="block text-sm font-medium text-on-surface-variant">{t('audioProcessingModeLabel')}</label>
          {warningMessage && (
            <div className="ml-2 relative group cursor-help">
              <FontAwesomeIcon icon={faExclamationTriangle} className="text-error" />
              <div className="absolute left-1/2 -translate-x-1/2 bottom-full mb-2 px-2 py-1 text-xs text-on-error bg-error rounded-md opacity-0 group-hover:opacity-100 transition-opacity duration-300 whitespace-nowrap z-10">
                {warningMessage}
              </div>
            </div>
          )}
        </div>
        <div className="flex items-center justify-between">
          {/* MD3 Segmented Button */}
          <div className="inline-flex rounded-full border border-outline overflow-hidden">
            <button
              onClick={selectClientMode}
              disabled={!isClientAudioAvailable || uiContext.isConverting}
              className={`px-5 py-1.5 text-xs font-semibold transition-all duration-150 ${
                audioState === "client"
                  ? "bg-secondary-container text-on-secondary-container"
                  : "bg-transparent text-on-surface hover:bg-surface-variant/20"
              } ${(!isClientAudioAvailable || uiContext.isConverting) ? "opacity-50 cursor-not-allowed" : ""}`}
            >
              {t('clientModeLabel')}
            </button>
            <div className="w-[1px] bg-outline" />
            <button
              onClick={selectServerMode}
              disabled={!isServerAudioAvailable || uiContext.isConverting}
              className={`px-5 py-1.5 text-xs font-semibold transition-all duration-150 ${
                audioState === "server"
                  ? "bg-secondary-container text-on-secondary-container"
                  : "bg-transparent text-on-surface hover:bg-surface-variant/20"
              } ${(!isServerAudioAvailable || uiContext.isConverting) ? "opacity-50 cursor-not-allowed" : ""}`}
            >
              {t('serverModeLabel')}
            </button>
          </div>

          {audioState === "client" && (
            <button
              onClick={() => uiContext.reloadDeviceInfo()}
              className="px-4 py-1.5 text-xs font-semibold border border-outline text-primary rounded-full hover:bg-primary/8 active:scale-97 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              disabled={uiContext.isConverting || appState.serverSetting.serverSetting.serverAudioStated === 1}
            >
              {t('reloadDeviceListLabel')}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

export default AudioMode;