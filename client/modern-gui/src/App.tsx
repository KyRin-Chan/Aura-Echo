import LeftSidebar from './components/LeftSideBar/Sidebar';
import BottomBar from './components/BottomBar/BottomBar';
import { JSX, useEffect, ReactNode } from 'react';
import GenericModal from './components/Modals/GenericModal';
import { PassthroughConfirmModalProps } from './components/BottomBar/Modals/PassthroughConfirmModal';
import { RVCModelSlot } from '@dannadori/voice-changer-client-js';
import MainContent from './components/MainContent';
import { useAppState } from './context/AppContext';
import { useUIContext } from './context/UIContext';
import { getFormantBinding, getFormantProfile, serializePitchStats, serializeProfileEnvelope, syncBindingsFromServer, syncProfilesFromServer } from './utils/formantStorage';

function App(): JSX.Element {
  // ---------------- State ----------------
  const appState = useAppState();
  const { showError } = useUIContext();

  // ---------------- Hooks ----------------

  // Monitor error messages from appState
  useEffect(() => {
    if (appState.errorMessage && appState.errorMessage !== '') {
      showError(appState.errorMessage, 'Error');
      appState.resetErrorMessage();
    }
  }, [appState.errorMessage, showError, appState]);

  // Auto-restore formant profile when model slot changes (server-authoritative first)
  useEffect(() => {
    const slotIndex = appState.serverSetting?.serverSetting?.modelSlotIndex;
    if (slotIndex === undefined || slotIndex === -1) return;

    let cancelled = false;
    const restoreFormantProfile = async () => {
      try {
        const [syncedProfiles, syncedBindings] = await Promise.all([
          syncProfilesFromServer(),
          syncBindingsFromServer()
        ]);
        if (cancelled) return;

        const binding =
          syncedBindings[slotIndex] ??
          syncedBindings[String(slotIndex)] ??
          getFormantBinding(slotIndex);

        if (binding && binding.targetProfileId && binding.inputProfileId) {
          const targetProfile =
            syncedProfiles.find((p) => p.id === binding.targetProfileId) ??
            getFormantProfile(binding.targetProfileId);
          const inputProfile =
            syncedProfiles.find((p) => p.id === binding.inputProfileId) ??
            getFormantProfile(binding.inputProfileId);

          if (targetProfile && inputProfile) {
            const isActive = binding.active !== undefined ? Boolean(binding.active) : true;
            const currentSetting = appState.serverSetting.serverSetting;
            appState.serverSetting.updateServerSettings({
              ...currentSetting,
              formantProfileActive: isActive,
              formantProfileTargetEnvelope: serializeProfileEnvelope(targetProfile),
              formantProfileTargetSr: targetProfile.sr,
              formantProfileInputEnvelope: serializeProfileEnvelope(inputProfile),
              formantProfileInputSr: inputProfile.sr,
              formantProfileStrength: binding.strength !== undefined ? binding.strength : (currentSetting.formantProfileStrength ?? 0.35),
              formantProfilePitchStats: serializePitchStats(inputProfile, targetProfile),
              formantProfilePitchScale: binding.pitchScale !== undefined ? binding.pitchScale : (currentSetting.formantProfilePitchScale ?? 0.85),
              formantProfilePitchOffset: binding.pitchOffset !== undefined ? binding.pitchOffset : (currentSetting.formantProfilePitchOffset ?? 0.0)
            });
            console.log(`Auto-loaded Match EQ profiles for model slot ${slotIndex}, active: ${isActive}`);
          } else if (appState.serverSetting?.serverSetting?.formantProfileActive) {
            appState.serverSetting.updateServerSettings({
              ...appState.serverSetting.serverSetting,
              formantProfileActive: false,
              formantProfilePitchStats: '{}'
            });
          }
        } else if (appState.serverSetting?.serverSetting?.formantProfileActive) {
          appState.serverSetting.updateServerSettings({
            ...appState.serverSetting.serverSetting,
            formantProfileActive: false,
            formantProfilePitchStats: '{}'
          });
        }
      } catch (e) {
        console.error('Error auto-restoring formant profile:', e);
      }
    };

    restoreFormantProfile();
    return () => {
      cancelled = true;
    };
  }, [appState.serverSetting?.serverSetting?.modelSlotIndex]);

  // ---------------- Render ----------------

  return (
    <div className="flex flex-col h-screen font-sans bg-surface text-on-surface">
      <div className="flex flex-grow overflow-hidden">
        <LeftSidebar />

        <main className="flex-grow p-4 overflow-y-auto bg-surface">
          <MainContent />
        </main>
      </div>
      <BottomBar />
    </div>
  );
}

export default App;