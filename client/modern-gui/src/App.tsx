import LeftSidebar from './components/LeftSideBar/Sidebar';
import BottomBar from './components/BottomBar/BottomBar';
import { JSX, useEffect, ReactNode } from 'react';
import GenericModal from './components/Modals/GenericModal';
import { PassthroughConfirmModalProps } from './components/BottomBar/Modals/PassthroughConfirmModal';
import { RVCModelSlot } from '@dannadori/voice-changer-client-js';
import MainContent from './components/MainContent';
import { useAppState } from './context/AppContext';
import { useUIContext } from './context/UIContext';

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

  // Auto-restore formant profile when model slot changes
  useEffect(() => {
    const slotIndex = appState.serverSetting?.serverSetting?.modelSlotIndex;
    if (slotIndex === undefined || slotIndex === -1) return;

    try {
      const storedBindingsStr = localStorage.getItem('model_formant_bindings') || '{}';
      const storedBindings = JSON.parse(storedBindingsStr);
      const binding = storedBindings[slotIndex];

      if (binding) {
        const storedProfilesStr = localStorage.getItem('formant_profiles') || '[]';
        const storedProfiles = JSON.parse(storedProfilesStr);

        const targetProfile = storedProfiles.find((p: any) => p.id === binding.targetProfileId);
        const inputProfile = storedProfiles.find((p: any) => p.id === binding.inputProfileId);

        if (targetProfile && inputProfile) {
          appState.serverSetting.updateServerSettings({
            ...appState.serverSetting.serverSetting,
            tran: binding.recommendedPitch !== undefined ? binding.recommendedPitch : appState.serverSetting.serverSetting.tran,
            formantShift: binding.recommendedFormantShift !== undefined ? binding.recommendedFormantShift : appState.serverSetting.serverSetting.formantShift,
            formantProfileActive: true,
            formantProfileTargetEnvelope: JSON.stringify(targetProfile.envelope),
            formantProfileTargetSr: targetProfile.sr,
            formantProfileInputEnvelope: JSON.stringify(inputProfile.envelope),
            formantProfileInputSr: inputProfile.sr,
            formantProfileStrength: 0.35
          });
          console.log(`Auto-loaded formant profiles for model slot ${slotIndex}`);
        }
      } else {
        if (appState.serverSetting?.serverSetting?.formantProfileActive) {
          appState.serverSetting.updateServerSettings({
            ...appState.serverSetting.serverSetting,
            formantProfileActive: false
          });
        }
      }
    } catch (e) {
      console.error('Error auto-restoring formant profile:', e);
    }
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