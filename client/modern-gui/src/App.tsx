import LeftSidebar from './components/LeftSideBar/Sidebar';
import BottomBar from './components/BottomBar/BottomBar';
import { JSX, useEffect, ReactNode } from 'react';
import GenericModal from './components/Modals/GenericModal';
import { PassthroughConfirmModalProps } from './components/BottomBar/Modals/PassthroughConfirmModal';
import { RVCModelSlot } from '@dannadori/voice-changer-client-js';
import MainContent from './components/MainContent';
import { useAppState } from './context/AppContext';
import { useUIContext } from './context/UIContext';
import { getFormantBinding, getFormantProfile } from './utils/formantStorage';

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
      const binding = getFormantBinding(slotIndex);

      if (binding && binding.targetProfileId && binding.inputProfileId) {
        const targetProfile = getFormantProfile(binding.targetProfileId);
        const inputProfile = getFormantProfile(binding.inputProfileId);

        if (targetProfile && inputProfile) {
          const isActive = binding.active !== undefined ? Boolean(binding.active) : true;
          appState.serverSetting.updateServerSettings({
            ...appState.serverSetting.serverSetting,
            tran: binding.recommendedPitch !== undefined ? binding.recommendedPitch : appState.serverSetting.serverSetting.tran,
            formantShift: binding.recommendedFormantShift !== undefined ? binding.recommendedFormantShift : appState.serverSetting.serverSetting.formantShift,
            formantProfileActive: isActive,
            formantProfileTargetEnvelope: JSON.stringify(targetProfile.envelopes || targetProfile.envelope),
            formantProfileTargetSr: targetProfile.sr,
            formantProfileInputEnvelope: JSON.stringify(inputProfile.envelopes || inputProfile.envelope),
            formantProfileInputSr: inputProfile.sr,
            formantProfileStrength: binding.strength !== undefined ? binding.strength : (appState.serverSetting.serverSetting.formantProfileStrength ?? 0.35)
          });
          console.log(`Auto-loaded formant profiles for model slot ${slotIndex}, active: ${isActive}`);
        } else {
          if (appState.serverSetting?.serverSetting?.formantProfileActive) {
            appState.serverSetting.updateServerSettings({
              ...appState.serverSetting.serverSetting,
              formantProfileActive: false
            });
          }
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