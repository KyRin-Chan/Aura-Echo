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