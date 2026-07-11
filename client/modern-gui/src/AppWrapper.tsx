import React from 'react';
import { AppContextProvider } from './context/AppContext';
import App from './App';
import GenericModal from './components/Modals/GenericModal';
import WelcomeModal from './components/Modals/WelcomeModal';
import { UIContextProvider } from './context/UIContext';
import { ThemeProvider } from './context/ThemeContext';
import { AppRootProvider } from './context/AppRootProvider';

const AppContent: React.FC = () => {
  // ---------------- State ----------------
  const [showWelcome, setShowWelcome] = React.useState<boolean>(true);

  // ---------------- Functions ----------------

  // Handle welcome modal completion
  const handleWelcomeComplete = async () => {
    setShowWelcome(false);
  };

  // ---------------- Render ----------------

  // Render welcome modal if showWelcome is true
  if (showWelcome) {
    return (
      <div className="fixed inset-0 w-screen h-screen bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-primary/15 via-surface to-surface flex justify-center items-center">
        <GenericModal
          isOpen={true}
          transparent={true}
          onClose={handleWelcomeComplete}
          title="AuraEcho"
          primaryButton={{
            text: "Continue",
            onClick: handleWelcomeComplete
          }}
        >
          <WelcomeModal />
        </GenericModal>
      </div>
    );
  }

  // Render app if showWelcome is false
  return (
    <>
      <AppContextProvider>
        <UIContextProvider>
          <App />
        </UIContextProvider>
      </AppContextProvider>
    </>
  );
};

// Wrapper around the AppContent with relevant context providers
export const AppWrapper: React.FC = () => {
  // ---------------- Render ----------------

  return (
    <ThemeProvider>
      <AppRootProvider>
        <AppContent />
      </AppRootProvider>
    </ThemeProvider>
  );
};