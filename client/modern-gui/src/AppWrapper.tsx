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
  const [fadeOut, setFadeOut] = React.useState<boolean>(false);

  // ---------------- Functions ----------------

  // Handle welcome modal completion
  const handleWelcomeComplete = async () => {
    setFadeOut(true);
    setTimeout(() => {
      setShowWelcome(false);
    }, 300); // 300ms matches the transition duration
  };

  // ---------------- Render ----------------

  return (
    <>
      <AppContextProvider>
        <UIContextProvider>
          <App />
        </UIContextProvider>
      </AppContextProvider>

      {showWelcome && (
        <>
          {/* Welcome screen background gradient overlay */}
          <div
            className={`fixed inset-0 z-30 w-screen h-screen bg-surface transition-all duration-300 ease-in-out ${
              fadeOut ? 'opacity-0 scale-95 pointer-events-none' : 'opacity-100'
            }`}
          >
            {/* Subtle primary radial gradient glow at the top */}
            <div
              className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-primary to-transparent opacity-15 pointer-events-none"
            />
          </div>
          <GenericModal
            isOpen={true}
            transparent={true}
            onClose={handleWelcomeComplete}
            title="AuraEcho"
            className={`transition-all duration-300 ease-in-out ${
              fadeOut ? 'opacity-0 scale-95 pointer-events-none' : 'opacity-100'
            }`}
            primaryButton={{
              text: "Continue",
              onClick: handleWelcomeComplete
            }}
          >
            <WelcomeModal />
          </GenericModal>
        </>
      )}
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