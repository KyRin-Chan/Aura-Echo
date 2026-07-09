import React from 'react';
import { AppContextProvider } from './context/AppContext';
import App from './App';
import ParticleBackground from './components/Helpers/ParticleBackground';
import GenericModal from './components/Modals/GenericModal';
import WelcomeModal from './components/Modals/WelcomeModal';
import { UIContextProvider } from './context/UIContext';
import { ThemeProvider, useThemeContext } from './context/ThemeContext';
import { AppRootProvider } from './context/AppRootProvider';

const AppContent: React.FC = () => {
  // ---------------- State ----------------
  const [showWelcome, setShowWelcome] = React.useState<boolean>(true);
  const { theme } = useThemeContext();

  // ---------------- Functions ----------------

  // Handle welcome modal completion
  const handleWelcomeComplete = async () => {
    setShowWelcome(false);
  };

  // Theme-based particle configuration
  const getParticleConfig = () => {
    if (theme === 'dark') {
      return {
        particleColor: "rgba(13, 242, 163, 0.4)", // Cyber-mint glowing particles
        backgroundColor: "#070913" // Space obsidian dark background
      };
    } else {
      return {
        particleColor: "rgba(58, 225, 165, 0.3)", // Soft mint particles
        backgroundColor: "#faf9f5" // Alabaster cream light background
      };
    }
  };

  const particleConfig = getParticleConfig();

  // ---------------- Render ----------------

  // Render welcome modal if showWelcome is true
  if (showWelcome) {
    return (
      <>
        <ParticleBackground
          zIndex={1}
          particleCount={100}
          particleColor={particleConfig.particleColor}
          backgroundColor={particleConfig.backgroundColor}
        />
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

      </>
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