import React from 'react';

interface LoadingScreenProps {
  message?: string;
}

const LoadingScreen: React.FC<LoadingScreenProps> = ({ message }) => (
  <div className="fixed inset-0 bg-scrim/40 backdrop-blur-sm flex justify-center items-center z-50 p-4 animate-fadeIn">
    <div className="bg-surface-container-high p-6 rounded-3xl border border-outline-variant shadow-elevation-3 flex flex-col items-center justify-center max-w-xs w-full">
      <div className="w-12 h-12 border-4 border-primary border-t-transparent rounded-full animate-spin mb-4"></div>
      <p className="text-on-surface text-sm font-bold">Loading...</p>
      {message && <p className="text-on-surface-variant text-xs mt-1.5 text-center leading-relaxed">{message}</p>}
    </div>
  </div>
);

export default LoadingScreen;