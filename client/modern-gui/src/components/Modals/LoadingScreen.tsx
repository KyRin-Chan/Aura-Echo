import React from 'react';

interface LoadingScreenProps {
  message?: string;
}

const LoadingScreen: React.FC<LoadingScreenProps> = ({ message }) => (
  <div className="fixed inset-0 bg-black bg-opacity-75 backdrop-blur-md flex flex-col justify-center items-center z-50 p-4">
    <div className="w-16 h-16 border-4 rounded-full animate-spin mb-4" style={{ borderColor: 'var(--macaron-mint)', borderTopColor: 'transparent' }}></div>
    <p className="text-white text-lg font-semibold">Loading...</p>
    <p className="text-slate-300 text-sm">{message}</p>
  </div>
);

export default LoadingScreen;