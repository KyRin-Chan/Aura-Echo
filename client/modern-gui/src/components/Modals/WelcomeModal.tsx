import React from 'react';

const WelcomeModal: React.FC = () => {
  return (
    <div className="text-center pt-2">
      <div className="mb-6">
        <h2 className="text-xl font-bold text-on-surface mb-2">
          Welcome to AuraEcho
        </h2>
        <p className="text-xs text-on-surface-variant leading-relaxed">
          Click on &quot;Continue&quot; to start the application and initialize the audio engine.
        </p>
      </div>

      <div className="mb-2 p-4 bg-surface-container-low border border-outline-variant rounded-xl">
        <p className="text-xs text-on-surface-variant leading-relaxed">
          <strong className="text-primary font-bold">Note:</strong> For optimal functionality, this app requires access to your audio system.
        </p>
      </div>
    </div>
  );
};

export default WelcomeModal;