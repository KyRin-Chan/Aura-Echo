import React from 'react';
import { t } from '../../locales';

const WelcomeModal: React.FC = () => {
  return (
    <div className="text-center py-6 flex flex-col items-center select-none">
      <h2 className="text-4xl font-black text-on-surface tracking-widest mb-4">
        <span className="bg-gradient-to-r from-primary to-tertiary bg-clip-text text-transparent">AuraEcho</span>
      </h2>
      <p className="text-xs text-on-surface-variant tracking-wider max-w-xs leading-relaxed font-medium">
        {t('welcomeModalDesc')}
      </p>
    </div>
  );
};

export default WelcomeModal;