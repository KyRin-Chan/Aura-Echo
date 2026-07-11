import React from 'react';

interface MD3SwitchProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
  label?: string;
  id?: string;
}

const MD3Switch: React.FC<MD3SwitchProps> = ({
  checked,
  onChange,
  disabled = false,
  label,
  id
}) => {
  const handleToggle = () => {
    if (!disabled) {
      onChange(!checked);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === ' ' || e.key === 'Enter') {
      e.preventDefault();
      handleToggle();
    }
  };

  return (
    <div className={`flex items-center select-none ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`} onClick={handleToggle}>
      <button
        id={id}
        role="switch"
        aria-checked={checked}
        disabled={disabled}
        onKeyDown={handleKeyDown}
        className={`relative inline-flex h-8 w-[52px] shrink-0 items-center rounded-full border-2 transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 ${
          checked
            ? 'bg-primary border-primary'
            : 'bg-surface-container-highest border-outline'
        } ${disabled ? 'cursor-not-allowed' : 'cursor-pointer'}`}
      >
        <span
          className={`pointer-events-none block rounded-full bg-white shadow-sm transition-transform duration-200 ease-in-out ${
            checked
              ? 'translate-x-[22px] w-6 h-6 bg-on-primary'
              : 'translate-x-[4px] w-4 h-4 bg-outline'
          }`}
        />
      </button>
      {label && (
        <span className="ml-3 text-sm font-medium text-on-surface">
          {label}
        </span>
      )}
    </div>
  );
};

export default MD3Switch;
