import React from 'react';

interface MD3CheckboxProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
  label?: string;
  id?: string;
}

const MD3Checkbox: React.FC<MD3CheckboxProps> = ({
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
    <div
      className={`flex items-center select-none ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
      onClick={handleToggle}
    >
      <div
        id={id}
        role="checkbox"
        aria-checked={checked}
        tabIndex={disabled ? -1 : 0}
        onKeyDown={handleKeyDown}
        className={`w-[18px] h-[18px] flex items-center justify-center rounded-[2px] border-2 transition-all duration-150 focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 ${
          checked
            ? 'bg-primary border-primary'
            : 'bg-transparent border-on-surface-variant'
        }`}
      >
        {checked && (
          <svg
            className="w-3 h-3 text-on-primary stroke-current stroke-2 fill-none"
            viewBox="0 0 16 16"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M3 8.5L6.5 12L13 4"
            />
          </svg>
        )}
      </div>
      {label && (
        <span className="ml-3 text-sm font-medium text-on-surface">
          {label}
        </span>
      )}
    </div>
  );
};

export default MD3Checkbox;
