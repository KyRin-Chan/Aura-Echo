import React from 'react';

interface MD3RadioOption {
  value: string | number;
  label: string;
}

interface MD3RadioGroupProps {
  options: MD3RadioOption[];
  value: string | number;
  onChange: (value: any) => void;
  name: string;
  disabled?: boolean;
  direction?: 'row' | 'column';
  label?: string;
}

const MD3RadioGroup: React.FC<MD3RadioGroupProps> = ({
  options,
  value,
  onChange,
  name,
  disabled = false,
  direction = 'row',
  label
}) => {
  return (
    <div className="flex flex-col space-y-2">
      {label && (
        <span className={`text-xs font-semibold text-on-surface-variant ${disabled ? 'opacity-50' : ''}`}>
          {label}
        </span>
      )}
      <div className={`flex ${direction === 'row' ? 'flex-row space-x-6' : 'flex-col space-y-3'}`}>
        {options.map((opt) => {
          const isSelected = opt.value === value;
          const handleSelect = () => {
            if (!disabled) {
              onChange(opt.value);
            }
          };

          return (
            <div
              key={opt.value}
              className={`flex items-center select-none ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
              onClick={handleSelect}
            >
              <div className="relative flex items-center justify-center w-5 h-5 rounded-full border-2 transition-all duration-150"
                style={{
                  borderColor: isSelected ? 'var(--md-sys-color-primary)' : 'var(--md-sys-color-on-surface-variant)'
                }}
              >
                {isSelected && (
                  <div className="w-2.5 h-2.5 rounded-full bg-primary animate-[scale-in_0.15s_ease-out]" />
                )}
              </div>
              <span className="ml-3 text-sm font-medium text-on-surface">
                {opt.label}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default MD3RadioGroup;
