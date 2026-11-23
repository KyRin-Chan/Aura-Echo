import React from 'react';

interface MD3SelectOption {
  value: string | number;
  label: string;
}

interface MD3SelectProps {
  value: string | number;
  onChange: (e: React.ChangeEvent<HTMLSelectElement>) => void;
  options: MD3SelectOption[];
  label?: string;
  disabled?: boolean;
  id?: string;
  className?: string;
}

const MD3Select: React.FC<MD3SelectProps> = ({
  value,
  onChange,
  options,
  label,
  disabled = false,
  id,
  className = ''
}) => {
  return (
    <div className={`relative w-full ${className}`}>
      {label && (
        <label
          htmlFor={id}
          className={`absolute left-3 top-1.5 text-[10px] font-semibold leading-none tracking-wide pointer-events-none bg-transparent text-on-surface-variant truncate max-w-[calc(100%-2.5rem)] transition-all ${
            disabled ? 'opacity-50' : ''
          }`}
        >
          {label}
        </label>
      )}
      <select
        id={id}
        value={value}
        onChange={onChange}
        disabled={disabled}
        className={`w-full pr-9 bg-surface-container/60 border rounded-sm focus:outline-none focus:ring-1 focus:ring-primary focus:border-primary text-on-surface text-sm transition-all duration-200 cursor-pointer appearance-none ${
          label ? 'h-[50px] pt-4 pb-1 px-3' : 'h-[44px] py-2 px-3'
        } ${
          disabled
            ? 'border-outline-variant/40 text-on-surface/30 cursor-not-allowed'
            : 'border-outline-variant hover:border-outline'
        }`}
      >
        {options.map((opt) => (
          <option key={opt.value} value={opt.value} className="bg-surface-container-high text-on-surface">
            {opt.label}
          </option>
        ))}
      </select>
      <div className={`absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none transition-transform duration-200 ${
        disabled ? 'text-on-surface/30' : 'text-on-surface-variant'
      }`}>
        <svg className="w-4 h-4 fill-current" viewBox="0 0 20 20">
          <path d="M5.516 7.548c.436-.446 1.043-.481 1.576 0L10 10.405l2.908-2.857c.533-.481 1.141-.446 1.576 0 .436.445.408 1.197 0 1.615l-3.695 3.63a1.08 1.08 0 01-1.576 0L4.824 9.163c-.408-.418-.436-1.17 0-1.615z" />
        </svg>
      </div>
    </div>
  );
};

export default MD3Select;
