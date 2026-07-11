import React, { useState, useEffect } from 'react';
import { useDebouncedCallback } from './DebouncedSlider';

interface MD3SliderProps {
  value: number;
  onChange: (value: number) => void;
  onImmediateChange?: (value: number) => void;
  min?: number | string;
  max?: number | string;
  step?: number | string;
  disabled?: boolean;
  id?: string;
  className?: string;
  showValue?: boolean;
  valueFormatter?: (val: number) => string;
}

const MD3Slider: React.FC<MD3SliderProps> = ({
  value,
  onChange,
  onImmediateChange,
  min = 0,
  max = 100,
  step = 1,
  disabled = false,
  id,
  className = '',
  showValue = false,
  valueFormatter
}) => {
  const minNum = typeof min === 'string' ? parseFloat(min) : min;
  const maxNum = typeof max === 'string' ? parseFloat(max) : max;

  const [internalValue, setInternalValue] = useState<number>(value);

  useEffect(() => {
    setInternalValue(value);
  }, [value]);

  const debouncedChange = useDebouncedCallback((val: number) => {
    onChange(val);
  }, 300);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.valueAsNumber;
    setInternalValue(val);
    if (onImmediateChange) {
      onImmediateChange(val);
    }
    debouncedChange(val);
  };

  const percent = ((internalValue - minNum) / (maxNum - minNum)) * 100;

  // Track style with dynamic fill gradient
  const trackStyle = {
    background: disabled
      ? `linear-gradient(to right, var(--md-sys-color-outline) 0%, var(--md-sys-color-outline) ${percent}%, var(--md-sys-color-surface-container) ${percent}%, var(--md-sys-color-surface-container) 100%)`
      : `linear-gradient(to right, var(--md-sys-color-primary) 0%, var(--md-sys-color-primary) ${percent}%, var(--md-sys-color-surface-container-highest) ${percent}%, var(--md-sys-color-surface-container-highest) 100%)`
  };

  const displayVal = valueFormatter ? valueFormatter(internalValue) : internalValue.toString();

  return (
    <div className={`w-full flex flex-col space-y-1 ${className}`}>
      <div className="flex items-center space-x-4">
        <input
          id={id}
          type="range"
          min={min}
          max={max}
          step={step}
          value={internalValue}
          onChange={handleChange}
          disabled={disabled}
          style={trackStyle}
          className={`w-full h-1 rounded-full appearance-none cursor-pointer focus:outline-none transition-all duration-150 ${
            disabled ? 'opacity-50 cursor-not-allowed' : ''
          }`}
        />
        {showValue && (
          <span className={`text-xs font-semibold select-none w-12 text-right ${disabled ? 'text-on-surface-variant/50' : 'text-on-surface-variant'}`}>
            {displayVal}
          </span>
        )}
      </div>
    </div>
  );
};

export default MD3Slider;
