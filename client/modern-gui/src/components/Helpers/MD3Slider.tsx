import React, { useState, useEffect, useRef, useCallback } from 'react';
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
  flushOnUnmount?: boolean;
  debounceMs?: number;
  commitOnRelease?: boolean;
  onDragStart?: () => void;
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
  valueFormatter,
  flushOnUnmount = false,
  debounceMs = 300,
  commitOnRelease = false,
  onDragStart
}) => {
  const minNum = typeof min === 'string' ? parseFloat(min) : min;
  const maxNum = typeof max === 'string' ? parseFloat(max) : max;

  const [internalValue, setInternalValue] = useState<number>(value);
  const isDraggingRef = useRef<boolean>(false);
  const hasPendingReleaseCommitRef = useRef<boolean>(false);
  const latestValueRef = useRef<number>(value);

  useEffect(() => {
    setInternalValue(value);
    latestValueRef.current = value;
  }, [value]);

  const debouncedChange = useDebouncedCallback(
    (val: number) => {
      onChange(val);
    },
    debounceMs,
    { flushOnUnmount }
  );

  const handleDragStart = useCallback(() => {
    if (disabled || !commitOnRelease) return;
    if (!isDraggingRef.current) {
      isDraggingRef.current = true;
      hasPendingReleaseCommitRef.current = false;
      debouncedChange.cancel();
      onDragStart?.();
    }
  }, [disabled, commitOnRelease, debouncedChange, onDragStart]);

  useEffect(() => {
    if (!commitOnRelease) return;

    const handleRelease = () => {
      if (!isDraggingRef.current) return;
      isDraggingRef.current = false;
      if (hasPendingReleaseCommitRef.current) {
        hasPendingReleaseCommitRef.current = false;
        debouncedChange(latestValueRef.current);
      }
    };

    window.addEventListener('pointerup', handleRelease);
    window.addEventListener('pointercancel', handleRelease);
    window.addEventListener('mouseup', handleRelease);
    window.addEventListener('touchend', handleRelease);
    window.addEventListener('touchcancel', handleRelease);
    return () => {
      window.removeEventListener('pointerup', handleRelease);
      window.removeEventListener('pointercancel', handleRelease);
      window.removeEventListener('mouseup', handleRelease);
      window.removeEventListener('touchend', handleRelease);
      window.removeEventListener('touchcancel', handleRelease);
    };
  }, [commitOnRelease, debouncedChange]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.valueAsNumber;
    setInternalValue(val);
    latestValueRef.current = val;
    if (onImmediateChange) {
      onImmediateChange(val);
    }
    if (commitOnRelease && isDraggingRef.current) {
      hasPendingReleaseCommitRef.current = true;
    } else {
      debouncedChange(val);
    }
  };

  const range = maxNum - minNum;
  const rawPercent = range > 0 ? ((internalValue - minNum) / range) * 100 : 0;
  const percent = Math.min(100, Math.max(0, rawPercent));

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
          onPointerDown={handleDragStart}
          onMouseDown={handleDragStart}
          onTouchStart={handleDragStart}
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
