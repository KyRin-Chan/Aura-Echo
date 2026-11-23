import React, { useState, useEffect, useCallback, useRef, InputHTMLAttributes } from 'react';

interface DebouncedSliderProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'onChange' | 'value'> {
  value: number; // Controlled component value
  onChange: (value: number) => void; // Debounced change callback
  onImmediateChange?: (value: number) => void; // Optional: Callback for immediate value change
  // id, name, min, max, step, className, disabled, etc., will be inherited from InputHTMLAttributes
}

export type DebouncedFn<T extends (...args: any[]) => void> = ((...args: Parameters<T>) => void) & {
  cancel: () => void;
};

export const useDebouncedCallback = <T extends (...args: any[]) => void>(
  callback: T,
  delay: number,
  options?: { flushOnUnmount?: boolean }
): DebouncedFn<T> => {
  const timeoutRef = useRef<number>();
  const pendingArgsRef = useRef<Parameters<T> | null>(null);
  const pendingCallbackRef = useRef<T | null>(null);
  const latestCallbackRef = useRef<T>(callback);
  const flushOnUnmount = Boolean(options?.flushOnUnmount);
  const flushOnUnmountRef = useRef<boolean>(flushOnUnmount);

  latestCallbackRef.current = callback;
  flushOnUnmountRef.current = flushOnUnmount;

  const cancel = useCallback(() => {
    window.clearTimeout(timeoutRef.current);
    pendingArgsRef.current = null;
    pendingCallbackRef.current = null;
  }, []);

  const debouncedFn = useCallback((...args: Parameters<T>) => {
    window.clearTimeout(timeoutRef.current);
    if (delay <= 0) {
      pendingArgsRef.current = null;
      pendingCallbackRef.current = null;
      latestCallbackRef.current(...args);
      return;
    }
    pendingArgsRef.current = args;
    pendingCallbackRef.current = callback;
    timeoutRef.current = window.setTimeout(() => {
      const pendingArgs = pendingArgsRef.current;
      const pendingCallback = pendingCallbackRef.current;
      pendingArgsRef.current = null;
      pendingCallbackRef.current = null;
      if (pendingArgs !== null) {
        const target = flushOnUnmountRef.current ? latestCallbackRef.current : pendingCallback;
        target?.(...pendingArgs);
      }
    }, delay);
  }, [callback, delay]) as DebouncedFn<T>;

  debouncedFn.cancel = cancel;

  useEffect(
    () => () => {
      window.clearTimeout(timeoutRef.current);
      const pendingArgs = pendingArgsRef.current;
      pendingArgsRef.current = null;
      pendingCallbackRef.current = null;
      if (flushOnUnmountRef.current && pendingArgs !== null) {
        latestCallbackRef.current(...pendingArgs);
      }
    },
    []
  );

  return debouncedFn;
};

const DebouncedSlider: React.FC<DebouncedSliderProps> = ({
  value,
  onChange,
  onImmediateChange,
  ...rest
}) => {
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
    if (onImmediateChange) onImmediateChange(val);
    debouncedChange(val);
  };

  return (
    <input
      type="range"
      {...rest} // Spread all other native input attributes
      value={internalValue} // Controlled internal value
      onChange={handleChange} // Internal handler
    />
  );
};

export default DebouncedSlider;
