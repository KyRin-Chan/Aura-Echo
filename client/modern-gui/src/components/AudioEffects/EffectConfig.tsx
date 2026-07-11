import { JSX, useState, useEffect } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faQuestionCircle } from '@fortawesome/free-solid-svg-icons';
import { AudioEffect, AudioEffectParameterDefinition } from '@dannadori/voice-changer-client-js';
import { CSS_CLASSES } from '../../styles/constants';
import MD3Slider from '../Helpers/MD3Slider';
import MD3Switch from '../Helpers/MD3Switch';
import MD3Select from '../Helpers/MD3Select';
import { getEffectDefinition } from './serverEffectsUtils';

// UI type with index for client-side management
type AudioEffectWithIndex = AudioEffect & { index: number };

interface EffectConfigProps {
  effect: AudioEffectWithIndex | null;
  onParameterChange: (effectIndex: number, parameterId: string, value: number | boolean | string) => void;
  serverSchema?: any;
}

interface SliderParameterProps {
  paramKey: string;
  definition: AudioEffectParameterDefinition;
  value: number;
  onChange: (value: number) => void;
}

function SliderParameter({ paramKey, definition, value, onChange }: SliderParameterProps) {
  const [displayValue, setDisplayValue] = useState(value);

  // Update display value when parameter value changes from server
  useEffect(() => {
    setDisplayValue(value);
  }, [value]);

  const decimalPlaces = definition.step && definition.step < 1 ? 2 : 0;

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-1.5">
          <label className={CSS_CLASSES.label}>{definition.name}</label>
          {definition.description && (
            <FontAwesomeIcon
              icon={faQuestionCircle}
              className="h-3.5 w-3.5 text-on-surface-variant/40 hover:text-on-surface-variant cursor-help transition-colors"
              title={definition.description}
            />
          )}
        </div>
      </div>
      <MD3Slider
        min={definition.min || 0}
        max={definition.max || 1}
        step={definition.step || 0.01}
        value={value}
        onChange={onChange}
        onImmediateChange={setDisplayValue}
        showValue={true}
        valueFormatter={(val) =>
          `${val.toFixed(decimalPlaces)}${definition.unit ? ` ${definition.unit}` : ''}`
        }
      />
    </div>
  );
}

function EffectConfig({ effect, onParameterChange, serverSchema }: EffectConfigProps): JSX.Element {
  if (!effect) {
    return (
      <div className="flex flex-col h-full bg-surface-container-low p-4 rounded-md border border-outline-variant">
        <div className="flex items-center justify-between mb-3 pb-2 border-b border-outline-variant">
          <h5 className="font-semibold text-on-surface">Configuration</h5>
        </div>
        <div className="flex-1 flex items-center justify-center text-on-surface-variant/60 text-sm italic">
          Select an effect to configure parameters
        </div>
      </div>
    );
  }

  const effectDefinition = getEffectDefinition(effect.type, serverSchema);
  if (!effectDefinition) {
    return (
      <div className="flex flex-col h-full bg-surface-container-low p-4 rounded-md border border-outline-variant">
        <div className="flex items-center justify-between mb-3 pb-2 border-b border-outline-variant">
          <h5 className="font-semibold text-on-surface">Unknown Effect</h5>
        </div>
        <div className="flex-1 flex items-center justify-center text-on-surface-variant/60 text-sm italic">
          Effect definition not found for type: {effect.type}
        </div>
      </div>
    );
  }

  const handleChange = (paramKey: string, value: number | boolean | string) => {
    onParameterChange(effect.index, paramKey, value);
  };

  const renderParameter = (paramKey: string, definition: AudioEffectParameterDefinition) => {
    const value = effect.parameters[paramKey] ?? definition.defaultValue;

    switch (definition.type) {
      case 'slider':
        return (
          <SliderParameter
            key={paramKey}
            paramKey={paramKey}
            definition={definition}
            value={value as number}
            onChange={(value) => handleChange(paramKey, value)}
          />
        );

      case 'toggle':
        const boolValue = value as boolean;
        return (
          <div key={paramKey} className="flex items-center justify-between py-1.5">
            <div className="flex items-center space-x-1.5">
              <label className="text-sm font-medium text-on-surface-variant">
                {definition.name}
              </label>
              {definition.description && (
                <FontAwesomeIcon
                  icon={faQuestionCircle}
                  className="h-3.5 w-3.5 text-on-surface-variant/40 hover:text-on-surface-variant cursor-help transition-colors"
                  title={definition.description}
                />
              )}
            </div>
            <MD3Switch
              checked={boolValue}
              onChange={(checked) => handleChange(paramKey, checked)}
            />
          </div>
        );

      case 'select':
        const stringValue = value as string;
        const selectOptions =
          definition.options?.map((option) => ({
            value: option,
            label: option
          })) || [];

        return (
          <div key={paramKey} className="py-1">
            <MD3Select
              label={definition.name}
              value={stringValue}
              onChange={(e) => handleChange(paramKey, e.target.value)}
              options={selectOptions}
            />
            {definition.description && (
              <p className="text-[10px] text-on-surface-variant/70 mt-1 px-1">
                {definition.description}
              </p>
            )}
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="flex flex-col h-full bg-surface-container-low p-4 rounded-md border border-outline-variant">
      {/* Header */}
      <div className="flex items-center justify-between mb-4 pb-3 border-b border-outline-variant">
        <div>
          <h5 className="font-semibold text-on-surface text-base">{effectDefinition.name}</h5>
          <div className="flex items-center space-x-2 mt-1">
            <p className="text-xs text-on-surface-variant capitalize">{effect.type} Effect</p>
            <span
              className={`px-2.5 py-0.5 rounded-full text-xs font-semibold ${
                effect.channel === 'input'
                  ? 'bg-primary-container text-on-primary-container'
                  : 'bg-secondary-container text-on-secondary-container'
              }`}
            >
              {effect.channel} channel
            </span>
          </div>
        </div>
        <div
          className={`px-3 py-1 rounded-full text-xs font-semibold ${
            effect.enabled
              ? 'bg-primary-container text-on-primary-container'
              : 'bg-surface-container-highest text-on-surface-variant'
          }`}
        >
          {effect.enabled ? 'Enabled' : 'Disabled'}
        </div>
      </div>

      {/* Parameters */}
      <div className="flex-1 overflow-y-auto space-y-4 pr-1">
        {Object.entries(effectDefinition.parameters).map(([paramKey, definition]) =>
          renderParameter(paramKey, definition)
        )}
      </div>
    </div>
  );
}

export default EffectConfig;