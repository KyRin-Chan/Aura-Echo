import { JSX } from 'react';
import { RVCModelSlot } from '@dannadori/voice-changer-client-js';
import MD3Checkbox from '../../../Helpers/MD3Checkbox';
import MD3Slider from '../../../Helpers/MD3Slider';
import { t } from '../../../../locales';

interface ModelMergeInfo {
  slot: RVCModelSlot;
  percentage: number;
}

interface MergeModelListProps {
  models: RVCModelSlot[];
  selectedModels: ModelMergeInfo[];
  onModelToggle: (slot: RVCModelSlot) => void;
  onPercentageChange: (slotIndex: number, percentage: number) => void;
  modelDir: string;
}

function MergeModelList({
  models,
  selectedModels,
  onModelToggle,
  onPercentageChange,
  modelDir
}: MergeModelListProps): JSX.Element {
  // ---------------- Functions ----------------

  // Check if a model is selected
  const isModelSelected = (slot: RVCModelSlot) => {
    return selectedModels.some((m) => m.slot.slotIndex === slot.slotIndex);
  };

  // Get model percentage
  const getModelPercentage = (slot: RVCModelSlot) => {
    const found = selectedModels.find((m) => m.slot.slotIndex === slot.slotIndex);
    return found ? found.percentage : 50;
  };

  // Generate placeholder (inline SVG generation)
  const generatePlaceholder = (name: string): string => {
    const initial = name.charAt(0).toUpperCase();
    const hash = name.split('').reduce((a, b) => {
      a = (a << 5) - a + b.charCodeAt(0);
      return a & a;
    }, 0);
    const colors = [
      '#6750A4', // MD3 Purple
      '#625B71', // MD3 Secondary
      '#7D5260', // MD3 Tertiary
      '#386A20', // MD3 Green
      '#00639B', // MD3 Blue
      '#9C423E', // MD3 Red
      '#8F4E00', // MD3 Orange
      '#7F5600'  // MD3 Yellow
    ];
    const bgColor = colors[Math.abs(hash) % colors.length];

    const svgString = `
      <svg width="32" height="32" xmlns="http://www.w3.org/2000/svg">
        <rect width="100%" height="100%" fill="${bgColor}" rx="6"/>
        <text x="50%" y="55%" font-family="system-ui, sans-serif" font-weight="bold" font-size="14" 
              fill="#ffffff" text-anchor="middle" dy=".3em">${initial}</text>
      </svg>
    `;
    return `data:image/svg+xml;base64,${btoa(unescape(encodeURIComponent(svgString.trim())))}`;
  };

  // Handle model card click
  const handleModelCardClick = (model: RVCModelSlot) => {
    const isSelected = isModelSelected(model);
    if (!isSelected) {
      onModelToggle(model);
    }
  };

  // ---------------- Render ----------------

  return (
    <div className="space-y-3">
      <h4 className="text-sm font-bold text-on-surface uppercase tracking-wider pl-1">{t('availableModelsLabel')}</h4>

      {models.length === 0 ? (
        <div className="text-center py-12 text-on-surface-variant/60 italic bg-surface-container-low rounded-lg border border-outline-variant text-sm">
          <p>{t('noModelsMatchFilter')}</p>
        </div>
      ) : (
        <div className="space-y-2">
          {models.map((model) => {
            const isSelected = isModelSelected(model);
            const percentage = getModelPercentage(model);
            const icon =
              model.iconFile.length > 0
                ? '/model_dir/' + model.slotIndex + '/' + model.iconFile.split(/[\/\\]/).pop()
                : '';
            const placeholder = generatePlaceholder(model.name);

            return (
              <div
                key={model.slotIndex}
                className={`p-3 rounded-lg border transition-all duration-150 ${
                  isSelected
                    ? 'bg-primary/8 border-primary shadow-elevation-1'
                    : 'bg-surface-container-low border-outline-variant hover:border-outline hover:bg-primary/4 cursor-pointer'
                }`}
                onClick={() => handleModelCardClick(model)}
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-3">
                    <div onClick={(e) => e.stopPropagation()}>
                      <MD3Checkbox checked={isSelected} onChange={() => onModelToggle(model)} />
                    </div>
                    <img
                      src={icon.length > 0 ? icon : placeholder}
                      alt={model.name}
                      className="w-9 h-9 rounded-md object-cover flex-shrink-0 border border-outline-variant/30"
                    />
                    <div>
                      <div className="font-semibold text-on-surface text-sm">
                        {model.name || `Model ${model.slotIndex}`}
                      </div>
                      <div className="text-[11px] text-on-surface-variant">
                        {model.embedder === 'hubert_base'
                          ? 'ContentVec / Hubert'
                          : model.embedder === 'spin_base'
                          ? 'SPIN'
                          : model.embedder === 'spin_v2'
                          ? 'SPIN V2'
                          : model.embedder || 'Unknown'}{' '}
                        • {model.samplingRate || 'Unknown'} Hz • {model.voiceChangerType || 'RVC'}
                        {model.version || '1'}
                      </div>
                    </div>
                  </div>

                  {isSelected && (
                    <div className="text-sm font-semibold text-primary ml-2">{percentage}%</div>
                  )}
                </div>

                {isSelected && (
                  <div className="mt-3.5 px-1.5" onClick={(e) => e.stopPropagation()}>
                    <MD3Slider
                      min={0}
                      max={100}
                      step={1}
                      value={percentage}
                      onChange={(val) => onPercentageChange(model.slotIndex, val)}
                    />
                    <div className="flex justify-between text-[10px] text-on-surface-variant font-semibold mt-1">
                      <span>0%</span>
                      <span>50%</span>
                      <span>100%</span>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {selectedModels.length > 0 && (
        <div className="pt-3 border-t border-outline-variant flex justify-between items-center text-xs text-on-surface-variant font-semibold px-1">
          <div>{t('selectedCountLabel')}<span className="text-primary font-bold">{selectedModels.length}</span></div>
          <div>{t('totalWeightLabel')}<span className="text-primary font-bold">{selectedModels.reduce((sum, m) => sum + m.percentage, 0)}%</span></div>
        </div>
      )}
    </div>
  );
}

export default MergeModelList;