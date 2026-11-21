import { JSX } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faFilter } from '@fortawesome/free-solid-svg-icons';
import { CSS_CLASSES } from '../../../../styles/constants';
import { ModelInfoDict } from '@dannadori/voice-changer-client-js';
import MD3Select from '../../../Helpers/MD3Select';
import { t } from '../../../../locales';

interface MergeFilterProps {
  embedders: ModelInfoDict;
  sampleRate: number;
  setSampleRate: (sampleRate: number) => void;
  selectedEmbedder: string;
  setSelectedEmbedder: (embedder: string) => void;
  selectedVocoder: string;
  setSelectedVocoder: (vocoder: string) => void;
  searchText: string;
  setSearchText: (searchText: string) => void;
  onFilterChange: () => void;
}

function MergeFilter({
  sampleRate,
  setSampleRate,
  selectedEmbedder,
  embedders,
  setSelectedEmbedder,
  selectedVocoder,
  setSelectedVocoder,
  searchText,
  setSearchText,
  onFilterChange
}: MergeFilterProps): JSX.Element {
  // ---------------- States ----------------

  const sampleRates = [32000, 40000, 48000];
  const vocoderOptions = [
    { value: 'embedded', label: 'NSF-HiFiGAN' },
    { value: 'refinegan', label: 'RefineGAN' }
  ];

  // ---------------- Handlers ----------------

  // Handle sample rate change
  const handleSampleRateChange = (newSampleRate: number) => {
    setSampleRate(newSampleRate);
    onFilterChange();
  };

  // Handle embedder change
  const handleEmbedderChange = (newEmbedder: string) => {
    setSelectedEmbedder(newEmbedder);
    onFilterChange();
  };

  // Handle vocoder change
  const handleVocoderChange = (newVocoder: string) => {
    setSelectedVocoder(newVocoder);
    onFilterChange();
  };

  // ---------------- Render ----------------

  const sampleRateOptions = sampleRates.map((rate) => ({
    value: rate,
    label: `${rate} Hz`
  }));

  const embedderOptions =
    Object.keys(embedders || {}).length === 0
      ? [{ value: '', label: t('noEmbeddersAvailable') }]
      : Object.entries(embedders || {}).map(([key, embedderInfo]) => ({
          value: key,
          label: embedderInfo.name
        }));

  return (
    <div className="space-y-4 p-4 bg-surface-container-low rounded-lg border border-outline-variant">
      <div className="flex justify-between items-center pb-2 border-b border-outline-variant/30">
        <h4 className="text-sm font-bold text-on-surface uppercase tracking-wider">{t('filterSettingsLabel')}</h4>
        <FontAwesomeIcon icon={faFilter} className="h-4 w-4 text-primary" />
      </div>

      <div className="space-y-4">
        <div>
          <label className={CSS_CLASSES.label}>{t('searchModelsLabel')}</label>
          <input
            type="text"
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
            placeholder={t('searchByModelNamePlaceholder')}
            className={CSS_CLASSES.input}
          />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <MD3Select
            id="sampleRateFilter"
            label={t('sampleRateLabel')}
            value={sampleRate}
            onChange={(e) => handleSampleRateChange(Number(e.target.value))}
            options={sampleRateOptions}
          />

          <MD3Select
            id="embedderFilter"
            label={t('embedderLabel')}
            value={selectedEmbedder}
            onChange={(e) => handleEmbedderChange(e.target.value)}
            options={embedderOptions}
            disabled={Object.keys(embedders || {}).length === 0}
          />

          <MD3Select
            id="vocoderFilter"
            label={t('vocoderLabel')}
            value={selectedVocoder}
            onChange={(e) => handleVocoderChange(e.target.value)}
            options={vocoderOptions}
          />
        </div>
      </div>
    </div>
  );
}

export default MergeFilter;