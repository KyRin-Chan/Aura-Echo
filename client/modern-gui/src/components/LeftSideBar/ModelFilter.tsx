import { ClientState, ModelSlotUnion, RVCModelSlot, VoiceChangerType } from "@dannadori/voice-changer-client-js";
import { faFilter, faSearch, faSort, faTimes, faArrowUpAZ, faArrowDownAZ } from "@fortawesome/free-solid-svg-icons";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { useMemo, useState, useEffect } from "react";
import { CSS_CLASSES } from "../../styles/constants";
import MD3Select from "../Helpers/MD3Select";
import { t } from "../../locales";

interface ModelFilterProps {
  appState: ClientState;
  setFilteredAndSortedModels: (models: RVCModelSlot[]) => void;
}

type SortOption = 'slot' | 'name';
type SampleRateFilter = number | 'All';

const sortOptions = [
  { value: 'slot', label: 'Slot' },
  { value: 'name', label: 'Name' }
];

function ModelFilter({ appState, setFilteredAndSortedModels }: ModelFilterProps) {
  // ---------------- State ----------------
  const [searchTerm, setSearchTerm] = useState('');
  const [currentSort, setCurrentSort] = useState<SortOption>('slot');
  const [typeVersionFilter, setTypeVersionFilter] = useState<string>('All');
  const [rateFilter, setRateFilter] = useState<SampleRateFilter>('All');
  const [embedderFilter, setEmbedderFilter] = useState<string>('All');
  const [isSortFilterVisible, setIsSortFilterVisible] = useState(false);
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');

  // ---------------- Hooks ----------------

  // Extract and validate RVC models from server state
  const localModels: RVCModelSlot[] = useMemo(() => {
    if (appState.serverSetting?.serverSetting?.modelSlots) {
      return appState.serverSetting.serverSetting.modelSlots
        .filter(
          (slot: ModelSlotUnion): slot is RVCModelSlot =>
            slot.voiceChangerType === VoiceChangerType.RVC &&
            slot.name !== '' &&
            typeof slot.slotIndex === 'number'
        )
        .map(
          (slot: RVCModelSlot): RVCModelSlot => ({
            ...slot,
            slotIndex: slot.slotIndex as number
          })
        );
    }
    return [];
  }, [appState.serverSetting?.serverSetting?.modelSlots]);

  // Generate dynamic filter options based on available models
  const modelTypeVersionOptions = useMemo(() => {
    const types = new Set<string>();
    localModels.forEach((model) => {
      if (model.voiceChangerType) {
        types.add(`${model.voiceChangerType}`);
      }
    });
    return ['All', ...Array.from(types).sort()];
  }, [localModels]);

  // Extract unique sample rates from models and sort numerically
  const sampleRateOptions = useMemo(() => {
    const rates = new Set<number>();
    localModels.forEach((model) => {
      if (model.samplingRate) {
        rates.add(model.samplingRate);
      }
    });
    return ['All' as SampleRateFilter, ...Array.from(rates).sort((a, b) => a - b)];
  }, [localModels]);

  // Generate list of available embedder types from models
  const embedderOptions = useMemo(() => {
    const embedders = new Set<string>();
    localModels.forEach((model) => {
      if (model.embedder) {
        embedders.add(model.embedder);
      }
    });
    return ['All', ...Array.from(embedders).sort()];
  }, [localModels]);

  // Main processing logic: applies all filters and sorting
  const filteredAndSortedModels = useMemo(() => {
    let processedModels = [...localModels];

    if (searchTerm) {
      processedModels = processedModels.filter((model) =>
        model.name.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    if (typeVersionFilter !== 'All') {
      processedModels = processedModels.filter(
        (model) => model.voiceChangerType && `${model.voiceChangerType}` === typeVersionFilter
      );
    }

    if (rateFilter !== 'All') {
      processedModels = processedModels.filter((model) => model.samplingRate === rateFilter);
    }

    if (embedderFilter !== 'All') {
      processedModels = processedModels.filter((model) => model.embedder === embedderFilter);
    }

    processedModels.sort((a, b) => {
      let comparison = 0;
      if (currentSort === 'slot') {
        comparison = a.slotIndex - b.slotIndex;
      } else if (currentSort === 'name') {
        comparison = a.name.localeCompare(b.name);
      }
      return sortDirection === 'asc' ? comparison : -comparison;
    });

    return processedModels;
  }, [localModels, searchTerm, currentSort, typeVersionFilter, rateFilter, embedderFilter, sortDirection]);

  // Update parent component with processed model list
  useEffect(() => {
    setFilteredAndSortedModels(filteredAndSortedModels);
  }, [filteredAndSortedModels, setFilteredAndSortedModels]);

  // ---------------- Render ----------------

  const typeOptions = modelTypeVersionOptions.map((opt) => ({ value: opt, label: opt }));
  const rateOptions = sampleRateOptions.map((opt) => ({
    value: opt === 'All' ? 'All' : opt,
    label: opt === 'All' ? 'All' : `${opt / 1000}kHz`
  }));
  const embedderFilterOptions = embedderOptions.map((opt) => ({
    value: opt,
    label:
      opt === 'hubert_base'
        ? 'ContentVec / Hubert'
        : opt === 'spin_base'
        ? 'SPIN'
        : opt === 'spin_v2'
        ? 'SPIN V2'
        : opt
  }));

  return (
    <>
      {/* Search Field */}
      <div className="relative mb-2.5">
        <input
          type="search"
          placeholder={t('searchModelsPlaceholder')}
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="w-full p-2.5 pr-10 rounded-full focus:outline-none focus:ring-1 focus:ring-primary text-xs bg-surface-container border border-outline-variant text-on-surface [&::-webkit-search-cancel-button]:appearance-none [&::-webkit-search-cancel-button]:hidden"
        />
        <FontAwesomeIcon
          icon={faSearch}
          className="absolute right-3.5 top-1/2 transform -translate-y-1/2 text-on-surface-variant/50"
        />
      </div>

      {/* Filter and Sort Toggle Button */}
      <div className="mb-2.5">
        <button
          onClick={() => setIsSortFilterVisible(!isSortFilterVisible)}
          className="w-full flex items-center justify-between px-3 py-2 text-xs font-semibold rounded-full bg-surface-container border border-outline-variant text-on-surface hover:bg-primary/8 transition-all"
        >
          <span>
            <FontAwesomeIcon icon={faFilter} className="mr-2 text-primary" /> {t('filterAndSort')}
          </span>
          <FontAwesomeIcon icon={isSortFilterVisible ? faTimes : faSort} className="text-on-surface-variant" />
        </button>
      </div>

      {/* Filter and Sort Controls - Conditional Rendering */}
      {isSortFilterVisible && (
        <div className="space-y-3.5 mb-3.5 p-3.5 rounded-lg border border-outline-variant bg-surface-container-low animate-fadeIn">
          {/* Sort Controls */}
          <div className="space-y-2 pb-2.5 border-b border-outline-variant/30">
            <label className="text-[10px] font-bold text-on-surface-variant flex items-center uppercase tracking-wider">
              <FontAwesomeIcon icon={faSort} className="mr-1.5 text-primary" />
              {t('sortBy')}
            </label>
            <div className="flex gap-2 items-center">
              <div className="flex-1">
                <MD3Select
                  id="sortOption"
                  value={currentSort}
                  onChange={(e) => setCurrentSort(e.target.value as SortOption)}
                  options={sortOptions}
                />
              </div>
              <button
                onClick={() => setSortDirection((prev) => (prev === 'asc' ? 'desc' : 'asc'))}
                className="p-2 rounded-full border border-outline-variant text-on-surface hover:bg-primary/8 transition-all"
                title={sortDirection === 'asc' ? t('sortDescending') : t('sortAscending')}
              >
                <FontAwesomeIcon icon={sortDirection === 'asc' ? faArrowUpAZ : faArrowDownAZ} className="text-xs" />
              </button>
            </div>
          </div>

          {/* Filter Controls */}
          <div className="space-y-2">
            <p className="text-[10px] font-bold text-on-surface-variant flex items-center uppercase tracking-wider">
              <FontAwesomeIcon icon={faFilter} className="mr-1.5 text-primary" />
              {t('filterBy')}
            </p>
            <div className="space-y-2">
              <MD3Select
                id="typeVersionFilter"
                label="Type"
                value={typeVersionFilter}
                onChange={(e) => setTypeVersionFilter(e.target.value)}
                options={typeOptions}
              />
              <MD3Select
                id="rateFilter"
                label="Rate"
                value={rateFilter === 'All' ? 'All' : rateFilter}
                onChange={(e) =>
                  setRateFilter(e.target.value === 'All' ? 'All' : (Number(e.target.value) as SampleRateFilter))
                }
                options={rateOptions}
              />
              <MD3Select
                id="embedderFilter"
                label="Embedder"
                value={embedderFilter}
                onChange={(e) => setEmbedderFilter(e.target.value)}
                options={embedderFilterOptions}
              />
            </div>
          </div>
        </div>
      )}
    </>
  );
}

export default ModelFilter;