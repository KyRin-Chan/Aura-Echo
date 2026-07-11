import { JSX, useState, useMemo } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faSearch, faFilter } from '@fortawesome/free-solid-svg-icons';
import { AudioChannel } from '@dannadori/voice-changer-client-js';
import { getAvailableEffectTypesFromServer } from './serverEffectsUtils';
import { CSS_CLASSES } from '../../styles/constants';
import GenericModal from '../Modals/GenericModal';
import MD3Select from '../Helpers/MD3Select';

interface AddEffectModalProps {
  isOpen: boolean;
  onClose: () => void;
  onAddEffect: (effectType: string, channel: AudioChannel) => void;
  channel: AudioChannel;
  serverSchema?: any;
  providersInfo?: any;
}

function AddEffectModal({
  isOpen,
  onClose,
  onAddEffect,
  channel,
  serverSchema,
  providersInfo
}: AddEffectModalProps): JSX.Element {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedProvider, setSelectedProvider] = useState<string>('all');

  // Get available effects and providers
  const availableEffects = useMemo(() => getAvailableEffectTypesFromServer(serverSchema), [serverSchema]);

  const providers = useMemo(() => {
    if (providersInfo?.providers) {
      return providersInfo.providers.filter((p: any) => p.available);
    }
    return [];
  }, [providersInfo]);

  // Filter effects based on search term and selected provider
  const filteredEffects = useMemo(() => {
    return availableEffects.filter((effect) => {
      const matchesSearch =
        !searchTerm ||
        effect.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        effect.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
        effect.type.toLowerCase().includes(searchTerm.toLowerCase());

      const matchesProvider = selectedProvider === 'all' || effect.provider === selectedProvider;

      return matchesSearch && matchesProvider;
    });
  }, [availableEffects, searchTerm, selectedProvider]);

  // Group effects by provider for better organization
  const effectsByProvider = useMemo(() => {
    const grouped: Record<string, typeof filteredEffects> = {};
    filteredEffects.forEach((effect) => {
      const provider = effect.provider || 'unknown';
      if (!grouped[provider]) {
        grouped[provider] = [];
      }
      grouped[provider].push(effect);
    });
    return grouped;
  }, [filteredEffects]);

  const handleAddEffect = (effectType: string) => {
    onAddEffect(effectType, channel);
    onClose();
    setSearchTerm('');
    setSelectedProvider('all');
  };

  const handleClose = () => {
    onClose();
    setSearchTerm('');
    setSelectedProvider('all');
  };

  const providerOptions = [
    { value: 'all', label: `All Providers (${availableEffects.length} effects)` },
    ...providers.map((provider: any) => {
      const providerEffects = availableEffects.filter((effect) => effect.provider === provider.name);
      return {
        value: provider.name,
        label: `${provider.name} (${providerEffects.length} effects)`
      };
    })
  ];

  return (
    <GenericModal
      isOpen={isOpen}
      onClose={handleClose}
      title="Add Audio Effect"
      size="large"
      secondaryButton={{
        text: 'Cancel',
        onClick: handleClose
      }}
    >
      <div className="space-y-4 pt-1">
        {/* Subtitle */}
        <p className="text-sm text-on-surface-variant">
          Choose an effect for the <span className="font-semibold capitalize text-primary">{channel}</span>{' '}
          channel
        </p>

        {/* Search and Filter */}
        <div className="space-y-3.5 bg-surface-container-low p-3.5 rounded-md border border-outline-variant">
          {/* Search Bar */}
          <div className="relative">
            <FontAwesomeIcon
              icon={faSearch}
              className="absolute left-3.5 top-1/2 transform -translate-y-1/2 text-on-surface-variant/50 h-4 w-4"
            />
            <input
              type="text"
              placeholder="Search effects..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className={`${CSS_CLASSES.input} pl-10`}
            />
          </div>

          {/* Provider Filter */}
          <div className="flex items-center gap-3">
            <FontAwesomeIcon icon={faFilter} className="text-on-surface-variant/50 h-4 w-4" />
            <div className="flex-1">
              <MD3Select
                id="providerFilter"
                label="Filter by Provider"
                value={selectedProvider}
                onChange={(e) => setSelectedProvider(e.target.value)}
                options={providerOptions}
              />
            </div>
          </div>
        </div>

        {/* Effects List */}
        <div className="max-h-96 overflow-y-auto pr-1">
          {filteredEffects.length === 0 ? (
            <div className="text-center py-12 text-on-surface-variant/60 italic">
              <p>No effects found</p>
              {searchTerm && <p className="text-sm mt-2">Try adjusting your search terms</p>}
            </div>
          ) : selectedProvider === 'all' ? (
            // Group by provider when showing all
            <div className="space-y-6">
              {Object.entries(effectsByProvider).map(([provider, effects]) => (
                <div key={provider} className="space-y-3">
                  <h4 className="font-bold text-primary text-xs uppercase tracking-wider pl-1">
                    {provider} ({effects.length})
                  </h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {effects.map((effect) => (
                      <EffectCard
                        key={effect.type}
                        effect={effect}
                        onAdd={() => handleAddEffect(effect.type)}
                      />
                    ))}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            // Single provider view
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {filteredEffects.map((effect) => (
                <EffectCard
                  key={effect.type}
                  effect={effect}
                  onAdd={() => handleAddEffect(effect.type)}
                />
              ))}
            </div>
          )}
        </div>

        {/* Footer Info */}
        <div className="text-xs text-on-surface-variant/80 text-center pt-3 border-t border-outline-variant font-semibold">
          {filteredEffects.length} of {availableEffects.length} effects shown
        </div>
      </div>
    </GenericModal>
  );
}

interface EffectCardProps {
  effect: {
    type: string;
    name: string;
    description: string;
    provider?: string;
  };
  onAdd: () => void;
}

function EffectCard({ effect, onAdd }: EffectCardProps): JSX.Element {
  return (
    <button
      onClick={onAdd}
      className="p-4 border border-outline-variant bg-surface-container-low rounded-lg hover:border-outline hover:bg-primary/8 transition-all text-left group"
    >
      <div className="flex items-start justify-between">
        <div className="flex-1 min-w-0">
          <h5 className="font-semibold text-on-surface group-hover:text-primary transition-colors text-sm">
            {effect.name}
          </h5>
          <p className="text-xs text-on-surface-variant mt-1.5 break-words line-clamp-2">
            {effect.description}
          </p>
          {effect.provider && (
            <span className="inline-block mt-2.5 px-2.5 py-0.5 bg-surface-container-highest text-on-surface-variant rounded-full text-[10px] font-semibold">
              {effect.provider}
            </span>
          )}
        </div>
      </div>
    </button>
  );
}

export default AddEffectModal;