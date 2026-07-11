import { JSX } from 'react';
import { RVCModelSlot } from '@dannadori/voice-changer-client-js';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faCog } from '@fortawesome/free-solid-svg-icons';
import MD3Switch from '../../../Helpers/MD3Switch';
import { t } from '../../../../locales';

interface MergeConfigurationProps {
  downloadModel: boolean;
  setDownloadModel: (download: boolean) => void;
  saveToMergeSlot: boolean;
  setSaveToMergeSlot: (save: boolean) => void;
  saveToEmptySlot: boolean;
  setSaveToEmptySlot: (save: boolean) => void;
  emptySlots: RVCModelSlot[];
}

function MergeConfiguration({
  downloadModel,
  setDownloadModel,
  saveToMergeSlot,
  setSaveToMergeSlot,
  saveToEmptySlot,
  setSaveToEmptySlot,
  emptySlots
}: MergeConfigurationProps): JSX.Element {
  // ---------------- Handlers ----------------

  // Handle merge slot change
  const handleMergeSlotChange = (checked: boolean) => {
    setSaveToMergeSlot(checked);
    if (checked) {
      setSaveToEmptySlot(false);
    }
  };

  // Handle empty slot change
  const handleEmptySlotChange = (checked: boolean) => {
    setSaveToEmptySlot(checked);
    if (checked) {
      setSaveToMergeSlot(false);
    }
  };

  // Handle download model change
  const handleDownloadModelChange = (checked: boolean) => {
    setDownloadModel(checked);
  };

  // ---------------- Render ----------------

  return (
    <div className="space-y-4 p-4 bg-surface-container-low rounded-lg border border-outline-variant">
      <div className="flex justify-between items-center pb-2 border-b border-outline-variant/30">
        <h4 className="text-sm font-bold text-on-surface uppercase tracking-wider">{t('mergeOptionsLabel')}</h4>
        <FontAwesomeIcon icon={faCog} className="h-4 w-4 text-primary" />
      </div>

      <div className="space-y-3.5">
        <MD3Switch
          id="download-model"
          checked={downloadModel}
          onChange={handleDownloadModelChange}
          label={t('downloadMergedModelLabel')}
        />

        <MD3Switch
          id="save-merge-slot"
          checked={saveToMergeSlot}
          onChange={handleMergeSlotChange}
          label={t('saveToMergeSlotLabel')}
        />

        <div className="space-y-1.5">
          <MD3Switch
            id="save-empty-slot"
            checked={saveToEmptySlot}
            onChange={handleEmptySlotChange}
            disabled={emptySlots.length === 0}
            label={t('saveToEmptySlotLabel')}
          />

          {emptySlots.length === 0 && (
            <div className="ml-12 text-xs text-on-surface-variant italic">
              {t('noEmptySlotsAvailableLabel')}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default MergeConfiguration;