import { JSX, useRef } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faPlus, faTrash, faShuffle, faRepeat } from '@fortawesome/free-solid-svg-icons';
import { CSS_CLASSES } from '../../styles/constants';
import { BackgroundTrack } from '@dannadori/voice-changer-client-js';
import { t } from '../../locales';

export type BackgroundListProps = {
  tracks: BackgroundTrack[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onAddFiles: (files: FileList) => void;
  onDelete: (id: string) => void;
  onToggle: (id: string) => void;
};

function TrackItem({
  track,
  isSelected,
  onSelect,
  onDelete,
  onToggle
}: {
  track: BackgroundTrack;
  isSelected: boolean;
  onSelect: () => void;
  onDelete: () => void;
  onToggle: () => void;
}) {
  return (
    <div
      className={`p-3 rounded-md border cursor-pointer transition-all duration-150 ${
        isSelected
          ? 'border-primary bg-primary/8 shadow-elevation-1'
          : 'border-outline-variant bg-surface-container-low hover:border-outline'
      }`}
      onClick={onSelect}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <button
            onClick={(e) => {
              e.stopPropagation();
              onToggle();
            }}
            className={`${CSS_CLASSES.iconButton} ${
              track.enabled ? 'text-primary' : 'text-on-surface-variant/40'
            }`}
            title={track.enabled ? t('disableTooltip') : t('enableTooltip')}
          >
            <FontAwesomeIcon icon={track.mode === 'loop' ? faRepeat : faShuffle} className="h-4 w-4" />
          </button>

          <div>
            <div className="font-semibold text-on-surface text-sm flex items-center space-x-2">
              <span className="truncate max-w-[180px]" title={track.name || track.filename}>
                {track.name || track.filename || t('untitled')}
              </span>
            </div>
            <div className="text-xs text-on-surface-variant flex items-center space-x-2">
              <span className="capitalize">{track.mode}</span>
              <span>•</span>
              {track.mode === 'loop' ? (
                <span>{t('pauseLabel')} {track.loopPauseSec?.toFixed(1) ?? '0.0'}s</span>
              ) : (
                <span>
                  {t('pauseLabel')} {track.random?.minPauseSec?.toFixed(1) ?? '2.0'}–
                  {track.random?.maxPauseSec?.toFixed(1) ?? '5.0'}s
                </span>
              )}
              <span>•</span>
              <span>{track.gainDb.toFixed(1)} dB</span>
            </div>
          </div>
        </div>

        <button
          onClick={(e) => {
            e.stopPropagation();
            onDelete();
          }}
          className={`${CSS_CLASSES.iconButton} text-error hover:text-error/80`}
          title={t('deleteTooltip')}
        >
          <FontAwesomeIcon icon={faTrash} className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}

export default function BackgroundList({
  tracks,
  selectedId,
  onSelect,
  onAddFiles,
  onDelete,
  onToggle
}: BackgroundListProps): JSX.Element {
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  return (
    <div className="flex flex-col h-full bg-surface-container-low p-4 rounded-md border border-outline-variant">
      {/* Header */}
      <div className="flex items-center justify-between mb-4 pb-3 border-b border-outline-variant">
        <div>
          <h5 className="font-semibold text-on-surface text-base">{t('backgroundTracksTitle')}</h5>
          <div className="text-xs text-on-surface-variant mt-1">
            {t('bgTracksDesc')}
          </div>
        </div>
        <div>
          <input
            ref={fileInputRef}
            type="file"
            accept="audio/*"
            className="hidden"
            onChange={(e) => {
              if (e.target.files?.length) onAddFiles(e.target.files);
              if (fileInputRef.current) fileInputRef.current.value = '';
            }}
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            className={`${CSS_CLASSES.iconButton} text-primary`}
            title={t('addAudioFilesTooltip')}
          >
            <FontAwesomeIcon icon={faPlus} className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* List */}
      <div className="flex-1 min-h-0 overflow-y-auto pr-1">
        {tracks.length === 0 ? (
          <div className="text-center py-12 text-on-surface-variant/60 text-sm italic">
            {t('noBgTracksDesc')}
          </div>
        ) : (
          <div className="space-y-1">
            {tracks.map((track) => (
              <TrackItem
                key={track.id}
                track={track}
                isSelected={selectedId === track.id}
                onSelect={() => onSelect(track.id)}
                onDelete={() => onDelete(track.id)}
                onToggle={() => onToggle(track.id)}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
