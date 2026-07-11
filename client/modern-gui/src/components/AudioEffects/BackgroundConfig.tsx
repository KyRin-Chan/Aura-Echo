import { JSX, useEffect, useState } from 'react';
import { CSS_CLASSES } from '../../styles/constants';
import MD3Slider from '../Helpers/MD3Slider';
import MD3Select from '../Helpers/MD3Select';
import AudioPlayer from '../Helpers/AudioPlayer';
import { BackgroundTrack } from '@dannadori/voice-changer-client-js';

type BackgroundConfigProps = {
  track: BackgroundTrack | null;
  onChange: (id: string, key: string, value: any) => void;
};

function BackgroundConfig({ track, onChange }: BackgroundConfigProps): JSX.Element {
  const [local, setLocal] = useState<BackgroundTrack | null>(track);
  const [displayGain, setDisplayGain] = useState<number>(track?.gainDb ?? -6);

  useEffect(() => {
    setLocal(track);
    if (track) setDisplayGain(track.gainDb);
  }, [track]);

  const handle = (key: keyof Omit<BackgroundTrack, 'id'>, value: any) => {
    if (!local) return;
    const updated = { ...local, [key]: value } as BackgroundTrack;
    setLocal(updated);
    onChange(local.id, key, value);
  };

  if (!local) {
    return (
      <div className="flex flex-col h-full bg-surface-container-low p-4 rounded-md border border-outline-variant">
        <div className="flex items-center justify-between mb-3 pb-2 border-b border-outline-variant">
          <h5 className="font-semibold text-on-surface">Background Config</h5>
        </div>
        <div className="flex-1 flex items-center justify-center text-on-surface-variant/60 text-sm italic">
          Select a background track
        </div>
      </div>
    );
  }

  const modeOptions = [
    { value: 'loop', label: 'Loop' },
    { value: 'random', label: 'Random' }
  ];

  return (
    <div className="flex flex-col h-full bg-surface-container-low p-4 rounded-md border border-outline-variant">
      {/* Header */}
      <div className="flex items-center justify-between mb-4 pb-3 border-b border-outline-variant">
        <div>
          <h5 className="font-semibold text-on-surface text-base">{local.name}</h5>
          <div className="text-xs text-on-surface-variant mt-1">
            {local.filename}
          </div>
        </div>
        <div
          className={`px-3 py-1 rounded-full text-xs font-semibold ${
            local.enabled
              ? 'bg-primary-container text-on-primary-container'
              : 'bg-surface-container-highest text-on-surface-variant'
          }`}
        >
          {local.enabled ? 'Enabled' : 'Disabled'}
        </div>
      </div>

      {/* Parameters */}
      <div className="flex-1 overflow-y-auto space-y-4">
        {/* Name */}
        <div>
          <label className={CSS_CLASSES.label}>Name</label>
          <input
            type="text"
            value={local.name}
            onChange={(e) => handle('name', e.target.value)}
            className={CSS_CLASSES.input}
          />
        </div>

        {/* Gain */}
        <div>
          <label className={CSS_CLASSES.label}>Gain</label>
          <MD3Slider
            min={-60}
            max={12}
            step={0.1}
            value={local.gainDb}
            onImmediateChange={(v) => setDisplayGain(v)}
            onChange={(v) => {
              setDisplayGain(v);
              handle('gainDb', v);
            }}
            showValue={true}
            valueFormatter={(val) => `${val.toFixed(1)} dB`}
          />
        </div>

        {/* Mode */}
        <div>
          <MD3Select
            label="Mode"
            value={local.mode}
            onChange={(e) => handle('mode', e.target.value as 'loop' | 'random')}
            options={modeOptions}
          />
        </div>

        {/* Loop pause (seconds) when in Loop mode */}
        {local.mode === 'loop' && (
          <div>
            <label className={CSS_CLASSES.label}>Loop Pause (s)</label>
            <input
              type="number"
              className={CSS_CLASSES.input}
              step={0.1}
              value={local.loopPauseSec ?? 0}
              onChange={(e) => handle('loopPauseSec', parseFloat(e.target.value || '0'))}
            />
          </div>
        )}

        {local.mode === 'random' && (
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className={CSS_CLASSES.label}>Min Pause (s)</label>
              <input
                type="number"
                className={CSS_CLASSES.input}
                step={0.1}
                value={local.random?.minPauseSec ?? 2}
                onChange={(e) => {
                  const newMin = parseFloat(e.target.value || '0');
                  handle('random', {
                    minPauseSec: newMin,
                    maxPauseSec: local.random?.maxPauseSec ?? 5
                  });
                }}
              />
            </div>
            <div>
              <label className={CSS_CLASSES.label}>Max Pause (s)</label>
              <input
                type="number"
                className={CSS_CLASSES.input}
                step={0.1}
                value={local.random?.maxPauseSec ?? 5}
                onChange={(e) => {
                  const newMax = parseFloat(e.target.value || '0');
                  handle('random', {
                    minPauseSec: local.random?.minPauseSec ?? 2,
                    maxPauseSec: newMax
                  });
                }}
              />
            </div>
          </div>
        )}

        {/* Preview & Info (bottom) */}
        {local.filename && (
          <div className="pt-2">
            <label className={CSS_CLASSES.label}>Preview</label>
            <div className="mt-1">
              <AudioPlayer src={`/sound_dir/${local.id}/${local.filename}`} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default BackgroundConfig;
