import { JSX, useState, useRef, useEffect } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faPlay, faPause, faVolumeUp, faVolumeMute, faDownload } from '@fortawesome/free-solid-svg-icons';
import { CSS_CLASSES } from '../../styles/constants';
import { t } from '../../locales';

interface AudioPlayerProps {
  src: string;
  title?: string;
  className?: string;
  id?: string;
  outputDeviceId?: string;
  modelName?: string;
  audioType?: 'Input' | 'Output';
}

function AudioPlayer({
  src,
  title,
  className = '',
  id,
  outputDeviceId,
  modelName,
  audioType
}: AudioPlayerProps): JSX.Element {
  // ---------------- States ----------------
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(0.7);
  const [isMuted, setIsMuted] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  const audioRef = useRef<HTMLAudioElement>(null);

  // ---------------- Hooks ----------------

  // Setup audio player
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    // Set initial volume
    audio.volume = volume;

    // Reset states when src changes
    setDuration(0);
    setCurrentTime(0);
    setIsPlaying(false);
    setIsLoading(true);

    const handleLoadedMetadata = () => {
      if (audio.duration && isFinite(audio.duration)) {
        setDuration(audio.duration);
      }
      setIsLoading(false);
    };

    const handleTimeUpdate = () => {
      if (audio.currentTime && isFinite(audio.currentTime)) {
        setCurrentTime(audio.currentTime);
      }
    };

    const handleEnded = () => {
      setIsPlaying(false);
    };

    const handleCanPlay = () => {
      setIsLoading(false);
    };

    const handleWaiting = () => {
      setIsLoading(true);
    };

    const handleCanPlayThrough = () => {
      setIsLoading(false);
    };

    const handleLoadStart = () => {
      setIsLoading(true);
    };

    const handleError = () => {
      setIsLoading(false);
      setDuration(0);
      setCurrentTime(0);
    };

    // Add event listeners
    audio.addEventListener('loadedmetadata', handleLoadedMetadata);
    audio.addEventListener('timeupdate', handleTimeUpdate);
    audio.addEventListener('ended', handleEnded);
    audio.addEventListener('canplay', handleCanPlay);
    audio.addEventListener('waiting', handleWaiting);
    audio.addEventListener('canplaythrough', handleCanPlayThrough);
    audio.addEventListener('loadstart', handleLoadStart);
    audio.addEventListener('error', handleError);

    // Force load
    audio.load();

    return () => {
      audio.removeEventListener('loadedmetadata', handleLoadedMetadata);
      audio.removeEventListener('timeupdate', handleTimeUpdate);
      audio.removeEventListener('ended', handleEnded);
      audio.removeEventListener('canplay', handleCanPlay);
      audio.removeEventListener('waiting', handleWaiting);
      audio.removeEventListener('canplaythrough', handleCanPlayThrough);
      audio.removeEventListener('loadstart', handleLoadStart);
      audio.removeEventListener('error', handleError);
    };
  }, [src]);

  // Update sink device when outputDeviceId changes
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio || !outputDeviceId) return;

    try {
      if ((audio as any).setSinkId) {
        (audio as any).setSinkId(outputDeviceId);
      }
    } catch (error) {
      console.error('Error setting audio output device:', error);
    }
  }, [outputDeviceId]);

  // Toggle play/pause
  const togglePlayPause = async () => {
    const audio = audioRef.current;
    if (!audio) return;

    try {
      if (isPlaying) {
        audio.pause();
        setIsPlaying(false);
      } else {
        await audio.play();
        setIsPlaying(true);
      }
    } catch (error) {
      console.error('Error playing audio:', error);
      setIsPlaying(false);
    }
  };

  // ---------------- Handlers ----------------

  // Handle progress change
  const handleProgressChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const audio = audioRef.current;
    if (!audio || !duration || !isFinite(duration)) return;

    const newTime = (parseFloat(e.target.value) / 100) * duration;
    if (isFinite(newTime)) {
      audio.currentTime = newTime;
      setCurrentTime(newTime);
    }
  };

  // Handle volume change
  const handleVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const audio = audioRef.current;
    const newVolume = parseFloat(e.target.value) / 100;
    setVolume(newVolume);
    setIsMuted(newVolume === 0);

    if (audio) {
      audio.volume = newVolume;
    }
  };

  // Handle download
  const handleDownload = async () => {
    try {
      const response = await fetch(src);
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);

      const now = new Date();
      const timestamp = now
        .toISOString()
        .replace(/T/, '_')
        .replace(/:/g, '-')
        .split('.')[0];

      const safeModelName = (modelName || 'Unknown').replace(/[^a-zA-Z0-9-_]/g, '_');
      const prefix = audioType || 'Audio';
      const filename = `${prefix}-${safeModelName}-${timestamp}.wav`;

      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Error downloading audio:', error);
    }
  };

  // Toggle mute
  const toggleMute = () => {
    const audio = audioRef.current;
    if (!audio) return;

    if (isMuted) {
      audio.volume = volume;
      setIsMuted(false);
    } else {
      audio.volume = 0;
      setIsMuted(true);
    }
  };

  // Format time
  const formatTime = (time: number): string => {
    if (!isFinite(time) || isNaN(time) || time < 0) return '0:00';
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  // Calculate progress percentage
  const progressPercentage =
    duration > 0 && isFinite(duration) && isFinite(currentTime) ? (currentTime / duration) * 100 : 0;

  // ---------------- Render ----------------

  const fillPercent = isFinite(progressPercentage) ? progressPercentage : 0;

  return (
    <div className={`bg-surface-container-low border border-outline-variant rounded-xl p-3.5 ${className}`}>
      <audio ref={audioRef} src={src} preload="metadata" id={id} controls={false} />

      {title && <div className="text-xs font-bold text-on-surface mb-2.5 truncate">{title}</div>}

      <div className="flex items-center gap-3 overflow-hidden">
        {/* Play/Pause Button */}
        <button
          onClick={togglePlayPause}
          disabled={isLoading}
          className="flex-shrink-0 w-9 h-9 flex items-center justify-center bg-primary hover:bg-primary/90 hover:shadow-elevation-1 disabled:bg-surface-container-highest text-on-primary rounded-full transition-all duration-150 active:scale-95"
        >
          {isLoading ? (
            <div className="w-4 h-4 border-2 border-on-primary border-t-transparent rounded-full animate-spin"></div>
          ) : (
            <FontAwesomeIcon icon={isPlaying ? faPause : faPlay} className="text-xs" />
          )}
        </button>

        {/* Progress Section */}
        <div className="flex-1 min-w-0">
          {/* Time Display */}
          <div className="flex justify-between text-[10px] text-on-surface-variant font-semibold mb-1">
            <span className="tabular-nums">{formatTime(currentTime)}</span>
            <span className="tabular-nums">{formatTime(duration)}</span>
          </div>

          {/* Progress Bar */}
          <div className="relative">
            <input
              type="range"
              min="0"
              max="100"
              value={fillPercent}
              onChange={handleProgressChange}
              disabled={!duration || !isFinite(duration)}
              className="w-full h-1.5 rounded-lg appearance-none cursor-pointer accent-primary disabled:opacity-50 disabled:cursor-not-allowed"
              style={{
                background: `linear-gradient(to right, var(--md-sys-color-primary) 0%, var(--md-sys-color-primary) ${fillPercent}%, var(--md-sys-color-surface-container-highest) ${fillPercent}%, var(--md-sys-color-surface-container-highest) 100%)`
              }}
            />
          </div>
        </div>

        {/* Volume Control */}
        <div className="flex items-center gap-1.5 flex-shrink-0">
          <button onClick={toggleMute} className={`${CSS_CLASSES.iconButton} p-1.5`}>
            <FontAwesomeIcon
              icon={isMuted || volume === 0 ? faVolumeMute : faVolumeUp}
              className="text-xs text-on-surface-variant"
            />
          </button>

          <input
            type="range"
            min="0"
            max="100"
            value={isMuted ? 0 : volume * 100}
            onChange={handleVolumeChange}
            className="w-12 h-1 rounded-lg appearance-none cursor-pointer accent-primary"
            style={{
              background: `linear-gradient(to right, var(--md-sys-color-primary) 0%, var(--md-sys-color-primary) ${
                isMuted ? 0 : volume * 100
              }%, var(--md-sys-color-surface-container-highest) ${
                isMuted ? 0 : volume * 100
              }%, var(--md-sys-color-surface-container-highest) 100%)`
            }}
          />
        </div>

        {/* Download Button */}
        <button
          onClick={handleDownload}
          className={`${CSS_CLASSES.iconButton} p-2 rounded-full flex-shrink-0 text-on-surface-variant hover:bg-surface-variant/20`}
          title={t('downloadAudioTooltip')}
        >
          <FontAwesomeIcon icon={faDownload} className="text-xs" />
        </button>
      </div>
    </div>
  );
}

export default AudioPlayer;