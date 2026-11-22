import { useState, useEffect, useCallback, useRef } from 'react';
import { TensorRTStatus } from '@dannadori/voice-changer-client-js';
import { useAppState } from '../context/AppContext';
import { useUIContext } from '../context/UIContext';
import { t } from '../locales';

const ACTIVE_STATES = new Set(['resolving', 'downloading', 'verifying', 'extracting']);

export function formatBytes(bytes: number): string {
  if (!bytes || bytes <= 0) return '0 B';
  if (bytes >= 1024 * 1024 * 1024) {
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
  }
  if (bytes >= 1024 * 1024) {
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }
  if (bytes >= 1024) {
    return `${Math.round(bytes / 1024)} KB`;
  }
  return `${bytes} B`;
}

export function getTensorRTPhaseText(status: TensorRTStatus): string {
  switch (status.state) {
    case 'resolving':
      return t('tensorRtResolving');
    case 'verifying':
      return t('tensorRtVerifying');
    case 'extracting':
      return t('tensorRtExtracting');
    case 'downloading': {
      const downloaded = formatBytes(status.downloadedBytes);
      const total = status.totalBytes > 0 ? ` / ${formatBytes(status.totalBytes)}` : '';
      const speed = status.speedBytesPerSec > 0 ? ` · ${formatBytes(status.speedBytesPerSec)}/s` : '';
      return `${t('tensorRtDownloading')} ${downloaded}${total}${speed}`;
    }
    default:
      return '';
  }
}

export function useTensorRT() {
  const appState = useAppState();
  const uiState = useUIContext();

  const [status, setStatus] = useState<TensorRTStatus | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const prevStateRef = useRef<string | null>(null);
  const appStateRef = useRef(appState);
  appStateRef.current = appState;
  const uiStateRef = useRef(uiState);
  uiStateRef.current = uiState;

  const refreshStatus = useCallback(async () => {
    const next = await appStateRef.current.serverSetting.getTensorRTStatus();
    const prevState = prevStateRef.current;
    prevStateRef.current = next.state;
    setStatus(next);

    if (prevState && ACTIVE_STATES.has(prevState)) {
      if (next.state === 'installed') {
        await appStateRef.current.serverSetting.reloadServerInfo();
        uiStateRef.current.showError(t('tensorRtInstallSuccess'), 'Confirm');
      } else if (next.state === 'error') {
        uiStateRef.current.showError(
          t('tensorRtInstallFailed').replace('{error}', next.error || 'Unknown error'),
          'Error'
        );
      }
    }
    return next;
  }, []);

  useEffect(() => {
    let mounted = true;
    refreshStatus().catch((err) => {
      if (mounted) {
        console.error('Failed to fetch TensorRT status:', err);
      }
    });
    return () => {
      mounted = false;
    };
  }, [refreshStatus]);

  const isActive = status != null && ACTIVE_STATES.has(status.state);

  useEffect(() => {
    if (!isActive) return;
    const timer = setInterval(() => {
      refreshStatus().catch((err) => {
        console.error('Failed to poll TensorRT status:', err);
      });
    }, 500);
    return () => clearInterval(timer);
  }, [isActive, refreshStatus]);

  const startDownload = useCallback(async () => {
    setIsSubmitting(true);
    try {
      const next = await appStateRef.current.serverSetting.downloadTensorRT();
      prevStateRef.current = next.state;
      setStatus(next);
      if (next.state === 'installed') {
        await appStateRef.current.serverSetting.reloadServerInfo();
      }
      return next;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      uiStateRef.current.showError(t('tensorRtInstallFailed').replace('{error}', msg), 'Error');
      try {
        await refreshStatus();
      } catch {
        // ignore secondary refresh failure
      }
      return null;
    } finally {
      setIsSubmitting(false);
    }
  }, [refreshStatus]);

  const cancelDownload = useCallback(async () => {
    setIsSubmitting(true);
    try {
      const next = await appStateRef.current.serverSetting.cancelTensorRTDownload();
      prevStateRef.current = next.state;
      setStatus(next);
      return next;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      uiStateRef.current.showError(msg, 'Error');
      return null;
    } finally {
      setIsSubmitting(false);
    }
  }, []);

  const deleteRuntime = useCallback(async () => {
    setIsDeleting(true);
    try {
      const next = await appStateRef.current.serverSetting.deleteTensorRT();
      prevStateRef.current = next.state;
      setStatus(next);
      await appStateRef.current.serverSetting.reloadServerInfo();
      if (next.pendingDelete || next.status === 'pending_restart') {
        uiStateRef.current.showError(t('tensorRtPendingRestartMsg'), 'Warning');
      } else {
        uiStateRef.current.showError(t('tensorRtDeleteSuccess'), 'Confirm');
      }
      return next;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      uiStateRef.current.showError(msg, 'Error');
      return null;
    } finally {
      setIsDeleting(false);
    }
  }, []);

  return {
    status,
    isActive,
    isSubmitting,
    isDeleting,
    startDownload,
    cancelDownload,
    deleteRuntime,
    refreshStatus,
  };
}
