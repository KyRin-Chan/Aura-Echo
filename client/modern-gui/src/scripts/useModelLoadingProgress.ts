import { useEffect, useRef } from 'react';

export type LoadingProgress = {
  step: number;
  total: number;
  label: string;
};

type UseModelLoadingProgressProps = {
  protocol: 'sio' | 'ws' | 'rest';
  serverUrl: string;
  onProgress: (progress: LoadingProgress | null) => void;
};

/**
 * Subscribes to model loading progress events from the server via the dedicated
 * /ws/progress WebSocket endpoint, avoiding Socket.IO sid collisions on /test.
 */
export function useModelLoadingProgress({
  protocol,
  serverUrl,
  onProgress,
}: UseModelLoadingProgressProps): void {
  // Stable ref so the socket callback always calls the latest onProgress
  const onProgressRef = useRef(onProgress);
  onProgressRef.current = onProgress;

  useEffect(() => {
    const baseUrl = serverUrl || window.location.origin;
    const wsUrl = baseUrl.replace(/^http/, 'ws') + '/ws/progress';

    let isDisposed = false;
    let ws: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    const connect = () => {
      if (isDisposed) return;
      ws = new WebSocket(wsUrl);

      ws.onmessage = (event: MessageEvent) => {
        try {
          const data: LoadingProgress = JSON.parse(event.data as string);
          onProgressRef.current(data);
        } catch {
          // ignore malformed messages
        }
      };

      ws.onerror = () => {
        // silently ignore — server may not be ready yet
      };

      ws.onclose = () => {
        ws = null;
        if (!isDisposed) {
          reconnectTimer = setTimeout(connect, 2000);
        }
      };
    };

    connect();

    return () => {
      isDisposed = true;
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
      if (ws) {
        ws.onclose = null;
        ws.close();
        ws = null;
      }
    };
  }, [protocol, serverUrl]);
}
