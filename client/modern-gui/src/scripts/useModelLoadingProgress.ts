import { useEffect, useRef } from 'react';
import { io, Socket } from 'socket.io-client';

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
 * Subscribes to model loading progress events from the server.
 *
 * - 'sio' mode: connects to the /test Socket.IO namespace and listens for
 *   `model_loading_progress` events (JSON object payload).
 * - 'ws'  mode: opens a dedicated WebSocket to /ws/progress and parses JSON text frames.
 * - 'rest' mode: no real-time channel — does nothing.
 *
 * Calls onProgress(data) for each step, so the UI can display a progress bar.
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
    if (!serverUrl) return;

    if (protocol === 'sio') {
      // Connect with the default JSON parser — no msgpack needed for progress events
      const socket: Socket = io(serverUrl + '/test', {
        transports: ['websocket'],
        reconnection: false,
      });

      socket.on('model_loading_progress', (data: LoadingProgress) => {
        onProgressRef.current(data);
      });

      return () => {
        socket.disconnect();
      };
    }

    if (protocol === 'ws') {
      const wsUrl = serverUrl.replace(/^http/, 'ws') + '/ws/progress';
      let ws: WebSocket | null = new WebSocket(wsUrl);

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

      return () => {
        ws?.close();
        ws = null;
      };
    }

    // 'rest' mode: no persistent connection for progress
  }, [protocol, serverUrl]);
}
