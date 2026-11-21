from collections import deque
from time import monotonic
import numpy as np


class ClockDriftTracker:
    """
    O(1) sliding-window minimum tracker for `diff = recv_ms - client_ts_ms`.
    Prevents long-term clock drift or NTP jumps from locking `relative_age` above threshold.
    """

    def __init__(self, window_sec: float = 60.0):
        self.window_sec = float(window_sec)
        self._min_deque: deque[tuple[float, int]] = deque()

    def update(self, diff: int, now_sec: float | None = None) -> int:
        if now_sec is None:
            now_sec = monotonic()
        cutoff = now_sec - self.window_sec
        while self._min_deque and self._min_deque[0][0] < cutoff:
            self._min_deque.popleft()
        while self._min_deque and self._min_deque[-1][1] >= diff:
            self._min_deque.pop()
        self._min_deque.append((now_sec, diff))
        return diff - self._min_deque[0][1]


class PacketLossConcealment:
    """
    Packet Loss Concealment (PLC) helper to mitigate audio dropouts (哑音/吞音)
    during real-time streaming when inference chunks are skipped or late.

    Instead of repeating long blocks (which causes robotic stutter and pitch shifts),
    this implementation applies a quick 10ms fade-out to silence when a frame is skipped,
    and a quick 10ms fade-in when normal stream resumes, ensuring click-free transitions.
    """

    def __init__(self):
        self.last_out = None
        self.consecutive_skips = 0

    def conceal(self, length: int) -> np.ndarray:
        """Generates a click-free fade-out to silence for the skipped frame."""
        out = np.zeros(length, dtype=np.float32)
        if self.last_out is not None and len(self.last_out) > 0:
            fade_len = min(512, len(self.last_out), length)
            out[:fade_len] = self.last_out[-fade_len:]
            fade = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
            out[:fade_len] *= fade
        self.consecutive_skips += 1
        return out

    def process_normal(self, out_audio: np.ndarray) -> np.ndarray:
        """Processes a normal output block, applying a fade-in if returning from a skip."""
        if out_audio is None or len(out_audio) == 0:
            return out_audio

        self.last_out = out_audio.copy()

        if self.consecutive_skips > 0:
            fade_len = min(512, len(out_audio))
            if fade_len > 0:
                fade = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
                out_audio = out_audio.copy()
                out_audio[:fade_len] *= fade
            self.consecutive_skips = 0

        return out_audio
