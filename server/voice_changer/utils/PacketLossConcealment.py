import numpy as np

class PacketLossConcealment:
    """
    Packet Loss Concealment (PLC) helper to mitigate audio dropouts (哑音/吞音) 
    during real-time streaming when inference chunks are skipped or late.
    
    Uses windowed waveform replication with a gain decay profile across consecutive skips.
    """
    def __init__(self):
        self.last_out = None
        self.consecutive_skips = 0

    def conceal(self, length: int) -> np.ndarray:
        """Generates replacement audio chunk of the given length using the last valid output."""
        if self.last_out is None or len(self.last_out) == 0:
            return np.zeros(length, dtype=np.float32)
        
        last_len = len(self.last_out)
        self.consecutive_skips += 1
        
        # Tile last_out to cover the requested length
        repeats = (length + last_len - 1) // last_len
        replicated = np.tile(self.last_out, repeats)[:length]
        
        # Apply a step-down linear fade-out to prevent robotic sound or loop artifacts
        # We fade out to 0 over 3 consecutive skips (approx 200-300ms)
        start_gain = max(0.0, 1.0 - (self.consecutive_skips - 1) * 0.35)
        end_gain = max(0.0, 1.0 - self.consecutive_skips * 0.35)
        
        fade = np.linspace(start_gain, end_gain, length, dtype=np.float32)
        concealed_audio = replicated * fade
        return concealed_audio

    def update(self, out_audio: np.ndarray):
        """Updates the internal buffer with the last successfully generated audio chunk."""
        if out_audio is not None and len(out_audio) > 0:
            self.last_out = out_audio.copy()
            self.consecutive_skips = 0
