import torch
from abc import ABC, abstractmethod

class Vocoder(ABC):
    """Abstract Base Class for Standalone Inference Vocoders."""

    @abstractmethod
    def infer(
        self,
        z: torch.Tensor,
        pitchf: torch.Tensor,
        g: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Synthesize audio waveform from latent representation z and F0 pitch contour.
        
        Args:
            z: Latent feature Tensor [B, C, T]
            pitchf: Pitch frequency Tensor [B, T]
            g: Speaker embedding Tensor [B, C_g, 1] (optional)
            
        Returns:
            Audio waveform Tensor [B, 1, T_samples]
        """
        pass
