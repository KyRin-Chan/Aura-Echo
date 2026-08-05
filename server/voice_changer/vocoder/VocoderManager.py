import os
import logging
from typing import Optional, Dict
from voice_changer.vocoder.Vocoder import Vocoder
from voice_changer.vocoder.RefineGANVocoder import RefineGANVocoder

logger = logging.getLogger(__name__)

class VocoderManager:
    """Singleton Manager to handle loading, caching, and switching of inference vocoders."""

    _instance: Optional['VocoderManager'] = None
    _vocoders: Dict[str, Vocoder] = {}
    _current_vocoder_type: str = "embedded"

    @classmethod
    def get_instance(cls) -> 'VocoderManager':
        if cls._instance is None:
            cls._instance = VocoderManager()
        return cls._instance

    def set_vocoder_type(self, vocoder_type: str, target_sr: int = 40000):
        """Set the active vocoder type (e.g. 'embedded', 'refinegan')."""
        self._current_vocoder_type = vocoder_type
        logger.info(f"VocoderManager set active vocoder type to: '{vocoder_type}'")

        if vocoder_type == "refinegan" and "refinegan" not in self._vocoders:
            model_path_f0G32k = os.path.join("pretrain", "vocoder", "f0G32k.pth")
            model_path_refinegan = os.path.join("pretrain", "vocoder", "refinegan.pth")
            model_path = model_path_f0G32k if os.path.exists(model_path_f0G32k) else model_path_refinegan
            try:
                self._vocoders["refinegan"] = RefineGANVocoder(model_path=model_path, target_sr=target_sr)
            except Exception as e:
                logger.error(f"Failed to instantiate RefineGAN vocoder: {e}")

    def get_active_vocoder(self) -> Optional[Vocoder]:
        """Returns the active standalone vocoder, or None if using default embedded decoder."""
        if self._current_vocoder_type == "embedded" or self._current_vocoder_type == "nsf-hifigan":
            return None
        return self._vocoders.get(self._current_vocoder_type, None)
