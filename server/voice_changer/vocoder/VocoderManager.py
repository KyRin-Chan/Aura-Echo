import os
import logging
from typing import Optional, Dict

from voice_changer.vocoder.Vocoder import Vocoder
from voice_changer.vocoder.RefineGANVocoder import RefineGANVocoder

logger = logging.getLogger(__name__)


class VocoderManager:
    """
    Singleton manager for standalone inference vocoders.

    Responsibilities:
    - Lazy-load and cache vocoder instances by type name
    - Switch the active vocoder at runtime
    - Propagate the current RVC model's sample rate to loaded vocoders
    """

    _instance: Optional["VocoderManager"] = None
    _vocoders: Dict[str, Vocoder] = {}
    _current_vocoder_type: str = "embedded"
    _target_sr: int = 40000          # updated when an RVC model is loaded

    @classmethod
    def get_instance(cls) -> "VocoderManager":
        if cls._instance is None:
            cls._instance = VocoderManager()
        return cls._instance

    # ------------------------------------------------------------------
    # Vocoder selection
    # ------------------------------------------------------------------

    def set_vocoder_type(self, vocoder_type: str, target_sr: Optional[int] = None):
        """
        Activate a vocoder by type name (e.g. ``'embedded'``, ``'refinegan'``).

        Args:
            vocoder_type: Identifier string selected in the UI.
            target_sr   : Override the output sample rate.  If *None*, the
                          last value set by :meth:`update_target_sr` is used.
        """
        self._current_vocoder_type = vocoder_type
        if target_sr is not None:
            self._target_sr = target_sr

        logger.info(
            f"VocoderManager: active vocoder → '{vocoder_type}'  "
            f"(target_sr={self._target_sr} Hz)"
        )

        if vocoder_type == "refinegan" and "refinegan" not in self._vocoders:
            self._load_refinegan()

    def get_active_vocoder(self) -> Optional[Vocoder]:
        """
        Return the active standalone vocoder, or *None* when the embedded
        decoder (NSF-HiFiGAN) should be used.
        """
        if self._current_vocoder_type in ("embedded", "nsf-hifigan"):
            return None
        return self._vocoders.get(self._current_vocoder_type)

    # ------------------------------------------------------------------
    # Sample-rate propagation
    # ------------------------------------------------------------------

    def update_target_sr(self, target_sr: int) -> None:
        """
        Notify the manager that the loaded RVC model uses *target_sr* Hz.

        This is called by :class:`RVCr2` after a model is initialised so that
        the RefineGAN resampler is aligned with the model's native sample rate.
        """
        if target_sr == self._target_sr:
            return

        self._target_sr = target_sr
        logger.info(f"VocoderManager: target_sr updated → {target_sr} Hz")

        # If RefineGAN is already loaded, update its resampler immediately
        if "refinegan" in self._vocoders:
            vocoder = self._vocoders["refinegan"]
            if isinstance(vocoder, RefineGANVocoder):
                vocoder.update_target_sr(target_sr)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_refinegan(self) -> None:
        """Instantiate and cache the RefineGAN vocoder."""
        model_path_primary = os.path.join("pretrain", "vocoder", "f0G32k.pth")
        model_path_fallback = os.path.join("pretrain", "vocoder", "refinegan.pth")
        model_path = (
            model_path_primary
            if os.path.exists(model_path_primary)
            else model_path_fallback
        )
        try:
            self._vocoders["refinegan"] = RefineGANVocoder(
                model_path=model_path,
                target_sr=self._target_sr,
            )
        except Exception as exc:
            logger.error(f"VocoderManager: failed to load RefineGAN: {exc}")
