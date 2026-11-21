from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
import logging
from .AudioEffect import AudioEffect, AudioChannel
from .AudioEffectProvider import ProviderRegistry

logger = logging.getLogger(__name__)

try:
    from pedalboard import Pedalboard
    PEDALBOARD_AVAILABLE = True
except ImportError:
    Pedalboard = None
    PEDALBOARD_AVAILABLE = False


class AudioEffectsManager:
    def __init__(self):
        self.input_effects: List[AudioEffect] = []
        self.output_effects: List[AudioEffect] = []
        self._compiled_input_chain: List[Tuple[str, Any]] = []
        self._compiled_output_chain: List[Tuple[str, Any]] = []
        self.provider_registry = ProviderRegistry()
        self._setup_default_providers()

    def _setup_default_providers(self) -> None:
        """Setup default audio effect providers"""
        logger.info("Setting up default audio effect providers...")

        try:
            from .providers.pedalboard.PedalboardProvider import PedalboardProvider
            pedalboard_provider = PedalboardProvider()
            if pedalboard_provider.is_available:
                self.provider_registry.register_provider(pedalboard_provider)
                logger.info(
                    f"PedalboardProvider registered successfully with {len(pedalboard_provider.supported_effects)} effects"
                )
            else:
                logger.warning("PedalboardProvider not available (pedalboard library not installed)")
        except ImportError as e:
            logger.warning(f"PedalboardProvider import failed: {e}")
        except Exception as e:
            logger.error(f"Error setting up PedalboardProvider: {e}")

        try:
            from .providers.simple.SimpleProvider import SimpleProvider
            simple_provider = SimpleProvider()
            if simple_provider.is_available:
                self.provider_registry.register_provider(simple_provider)
                logger.info(
                    f"SimpleProvider registered successfully with {len(simple_provider.supported_effects)} effects"
                )
            else:
                logger.warning("SimpleProvider not available")
        except ImportError as e:
            logger.warning(f"SimpleProvider import failed: {e}")
        except Exception as e:
            logger.error(f"Error setting up SimpleProvider: {e}")

        total_providers = len(self.provider_registry.providers)
        total_effects = len(self.get_supported_effects())
        logger.info(f"Audio effects setup complete: {total_providers} providers, {total_effects} total effects")

    def register_provider(self, provider) -> None:
        """Register a new audio effect provider"""
        self.provider_registry.register_provider(provider)
        logger.info(f"Registered audio effect provider: {provider.provider_name}")

    @staticmethod
    def _compile_chain(effects: List[AudioEffect]) -> List[Tuple[str, Any]]:
        compiled: List[Tuple[str, Any]] = []
        pending_pb: List[Any] = []

        def _flush_pb():
            if pending_pb:
                if PEDALBOARD_AVAILABLE and Pedalboard is not None:
                    compiled.append(("pedalboard", Pedalboard(list(pending_pb))))
                pending_pb.clear()

        for effect in effects:
            if not effect.is_enabled():
                continue
            # autoTune modifies F0 inside Pipeline.extract_pitch rather than waveform DSP
            if effect.effect_type == "autoTune":
                continue

            if PEDALBOARD_AVAILABLE and hasattr(effect, "get_pedalboard_plugins"):
                plugins = effect.get_pedalboard_plugins()
                if plugins:
                    pending_pb.extend(plugins)
                continue

            _flush_pb()
            compiled.append(("effect", effect))

        _flush_pb()
        return compiled

    def _rebuild_compiled_chains(self) -> None:
        self._compiled_input_chain = self._compile_chain(self.input_effects)
        self._compiled_output_chain = self._compile_chain(self.output_effects)

    def swap_chains(self, input_effects: List[AudioEffect], output_effects: List[AudioEffect]) -> None:
        """Atomically swap prebuilt input/output chains and their compiled Pedalboard batches."""
        in_sorted = sorted(input_effects, key=lambda x: x.order)
        out_sorted = sorted(output_effects, key=lambda x: x.order)
        compiled_in = self._compile_chain(in_sorted)
        compiled_out = self._compile_chain(out_sorted)
        self.input_effects = in_sorted
        self.output_effects = out_sorted
        self._compiled_input_chain = compiled_in
        self._compiled_output_chain = compiled_out

    def add_effect(
        self,
        effect_type: str,
        channel: AudioChannel,
        order: int = 0,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> AudioEffect:
        effect = self.provider_registry.create_effect(effect_type, channel, order)
        if effect is None:
            raise ValueError(f"Unknown effect type: {effect_type}")

        if parameters:
            if self.provider_registry.validate_effect_parameters(effect_type, parameters):
                effect.set_parameters(parameters)
            else:
                logger.warning(f"Invalid parameters for effect {effect_type}, using defaults")

        if channel == "input":
            self.input_effects.append(effect)
            self.input_effects.sort(key=lambda x: x.order)
        elif channel == "output":
            self.output_effects.append(effect)
            self.output_effects.sort(key=lambda x: x.order)

        self._rebuild_compiled_chains()
        logger.debug(f"Added {effect_type} effect to {channel} chain with order {order}")
        return effect

    def remove_effect(self, effect: AudioEffect) -> None:
        if effect in self.input_effects:
            self.input_effects.remove(effect)
        if effect in self.output_effects:
            self.output_effects.remove(effect)
        self._rebuild_compiled_chains()
        logger.info(f"Removed {effect.effect_type} effect")

    def clear_effects(self, channel: Optional[AudioChannel] = None) -> None:
        if channel is None or channel == "input":
            self.input_effects.clear()
        if channel is None or channel == "output":
            self.output_effects.clear()
        self._rebuild_compiled_chains()
        logger.debug(f"Cleared effects for {channel or 'all'} channel(s)")

    @staticmethod
    def _run_compiled_chain_np(
        chain: List[Tuple[str, Any]], audio_np: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        if not chain or audio_np is None or audio_np.size == 0:
            return audio_np

        current = np.ascontiguousarray(audio_np, dtype=np.float32)
        for kind, stage in chain:
            if kind == "pedalboard":
                arr_2d = current.reshape(1, -1) if current.ndim == 1 else current
                processed = stage(arr_2d, sample_rate, reset=False)
                if processed.ndim == 2 and processed.shape[0] == 1:
                    current = np.ascontiguousarray(processed[0], dtype=np.float32)
                else:
                    current = np.ascontiguousarray(processed, dtype=np.float32)
            else:
                if hasattr(stage, "process_numpy"):
                    current = stage.process_numpy(current, sample_rate)
                else:
                    t = torch.as_tensor(current, dtype=torch.float32)
                    current = stage.process(t, sample_rate).detach().cpu().numpy().astype(np.float32, copy=False)
        return current

    def process_input_chain_np(self, audio_np: np.ndarray, sample_rate: int) -> np.ndarray:
        chain = self._compiled_input_chain
        if not chain:
            return audio_np
        try:
            return self._run_compiled_chain_np(chain, audio_np, sample_rate)
        except Exception as e:
            logger.error(f"Error in input effects chain (numpy): {e}")
            return audio_np

    def process_output_chain_np(self, audio_np: np.ndarray, sample_rate: int) -> np.ndarray:
        chain = self._compiled_output_chain
        if not chain:
            return audio_np
        try:
            return self._run_compiled_chain_np(chain, audio_np, sample_rate)
        except Exception as e:
            logger.error(f"Error in output effects chain (numpy): {e}")
            return audio_np

    def process_input_chain(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        chain = self._compiled_input_chain
        if not chain:
            return audio
        try:
            audio_np = audio.detach().cpu().numpy()
            processed = self._run_compiled_chain_np(chain, audio_np, sample_rate)
            return torch.as_tensor(processed, dtype=audio.dtype, device=audio.device)
        except Exception as e:
            logger.error(f"Error in input effects chain: {e}")
            return audio

    def process_output_chain(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        chain = self._compiled_output_chain
        if not chain:
            return audio
        try:
            audio_np = audio.detach().cpu().numpy()
            processed = self._run_compiled_chain_np(chain, audio_np, sample_rate)
            return torch.as_tensor(processed, dtype=audio.dtype, device=audio.device)
        except Exception as e:
            logger.error(f"Error in output effects chain: {e}")
            return audio

    def get_effects_info(self) -> Dict[str, Any]:
        return {
            "input_effects": [effect.get_effect_info() for effect in self.input_effects],
            "output_effects": [effect.get_effect_info() for effect in self.output_effects],
            "available_effects": self.provider_registry.get_all_supported_effects(),
            "providers": self.provider_registry.get_providers_info()
        }

    def update_effect_parameters(
        self, effect_type: str, channel: AudioChannel, order: int, parameters: Dict[str, Any]
    ) -> bool:
        effects_list = self.input_effects if channel == "input" else (self.output_effects if channel == "output" else [])

        updated = False
        for effect in effects_list:
            if effect.effect_type == effect_type and effect.channel == channel and effect.order == order:
                effect.set_parameters(parameters)
                updated = True

        if updated:
            self._rebuild_compiled_chains()
        return updated

    def reorder_effects(self) -> None:
        """Re-sort effects by order in case orders were changed"""
        self.input_effects.sort(key=lambda x: x.order)
        self.output_effects.sort(key=lambda x: x.order)
        self._rebuild_compiled_chains()
        logger.info("Effects reordered by priority")

    def get_effect_by_id(self, effect_type: str, channel: AudioChannel, order: int) -> Optional[AudioEffect]:
        """Get a specific effect by type, channel and order"""
        effects_list = self.input_effects if channel == "input" else self.output_effects

        for effect in effects_list:
            if effect.effect_type == effect_type and effect.order == order:
                return effect
        return None

    def get_supported_effects(self) -> Dict[str, Dict[str, Any]]:
        """Get all supported effects with their parameter schemas"""
        return self.provider_registry.get_all_supported_effects()

    def get_providers_info(self) -> List[Dict[str, Any]]:
        """Get information about all registered providers"""
        return self.provider_registry.get_providers_info()

    def validate_effect_parameters(self, effect_type: str, parameters: Dict[str, Any]) -> bool:
        """Validate parameters for a specific effect type"""
        return self.provider_registry.validate_effect_parameters(effect_type, parameters)
