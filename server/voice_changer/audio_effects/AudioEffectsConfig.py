from copy import deepcopy
from typing import Dict, List, Any, Tuple
import logging
from .AudioEffect import AudioEffect
from .AudioEffectsManager import AudioEffectsManager

logger = logging.getLogger(__name__)


class AudioEffectsConfig:
    @staticmethod
    def parse_effects_from_config(config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parse audioEffects from frontend configuration
        """
        audio_effects = config.get("audioEffects", [])
        if not isinstance(audio_effects, list):
            logger.warning("audioEffects should be a list, got: %s", type(audio_effects))
            return []

        parsed_effects = []
        for i, effect_config in enumerate(audio_effects):
            try:
                if not isinstance(effect_config, dict):
                    logger.warning("Effect config at index %d is not a dict: %s", i, effect_config)
                    continue

                effect_type = effect_config.get("type")
                channel = effect_config.get("channel")
                enabled = bool(effect_config.get("enabled", True))
                parameters = effect_config.get("parameters", {})

                if not effect_type:
                    logger.warning("Effect at index %d missing 'type' field", i)
                    continue

                if channel not in ["input", "output"]:
                    logger.warning("Effect at index %d has invalid channel: %s", i, channel)
                    continue

                parsed_effect = {
                    "type": effect_type,
                    "channel": channel,
                    "enabled": enabled,
                    "parameters": parameters if isinstance(parameters, dict) else {},
                    "order": i,
                }

                parsed_effects.append(parsed_effect)
                logger.debug("Parsed effect: %s on %s channel", effect_type, channel)

            except Exception as e:
                logger.error("Error parsing effect at index %d: %s", i, e)
                continue

        return parsed_effects

    @staticmethod
    def build_updated_chains(
        effects_manager: AudioEffectsManager, settings: Dict[str, Any]
    ) -> Tuple[List[AudioEffect], List[AudioEffect], int, int]:
        """
        Build updated input and output effect lists incrementally without mutating `effects_manager`.
        Reuses existing instances (preserving stateful buffers/reverb tails) whenever
        (type, channel, order, parameters) match; only constructs new effects when changed.
        Safe to execute outside audio/device locks.
        """
        effects_config = AudioEffectsConfig.parse_effects_from_config(settings)

        existing_map: Dict[Tuple[str, str, int], AudioEffect] = {}
        for eff in list(effects_manager.input_effects) + list(effects_manager.output_effects):
            existing_map[(eff.effect_type, eff.channel, eff.order)] = eff

        new_input: List[AudioEffect] = []
        new_output: List[AudioEffect] = []

        for item in effects_config:
            effect_type = item["type"]
            channel = item["channel"]
            order = item["order"]
            enabled = item["enabled"]
            params = item["parameters"]

            existing = existing_map.get((effect_type, channel, order))
            if (
                existing is not None
                and getattr(existing, "_applied_config_params", None) == params
            ):
                existing.set_enabled(enabled)
                effect_obj = existing
            elif existing is not None and not hasattr(existing, "get_pedalboard_plugins"):
                if params:
                    if effects_manager.provider_registry.validate_effect_parameters(
                        effect_type, params
                    ):
                        existing.set_parameters(params)
                    else:
                        logger.warning(
                            f"Invalid parameters for effect {effect_type}, keeping previous"
                        )
                setattr(existing, "_applied_config_params", deepcopy(params))
                existing.set_enabled(enabled)
                effect_obj = existing
            else:
                try:
                    effect_obj = effects_manager.provider_registry.create_effect(
                        effect_type, channel, order
                    )
                    if effect_obj is None:
                        raise ValueError(f"Unknown effect type: {effect_type}")
                    if params:
                        if effects_manager.provider_registry.validate_effect_parameters(
                            effect_type, params
                        ):
                            effect_obj.set_parameters(params)
                        else:
                            logger.warning(
                                f"Invalid parameters for effect {effect_type}, using defaults"
                            )
                    setattr(effect_obj, "_applied_config_params", deepcopy(params))
                    effect_obj.set_enabled(enabled)
                except Exception as e:
                    logger.error("Failed to create %s effect %s: %s", channel, effect_type, e)
                    continue

            if channel == "input":
                new_input.append(effect_obj)
            else:
                new_output.append(effect_obj)

        active_count = sum(1 for e in effects_config if e.get("enabled", False))
        return new_input, new_output, len(effects_config), active_count

    @staticmethod
    def apply_effects_to_manager(
        effects_manager: AudioEffectsManager, effects_config: List[Dict[str, Any]]
    ) -> None:
        """Apply parsed effects configuration to AudioEffectsManager."""
        new_in, new_out, _, _ = AudioEffectsConfig.build_updated_chains(
            effects_manager, {"audioEffects": effects_config}
        )
        effects_manager.swap_chains(new_in, new_out)

    @staticmethod
    def configure_effects_from_settings(
        effects_manager: AudioEffectsManager, settings: Dict[str, Any]
    ) -> bool:
        """Configure effects manager from complete settings dictionary."""
        try:
            new_in, new_out, total_effects, active_count = AudioEffectsConfig.build_updated_chains(
                effects_manager, settings
            )
            effects_manager.swap_chains(new_in, new_out)

            if total_effects > 0:
                if active_count > 0:
                    logger.info(
                        "Successfully configured %d audio effects (%d active)",
                        total_effects,
                        active_count,
                    )
                else:
                    logger.info(
                        "Successfully configured %d audio effects (all disabled)",
                        total_effects,
                    )
            else:
                logger.debug("Successfully configured %d audio effects", total_effects)
            return True
        except Exception as e:
            logger.error("Failed to configure audio effects: %s", e)
            return False

    @staticmethod
    def validate_effect_config(effect_config: Dict[str, Any]) -> bool:
        """Validate a single effect configuration."""
        required_fields = ["type", "channel"]

        for field in required_fields:
            if field not in effect_config:
                logger.error("Effect config missing required field: %s", field)
                return False

        if effect_config["channel"] not in ["input", "output"]:
            logger.error("Invalid channel: %s", effect_config["channel"])
            return False

        return True
