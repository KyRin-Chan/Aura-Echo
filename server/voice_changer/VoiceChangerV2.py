import logging
from typing import Any, Union

import numpy as np
import torch
import torch.nn.functional as F

from const import VoiceChangerType
from data.ModelSlot import ModelSlots
from Exceptions import VoiceChangerIsNotSelectedException
from voice_changer.common.AudioDSP import AdaptiveNoiseGate, StatefulPeakLimiter, soft_knee_limit_np
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
from voice_changer.IORecorder import IORecorder
from voice_changer.utils.Timer import Timer2
from voice_changer.utils.VoiceChangerModel import AudioInOutFloat, VoiceChangerModel
from voice_changer.VoiceChangerSettings import VoiceChangerSettings

logger = logging.getLogger(__name__)


class VoiceChangerV2:
    def __init__(self, settings: VoiceChangerSettings):
        self.settings = settings

        self.block_frame = self._compute_block_frame()
        self.crossfade_frame = int(self.settings.crossFadeOverlapSize * self.settings.inputSampleRate)
        self.extra_frame = int(self.settings.extraConvertSize * self.settings.inputSampleRate)
        self.sola_search_frame = max(1, int(round(0.012 * self.settings.inputSampleRate)))

        self.vcmodel: VoiceChangerModel | None = None
        self.device_manager = DeviceManager.get_instance()
        self.sola_buffer: torch.Tensor | None = None
        self._zero_offset: torch.Tensor | None = None
        self._was_voiced: bool = False
        self._peak_limiter = StatefulPeakLimiter(threshold=0.85, ceiling=0.98)

        self.io_recorder = IORecorder(
            self.settings.inputSampleRate,
            self.settings.outputSampleRate,
        )
        if self.settings.recordIO:
            self.io_recorder.open(self.settings.inputSampleRate, self.settings.outputSampleRate)
        self._generate_strength()

        # Optional background audio mixer & shared audio effects manager
        self.audio_mixer = None
        self.audio_effects_manager = None
        self._suppress_warmup: int = 0
        self._pt_gate = AdaptiveNoiseGate()
        self._pt_was_gated_open: bool = False
        self._pt_resampler = None
        self._vc_dirty: bool = False

    def begin_batch_update(self) -> None:
        self._suppress_warmup += 1

    def end_batch_update(self) -> None:
        self._suppress_warmup = max(0, self._suppress_warmup - 1)

    def _compute_block_frame(self) -> int:
        return max(
            128,
            int(round((self.settings.serverReadChunkSize * 128 / 48000.0) * self.settings.inputSampleRate)),
        )

    def set_audio_mixer(self, mixer):
        self.audio_mixer = mixer

    def set_audio_effects_manager(self, manager):
        self.audio_effects_manager = manager
        if self.vcmodel is not None and hasattr(self.vcmodel, "set_audio_effects_manager"):
            self.vcmodel.set_audio_effects_manager(manager)

    def _warmup_post_sola(self, audio: torch.Tensor) -> None:
        """Warm up SOLA conv1d kernels and GPU post-formant filter without touching CPU effects/mixer."""
        if audio is None or audio.numel() == 0:
            return
        needed_conv = self.crossfade_frame + self.sola_search_frame
        if self.sola_buffer is not None and self.sola_ones is not None and needed_conv > 0:
            if audio.shape[0] < needed_conv:
                conv_in_1d = F.pad(audio, (0, needed_conv - audio.shape[0]))
            else:
                conv_in_1d = audio[:needed_conv]
            conv_input = conv_in_1d[None, None, :]
            cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
            cor_den = torch.sqrt(F.conv1d(conv_input ** 2, self.sola_ones) + 1e-8)
            _ = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])

        needed_out = self.block_frame + self.crossfade_frame
        if audio.shape[0] < needed_out:
            mixed = F.pad(audio, (0, needed_out - audio.shape[0]))[: self.block_frame]
        else:
            mixed = audio[: self.block_frame]
        if self.vcmodel is not None and hasattr(self.vcmodel, "formant_filter") and self.vcmodel.formant_filter is not None:
            _ = self.vcmodel.formant_filter.apply_post(mixed)

    def warmup(self, step: int = 5, total: int = 5) -> None:
        if self._suppress_warmup > 0 or self.vcmodel is None:
            return
        if getattr(type(self.vcmodel), "warmup", None) is None:
            return
        if not getattr(self.vcmodel, "_init_succeeded", True):
            return
        try:
            self.vcmodel.warmup(
                block_frame=self.block_frame,
                on_audio_out=self._warmup_post_sola,
                step=step,
                total=total,
            )
        finally:
            self._was_voiced = False
            self._vc_dirty = False
            self._peak_limiter.reset()
            if self.sola_buffer is not None:
                self.sola_buffer.zero_()

    def initialize(self, vcmodel: VoiceChangerModel):
        self.vcmodel = vcmodel
        if self.audio_effects_manager is not None and hasattr(self.vcmodel, "set_audio_effects_manager"):
            self.vcmodel.set_audio_effects_manager(self.audio_effects_manager)
        self.vcmodel.realloc(self.block_frame, self.extra_frame, self.crossfade_frame, self.sola_search_frame)
        self.vcmodel.initialize()
        self._was_voiced = False
        self._vc_dirty = False
        self._peak_limiter.reset()
        if self.sola_buffer is not None:
            self.sola_buffer.zero_()
        self.warmup(step=5, total=5)

    def set_slot_info(self, slot_info: ModelSlots):
        self.vcmodel.set_slot_info(slot_info)
        if self.audio_effects_manager is not None and hasattr(self.vcmodel, "set_audio_effects_manager"):
            self.vcmodel.set_audio_effects_manager(self.audio_effects_manager)
        self.vcmodel.realloc(self.block_frame, self.extra_frame, self.crossfade_frame, self.sola_search_frame)
        self.vcmodel.initialize()
        self._was_voiced = False
        self._vc_dirty = False
        self._peak_limiter.reset()
        if self.sola_buffer is not None:
            self.sola_buffer.zero_()
        self.warmup(step=5, total=5)

    def get_type(self) -> VoiceChangerType:
        if self.vcmodel is None:
            return "None"
        return self.vcmodel.voiceChangerType

    def set_input_sample_rate(self):
        if self.settings.recordIO:
            self.io_recorder.open(self.settings.inputSampleRate, self.settings.outputSampleRate)

        self.block_frame = self._compute_block_frame()
        self.extra_frame = int(self.settings.extraConvertSize * self.settings.inputSampleRate)
        self.crossfade_frame = int(self.settings.crossFadeOverlapSize * self.settings.inputSampleRate)
        self.sola_search_frame = max(1, int(round(0.012 * self.settings.inputSampleRate)))
        self._generate_strength()
        self._peak_limiter.reset()
        self._pt_gate.reset()
        self._pt_was_gated_open = False
        self._pt_resampler = None

        if self.vcmodel is not None:
            self.vcmodel.set_sampling_rate(self.settings.inputSampleRate, self.settings.outputSampleRate)
            self.vcmodel.realloc(self.block_frame, self.extra_frame, self.crossfade_frame, self.sola_search_frame)

    def set_output_sample_rate(self):
        if self.settings.recordIO:
            self.io_recorder.open(self.settings.inputSampleRate, self.settings.outputSampleRate)

        self._generate_strength()
        self._peak_limiter.reset()
        self._pt_gate.reset()
        self._pt_was_gated_open = False
        self._pt_resampler = None
        if self.vcmodel is not None:
            self.vcmodel.set_sampling_rate(self.settings.inputSampleRate, self.settings.outputSampleRate)

    def get_info(self):
        if self.vcmodel is not None:
            return self.vcmodel.get_info()
        return {}

    def update_settings(self, key: str, val: Any, old_val: Any):
        if key == "serverReadChunkSize":
            self.block_frame = self._compute_block_frame()
        elif key == 'gpu':
            self._generate_strength()
        elif key == "inputSampleRate":
            self.set_input_sample_rate()
        elif key == "outputSampleRate":
            self.set_output_sample_rate()
        elif key == 'extraConvertSize':
            self.extra_frame = int(val * self.settings.inputSampleRate)
        elif key == 'crossFadeOverlapSize':
            self.crossfade_frame = int(val * self.settings.inputSampleRate)
            self._generate_strength()
        elif key == 'recordIO':
            enable = int(val)
            if enable == 1:
                self.io_recorder.open(self.settings.inputSampleRate, self.settings.outputSampleRate)
            else:
                self.io_recorder.close()
        elif key == 'passThrough':
            self._pt_gate.reset()
            self._pt_was_gated_open = False
            if self._pt_resampler is not None:
                self._pt_resampler.reset()

        if self.vcmodel is not None:
            if key in {'gpu', 'serverReadChunkSize', 'extraConvertSize', 'crossFadeOverlapSize', 'silenceFront', 'forceFp32'}:
                self.vcmodel.realloc(self.block_frame, self.extra_frame, self.crossfade_frame, self.sola_search_frame)
                self._was_voiced = False
                self._vc_dirty = False
                self._peak_limiter.reset()
                if self.sola_buffer is not None:
                    self.sola_buffer.zero_()

            self.vcmodel.update_settings(key, val, old_val)

            if key in {'gpu', 'forceFp32', 'disableJit', 'useTensorRT', 'useONNX'}:
                self.warmup(step=5, total=5)
            elif key == 'f0Detector':
                self.warmup(step=2, total=2)
            elif key in {'serverReadChunkSize', 'extraConvertSize', 'crossFadeOverlapSize', 'silenceFront', 'inputSampleRate', 'outputSampleRate', 'formantShift', 'indexRatio'}:
                self.warmup(step=1, total=1)

    def _generate_strength(self):
        linspace = torch.linspace(
            0.0,
            1.0,
            steps=self.crossfade_frame,
            device=self.device_manager.device,
            dtype=torch.float32,
        )
        # Equal-gain Hann crossfade (sin^2 + cos^2 == 1) prevents +3 dB bumps on phase-aligned SOLA signals
        self.fade_in_window: torch.Tensor = torch.sin(0.5 * np.pi * linspace) ** 2
        self.fade_out_window: torch.Tensor = torch.cos(0.5 * np.pi * linspace) ** 2

        self.sola_buffer = torch.zeros(self.crossfade_frame, device=self.device_manager.device, dtype=torch.float32)
        self.sola_ones = torch.ones(1, 1, self.crossfade_frame, device=self.device_manager.device, dtype=torch.float32)
        self._zero_offset = torch.tensor(0, device=self.device_manager.device, dtype=torch.int64)
        self._was_voiced = False
        logger.info(f'Allocated SOLA buffer size: {self.crossfade_frame}')

    def get_processing_sampling_rate(self) -> int:
        if self.vcmodel is None:
            return 0
        return self.vcmodel.get_processing_sampling_rate()

    def _flush_silence_sola_np(self, block_size: int) -> np.ndarray:
        """Fade out residual SOLA overlap buffer smoothly on voiced->silence transition."""
        mixed_np = np.zeros(block_size, dtype=np.float32)
        if self._was_voiced and self.sola_buffer is not None:
            fade_len = min(block_size, self.crossfade_frame)
            if fade_len > 0:
                tail_t = self.sola_buffer[:fade_len] * self.fade_out_window[:fade_len]
                mixed_np[:fade_len] = tail_t.detach().cpu().numpy()
            self.sola_buffer.zero_()
            self._was_voiced = False
        return mixed_np

    def _apply_limiter_np(self, mixed_np: np.ndarray) -> np.ndarray:
        """Apply optional stateful AGC peak limiter and soft-knee safety ceiling."""
        if getattr(self.settings, "agcEnabled", False):
            mixed_np = self._peak_limiter.process(mixed_np, sample_rate=self.settings.outputSampleRate)
            mixed_np = soft_knee_limit_np(mixed_np, threshold=0.985, ceiling=0.999)
        else:
            mixed_np = soft_knee_limit_np(mixed_np, threshold=0.95, ceiling=0.999)
        return mixed_np

    def _apply_post_cpu_chain(self, mixed_np: np.ndarray, block_size: int) -> np.ndarray:
        """Run non-overlapping output effects, background mixing, and smooth soft-knee AGC on CPU."""
        if self.audio_effects_manager is not None:
            mixed_np = self.audio_effects_manager.process_output_chain_np(
                mixed_np, sample_rate=self.settings.outputSampleRate
            )

        if self.audio_mixer is not None:
            try:
                bg_np = self.audio_mixer.render_numpy(block_size, self.settings.outputSampleRate)
                if bg_np is not None:
                    mixed_np = mixed_np + bg_np
            except Exception:
                pass

        return self._apply_limiter_np(mixed_np)

    def _apply_passthrough_gate_np(self, audio_in: np.ndarray) -> tuple[np.ndarray, float]:
        if audio_in.size == 0:
            return np.empty(0, dtype=np.float32), 0.0

        gate_active, vol = self._pt_gate.evaluate(
            audio_in,
            sample_rate=self.settings.inputSampleRate,
            silent_threshold_db=self.settings.silentThreshold,
            vad_enabled=bool(getattr(self.settings, "vadEnabled", False)),
            vad_threshold=float(getattr(self.settings, "vadThreshold", 0.45)),
        )

        if gate_active:
            out = np.array(audio_in, dtype=np.float32, copy=True)
            if not self._pt_was_gated_open:
                fade_len = min(64, int(out.shape[0]))
                if fade_len > 1:
                    lin = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
                    out[:fade_len] *= np.square(np.sin(0.5 * np.pi * lin)).astype(np.float32)
                elif fade_len == 1:
                    out[0] = 0.0
                self._pt_was_gated_open = True
            return out, vol

        out = np.zeros(audio_in.shape[0], dtype=np.float32)
        if self._pt_was_gated_open:
            fade_len = min(128, int(audio_in.shape[0]))
            if fade_len > 1:
                lin = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
                out[:fade_len] = audio_in[:fade_len] * np.square(np.cos(0.5 * np.pi * lin)).astype(np.float32)
            self._pt_was_gated_open = False
        return out, vol

    def process_passthrough(self, audio_in: AudioInOutFloat) -> tuple[AudioInOutFloat, float]:
        audio_np = np.asarray(audio_in, dtype=np.float32).reshape(-1)
        block_size = int(audio_np.shape[0])
        if block_size == 0:
            return np.empty(0, dtype=np.float32), 0.0

        use_effects = bool(getattr(self.settings, "passThroughEffects", False))
        if use_effects and self.audio_effects_manager is not None:
            audio_np = self.audio_effects_manager.process_input_chain_np(
                audio_np, sample_rate=self.settings.inputSampleRate
            )

        gated_np, vol = self._apply_passthrough_gate_np(audio_np)

        in_sr = int(self.settings.inputSampleRate)
        out_sr = int(self.settings.outputSampleRate)
        if in_sr != out_sr and gated_np.size > 0:
            if (
                self._pt_resampler is None
                or self._pt_resampler.orig_freq != in_sr
                or self._pt_resampler.new_freq != out_sr
            ):
                from voice_changer.common.ServerAEC import NumpyCausalPolyphaseResampler

                self._pt_resampler = NumpyCausalPolyphaseResampler(in_sr, out_sr)
            gated_np = self._pt_resampler.process(gated_np)

        out_block_size = int(gated_np.shape[0])
        if self._was_voiced and out_block_size > 0:
            fade_len = min(out_block_size, self.crossfade_frame)
            sola_tail = self._flush_silence_sola_np(out_block_size)
            if fade_len > 0:
                fade_in_np = self.fade_in_window[:fade_len].detach().cpu().numpy()
                gated_np[:fade_len] *= fade_in_np
            gated_np = gated_np + sola_tail
        elif self._was_voiced:
            if self.sola_buffer is not None:
                self.sola_buffer.zero_()
            self._was_voiced = False

        if self._vc_dirty:
            if self.vcmodel is not None and hasattr(self.vcmodel, "reset_state"):
                self.vcmodel.reset_state()
            self._vc_dirty = False

        if use_effects:
            mixed_np = self._apply_post_cpu_chain(gated_np, out_block_size)
        else:
            mixed_np = self._apply_limiter_np(gated_np)
        return mixed_np, vol

    def process_audio(self, audio_in: AudioInOutFloat) -> tuple[AudioInOutFloat, float]:
        self._vc_dirty = True
        block_size = int(audio_in.shape[0])

        if block_size > 0 and block_size != self.block_frame and self.vcmodel is not None:
            self.block_frame = block_size
            self.vcmodel.realloc(self.block_frame, self.extra_frame, self.crossfade_frame, self.sola_search_frame)
            self._was_voiced = False
            self._peak_limiter.reset()
            if self.sola_buffer is not None:
                self.sola_buffer.zero_()

        # Input effects on continuous non-overlapping incoming block
        if self.audio_effects_manager is not None:
            audio_in = self.audio_effects_manager.process_input_chain_np(
                audio_in, sample_rate=self.settings.inputSampleRate
            )

        audio, vol = self.vcmodel.inference(audio_in)

        if audio is None:
            mixed_np = self._flush_silence_sola_np(block_size)
            mixed_np = self._apply_post_cpu_chain(mixed_np, block_size)
            return mixed_np, vol

        # Single-sync SOLA alignment
        sola_energy = torch.mean(self.sola_buffer ** 2)
        conv_input = audio[None, None, : self.crossfade_frame + self.sola_search_frame]
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(F.conv1d(conv_input ** 2, self.sola_ones) + 1e-8)
        best_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])
        sola_offset = int(torch.where(sola_energy < 1e-6, self._zero_offset, best_offset).item())

        audio = audio[sola_offset:]
        needed = block_size + self.crossfade_frame
        if audio.shape[0] < needed:
            audio = F.pad(audio, (0, needed - audio.shape[0]))

        audio[: self.crossfade_frame] *= self.fade_in_window
        audio[: self.crossfade_frame] += self.sola_buffer * self.fade_out_window

        self.sola_buffer[:] = audio[block_size : block_size + self.crossfade_frame]
        self._was_voiced = True

        mixed = audio[: block_size]

        # Post-filter (minimum-phase formant filter) on SOLA-stitched continuous stream
        if hasattr(self.vcmodel, "formant_filter") and self.vcmodel.formant_filter is not None:
            mixed = self.vcmodel.formant_filter.apply_post(mixed)

        # Single GPU->CPU transfer for the entire pipeline
        mixed_np = mixed.detach().cpu().numpy().astype(np.float32, copy=False)
        mixed_np = self._apply_post_cpu_chain(mixed_np, block_size)
        return mixed_np, vol

    def skip_audio(self, audio_in: AudioInOutFloat) -> tuple[AudioInOutFloat, float]:
        """Advance audio context without running heavy neural inference when packets arrive stale."""
        self._vc_dirty = True
        block_size = audio_in.shape[0]
        if self.audio_effects_manager is not None:
            audio_in = self.audio_effects_manager.process_input_chain_np(
                audio_in, sample_rate=self.settings.inputSampleRate
            )
        if hasattr(self.vcmodel, "feed_context"):
            self.vcmodel.feed_context(audio_in)

        mixed_np = self._flush_silence_sola_np(block_size)
        mixed_np = self._apply_post_cpu_chain(mixed_np, block_size)
        vol = float(np.sqrt(np.mean(np.square(audio_in, dtype=np.float32)))) if audio_in.size > 0 else 0.0
        return mixed_np, vol

    @torch.no_grad()
    def on_request(self, audio_in: AudioInOutFloat) -> tuple[AudioInOutFloat, float, list[Union[int, float]]]:
        if self.vcmodel is None:
            raise VoiceChangerIsNotSelectedException("Voice Changer is not selected.")

        with Timer2("main-process", True) as t:
            result, vol = self.process_audio(audio_in)

        mainprocess_time = t.secs

        if self.settings.recordIO:
            self.io_recorder.write_input((np.clip(audio_in, -1.0, 1.0) * 32767).astype(np.int16).tobytes())
            self.io_recorder.write_output((np.clip(result, -1.0, 1.0) * 32767).astype(np.int16).tobytes())

        return result, vol, [0, mainprocess_time, 0]

    @torch.no_grad()
    def on_passthrough_request(self, audio_in: AudioInOutFloat) -> tuple[AudioInOutFloat, float, list[Union[int, float]]]:
        with Timer2("main-process", True) as t:
            result, vol = self.process_passthrough(audio_in)

        mainprocess_time = t.secs

        if self.settings.recordIO:
            self.io_recorder.write_input((np.clip(audio_in, -1.0, 1.0) * 32767).astype(np.int16).tobytes())
            self.io_recorder.write_output((np.clip(result, -1.0, 1.0) * 32767).astype(np.int16).tobytes())

        return result, vol, [0, mainprocess_time, 0]

    @torch.no_grad()
    def on_skip_request(self, audio_in: AudioInOutFloat) -> tuple[AudioInOutFloat, float, list[Union[int, float]]]:
        if self.vcmodel is None:
            raise VoiceChangerIsNotSelectedException("Voice Changer is not selected.")
        result, vol = self.skip_audio(audio_in)
        return result, vol, [0.0, 0.0, 0.0]

    @torch.no_grad()
    def export2onnx(self):
        return self.vcmodel.export2onnx()

    def get_current_model_settings(self) -> dict:
        return self.vcmodel.get_model_current()
