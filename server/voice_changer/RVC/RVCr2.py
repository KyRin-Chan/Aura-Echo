"""
VoiceChangerV2向け
"""
import logging
import os
from time import perf_counter
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torchaudio import transforms as tat

from const import EnumInferenceTypes, HUBERT_SAMPLE_RATE, WINDOW_SIZE
from data.ModelSlot import RVCModelSlot, saveSlotInfo
from Exceptions import PipelineNotInitializedException
from settings import get_settings
from voice_changer.common.AudioDSP import (
    AdaptiveNoiseGate,
    CausalPolyphaseResampler,
    MinimumPhaseFormantFilter,
    _select_highest_energy_segment,
    _to_mono_float32,
)
from voice_changer.common.TorchUtils import circular_write
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
from voice_changer.embedder.EmbedderManager import EmbedderManager
from voice_changer.pitch_extractor.PitchExtractorManager import PitchExtractorManager
from voice_changer.RVC.onnx_exporter.export2onnx import export2onnx
from voice_changer.RVC.pipeline.Pipeline import Pipeline
from voice_changer.RVC.pipeline.PipelineGenerator import createPipeline
from voice_changer.utils.VoiceChangerModel import AudioInOutFloat, VoiceChangerModel
from voice_changer.VoiceChangerSettings import VoiceChangerSettings
from voice_changer.vocoder.VocoderManager import VocoderManager

logger = logging.getLogger(__name__)


class BypassResampler(torch.nn.Module):
    def forward(self, x):
        return x


class RVCr2(VoiceChangerModel):
    def __init__(self, slotInfo: RVCModelSlot, settings: VoiceChangerSettings):
        self.voiceChangerType = "RVC"

        self.device_manager = DeviceManager.get_instance()
        EmbedderManager.initialize()
        PitchExtractorManager.initialize()
        self.settings = settings
        self.params = get_settings()

        self.pipeline: Pipeline | None = None
        self._audio_effects_manager = None

        self.convert_buffer: torch.Tensor | None = None
        self.pitch_buffer: torch.Tensor | None = None
        self.pitchf_buffer: torch.Tensor | None = None
        self._audio_scratch: torch.Tensor | None = None
        self._convert_scratch: torch.Tensor | None = None
        self._pitch_scratch: torch.Tensor | None = None
        self._pitchf_scratch: torch.Tensor | None = None

        self.return_length = 0
        self.skip_head = 0
        self.silence_front = 0
        self.slotInfo = slotInfo

        self.resampler_in: CausalPolyphaseResampler | None = None
        self.resampler_out: torch.nn.Module | None = None
        self.formant_filter = MinimumPhaseFormantFilter().to(self.device_manager.device)

        self.input_sample_rate = self.settings.inputSampleRate
        self.output_sample_rate = self.settings.outputSampleRate

        # Convert dB to RMS
        self.inputSensitivity = 10 ** (self.settings.silentThreshold / 20)

        self.is_half = self.device_manager.use_fp16()
        self.dtype = torch.float16 if self.is_half else torch.float32
        self._progress_callback: Optional[Callable[[int, int, str], None]] = None

        self._pitch_sample_acc: int = 0
        self._rms_gain_ema_t: torch.Tensor | None = None
        self._noise_gate = AdaptiveNoiseGate()
        self._last_block_frame: int = 0
        self._last_pitch_hop_frames: int = 4
        self._init_succeeded: bool = False

    @property
    def _gate_open(self) -> bool:
        return self._noise_gate.is_open

    @_gate_open.setter
    def _gate_open(self, value: bool) -> None:
        self._noise_gate.is_open = bool(value)

    @property
    def _gate_hangover_chunks(self) -> int:
        return self._noise_gate.hangover_chunks

    @_gate_hangover_chunks.setter
    def _gate_hangover_chunks(self, value: int) -> None:
        self._noise_gate.hangover_chunks = int(value)

    def set_audio_effects_manager(self, manager) -> None:
        self._audio_effects_manager = manager
        if self.pipeline is not None:
            self.pipeline.audio_effects_manager = manager

    def set_progress_callback(self, callback: Optional[Callable[[int, int, str], None]]):
        """Register a callback to receive model loading progress updates."""
        self._progress_callback = callback

    def _emit_progress(
        self,
        step: int,
        total: int,
        label: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> None:
        cb = progress_callback if progress_callback is not None else self._progress_callback
        if cb is not None:
            try:
                cb(step, total, label)
            except Exception:
                pass

    def initialize(self, force_reload: bool = False, progress_callback: Optional[Callable[[int, int, str], None]] = None):
        logger.info("Initializing...")
        self._init_succeeded = False

        if self.settings.useONNX and not self.slotInfo.modelFileOnnx:
            self.export2onnx()

        _cb = progress_callback if progress_callback is not None else self._progress_callback

        try:
            self.pipeline = createPipeline(
                self.slotInfo,
                self.settings.f0Detector,
                self.settings.useONNX,
                force_reload,
                progress_callback=_cb,
                audio_effects_manager=self._audio_effects_manager,
            )
            if self.pipeline is not None:
                self.pipeline.settings = self.settings
                if self._audio_effects_manager is None:
                    self._audio_effects_manager = self.pipeline.audio_effects_manager

            if self.pipeline is not None and hasattr(self.settings, 'audioEffects'):
                self.pipeline.configure_audio_effects(self.settings.to_dict())
                logger.info("Initial audio effects configuration loaded")
        except Exception as e:  # NOQA
            logger.error("Failed to create pipeline.")
            logger.exception(e)
            return

        # 処理は16Kで実施(Pitch, embed, (infer))
        self.resampler_in = CausalPolyphaseResampler(
            orig_freq=self.input_sample_rate,
            new_freq=HUBERT_SAMPLE_RATE,
        ).to(self.device_manager.device)

        if self.slotInfo.samplingRate == self.output_sample_rate:
            self.resampler_out = BypassResampler().to(self.device_manager.device)
        else:
            self.resampler_out = tat.Resample(
                orig_freq=self.slotInfo.samplingRate,
                new_freq=self.output_sample_rate,
                dtype=torch.float32,
            ).to(self.device_manager.device)

        try:
            VocoderManager.get_instance().update_target_sr(self.output_sample_rate)
        except Exception:
            pass

        self.formant_filter = MinimumPhaseFormantFilter().to(self.device_manager.device)
        self._sync_formant_filter()

        try:
            model_vocoder = getattr(self.slotInfo, "vocoder", "embedded")
            logger.info(f"Model initialization auto-detected model vocoder → '{model_vocoder}'")
        except Exception as _e:
            logger.warning(f"Could not check model vocoder: {_e}")

        self._init_succeeded = True
        logger.info("Initialized.")

    def set_sampling_rate(self, input_sample_rate: int, output_sample_rate: int):
        if self.input_sample_rate != input_sample_rate:
            self.input_sample_rate = input_sample_rate
            self.resampler_in = CausalPolyphaseResampler(
                orig_freq=self.input_sample_rate,
                new_freq=HUBERT_SAMPLE_RATE,
            ).to(self.device_manager.device)
        if self.output_sample_rate != output_sample_rate:
            self.output_sample_rate = output_sample_rate
            if self.slotInfo.samplingRate == self.output_sample_rate:
                self.resampler_out = BypassResampler().to(self.device_manager.device)
            else:
                self.resampler_out = tat.Resample(
                    orig_freq=self.slotInfo.samplingRate,
                    new_freq=self.output_sample_rate,
                    dtype=torch.float32,
                ).to(self.device_manager.device)
            try:
                VocoderManager.get_instance().update_target_sr(self.output_sample_rate)
            except Exception:
                pass
            self.formant_filter.reset()

    def change_pitch_extractor(self):
        self._emit_progress(1, 2, "Loading pitch extractor...")
        pitchExtractor = PitchExtractorManager.getPitchExtractor(
            self.settings.f0Detector, False
        )
        self.pipeline.setPitchExtractor(pitchExtractor)

    def update_settings(self, key: str, val, old_val):
        if key in {"gpu", "forceFp32", "disableJit", "useTensorRT"}:
            self.is_half = self.device_manager.use_fp16()
            self.dtype = torch.float16 if self.is_half else torch.float32
            self.initialize(True)
        elif key == 'useONNX':
            self.initialize()
        elif key == "f0Detector" and self.pipeline is not None:
            self.change_pitch_extractor()
        elif key == 'silentThreshold':
            self.inputSensitivity = 10 ** (self.settings.silentThreshold / 20)

    def set_slot_info(self, slotInfo: RVCModelSlot):
        self.slotInfo = slotInfo

    def get_info(self):
        data = {}
        if self.pipeline is not None:
            pipelineInfo = self.pipeline.getPipelineInfo()
            data["pipelineInfo"] = pipelineInfo
        else:
            data["pipelineInfo"] = "None"
        return data

    def get_processing_sampling_rate(self):
        return self.slotInfo.samplingRate

    def realloc(self, block_frame: int, extra_frame: int, crossfade_frame: int, sola_search_frame: int):
        self.is_half = self.device_manager.use_fp16()
        self.dtype = torch.float16 if self.is_half else torch.float32
        self._last_block_frame = int(block_frame)

        block_frame_16k = int(block_frame / self.input_sample_rate * HUBERT_SAMPLE_RATE)
        crossfade_frame_16k = int(crossfade_frame / self.input_sample_rate * HUBERT_SAMPLE_RATE)
        sola_search_frame_16k = int(sola_search_frame / self.input_sample_rate * HUBERT_SAMPLE_RATE)
        extra_frame_16k = int(extra_frame / self.input_sample_rate * HUBERT_SAMPLE_RATE)

        return_size_16k = block_frame_16k + sola_search_frame_16k + crossfade_frame_16k
        return_frames = (return_size_16k + WINDOW_SIZE - 1) // WINDOW_SIZE
        extra_frames = extra_frame_16k // WINDOW_SIZE
        total_frames = return_frames + extra_frames
        # Align total frames to even count (WINDOW_SIZE * 2 = 320 @16kHz = HuBERT stride)
        # by absorbing 1 padding frame into left context (skip_head) so SOLA right-alignment stays exact.
        if total_frames % 2 != 0:
            total_frames += 1
            extra_frames += 1

        self.convert_feature_size_16k = total_frames
        convert_size_16k = total_frames * WINDOW_SIZE

        self.skip_head = extra_frames
        self.return_length = return_frames
        self.silence_front = max(0, (self.skip_head - 12) * WINDOW_SIZE) if self.settings.silenceFront else 0

        audio_buffer_size = block_frame_16k + crossfade_frame_16k
        dev = self.device_manager.device
        self.audio_buffer = torch.zeros(audio_buffer_size, dtype=torch.float32, device=dev)
        self.convert_buffer = torch.zeros(convert_size_16k, dtype=torch.float32, device=dev)
        self.pitch_buffer = torch.zeros(self.convert_feature_size_16k + 1, dtype=torch.int64, device=dev)
        self.pitchf_buffer = torch.zeros(self.convert_feature_size_16k + 1, dtype=self.dtype, device=dev)

        self._audio_scratch = torch.empty_like(self.audio_buffer)
        self._convert_scratch = torch.empty_like(self.convert_buffer)
        self._pitch_scratch = torch.empty_like(self.pitch_buffer)
        self._pitchf_scratch = torch.empty_like(self.pitchf_buffer)

        self._pitch_sample_acc = 0
        self._rms_gain_ema_t = None
        self._noise_gate.reset()
        if self.resampler_in is not None:
            self.resampler_in.reset()
        self.formant_filter.reset()

        logger.info(f'Allocated audio buffer size: {audio_buffer_size}')
        logger.info(f'Allocated convert buffer size: {convert_size_16k}')
        logger.info(f'Allocated pitchf buffer size: {self.convert_feature_size_16k + 1}')

    def reset_state(self) -> None:
        """Reset streaming buffers and filter/F0 state without reallocating memory."""
        for attr in ("audio_buffer", "convert_buffer", "pitch_buffer", "pitchf_buffer"):
            buf = getattr(self, attr, None)
            if buf is not None:
                buf.zero_()
        self._pitch_sample_acc = 0
        self._rms_gain_ema_t = None
        self._noise_gate.reset()
        if self.resampler_in is not None:
            self.resampler_in.reset()
        if self.formant_filter is not None:
            self.formant_filter.reset()
        if self.pipeline is not None:
            self.pipeline.last_f0_smoothed = None
            if hasattr(self.pipeline.pitchExtractor, "reset"):
                self.pipeline.pitchExtractor.reset()

    def _sync_formant_filter(self) -> None:
        self.formant_filter.update_config(
            active=getattr(self.settings, "formantProfileActive", False),
            target_env=getattr(self.settings, "formantProfileTargetEnvelope", "[]"),
            target_sr=getattr(self.settings, "formantProfileTargetSr", 0),
            input_env=getattr(self.settings, "formantProfileInputEnvelope", "[]"),
            input_sr=getattr(self.settings, "formantProfileInputSr", 0),
            strength=getattr(self.settings, "formantProfileStrength", 0.35),
            pre_sr=HUBERT_SAMPLE_RATE,
            post_sr=self.output_sample_rate,
            device=self.device_manager.device,
        )

    def _shift_pitch_buffers_zero(self, pitch_hop_frames: int) -> None:
        if self.pitch_buffer is None or self.pitchf_buffer is None:
            return
        hop = min(max(0, pitch_hop_frames), self.pitchf_buffer.shape[0])
        if hop > 0:
            keep = self.pitchf_buffer.shape[0] - hop
            if keep > 0:
                if self._pitch_scratch is not None:
                    self._pitch_scratch[:keep].copy_(self.pitch_buffer[hop:])
                    self.pitch_buffer[:keep].copy_(self._pitch_scratch[:keep])
                else:
                    self.pitch_buffer[:keep] = self.pitch_buffer[hop:].detach().clone()
                if self._pitchf_scratch is not None:
                    self._pitchf_scratch[:keep].copy_(self.pitchf_buffer[hop:])
                    self.pitchf_buffer[:keep].copy_(self._pitchf_scratch[:keep])
                else:
                    self.pitchf_buffer[:keep] = self.pitchf_buffer[hop:].detach().clone()
            self.pitch_buffer[-hop:].zero_()
            self.pitchf_buffer[-hop:].zero_()
        if self.pipeline is not None:
            self.pipeline.last_f0_smoothed = None
            if hasattr(self.pipeline.pitchExtractor, "reset"):
                self.pipeline.pitchExtractor.reset()

    def _prepare_16k_input(self, audio_in: AudioInOutFloat) -> tuple[torch.Tensor, int]:
        audio_in_t = torch.as_tensor(audio_in, dtype=torch.float32, device=self.device_manager.device)
        audio_in_16k = self.resampler_in(audio_in_t)
        self._sync_formant_filter()

        self._pitch_sample_acc += int(audio_in_16k.shape[0])
        pitch_hop_frames = self._pitch_sample_acc // WINDOW_SIZE
        self._pitch_sample_acc %= WINDOW_SIZE

        return audio_in_16k, pitch_hop_frames

    def feed_context(self, audio_in: AudioInOutFloat) -> None:
        """Advance resampler/conversion context buffers when a chunk's inference is skipped."""
        if self.pipeline is None or self.resampler_in is None:
            return
        audio_in_16k, pitch_hop_frames = self._prepare_16k_input(audio_in)
        circular_write(audio_in_16k, self.audio_buffer, scratch=self._audio_scratch)
        circular_write(audio_in_16k, self.convert_buffer, scratch=self._convert_scratch)
        self._shift_pitch_buffers_zero(pitch_hop_frames)

    @torch.no_grad()
    def convert_offline(
        self,
        audio_in: AudioInOutFloat,
        sample_rate: int,
        max_duration_sec: float = 12.0,
    ) -> tuple[np.ndarray, int]:
        """
        Convert an offline dry vocal sample through the active RVC pipeline using
        streaming-aligned chunk shapes (preserving TensorRT/ONNX/JIT shape caches
        and leaving live streaming buffers untouched).
        """
        if self.pipeline is None:
            raise PipelineNotInitializedException()

        sr_in = max(1, int(sample_rate))
        if isinstance(audio_in, torch.Tensor):
            audio_np = audio_in.detach().cpu().numpy()
        else:
            audio_np = np.asarray(audio_in, dtype=np.float32)
        audio_np = _to_mono_float32(audio_np)

        out_sr = int(self.output_sample_rate or self.slotInfo.samplingRate or 48000)
        if audio_np.size == 0:
            return np.empty(0, dtype=np.float32), out_sr

        audio_np = _select_highest_energy_segment(audio_np, sr_in, max_duration_sec)
        dev = self.device_manager.device

        audio_t = torch.as_tensor(audio_np, dtype=torch.float32, device=dev)
        if sr_in != HUBERT_SAMPLE_RATE:
            resampler_temp = tat.Resample(
                orig_freq=sr_in,
                new_freq=HUBERT_SAMPLE_RATE,
                lowpass_filter_width=16,
                rolloff=0.94,
                resampling_method="sinc_interp_kaiser",
                dtype=torch.float32,
            ).to(dev)
            audio_16k = resampler_temp(audio_t)
        else:
            audio_16k = audio_t

        n_16k = int(audio_16k.shape[0])
        if n_16k == 0:
            return np.empty(0, dtype=np.float32), out_sr

        if (
            self.convert_buffer is not None
            and getattr(self, "convert_feature_size_16k", 0) > 0
            and self.return_length > 0
        ):
            feats_len = int(self.convert_feature_size_16k)
            skip_head = int(self.skip_head)
            return_length = int(self.return_length)
            silence_front = int(self.silence_front)
        else:
            feats_len = 100
            skip_head = 20
            return_length = 80
            silence_front = 0

        overlap_frames = min(4, max(0, return_length // 4))
        step_frames = max(1, return_length - overlap_frames)
        step_16k = step_frames * WINDOW_SIZE
        left_pad_16k = skip_head * WINDOW_SIZE
        win_16k = feats_len * WINDOW_SIZE

        padded_16k = F.pad(audio_16k, (left_pad_16k, win_16k))
        n_chunks = (n_16k + step_16k - 1) // step_16k

        out_per_frame = out_sr * (WINDOW_SIZE / float(HUBERT_SAMPLE_RATE))
        expected_chunk_len = max(1, int(round(return_length * out_per_frame)))
        step_out = max(1, int(round(step_frames * out_per_frame)))
        overlap_out = max(0, expected_chunk_len - step_out)
        target_out_len = max(1, int(round(n_16k * (out_sr / float(HUBERT_SAMPLE_RATE)))))

        out_buf_len = max(target_out_len, (n_chunks - 1) * step_out + expected_chunk_len)
        out_buf = torch.zeros(out_buf_len, dtype=torch.float32, device=dev)

        if overlap_out > 0:
            lin = torch.linspace(0.0, 1.0, steps=overlap_out, device=dev, dtype=torch.float32)
            fade_in = torch.sin(0.5 * np.pi * lin) ** 2
            fade_out = torch.cos(0.5 * np.pi * lin) ** 2
        else:
            fade_in = None
            fade_out = None

        tmp_pitch = torch.zeros(feats_len + 1, dtype=torch.int64, device=dev)
        tmp_pitchf = torch.zeros(feats_len + 1, dtype=self.dtype, device=dev)
        formant_shift = float(getattr(self.settings, "formantShift", 0.0))

        saved_last_f0 = self.pipeline.last_f0_smoothed
        self.pipeline.last_f0_smoothed = None
        saved_pe_cent = getattr(self.pipeline.pitchExtractor, "_last_voiced_cent", None)
        if hasattr(self.pipeline.pitchExtractor, "reset"):
            self.pipeline.pitchExtractor.reset()
        try:
            for k in range(n_chunks):
                pos_16k = k * step_16k
                chunk_16k = padded_16k[pos_16k : pos_16k + win_16k]
                tmp_pitch.zero_()
                tmp_pitchf.zero_()

                audio_model = self.pipeline.exec(
                    self.settings.dstId,
                    chunk_16k,
                    tmp_pitch,
                    tmp_pitchf,
                    self.settings.tran,
                    self.settings.indexRatio,
                    feats_len,
                    silence_front,
                    self.slotInfo.embOutputLayer,
                    self.slotInfo.useFinalProj,
                    skip_head,
                    return_length,
                    self.settings.protect,
                    formant_shift=formant_shift,
                )

                uses_standalone = getattr(
                    self.pipeline.inferencer, "uses_standalone_vocoder", lambda: False
                )()
                if uses_standalone:
                    try:
                        VocoderManager.get_instance().update_target_sr(out_sr)
                    except Exception:
                        pass
                    chunk_out = audio_model.float()
                elif self.resampler_out is not None:
                    chunk_out = self.resampler_out(audio_model.float())
                else:
                    chunk_out = audio_model.float()

                chunk_out = chunk_out.reshape(-1)
                if chunk_out.shape[0] > expected_chunk_len:
                    chunk_out = chunk_out[:expected_chunk_len]
                elif chunk_out.shape[0] < expected_chunk_len:
                    chunk_out = F.pad(chunk_out, (0, expected_chunk_len - chunk_out.shape[0]))

                out_pos = k * step_out
                has_next = k + 1 < n_chunks

                if k == 0 or overlap_out <= 0:
                    if has_next and overlap_out > 0 and fade_out is not None:
                        out_buf[out_pos : out_pos + step_out] = chunk_out[:step_out]
                        out_buf[out_pos + step_out : out_pos + expected_chunk_len] = (
                            chunk_out[step_out:] * fade_out
                        )
                    else:
                        out_buf[out_pos : out_pos + expected_chunk_len] = chunk_out
                else:
                    if fade_in is not None:
                        out_buf[out_pos : out_pos + overlap_out] += (
                            chunk_out[:overlap_out] * fade_in
                        )
                    if has_next and overlap_out > 0 and fade_out is not None:
                        if step_out > overlap_out:
                            out_buf[out_pos + overlap_out : out_pos + step_out] = chunk_out[
                                overlap_out:step_out
                            ]
                        out_buf[out_pos + step_out : out_pos + expected_chunk_len] = (
                            chunk_out[step_out:] * fade_out
                        )
                    else:
                        out_buf[out_pos + overlap_out : out_pos + expected_chunk_len] = (
                            chunk_out[overlap_out:]
                        )
        finally:
            self.pipeline.last_f0_smoothed = saved_last_f0
            if saved_pe_cent is not None:
                self.pipeline.pitchExtractor._last_voiced_cent = saved_pe_cent

        out_np = out_buf[:target_out_len].detach().cpu().numpy().astype(np.float32, copy=False)
        return out_np, out_sr

    def convert(self, audio_in: AudioInOutFloat, sample_rate: int) -> torch.Tensor:
        out_np, _ = self.convert_offline(audio_in, sample_rate)
        return torch.as_tensor(out_np, dtype=torch.float32, device=self.device_manager.device)

    def _has_sustained_f0(self) -> bool:
        if self.pipeline is None or not getattr(self.pipeline, "use_f0", False):
            return False
        if not hasattr(self, "pitchf_buffer") or self.pitchf_buffer is None or self.pitchf_buffer.numel() == 0:
            return False
        hop = min(max(2, int(self._last_pitch_hop_frames)), int(self.pitchf_buffer.shape[0]))
        tail = self.pitchf_buffer[-hop:]
        voiced_ratio = float((tail > 0.0).float().mean().item())
        return voiced_ratio >= 0.25

    def inference(self, audio_in: AudioInOutFloat, bypass_silence_gate: bool = False):
        if self.pipeline is None:
            raise PipelineNotInitializedException()

        if bypass_silence_gate:
            audio_f32 = np.asarray(audio_in, dtype=np.float32).reshape(-1)
            vol = float(np.sqrt(np.mean(np.square(audio_f32, dtype=np.float32)))) if audio_f32.size > 0 else 0.0
        else:
            gate_active, vol = self._noise_gate.evaluate(
                audio_in,
                sample_rate=self.input_sample_rate,
                silent_threshold_db=self.settings.silentThreshold,
                vad_enabled=bool(getattr(self.settings, "vadEnabled", False)),
                vad_threshold=float(getattr(self.settings, "vadThreshold", 0.45)),
                f0_voiced_fn=self._has_sustained_f0,
            )
            if not gate_active:
                # When a neural/DSP denoiser (DFN3 / NS) is active, stationary mic noise/hum is
                # already suppressed upstream, so feed raw audio into context buffers to preserve
                # soft speech onsets. Without a denoiser, attenuate by -12 dB to avoid noise pollution.
                denoiser_active = bool(
                    getattr(self.settings, "noiseSuppression", False)
                    or getattr(self.settings, "noiseSuppression2", False)
                )
                context_in = (
                    np.asarray(audio_in, dtype=np.float32)
                    if denoiser_active
                    else np.asarray(audio_in, dtype=np.float32) * np.float32(0.25)
                )
                self.feed_context(context_in)
                if hasattr(self.pipeline.pitchExtractor, "reset"):
                    self.pipeline.pitchExtractor.reset()
                if self._rms_gain_ema_t is not None:
                    self._rms_gain_ema_t = 0.9 * self._rms_gain_ema_t + 0.1
                return None, vol

        audio_in_16k, pitch_hop_frames = self._prepare_16k_input(audio_in)
        self._last_pitch_hop_frames = max(1, int(pitch_hop_frames))
        circular_write(audio_in_16k, self.audio_buffer, scratch=self._audio_scratch)
        circular_write(audio_in_16k, self.convert_buffer, scratch=self._convert_scratch)

        buf_rms = torch.sqrt(torch.mean(self.audio_buffer.float() ** 2) + 1e-8)

        audio_model = self.pipeline.exec(
            self.settings.dstId,
            self.convert_buffer,
            self.pitch_buffer,
            self.pitchf_buffer,
            self.settings.tran,
            self.settings.indexRatio,
            self.convert_feature_size_16k,
            self.silence_front,
            self.slotInfo.embOutputLayer,
            self.slotInfo.useFinalProj,
            self.skip_head,
            self.return_length,
            self.settings.protect,
            pitch_hop_frames=pitch_hop_frames,
            pitch_scratch=self._pitch_scratch,
            pitchf_scratch=self._pitchf_scratch,
            formant_shift=float(getattr(self.settings, "formantShift", 0.0)),
        )

        # Smoothed adaptive RMS normalization (zero GPU->CPU sync, prevents pumping and noise boost)
        if vol > 1e-3:
            model_rms = torch.sqrt(torch.mean(audio_model ** 2) + 1e-8)
            target_scale = torch.where(
                model_rms > 1e-4,
                (buf_rms / model_rms).clamp(0.25, 2.5),
                torch.ones_like(model_rms),
            )
            if self._rms_gain_ema_t is None or self._rms_gain_ema_t.device != audio_model.device:
                self._rms_gain_ema_t = target_scale.detach()
            else:
                self._rms_gain_ema_t = 0.7 * self._rms_gain_ema_t + 0.3 * target_scale.detach()
            audio_model = audio_model * self._rms_gain_ema_t

        uses_standalone = getattr(self.pipeline.inferencer, "uses_standalone_vocoder", lambda: False)()
        if uses_standalone:
            try:
                VocoderManager.get_instance().update_target_sr(self.output_sample_rate)
            except Exception:
                pass
            audio_out = audio_model
        else:
            audio_out = self.resampler_out(audio_model)

        # Exact length alignment via trim/pad (avoids linear-interpolation aliasing)
        expected_len = int(round(self.return_length * (WINDOW_SIZE / HUBERT_SAMPLE_RATE) * self.output_sample_rate))
        if audio_out.dim() == 1 and audio_out.shape[-1] != expected_len:
            if audio_out.shape[-1] > expected_len:
                audio_out = audio_out[:expected_len]
            else:
                audio_out = F.pad(audio_out, (0, expected_len - audio_out.shape[-1]))

        return audio_out, vol

    def warmup(
        self,
        block_frame: Optional[int] = None,
        on_audio_out: Optional[Callable[[torch.Tensor], None]] = None,
        step: int = 5,
        total: int = 5,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> None:
        """
        Pre-compile and warm up the full inference pipeline (TensorRT engines, TorchScript
        JIT kernel fusion, cuDNN autotuning, and CUDA lazy modules) using deterministic
        synthetic voiced audio, then reset all streaming state.
        """
        if self.pipeline is None or self.convert_buffer is None or self.resampler_in is None:
            return

        self._emit_progress(
            step, total, "Compiling & warming up...", progress_callback=progress_callback
        )

        if block_frame is not None and block_frame > 0:
            n_samples = int(block_frame)
        elif self._last_block_frame > 0:
            n_samples = self._last_block_frame
        else:
            n_samples = max(
                128,
                int(round((self.settings.serverReadChunkSize * 128 / 48000.0) * self.input_sample_rate)),
            )

        t = np.arange(n_samples, dtype=np.float32) / float(max(1, self.input_sample_rate))
        dummy_audio = (0.15 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32, copy=False)

        dev = self.device_manager.device
        warmup_runs = 2 if dev.type == "cuda" else 1

        logger.info(
            f"Warming up pipeline ({warmup_runs} pass(es), chunk={n_samples} samples @ {self.input_sample_rate}Hz, device={dev})..."
        )
        t0 = perf_counter()
        try:
            with torch.no_grad():
                for _ in range(warmup_runs):
                    out_audio, _ = self.inference(dummy_audio, bypass_silence_gate=True)
                    if on_audio_out is not None and out_audio is not None:
                        on_audio_out(out_audio)
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
            elapsed_ms = (perf_counter() - t0) * 1000.0
            logger.info(f"Pipeline warmup completed in {elapsed_ms:.1f} ms.")
        except Exception as e:
            logger.warning(f"Pipeline warmup failed (will warm up on first live chunk): {e}")
        finally:
            self.reset_state()

    def __del__(self):
        del self.pipeline

    def export2onnx(self):
        modelSlot = self.slotInfo

        if modelSlot.isONNX:
            logger.error(f"{modelSlot.modelFile} is already in ONNX format.")
            return

        output_path = export2onnx(modelSlot)

        self.slotInfo.modelFileOnnx = os.path.basename(output_path)
        self.slotInfo.modelTypeOnnx = EnumInferenceTypes.onnxRVC.value if self.slotInfo.f0 else EnumInferenceTypes.onnxRVCNono.value
        saveSlotInfo(self.params.model_dir, self.slotInfo.slotIndex, self.slotInfo)

    def get_model_current(self) -> dict:
        return [
            {
                "key": "defaultTune",
                "val": self.settings.tran,
            },
            {
                "key": "defaultIndexRatio",
                "val": self.settings.indexRatio,
            },
            {
                "key": "defaultFormantShift",
                "val": self.settings.formantShift,
            },
        ]
