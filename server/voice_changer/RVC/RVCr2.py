"""
VoiceChangerV2向け
"""
import logging
import os
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torchaudio import transforms as tat

from const import EnumInferenceTypes, HUBERT_SAMPLE_RATE, WINDOW_SIZE
from data.ModelSlot import RVCModelSlot, saveSlotInfo
from Exceptions import PipelineNotInitializedException
from settings import get_settings
from voice_changer.common.AudioDSP import CausalPolyphaseResampler, MinimumPhaseFormantFilter
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
        self._keepalive_a: torch.Tensor | None = None
        self._keepalive_b: torch.Tensor | None = None

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
        self._gate_hangover_chunks: int = 0

    def set_audio_effects_manager(self, manager) -> None:
        self._audio_effects_manager = manager
        if self.pipeline is not None:
            self.pipeline.audio_effects_manager = manager

    def set_progress_callback(self, callback: Optional[Callable[[int, int, str], None]]):
        """Register a callback to receive model loading progress updates."""
        self._progress_callback = callback

    def initialize(self, force_reload: bool = False, progress_callback: Optional[Callable[[int, int, str], None]] = None):
        logger.info("Initializing...")

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
        block_frame_16k = int(block_frame / self.input_sample_rate * HUBERT_SAMPLE_RATE)
        crossfade_frame_16k = int(crossfade_frame / self.input_sample_rate * HUBERT_SAMPLE_RATE)
        sola_search_frame_16k = int(sola_search_frame / self.input_sample_rate * HUBERT_SAMPLE_RATE)
        extra_frame_16k = int(extra_frame / self.input_sample_rate * HUBERT_SAMPLE_RATE)

        convert_size_16k = block_frame_16k + sola_search_frame_16k + extra_frame_16k + crossfade_frame_16k
        if (modulo := convert_size_16k % WINDOW_SIZE) != 0:
            convert_size_16k = convert_size_16k + (WINDOW_SIZE - modulo)
        self.convert_feature_size_16k = convert_size_16k // WINDOW_SIZE

        self.skip_head = extra_frame_16k // WINDOW_SIZE
        self.return_length = self.convert_feature_size_16k - self.skip_head
        self.silence_front = max(0, extra_frame_16k - (WINDOW_SIZE * 12)) if self.settings.silenceFront else 0

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

        if dev.type == 'cuda':
            self._keepalive_a = torch.ones((64, 64), dtype=self.dtype, device=dev)
            self._keepalive_b = torch.ones((64, 64), dtype=self.dtype, device=dev)
        else:
            self._keepalive_a = None
            self._keepalive_b = None

        self._pitch_sample_acc = 0
        self._rms_gain_ema_t = None
        self._gate_hangover_chunks = 0
        if self.resampler_in is not None:
            self.resampler_in.reset()
        self.formant_filter.reset()

        logger.info(f'Allocated audio buffer size: {audio_buffer_size}')
        logger.info(f'Allocated convert buffer size: {convert_size_16k}')
        logger.info(f'Allocated pitchf buffer size: {self.convert_feature_size_16k + 1}')

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

    def _prepare_16k_input(self, audio_in: AudioInOutFloat) -> tuple[torch.Tensor, int]:
        audio_in_t = torch.as_tensor(audio_in, dtype=torch.float32, device=self.device_manager.device)
        audio_in_16k = self.resampler_in(audio_in_t)
        self._sync_formant_filter()
        audio_in_16k = self.formant_filter.apply_pre(audio_in_16k)

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

    def convert(self, audio_in: AudioInOutFloat, sample_rate: int) -> torch.Tensor:
        if self.pipeline is None:
            raise PipelineNotInitializedException()

        audio_in_t = torch.as_tensor(audio_in, dtype=torch.float32, device=self.device_manager.device)
        if self.is_half:
            audio_in_t = audio_in_t.half()

        convert_feature_size_16k = audio_in_t.shape[0] // WINDOW_SIZE

        resampler_temp = tat.Resample(
            orig_freq=sample_rate,
            new_freq=HUBERT_SAMPLE_RATE,
            lowpass_filter_width=16,
            rolloff=0.94,
            resampling_method="kaiser_window",
            dtype=self.dtype
        ).to(self.device_manager.device)
        audio_in_16k = resampler_temp(audio_in_t)

        vol_t = torch.sqrt(
            torch.square(audio_in_16k).mean()
        )

        audio_model = self.pipeline.exec(
            self.settings.dstId,
            audio_in_16k,
            None,
            None,
            self.settings.tran,
            self.settings.indexRatio,
            convert_feature_size_16k,
            0,
            self.slotInfo.embOutputLayer,
            self.slotInfo.useFinalProj,
            0,
            convert_feature_size_16k,
            self.settings.protect,
            formant_shift=float(getattr(self.settings, "formantShift", 0.0)),
        )

        audio_out: torch.Tensor = self.resampler_out(audio_model * torch.sqrt(vol_t))
        return audio_out

    def inference(self, audio_in: AudioInOutFloat):
        if self.pipeline is None:
            raise PipelineNotInitializedException()

        # Precompute RMS volume on CPU before GPU upload to avoid mid-pipeline GPU sync
        vol = float(np.sqrt(np.mean(np.square(audio_in, dtype=np.float32)))) if audio_in.size > 0 else 0.0

        if vol >= self.inputSensitivity:
            self._gate_hangover_chunks = 2
        elif self._gate_hangover_chunks > 0:
            self._gate_hangover_chunks -= 1
        else:
            self.feed_context(audio_in)
            if self._rms_gain_ema_t is not None:
                self._rms_gain_ema_t = 0.9 * self._rms_gain_ema_t + 0.1

            # Lightweight CUDA clock keepalive instead of running full RVC pipeline on silence
            dev = self.device_manager.device
            if getattr(self.settings, 'powerSavingMode', 0) == 0 and dev.type == 'cuda':
                if self._keepalive_a is not None and self._keepalive_b is not None:
                    _ = torch.mm(self._keepalive_a, self._keepalive_b)
            return None, vol

        audio_in_16k, pitch_hop_frames = self._prepare_16k_input(audio_in)
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
                "key": "defaultProtect",
                "val": self.settings.protect,
            },
            {
                "key": "defaultFormantShift",
                "val": self.settings.formantShift,
            },
        ]
