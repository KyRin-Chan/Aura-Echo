import numpy as np
import onnxruntime
import torch
import torch.nn.functional as F

from const import F0_MIN, TRACKER_F0_MAX, PitchExtractorType
from voice_changer.common.OnnxIOHelper import (
    OnnxCudaIOBufferCache,
    create_ort_session_with_stream,
    ort_cuda_context,
)
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
from voice_changer.pitch_extractor.PitchExtractor import (
    PitchExtractor,
    load_onnx_model_safe,
)

SWIFT_HOP = 256
SWIFT_CENTER_OFFSET = 128.0
SWIFT_MIN_SAMPLES = 256


class SwiftF0OnnxPitchExtractor(PitchExtractor):
    """ONNX Runtime pitch extractor for SwiftF0 (swift3.onnx)."""

    def __init__(self, file: str, confidence_threshold: float = 0.9):
        super().__init__()
        self.file = file
        self.type: PitchExtractorType = "swiftf0_onnx"

        device_manager = DeviceManager.get_instance()
        self._cuda_stream = device_manager.pitch_stream
        # Keep SwiftF0 in FP32 so embedded STFT ops preserve numerical accuracy.
        model, self.is_half = load_onnx_model_safe(file, False)

        so = onnxruntime.SessionOptions()
        so.log_severity_level = 3
        self.onnx_session, self._bound_to_stream = create_ort_session_with_stream(
            model, so, device_manager, cuda_stream=self._cuda_stream
        )
        self._io_cache = OnnxCudaIOBufferCache()

        inputs = self.onnx_session.get_inputs()
        outputs = self.onnx_session.get_outputs()
        self.input_name = inputs[0].name if inputs else "input_audio"
        self.output_names = (
            [out.name for out in outputs] if outputs else ["pitch_hz", "confidence"]
        )
        self.input_is_fp16 = bool(inputs) and inputs[0].type == "tensor(float16)"
        self.in_dtype_t = torch.float16 if self.input_is_fp16 else torch.float32
        self.in_dtype_np = np.float16 if self.input_is_fp16 else np.float32

        self.confidence_threshold = float(confidence_threshold)
        self.fmin = float(F0_MIN)
        self.fmax = float(TRACKER_F0_MAX)

    def _run_session(self, audio_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if audio_in.device.type == "cuda":
            audio_in = audio_in.to(dtype=self.in_dtype_t).contiguous()
            with ort_cuda_context(audio_in.device, self._bound_to_stream, self._cuda_stream):
                binding = self.onnx_session.io_binding()
                binding.bind_input(
                    self.input_name,
                    device_type="cuda",
                    device_id=audio_in.device.index,
                    element_type=self.in_dtype_np,
                    shape=tuple(audio_in.shape),
                    buffer_ptr=audio_in.data_ptr(),
                )
                outs = self._io_cache.run(
                    self.onnx_session,
                    binding,
                    self.output_names,
                    (tuple(audio_in.shape), tuple(self.output_names)),
                    audio_in.device,
                )
            pitch_hz = outs[0].reshape(-1).float()
            conf = (
                outs[1].reshape(-1).float()
                if len(outs) > 1
                else (pitch_hz > 0).float()
            )
            return pitch_hz, conf

        outs = self.onnx_session.run(
            self.output_names,
            {self.input_name: audio_in.detach().cpu().numpy().astype(self.in_dtype_np)},
        )
        pitch_hz = torch.as_tensor(
            outs[0], dtype=torch.float32, device=audio_in.device
        ).reshape(-1)
        conf = (
            torch.as_tensor(
                outs[1], dtype=torch.float32, device=audio_in.device
            ).reshape(-1)
            if len(outs) > 1
            else (pitch_hz > 0).float()
        )
        return pitch_hz, conf

    @staticmethod
    def _resample_to_rvc_grid(
        pitch_hz: torch.Tensor,
        conf: torch.Tensor,
        target_frames: int,
        window: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resample SwiftF0 frames (hop=256, center=128) onto the RVC frame grid."""
        if target_frames <= 0 or pitch_hz.numel() == 0:
            zeros = torch.zeros(
                max(0, target_frames), dtype=torch.float32, device=pitch_hz.device
            )
            return zeros, zeros.clone()

        t_swift = pitch_hz.shape[0]
        pos = (
            torch.arange(target_frames, device=pitch_hz.device, dtype=torch.float32)
            * float(window)
            - SWIFT_CENTER_OFFSET
        ) / float(SWIFT_HOP)
        pos = pos.clamp(0.0, float(t_swift - 1))

        idx0 = pos.floor().long()
        idx1 = (idx0 + 1).clamp(max=t_swift - 1)
        w = pos - idx0.float()

        conf_interp = conf[idx0] * (1.0 - w) + conf[idx1] * w

        p0 = pitch_hz[idx0]
        p1 = pitch_hz[idx1]
        both_pos = (p0 > 0.0) & (p1 > 0.0)
        log_interp = torch.exp(
            torch.log(p0.clamp(min=1.0)) * (1.0 - w)
            + torch.log(p1.clamp(min=1.0)) * w
        )
        nearest = torch.where(w < 0.5, p0, p1)
        f0_interp = torch.where(both_pos, log_interp, nearest)
        return f0_interp, conf_interp

    def extract_with_confidence(
        self,
        audio: torch.Tensor,
        sr: int,
        window: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if audio.ndim > 1:
            audio = audio.reshape(-1)
        if audio.numel() == 0:
            empty = torch.empty(0, dtype=torch.float32, device=audio.device)
            return empty, empty.clone()

        n_samples = audio.shape[-1]
        target_frames = n_samples // window + 1

        audio_in = audio.float().unsqueeze(0)
        if n_samples < SWIFT_MIN_SAMPLES:
            audio_in = F.pad(audio_in, (0, SWIFT_MIN_SAMPLES - n_samples))

        pitch_hz, conf = self._run_session(audio_in)
        pitch_hz = torch.nan_to_num(pitch_hz, nan=0.0, posinf=0.0, neginf=0.0)
        conf = torch.nan_to_num(conf, nan=0.0, posinf=0.0, neginf=0.0)
        valid_raw = (
            (conf >= self.confidence_threshold)
            & (pitch_hz >= self.fmin)
            & (pitch_hz <= self.fmax)
        )
        pitch_hz = torch.where(valid_raw, pitch_hz, torch.zeros_like(pitch_hz))

        f0_interp, conf_interp = self._resample_to_rvc_grid(
            pitch_hz, conf, target_frames, window
        )

        voiced = (
            (conf_interp >= self.confidence_threshold)
            & (f0_interp >= self.fmin)
            & (f0_interp <= self.fmax)
        )
        f0 = torch.where(voiced, f0_interp, torch.zeros_like(f0_interp))
        voicing_conf = torch.where(
            voiced, conf_interp.clamp(0.0, 1.0), torch.zeros_like(conf_interp)
        )
        return f0, voicing_conf

    def extract(
        self,
        audio: torch.Tensor,
        sr: int,
        window: int,
    ) -> torch.Tensor:
        f0, _ = self.extract_with_confidence(audio, sr, window)
        return f0
