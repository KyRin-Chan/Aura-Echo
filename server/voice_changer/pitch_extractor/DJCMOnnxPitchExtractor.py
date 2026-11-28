import numpy as np
import onnxruntime
import torch
import torch.nn.functional as F

from const import PitchExtractorType
from voice_changer.common.OnnxIOHelper import (
    OnnxCudaIOBufferCache,
    create_ort_session_with_stream,
    ort_cuda_context,
)
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
from voice_changer.common.djcm.djcm import (
    N_CLASS,
    SAMPLE_RATE,
    Spectrogram,
    pad_audio_segments,
)
from voice_changer.pitch_extractor.PitchExtractor import (
    PitchExtractor,
    calibrate_voicing_confidence,
    decode_salience_to_f0,
    load_onnx_model_safe,
)


class DJCMOnnxPitchExtractor(PitchExtractor):
    """ONNX Runtime pitch extractor for DJCM."""

    def __init__(self, file: str, segment_len: float = 5.12):
        super().__init__()
        self.file = file
        self.type: PitchExtractorType = "djcm_onnx"

        device_manager = DeviceManager.get_instance()
        self._cuda_stream = device_manager.pitch_stream
        model, self.is_half = load_onnx_model_safe(file, device_manager.use_fp16())

        so = onnxruntime.SessionOptions()
        so.log_severity_level = 3
        self.onnx_session, self._bound_to_stream = create_ort_session_with_stream(
            model, so, device_manager, cuda_stream=self._cuda_stream
        )
        self._io_cache = OnnxCudaIOBufferCache()

        inputs = self.onnx_session.get_inputs()
        outputs = self.onnx_session.get_outputs()
        self.input_name = inputs[0].name if inputs else "spec"
        self.output_name = outputs[0].name if outputs else "f0"

        spec_inp = inputs[0]
        self.input_is_fp16 = spec_inp.type == "tensor(float16)"
        self.in_dtype_t = torch.float16 if self.input_is_fp16 else torch.float32
        self.in_dtype_np = np.float16 if self.input_is_fp16 else np.float32

        inp_shape = getattr(spec_inp, "shape", [])
        self.fixed_t = (
            inp_shape[2]
            if len(inp_shape) >= 4 and isinstance(inp_shape[2], int)
            else None
        )
        self.fixed_batch = (
            inp_shape[0]
            if len(inp_shape) >= 4 and isinstance(inp_shape[0], int)
            else None
        )

        self.seg_len = int(segment_len * SAMPLE_RATE)
        self.seg_frames = self.seg_len // (SAMPLE_RATE // 100)
        self.threshold_val = 0.03

        self.spec_extractor = Spectrogram(hop_length=160, win_length=1024).to(
            device_manager.device
        )
        self.idx = torch.arange(N_CLASS, device=device_manager.device)[None, None, :]
        self.idx_cents = self.idx * 20 + 1997.3794084376191

    def _run_session(self, spec: torch.Tensor) -> torch.Tensor:
        if self.fixed_t is not None and spec.shape[2] != self.fixed_t:
            if spec.shape[2] > self.fixed_t:
                spec = spec[:, :, : self.fixed_t, :]
            else:
                spec = F.pad(spec, (0, 0, 0, self.fixed_t - spec.shape[2]))

        if spec.device.type == "cuda":
            spec = spec.to(dtype=self.in_dtype_t).contiguous()
            with ort_cuda_context(spec.device, self._bound_to_stream, self._cuda_stream):
                binding = self.onnx_session.io_binding()
                binding.bind_input(
                    self.input_name,
                    device_type="cuda",
                    device_id=spec.device.index,
                    element_type=self.in_dtype_np,
                    shape=tuple(spec.shape),
                    buffer_ptr=spec.data_ptr(),
                )
                out = self._io_cache.run(
                    self.onnx_session,
                    binding,
                    [self.output_name],
                    (tuple(spec.shape), self.output_name),
                    spec.device,
                )[0]
            return out

        out = self.onnx_session.run(
            [self.output_name],
            {self.input_name: spec.detach().cpu().numpy().astype(self.in_dtype_np)},
        )[0]
        return torch.as_tensor(out, dtype=torch.float32, device=spec.device)

    def extract_with_confidence(
        self,
        audio: torch.Tensor,
        sr: int,
        window: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if audio.ndim > 1:
            audio = audio.reshape(-1)
        n_frames = audio.shape[-1] // (SAMPLE_RATE // 100) + 1

        if self.fixed_t is None and audio.shape[-1] <= self.seg_len:
            spec = self.spec_extractor(audio.unsqueeze(0).unsqueeze(0).float())
            raw_out = self._run_session(spec)
            salience = raw_out.unsqueeze(0) if raw_out.ndim == 2 else raw_out
            salience = salience[:, :n_frames, :].float()
        else:
            segments = pad_audio_segments(audio.float(), self.seg_len)
            batch_sz = 1 if self.fixed_batch == 1 else len(segments)
            seg_outs = []
            for i in range(0, len(segments), batch_sz):
                spec = self.spec_extractor(segments[i : i + batch_sz])
                out = self._run_session(spec)
                if out.ndim == 2:
                    out = out.unsqueeze(0)
                seg_outs.append(out)
            all_segs = torch.cat(seg_outs, dim=0)
            start = self.seg_frames // 4
            end = int(self.seg_frames * 0.75)
            stitched = torch.cat([seg[start:end] for seg in all_segs], dim=0)[:n_frames]
            salience = stitched.unsqueeze(0).float()

        f0, raw_conf = decode_salience_to_f0(
            salience, self.threshold_val, self.idx, self.idx_cents
        )
        f0 = f0.squeeze()
        raw_conf = raw_conf.squeeze()
        return f0, calibrate_voicing_confidence(f0, raw_conf)

    def extract(
        self,
        audio: torch.Tensor,
        sr: int,
        window: int,
    ) -> torch.Tensor:
        f0, _ = self.extract_with_confidence(audio, sr, window)
        return f0
