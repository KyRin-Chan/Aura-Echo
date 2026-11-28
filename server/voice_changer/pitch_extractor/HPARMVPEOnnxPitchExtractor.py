import numpy as np
import onnxruntime
import torch
import torch.nn.functional as F

from const import PitchExtractorType
from voice_changer.common.MelExtractor import MelSpectrogram
from voice_changer.common.OnnxIOHelper import (
    OnnxCudaIOBufferCache,
    create_ort_session_with_stream,
    ort_cuda_context,
)
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
from voice_changer.pitch_extractor.PitchExtractor import (
    PitchExtractor,
    calibrate_voicing_confidence,
    decode_salience_to_f0,
    load_onnx_model_safe,
)


class HPARMVPEOnnxPitchExtractor(PitchExtractor):
    """ONNX Runtime pitch extractor for HPA-RMVPE."""

    def __init__(self, file: str):
        super().__init__()
        self.file = file
        self.type: PitchExtractorType = "hpa_rmvpe_onnx"

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
        self.input_names = [inp.name for inp in inputs]
        self.mel_input_name = "mel" if "mel" in self.input_names else self.input_names[0]
        self.output_name = outputs[0].name if outputs else "f0"

        mel_inp = next((inp for inp in inputs if inp.name == self.mel_input_name), inputs[0])
        self.input_is_fp16 = mel_inp.type == "tensor(float16)"
        self.in_dtype_t = torch.float16 if self.input_is_fp16 else torch.float32
        self.in_dtype_np = np.float16 if self.input_is_fp16 else np.float32
        self.threshold_val = 0.032
        self.threshold_np = np.array(self.threshold_val, dtype=self.in_dtype_np)

        self.mel_extractor = MelSpectrogram(
            self.input_is_fp16, 128, 16000, 1024, 160, mel_fmin=30, mel_fmax=8000
        ).to(device_manager.device)
        self.idx = torch.arange(360, device=device_manager.device)[None, None, :]
        self.idx_cents = self.idx * 20 + 1997.3794084376191

    def _run_session(self, mel: torch.Tensor) -> torch.Tensor:
        if mel.device.type == "cuda":
            mel = mel.to(dtype=self.in_dtype_t).contiguous()
            with ort_cuda_context(mel.device, self._bound_to_stream, self._cuda_stream):
                binding = self.onnx_session.io_binding()
                binding.bind_input(
                    self.mel_input_name,
                    device_type="cuda",
                    device_id=mel.device.index,
                    element_type=self.in_dtype_np,
                    shape=tuple(mel.shape),
                    buffer_ptr=mel.data_ptr(),
                )
                if "threshold" in self.input_names:
                    binding.bind_cpu_input("threshold", self.threshold_np)
                out = self._io_cache.run(
                    self.onnx_session,
                    binding,
                    [self.output_name],
                    (tuple(mel.shape), self.output_name),
                    mel.device,
                )[0]
            return out

        inputs_dict = {self.mel_input_name: mel.detach().cpu().numpy().astype(self.in_dtype_np)}
        if "threshold" in self.input_names:
            inputs_dict["threshold"] = self.threshold_np
        out = self.onnx_session.run([self.output_name], inputs_dict)[0]
        return torch.as_tensor(out, dtype=torch.float32, device=mel.device)

    def extract_with_confidence(
        self,
        audio: torch.Tensor,
        sr: int,
        window: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mel = self.mel_extractor(audio.unsqueeze(0).float())
        n_frames = mel.shape[-1]

        pad_len = 32 * ((n_frames - 1) // 32 + 1) - n_frames
        if pad_len > 0:
            mel = F.pad(mel, (0, pad_len), mode="reflect")

        raw_out = self._run_session(mel)
        if raw_out.ndim >= 2 and raw_out.shape[-1] == 360:
            salience = raw_out.unsqueeze(0) if raw_out.ndim == 2 else raw_out
            salience = salience[:, :n_frames, :].float()
            f0, raw_conf = decode_salience_to_f0(
                salience, self.threshold_val, self.idx, self.idx_cents
            )
            f0 = f0.squeeze()
            raw_conf = raw_conf.squeeze()
            return f0, calibrate_voicing_confidence(f0, raw_conf)

        f0 = raw_out.reshape(-1)[:n_frames].float()
        return f0, (f0 > 0).float()

    def extract(
        self,
        audio: torch.Tensor,
        sr: int,
        window: int,
    ) -> torch.Tensor:
        f0, _ = self.extract_with_confidence(audio, sr, window)
        return f0
