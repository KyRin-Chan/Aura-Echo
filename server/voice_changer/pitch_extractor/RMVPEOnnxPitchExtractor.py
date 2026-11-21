import numpy as np
import torch
import torch.nn.functional as F
import onnxruntime
from const import PitchExtractorType
from voice_changer.common.OnnxLoader import load_onnx_model
from voice_changer.common.OnnxIOHelper import (
    OnnxCudaIOBufferCache,
    create_ort_session_with_stream,
    ort_cuda_context,
)
from voice_changer.pitch_extractor.PitchExtractor import PitchExtractor
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
from voice_changer.common.MelExtractor import MelSpectrogram

class RMVPEOnnxPitchExtractor(PitchExtractor):

    def __init__(self, file: str):
        super().__init__()
        self.file = file
        self.type: PitchExtractorType = "rmvpe_onnx"

        device_manager = DeviceManager.get_instance()
        self.is_half = device_manager.use_fp16()
        self._cuda_stream = device_manager.pitch_stream

        model = load_onnx_model(file, self.is_half)

        self.fp_dtype_t = torch.float16 if self.is_half else torch.float32
        self.fp_dtype_np = np.float16 if self.is_half else np.float32

        self.threshold = np.array(0.045, dtype=self.fp_dtype_np)

        so = onnxruntime.SessionOptions()
        so.log_severity_level = 3
        self.mel_extractor = MelSpectrogram(
            self.is_half, 128, 16000, 1024, 160, mel_fmin=30, mel_fmax=8000
        ).to(device_manager.device)
        self.onnx_session, self._bound_to_stream = create_ort_session_with_stream(
            model, so, device_manager, cuda_stream=self._cuda_stream
        )
        self._io_cache = OnnxCudaIOBufferCache()

    def extract(
        self,
        audio: torch.Tensor,
        sr: int,
        window: int,
    ) -> torch.Tensor:
        mel = self.mel_extractor(audio.unsqueeze(0).float())
        n_frames = mel.shape[-1]

        # UNet 5-level downsampling requires T dimension to be a multiple of 32 (2^5).
        # Reflect padding prevents boundary artifacts, duplicate overlaps, and frame dimension mismatches.
        pad_len = 32 * ((n_frames - 1) // 32 + 1) - n_frames
        if pad_len > 0:
            mel = F.pad(mel, (0, pad_len), mode='reflect')

        if audio.device.type == 'cuda':
            mel = mel.to(dtype=self.fp_dtype_t).contiguous()
            with ort_cuda_context(audio.device, self._bound_to_stream, self._cuda_stream):
                binding = self.onnx_session.io_binding()
                binding.bind_input(
                    'mel',
                    device_type='cuda',
                    device_id=audio.device.index,
                    element_type=self.fp_dtype_np,
                    shape=tuple(mel.shape),
                    buffer_ptr=mel.data_ptr(),
                )
                binding.bind_cpu_input('threshold', self.threshold)
                output_tensor = self._io_cache.run(
                    self.onnx_session,
                    binding,
                    ['pitchf'],
                    (tuple(mel.shape),),
                    audio.device,
                )[0]

            output_tensor = output_tensor.to(dtype=self.fp_dtype_t).squeeze()
            return output_tensor[..., :n_frames]
        else:
            output: list[np.ndarray] = self.onnx_session.run(
                ["pitchf"],
                {
                    "mel": mel.detach().cpu().numpy(),
                    "threshold": self.threshold,
                },
            )
            # self.onnx_session.end_profiling()

            output_tensor = torch.as_tensor(output[0], dtype=self.fp_dtype_t, device=audio.device).squeeze()
            return output_tensor[..., :n_frames]