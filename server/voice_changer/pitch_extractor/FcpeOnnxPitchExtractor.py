import numpy as np
import torch
import onnxruntime
from const import PitchExtractorType
from voice_changer.pitch_extractor.PitchExtractor import PitchExtractor
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
from voice_changer.common.OnnxLoader import load_onnx_model
from voice_changer.common.OnnxIOHelper import (
    OnnxCudaIOBufferCache,
    create_ort_session_with_stream,
    ort_cuda_context,
)
from voice_changer.common.MelExtractorFcpe import Wav2MelModule

class FcpeOnnxPitchExtractor(PitchExtractor):

    def __init__(self, file: str):
        super().__init__()
        self.file = file
        self.type: PitchExtractorType = "fcpe_onnx"

        device_manager = DeviceManager.get_instance()
        # NOTE: FCPE doesn't seem to behave correctly in FP16 mode.
        self.is_half = False
        self._cuda_stream = device_manager.pitch_stream

        model = load_onnx_model(file, self.is_half)

        self.fp_dtype_t = torch.float16 if self.is_half else torch.float32
        self.fp_dtype_np = np.float16 if self.is_half else np.float32

        self.threshold = np.array([0.006], dtype=self.fp_dtype_np)

        so = onnxruntime.SessionOptions()
        so.log_severity_level = 3
        self.mel_extractor = Wav2MelModule(
            sr=16000,
            n_mels=128,
            n_fft=1024,
            win_size=1024,
            hop_length=160,
            fmin=0,
            fmax=8000,
            clip_val=1e-05,
            is_half=self.is_half
        ).to(device_manager.device)
        self.onnx_session, self._bound_to_stream = create_ort_session_with_stream(
            model, so, device_manager, cuda_stream=self._cuda_stream
        )
        self._io_cache = OnnxCudaIOBufferCache()

        # Inspect actual input and output node names from the ONNX session
        self.input_names = {inp.name: inp for inp in self.onnx_session.get_inputs()}
        self.output_names = [out.name for out in self.onnx_session.get_outputs()]

    def extract(
        self,
        audio: torch.Tensor,
        sr: int,
        window: int,
    ) -> torch.Tensor:
        mel = self.mel_extractor(audio.unsqueeze(0).float())
        mel = torch.nan_to_num(mel, nan=0.0, posinf=0.0, neginf=-11.51)

        # Check if ONNX model expects [B, 128, T] instead of [B, T, 128]
        if len(self.input_names) > 0:
            first_input_key = 'mel' if 'mel' in self.input_names else list(self.input_names.keys())[0]
            inp_shape = getattr(self.input_names[first_input_key], 'shape', [])
            if len(inp_shape) == 3 and inp_shape[1] == 128 and mel.shape[1] != 128 and mel.shape[2] == 128:
                mel = mel.transpose(1, 2)

        mel = mel.to(dtype=self.fp_dtype_t).contiguous()
        n_samples = np.array([mel.shape[1] if mel.shape[1] != 128 else mel.shape[2]], dtype=np.int64)
        output_node = self.output_names[0] if len(self.output_names) > 0 else "pitchf"

        if audio.device.type == 'cuda':
            audio_t = audio.to(dtype=self.fp_dtype_t).contiguous()
            with ort_cuda_context(audio.device, self._bound_to_stream, self._cuda_stream):
                binding = self.onnx_session.io_binding()

                # Dynamically bind inputs matching actual ONNX node definitions
                if 'mel' in self.input_names:
                    binding.bind_input('mel', device_type='cuda', device_id=audio.device.index, element_type=self.fp_dtype_np, shape=tuple(mel.shape), buffer_ptr=mel.data_ptr())
                    sig_shape = tuple(mel.shape)
                elif 'audio' in self.input_names:
                    binding.bind_input('audio', device_type='cuda', device_id=audio.device.index, element_type=self.fp_dtype_np, shape=tuple(audio_t.shape), buffer_ptr=audio_t.data_ptr())
                    sig_shape = tuple(audio_t.shape)
                elif len(self.input_names) > 0:
                    first_input = list(self.input_names.keys())[0]
                    binding.bind_input(first_input, device_type='cuda', device_id=audio.device.index, element_type=self.fp_dtype_np, shape=tuple(mel.shape), buffer_ptr=mel.data_ptr())
                    sig_shape = tuple(mel.shape)
                else:
                    sig_shape = tuple(mel.shape)

                if 'n_samples' in self.input_names:
                    binding.bind_cpu_input('n_samples', n_samples)

                if 'threshold' in self.input_names:
                    binding.bind_cpu_input('threshold', self.threshold)

                output_tensor = self._io_cache.run(
                    self.onnx_session,
                    binding,
                    [output_node],
                    (sig_shape, output_node),
                    audio.device,
                )[0]

            output_tensor = output_tensor.to(dtype=self.fp_dtype_t).squeeze()
            return torch.nan_to_num(output_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            inputs_dict = {}
            if 'mel' in self.input_names:
                inputs_dict['mel'] = mel.detach().cpu().numpy()
            elif 'audio' in self.input_names:
                inputs_dict['audio'] = audio.detach().cpu().numpy()
            elif len(self.input_names) > 0:
                first_input = list(self.input_names.keys())[0]
                inputs_dict[first_input] = mel.detach().cpu().numpy()

            if 'n_samples' in self.input_names:
                inputs_dict['n_samples'] = n_samples

            if 'threshold' in self.input_names:
                inputs_dict['threshold'] = self.threshold

            output: list[np.ndarray] = self.onnx_session.run(
                [output_node],
                inputs_dict,
            )

            output_tensor = torch.as_tensor(output[0], dtype=self.fp_dtype_t, device=audio.device).squeeze()
            return torch.nan_to_num(output_tensor, nan=0.0, posinf=0.0, neginf=0.0)