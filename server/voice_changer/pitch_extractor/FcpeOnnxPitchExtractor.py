import numpy as np
import torch
import onnxruntime
from const import PitchExtractorType
from voice_changer.pitch_extractor.PitchExtractor import PitchExtractor
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
from voice_changer.common.OnnxLoader import load_onnx_model
from voice_changer.common.MelExtractorFcpe import Wav2MelModule

class FcpeOnnxPitchExtractor(PitchExtractor):

    def __init__(self, file: str):
        super().__init__()
        self.file = file
        self.type: PitchExtractorType = "fcpe_onnx"

        device_manager = DeviceManager.get_instance()
        # NOTE: FCPE doesn't seem to be behave correctly in FP16 mode.
        # self.is_half = device_manager.use_fp16()
        self.is_half = False
        (
            onnxProviders,
            onnxProviderOptions,
        ) = device_manager.get_onnx_execution_provider()

        model = load_onnx_model(file, self.is_half)

        self.fp_dtype_t = torch.float16 if self.is_half else torch.float32
        self.fp_dtype_np = np.float16 if self.is_half else np.float32

        self.threshold = np.array([0.006], dtype=self.fp_dtype_np)

        so = onnxruntime.SessionOptions()
        so.log_severity_level = 3
        # so.enable_profiling = True
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
        self.onnx_session = onnxruntime.InferenceSession(model.SerializeToString(), sess_options=so, providers=onnxProviders, provider_options=onnxProviderOptions)

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
        n_samples = np.array([mel.shape[1]], dtype=np.int64)

        output_node = self.output_names[0] if len(self.output_names) > 0 else "pitchf"

        if audio.device.type == 'cuda':
            binding = self.onnx_session.io_binding()

            # Dynamically bind inputs matching actual ONNX node definitions
            if 'mel' in self.input_names:
                binding.bind_input('mel', device_type='cuda', device_id=audio.device.index, element_type=self.fp_dtype_np, shape=tuple(mel.shape), buffer_ptr=mel.contiguous().data_ptr())
            elif 'audio' in self.input_names:
                binding.bind_input('audio', device_type='cuda', device_id=audio.device.index, element_type=self.fp_dtype_np, shape=tuple(audio.shape), buffer_ptr=audio.contiguous().data_ptr())
            elif len(self.input_names) > 0:
                first_input = list(self.input_names.keys())[0]
                binding.bind_input(first_input, device_type='cuda', device_id=audio.device.index, element_type=self.fp_dtype_np, shape=tuple(mel.shape), buffer_ptr=mel.contiguous().data_ptr())

            if 'n_samples' in self.input_names:
                binding.bind_cpu_input('n_samples', n_samples)

            if 'threshold' in self.input_names:
                binding.bind_cpu_input('threshold', self.threshold)

            binding.bind_output(output_node, device_type='cuda', device_id=audio.device.index)

            self.onnx_session.run_with_iobinding(binding)

            outputs = binding.get_outputs()
            from torch.utils.dlpack import from_dlpack
            out_val = outputs[0]
            if hasattr(out_val, 'to_dlpack'):
                output_tensor = from_dlpack(out_val.to_dlpack())
            elif hasattr(out_val, '_ortvalue') and hasattr(out_val._ortvalue, 'to_dlpack'):
                output_tensor = from_dlpack(out_val._ortvalue.to_dlpack())
            else:
                output_tensor = torch.from_numpy(out_val.numpy()).to(audio.device)

            return output_tensor.to(dtype=self.fp_dtype_t).squeeze()
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

            return torch.as_tensor(output[0], dtype=self.fp_dtype_t, device=audio.device).squeeze()