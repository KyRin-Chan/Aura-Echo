import torch
from voice_changer.common.OnnxLoader import load_onnx_model
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
from voice_changer.embedder.Embedder import Embedder
import onnxruntime
import numpy as np

class OnnxEmbedder(Embedder):

    def load_model(self, file: str) -> Embedder:
        device_manager = DeviceManager.get_instance()
        self.is_half = device_manager.use_fp16()
        (
            onnxProviders,
            onnxProviderOptions,
        ) = device_manager.get_onnx_execution_provider()

        model = load_onnx_model(file, self.is_half, device_manager.is_int8_avalable())

        so = onnxruntime.SessionOptions()
        so.log_severity_level = 3
        # so.enable_profiling = True
        # so.add_free_dimension_override_by_name('audio_dynamic_axes_1', 45600)
        self.fp_dtype_t = torch.float16 if self.is_half else torch.float32
        self.fp_dtype_np = np.float16 if self.is_half else np.float32
        self.onnx_session = onnxruntime.InferenceSession(model.SerializeToString(), sess_options=so, providers=onnxProviders, provider_options=onnxProviderOptions)
        super().set_props(self.embedderType, file)
        return self

    def extract_features(
        self, feats: torch.Tensor, embOutputLayer=9, useFinalProj=True
    ) -> torch.Tensor:
        if feats.device.type == 'cuda':
            binding = self.onnx_session.io_binding()

            binding.bind_input('audio', device_type='cuda', device_id=feats.device.index, element_type=self.fp_dtype_np, shape=tuple(feats.shape), buffer_ptr=feats.data_ptr())
            for output in self.onnx_session.get_outputs():
                binding.bind_output(output.name, device_type='cuda', device_id=feats.device.index)

            self.onnx_session.run_with_iobinding(binding)

            outputs = binding.get_outputs()
            from torch.utils.dlpack import from_dlpack
            units = []
            for out_val in outputs:
                if hasattr(out_val, 'to_dlpack'):
                    units.append(from_dlpack(out_val.to_dlpack()))
                elif hasattr(out_val, '_ortvalue') and hasattr(out_val._ortvalue, 'to_dlpack'):
                    units.append(from_dlpack(out_val._ortvalue.to_dlpack()))
                else:
                    units.append(torch.from_numpy(out_val.numpy()).to(feats.device))
        else:
            output_np = self.onnx_session.run(
                ['units9', 'unit12', 'unit12s'],
                { 'audio': feats.detach().cpu().numpy() }
            )
            units = [torch.as_tensor(u, dtype=self.fp_dtype_t, device=feats.device) for u in output_np]
        # self.onnx_session.end_profiling()

        res = units[0] if embOutputLayer == 9 else units[1]
        return res.to(dtype=self.fp_dtype_t)
