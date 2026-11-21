import torch
from const import EmbedderType
from voice_changer.common.OnnxLoader import load_onnx_model
from voice_changer.common.OnnxIOHelper import (
    OnnxCudaIOBufferCache,
    create_ort_session_with_stream,
    ort_cuda_context,
)
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
from voice_changer.embedder.Embedder import Embedder
import onnxruntime
import numpy as np

class OnnxEmbedder(Embedder):

    def load_model(self, file: str, embedder_type: EmbedderType = "hubert_base") -> Embedder:
        device_manager = DeviceManager.get_instance()
        self.is_half = device_manager.use_fp16()
        self._cuda_stream = device_manager.embedder_stream

        model = load_onnx_model(file, self.is_half, device_manager.is_int8_avalable())

        so = onnxruntime.SessionOptions()
        so.log_severity_level = 3
        self.fp_dtype_t = torch.float16 if self.is_half else torch.float32
        self.fp_dtype_np = np.float16 if self.is_half else np.float32
        self.onnx_session, self._bound_to_stream = create_ort_session_with_stream(
            model, so, device_manager, cuda_stream=self._cuda_stream
        )
        self._output_names = [o.name for o in self.onnx_session.get_outputs()]
        self._io_cache = OnnxCudaIOBufferCache()
        super().set_props(embedder_type, file)
        return self

    def _select_output_name(self, embOutputLayer: int) -> str:
        if len(self._output_names) <= 1 or embOutputLayer == 9:
            return self._output_names[0]
        return self._output_names[1]

    def extract_features(
        self, feats: torch.Tensor, embOutputLayer=9, useFinalProj=True
    ) -> torch.Tensor:
        target_output = self._select_output_name(embOutputLayer)
        if feats.device.type == 'cuda':
            feats_t = feats.to(dtype=self.fp_dtype_t).contiguous()
            with ort_cuda_context(feats_t.device, self._bound_to_stream, self._cuda_stream):
                binding = self.onnx_session.io_binding()
                binding.bind_input(
                    'audio',
                    device_type='cuda',
                    device_id=feats_t.device.index,
                    element_type=self.fp_dtype_np,
                    shape=tuple(feats_t.shape),
                    buffer_ptr=feats_t.data_ptr(),
                )
                res = self._io_cache.run(
                    self.onnx_session,
                    binding,
                    [target_output],
                    (tuple(feats_t.shape), target_output),
                    feats_t.device,
                )[0]
        else:
            output_np = self.onnx_session.run(
                [target_output],
                {'audio': feats.to(dtype=self.fp_dtype_t).detach().cpu().numpy()},
            )
            res = torch.as_tensor(output_np[0], dtype=self.fp_dtype_t, device=feats.device)

        return res.to(dtype=self.fp_dtype_t)
