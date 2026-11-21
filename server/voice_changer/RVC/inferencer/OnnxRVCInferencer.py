import torch
import onnxruntime
import json
from const import EnumInferenceTypes
from voice_changer.common.OnnxLoader import load_onnx_model
from voice_changer.common.OnnxIOHelper import (
    OnnxCudaIOBufferCache,
    create_ort_session_with_stream,
    ort_cuda_context,
)
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
from voice_changer.RVC.inferencer.Inferencer import Inferencer
import numpy as np

class OnnxRVCInferencer(Inferencer):
    def load_model(self, file: str):
        device_manager = DeviceManager.get_instance()
        self.is_half = device_manager.use_fp16()
        self._cuda_stream = device_manager.infer_stream

        self.set_props(EnumInferenceTypes.onnxRVC, file)

        model = load_onnx_model(file, self.is_half)

        self.fp_dtype_t = torch.float16 if self.is_half else torch.float32
        self.fp_dtype_np = np.float16 if self.is_half else np.float32

        so = onnxruntime.SessionOptions()
        so.log_severity_level = 3
        self.model, self._bound_to_stream = create_ort_session_with_stream(
            model, so, device_manager, cuda_stream=self._cuda_stream
        )
        self._io_cache = OnnxCudaIOBufferCache()

        metadata = json.loads(self.model.get_modelmeta().custom_metadata_map["metadata"])
        self.inferencerTypeVersion = metadata['version']

        return self

    def infer(
        self,
        feats: torch.Tensor,
        pitch_length: torch.Tensor,
        pitch: torch.Tensor,
        pitchf: torch.Tensor,
        sid: torch.Tensor,
        skip_head: int,
        return_length: int,
        formant_length: int,
    ) -> torch.Tensor:
        assert pitch is not None or pitchf is not None, "Pitch or Pitchf is not found."

        if feats.device.type == 'cuda':
            feats_t = feats.to(dtype=self.fp_dtype_t).contiguous()
            pitch_len_t = pitch_length.to(dtype=torch.int64).contiguous()
            pitch_t = pitch.to(dtype=torch.int64).contiguous()
            pitchf_t = pitchf.to(dtype=self.fp_dtype_t).contiguous()
            sid_t = sid.to(dtype=torch.int64).contiguous()
            with ort_cuda_context(feats_t.device, self._bound_to_stream, self._cuda_stream):
                binding = self.model.io_binding()

                binding.bind_input('feats', device_type='cuda', device_id=feats_t.device.index, element_type=self.fp_dtype_np, shape=tuple(feats_t.shape), buffer_ptr=feats_t.data_ptr())
                binding.bind_input('p_len', device_type='cuda', device_id=feats_t.device.index, element_type=np.int64, shape=tuple(pitch_len_t.shape), buffer_ptr=pitch_len_t.data_ptr())
                binding.bind_input('pitch', device_type='cuda', device_id=feats_t.device.index, element_type=np.int64, shape=tuple(pitch_t.shape), buffer_ptr=pitch_t.data_ptr())
                binding.bind_input('pitchf', device_type='cuda', device_id=feats_t.device.index, element_type=self.fp_dtype_np, shape=tuple(pitchf_t.shape), buffer_ptr=pitchf_t.data_ptr())
                binding.bind_input('sid', device_type='cuda', device_id=feats_t.device.index, element_type=np.int64, shape=tuple(sid_t.shape), buffer_ptr=sid_t.data_ptr())
                binding.bind_cpu_input('skip_head', np.array(skip_head, dtype=np.int64))
                binding.bind_cpu_input('return_length', np.array(return_length, dtype=np.int64))
                binding.bind_cpu_input('formant_length', np.array(formant_length, dtype=np.int64))

                input_sig = (tuple(feats_t.shape), skip_head, return_length, formant_length)
                res = self._io_cache.run(self.model, binding, ['audio'], input_sig, feats_t.device)[0]
        else:
            output = self.model.run(
                ["audio"],
                {
                    "feats": feats.to(dtype=self.fp_dtype_t).detach().cpu().numpy(),
                    "p_len": pitch_length.detach().cpu().numpy(),
                    "pitch": pitch.detach().cpu().numpy(),
                    "pitchf": pitchf.to(dtype=self.fp_dtype_t).detach().cpu().numpy(),
                    "sid": sid.detach().cpu().numpy(),
                    "skip_head": np.array(skip_head, dtype=np.int64),
                    "return_length": np.array(return_length, dtype=np.int64),
                    "formant_length": np.array(formant_length, dtype=np.int64),
                },
            )
            res = torch.as_tensor(output[0], dtype=self.fp_dtype_t, device=feats.device)
        # self.model.end_profiling()

        res = res.to(dtype=self.fp_dtype_t)

        if self.inferencerTypeVersion == "2.1" or self.inferencerTypeVersion == "2.2" or self.inferencerTypeVersion == "1.1":
            return res
        return torch.clip(res[0, 0], -1.0, 1.0)

    def getInferencerInfo(self):
        inferencer = super().getInferencerInfo()
        inferencer["onnxExecutionProvider"] = self.model.get_providers()
        return inferencer
