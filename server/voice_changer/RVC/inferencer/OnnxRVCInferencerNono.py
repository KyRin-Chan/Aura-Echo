import torch
import numpy as np
from const import EnumInferenceTypes

from voice_changer.common.OnnxIOHelper import ort_cuda_context
from voice_changer.RVC.inferencer.OnnxRVCInferencer import OnnxRVCInferencer


class OnnxRVCInferencerNono(OnnxRVCInferencer):
    def load_model(self, file: str):
        super().load_model(file)
        self.set_props(EnumInferenceTypes.onnxRVCNono, file)
        return self

    def infer(
        self,
        feats: torch.Tensor,
        pitch_length: torch.Tensor,
        pitch: torch.Tensor | None,
        pitchf: torch.Tensor | None,
        sid: torch.Tensor,
        skip_head: int,
        return_length: int,
        formant_length: int,
    ) -> torch.Tensor:
        if feats.device.type == 'cuda':
            feats_t = feats.to(dtype=self.fp_dtype_t).contiguous()
            pitch_len_t = pitch_length.to(dtype=torch.int64).contiguous()
            sid_t = sid.to(dtype=torch.int64).contiguous()
            with ort_cuda_context(feats_t.device, self._bound_to_stream, self._cuda_stream):
                binding = self.model.io_binding()

                binding.bind_input('feats', device_type='cuda', device_id=feats_t.device.index, element_type=self.fp_dtype_np, shape=tuple(feats_t.shape), buffer_ptr=feats_t.data_ptr())
                binding.bind_input('p_len', device_type='cuda', device_id=feats_t.device.index, element_type=np.int64, shape=tuple(pitch_len_t.shape), buffer_ptr=pitch_len_t.data_ptr())
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
                    "sid": sid.detach().cpu().numpy(),
                    "skip_head": np.array(skip_head, dtype=np.int64),
                    "return_length": np.array(return_length, dtype=np.int64),
                    "formant_length": np.array(formant_length, dtype=np.int64),
                },
            )
            res = torch.as_tensor(output[0], dtype=self.fp_dtype_t, device=feats.device)
        # self.model.end_profiling()

        res = res.to(dtype=self.fp_dtype_t)

        if self.inferencerTypeVersion == "v2.1" or self.inferencerTypeVersion == "v2.2" or self.inferencerTypeVersion == "v1.1":
            return res
        return torch.clip(res[0, 0], -1.0, 1.0)