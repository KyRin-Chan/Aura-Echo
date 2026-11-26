import torch
import json
from const import EnumInferenceTypes
from safetensors import safe_open
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
from voice_changer.RVC.inferencer.Inferencer import Inferencer
from .rvc_models.infer_pack.models import SynthesizerTrnMs256NSFsid_nono
from voice_changer.common.SafetensorsUtils import load_model


class RVCInferencerNono(Inferencer):
    def load_model(self, file: str):
        device_manager = DeviceManager.get_instance()
        dev = device_manager.device
        target_dtype = device_manager.get_torch_dtype()
        is_half = target_dtype != torch.float32
        self.set_props(EnumInferenceTypes.pyTorchRVCNono, file)

        # Keep torch.load for backward compatibility, but discourage the use of this loading method
        if file.endswith('.safetensors'):
            with safe_open(file, 'pt', device=str(dev) if dev.type == 'cuda' else 'cpu') as cpt:
                config = json.loads(cpt.metadata()['config'])
                model = SynthesizerTrnMs256NSFsid_nono(*config, is_half=is_half).to(dev)
                load_model(model, cpt, strict=False)
        else:
            cpt = torch.load(file, map_location=dev if dev.type == 'cuda' else 'cpu')
            model = SynthesizerTrnMs256NSFsid_nono(*cpt["config"], is_half=is_half).to(dev)
            model.load_state_dict(cpt["weight"], strict=False)

        model.eval().remove_weight_norm()

        if target_dtype != torch.float32:
            model = model.to(dtype=target_dtype)
        self.model_dtype = target_dtype

        self.model = model
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
        model_dtype = getattr(self, "model_dtype", feats.dtype)
        res = self.model.infer(
            feats.to(dtype=model_dtype),
            pitch_length,
            sid,
            skip_head=skip_head,
            return_length=return_length,
            formant_length=formant_length
        )
        return res[0][0, 0]
