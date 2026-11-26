import torch
from const import EnumInferenceTypes
from voice_changer.common.deviceManager.DeviceManager import DeviceManager

from voice_changer.RVC.inferencer.Inferencer import Inferencer
from voice_changer.RVC.inferencer.rvc_models.infer_pack.models_onnx import SynthesizerTrnMsNSFsidM


class WebUIInferencer(Inferencer):
    def load_model(self, file: str):
        self.set_props(EnumInferenceTypes.pyTorchWebUI, file)

        device_manager = DeviceManager.get_instance()
        dev = device_manager.device
        target_dtype = device_manager.get_torch_dtype()
        is_half = target_dtype != torch.float32

        cpt = torch.load(file, map_location="cpu")
        model = SynthesizerTrnMsNSFsidM(**cpt["params"], is_half=is_half)

        model.eval()
        model.load_state_dict(cpt["weight"], strict=False)

        model = model.to(dev)
        if target_dtype != torch.float32:
            model = model.to(dtype=target_dtype)
        self.model_dtype = target_dtype

        self.model = model
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

        model_dtype = getattr(self, "model_dtype", feats.dtype)
        res = self.model.infer(
            feats.to(dtype=model_dtype),
            pitch_length,
            pitch,
            pitchf.to(dtype=model_dtype),
            sid,
            skip_head=skip_head,
            return_length=return_length,
            formant_length=formant_length
        )
        return res[0][0, 0]
