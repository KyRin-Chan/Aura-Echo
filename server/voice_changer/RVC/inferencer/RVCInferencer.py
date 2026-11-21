import torch
import json
from const import EnumInferenceTypes
from safetensors import safe_open
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
from voice_changer.RVC.inferencer.Inferencer import Inferencer
from voice_changer.RVC.inferencer.rvc_models.infer_pack.models import SynthesizerTrnMs256NSFsid
import logging
from voice_changer.common.SafetensorsUtils import load_model
from voice_changer.vocoder.VocoderManager import VocoderManager

logger = logging.getLogger(__name__)


class RVCInferencer(Inferencer):
    def __init__(self):
        self._last_used_standalone_vocoder: bool = False

    def load_model(self, file: str):
        self._last_used_standalone_vocoder = False
        device_manager = DeviceManager.get_instance()
        dev = device_manager.device
        is_half = device_manager.use_fp16()
        self.set_props(EnumInferenceTypes.pyTorchRVC, file)

        # Keep torch.load for backward compatibility, but discourage the use of this loading method
        if file.endswith('.safetensors'):
            with safe_open(file, 'pt', device=str(dev) if dev.type == 'cuda' else 'cpu') as cpt:
                metadata = cpt.metadata() or {}
                config = json.loads(metadata['config'])
                vocoder = metadata.get('vocoder', 'embedded')
                model = SynthesizerTrnMs256NSFsid(*config, is_half=is_half, vocoder=vocoder).to(dev)
                load_model(model, cpt, strict=False)
        else:
            cpt = torch.load(file, map_location=dev if dev.type == 'cuda' else 'cpu')
            vocoder = cpt.get('vocoder', 'embedded') if isinstance(cpt, dict) else 'embedded'
            model = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=is_half, vocoder=vocoder).to(dev)
            model.load_state_dict(cpt["weight"], strict=False)

        model.eval().remove_weight_norm()
        self.is_refinegan = getattr(model, "is_refinegan", False)

        if is_half:
            model = model.half()

        self.model = model
        return self

    def uses_standalone_vocoder(self) -> bool:
        return bool(getattr(self, "_last_used_standalone_vocoder", False))

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

        self._last_used_standalone_vocoder = False
        if not getattr(self, "is_refinegan", False):
            try:
                active_vocoder = VocoderManager.get_instance().get_active_vocoder()
                if active_vocoder is not None and getattr(active_vocoder, "is_loaded", False):
                    z_masked, nsff0_slice, g = self.model.infer_latent(
                        feats,
                        pitch_length,
                        pitch,
                        pitchf,
                        sid,
                        skip_head,
                        return_length,
                    )
                    o = active_vocoder.infer(z_masked, nsff0_slice, g=g)
                    if o is not None and o.shape[-1] > 0:
                        self._last_used_standalone_vocoder = True
                        res = o[0, 0]
                        return torch.clip(res, -1.0, 1.0, out=res)
            except Exception as e:
                logger.warning(f"Error during active vocoder inference: {e}. Falling back to default decoder.")

        res = self.model.infer(
            feats,
            pitch_length,
            pitch,
            pitchf,
            sid,
            skip_head=skip_head,
            return_length=return_length,
            formant_length=formant_length
        )
        res = res[0][0, 0]
        return torch.clip(res, -1.0, 1.0, out=res)
