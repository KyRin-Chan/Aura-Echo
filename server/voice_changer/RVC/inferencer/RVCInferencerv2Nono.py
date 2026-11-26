import torch
import json
import logging
from safetensors import safe_open
from const import EnumInferenceTypes
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
from voice_changer.RVC.inferencer.Inferencer import Inferencer
from .rvc_models.infer_pack.models import SynthesizerTrnMs768NSFsid_nono
from voice_changer.common.SafetensorsUtils import load_model
from voice_changer.RVC.inferencer.RVCInferencerv2 import _load_state_dict_compat

logger = logging.getLogger(__name__)

class RVCInferencerv2Nono(Inferencer):
    def load_model(self, file: str):
        device_manager = DeviceManager.get_instance()
        dev = device_manager.device
        target_dtype = device_manager.get_torch_dtype()
        is_half = target_dtype != torch.float32
        use_jit_compile = device_manager.use_jit_compile()
        self.set_props(EnumInferenceTypes.pyTorchRVCv2Nono, file)

        # Keep torch.load for backward compatibility, but discourage the use of this loading method
        if file.endswith('.safetensors'):
            with safe_open(file, 'pt', device=str(dev) if dev.type == 'cuda' else 'cpu') as cpt:
                config = json.loads(cpt.metadata()['config'])
                model = SynthesizerTrnMs768NSFsid_nono(*config, is_half=is_half).to(dev)
                load_model(model, cpt, strict=False)
        else:
            cpt = torch.load(file, map_location=dev if dev.type == 'cuda' else 'cpu')
            model = SynthesizerTrnMs768NSFsid_nono(*cpt["config"], is_half=is_half).to(dev)
            _load_state_dict_compat(model, cpt["weight"], strict=False)
        # Strip parametrizations (weight_norm) exactly like Applio Realtime Voice Converter
        try:
            from torch.nn.utils.parametrize import strip_parametrizations
            strip_parametrizations(model)
        except Exception:
            model.remove_weight_norm()

        model.eval()

        if target_dtype != torch.float32:
            model = model.to(dtype=target_dtype)
        self.model_dtype = target_dtype

        self.use_jit = use_jit_compile
        if use_jit_compile:
            logger.info('Compiling JIT model...')
            try:
                model = torch.jit.optimize_for_inference(torch.jit.script(model), other_methods=['infer'])
            except Exception as e:
                logger.warning(f"JIT compilation failed: {e}. Falling back to eager mode.")
                self.use_jit = False

        self.model = model
        return self

    @torch.no_grad()
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
        with torch.jit.optimized_execution(self.use_jit):
            out = self.model.infer(
                feats.to(dtype=model_dtype),
                pitch_length,
                sid,
                skip_head=skip_head,
                return_length=return_length,
                formant_length=formant_length,
            )
            res = out[0][0, 0]
        return res
