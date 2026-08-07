import torch
import json
import logging
from safetensors import safe_open
from const import EnumInferenceTypes
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
from voice_changer.RVC.inferencer.Inferencer import Inferencer
from .rvc_models.infer_pack.models import SynthesizerTrnMs768NSFsid
from voice_changer.common.SafetensorsUtils import load_model

logger = logging.getLogger(__name__)


def _load_state_dict_compat(model: torch.nn.Module, raw: dict, strict: bool = False):
    """
    Load state dict with compatibility for both:
    - Old weight_norm format: .weight_g / .weight_v
    - New parametrizations format: .parametrizations.weight.original0 / .original1
    Merges old-format weight_g/weight_v pairs into a single .weight tensor
    so that models using the new parametrizations API can load old checkpoints.
    """
    def _wn_merge(wg: torch.Tensor, wv: torch.Tensor) -> torch.Tensor:
        norm = wv.reshape(wv.shape[0], -1).norm(dim=1)
        for _ in range(wv.dim() - 1):
            norm = norm.unsqueeze(-1)
        return wg * wv / (norm + 1e-8)

    merged: dict = {}
    skip: set = set()
    for k, v in raw.items():
        if k.endswith(".weight_g") and k not in skip:
            base = k[:-len(".weight_g")]
            kv = base + ".weight_v"
            if kv in raw:
                merged[base + ".weight"] = _wn_merge(v, raw[kv])
                skip.update([k, kv])
        elif k.endswith(".weight_v") and k not in skip:
            skip.add(k)  # already handled above
    for k, v in raw.items():
        if k not in skip:
            merged[k] = v

    missing, unexpected = model.load_state_dict(merged, strict=strict)
    if missing:
        logger.debug(f"load_state_dict missing keys ({len(missing)}): {missing[:3]}")
    if unexpected:
        logger.debug(f"load_state_dict unexpected keys ({len(unexpected)}): {unexpected[:3]}")
    return missing, unexpected

class RVCInferencerv2(Inferencer):
    def load_model(self, file: str):
        device_manager = DeviceManager.get_instance()
        dev = device_manager.device
        is_half = device_manager.use_fp16()
        use_jit_compile = device_manager.use_jit_compile()
        self.set_props(EnumInferenceTypes.pyTorchRVCv2, file)

        # Keep torch.load for backward compatibility, but discourage the use of this loading method
        if file.endswith('.safetensors'):
            with safe_open(file, 'pt', device=str(dev) if dev.type == 'cuda' else 'cpu') as cpt:
                metadata = cpt.metadata() or {}
                config = json.loads(metadata['config'])
                vocoder = metadata.get('vocoder', 'embedded')
                model = SynthesizerTrnMs768NSFsid(*config, is_half=is_half, vocoder=vocoder).to(dev)
                load_model(model, cpt, strict=False)
        else:
            cpt = torch.load(file, map_location=dev if dev.type == 'cuda' else 'cpu')
            vocoder = cpt.get('vocoder', 'embedded') if isinstance(cpt, dict) else 'embedded'
            model = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=is_half, vocoder=vocoder).to(dev)
            # Use compat loader to handle both old weight_g/weight_v and new parametrizations formats
            # This is critical for RefineGAN models whose dec.* weights may use old weight_norm API
            _load_state_dict_compat(model, cpt["weight"], strict=False)
        # Strip parametrizations (weight_norm) exactly like Applio Realtime Voice Converter
        try:
            from torch.nn.utils.parametrize import strip_parametrizations
            strip_parametrizations(model)
        except Exception:
            model.remove_weight_norm()

        if is_half:
            model = model.half()

        self.use_jit_eager = not use_jit_compile
        if use_jit_compile:
            logger.info('Compiling JIT model...')
            try:
                model = torch.jit.optimize_for_inference(torch.jit.script(model), other_methods=['infer'])
            except Exception as e:
                logger.warning(f"JIT compilation failed: {e}. Falling back to eager mode.")
                self.use_jit_eager = True

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

        with torch.jit.optimized_execution(self.use_jit_eager):
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
