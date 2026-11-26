import hashlib
import os
from time import perf_counter
import torch
import json
import logging
from safetensors import safe_open
from const import EnumInferenceTypes, TMP_DIR
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
from voice_changer.RVC.inferencer.Inferencer import Inferencer
from .rvc_models.infer_pack.models import SynthesizerTrnMs768NSFsid
from voice_changer.common.SafetensorsUtils import load_model
from voice_changer.vocoder.VocoderManager import VocoderManager

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

def _get_jit_cache_path(file: str, dev: torch.device, target_dtype: torch.dtype) -> str | None:
    """Compute a deterministic TorchScript disk cache path for a checkpoint + device + dtype."""
    try:
        abs_path = os.path.abspath(file)
        st = os.stat(abs_path)
        cuda_cap = ""
        if dev.type == "cuda":
            major, minor = torch.cuda.get_device_capability(dev)
            cuda_cap = f"sm_{major}{minor}"
        key = "|".join([
            "rvc_v2_jit_v1",
            abs_path,
            str(st.st_size),
            str(st.st_mtime_ns),
            dev.type,
            cuda_cap,
            str(target_dtype),
            torch.__version__,
        ])
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
        return os.path.join(TMP_DIR, "jit_cache", f"{digest}.ts")
    except Exception:
        return None


class RVCInferencerv2(Inferencer):
    def __init__(self):
        self._last_used_standalone_vocoder: bool = False

    def load_model(self, file: str):
        self._last_used_standalone_vocoder = False
        device_manager = DeviceManager.get_instance()
        dev = device_manager.device
        target_dtype = device_manager.get_torch_dtype()
        is_half = target_dtype != torch.float32
        use_jit_compile = device_manager.use_jit_compile()
        self.set_props(EnumInferenceTypes.pyTorchRVCv2, file)

        cache_path = _get_jit_cache_path(file, dev, target_dtype) if use_jit_compile else None
        if cache_path and os.path.isfile(cache_path):
            t0 = perf_counter()
            try:
                extra_files = {"meta.json": ""}
                model = torch.jit.load(cache_path, map_location=dev, _extra_files=extra_files)
                if not hasattr(model, "infer"):
                    raise RuntimeError("Cached TorchScript module is missing 'infer' method")
                meta_raw = extra_files.get("meta.json", b"")
                if isinstance(meta_raw, bytes):
                    meta_raw = meta_raw.decode("utf-8", errors="ignore")
                meta = json.loads(meta_raw) if meta_raw else {}
                self.is_refinegan = bool(meta.get("is_refinegan", False))
                self.model_dtype = target_dtype
                self.use_jit = True
                self.model = model
                elapsed_ms = (perf_counter() - t0) * 1000.0
                logger.info(f"Loaded cached JIT model from {cache_path} in {elapsed_ms:.1f} ms.")
                return self
            except Exception as e:
                logger.warning(f"Failed to load cached JIT model ({e}); recompiling.")
                try:
                    os.remove(cache_path)
                except OSError:
                    pass

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

        model.eval()
        self.is_refinegan = getattr(model, "is_refinegan", False)

        if target_dtype != torch.float32:
            model = model.to(dtype=target_dtype)
        self.model_dtype = target_dtype

        self.use_jit = use_jit_compile
        if use_jit_compile:
            logger.info('Compiling JIT model...')
            try:
                model = torch.jit.optimize_for_inference(
                    torch.jit.script(model),
                    other_methods=['infer', 'infer_latent'],
                )
                if cache_path:
                    tmp_path = f"{cache_path}.{os.getpid()}.tmp"
                    try:
                        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                        extra_files = {"meta.json": json.dumps({"is_refinegan": self.is_refinegan})}
                        torch.jit.save(model, tmp_path, _extra_files=extra_files)
                        os.replace(tmp_path, cache_path)
                        logger.info(f"Saved compiled JIT model cache to {cache_path}")
                    except Exception as cache_err:
                        logger.warning(f"Failed to save JIT model cache: {cache_err}")
                        try:
                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)
                        except OSError:
                            pass
            except Exception as e:
                logger.warning(f"JIT compilation failed: {e}. Falling back to eager mode.")
                self.use_jit = False

        self.model = model
        return self

    def uses_standalone_vocoder(self) -> bool:
        return bool(getattr(self, "_last_used_standalone_vocoder", False))

    @torch.no_grad()
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

        active_vocoder = None
        if not getattr(self, "is_refinegan", False):
            try:
                candidate = VocoderManager.get_instance().get_active_vocoder()
                if candidate is not None and getattr(candidate, "is_loaded", False):
                    active_vocoder = candidate
            except Exception:
                active_vocoder = None

        model_dtype = getattr(self, "model_dtype", feats.dtype)
        feats_t = feats.to(dtype=model_dtype)
        pitchf_t = pitchf.to(dtype=model_dtype) if pitchf is not None else pitchf

        self._last_used_standalone_vocoder = False
        with torch.jit.optimized_execution(self.use_jit):
            if active_vocoder is not None:
                try:
                    z_masked, nsff0_slice, g = self.model.infer_latent(
                        feats_t,
                        pitch_length,
                        pitch,
                        pitchf_t,
                        sid,
                        skip_head,
                        return_length,
                    )
                    o = active_vocoder.infer(z_masked, nsff0_slice, g=g)
                    if o is not None and o.shape[-1] > 0:
                        self._last_used_standalone_vocoder = True
                        return o[0, 0]
                except Exception as e:
                    logger.warning(f"Error during active vocoder inference: {e}. Falling back to default decoder.")

            out = self.model.infer(
                feats_t,
                pitch_length,
                pitch,
                pitchf_t,
                sid,
                skip_head=skip_head,
                return_length=return_length,
                formant_length=formant_length,
            )
            res = out[0][0, 0]
        return res
