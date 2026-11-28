from typing import Protocol
import torch


def decode_salience_to_f0(
    hidden: torch.Tensor,
    threshold: float,
    idx: torch.Tensor | None = None,
    idx_cents: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode [B, T, 360] pitch salience into (f0_hz, raw_confidence) using 9-bin local average."""
    if idx is None:
        idx = torch.arange(hidden.shape[-1], device=hidden.device)[None, None, :]
    if idx_cents is None:
        idx_cents = idx * 20 + 1997.3794084376191

    center = torch.argmax(hidden, dim=2, keepdim=True)
    start = torch.clip(center - 4, min=0)
    end = torch.clip(center + 5, max=hidden.shape[-1])
    idx_mask = (idx >= start) & (idx < end)
    weights = hidden * idx_mask
    product_sum = torch.sum(weights * idx_cents, dim=2)
    weight_sum = torch.sum(weights, dim=2)
    cents = product_sum / (weight_sum + (weight_sum == 0))
    f0 = 10 * 2 ** (cents / 1200)
    conf = hidden.max(dim=2)[0]
    uv = conf < threshold
    return f0 * ~uv, conf


def calibrate_voicing_confidence(
    f0: torch.Tensor,
    raw_conf: torch.Tensor,
    floor: float = 0.020,
    scale: float = 0.08,
) -> torch.Tensor:
    """Calibrate raw salience peak probability into voicing confidence in [0, 1]."""
    raw_conf = raw_conf.float()
    above = torch.clamp(raw_conf - floor, min=0.0)
    return torch.where(f0 > 0, above / (above + scale + 1e-6), torch.zeros_like(raw_conf))


def load_onnx_model_safe(file: str, is_half: bool):
    """Load an ONNX model with FP16 conversion, falling back to FP32 if conversion fails."""
    import logging
    import onnx
    from voice_changer.common.OnnxLoader import load_onnx_model

    if is_half:
        try:
            return load_onnx_model(file, True), True
        except Exception as e:
            logging.getLogger(__name__).warning(
                f"FP16 ONNX conversion failed for {file} ({e}); falling back to FP32."
            )
    return onnx.load(file), False


class PitchExtractor(Protocol):
    type: str

    def extract(
        self,
        audio: torch.Tensor,
        sr: int,
        window: int,
    ) -> torch.Tensor:
        ...

    def getPitchExtractorInfo(self):
        return {
            "pitchExtractorType": self.type,
        }
