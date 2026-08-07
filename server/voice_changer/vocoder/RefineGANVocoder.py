"""
RefineGAN standalone vocoder for RVC voice changer.

Key design:
  - SineGenerator: synthesises harmonic excitation from F0
  - AdaIN: adaptive instance normalisation with learnable Gaussian noise
  - ResBlock: stacked dilated + unit-dilation conv residual pairs
  - ParallelResBlock: three parallel ResBlocks (kernels 3/7/11) averaged
  - RefineGANGenerator:
      F0 → SineGenerator → pre_conv → sinc-kaiser multi-scale downsamples
      z   → mel_conv (+ speaker cond) → concat with lowest F0 scale
           → iterative Upsample + ParallelResBlock (fusing each F0 scale)
           → conv_post + tanh → waveform

Default pretrained model: f0G32k.pth
  - native sample rate : 32 000 Hz
  - upsample_rates     : (10, 8, 2, 2) → upp = 320
  - frame rate         : 32 000 / 320 = 100 fps  (= 10 ms / frame)
  - num_mels           : 192 (RVC inter_channels / latent z channels)

After resampling 32 k → model_sr the output length per frame is:
    320 × (model_sr / 32 000) = model_sr / 100 = embedded_upp
so the output length always matches the embedded decoder's upp, for any
supported model sample rate (32 k, 40 k, 48 k …).
"""

import math
import os
import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio import transforms as tat

try:
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:          # PyTorch < 1.8 fallback
    from torch.nn.utils import weight_norm  # type: ignore[assignment]

from torch.nn.utils import remove_weight_norm

from voice_changer.vocoder.Vocoder import Vocoder
from voice_changer.common.deviceManager.DeviceManager import DeviceManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (inlined to avoid dependency on Applio's commons module)
# ---------------------------------------------------------------------------

def _safe_remove_weight_norm(m: nn.Module) -> None:
    """Safely remove weight norm or parametrization hook from a module."""
    if not isinstance(m, nn.Module):
        return
    # PyTorch 2.0+ parametrizations
    try:
        from torch.nn.utils.parametrize import remove_parametrizations
        if hasattr(m, "parametrizations") and "weight" in m.parametrizations:
            remove_parametrizations(m, "weight")
            return
    except Exception:
        pass
    # Legacy PyTorch weight_norm hook
    try:
        remove_weight_norm(m)
    except Exception:
        pass


def _get_padding(kernel_size: int, dilation: int = 1) -> int:
    """Same-length padding for a causal-free dilated Conv1d."""
    return (kernel_size * dilation - dilation) // 2


def _init_weights(m: nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    """Initialise Conv weight with a small normal distribution."""
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        m.weight.data.normal_(mean, std)


def _precompute_kaiser_sinc_kernel(
    orig_ratio: int,
    new_ratio: int,
    lowpass_filter_width: int = 64,
    rolloff: float = 0.9475937167399596,
    beta: float = 14.769656459379492,
) -> torch.Tensor:
    """
    Precompute a Kaiser-windowed sinc FIR resampling kernel (polyphase form).

    Returns shape (new_ratio, 1, 2*width) for use with F.conv1d(stride=orig_ratio).
    Functionally identical to torchaudio.functional.resample(sinc_interp_kaiser)
    but materialised once at __init__ so it can live as an nn.Buffer and be
    cast to any dtype (including FP16) via the normal .to() / .half() path.

    Algorithm mirrors torchaudio/_resample.py::_get_sinc_resample_kernel.
    """
    import math as _math
    base_freq = min(orig_ratio, new_ratio) * rolloff
    width = _math.ceil(lowpass_filter_width * orig_ratio / base_freq)

    kernels = []
    for i in range(new_ratio):
        t = torch.arange(-width, width, dtype=torch.float64) + i / new_ratio
        t = t * base_freq / orig_ratio
        t = t.clamp(-lowpass_filter_width, lowpass_filter_width)

        # Sinc  sinc(πt) = sin(πt)/(πt), with sinc(0)=1
        kernel = torch.where(
            t == 0,
            torch.ones_like(t),
            torch.sin(_math.pi * t) / (_math.pi * t),
        )
        # Kaiser window  I₀(β√(1-(t/M)²)) / I₀(β)
        arg = beta * torch.sqrt(torch.clamp(1.0 - (t / lowpass_filter_width) ** 2, min=0.0))
        window = torch.special.i0(arg) / torch.special.i0(torch.tensor(beta, dtype=torch.float64))
        kernel = kernel * window
        kernels.append(kernel)

    scale = base_freq / orig_ratio
    out = torch.stack(kernels).view(new_ratio, 1, -1).float() * scale
    return out  # (new_ratio, 1, 2*width)


# ---------------------------------------------------------------------------
# RefineGAN building blocks
# ---------------------------------------------------------------------------

class _SineGenerator(nn.Module):
    """
    Harmonic sine excitation source driven by F0.

    Generates voiced sine waves + harmonics; replaces them with Gaussian
    noise in unvoiced regions.

    Args:
        samp_rate       : audio sample rate (Hz)
        harmonic_num    : number of additional harmonic overtones (0 = F0 only)
        sine_amp        : amplitude of sine waveforms
        noise_std       : std-dev of additive Gaussian noise for voiced regions
        voiced_threshold: F0 threshold below which a frame is treated as unvoiced
    """

    def __init__(
        self,
        samp_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0.0,
    ):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

        # Linearly mix harmonics → 1 channel, then apply tanh
        self.merge = nn.Sequential(
            nn.Linear(self.dim, 1, bias=False),
            nn.Tanh(),
        )

    def _f02uv(self, f0: torch.Tensor) -> torch.Tensor:
        """Binary voiced/unvoiced mask from F0 values."""
        return (f0 > self.voiced_threshold).float()

    def _f02sine(self, f0_values: torch.Tensor) -> torch.Tensor:
        """
        Convert F0 (B, T, dim) to sine waves (B, T, dim).
        Uses phase accumulation to handle discontinuities at pitch changes.
        """
        rad_values = (f0_values / self.sampling_rate) % 1.0

        # Random initial phase for harmonics (not for the fundamental)
        rand_ini = torch.rand(
            f0_values.shape[0], f0_values.shape[2], device=f0_values.device
        )
        rand_ini[:, 0] = 0.0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        # Cumulative phase with wrap-around correction
        tmp = torch.cumsum(rad_values, dim=1) % 1.0
        idx = (tmp[:, 1:, :] - tmp[:, :-1, :]) < 0
        shift = torch.zeros_like(rad_values)
        shift[:, 1:, :] = idx * -1.0

        sines = torch.sin(
            torch.cumsum(rad_values + shift, dim=1) * (2.0 * math.pi)
        )
        return sines

    def forward(self, f0: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f0: (B, T, 1) — F0 contour at waveform sample rate
        Returns:
            (B, T, 1) — harmonic + noise excitation signal
        """
        # Phase accumulation via cumsum is precision-sensitive: always run in float32
        # to avoid overflow when operating in FP16 mode (float16 max ≈ 65504).
        f0_fp32 = f0.float()
        with torch.no_grad():
            f0_buf = torch.zeros(
                f0_fp32.shape[0], f0_fp32.shape[1], self.dim,
                device=f0_fp32.device, dtype=torch.float32
            )
            f0_buf[:, :, 0] = f0_fp32[:, :, 0]
            for i in range(self.harmonic_num):
                f0_buf[:, :, i + 1] = f0_buf[:, :, 0] * (i + 2)

            sine_waves = self._f02sine(f0_buf) * self.sine_amp

            uv = self._f02uv(f0_fp32)
            noise_amp = uv * self.noise_std + (1.0 - uv) * (self.sine_amp / 3.0)
            noise = noise_amp * torch.randn_like(sine_waves)

            sine_waves = sine_waves * uv + noise

        # Cast float32 result to the Linear layer's dtype (float16 or float32)
        target_dtype = self.merge[0].weight.dtype
        return self.merge(sine_waves.to(target_dtype))


class _AdaIN(nn.Module):
    """
    Adaptive Instance Normalisation with learnable Gaussian noise injection.

    A tiny learnable scale controls the variance of per-sample noise,
    allowing the generator to model stochastic fine structure.
    """

    def __init__(self, channels: int, leaky_relu_slope: float = 0.2):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels) * 1e-4)
        self.activation = nn.LeakyReLU(leaky_relu_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gaussian = torch.randn_like(x) * self.weight[None, :, None]
        return self.activation(x + gaussian)


class _ResBlock(nn.Module):
    """
    Residual block: for each dilation d, applies
        x ← x + Conv1d(LReLU(Conv1d(LReLU(x), d)), 1)
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
        dilation: tuple = (1, 3, 5),
        leaky_relu_slope: float = 0.2,
    ):
        super().__init__()
        self.leaky_relu_slope = leaky_relu_slope

        self.convs1 = nn.ModuleList([
            weight_norm(
                nn.Conv1d(
                    channels, channels, kernel_size,
                    dilation=d,
                    padding=_get_padding(kernel_size, d),
                )
            )
            for d in dilation
        ])
        self.convs1.apply(_init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(
                nn.Conv1d(
                    channels, channels, kernel_size,
                    dilation=1,
                    padding=_get_padding(kernel_size, 1),
                )
            )
            for _ in dilation
        ])
        self.convs2.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, self.leaky_relu_slope)
            xt = c1(xt)
            xt = F.leaky_relu(xt, self.leaky_relu_slope)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for c in list(self.convs1) + list(self.convs2):
            _safe_remove_weight_norm(c)


class _ParallelResBlock(nn.Module):
    """
    Parallel multi-kernel residual block with AdaIN.

    Applies ``len(kernel_sizes)`` independent ResBlocks in parallel and
    returns their mean — each wrapped by two AdaIN layers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: tuple = (3, 7, 11),
        dilation: tuple = (1, 3, 5),
        leaky_relu_slope: float = 0.2,
    ):
        super().__init__()

        self.input_conv = nn.Conv1d(in_channels, out_channels, 7, padding=3)
        self.input_conv.apply(_init_weights)

        self.blocks = nn.ModuleList([
            nn.Sequential(
                _AdaIN(channels=out_channels, leaky_relu_slope=leaky_relu_slope),
                _ResBlock(
                    out_channels,
                    kernel_size=ks,
                    dilation=dilation,
                    leaky_relu_slope=leaky_relu_slope,
                ),
                _AdaIN(channels=out_channels, leaky_relu_slope=leaky_relu_slope),
            )
            for ks in kernel_sizes
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)
        # average across the parallel branches
        return torch.stack([blk(x) for blk in self.blocks], dim=0).mean(dim=0)

    def remove_weight_norm(self):
        _safe_remove_weight_norm(self.input_conv)
        for blk in self.blocks:
            if hasattr(blk, "__getitem__") and len(blk) > 1 and hasattr(blk[1], "remove_weight_norm"):
                blk[1].remove_weight_norm()


class _RefineGANGenerator(nn.Module):
    """
    RefineGAN Generator.

    Forward:
        mel  : (B, num_mels, T)      — latent z from the VITS flow model
        f0   : (B, T)                — F0 in Hz (0 = unvoiced)
        g    : (B, gin_channels, 1)  — speaker embedding (optional)
    Returns:
        (B, 1, T × upp)              — audio waveform in [-1, 1]

    Architecture:
        1. F0 → upsample to waveform length → SineGenerator → pre_conv
           → sinc-kaiser anti-alias downsamples × len(upsample_rates)
        2. mel_conv(mel) [+ cond(g)] → concat with coarsest F0 branch
        3. Iterative Upsample(linear) + cat(F0 branch) + ParallelResBlock
        4. conv_post → tanh
    """

    def __init__(
        self,
        sample_rate: int = 32000,
        upsample_rates: tuple = (10, 8, 2, 2),   # product = 320 → 100 fps @ 32 kHz
        num_mels: int = 192,                       # RVC inter_channels
        start_channels: int = 16,
        gin_channels: int = 256,
        upsample_initial_channel: int = 512,
        leaky_relu_slope: float = 0.2,
    ):
        super().__init__()
        self.upsample_rates = upsample_rates
        self.leaky_relu_slope = leaky_relu_slope
        self.upp: int = int(np.prod(upsample_rates))

        # --- Harmonic source ---
        self.m_source = _SineGenerator(sample_rate)

        # 1-channel sine → start_channels
        self.pre_conv = weight_norm(
            nn.Conv1d(1, start_channels, 7, padding=3)
        )

        # Multi-scale F0 downsampling branches
        channels = start_channels
        size = self.upp
        self.downsample_blocks = nn.ModuleList()
        self.df0: list = []                       # [(old_T_factor, new_T_factor), …]
        for i, _ in enumerate(upsample_rates):
            new_size = int(size / upsample_rates[-(i + 1)])
            self.df0.append((size, new_size))
            size = new_size
            new_ch = channels * 2
            self.downsample_blocks.append(
                weight_norm(nn.Conv1d(channels, new_ch, 7, padding=3))
            )
            channels = new_ch

        # --- Mel / latent z projection ---
        ch = upsample_initial_channel
        self.mel_conv = weight_norm(
            nn.Conv1d(num_mels, ch // 2, 7, padding=3)
        )
        self.mel_conv.apply(_init_weights)

        if gin_channels > 0:
            self.cond = nn.Conv1d(gin_channels, ch // 2, 1)

        # --- Decoder ---
        self.upsample_blocks = nn.ModuleList()
        self.upsample_conv_blocks = nn.ModuleList()
        for rate in upsample_rates:
            new_ch = ch // 2
            self.upsample_blocks.append(
                nn.Upsample(scale_factor=rate, mode="linear")
            )
            # in_channels = ch (mel branch) + ch // 4 (F0 branch at this scale)
            self.upsample_conv_blocks.append(
                _ParallelResBlock(
                    in_channels=ch + ch // 4,
                    out_channels=new_ch,
                    kernel_sizes=(3, 7, 11),
                    dilation=(1, 3, 5),
                    leaky_relu_slope=leaky_relu_slope,
                )
            )
            ch = new_ch

        self.conv_post = weight_norm(
            nn.Conv1d(ch, 1, 7, padding=3, bias=False)
        )
        self.conv_post.apply(_init_weights)

    def forward(
        self,
        mel: torch.Tensor,
        f0: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        target_dtype = self.pre_conv.weight.dtype
        mel = mel.to(target_dtype)
        if g is not None:
            g = g.to(target_dtype)

        f0_frames = mel.shape[-1]

        # Upsample F0 to full waveform resolution
        f0_up = F.interpolate(
            f0.unsqueeze(1), size=f0_frames * self.upp, mode="linear"
        )                                                    # (B, 1, T*upp)

        # Voiced/unvoiced sine harmonics excitation
        har = self.m_source(f0_up.transpose(1, 2)).transpose(1, 2)   # (B, 1, T*upp)

        # pre_conv: 1 ch → start_channels
        x = self.pre_conv(har)

        # Build F0 down-branches with sinc-Kaiser anti-aliasing resampling
        downs = []
        for blk, (old_sz, new_sz) in zip(self.downsample_blocks, self.df0):
            x = F.leaky_relu(x, self.leaky_relu_slope)
            downs.append(x)
            # torchaudio.functional.resample (sinc_interp_kaiser) requires float32 input.
            # Localized float32 cast ensures 100% exact Applio audio quality while keeping convs in FP16.
            x = torchaudio.functional.resample(
                x.float().contiguous(),
                orig_freq=int(f0_frames * old_sz),
                new_freq=int(f0_frames * new_sz),
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method="sinc_interp_kaiser",
                beta=14.769656459379492,
            ).to(target_dtype)
            x = blk(x)

        # Mel/z projection + optional speaker conditioning
        mel = self.mel_conv(mel)
        if g is not None:
            mel = mel + self.cond(g)

        # Concat mel branch with the coarsest F0 branch
        x = torch.cat([mel, x], dim=1)

        # Decode: linear upsample + fuse next F0 branch + ParallelResBlock
        for ups, res, down in zip(
            self.upsample_blocks, self.upsample_conv_blocks, reversed(downs)
        ):
            x = F.leaky_relu(x, self.leaky_relu_slope)
            x = ups(x)
            x = torch.cat([x, down], dim=1)
            x = res(x)

        x = F.leaky_relu(x, self.leaky_relu_slope)
        x = self.conv_post(x)
        return torch.tanh(x)

    def remove_weight_norm(self):
        _safe_remove_weight_norm(self.pre_conv)
        _safe_remove_weight_norm(self.mel_conv)
        _safe_remove_weight_norm(self.conv_post)
        for blk in self.downsample_blocks:
            _safe_remove_weight_norm(blk)
        for blk in self.upsample_conv_blocks:
            if hasattr(blk, "remove_weight_norm"):
                blk.remove_weight_norm()


# ---------------------------------------------------------------------------
# Checkpoint loading helper
# ---------------------------------------------------------------------------

def _load_state_dict_compatible(
    model: nn.Module,
    raw: dict,
    model_path: str,
) -> bool:
    """
    Load ``raw`` state-dict into ``model``, handling four possible key formats:

    1. Plain weights  —  ``conv.weight``
    2. Old weight_norm  —  ``conv.weight_g`` / ``conv.weight_v``
    3. New parametrizations  —  ``conv.parametrizations.weight.original0/1``
    4. Any of the above prefixed with ``generator.`` / ``module.`` / ``model.``
    """

    # --- Strip outer wrapper prefixes ---
    def _strip_prefixes(k: str) -> str:
        for pfx in ("generator.", "module.", "model."):
            if k.startswith(pfx):
                k = k[len(pfx):]
                break
        return k

    stripped = {_strip_prefixes(k): v for k, v in raw.items()}

    # --- Detect and merge weight_norm parameters ---
    def _compute_wn_weight(wg: torch.Tensor, wv: torch.Tensor) -> torch.Tensor:
        """Compute weight = wg * wv / ‖wv‖  (weight_norm formula)."""
        norm = wv.reshape(wv.shape[0], -1).norm(dim=1)
        # broadcast back to weight shape
        for _ in range(wv.dim() - 1):
            norm = norm.unsqueeze(-1)
        return wg * wv / (norm + 1e-8)

    merged: dict[str, torch.Tensor] = {}
    skip: set[str] = set()

    # parametrizations format: .parametrizations.weight.original0 / .original1
    for k in list(stripped):
        if ".parametrizations.weight.original0" in k and k not in skip:
            base = k.replace(".parametrizations.weight.original0", "")
            k1 = k
            k2 = k.replace(".original0", ".original1")
            if k2 in stripped:
                # Applio saves weight_g as original0, weight_v as original1
                # (confirmed from torch.nn.utils.parametrizations source)
                wg = stripped[k1]
                wv = stripped[k2]
                merged[base + ".weight"] = _compute_wn_weight(wg, wv)
                skip.update([k1, k2])

    # Old weight_norm format: .weight_g / .weight_v
    for k in list(stripped):
        if k.endswith(".weight_g") and k not in skip:
            base = k[: -len(".weight_g")]
            kv = base + ".weight_v"
            if kv in stripped:
                merged[base + ".weight"] = _compute_wn_weight(stripped[k], stripped[kv])
                skip.update([k, kv])

    # Copy everything else unchanged
    for k, v in stripped.items():
        if k not in skip:
            merged[k] = v

    # --- Load ---
    missing, unexpected = model.load_state_dict(merged, strict=False)

    n_loaded = len(merged) - len(unexpected)
    n_model = sum(1 for _ in model.state_dict())
    missing_ratio = len(missing) / max(n_model, 1)

    if missing:
        logger.warning(
            f"RefineGAN checkpoint '{os.path.basename(model_path)}': "
            f"{len(missing)} missing keys (first 5: {missing[:5]})"
        )
    if unexpected:
        logger.warning(
            f"RefineGAN checkpoint: "
            f"{len(unexpected)} unexpected keys (first 5: {unexpected[:5]})"
        )

    if missing_ratio > 0.5:
        logger.error(
            f"RefineGAN: over 50 % of model keys missing ({len(missing)}/{n_model}). "
            "The checkpoint architecture likely does not match upsample_rates=(10,8,2,2). "
            "Try setting upsample_rates=(8,8,2,2) in RefineGANVocoder if loading fails."
        )
        return False

    logger.info(
        f"RefineGAN weights loaded from '{model_path}' "
        f"({n_loaded}/{n_model + len(unexpected)} keys matched, "
        f"missing={len(missing)}, unexpected={len(unexpected)})"
    )
    return True


# ---------------------------------------------------------------------------
# Public Vocoder wrapper
# ---------------------------------------------------------------------------

class RefineGANVocoder(Vocoder):
    """
    Standalone RefineGAN vocoder with GPU-accelerated output resampling.

    Wraps ``_RefineGANGenerator``, handles checkpoint loading, weight-norm
    stripping, half-precision, and resampling from the vocoder's native 32 kHz
    to whatever sample rate the loaded RVC model uses.

    Because the generator's ``upp = 320`` at 32 kHz gives 10 ms / frame,
    resampling to ``model_sr`` Hz always yields exactly ``model_sr / 100``
    samples per frame — identical to the embedded NSF-HiFiGAN decoder.
    """

    VOCODER_SR: int = 32000

    def __init__(self, model_path: str, target_sr: int = 40000):
        super().__init__()
        self._device_manager = DeviceManager.get_instance()
        self.device = self._device_manager.device
        self.is_half = self._device_manager.use_fp16()
        self.target_sr = target_sr
        self.is_loaded = False

        logger.info(
            f"Initialising RefineGAN vocoder "
            f"(native {self.VOCODER_SR} Hz → target {target_sr} Hz, "
            f"device={self.device}, fp16={self.is_half})"
        )

        self.model = _RefineGANGenerator().to(self.device)

        if os.path.exists(model_path):
            self._load(model_path)
        else:
            logger.warning(
                f"RefineGAN weight file not found: '{model_path}'. "
                "Download it from the Downloader panel."
            )

        # Strip weight_norm for faster inference (no-op if already plain weights)
        if self.is_loaded:
            try:
                self.model.remove_weight_norm()
                logger.debug("RefineGAN: weight_norm stripped.")
            except Exception:
                pass

        self.model.eval()
        if self.is_half:
            self.model = self.model.half()

        self._build_resampler()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load(self, model_path: str) -> None:
        try:
            cpt = torch.load(model_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            logger.error(f"RefineGAN: cannot read checkpoint '{model_path}': {exc}")
            return

        # Unwrap outer dict
        if isinstance(cpt, dict):
            raw: Optional[dict] = None
            for key in ("generator", "model", "g", "weight", "state_dict"):
                if key in cpt and isinstance(cpt[key], dict):
                    raw = cpt[key]
                    break
            if raw is None:
                # Treat the whole dict as a flat state_dict
                raw = {k: v for k, v in cpt.items() if isinstance(v, torch.Tensor)}
        else:
            logger.error(
                f"RefineGAN: unsupported checkpoint type {type(cpt)!r}. "
                "Expected a dict containing model weights."
            )
            return

        if not raw:
            logger.error("RefineGAN: checkpoint contains no weight tensors.")
            return

        self.is_loaded = _load_state_dict_compatible(self.model, raw, model_path)

    def _build_resampler(self) -> None:
        """(Re)build GPU resampler; no-op if native SR == target SR."""
        if self.target_sr != self.VOCODER_SR:
            self.resampler_out: Optional[tat.Resample] = tat.Resample(
                orig_freq=self.VOCODER_SR,
                new_freq=self.target_sr,
                dtype=torch.float32,
            ).to(self.device)
        else:
            self.resampler_out = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def update_target_sr(self, target_sr: int) -> None:
        """
        Update the output sample rate without reloading model weights.
        Called by VocoderManager when the active RVC model changes.
        """
        if target_sr != self.target_sr:
            self.target_sr = target_sr
            self._build_resampler()
            logger.info(f"RefineGAN: output resampler updated → {target_sr} Hz")

    def infer(
        self,
        z: torch.Tensor,
        pitchf: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Synthesise audio from latent features and pitch.

        Args:
            z      : (B, 192, T)       VITS latent representation
            pitchf : (B, T)            F0 contour in Hz
            g      : (B, 256, 1)       speaker embedding (optional)

        Returns:
            (B, 1, T × embedded_upp)   waveform tensor at target_sr
        """
        dtype = torch.float16 if self.is_half else torch.float32

        z = z.to(device=self.device, dtype=dtype)
        pitchf = pitchf.to(device=self.device, dtype=dtype)
        if g is not None:
            g = g.to(device=self.device, dtype=dtype)

        with torch.no_grad():
            audio = self.model(z, pitchf, g)       # (B, 1, T*320) @ 32 kHz

            if self.resampler_out is not None:
                # Resample 32 k → target_sr (GPU-accelerated, pitch-preserving)
                # Output length = T * 320 * target_sr / 32000 = T * (target_sr/100)
                #               = T * embedded_upp  ✓
                audio = self.resampler_out(audio.to(torch.float32)).to(dtype)

        return audio
