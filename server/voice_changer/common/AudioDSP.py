import json
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import signal as scipy_signal
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalPolyphaseResampler(nn.Module):
    """
    Stateful causal polyphase Kaiser-windowed sinc resampler for real-time streaming.

    Maintains an exact sample history buffer and fractional output phase across chunks
    without right-side zero padding, guaranteeing:
      - Zero chunk-boundary attenuation or clicks
      - Zero cumulative sample-count drift (exact rational rate P/Q)
      - Exact unity DC gain on every polyphase branch
    """

    def __init__(
        self,
        orig_freq: int,
        new_freq: int,
        lowpass_filter_width: int = 8,
        rolloff: float = 0.945,
        beta: float = 10.0,
    ):
        super().__init__()
        self.orig_freq = int(orig_freq)
        self.new_freq = int(new_freq)

        g = math.gcd(self.orig_freq, self.new_freq)
        self.p = self.new_freq // g
        self.q = self.orig_freq // g
        self.is_bypass = self.p == self.q

        if self.is_bypass:
            self.half_width = 0
            self.kernel_len = 1
            self.register_buffer("kernels", torch.ones(1, 1, dtype=torch.float32))
            self.register_buffer("history", torch.empty(0, dtype=torch.float32), persistent=False)
        else:
            base_freq = min(self.p, self.q) * rolloff
            self.half_width = max(1, math.ceil(lowpass_filter_width * self.q / base_freq))
            self.kernel_len = 2 * self.half_width + 1

            idx = torch.arange(self.kernel_len, dtype=torch.float64)
            i0_beta = torch.special.i0(torch.tensor(beta, dtype=torch.float64))
            scale = base_freq / self.q

            kernels = []
            for ph in range(self.p):
                # Distance from sample tap j to target continuous position (half_width + ph/p)
                delta = (self.half_width + ph / self.p) - idx
                t = delta * scale
                t_clamped = t.clamp(-lowpass_filter_width, lowpass_filter_width)

                sinc_val = torch.where(
                    t == 0.0,
                    torch.ones_like(t),
                    torch.sin(math.pi * t) / (math.pi * t),
                )
                kaiser_arg = beta * torch.sqrt(
                    torch.clamp(1.0 - (t_clamped / lowpass_filter_width) ** 2, min=0.0)
                )
                window = torch.special.i0(kaiser_arg) / i0_beta
                # Zero taps outside support window
                mask = (t.abs() <= lowpass_filter_width).to(torch.float64)
                k_ph = sinc_val * window * mask

                # Normalize each polyphase branch for exact unity DC gain
                k_sum = k_ph.sum()
                if k_sum.abs() > 1e-12:
                    k_ph = k_ph / k_sum
                kernels.append(k_ph.float())

            self.register_buffer("kernels", torch.stack(kernels, dim=0).contiguous())
            self.register_buffer(
                "history",
                torch.zeros(2 * self.half_width, dtype=torch.float32),
                persistent=False,
            )

        self._next_pos_num: int = 0

    def reset(self) -> None:
        self._next_pos_num = 0
        if self.history.numel() > 0:
            self.history.zero_()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x.float()
        if self.is_bypass:
            return x.float() if x.dtype != torch.float32 else x

        x_f32 = x.reshape(-1).to(dtype=torch.float32)
        if self.kernels.device != x_f32.device:
            self.to(x_f32.device)

        n = x_f32.shape[0]
        hist_len = 2 * self.half_width
        buf = torch.cat([self.history, x_f32], dim=0)
        total_num = n * self.p

        if total_num <= self._next_pos_num:
            self._next_pos_num -= total_num
            self.history = buf[-hist_len:].detach().clone()
            return torch.empty(0, dtype=torch.float32, device=x_f32.device)

        m = (total_num - self._next_pos_num + self.q - 1) // self.q

        if self.p == 1:
            # Fast decimation path (e.g. 48k -> 16k, 32k -> 16k, 96k -> 16k)
            start = self._next_pos_num
            end = start + (m - 1) * self.q + self.kernel_len
            conv_in = buf[start:end].view(1, 1, -1)
            weight = self.kernels.view(1, 1, self.kernel_len)
            out = F.conv1d(conv_in, weight, stride=self.q).view(-1)
        else:
            # General rational resampling path (e.g. 44.1k -> 16k)
            k_idx = torch.arange(m, dtype=torch.int64, device=x_f32.device)
            nums = self._next_pos_num + k_idx * self.q
            idx_k = torch.div(nums, self.p, rounding_mode="floor")
            ph_k = nums % self.p
            windows = buf.unfold(0, self.kernel_len, 1)[idx_k]
            out = (windows * self.kernels[ph_k]).sum(dim=-1)

        self._next_pos_num = self._next_pos_num + m * self.q - total_num
        self.history = buf[-hist_len:].detach().clone()
        return out


def compute_minimum_phase_fir(
    log_H_normalized: torch.Tensor,
    gain_scale: float,
    num_taps: int,
    causal_mask_cache: Optional[Dict[tuple, torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Computes a flipped causal minimum-phase FIR kernel (for F.conv1d) directly on GPU
    from a normalized one-sided log-magnitude spectrum.
    """
    device = log_H_normalized.device
    log_H = gain_scale * log_H_normalized.to(dtype=torch.float32)

    # Construct conjugate-symmetric full-spectrum log-magnitude
    log_H_full = torch.cat([log_H, torch.flip(log_H[1:-1], dims=[0])])
    n_fft = log_H_full.shape[0]

    cepstrum = torch.fft.ifft(log_H_full).real

    cache_key = (n_fft, str(device))
    if causal_mask_cache is not None and cache_key in causal_mask_cache:
        causal_mask = causal_mask_cache[cache_key]
    else:
        causal_mask = torch.zeros(n_fft, device=device, dtype=torch.float32)
        causal_mask[0] = 1.0
        causal_mask[n_fft // 2] = 1.0
        causal_mask[1 : n_fft // 2] = 2.0
        if causal_mask_cache is not None:
            causal_mask_cache[cache_key] = causal_mask

    complex_log_spectrum = torch.fft.fft(cepstrum * causal_mask)
    H_min_phase = torch.exp(complex_log_spectrum[: log_H.shape[0]])

    h_ir = torch.fft.irfft(H_min_phase, n=n_fft)[:num_taps]

    # Smooth half-Hann tail taper to avoid truncation ringing
    taper_len = min(32, num_taps // 4)
    if taper_len > 1:
        taper = 0.5 * (
            1.0
            + torch.cos(
                torch.linspace(0.0, math.pi, taper_len, device=device, dtype=torch.float32)
            )
        )
        h_ir = h_ir.clone()
        h_ir[-taper_len:] *= taper

    # Flip for PyTorch F.conv1d cross-correlation semantics
    return torch.flip(h_ir, dims=[0]).contiguous()


def _extract_voiced_f0_yin_16k(
    y_16k: np.ndarray,
    sr_16k: int = 16000,
    frame_length: int = 1024,
    hop_length: int = 160,
    fmin: float = 55.0,
    fmax: float = 800.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized FFT-based YIN pitch & aperiodicity estimator at 16 kHz.
    Returns (f0_hz, aperiodicity, rms) per frame.
    """
    x = np.asarray(y_16k, dtype=np.float32).reshape(-1)
    pad = frame_length // 2
    if x.size == 0:
        return (
            np.zeros(1, dtype=np.float32),
            np.ones(1, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
        )
    if x.size < frame_length:
        x = np.pad(x, (0, frame_length - x.size), mode="constant")
    x_padded = np.pad(x, (pad, pad), mode="constant")
    frames = np.lib.stride_tricks.sliding_window_view(x_padded, frame_length)[::hop_length]
    n_frames = frames.shape[0]

    w_int = frame_length // 2
    n_fft_yin = 2 * frame_length
    tau_min = max(2, int(math.floor(sr_16k / float(fmax))))
    tau_max = min(w_int - 2, int(math.ceil(sr_16k / float(fmin))))

    f0_out = np.zeros(n_frames, dtype=np.float32)
    aper_out = np.ones(n_frames, dtype=np.float32)
    rms_out = np.zeros(n_frames, dtype=np.float32)

    batch_size = 512
    for start in range(0, n_frames, batch_size):
        end = min(start + batch_size, n_frames)
        f = frames[start:end].astype(np.float64, copy=False)
        bsz = f.shape[0]
        rms_b = np.sqrt(np.mean(f * f, axis=1))
        rms_out[start:end] = rms_b.astype(np.float32)

        x0 = f[:, :w_int]
        X0 = np.fft.rfft(x0, n=n_fft_yin, axis=1)
        X1 = np.fft.rfft(f, n=n_fft_yin, axis=1)
        acf = np.fft.irfft(np.conj(X0) * X1, n=n_fft_yin, axis=1)[:, :w_int]

        sq = f * f
        cs = np.concatenate(
            [np.zeros((bsz, 1), dtype=np.float64), np.cumsum(sq, axis=1)], axis=1
        )
        e0 = cs[:, w_int : w_int + 1]
        e_tau = cs[:, w_int:frame_length] - cs[:, :w_int]
        d = np.maximum(e0 + e_tau - 2.0 * acf, 0.0)

        taus = np.arange(1, w_int, dtype=np.float64)[np.newaxis, :]
        d_pos = d[:, 1:]
        cum_d = np.cumsum(d_pos, axis=1)
        cmndf = np.ones((bsz, w_int), dtype=np.float64)
        cmndf[:, 1:] = np.where(cum_d > 1e-12, d_pos * taus / np.maximum(cum_d, 1e-12), 1.0)

        for i in range(bsz):
            if rms_b[i] < 1e-5 or e0[i, 0] < 1e-10:
                continue
            sub = cmndf[i, tau_min : tau_max + 1]
            below = np.flatnonzero(sub < 0.20)
            if below.size > 0:
                t = tau_min + int(below[0])
                while t + 1 <= tau_max and cmndf[i, t + 1] < cmndf[i, t]:
                    t += 1
            else:
                t = tau_min + int(np.argmin(sub))

            a, b, c = cmndf[i, t - 1], cmndf[i, t], cmndf[i, t + 1]
            denom = a - 2.0 * b + c
            if denom > 1e-12:
                shift = float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))
                tau_ref = float(t) + shift
            else:
                tau_ref = float(t)

            if tau_ref > 0.0:
                f0_out[start + i] = float(sr_16k / tau_ref)
                aper_out[start + i] = float(np.clip(b, 0.0, 1.0))

    return f0_out, aper_out, rms_out


def _to_mono_float32(y: np.ndarray) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float32)
    if arr.ndim == 0 or arr.size == 0:
        return np.empty(0, dtype=np.float32)
    if arr.ndim > 1:
        # Collapse whichever axis looks like channels (smaller dimension)
        ch_axis = 0 if arr.shape[0] <= arr.shape[-1] else -1
        arr = np.mean(arr, axis=ch_axis, dtype=np.float32)
    return arr.reshape(-1)


def _select_highest_energy_segment(y: np.ndarray, sr: int, max_duration_sec: float) -> np.ndarray:
    max_len = int(round(max_duration_sec * sr))
    if max_len <= 0 or y.shape[0] <= max_len:
        return y
    step = max(1, sr // 4)
    sq = y.astype(np.float64, copy=False) ** 2
    cs = np.concatenate([[0.0], np.cumsum(sq)])
    starts = np.arange(0, y.shape[0] - max_len + 1, step, dtype=np.int64)
    energies = cs[starts + max_len] - cs[starts]
    best_start = int(starts[int(np.argmax(energies))])
    return y[best_start : best_start + max_len]


def analyze_voice_acoustics(
    y: np.ndarray,
    sr: int,
    max_duration_sec: float = 30.0,
) -> Dict[str, Any]:
    """
    Extract pitch (F0), SR-invariant formant centroid, and physical-time cepstral
    spectral envelope over strictly voiced frames.
    """
    sr_int = max(1, int(sr))
    y_mono = _to_mono_float32(y)
    if y_mono.size > 0:
        y_mono = y_mono - float(np.mean(y_mono))

    n_fft = 1024 if sr_int < 28000 else (2048 if sr_int < 64000 else 4096)
    flat_env = np.ones(n_fft // 2 + 1, dtype=np.float32)

    if y_mono.size < 512 or float(np.max(np.abs(y_mono))) < 1e-6:
        return {
            "f0": 0.0,
            "centroid": 0.0,
            "envelope": flat_env,
            "sr": sr_int,
            "voiced_ratio": 0.0,
        }

    y_mono = _select_highest_energy_segment(y_mono, sr_int, max_duration_sec)

    # Resample to 16 kHz solely for YIN F0 / periodicity analysis
    if sr_int == 16000:
        y_16k = y_mono
    else:
        g = math.gcd(16000, sr_int)
        up, down = 16000 // g, sr_int // g
        if max(up, down) <= 2000:
            y_16k = scipy_signal.resample_poly(y_mono, up, down).astype(
                np.float32, copy=False
            )
        else:
            target_len = max(1, int(round(y_mono.shape[0] * 16000.0 / float(sr_int))))
            y_16k = scipy_signal.resample(y_mono, target_len).astype(
                np.float32, copy=False
            )

    yin_hop = 160
    f0_yin, aper_yin, _ = _extract_voiced_f0_yin_16k(
        y_16k, sr_16k=16000, frame_length=1024, hop_length=yin_hop, fmin=55.0, fmax=800.0
    )

    # STFT on native sample rate with physically consistent window size
    hop_length = n_fft // 4
    pad = n_fft // 2
    pad_mode = "reflect" if y_mono.size > pad else "constant"
    y_pad = np.pad(y_mono, (pad, pad), mode=pad_mode)
    frames = np.lib.stride_tricks.sliding_window_view(y_pad, n_fft)[::hop_length]
    win = np.hanning(n_fft).astype(np.float32)
    win_rms = float(np.sqrt(np.mean(win * win))) or 1.0
    rms_stft = np.sqrt(np.mean((frames * win) ** 2, axis=1)) / win_rms

    spec_mag = np.abs(np.fft.rfft(frames * win, axis=-1)).T  # (n_bins, n_frames)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sr_int))
    n_frames = spec_mag.shape[1]

    # Map 16 kHz YIN estimates to native STFT frames by physical timestamp
    t_stft = np.arange(n_frames, dtype=np.float64) * (hop_length / float(sr_int))
    yin_dt = yin_hop / 16000.0
    idx_map = np.clip(np.round(t_stft / yin_dt).astype(np.int64), 0, f0_yin.shape[0] - 1)
    f0_stft = f0_yin[idx_map]
    aper_stft = aper_yin[idx_map]

    rms_p95 = float(np.percentile(rms_stft, 95)) if rms_stft.size > 0 else 0.0
    rms_thresh = max(1e-4, 0.05 * rms_p95)
    energy_mask = rms_stft > rms_thresh

    low_mid_mask = (freqs >= 80.0) & (freqs <= 3800.0)
    high_fric_mask = (freqs > 3800.0) & (freqs <= min(8000.0, sr_int / 2.0))
    if np.any(high_fric_mask) and np.any(low_mid_mask):
        e_lm = np.sum(spec_mag[low_mid_mask, :] ** 2, axis=0) + 1e-12
        e_hf = np.sum(spec_mag[high_fric_mask, :] ** 2, axis=0) + 1e-12
        tonal_band_mask = e_lm > e_hf
    else:
        tonal_band_mask = np.ones(n_frames, dtype=bool)

    voiced_mask = (
        energy_mask
        & (aper_stft < 0.25)
        & (f0_stft >= 55.0)
        & (f0_stft <= 800.0)
        & tonal_band_mask
    )

    active_count = int(np.sum(energy_mask))
    voiced_count = int(np.sum(voiced_mask))
    voiced_ratio = float(voiced_count) / float(active_count) if active_count > 0 else 0.0

    if voiced_count >= 3:
        f0_median = float(np.median(f0_stft[voiced_mask]))
        use_mask = voiced_mask
    else:
        fallback_f0 = (f0_stft >= 55.0) & (f0_stft <= 800.0) & energy_mask
        f0_median = float(np.median(f0_stft[fallback_f0])) if np.any(fallback_f0) else 0.0
        use_mask = energy_mask if active_count > 0 else np.ones(n_frames, dtype=bool)

    # Physical-time cepstral liftering (symmetric raised cosine)
    tau_c = float(np.clip(0.65 / f0_median, 0.0012, 0.0022)) if f0_median > 0.0 else 0.0018
    q_cut = max(8, min(n_fft // 4, int(round(tau_c * sr_int))))
    q_pass = max(2, int(round(0.7 * q_cut)))

    lifter = np.zeros(n_fft, dtype=np.float32)
    lifter[: q_pass + 1] = 1.0
    if q_cut > q_pass:
        idx_t = np.arange(q_pass + 1, q_cut, dtype=np.float32)
        taper = 0.5 * (1.0 + np.cos(math.pi * (idx_t - q_pass) / float(q_cut - q_pass)))
        lifter[q_pass + 1 : q_cut] = taper
    lifter[n_fft - q_cut + 1 :] = lifter[1:q_cut][::-1]

    log_S = np.log(spec_mag[:, use_mask] + 1e-7)
    cepstrum = np.fft.irfft(log_S, n=n_fft, axis=0)
    log_env_frames = np.fft.rfft(cepstrum * lifter[:, np.newaxis], axis=0).real

    core_mask = (freqs >= 150.0) & (freqs <= min(4500.0, 0.85 * sr_int / 2.0))
    if not np.any(core_mask):
        core_mask = slice(None)

    frame_means = np.mean(log_env_frames[core_mask, :], axis=0, keepdims=True)
    log_env_norm = log_env_frames - frame_means

    weights = rms_stft[use_mask].astype(np.float64)
    w_sum = float(np.sum(weights))
    if w_sum > 1e-12:
        mean_log_env = np.sum(log_env_norm * weights[np.newaxis, :], axis=1) / w_sum
    else:
        mean_log_env = np.mean(log_env_norm, axis=1)

    mean_log_env = mean_log_env - np.mean(mean_log_env[core_mask])
    envelope = np.exp(np.clip(mean_log_env, -10.0, 10.0)).astype(np.float32)

    # SR-independent formant centroid in [200, min(4000, 0.9 * Nyquist)] Hz
    cent_mask = (freqs >= 200.0) & (freqs <= min(4000.0, 0.9 * sr_int / 2.0))
    if np.any(cent_mask):
        cent_sum = float(np.sum(envelope[cent_mask]))
        centroid = (
            float(np.sum(envelope[cent_mask] * freqs[cent_mask]) / cent_sum)
            if cent_sum > 1e-12
            else 0.0
        )
    else:
        centroid = 0.0

    return {
        "f0": f0_median,
        "centroid": centroid,
        "envelope": envelope,
        "sr": sr_int,
        "voiced_ratio": voiced_ratio,
    }


def _parse_envelope_to_log(env_raw: Any) -> Optional[np.ndarray]:
    """Parse 1D envelope array (or legacy vowel-dict) into log-magnitude domain."""
    if env_raw is None:
        return None
    if isinstance(env_raw, str):
        env_raw = json.loads(env_raw)

    if isinstance(env_raw, dict):
        logs = []
        for k in ("close", "open", "front"):
            if k in env_raw:
                parsed_k = _parse_envelope_to_log(env_raw[k])
                if parsed_k is not None:
                    logs.append(parsed_k)
        if not logs:
            return None
        min_len = min(arr.shape[0] for arr in logs)
        return np.mean([arr[:min_len] for arr in logs], axis=0)

    arr = np.asarray(env_raw, dtype=np.float64)
    if arr.ndim != 1 or arr.size < 2:
        return None
    if np.any(arr <= 0.0):
        return arr
    return np.log(np.maximum(arr, 1e-8))


def compute_residual_match_eq_curve(
    target_env: Any,
    target_sr: int,
    base_env: Any,
    base_sr: int,
    post_sr: int,
    n_fft_design: int = 4096,
    max_db: float = 6.0,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Compute 1/3-octave smoothed, band-limited residual Match EQ log-response
    delta_L(f) = log E_target(f) - log E_base(f) on post_sr grid.

    Returns (delta_log_masked, log_base_centered, rms_residual_db, max_correction_db).
    """
    log_tgt_src = _parse_envelope_to_log(target_env)
    log_base_src = _parse_envelope_to_log(base_env)
    if log_tgt_src is None or log_base_src is None:
        raise ValueError("Invalid target or baseline spectral envelope.")

    f_tgt = np.linspace(0.0, float(target_sr) / 2.0, log_tgt_src.shape[0])
    f_base = np.linspace(0.0, float(base_sr) / 2.0, log_base_src.shape[0])
    f_out = np.linspace(0.0, float(post_sr) / 2.0, n_fft_design // 2 + 1)

    log_tgt = np.interp(f_out, f_tgt, log_tgt_src)
    log_base = np.interp(f_out, f_base, log_base_src)

    nyq_common = min(float(post_sr) / 2.0, float(target_sr) / 2.0, float(base_sr) / 2.0)
    f_high_pass = min(7500.0, 0.80 * nyq_common)
    f_high_stop = min(9500.0, 0.92 * nyq_common)

    core_mask = (f_out >= 150.0) & (f_out <= min(4500.0, max(300.0, f_high_pass)))
    if not np.any(core_mask):
        core_mask = f_out <= nyq_common

    # Preserve genuine spectral tilt by subtracting only the core-band scalar mean
    log_tgt_c = log_tgt - float(np.mean(log_tgt[core_mask]))
    log_base_c = log_base - float(np.mean(log_base[core_mask]))
    delta_log = log_tgt_c - log_base_c

    # Log-frequency ~1/3-octave Gaussian smoothing
    f_lo = 60.0
    f_hi = max(120.0, nyq_common)
    n_log = 256
    f_log = np.geomspace(f_lo, f_hi, n_log)
    delta_on_log = np.interp(f_log, f_out, delta_log)

    octaves_span = math.log2(f_hi / f_lo)
    d_oct = octaves_span / float(n_log - 1)
    sigma_bins = 0.25 / max(d_oct, 1e-6)
    half_k = max(1, int(math.ceil(3.0 * sigma_bins)))
    kx = np.arange(-half_k, half_k + 1, dtype=np.float64)
    gauss_k = np.exp(-0.5 * (kx / max(sigma_bins, 1e-6)) ** 2)
    gauss_k /= np.sum(gauss_k)

    padded = np.pad(delta_on_log, (half_k, half_k), mode="edge")
    smoothed_on_log = np.convolve(padded, gauss_k, mode="valid")
    delta_smooth = np.interp(
        f_out, f_log, smoothed_on_log, left=smoothed_on_log[0], right=smoothed_on_log[-1]
    )

    max_ln = float(max_db) * math.log(10.0) / 20.0
    delta_capped = np.clip(delta_smooth, -max_ln, max_ln)

    # Raised-cosine band limits [60..100 Hz] -> [f_high_pass..f_high_stop]
    band_mask = np.ones_like(f_out)
    band_mask[f_out <= 60.0] = 0.0
    low_trans = (f_out > 60.0) & (f_out < 100.0)
    if np.any(low_trans):
        band_mask[low_trans] = 0.5 * (
            1.0 - np.cos(math.pi * (f_out[low_trans] - 60.0) / 40.0)
        )
    high_trans = (f_out > f_high_pass) & (f_out < f_high_stop)
    if np.any(high_trans):
        band_mask[high_trans] = 0.5 * (
            1.0
            + np.cos(
                math.pi
                * (f_out[high_trans] - f_high_pass)
                / max(f_high_stop - f_high_pass, 1.0)
            )
        )
    band_mask[f_out >= f_high_stop] = 0.0

    delta_masked = (delta_capped * band_mask).astype(np.float32)

    ln_to_db = 20.0 / math.log(10.0)
    rms_residual_db = float(np.sqrt(np.mean(delta_masked[core_mask] ** 2)) * ln_to_db)
    max_correction_db = float(np.max(np.abs(delta_masked)) * ln_to_db)

    return delta_masked, log_base_c.astype(np.float32), rms_residual_db, max_correction_db


class MinimumPhaseFormantFilter(nn.Module):
    """
    Stateful post-SOLA residual minimum-phase Match EQ filter.
    - apply_pre is an explicit pass-through no-op so HuBERT/ContentVec and F0 extraction
      receive unmodified dry input speech.
    - apply_post applies a static (crossfade-on-change) minimum-phase FIR matching
      the residual spectral difference between target reference and RVC model output,
      with speech-weighted RMS energy conservation.
    """

    def __init__(
        self,
        num_taps_post: int = 512,
        n_fft_design: int = 4096,
        max_db: float = 6.0,
    ):
        super().__init__()
        self.num_taps_post = int(num_taps_post)
        self.n_fft_design = int(n_fft_design)
        self.max_db = float(max_db)

        self.register_buffer(
            "post_history",
            torch.zeros(self.num_taps_post - 1, dtype=torch.float32),
            persistent=False,
        )

        self._causal_mask_cache: Dict[tuple, torch.Tensor] = {}
        self._cache_key: Any = None
        self._strength: float = 0.35

        self._h_post_conv: Optional[torch.Tensor] = None
        self._h_post_xfade_from: Optional[torch.Tensor] = None

    def reset(self) -> None:
        if self.post_history.numel() > 0:
            self.post_history.zero_()
        self._h_post_xfade_from = None

    def _clear_filters(self) -> None:
        self._h_post_conv = None
        self._h_post_xfade_from = None
        if self.post_history.numel() > 0:
            self.post_history.zero_()

    def update_config(
        self,
        active: bool,
        target_env: str,
        target_sr: int,
        input_env: str,
        input_sr: int,
        strength: float,
        pre_sr: int,
        post_sr: int,
        device: torch.device,
    ) -> None:
        del pre_sr  # Pre-filter removed: Match EQ operates exclusively post-RVC
        cache_key = (
            bool(active),
            target_env,
            int(target_sr),
            input_env,
            int(input_sr),
            float(strength),
            int(post_sr),
            str(device),
        )
        if cache_key == self._cache_key:
            return

        self._cache_key = cache_key
        self._strength = float(strength)

        if self.post_history.device != device:
            self.to(device)

        if not (
            active
            and target_env
            and input_env
            and target_sr > 0
            and input_sr > 0
            and post_sr > 0
            and self._strength > 1e-4
        ):
            self._clear_filters()
            return

        try:
            parsed_tgt = json.loads(target_env) if isinstance(target_env, str) else target_env
            parsed_base = json.loads(input_env) if isinstance(input_env, str) else input_env

            delta_masked, log_base_c, _, _ = compute_residual_match_eq_curve(
                parsed_tgt,
                int(target_sr),
                parsed_base,
                int(input_sr),
                int(post_sr),
                n_fft_design=self.n_fft_design,
                max_db=self.max_db,
            )

            delta_t = torch.tensor(delta_masked, dtype=torch.float32, device=device)
            h_new = compute_minimum_phase_fir(
                delta_t, self._strength, self.num_taps_post, self._causal_mask_cache
            )

            # Speech-weighted RMS energy conservation across full speech passband
            f_out = np.linspace(0.0, float(post_sr) / 2.0, self.n_fft_design // 2 + 1)
            nyq_common = min(float(post_sr) / 2.0, float(target_sr) / 2.0, float(input_sr) / 2.0)
            rms_mask = (f_out >= 100.0) & (f_out <= min(7500.0, 0.80 * nyq_common))
            if not np.any(rms_mask):
                rms_mask = f_out <= nyq_common
            p_base = np.exp(np.clip(2.0 * log_base_c[rms_mask], -20.0, 20.0))
            p_gain = np.exp(2.0 * self._strength * delta_masked[rms_mask])
            denom_p = max(float(np.sum(p_base)), 1e-12)
            rms_ratio = math.sqrt(float(np.sum(p_base * p_gain)) / denom_p)
            h_new = h_new / float(np.clip(rms_ratio, 0.5, 2.0))

            self._h_post_xfade_from = (
                self._h_post_conv.detach() if self._h_post_conv is not None else None
            )
            self._h_post_conv = h_new
        except Exception:
            self._clear_filters()

    @torch.no_grad()
    def apply_pre(self, x: torch.Tensor) -> torch.Tensor:
        """Pass-through no-op: input audio to HuBERT and F0 extractor is never altered."""
        return x

    @torch.no_grad()
    def apply_post(self, x: torch.Tensor) -> torch.Tensor:
        if self._h_post_conv is None or x.numel() == 0:
            return x

        orig_dtype = x.dtype
        x_f32 = x.reshape(-1).to(dtype=torch.float32)
        if self.post_history.device != x_f32.device:
            self.to(x_f32.device)

        n = x_f32.shape[0]
        hist_len = self.num_taps_post - 1
        buf = torch.cat([self.post_history, x_f32], dim=0)

        y = F.conv1d(buf.view(1, 1, -1), self._h_post_conv.view(1, 1, -1)).view(-1)

        if self._h_post_xfade_from is not None:
            xfade_len = min(64, n)
            if xfade_len > 1:
                prev_slice = buf[: hist_len + xfade_len]
                y_prev = F.conv1d(
                    prev_slice.view(1, 1, -1), self._h_post_xfade_from.view(1, 1, -1)
                ).view(-1)
                alpha = torch.linspace(
                    0.0, 1.0, xfade_len, device=x_f32.device, dtype=torch.float32
                )
                y[:xfade_len] = y[:xfade_len] * alpha + y_prev[:xfade_len] * (1.0 - alpha)
            self._h_post_xfade_from = None

        self.post_history = buf[-hist_len:].detach().clone()
        return y.to(dtype=orig_dtype)


MinimumPhaseMatchEQ = MinimumPhaseFormantFilter


class StatefulPeakLimiter:
    """
    Stateful sub-block envelope peak limiter for real-time audio streams.

    Leaves signals below `threshold` completely linear (unity gain) and tracks
    peak excursions with fast attack / smooth release, avoiding the steady-state
    harmonic distortion of stateless waveshapers on sustained loud audio.
    """

    def __init__(
        self,
        threshold: float = 0.85,
        ceiling: float = 0.98,
        attack_ms: float = 1.0,
        release_ms: float = 80.0,
    ):
        self.threshold = float(threshold)
        self.ceiling = float(ceiling)
        self.attack_ms = max(0.1, float(attack_ms))
        self.release_ms = max(1.0, float(release_ms))
        self._env: float = 0.0
        self._prev_gain: float = 1.0

    def reset(self) -> None:
        self._env = 0.0
        self._prev_gain = 1.0

    def _gain_for_env(self, env: float) -> float:
        if env <= self.threshold or env <= 1e-8:
            return 1.0
        headroom = max(self.ceiling - self.threshold, 1e-6)
        target_peak = self.threshold + headroom * math.tanh((env - self.threshold) / headroom)
        return min(1.0, target_peak / env)

    def process(self, audio: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
        if audio is None or audio.size == 0:
            return audio

        n = audio.shape[0]
        sr = max(1, int(sample_rate))
        sub_len = max(1, int(round(sr * 0.0005)))
        abs_x = np.abs(audio)

        full_subs = n // sub_len
        if full_subs > 0 and full_subs * sub_len == n:
            peaks = abs_x.reshape(full_subs, sub_len).max(axis=1)
        elif full_subs > 0:
            head_peaks = abs_x[: full_subs * sub_len].reshape(full_subs, sub_len).max(axis=1)
            tail_peak = np.array([abs_x[full_subs * sub_len :].max()], dtype=np.float32)
            peaks = np.concatenate([head_peaks, tail_peak])
        else:
            peaks = np.array([abs_x.max()], dtype=np.float32)

        dt_sub = sub_len / float(sr)
        g_attack = math.exp(-dt_sub / (self.attack_ms / 1000.0))
        g_release = math.exp(-dt_sub / (self.release_ms / 1000.0))

        n_subs = peaks.shape[0]
        gain_knots = np.empty(n_subs + 1, dtype=np.float32)
        gain_knots[0] = self._prev_gain

        env = self._env
        for i in range(n_subs):
            peak = float(peaks[i])
            if peak > env:
                env = g_attack * env + (1.0 - g_attack) * peak
            else:
                env = g_release * env + (1.0 - g_release) * peak
            gain_knots[i + 1] = self._gain_for_env(env)

        self._env = env
        self._prev_gain = float(gain_knots[-1])

        if np.all(gain_knots >= 0.999999):
            return audio

        knot_x = np.empty(n_subs + 1, dtype=np.float32)
        knot_x[0] = 0.0
        for i in range(full_subs):
            knot_x[i + 1] = float((i + 1) * sub_len)
        if full_subs * sub_len < n:
            knot_x[-1] = float(n)

        sample_x = np.arange(n, dtype=np.float32)
        gain_curve = np.interp(sample_x, knot_x, gain_knots).astype(np.float32)
        return (audio * gain_curve).astype(np.float32, copy=False)


def soft_knee_limit_np(
    audio: np.ndarray, threshold: float = 0.85, ceiling: float = 0.98
) -> np.ndarray:
    """
    C1-continuous soft-knee peak limiter without hard threshold jumps.
    Linear below `threshold`, smoothly compressed via tanh up to `ceiling`.
    """
    if audio is None or audio.size == 0:
        return audio
    abs_x = np.abs(audio)
    over = abs_x > threshold
    if not np.any(over):
        return audio
    out = audio.copy()
    headroom = max(ceiling - threshold, 1e-6)
    excess = abs_x[over] - threshold
    compressed = threshold + headroom * np.tanh(excess / headroom)
    out[over] = np.sign(audio[over]) * compressed
    return out
