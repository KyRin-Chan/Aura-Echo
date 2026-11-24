import json
import logging
import math
import os
from typing import Any, Callable, Dict, Optional, Tuple

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


def _select_uniform_f0_audio(y: np.ndarray, sr: int, max_f0_duration_sec: float = 90.0) -> np.ndarray:
    """Sample evenly spaced contiguous segments across the full recording for unbiased F0 statistics."""
    max_len = int(round(max_f0_duration_sec * sr))
    if max_len <= 0 or y.shape[0] <= max_len:
        return y
    n_segments = 6
    seg_len = max(sr, max_len // n_segments)
    if seg_len >= y.shape[0]:
        return y
    starts = np.linspace(0, y.shape[0] - seg_len, n_segments, dtype=np.int64)
    return np.concatenate([y[int(s) : int(s) + seg_len] for s in starts])


def _resample_to_16k(y_mono: np.ndarray, sr_int: int) -> np.ndarray:
    if sr_int == 16000:
        return y_mono
    g = math.gcd(16000, sr_int)
    up, down = 16000 // g, sr_int // g
    if max(up, down) <= 2000:
        return scipy_signal.resample_poly(y_mono, up, down).astype(np.float32, copy=False)
    target_len = max(1, int(round(y_mono.shape[0] * 16000.0 / float(sr_int))))
    return scipy_signal.resample(y_mono, target_len).astype(np.float32, copy=False)


def _filter_f0_octave_outliers(f0_vals: np.ndarray, max_oct_dev: float = 1.2) -> np.ndarray:
    """Remove gross octave-jump outliers relative to the log2 median."""
    valid = f0_vals[np.isfinite(f0_vals) & (f0_vals > 0.0)]
    if valid.size < 5:
        return valid
    log2_vals = np.log2(valid.astype(np.float64))
    med = float(np.median(log2_vals))
    inliers = valid[np.abs(log2_vals - med) <= max_oct_dev]
    return inliers if inliers.size >= 3 else valid


def _compute_pitch_stats(f0_voiced: np.ndarray) -> tuple[float, float, float, float, float]:
    """Compute robust F0 median, log2 median, log2 robust std, p10, and p90."""
    in_range = f0_voiced[np.isfinite(f0_voiced) & (f0_voiced >= 55.0) & (f0_voiced <= 800.0)]
    cleaned = _filter_f0_octave_outliers(in_range)
    if cleaned.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    f0_median = float(np.median(cleaned))
    log2_f0 = np.log2(np.maximum(cleaned.astype(np.float64), 1e-3))
    f0_log2_mean = float(np.median(log2_f0))
    f0_p10 = float(np.percentile(cleaned, 10))
    f0_p90 = float(np.percentile(cleaned, 90))
    if cleaned.size >= 5:
        f0_log2_std = float((np.percentile(log2_f0, 90) - np.percentile(log2_f0, 10)) / 2.5631)
    else:
        f0_log2_std = float(np.std(log2_f0))
    return f0_median, f0_log2_mean, f0_log2_std, f0_p10, f0_p90


def _compute_16k_frame_rms(y_16k: np.ndarray, frame_length: int = 1024, hop_length: int = 160) -> np.ndarray:
    if y_16k.size == 0:
        return np.empty(0, dtype=np.float32)
    pad = frame_length // 2
    pad_mode = "reflect" if y_16k.size > pad else "constant"
    y_pad = np.pad(y_16k, (pad, pad), mode=pad_mode)
    if y_pad.size < frame_length:
        y_pad = np.pad(y_pad, (0, frame_length - y_pad.size), mode="constant")
    frames = np.lib.stride_tricks.sliding_window_view(y_pad, frame_length)[::hop_length]
    return np.sqrt(np.mean(frames.astype(np.float32) ** 2, axis=1))


def _extract_f0_with_pitch_extractor(
    y_16k: np.ndarray,
    pitch_extractor: Any,
    device: Any = None,
    window_size: int = 160,
) -> np.ndarray:
    """Extract F0 at 16 kHz using the active neural pitch extractor in bounded chunks."""
    if y_16k.size < window_size:
        return np.empty(0, dtype=np.float32)

    dev = device or getattr(pitch_extractor, "device", None)
    if dev is None:
        dev = torch.device("cpu")

    saved_cent = getattr(pitch_extractor, "_last_voiced_cent", None)
    if hasattr(pitch_extractor, "reset"):
        pitch_extractor.reset()

    chunk_samples = 30 * 16000
    f0_chunks = []
    try:
        with torch.no_grad():
            for start in range(0, y_16k.shape[0], chunk_samples):
                chunk = np.ascontiguousarray(y_16k[start : start + chunk_samples], dtype=np.float32)
                if chunk.size < window_size:
                    continue
                chunk_t = torch.from_numpy(chunk).to(device=dev, dtype=torch.float32)
                out = pitch_extractor.extract(chunk_t, 16000, window_size)
                f0_chunks.append(out.detach().float().reshape(-1).cpu().numpy())
    finally:
        if hasattr(pitch_extractor, "reset"):
            pitch_extractor.reset()
        if saved_cent is not None and hasattr(pitch_extractor, "_last_voiced_cent"):
            pitch_extractor._last_voiced_cent = saved_cent

    if not f0_chunks:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(f0_chunks).astype(np.float32, copy=False)


def analyze_voice_acoustics(
    y: np.ndarray,
    sr: int,
    max_duration_sec: float = 30.0,
    pitch_extractor: Any = None,
    pitch_extractor_device: Any = None,
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
            "f0_log2_mean": 0.0,
            "f0_log2_std": 0.0,
            "f0_p10": 0.0,
            "f0_p90": 0.0,
            "centroid": 0.0,
            "envelope": flat_env,
            "sr": sr_int,
            "voiced_ratio": 0.0,
        }

    # Spectral envelope uses the highest-energy segment for high SNR;
    # F0 statistics cover the full recording (or uniform slices if very long) to avoid loud-burst pitch bias.
    y_env = _select_highest_energy_segment(y_mono, sr_int, max_duration_sec)
    y_f0 = _select_uniform_f0_audio(y_mono, sr_int, max_f0_duration_sec=max(90.0, max_duration_sec))

    y_16k_env = _resample_to_16k(y_env, sr_int)
    same_f0_segment = y_f0 is y_env or (y_f0.shape[0] == y_env.shape[0] and y_mono.shape[0] <= int(round(max_duration_sec * sr_int)))
    y_16k_f0 = y_16k_env if same_f0_segment else _resample_to_16k(y_f0, sr_int)

    yin_hop = 160
    f0_yin, aper_yin, rms_yin = _extract_voiced_f0_yin_16k(
        y_16k_env, sr_16k=16000, frame_length=1024, hop_length=yin_hop, fmin=55.0, fmax=800.0
    )

    # STFT on native sample rate with physically consistent window size
    hop_length = n_fft // 4
    pad = n_fft // 2
    pad_mode = "reflect" if y_env.size > pad else "constant"
    y_pad = np.pad(y_env, (pad, pad), mode=pad_mode)
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
        use_mask = voiced_mask
    else:
        use_mask = energy_mask if active_count > 0 else np.ones(n_frames, dtype=bool)

    # Extract unbiased F0 distribution across y_16k_f0 (preferring active neural pitch extractor)
    f0_voiced_full = np.empty(0, dtype=np.float32)
    if pitch_extractor is not None:
        try:
            f0_pe = _extract_f0_with_pitch_extractor(
                y_16k_f0, pitch_extractor, device=pitch_extractor_device, window_size=yin_hop
            )
            rms_16k = _compute_16k_frame_rms(y_16k_f0, frame_length=1024, hop_length=yin_hop)
            if f0_pe.size > 0 and rms_16k.size > 0:
                rms_pe_p95 = float(np.percentile(rms_16k, 95))
                rms_pe_thresh = max(1e-4, 0.05 * rms_pe_p95)
                idx_pe = np.linspace(0, rms_16k.size - 1, f0_pe.size).round().astype(np.int64)
                pe_mask = (
                    (rms_16k[idx_pe] > rms_pe_thresh)
                    & (f0_pe >= 55.0)
                    & (f0_pe <= 800.0)
                )
                if int(np.sum(pe_mask)) >= 3:
                    f0_voiced_full = f0_pe[pe_mask]
        except Exception:
            f0_voiced_full = np.empty(0, dtype=np.float32)

    if f0_voiced_full.size < 3:
        if same_f0_segment:
            f0_yin_f0, aper_yin_f0, rms_yin_f0 = f0_yin, aper_yin, rms_yin
        else:
            f0_yin_f0, aper_yin_f0, rms_yin_f0 = _extract_voiced_f0_yin_16k(
                y_16k_f0, sr_16k=16000, frame_length=1024, hop_length=yin_hop, fmin=55.0, fmax=800.0
            )
        rms_f0_p95 = float(np.percentile(rms_yin_f0, 95)) if rms_yin_f0.size > 0 else 0.0
        rms_f0_thresh = max(1e-4, 0.05 * rms_f0_p95)
        e_f0_mask = rms_yin_f0 > rms_f0_thresh
        yin_voiced = e_f0_mask & (aper_yin_f0 < 0.25) & (f0_yin_f0 >= 55.0) & (f0_yin_f0 <= 800.0)
        if int(np.sum(yin_voiced)) >= 3:
            f0_voiced_full = f0_yin_f0[yin_voiced]
        else:
            fallback_f0 = e_f0_mask & (f0_yin_f0 >= 55.0) & (f0_yin_f0 <= 800.0)
            f0_voiced_full = f0_yin_f0[fallback_f0] if np.any(fallback_f0) else np.empty(0, dtype=np.float32)

    f0_median, f0_log2_mean, f0_log2_std, f0_p10, f0_p90 = _compute_pitch_stats(f0_voiced_full)

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
        "f0_log2_mean": f0_log2_mean,
        "f0_log2_std": f0_log2_std,
        "f0_p10": f0_p10,
        "f0_p90": f0_p90,
        "centroid": centroid,
        "envelope": envelope,
        "sr": sr_int,
        "voiced_ratio": voiced_ratio,
    }


def map_f0_distribution(
    f0: torch.Tensor,
    src_log2_mean: float,
    src_log2_std: float,
    tgt_log2_mean: float,
    tgt_log2_std: float,
    tgt_f0_p10: float = 0.0,
    tgt_f0_p90: float = 0.0,
    pitch_scale: float = 1.0,
    pitch_offset_semitones: float = 0.0,
    soft_margin_semitones: float = 4.0,
    pitch_mode: str = "speech",
) -> torch.Tensor:
    """
    Map voiced F0 frames from source log2 center into target speaker's log2
    center.
    - speech: preserves natural semitone intonation (1:1 at pitch_scale=1.0)
      with gentle tanh soft-compression beyond a wide comfort zone.
    - singing: snaps center shift to the nearest integer semitone, locks scale
      to 1.0, and disables soft-compression so musical intervals stay in tune.
    Unvoiced frames (f0 <= 0) remain 0.
    """
    del src_log2_std, tgt_log2_std
    if f0.numel() == 0 or src_log2_mean <= 0.0 or tgt_log2_mean <= 0.0:
        return f0

    orig_dtype = f0.dtype
    f0_f32 = f0.float()
    voiced_mask = f0_f32 > 0.0
    if not torch.any(voiced_mask):
        return f0

    mode = "singing" if str(pitch_mode).strip().lower() == "singing" else "speech"
    offset_oct = float(pitch_offset_semitones) / 12.0
    log2_in = torch.log2(torch.clamp(f0_f32, min=1e-3))

    if mode == "singing":
        shift_oct = round((float(tgt_log2_mean) - float(src_log2_mean)) * 12.0) / 12.0
        soft_mapped = log2_in + shift_oct + offset_oct
    else:
        scale = max(0.30, min(1.50, float(pitch_scale)))
        tgt_center = float(tgt_log2_mean) + offset_oct
        mapped = tgt_center + scale * (log2_in - float(src_log2_mean))

        lo_min_comfort = tgt_center - 6.0 / 12.0
        hi_min_comfort = tgt_center + 7.0 / 12.0
        if tgt_f0_p10 > 0.0 and tgt_f0_p90 > tgt_f0_p10:
            lo = min(math.log2(float(tgt_f0_p10)) + offset_oct - 2.0 / 12.0, lo_min_comfort)
            hi = max(math.log2(float(tgt_f0_p90)) + offset_oct + 2.0 / 12.0, hi_min_comfort)
        else:
            lo = lo_min_comfort
            hi = hi_min_comfort

        margin = max(0.10, float(soft_margin_semitones) / 12.0)
        soft_mapped = torch.where(
            mapped > hi,
            hi + margin * torch.tanh((mapped - hi) / margin),
            torch.where(
                mapped < lo,
                lo - margin * torch.tanh((lo - mapped) / margin),
                mapped,
            ),
        )

    out_hz = torch.pow(2.0, soft_mapped).clamp(50.0, 1100.0)
    out = torch.where(voiced_mask, out_hz, torch.zeros_like(f0_f32))
    return out.to(dtype=orig_dtype)


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


class StatefulAutoGainControl:
    """
    Stateful RMS-tracking Auto Gain Control (AGC) for real-time speech streams.

    Smoothly boosts quiet speech toward `target_rms_db`, gates out silence/noise
    below `noise_floor_db` with hold and slow idle release, and attenuates loud
    segments with peak-aware clamping to avoid clipping.
    """

    def __init__(
        self,
        target_rms_db: float = -18.0,
        noise_floor_db: float = -45.0,
        max_gain_db: float = 15.0,
        min_gain_db: float = -12.0,
        attack_ms: float = 25.0,
        release_ms: float = 250.0,
        hold_ms: float = 250.0,
        idle_release_ms: float = 1200.0,
        rms_smooth_ms: float = 30.0,
        peak_ceiling: float = 0.90,
    ):
        self.target_rms_db = float(target_rms_db)
        self.noise_floor_db = float(noise_floor_db)
        self.max_gain_db = float(max_gain_db)
        self.min_gain_db = float(min_gain_db)
        self.attack_ms = max(0.5, float(attack_ms))
        self.release_ms = max(1.0, float(release_ms))
        self.hold_ms = max(0.0, float(hold_ms))
        self.idle_release_ms = max(1.0, float(idle_release_ms))
        self.rms_smooth_ms = max(1.0, float(rms_smooth_ms))
        self.peak_ceiling = max(0.05, float(peak_ceiling))
        self._power_env: float = 0.0
        self._gain_db: float = 0.0
        self._prev_gain: float = 1.0
        self._hold_remaining_s: float = 0.0

    def reset(self) -> None:
        self._power_env = 0.0
        self._gain_db = 0.0
        self._prev_gain = 1.0
        self._hold_remaining_s = 0.0

    def process(self, audio: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
        if audio is None or audio.size == 0:
            return audio

        n = audio.shape[0]
        sr = max(1, int(sample_rate))
        sub_len = max(1, int(round(sr * 0.005)))
        abs_x = np.abs(audio)
        sq_x = np.square(audio, dtype=np.float32)

        full_subs = n // sub_len
        if full_subs > 0 and full_subs * sub_len == n:
            powers = sq_x.reshape(full_subs, sub_len).mean(axis=1)
            peaks = abs_x.reshape(full_subs, sub_len).max(axis=1)
        elif full_subs > 0:
            head_sq = sq_x[: full_subs * sub_len].reshape(full_subs, sub_len).mean(axis=1)
            tail_sq = np.array([sq_x[full_subs * sub_len :].mean()], dtype=np.float32)
            powers = np.concatenate([head_sq, tail_sq])

            head_peaks = abs_x[: full_subs * sub_len].reshape(full_subs, sub_len).max(axis=1)
            tail_peak = np.array([abs_x[full_subs * sub_len :].max()], dtype=np.float32)
            peaks = np.concatenate([head_peaks, tail_peak])
        else:
            powers = np.array([sq_x.mean()], dtype=np.float32)
            peaks = np.array([abs_x.max()], dtype=np.float32)

        dt_sub = sub_len / float(sr)
        a_rms = math.exp(-dt_sub / (self.rms_smooth_ms / 1000.0))
        a_attack = math.exp(-dt_sub / (self.attack_ms / 1000.0))
        a_release = math.exp(-dt_sub / (self.release_ms / 1000.0))
        a_idle = math.exp(-dt_sub / (self.idle_release_ms / 1000.0))

        n_subs = powers.shape[0]
        gain_knots = np.empty(n_subs + 1, dtype=np.float32)
        gain_knots[0] = self._prev_gain

        power_env = self._power_env
        gain_db = self._gain_db
        hold_remaining_s = self._hold_remaining_s

        for i in range(n_subs):
            sub_power = float(powers[i])
            sub_peak = float(peaks[i])
            power_env = a_rms * power_env + (1.0 - a_rms) * sub_power
            rms = math.sqrt(max(power_env, 1e-12))
            rms_db = 20.0 * math.log10(rms)
            inst_rms_db = 10.0 * math.log10(max(sub_power, 1e-12))

            if rms_db >= self.noise_floor_db and inst_rms_db >= self.noise_floor_db - 6.0:
                hold_remaining_s = self.hold_ms / 1000.0
                desired_db = max(self.min_gain_db, min(self.max_gain_db, self.target_rms_db - rms_db))
                if sub_peak > 1e-6:
                    max_peak_db = 20.0 * math.log10(self.peak_ceiling / sub_peak)
                    desired_db = min(desired_db, max_peak_db)
                coeff = a_attack if desired_db < gain_db else a_release
                gain_db = coeff * gain_db + (1.0 - coeff) * desired_db
            else:
                if hold_remaining_s > 0.0:
                    hold_remaining_s = max(0.0, hold_remaining_s - dt_sub)
                    if sub_peak > 1e-6 and sub_peak * (10.0 ** (gain_db / 20.0)) > self.peak_ceiling:
                        gain_db = min(gain_db, 20.0 * math.log10(self.peak_ceiling / sub_peak))
                else:
                    gain_db = a_idle * gain_db

            gain_knots[i + 1] = 10.0 ** (gain_db / 20.0)

        self._power_env = power_env
        self._gain_db = gain_db
        self._prev_gain = float(gain_knots[-1])
        self._hold_remaining_s = hold_remaining_s

        if np.all(np.abs(gain_knots - 1.0) < 1e-6):
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


logger = logging.getLogger(__name__)


class SileroVAD:
    """
    Lightweight streaming Silero VAD v5 wrapper using CPU ONNX Runtime.

    Maintains a 16 kHz 512-sample (32 ms) frame buffer with 64-sample context
    and persistent LSTM state across arbitrary incoming chunk sizes.
    """

    VAD_SR = 16000
    FRAME_SAMPLES = 512
    CONTEXT_SAMPLES = 64

    def __init__(self, model_path: Optional[str] = None):
        self._model_path = model_path
        self._session = None
        self._load_attempted: bool = False
        self._available: bool = False
        self._sr_arr = np.array(self.VAD_SR, dtype=np.int64)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(self.CONTEXT_SAMPLES, dtype=np.float32)
        self._ring = np.empty(0, dtype=np.float32)
        self._last_prob: float = 0.0

    @property
    def is_available(self) -> bool:
        if not self._load_attempted:
            self._load_session()
        return self._available

    def _resolve_model_path(self) -> Optional[str]:
        candidates = []
        if self._model_path:
            candidates.append(self._model_path)
        candidates.append(os.path.join("pretrain", "vad", "silero_vad.onnx"))
        repo_server_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        candidates.append(os.path.join(repo_server_dir, "pretrain", "vad", "silero_vad.onnx"))
        for path in candidates:
            if path and os.path.isfile(path):
                return path
        return None

    def _load_session(self) -> None:
        self._load_attempted = True
        resolved = self._resolve_model_path()
        if not resolved:
            logger.warning("Silero VAD ONNX model not found at pretrain/vad/silero_vad.onnx; falling back to noise gate.")
            self._available = False
            return
        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            opts.log_severity_level = 3
            self._session = ort.InferenceSession(
                resolved,
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self._available = True
            logger.info("Loaded Silero VAD ONNX model from %s", resolved)
        except Exception as exc:
            logger.warning("Failed to load Silero VAD ONNX model (%s); falling back to noise gate.", exc)
            self._session = None
            self._available = False

    def reset(self) -> None:
        self._state.fill(0.0)
        self._context.fill(0.0)
        self._ring = np.empty(0, dtype=np.float32)
        self._last_prob = 0.0

    def process_chunk(self, audio_f32: np.ndarray, sample_rate: int) -> float:
        if not self.is_available or self._session is None:
            return 1.0
        if audio_f32.size == 0:
            return self._last_prob

        sr = max(800, int(sample_rate))
        if sr != self.VAD_SR:
            g = math.gcd(self.VAD_SR, sr)
            up = self.VAD_SR // g
            down = sr // g
            audio_16k = scipy_signal.resample_poly(audio_f32, up, down).astype(np.float32, copy=False)
        else:
            audio_16k = np.asarray(audio_f32, dtype=np.float32).reshape(-1)

        if audio_16k.size == 0:
            return self._last_prob

        self._ring = np.concatenate((self._ring, audio_16k)) if self._ring.size > 0 else audio_16k

        ran_frame = False
        max_prob = 0.0
        offset = 0
        total = int(self._ring.shape[0])
        while total - offset >= self.FRAME_SAMPLES:
            frame = self._ring[offset : offset + self.FRAME_SAMPLES]
            offset += self.FRAME_SAMPLES
            x = np.concatenate((self._context, frame))[np.newaxis, :]
            out, self._state = self._session.run(
                None,
                {"input": x, "state": self._state, "sr": self._sr_arr},
            )
            self._context = frame[-self.CONTEXT_SAMPLES :].copy()
            prob = float(out[0, 0])
            if not ran_frame or prob > max_prob:
                max_prob = prob
            ran_frame = True

        if offset > 0:
            self._ring = np.array(self._ring[offset:], dtype=np.float32, copy=True)

        if ran_frame:
            self._last_prob = max_prob
        return self._last_prob


class AdaptiveNoiseGate:
    """
    Input noise gate designed for mediocre/noisy microphones.
    """

    def __init__(
        self,
        bandpass_low_hz: float = 100.0,
        bandpass_high_hz: float = 3800.0,
        hysteresis_db: float = 3.0,
        min_hangover_sec: float = 0.10,
        max_hangover_sec: float = 0.35,
        noise_margin_db: float = 6.0,
        max_adaptive_boost_db: float = 10.0,
        impulse_crest_threshold: float = 3.8,
        impulse_quarter_ratio: float = 0.35,
        max_f0_hold_sec: float = 2.0,
        vad_model_path: Optional[str] = None,
    ):
        self.bandpass_low_hz = float(bandpass_low_hz)
        self.bandpass_high_hz = float(bandpass_high_hz)
        self.hysteresis_db = float(hysteresis_db)
        self.min_hangover_sec = float(min_hangover_sec)
        self.max_hangover_sec = float(max_hangover_sec)
        self.noise_margin_db = float(noise_margin_db)
        self.max_adaptive_boost_db = float(max_adaptive_boost_db)
        self.impulse_crest_threshold = float(impulse_crest_threshold)
        self.impulse_quarter_ratio = float(impulse_quarter_ratio)
        self.max_f0_hold_sec = float(max_f0_hold_sec)

        self.is_open: bool = False
        self.hangover_chunks: int = 0
        self._sr: int = 0
        self._sos: Optional[np.ndarray] = None
        self._zi: Optional[np.ndarray] = None
        self._noise_floor_rms: Optional[float] = None
        self._noise_obs_sec: float = 0.0
        self._speech_duration_sec: float = 0.0
        self._peak_snr_db: float = 0.0
        self._f0_hold_sec: float = 0.0
        self._vad = SileroVAD(model_path=vad_model_path)

    def reset(self) -> None:
        self.is_open = False
        self.hangover_chunks = 0
        if self._sos is not None:
            self._zi = np.zeros((self._sos.shape[0], 2), dtype=np.float64)
        else:
            self._zi = None
        self._noise_floor_rms = None
        self._noise_obs_sec = 0.0
        self._speech_duration_sec = 0.0
        self._peak_snr_db = 0.0
        self._f0_hold_sec = 0.0
        self._vad.reset()

    def _ensure_filter(self, sr: int) -> None:
        if sr == self._sr and self._sos is not None and self._zi is not None:
            return
        low = max(20.0, min(self.bandpass_low_hz, sr * 0.2))
        high = max(low + 100.0, min(self.bandpass_high_hz, sr * 0.45))
        self._sos = scipy_signal.butter(2, [low, high], btype="bandpass", fs=sr, output="sos")
        self._zi = np.zeros((self._sos.shape[0], 2), dtype=np.float64)
        self._sr = sr

    def _compute_detector_rms(
        self,
        audio_f32: np.ndarray,
        sr: int,
        apply_impulse_reject: bool,
    ) -> Tuple[float, float, float]:
        vol = float(np.sqrt(np.mean(np.square(audio_f32, dtype=np.float32))))
        self._ensure_filter(sr)
        key_sig, self._zi = scipy_signal.sosfilt(self._sos, audio_f32, zi=self._zi)
        key_f32 = np.asarray(key_sig, dtype=np.float32)

        n_samples = int(key_f32.shape[0])
        win = max(64, int(round(0.02 * sr)))
        sub_pairs = []
        if n_samples > win:
            n_full = (n_samples // win) * win
            for start in range(0, n_full, win):
                sub_pairs.append((audio_f32[start : start + win], key_f32[start : start + win]))
            if n_samples > n_full:
                sub_pairs.append((audio_f32[n_full:], key_f32[n_full:]))
        else:
            sub_pairs.append((audio_f32, key_f32))

        raw_sub_rms = []
        eff_sub_rms = []
        for raw_sub, key_sub in sub_pairs:
            sub_len = int(key_sub.shape[0])
            if sub_len == 0:
                continue
            raw_s_rms = float(np.sqrt(np.mean(np.square(raw_sub, dtype=np.float32))))
            key_s_rms = float(np.sqrt(np.mean(np.square(key_sub, dtype=np.float32))))
            # Passband gain <= 1.0, so capping by raw sub-frame RMS suppresses cross-chunk IIR ringdown
            s_rms = min(key_s_rms, raw_s_rms)
            raw_sub_rms.append(s_rms)
            eff = s_rms

            if apply_impulse_reject and sub_len >= 16:
                s_peak = float(np.max(np.abs(key_sub)))
                crest = s_peak / (s_rms + 1e-8)
                if crest > self.impulse_crest_threshold:
                    q_len = sub_len // 4
                    if q_len > 0:
                        quarters = key_sub[: q_len * 4].reshape(4, q_len)
                        q_rms = np.sqrt(np.mean(np.square(quarters, dtype=np.float32), axis=1))
                        q_sorted = np.sort(q_rms)[::-1]
                        q_max = float(q_sorted[0])
                        q_second = float(q_sorted[1])
                        if q_second < self.impulse_quarter_ratio * q_max:
                            eff = min(q_second, raw_s_rms)

            eff_sub_rms.append(eff)

        if not raw_sub_rms:
            return vol, 0.0, 0.0

        gate_rms = max(eff_sub_rms)
        frame_floor = float(np.percentile(raw_sub_rms, 25))
        return vol, gate_rms, frame_floor

    def evaluate(
        self,
        audio_in: np.ndarray,
        sample_rate: int,
        silent_threshold_db: float,
        vad_enabled: bool = False,
        vad_threshold: float = 0.45,
        f0_voiced_fn: Optional[Callable[[], bool]] = None,
    ) -> Tuple[bool, float]:
        audio_f32 = np.asarray(audio_in, dtype=np.float32).reshape(-1)
        n_samples = int(audio_f32.shape[0])
        if n_samples <= 0:
            return False, 0.0

        sr = max(800, int(sample_rate))
        chunk_dur = n_samples / float(sr)
        thresh_db = float(silent_threshold_db)
        gate_enabled = thresh_db > -89.0
        gate_closed = (not self.is_open) and (self.hangover_chunks == 0)

        vol, gate_rms, frame_floor = self._compute_detector_rms(
            audio_f32, sr, apply_impulse_reject=(gate_closed and gate_enabled)
        )

        use_vad = bool(vad_enabled) and self._vad.is_available
        vad_prob = self._vad.process_chunk(audio_f32, sr) if use_vad else 1.0

        user_open = 10.0 ** (thresh_db / 20.0)
        max_open = user_open * (10.0 ** (self.max_adaptive_boost_db / 20.0))
        if gate_enabled and self._noise_floor_rms is not None and self._noise_obs_sec >= 0.25:
            adaptive_open = self._noise_floor_rms * (10.0 ** (self.noise_margin_db / 20.0))
            open_thresh = max(user_open, min(adaptive_open, max_open))
        else:
            open_thresh = user_open

        hold_thresh = open_thresh * (10.0 ** (-self.hysteresis_db / 20.0))
        active_thresh = open_thresh if gate_closed else hold_thresh
        energy_ok = gate_rms >= active_thresh

        if not use_vad:
            speech_active = energy_ok
            f0_hold = False
            self._f0_hold_sec = 0.0
        else:
            vad_open_thresh = float(np.clip(vad_threshold, 0.05, 0.95))
            vad_keep_thresh = max(0.08, min(0.15, vad_open_thresh * 0.35))
            if gate_closed:
                # Opening strictly requires both energy above threshold and VAD speech detection
                speech_active = energy_ok and (vad_prob >= vad_open_thresh)
                f0_hold = False
                self._f0_hold_sec = 0.0
            elif energy_ok and vad_prob >= vad_keep_thresh:
                speech_active = True
                f0_hold = False
                self._f0_hold_sec = 0.0
            elif (
                energy_ok
                and f0_voiced_fn is not None
                and self._f0_hold_sec < self.max_f0_hold_sec
                and f0_voiced_fn()
            ):
                # Keep gate open while F0 remains continuously voiced (e.g. singing/humming)
                speech_active = False
                f0_hold = True
                self._f0_hold_sec += chunk_dur
            else:
                speech_active = False
                f0_hold = False

        eff_min_hangover = max(self.min_hangover_sec, 0.22) if use_vad else self.min_hangover_sec
        eff_max_hangover = max(self.max_hangover_sec, 0.45) if use_vad else self.max_hangover_sec

        if speech_active:
            self._speech_duration_sec = min(2.0, self._speech_duration_sec + chunk_dur)
            snr_db = 20.0 * math.log10(max(gate_rms, 1e-8) / max(open_thresh, 1e-8))
            snr_clamped = min(24.0, max(0.0, snr_db))
            self._peak_snr_db = max(snr_clamped, 0.85 * self._peak_snr_db + 0.15 * snr_clamped)

            speech_factor = min(1.0, max(0.0, self._speech_duration_sec - chunk_dur) / 0.50)
            snr_factor = min(1.0, self._peak_snr_db / 18.0)
            elastic = 0.80 * speech_factor + 0.20 * snr_factor
            hangover_sec = eff_min_hangover + (eff_max_hangover - eff_min_hangover) * elastic
            self.hangover_chunks = max(1, int(round(hangover_sec / max(chunk_dur, 1e-4))))
            self.is_open = True
            gate_active = True
        elif f0_hold:
            min_f0_hangover_chunks = max(1, int(round(eff_min_hangover / max(chunk_dur, 1e-4))))
            self.hangover_chunks = max(self.hangover_chunks, min_f0_hangover_chunks)
            self.is_open = True
            gate_active = True
        elif self.hangover_chunks > 0:
            self.hangover_chunks -= 1
            self.is_open = False
            gate_active = True
            if self.hangover_chunks == 0:
                self._speech_duration_sec = 0.0
                self._peak_snr_db = 0.0
                self._f0_hold_sec = 0.0
        else:
            self.is_open = False
            self._speech_duration_sec = 0.0
            self._peak_snr_db = 0.0
            self._f0_hold_sec = 0.0
            gate_active = False

        # Track stationary noise floor on closed gate, quieter valleys, or low-modulation
        # stationary background noise that sits within the plausible noise-floor band (<= max_open).
        is_stationary_noise = (
            gate_enabled
            and frame_floor <= max_open
            and gate_rms <= 1.8 * max(frame_floor, 1e-8)
        )
        should_update_floor = (
            (not gate_active)
            or (self._noise_floor_rms is not None and frame_floor < self._noise_floor_rms)
            or is_stationary_noise
        )
        if should_update_floor:
            if self._noise_floor_rms is None:
                self._noise_floor_rms = frame_floor
            elif frame_floor < self._noise_floor_rms:
                alpha = 1.0 - math.exp(-chunk_dur / 0.25)
                self._noise_floor_rms += alpha * (frame_floor - self._noise_floor_rms)
            else:
                tau_up = 2.0 if not gate_active else 1.5
                alpha = 1.0 - math.exp(-chunk_dur / tau_up)
                self._noise_floor_rms += alpha * (frame_floor - self._noise_floor_rms)

        if (not gate_active) or is_stationary_noise:
            self._noise_obs_sec = min(10.0, self._noise_obs_sec + chunk_dur)

        return gate_active, vol
