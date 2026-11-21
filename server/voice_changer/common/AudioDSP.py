import json
import math
from typing import Any, Dict, Optional

import numpy as np
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


class MinimumPhaseFormantFilter(nn.Module):
    """
    Stateful vowel-aware minimum-phase FIR formant filter.
    - Pre-filter operates at pre_sr (16 kHz) on each incoming non-overlapping block
    - Post-filter operates at post_sr (output_sr) on the SOLA-stitched continuous output block
    """

    def __init__(
        self,
        num_taps_pre: int = 256,
        num_taps_post: int = 256,
        n_fft_design: int = 2048,
        n_fft_vowel: int = 1024,
    ):
        super().__init__()
        self.num_taps_pre = num_taps_pre
        self.num_taps_post = num_taps_post
        self.n_fft_design = n_fft_design
        self.n_fft_vowel = n_fft_vowel

        self.register_buffer(
            "pre_history", torch.zeros(num_taps_pre - 1, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "post_history", torch.zeros(num_taps_post - 1, dtype=torch.float32), persistent=False
        )

        self._causal_mask_cache: Dict[tuple, torch.Tensor] = {}
        self._cache_key: Any = None
        self._strength: float = 0.35

        self._h_pre_conv: Optional[torch.Tensor] = None
        self._h_post_conv: Optional[torch.Tensor] = None
        self._is_vowel_aware: bool = False

        self._log_post_close: Optional[torch.Tensor] = None
        self._log_post_open: Optional[torch.Tensor] = None
        self._log_post_front: Optional[torch.Tensor] = None

        self._vowel_win: Optional[torch.Tensor] = None
        self._mask_f1: Optional[torch.Tensor] = None
        self._mask_f2_mid: Optional[torch.Tensor] = None
        self._mask_f2_high: Optional[torch.Tensor] = None

        self._vowel_weights_ema: Optional[torch.Tensor] = None
        self._h_post_prev: Optional[torch.Tensor] = None

    def reset(self) -> None:
        if self.pre_history.numel() > 0:
            self.pre_history.zero_()
        if self.post_history.numel() > 0:
            self.post_history.zero_()
        self._vowel_weights_ema = None
        self._h_post_prev = None

    def _clear_filters(self) -> None:
        self._h_pre_conv = None
        self._h_post_conv = None
        self._is_vowel_aware = False
        self._log_post_close = None
        self._log_post_open = None
        self._log_post_front = None
        self._vowel_weights_ema = None
        self._h_post_prev = None

    def _build_log_norm(
        self, env_raw: Any, src_sr: int, domain_sr: int, device: torch.device
    ) -> Optional[torch.Tensor]:
        if env_raw is None:
            return None
        env_arr = np.asarray(env_raw, dtype=np.float32)
        if env_arr.ndim != 1 or env_arr.size < 2:
            return None

        # Support both legacy log-domain envelopes (contain <= 0 values) and linear envelopes
        if np.any(env_arr <= 0.0):
            log_src = env_arr
        else:
            log_src = np.log(np.maximum(env_arr, 1e-8))

        f_src = np.linspace(0.0, src_sr / 2.0, len(log_src))
        f_out = np.linspace(0.0, domain_sr / 2.0, self.n_fft_design // 2 + 1)
        log_env = np.interp(f_out, f_src, log_src)

        # Fit spectral tilt within the shared valid frequency band
        nyq_common = min(domain_sr / 2.0, src_sr / 2.0)
        valid_mask = f_out <= nyq_common
        x_all = np.arange(len(log_env), dtype=np.float32)
        if np.sum(valid_mask) >= 4:
            x_fit = x_all[valid_mask]
            s_p, i_p = np.polyfit(x_fit, log_env[valid_mask], 1)
        else:
            s_p, i_p = np.polyfit(x_all, log_env, 1)

        log_norm = log_env - (s_p * x_all + i_p)
        log_norm = np.clip(log_norm, -3.0, 3.0)

        # Roll off gracefully above source Nyquist when domain_sr > src_sr
        if domain_sr > src_sr:
            roll_start = 0.9 * (src_sr / 2.0)
            roll_end = src_sr / 2.0
            taper_mask = f_out > roll_start
            if np.any(taper_mask):
                alpha = np.clip(
                    1.0 - (f_out[taper_mask] - roll_start) / max(roll_end - roll_start, 1.0),
                    0.0,
                    1.0,
                )
                log_norm[taper_mask] *= alpha

        return torch.tensor(log_norm, dtype=torch.float32, device=device)

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
        cache_key = (
            bool(active),
            target_env,
            int(target_sr),
            input_env,
            int(input_sr),
            float(strength),
            int(pre_sr),
            int(post_sr),
            str(device),
        )
        if cache_key == self._cache_key:
            return

        self._cache_key = cache_key
        self._strength = float(strength)

        if self.pre_history.device != device or self.post_history.device != device:
            self.to(device)

        if not (active and target_env and input_env and target_sr > 0 and input_sr > 0):
            self._clear_filters()
            return

        try:
            parsed_tgt = json.loads(target_env) if isinstance(target_env, str) else target_env
            parsed_in = json.loads(input_env) if isinstance(input_env, str) else input_env

            # 1. Pre-filter (input whitening at pre_sr = 16000 Hz)
            if isinstance(parsed_in, dict) and any(k in parsed_in for k in ("open", "front", "close")):
                logs_in = []
                for k in ("close", "open", "front"):
                    if k in parsed_in:
                        t_k = self._build_log_norm(parsed_in[k], input_sr, pre_sr, device)
                        if t_k is not None:
                            logs_in.append(t_k)
                log_pre = torch.stack(logs_in, dim=0).mean(dim=0) if logs_in else None
            else:
                log_pre = self._build_log_norm(parsed_in, input_sr, pre_sr, device)

            self._h_pre_conv = (
                compute_minimum_phase_fir(
                    log_pre, -0.5 * self._strength, self.num_taps_pre, self._causal_mask_cache
                )
                if log_pre is not None
                else None
            )

            # 2. Post-filter (target coloring at post_sr = output_sr)
            if (
                isinstance(parsed_tgt, dict)
                and "open" in parsed_tgt
                and "front" in parsed_tgt
                and "close" in parsed_tgt
            ):
                log_c = self._build_log_norm(parsed_tgt["close"], target_sr, post_sr, device)
                log_o = self._build_log_norm(parsed_tgt["open"], target_sr, post_sr, device)
                log_f = self._build_log_norm(parsed_tgt["front"], target_sr, post_sr, device)
                if log_c is not None and log_o is not None and log_f is not None:
                    self._is_vowel_aware = True
                    self._log_post_close = log_c
                    self._log_post_open = log_o
                    self._log_post_front = log_f
                    self._h_post_conv = compute_minimum_phase_fir(
                        log_o, self._strength, self.num_taps_post, self._causal_mask_cache
                    )

                    # Precompute physical frequency band masks for 1024-pt rFFT at post_sr
                    freqs = torch.linspace(
                        0.0, post_sr / 2.0, self.n_fft_vowel // 2 + 1, device=device, dtype=torch.float32
                    )
                    self._mask_f1 = (freqs >= 200.0) & (freqs < 900.0)
                    self._mask_f2_mid = (freqs >= 900.0) & (freqs < 2200.0)
                    self._mask_f2_high = (freqs >= 2200.0) & (freqs <= 4500.0)
                    self._vowel_win = torch.hann_window(
                        self.n_fft_vowel, device=device, dtype=torch.float32
                    )
                    self._vowel_weights_ema = None
                    self._h_post_prev = None
                else:
                    self._is_vowel_aware = False
                    self._h_post_conv = None
            else:
                self._is_vowel_aware = False
                log_post = self._build_log_norm(parsed_tgt, target_sr, post_sr, device)
                self._h_post_conv = (
                    compute_minimum_phase_fir(
                        log_post, self._strength, self.num_taps_post, self._causal_mask_cache
                    )
                    if log_post is not None
                    else None
                )
        except Exception:
            self._clear_filters()

    @torch.no_grad()
    def apply_pre(self, x: torch.Tensor) -> torch.Tensor:
        if self._h_pre_conv is None or x.numel() == 0:
            return x
        orig_dtype = x.dtype
        x_f32 = x.reshape(-1).to(dtype=torch.float32)
        if self.pre_history.device != x_f32.device:
            self.to(x_f32.device)
        buf = torch.cat([self.pre_history, x_f32], dim=0)
        y = F.conv1d(buf.view(1, 1, -1), self._h_pre_conv.view(1, 1, -1)).view(-1)
        self.pre_history = buf[-(self.num_taps_pre - 1) :].detach().clone()
        return y.to(dtype=orig_dtype)

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

        if (
            self._is_vowel_aware
            and self._log_post_close is not None
            and self._log_post_open is not None
            and self._log_post_front is not None
            and self._mask_f1 is not None
        ):
            # Analyze spectral band energy in physical Hz
            if n >= self.n_fft_vowel:
                frames = x_f32.unfold(0, self.n_fft_vowel, self.n_fft_vowel // 2)
                spec_mag = torch.abs(torch.fft.rfft(frames * self._vowel_win, dim=-1)).mean(dim=0)
            else:
                padded = F.pad(x_f32, (0, self.n_fft_vowel - n))
                spec_mag = torch.abs(torch.fft.rfft(padded * self._vowel_win, dim=-1))

            e_close = spec_mag[self._mask_f1].sum() + 1e-8
            e_open = spec_mag[self._mask_f2_mid].sum() + 1e-8
            e_front = spec_mag[self._mask_f2_high].sum() + 1e-8
            weights = torch.stack([e_close, e_open, e_front])
            weights = weights / weights.sum()

            if self._vowel_weights_ema is None:
                self._vowel_weights_ema = weights
            else:
                self._vowel_weights_ema = 0.65 * weights + 0.35 * self._vowel_weights_ema

            w_c, w_o, w_f = self._vowel_weights_ema[0], self._vowel_weights_ema[1], self._vowel_weights_ema[2]
            log_dyn = (
                w_c * self._log_post_close
                + w_o * self._log_post_open
                + w_f * self._log_post_front
            )
            h_curr = compute_minimum_phase_fir(
                log_dyn, self._strength, self.num_taps_post, self._causal_mask_cache
            )
            y = F.conv1d(buf.view(1, 1, -1), h_curr.view(1, 1, -1)).view(-1)

            # Smooth crossfade on leading boundary if filter response changed
            if self._h_post_prev is not None:
                xfade_len = min(64, n)
                if xfade_len > 1:
                    prev_slice = buf[: hist_len + xfade_len]
                    y_prev = F.conv1d(
                        prev_slice.view(1, 1, -1), self._h_post_prev.view(1, 1, -1)
                    ).view(-1)
                    alpha = torch.linspace(0.0, 1.0, xfade_len, device=x_f32.device, dtype=torch.float32)
                    y[:xfade_len] = y[:xfade_len] * alpha + y_prev[:xfade_len] * (1.0 - alpha)

            self._h_post_prev = h_curr.detach()
        else:
            y = F.conv1d(buf.view(1, 1, -1), self._h_post_conv.view(1, 1, -1)).view(-1)

        self.post_history = buf[-hist_len:].detach().clone()
        return y.to(dtype=orig_dtype)


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
