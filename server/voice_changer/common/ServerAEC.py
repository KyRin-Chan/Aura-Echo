import math
from typing import Optional

import numpy as np


class ServerAEC:
    """
    Real-time CPU-based Acoustic Echo Canceller (AEC) for Server Audio mode.

    Combines:
      1. Automatic bulk-delay tracker (0..300 ms) via sub-banded/decimated waveform
         cross-correlation so cross-device setups (e.g. external speaker + Realtek mic)
         align automatically without manual delay tuning.
      2. Dual-filter (background/foreground) Partitioned Block Frequency-Domain Adaptive
         Filter (PBFDAF / MDF) with linear-convolution gradient constraint.
      3. Zero-latency causal minimum-phase Wiener Residual Echo Suppressor (RES) to remove
         non-linear speaker distortion and late room reverberation while preserving
         near-end speech during double-talk.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        max_delay_ms: float = 300.0,
        tail_ms: float = 64.0,
    ):
        self.sample_rate = max(8000, int(sample_rate))
        # Choose power-of-two hop N (~5.3ms at 48kHz) matching sample rate scale
        if self.sample_rate >= 88200:
            self.hop_size = 512
        elif self.sample_rate >= 32000:
            self.hop_size = 256
        else:
            self.hop_size = 128

        self.fft_size = self.hop_size * 2
        self.n_bins = self.fft_size // 2 + 1
        self.n_partitions = max(8, int(math.ceil((tail_ms / 1000.0) * self.sample_rate / self.hop_size)))

        self.max_delay_hops = max(8, int(math.ceil((max_delay_ms / 1000.0) * self.sample_rate / self.hop_size)))
        self._fft_hist_len = self.max_delay_hops + self.n_partitions + 4

        # Decimation factor for fast ~3 kHz bulk-delay cross-correlation
        self._decim = max(4, self.hop_size // 16)
        self._decim_step = self.hop_size // self._decim
        self._corr_hops = max(64, int(math.ceil(0.65 * self.sample_rate / self.hop_size)))
        self._corr_len = self._corr_hops * self._decim_step

        self.reset()

    def reset(self) -> None:
        N = self.hop_size
        M = self.fft_size
        P = self.n_partitions
        K = self.n_bins

        # Continuous grid-aligned rolling time buffers for overlap-save and partial tails
        self._hist_len = (self.max_delay_hops + P + 4) * N
        self._mic_hist = np.zeros(self._hist_len, dtype=np.float32)
        self._ref_hist = np.zeros(self._hist_len, dtype=np.float32)
        self._y_fg_hist = np.zeros(M, dtype=np.float32)
        self._e_fg_hist = np.zeros(M, dtype=np.float32)

        # Accumulator for samples modulo hop_size so arbitrary block_frame lengths work at zero latency
        self._grid_pos = 0  # [0 .. N-1] position inside current N-aligned frame

        # Ring of M-point rfft(ref) frames
        self._ref_fft_hist = np.zeros((self._fft_hist_len, K), dtype=np.complex64)
        self._ref_fft_head = 0

        # Dual-filter MDF weights (background adapts continuously, foreground protects against double-talk)
        self._W_bg = np.zeros((P, K), dtype=np.complex64)
        self._W_fg = np.zeros((P, K), dtype=np.complex64)
        self._Px = np.full(K, 1e-4, dtype=np.float32)

        # Bulk delay tracking state (in hop_size units)
        self.bulk_delay_hops = 0
        self._candidate_delay_hops = 0
        self._candidate_streak = 0
        self._delay_check_counter = 0
        self._bg_diverge_streak = 0
        self._converged_erle_ema = 1.0

        # Decimated ~3 kHz buffers for delay cross-correlation
        self._mic_decim_buf = np.zeros(self._corr_len, dtype=np.float32)
        self._ref_decim_buf = np.zeros(self._corr_len, dtype=np.float32)
        self._mic_dc_prev = 0.0
        self._ref_dc_prev = 0.0

        # Residual Echo Suppressor (RES) state
        self._hann = np.hanning(M).astype(np.float32)
        self._P_dd = np.zeros(K, dtype=np.float32)
        self._P_ee = np.zeros(K, dtype=np.float32)
        self._P_yy = np.zeros(K, dtype=np.float32)
        self._P_xx_del = np.zeros(K, dtype=np.float32)
        self._G_smooth = np.ones(K, dtype=np.float32)
        self._echo_gain_est = 0.25

        # Causal minimum-phase lifter mask (for zero-latency RES FIR construction)
        lifter = np.zeros(M, dtype=np.float32)
        lifter[0] = 1.0
        lifter[M // 2] = 1.0
        lifter[1 : M // 2] = 2.0
        self._causal_lifter = lifter

        taper_len = min(32, N // 4)
        taper = np.ones(N, dtype=np.float32)
        if taper_len > 1:
            taper[-taper_len:] = 0.5 * (
                1.0 + np.cos(np.linspace(0.0, math.pi, taper_len, dtype=np.float32))
            )
        self._ir_taper = taper
        self._last_H_min = np.ones(K, dtype=np.complex64)
        self._res_active = False

    def _push_history(self, mic_samples: np.ndarray, ref_samples: np.ndarray) -> None:
        k = int(mic_samples.shape[0])
        if k <= 0:
            return
        if k >= self._hist_len:
            self._mic_hist[:] = mic_samples[-self._hist_len :]
            self._ref_hist[:] = ref_samples[-self._hist_len :]
        else:
            self._mic_hist[:-k] = self._mic_hist[k:]
            self._mic_hist[-k:] = mic_samples
            self._ref_hist[:-k] = self._ref_hist[k:]
            self._ref_hist[-k:] = ref_samples

    def _update_decim_and_delay(self, mic_hop: np.ndarray, ref_hop: np.ndarray) -> None:
        step = self._decim_step
        d = self._decim
        # Simple high-pass (1 - z^-1) before decimation removes DC / 50-60Hz hum bias
        mic_hp = np.empty_like(mic_hop)
        mic_hp[0] = mic_hop[0] - self._mic_dc_prev
        mic_hp[1:] = mic_hop[1:] - mic_hop[:-1]
        self._mic_dc_prev = float(mic_hop[-1])

        ref_hp = np.empty_like(ref_hop)
        ref_hp[0] = ref_hop[0] - self._ref_dc_prev
        ref_hp[1:] = ref_hop[1:] - ref_hop[:-1]
        self._ref_dc_prev = float(ref_hop[-1])

        mic_d = mic_hp.reshape(step, d).mean(axis=1)
        ref_d = ref_hp.reshape(step, d).mean(axis=1)

        self._mic_decim_buf[:-step] = self._mic_decim_buf[step:]
        self._mic_decim_buf[-step:] = mic_d
        self._ref_decim_buf[:-step] = self._ref_decim_buf[step:]
        self._ref_decim_buf[-step:] = ref_d

        self._delay_check_counter += 1
        if self._delay_check_counter < 6:
            return
        self._delay_check_counter = 0

        # Skip estimation if reference or microphone is quiet
        ref_energy = float(np.dot(self._ref_decim_buf, self._ref_decim_buf))
        mic_energy = float(np.dot(self._mic_decim_buf, self._mic_decim_buf))
        if ref_energy < 1e-6 or mic_energy < 1e-8:
            return

        # Normalized cross-correlation via FFT over [0 .. max_delay_hops]
        # Zero-pad to 2*L to perform linear (non-circular) cross-correlation
        L = self._corr_len
        n_fft = 2 * L
        # Compare only the recent half of mic against full ref history so delayed lags stay valid
        mic_recent = np.zeros(L, dtype=np.float32)
        half = L // 2
        mic_recent[half:] = self._mic_decim_buf[half:]
        mic_recent_energy = float(np.dot(mic_recent[half:], mic_recent[half:]))
        if mic_recent_energy < 1e-8:
            return

        F_mic = np.fft.rfft(mic_recent, n=n_fft)
        F_ref = np.fft.rfft(self._ref_decim_buf, n=n_fft)
        cc = np.fft.irfft(F_mic * np.conj(F_ref), n=n_fft)

        max_lag_decim = min(half - 1, self.max_delay_hops * step)
        if max_lag_decim <= step:
            return

        # Positive lag k means mic[t] correlates with ref[t - k]
        cc_lags = np.abs(cc[:max_lag_decim])
        # Compute local window energy of ref for each lag via cumulative sum
        ref_sq = self._ref_decim_buf.astype(np.float64) ** 2
        cs = np.concatenate(([0.0], np.cumsum(ref_sq)))
        lags = np.arange(max_lag_decim, dtype=np.int64)
        # mic_recent[half:L] aligns with ref_decim_buf[half - lag : L - lag]
        starts = half - lags
        ends = L - lags
        valid = starts >= 0
        norm_cc = np.zeros(max_lag_decim, dtype=np.float32)
        ref_win_e = np.maximum(cs[ends[valid]] - cs[starts[valid]], 1e-9)
        norm_cc[valid] = (
            cc_lags[valid] / np.sqrt(mic_recent_energy * ref_win_e)
        ).astype(np.float32)

        best_lag_decim = int(np.argmax(norm_cc))
        best_score = float(norm_cc[best_lag_decim])
        if best_score < 0.25:
            return

        best_hop = best_lag_decim // step
        # Check if best_hop is already comfortably inside [bulk_delay_hops, bulk_delay_hops + P - 3]
        if self.bulk_delay_hops <= best_hop <= self.bulk_delay_hops + max(2, self.n_partitions - 4):
            self._candidate_streak = 0
            return

        # Place target bulk delay 1 hop before the peak so pre-echo + main echo + tail fit cleanly
        target_delay = max(0, min(self.max_delay_hops, best_hop - 1))

        # Current window max score
        cur_start = self.bulk_delay_hops * step
        cur_end = min(max_lag_decim, (self.bulk_delay_hops + self.n_partitions) * step)
        cur_score = float(np.max(norm_cc[cur_start:cur_end])) if cur_end > cur_start else 0.0

        # Switch quickly on initial convergence or after 2 consistent strong detections
        if best_score > max(0.32, cur_score * 1.25):
            if abs(target_delay - self._candidate_delay_hops) <= 2:
                self._candidate_streak += 1
            else:
                self._candidate_delay_hops = target_delay
                self._candidate_streak = 1

            needed = 1 if (self._converged_erle_ema < 1.35 and best_score > 0.42) else 2
            if self._candidate_streak >= needed:
                self._shift_bulk_delay(target_delay)
                self._candidate_streak = 0

    def _shift_bulk_delay(self, new_delay_hops: int) -> None:
        old_delay = self.bulk_delay_hops
        delta = new_delay_hops - old_delay
        if delta == 0:
            return

        P = self.n_partitions
        if abs(delta) < P:
            new_fg = np.zeros_like(self._W_fg)
            new_bg = np.zeros_like(self._W_bg)
            for p in range(P):
                src_p = p + delta
                if 0 <= src_p < P:
                    new_fg[p] = self._W_fg[src_p]
                    new_bg[p] = self._W_bg[src_p]
            self._W_fg = new_fg
            self._W_bg = new_bg
        else:
            self._W_fg.fill(0.0)
            self._W_bg.fill(0.0)

        self.bulk_delay_hops = new_delay_hops

    def _get_partition_spectra(self, delay_hops: int) -> np.ndarray:
        idx = (self._ref_fft_head - delay_hops - np.arange(self.n_partitions)) % self._fft_hist_len
        return self._ref_fft_hist[idx]

    def _compute_res_filter(
        self,
        d_win: np.ndarray,
        e_win: np.ndarray,
        y_win: np.ndarray,
        X_del: np.ndarray,
        ref_rms: float,
    ) -> np.ndarray:
        K = self.n_bins
        M = self.fft_size
        N = self.hop_size

        D_w = np.fft.rfft(d_win * self._hann, n=M)
        E_w = np.fft.rfft(e_win * self._hann, n=M)
        Y_w = np.fft.rfft(y_win * self._hann, n=M)

        S_dd = (D_w.real * D_w.real + D_w.imag * D_w.imag).astype(np.float32)
        S_ee = (E_w.real * E_w.real + E_w.imag * E_w.imag).astype(np.float32)
        S_yy = (Y_w.real * Y_w.real + Y_w.imag * Y_w.imag).astype(np.float32)
        S_xx = (X_del.real * X_del.real + X_del.imag * X_del.imag).astype(np.float32)

        alpha = np.float32(0.55)
        self._P_dd = alpha * self._P_dd + (1.0 - alpha) * S_dd
        self._P_ee = alpha * self._P_ee + (1.0 - alpha) * S_ee
        self._P_yy = alpha * self._P_yy + (1.0 - alpha) * S_yy
        self._P_xx_del = alpha * self._P_xx_del + (1.0 - alpha) * S_xx

        yy_total = float(np.sum(self._P_yy))
        if ref_rms < 1e-5 and yy_total < 1e-7:
            self._G_smooth = 0.7 * self._G_smooth + 0.3
            if float(np.min(self._G_smooth)) > 0.985:
                self._res_active = False
                self._last_H_min.fill(1.0 + 0.0j)
                return self._last_H_min
        else:
            self._res_active = True

        # Update conservative speaker->mic coupling gain during likely single-talk echo
        xx_total = float(np.sum(self._P_xx_del))
        dd_total = float(np.sum(self._P_dd))
        ee_total = float(np.sum(self._P_ee))
        if xx_total > 1e-6 and ref_rms > 1e-4:
            erle = dd_total / max(ee_total, 1e-10)
            ratio = dd_total / max(xx_total, 1e-10)
            if erle > 1.35:
                # Linear filter is actively cancelling echo -> reliable gain observation
                self._echo_gain_est = 0.9 * self._echo_gain_est + 0.1 * min(50.0, ratio)
            elif ratio < self._echo_gain_est:
                self._echo_gain_est = 0.95 * self._echo_gain_est + 0.05 * ratio

        # Light 3-bin frequency smoothing of spectral powers reduces musical noise
        def _smooth3(arr: np.ndarray) -> np.ndarray:
            out = arr.copy()
            out[1:-1] = 0.2 * arr[:-2] + 0.6 * arr[1:-1] + 0.2 * arr[2:]
            return out

        P_ee_s = _smooth3(self._P_ee)
        P_yy_s = _smooth3(self._P_yy)
        P_xx_s = _smooth3(self._P_xx_del)

        # Broadband double-talk indicator: compare residual error against expected residual echo
        lin_scale = float(np.clip(1.0 / max(1.0, self._converged_erle_ema * 0.5), 0.10, 0.75))
        expected_res_total = lin_scale * yy_total + 0.04 * self._echo_gain_est * xx_total
        is_double_talk = ee_total > max(2.2 * expected_res_total, 1e-6) and ee_total > 1e-6

        # Adaptive residual over-subtraction:
        # Strong suppression on single-talk residual echo, minimal attenuation during double-talk
        gamma = (2.0 if not is_double_talk else 0.45) * lin_scale
        leak_beta = (0.22 if not is_double_talk else 0.02) * lin_scale
        g_min = 0.03 if not is_double_talk else 0.35

        S_res = gamma * P_yy_s + leak_beta * self._echo_gain_est * P_xx_s
        G_raw = np.clip((P_ee_s - S_res) / (P_ee_s + 1e-9), g_min, 1.0).astype(np.float32)

        # Temporal smoothing (fast attack, smooth release)
        attack = 0.65
        release = 0.35
        coef = np.where(G_raw < self._G_smooth, attack, release).astype(np.float32)
        self._G_smooth = (1.0 - coef) * self._G_smooth + coef * G_raw

        # Convert real magnitude target G_smooth into a causal FIR <= N taps via minimum-phase
        # cepstral factorization so applying it on [prev_N, cur_N] has 0-sample latency & 0 aliasing.
        log_G = np.log(np.maximum(self._G_smooth, 1e-4)).astype(np.float32)
        cepstrum = np.fft.irfft(log_G, n=M)
        H_min = np.exp(np.fft.rfft(cepstrum * self._causal_lifter, n=M))
        h_ir = np.fft.irfft(H_min, n=M)[:N] * self._ir_taper
        self._last_H_min = np.fft.rfft(h_ir, n=M).astype(np.complex64)
        return self._last_H_min

    def _process_aligned_hop(self, mic_hop: np.ndarray, ref_hop: np.ndarray) -> np.ndarray:
        N = self.hop_size
        M = self.fft_size
        P = self.n_partitions

        # Append aligned N samples to continuous history
        self._push_history(mic_hop, ref_hop)
        self._update_decim_and_delay(mic_hop, ref_hop)

        # Compute single M-point rfft of latest reference window and push to circular FFT history
        ref_win = self._ref_hist[-M:]
        X_new = np.fft.rfft(ref_win, n=M).astype(np.complex64)
        self._ref_fft_head = (self._ref_fft_head + 1) % self._fft_hist_len
        self._ref_fft_hist[self._ref_fft_head] = X_new

        X_part = self._get_partition_spectra(self.bulk_delay_hops)

        # Foreground & background linear echo estimates
        Y_fg_spec = np.sum(self._W_fg * X_part, axis=0)
        Y_bg_spec = np.sum(self._W_bg * X_part, axis=0)
        y_fg = np.fft.irfft(Y_fg_spec, n=M)[N:].astype(np.float32)
        y_bg = np.fft.irfft(Y_bg_spec, n=M)[N:].astype(np.float32)

        e_fg = mic_hop - y_fg
        e_bg = mic_hop - y_bg

        # Energies for dual-filter arbitration & step-size control
        e_d = float(np.dot(mic_hop, mic_hop))
        e_fg_en = float(np.dot(e_fg, e_fg))
        e_bg_en = float(np.dot(e_bg, e_bg))
        y_fg_en = float(np.dot(y_fg, y_fg))
        ref_rms = float(np.sqrt(np.mean(ref_win * ref_win)))

        is_dt_hop = y_fg_en > 0.10 * e_d and e_fg_en > 1.15 * y_fg_en and e_fg_en > 0.40 * e_d

        # Arbitrate between background and foreground filters
        if not is_dt_hop and e_bg_en < 0.85 * e_fg_en and e_bg_en < 0.90 * e_d:
            self._W_fg[:] = self._W_bg
            e_fg = e_bg
            y_fg = y_bg
            e_fg_en = e_bg_en
            y_fg_en = float(np.dot(y_fg, y_fg))
            self._bg_diverge_streak = 0
        elif e_bg_en > 1.6 * max(e_d, 1e-8) and e_d > 1e-7:
            self._bg_diverge_streak += 1
            if self._bg_diverge_streak >= 4:
                self._W_bg[:] = self._W_fg
                e_bg = e_fg
                self._bg_diverge_streak = 0
        else:
            self._bg_diverge_streak = 0

        if not is_dt_hop and e_d > 1e-7 and ref_rms > 1e-4:
            cur_erle = e_d / max(e_fg_en, 1e-9)
            self._converged_erle_ema = 0.92 * self._converged_erle_ema + 0.08 * min(20.0, cur_erle)

        # Update background adaptive filter (MDF NLMS + gradient constraint) when reference is active
        Px_inst = (X_part.real * X_part.real + X_part.imag * X_part.imag).sum(axis=0)
        self._Px = np.float32(0.85) * self._Px + np.float32(0.15) * Px_inst.astype(np.float32)

        if ref_rms > 1e-5:
            # Freeze/slow adaptation during strong double-talk so near-end speech does not corrupt W
            mu = np.float32(0.01) if is_dt_hop else np.float32(0.45)

            e_bg_pad = np.empty(M, dtype=np.float32)
            e_bg_pad[:N] = 0.0
            e_bg_pad[N:] = e_bg
            E_bg_spec = np.fft.rfft(e_bg_pad, n=M)

            px_floor = np.float32(max(1e-6, float(np.mean(self._Px)) * 0.01))
            step_spec = (mu * E_bg_spec / (self._Px + px_floor)).astype(np.complex64)
            self._W_bg += np.conj(X_part) * step_spec[None, :]

            # Enforce time-domain linear convolution constraint (zero second half of each partition)
            w_t = np.fft.irfft(self._W_bg, n=M, axis=-1)
            w_t[:, N:] = 0.0
            self._W_bg = np.fft.rfft(w_t, n=M, axis=-1).astype(np.complex64)

        # Slide post-filter windows & compute causal RES filter
        self._y_fg_hist[:N] = self._y_fg_hist[N:]
        self._y_fg_hist[N:] = y_fg
        self._e_fg_hist[:N] = self._e_fg_hist[N:]
        self._e_fg_hist[N:] = e_fg

        H_min = self._compute_res_filter(
            self._mic_hist[-M:],
            self._e_fg_hist,
            self._y_fg_hist,
            X_part[0],
            ref_rms,
        )

        if not self._res_active:
            return e_fg

        out_full = np.fft.irfft(np.fft.rfft(self._e_fg_hist, n=M) * H_min, n=M)
        return out_full[N:].astype(np.float32)

    def _process_partial_tail(self, mic_tail: np.ndarray, ref_tail: np.ndarray) -> np.ndarray:
        """
        Evaluate a trailing partial slice of length r (1 <= r < N) with exact zero latency
        without shifting the N-aligned FFT cache or adaptive filter grid.
        """
        r = int(mic_tail.shape[0])
        if r <= 0:
            return np.empty(0, dtype=np.float32)

        N = self.hop_size
        M = self.fft_size
        P = self.n_partitions

        # Build a temporary lookback history ending at the partial tail's end
        self._push_history(mic_tail, ref_tail)

        # Compute partition spectra at current continuous sample tail for linear echo estimation
        X_temp = np.empty((P, self.n_bins), dtype=np.complex64)
        for p in range(P):
            offset_end = (self.bulk_delay_hops + p) * N
            end_idx = self._hist_len - offset_end
            start_idx = end_idx - M
            if start_idx >= 0:
                win = self._ref_hist[start_idx:end_idx]
            else:
                win = np.zeros(M, dtype=np.float32)
            X_temp[p] = np.fft.rfft(win, n=M)

        y_fg_full = np.fft.irfft(np.sum(self._W_fg * X_temp, axis=0), n=M)
        e_tail = mic_tail - y_fg_full[M - r :].astype(np.float32)

        if not self._res_active:
            return e_tail

        # Apply the current causal minimum-phase RES FIR to the continuous error stream ending here
        # (last M - r samples of _e_fg_hist + r new samples of e_tail)
        e_win = np.concatenate((self._e_fg_hist[r:], e_tail))
        out_full = np.fft.irfft(np.fft.rfft(e_win, n=M) * self._last_H_min, n=M)
        return out_full[M - r :].astype(np.float32)

    def process(self, mic_chunk: np.ndarray, ref_chunk: np.ndarray) -> np.ndarray:
        """
        Cancel acoustic echo from `mic_chunk` using `ref_chunk` (both 1D float32 of equal length).
        Supports arbitrary chunk lengths with zero added block latency.
        """
        mic = np.asarray(mic_chunk, dtype=np.float32).reshape(-1)
        ref = np.asarray(ref_chunk, dtype=np.float32).reshape(-1)
        total = int(mic.shape[0])
        if total == 0:
            return mic
        if ref.shape[0] != total:
            if ref.shape[0] < total:
                ref = np.pad(ref, (0, total - ref.shape[0]))
            else:
                ref = ref[:total]

        N = self.hop_size
        out = np.empty(total, dtype=np.float32)
        pos = 0

        # If a previous call left a partial frame open (`_grid_pos > 0`), finish that N-aligned frame first
        if self._grid_pos > 0:
            need = N - self._grid_pos
            if total < need:
                out[:] = self._process_partial_tail(mic, ref)
                self._grid_pos += total
                return out

            # Reconstruct the full N-aligned frame by taking the earlier `_grid_pos` samples from history
            # (undoing the temporary tail push before committing the aligned N-sample hop)
            g = self._grid_pos
            prev_mic = self._mic_hist[-g:].copy()
            prev_ref = self._ref_hist[-g:].copy()
            # Roll history back by g samples so _process_aligned_hop commits the clean N-hop once
            self._mic_hist[g:] = self._mic_hist[:-g]
            self._mic_hist[:g] = 0.0
            self._ref_hist[g:] = self._ref_hist[:-g]
            self._ref_hist[:g] = 0.0

            mic_hop = np.concatenate((prev_mic, mic[:need]))
            ref_hop = np.concatenate((prev_ref, ref[:need]))
            out_hop = self._process_aligned_hop(mic_hop, ref_hop)
            out[:need] = out_hop[g:]
            pos += need
            self._grid_pos = 0

        # Process all complete N-aligned hops
        while pos + N <= total:
            out[pos : pos + N] = self._process_aligned_hop(mic[pos : pos + N], ref[pos : pos + N])
            pos += N

        # Process any trailing partial slice (< N) at zero latency
        rem = total - pos
        if rem > 0:
            out[pos:] = self._process_partial_tail(mic[pos:], ref[pos:])
            self._grid_pos = rem

        return out
