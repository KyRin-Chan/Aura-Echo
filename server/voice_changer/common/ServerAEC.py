import logging
import math
from typing import Optional

import numpy as np
from scipy import signal as scipy_signal
from scipy.special import i0 as _scipy_i0

logger = logging.getLogger(__name__)

try:
    from pywebrtc_audio import (
        EchoCanceller as _WebRtcEchoCanceller,
        NoiseSuppressor as _WebRtcNoiseSuppressor,
    )

    _HAS_WEBRTC = True
except Exception as _import_err:  # pragma: no cover - fallback when wheel absent
    _WebRtcEchoCanceller = None
    _WebRtcNoiseSuppressor = None
    _HAS_WEBRTC = False
    logger.warning(f"[ServerAEC] pywebrtc_audio unavailable, using NumPy fallback: {_import_err}")


class NumpyCausalPolyphaseResampler:
    """
    Pure-NumPy stateful causal polyphase Kaiser-windowed sinc resampler for
    real-time CPU audio streaming (e.g. 44.1 kHz <-> 48 kHz, 96 kHz -> 48 kHz).
    """

    def __init__(
        self,
        orig_freq: int,
        new_freq: int,
        lowpass_filter_width: int = 8,
        rolloff: float = 0.945,
        beta: float = 8.0,
    ):
        self.orig_freq = max(1, int(orig_freq))
        self.new_freq = max(1, int(new_freq))

        g = math.gcd(self.orig_freq, self.new_freq)
        self.p = self.new_freq // g
        self.q = self.orig_freq // g
        self.is_bypass = self.p == self.q
        self._use_linear = False
        self._ratio = float(self.orig_freq) / float(self.new_freq)
        self._lin_phase = 0.0
        self._lin_prev = 0.0
        self._lin_has_prev = False
        self._next_pos_num = 0

        if self.is_bypass:
            self.half_width = 0
            self.kernel_len = 1
            self.kernels = np.ones((1, 1), dtype=np.float32)
            self.history = np.empty(0, dtype=np.float32)
        elif self.p > 2048:
            # Guard against huge polyphase tables on non-standard exotic sample rates
            self._use_linear = True
            self.half_width = 1
            self.kernel_len = 2
            self.kernels = np.empty((0, 0), dtype=np.float32)
            self.history = np.empty(0, dtype=np.float32)
        else:
            base_freq = min(self.p, self.q) * rolloff
            self.half_width = max(1, math.ceil(lowpass_filter_width * self.q / base_freq))
            self.kernel_len = 2 * self.half_width + 1

            idx = np.arange(self.kernel_len, dtype=np.float64)
            i0_beta = float(_scipy_i0(beta))
            scale = base_freq / self.q

            kernels = np.empty((self.p, self.kernel_len), dtype=np.float32)
            for ph in range(self.p):
                delta = (self.half_width + ph / self.p) - idx
                t = delta * scale
                t_c = np.clip(t, -lowpass_filter_width, lowpass_filter_width)
                sinc_val = np.sinc(t)
                kaiser = _scipy_i0(
                    beta * np.sqrt(np.maximum(0.0, 1.0 - (t_c / lowpass_filter_width) ** 2))
                ) / i0_beta
                mask = (np.abs(t) <= lowpass_filter_width).astype(np.float64)
                k_ph = sinc_val * kaiser * mask
                k_sum = k_ph.sum()
                if abs(k_sum) > 1e-12:
                    k_ph /= k_sum
                kernels[ph] = k_ph.astype(np.float32)

            self.kernels = np.ascontiguousarray(kernels)
            self.history = np.zeros(2 * self.half_width, dtype=np.float32)

    def reset(self) -> None:
        self._next_pos_num = 0
        self._lin_phase = 0.0
        self._lin_prev = 0.0
        self._lin_has_prev = False
        if self.history.size > 0:
            self.history.fill(0.0)

    def _process_linear(self, x_f32: np.ndarray) -> np.ndarray:
        n_src = int(x_f32.shape[0])
        if n_src == 0:
            return np.empty(0, dtype=np.float32)
        if not self._lin_has_prev:
            self._lin_prev = float(x_f32[0])
            self._lin_has_prev = True
            self._lin_phase = 0.0

        start_pos = self._lin_phase
        # Keep ext_pos <= n_src - 1e-9 so idx1 = idx0 + 1 <= n_src (never flat-clamps tail)
        if start_pos > (n_src - 1.0) - 1e-9:
            self._lin_phase = start_pos - n_src
            self._lin_prev = float(x_f32[-1])
            return np.empty(0, dtype=np.float32)

        count = int(math.floor(((n_src - 1.0) - 1e-9 - start_pos) / self._ratio)) + 1
        if count <= 0:
            self._lin_phase = start_pos - n_src
            self._lin_prev = float(x_f32[-1])
            return np.empty(0, dtype=np.float32)

        ext = np.concatenate((np.array([self._lin_prev], dtype=np.float32), x_f32))
        pos = start_pos + np.arange(count, dtype=np.float64) * self._ratio
        ext_pos = pos + 1.0
        idx0 = np.clip(np.floor(ext_pos).astype(np.int64), 0, n_src - 1)
        idx1 = idx0 + 1
        frac = (ext_pos - idx0).astype(np.float32)
        out = ext[idx0] * (1.0 - frac) + ext[idx1] * frac

        self._lin_phase = (start_pos + count * self._ratio) - n_src
        self._lin_prev = float(x_f32[-1])
        return out.astype(np.float32, copy=False)

    def process(self, x: np.ndarray) -> np.ndarray:
        x_f32 = np.asarray(x, dtype=np.float32).reshape(-1)
        n = int(x_f32.shape[0])
        if n == 0 or self.is_bypass:
            return x_f32
        if self._use_linear:
            return self._process_linear(x_f32)

        hist_len = 2 * self.half_width
        buf = np.concatenate((self.history, x_f32))
        total_num = n * self.p

        if total_num <= self._next_pos_num:
            self._next_pos_num -= total_num
            self.history[:] = buf[-hist_len:]
            return np.empty(0, dtype=np.float32)

        m = (total_num - self._next_pos_num + self.q - 1) // self.q
        k_idx = np.arange(m, dtype=np.int64)
        nums = self._next_pos_num + k_idx * self.q
        idx_k = nums // self.p
        ph_k = nums % self.p
        windows = np.lib.stride_tricks.sliding_window_view(buf, self.kernel_len)[idx_k]
        out = np.einsum("ij,ij->i", windows, self.kernels[ph_k], optimize=True).astype(
            np.float32, copy=False
        )

        self._next_pos_num = int(self._next_pos_num + m * self.q - total_num)
        self.history[:] = buf[-hist_len:]
        return out


class _WebRtcAecEngine:
    """
    Exact 10ms-framed streaming wrapper around WebRTC's standalone C++
    EchoCanceller (AEC3 + built-in HPF) and NoiseSuppressor.

    Uses standalone EchoCanceller -> NoiseSuppressor rather than AudioProcessor
    because pywebrtc-audio 0.2.0's AudioProcessor limits multi-band NS >16 kHz to
    ~6.5 dB attenuation and introduces 100 Hz modulation sidebands.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        echo_cancellation: bool = True,
        noise_suppression: bool = False,
        ns_level: int = 2,
        stream_delay_ms: int = 0,
        causal_lead_ms: int = 10,
    ):
        self.sample_rate = max(8000, int(sample_rate))
        self.echo_cancellation = bool(echo_cancellation)
        self.noise_suppression = bool(noise_suppression)
        self.ns_level = int(np.clip(int(ns_level), 0, 3))
        self.stream_delay_ms = max(0, int(stream_delay_ms))

        if self.sample_rate in (16000, 32000, 48000):
            self._proc_sr = self.sample_rate
            self._resampler_mic_in: Optional[NumpyCausalPolyphaseResampler] = None
            self._resampler_ref_in: Optional[NumpyCausalPolyphaseResampler] = None
            self._resampler_out: Optional[NumpyCausalPolyphaseResampler] = None
        else:
            self._proc_sr = 48000 if self.sample_rate >= 40000 else (32000 if self.sample_rate >= 24000 else 16000)
            self._resampler_mic_in = NumpyCausalPolyphaseResampler(self.sample_rate, self._proc_sr)
            self._resampler_ref_in = (
                NumpyCausalPolyphaseResampler(self.sample_rate, self._proc_sr)
                if self.echo_cancellation
                else None
            )
            self._resampler_out = NumpyCausalPolyphaseResampler(self._proc_sr, self.sample_rate)

        self.frame_size = self._proc_sr // 100  # Exact 10ms WebRTC quantum (160, 320, or 480)

        self._ec = (
            _WebRtcEchoCanceller(
                sample_rate=self._proc_sr,
                num_channels=1,
                stream_delay_ms=self.stream_delay_ms,
            )
            if self.echo_cancellation
            else None
        )
        self._ns = (
            _WebRtcNoiseSuppressor(
                sample_rate=self._proc_sr,
                num_channels=1,
                level=self.ns_level,
            )
            if self.noise_suppression
            else None
        )

        self._hpf_sos: Optional[np.ndarray] = None
        self._hpf_zi: Optional[np.ndarray] = None
        if not self.echo_cancellation and self.noise_suppression:
            self._hpf_sos = scipy_signal.butter(
                2, 70.0, btype="highpass", fs=self._proc_sr, output="sos"
            )
            self._hpf_zi = np.zeros((self._hpf_sos.shape[0], 2), dtype=np.float64)

        lead_frames = max(0, int(round(causal_lead_ms / 10.0))) if self.echo_cancellation else 0
        self._causal_lead_samples = lead_frames * self.frame_size
        self._init_out_pad = int(
            math.ceil(self.frame_size * self.sample_rate / float(self._proc_sr))
        ) + (64 if self._proc_sr != self.sample_rate else 0)

        self._mic_fifo = np.zeros(self._causal_lead_samples, dtype=np.float32)
        self._ref_fifo = np.empty(0, dtype=np.float32)
        self._out_fifo = np.zeros(self._init_out_pad, dtype=np.float32)

    def set_noise_suppression(self, enabled: bool, ns_level: Optional[int] = None) -> None:
        want_ns = bool(enabled)
        if ns_level is not None:
            new_level = int(np.clip(int(ns_level), 0, 3))
        else:
            new_level = self.ns_level

        if want_ns:
            if self._ns is None or new_level != self.ns_level:
                self.ns_level = new_level
                self._ns = _WebRtcNoiseSuppressor(
                    sample_rate=self._proc_sr,
                    num_channels=1,
                    level=self.ns_level,
                )
            if not self.echo_cancellation and self._hpf_sos is None:
                self._hpf_sos = scipy_signal.butter(
                    2, 70.0, btype="highpass", fs=self._proc_sr, output="sos"
                )
                self._hpf_zi = np.zeros((self._hpf_sos.shape[0], 2), dtype=np.float64)
        else:
            self.ns_level = new_level
            self._ns = None
            self._hpf_sos = None
            self._hpf_zi = None

        self.noise_suppression = want_ns

    def reset(self) -> None:
        if self._ec is not None:
            self._ec.reset()
        if self._ns is not None:
            self._ns.reset()
        if self._hpf_zi is not None:
            self._hpf_zi.fill(0.0)
        if self._resampler_mic_in is not None:
            self._resampler_mic_in.reset()
        if self._resampler_ref_in is not None:
            self._resampler_ref_in.reset()
        if self._resampler_out is not None:
            self._resampler_out.reset()
        self._mic_fifo = np.zeros(self._causal_lead_samples, dtype=np.float32)
        self._ref_fifo = np.empty(0, dtype=np.float32)
        self._out_fifo = np.zeros(self._init_out_pad, dtype=np.float32)

    def process(self, mic_chunk: np.ndarray, ref_chunk: Optional[np.ndarray]) -> np.ndarray:
        total = int(mic_chunk.shape[0])
        if total == 0:
            return mic_chunk

        mic = np.ascontiguousarray(mic_chunk, dtype=np.float32).reshape(-1)
        if self._resampler_mic_in is not None:
            mic = self._resampler_mic_in.process(mic)
        if mic.size > 0:
            self._mic_fifo = np.concatenate((self._mic_fifo, mic))

        if self.echo_cancellation:
            if ref_chunk is None:
                ref = np.zeros(total, dtype=np.float32)
            else:
                ref = np.ascontiguousarray(ref_chunk, dtype=np.float32).reshape(-1)
                if ref.shape[0] < total:
                    ref = np.pad(ref, (0, total - ref.shape[0]))
                elif ref.shape[0] > total:
                    ref = ref[:total]
            if self._resampler_ref_in is not None:
                ref = self._resampler_ref_in.process(ref)
            if ref.size > 0:
                self._ref_fifo = np.concatenate((self._ref_fifo, ref))

        fs = self.frame_size
        if self.echo_cancellation:
            n_avail = min(int(self._mic_fifo.shape[0]), int(self._ref_fifo.shape[0]))
        else:
            n_avail = int(self._mic_fifo.shape[0])
        n_frames = n_avail // fs

        if n_frames > 0:
            produced = []
            for idx in range(n_frames):
                start = idx * fs
                end = start + fs
                frame = np.ascontiguousarray(self._mic_fifo[start:end], dtype=np.float32)
                if self._ec is not None:
                    ref_frame = np.ascontiguousarray(self._ref_fifo[start:end], dtype=np.float32)
                    frame = np.asarray(self._ec.process(frame, ref_frame), dtype=np.float32)
                elif self._hpf_sos is not None:
                    hpf_out, self._hpf_zi = scipy_signal.sosfilt(
                        self._hpf_sos, frame.astype(np.float64), zi=self._hpf_zi
                    )
                    frame = np.ascontiguousarray(hpf_out, dtype=np.float32)
                if self._ns is not None:
                    frame = np.asarray(self._ns.process(frame), dtype=np.float32)
                produced.append(frame)

            consumed = n_frames * fs
            self._mic_fifo = self._mic_fifo[consumed:]
            if self.echo_cancellation:
                self._ref_fifo = self._ref_fifo[consumed:]

            block_out = np.concatenate(produced)
            if self._resampler_out is not None:
                block_out = self._resampler_out.process(block_out)
            if block_out.size > 0:
                self._out_fifo = np.concatenate((self._out_fifo, block_out))

        if self._out_fifo.shape[0] >= total:
            out = self._out_fifo[:total]
            self._out_fifo = self._out_fifo[total:]
            return out

        # Defensive pad if an unexpected tiny starvation occurred
        out = np.zeros(total, dtype=np.float32)
        have = int(self._out_fifo.shape[0])
        if have > 0:
            out[:have] = self._out_fifo
            self._out_fifo = np.empty(0, dtype=np.float32)
        return out


class _NumpyFallbackAecEngine:
    """
    Pure-NumPy fallback AEC (PBFDAF + delay estimation + RES) used only when
    pywebrtc_audio is unavailable.
    """

    def __init__(self, sample_rate: int = 48000, max_delay_ms: float = 300.0, tail_ms: float = 64.0):
        self.sample_rate = max(8000, int(sample_rate))
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

        self._hist_len = (self.max_delay_hops + P + 4) * N
        self._mic_hist = np.zeros(self._hist_len, dtype=np.float32)
        self._ref_hist = np.zeros(self._hist_len, dtype=np.float32)
        self._y_fg_hist = np.zeros(M, dtype=np.float32)
        self._e_fg_hist = np.zeros(M, dtype=np.float32)
        self._grid_pos = 0

        self._ref_fft_hist = np.zeros((self._fft_hist_len, K), dtype=np.complex64)
        self._ref_fft_head = 0

        self._W_bg = np.zeros((P, K), dtype=np.complex64)
        self._W_fg = np.zeros((P, K), dtype=np.complex64)
        self._Px = np.full(K, 1e-4, dtype=np.float32)

        self.bulk_delay_hops = 0
        self._candidate_delay_hops = 0
        self._candidate_streak = 0
        self._delay_check_counter = 0
        self._bg_diverge_streak = 0
        self._converged_erle_ema = 1.0

        self._mic_decim_buf = np.zeros(self._corr_len, dtype=np.float32)
        self._ref_decim_buf = np.zeros(self._corr_len, dtype=np.float32)
        self._mic_dc_prev = 0.0
        self._ref_dc_prev = 0.0

        self._hann = np.hanning(M).astype(np.float32)
        self._P_dd = np.zeros(K, dtype=np.float32)
        self._P_ee = np.zeros(K, dtype=np.float32)
        self._P_yy = np.zeros(K, dtype=np.float32)
        self._P_xx_del = np.zeros(K, dtype=np.float32)
        self._G_smooth = np.ones(K, dtype=np.float32)
        self._echo_gain_est = 0.25

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
        mic_hp = np.empty_like(mic_hop)
        mic_hp[0] = mic_hop[0] - self._mic_dc_prev
        mic_hp[1:] = mic_hop[1:] - mic_hop[:-1]
        self._mic_dc_prev = float(mic_hop[-1])

        ref_hp = np.empty_like(ref_hop)
        ref_hp[0] = ref_hop[0] - self._ref_dc_prev
        ref_hp[1:] = ref_hop[1:] - ref_hop[:-1]
        self._ref_dc_prev = float(ref_hop[-1])

        self._mic_decim_buf[:-step] = self._mic_decim_buf[step:]
        self._mic_decim_buf[-step:] = mic_hp.reshape(step, d).mean(axis=1)
        self._ref_decim_buf[:-step] = self._ref_decim_buf[step:]
        self._ref_decim_buf[-step:] = ref_hp.reshape(step, d).mean(axis=1)

        self._delay_check_counter += 1
        if self._delay_check_counter < 6:
            return
        self._delay_check_counter = 0

        ref_energy = float(np.dot(self._ref_decim_buf, self._ref_decim_buf))
        mic_energy = float(np.dot(self._mic_decim_buf, self._mic_decim_buf))
        if ref_energy < 1e-6 or mic_energy < 1e-8:
            return

        L = self._corr_len
        n_fft = 2 * L
        mic_recent = np.zeros(L, dtype=np.float32)
        half = L // 2
        mic_recent[half:] = self._mic_decim_buf[half:]
        mic_recent_energy = float(np.dot(mic_recent[half:], mic_recent[half:]))
        if mic_recent_energy < 1e-8:
            return

        cc = np.fft.irfft(np.fft.rfft(mic_recent, n=n_fft) * np.conj(np.fft.rfft(self._ref_decim_buf, n=n_fft)), n=n_fft)
        max_lag_decim = min(half - 1, self.max_delay_hops * step)
        if max_lag_decim <= step:
            return

        cc_lags = np.abs(cc[:max_lag_decim])
        ref_sq = self._ref_decim_buf.astype(np.float64) ** 2
        cs = np.concatenate(([0.0], np.cumsum(ref_sq)))
        lags = np.arange(max_lag_decim, dtype=np.int64)
        starts = half - lags
        ends = L - lags
        valid = starts >= 0
        norm_cc = np.zeros(max_lag_decim, dtype=np.float32)
        ref_win_e = np.maximum(cs[ends[valid]] - cs[starts[valid]], 1e-9)
        norm_cc[valid] = (cc_lags[valid] / np.sqrt(mic_recent_energy * ref_win_e)).astype(np.float32)

        best_lag_decim = int(np.argmax(norm_cc))
        best_score = float(norm_cc[best_lag_decim])
        if best_score < 0.25:
            return

        best_hop = best_lag_decim // step
        if self.bulk_delay_hops <= best_hop <= self.bulk_delay_hops + max(2, self.n_partitions - 4):
            self._candidate_streak = 0
            return

        target_delay = max(0, min(self.max_delay_hops, best_hop - 1))
        cur_start = self.bulk_delay_hops * step
        cur_end = min(max_lag_decim, (self.bulk_delay_hops + self.n_partitions) * step)
        cur_score = float(np.max(norm_cc[cur_start:cur_end])) if cur_end > cur_start else 0.0

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
        delta = new_delay_hops - self.bulk_delay_hops
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

    def _compute_res_filter(self, d_win, e_win, y_win, X_del, ref_rms) -> np.ndarray:
        M = self.fft_size
        N = self.hop_size
        D_w = np.fft.rfft(d_win * self._hann, n=M)
        E_w = np.fft.rfft(e_win * self._hann, n=M)
        Y_w = np.fft.rfft(y_win * self._hann, n=M)

        alpha = np.float32(0.55)
        self._P_dd = alpha * self._P_dd + (1.0 - alpha) * (D_w.real**2 + D_w.imag**2).astype(np.float32)
        self._P_ee = alpha * self._P_ee + (1.0 - alpha) * (E_w.real**2 + E_w.imag**2).astype(np.float32)
        self._P_yy = alpha * self._P_yy + (1.0 - alpha) * (Y_w.real**2 + Y_w.imag**2).astype(np.float32)
        self._P_xx_del = alpha * self._P_xx_del + (1.0 - alpha) * (X_del.real**2 + X_del.imag**2).astype(np.float32)

        yy_total = float(np.sum(self._P_yy))
        if ref_rms < 1e-5 and yy_total < 1e-7:
            self._G_smooth = 0.7 * self._G_smooth + 0.3
            if float(np.min(self._G_smooth)) > 0.985:
                self._res_active = False
                self._last_H_min.fill(1.0 + 0.0j)
                return self._last_H_min
        else:
            self._res_active = True

        xx_total = float(np.sum(self._P_xx_del))
        dd_total = float(np.sum(self._P_dd))
        ee_total = float(np.sum(self._P_ee))
        if xx_total > 1e-6 and ref_rms > 1e-4:
            erle = dd_total / max(ee_total, 1e-10)
            ratio = dd_total / max(xx_total, 1e-10)
            if erle > 1.35:
                self._echo_gain_est = 0.9 * self._echo_gain_est + 0.1 * min(50.0, ratio)
            elif ratio < self._echo_gain_est:
                self._echo_gain_est = 0.95 * self._echo_gain_est + 0.05 * ratio

        def _smooth3(arr: np.ndarray) -> np.ndarray:
            out = arr.copy()
            out[1:-1] = 0.2 * arr[:-2] + 0.6 * arr[1:-1] + 0.2 * arr[2:]
            return out

        P_ee_s = _smooth3(self._P_ee)
        P_yy_s = _smooth3(self._P_yy)
        P_xx_s = _smooth3(self._P_xx_del)

        lin_scale = float(np.clip(1.0 / max(1.0, self._converged_erle_ema * 0.5), 0.10, 0.75))
        expected_res_total = lin_scale * yy_total + 0.04 * self._echo_gain_est * xx_total
        is_double_talk = ee_total > max(2.2 * expected_res_total, 1e-6) and ee_total > 1e-6

        gamma = (2.0 if not is_double_talk else 0.45) * lin_scale
        leak_beta = (0.22 if not is_double_talk else 0.02) * lin_scale
        g_min = 0.03 if not is_double_talk else 0.35

        S_res = gamma * P_yy_s + leak_beta * self._echo_gain_est * P_xx_s
        G_raw = np.clip((P_ee_s - S_res) / (P_ee_s + 1e-9), g_min, 1.0).astype(np.float32)

        coef = np.where(G_raw < self._G_smooth, 0.65, 0.35).astype(np.float32)
        self._G_smooth = (1.0 - coef) * self._G_smooth + coef * G_raw

        log_G = np.log(np.maximum(self._G_smooth, 1e-4)).astype(np.float32)
        cepstrum = np.fft.irfft(log_G, n=M)
        H_min = np.exp(np.fft.rfft(cepstrum * self._causal_lifter, n=M))
        h_ir = np.fft.irfft(H_min, n=M)[:N] * self._ir_taper
        self._last_H_min = np.fft.rfft(h_ir, n=M).astype(np.complex64)
        return self._last_H_min

    def _process_aligned_hop(self, mic_hop: np.ndarray, ref_hop: np.ndarray) -> np.ndarray:
        N = self.hop_size
        M = self.fft_size

        self._push_history(mic_hop, ref_hop)
        self._update_decim_and_delay(mic_hop, ref_hop)

        ref_win = self._ref_hist[-M:]
        X_new = np.fft.rfft(ref_win, n=M).astype(np.complex64)
        self._ref_fft_head = (self._ref_fft_head + 1) % self._fft_hist_len
        self._ref_fft_hist[self._ref_fft_head] = X_new

        X_part = self._get_partition_spectra(self.bulk_delay_hops)
        y_fg = np.fft.irfft(np.sum(self._W_fg * X_part, axis=0), n=M)[N:].astype(np.float32)
        y_bg = np.fft.irfft(np.sum(self._W_bg * X_part, axis=0), n=M)[N:].astype(np.float32)

        e_fg = mic_hop - y_fg
        e_bg = mic_hop - y_bg
        e_d = float(np.dot(mic_hop, mic_hop))
        e_fg_en = float(np.dot(e_fg, e_fg))
        e_bg_en = float(np.dot(e_bg, e_bg))
        y_fg_en = float(np.dot(y_fg, y_fg))
        ref_rms = float(np.sqrt(np.mean(ref_win * ref_win)))

        is_dt_hop = y_fg_en > 0.10 * e_d and e_fg_en > 1.15 * y_fg_en and e_fg_en > 0.40 * e_d
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

        Px_inst = (X_part.real * X_part.real + X_part.imag * X_part.imag).sum(axis=0)
        self._Px = np.float32(0.85) * self._Px + np.float32(0.15) * Px_inst.astype(np.float32)

        if ref_rms > 1e-5:
            mu = np.float32(0.01) if is_dt_hop else np.float32(0.45)
            e_bg_pad = np.empty(M, dtype=np.float32)
            e_bg_pad[:N] = 0.0
            e_bg_pad[N:] = e_bg
            E_bg_spec = np.fft.rfft(e_bg_pad, n=M)
            px_floor = np.float32(max(1e-6, float(np.mean(self._Px)) * 0.01))
            step_spec = (mu * E_bg_spec / (self._Px + px_floor)).astype(np.complex64)
            self._W_bg += np.conj(X_part) * step_spec[None, :]
            w_t = np.fft.irfft(self._W_bg, n=M, axis=-1)
            w_t[:, N:] = 0.0
            self._W_bg = np.fft.rfft(w_t, n=M, axis=-1).astype(np.complex64)

        self._y_fg_hist[:N] = self._y_fg_hist[N:]
        self._y_fg_hist[N:] = y_fg
        self._e_fg_hist[:N] = self._e_fg_hist[N:]
        self._e_fg_hist[N:] = e_fg

        H_min = self._compute_res_filter(self._mic_hist[-M:], self._e_fg_hist, self._y_fg_hist, X_part[0], ref_rms)
        if not self._res_active:
            return e_fg
        out_full = np.fft.irfft(np.fft.rfft(self._e_fg_hist, n=M) * H_min, n=M)
        return out_full[N:].astype(np.float32)

    def _process_partial_tail(self, mic_tail: np.ndarray, ref_tail: np.ndarray) -> np.ndarray:
        r = int(mic_tail.shape[0])
        if r <= 0:
            return np.empty(0, dtype=np.float32)
        N = self.hop_size
        M = self.fft_size
        P = self.n_partitions
        self._push_history(mic_tail, ref_tail)
        X_temp = np.empty((P, self.n_bins), dtype=np.complex64)
        for p in range(P):
            end_idx = self._hist_len - (self.bulk_delay_hops + p) * N
            start_idx = end_idx - M
            win = self._ref_hist[start_idx:end_idx] if start_idx >= 0 else np.zeros(M, dtype=np.float32)
            X_temp[p] = np.fft.rfft(win, n=M)
        y_fg_full = np.fft.irfft(np.sum(self._W_fg * X_temp, axis=0), n=M)
        e_tail = mic_tail - y_fg_full[M - r :].astype(np.float32)
        if not self._res_active:
            return e_tail
        e_win = np.concatenate((self._e_fg_hist[r:], e_tail))
        out_full = np.fft.irfft(np.fft.rfft(e_win, n=M) * self._last_H_min, n=M)
        return out_full[M - r :].astype(np.float32)

    def process(self, mic_chunk: np.ndarray, ref_chunk: Optional[np.ndarray]) -> np.ndarray:
        mic = np.asarray(mic_chunk, dtype=np.float32).reshape(-1)
        total = int(mic.shape[0])
        if total == 0:
            return mic
        if ref_chunk is None:
            ref = np.zeros(total, dtype=np.float32)
        else:
            ref = np.asarray(ref_chunk, dtype=np.float32).reshape(-1)
            if ref.shape[0] < total:
                ref = np.pad(ref, (0, total - ref.shape[0]))
            elif ref.shape[0] > total:
                ref = ref[:total]

        N = self.hop_size
        out = np.empty(total, dtype=np.float32)
        pos = 0
        if self._grid_pos > 0:
            need = N - self._grid_pos
            if total < need:
                out[:] = self._process_partial_tail(mic, ref)
                self._grid_pos += total
                return out
            g = self._grid_pos
            prev_mic = self._mic_hist[-g:].copy()
            prev_ref = self._ref_hist[-g:].copy()
            self._mic_hist[g:] = self._mic_hist[:-g]
            self._mic_hist[:g] = 0.0
            self._ref_hist[g:] = self._ref_hist[:-g]
            self._ref_hist[:g] = 0.0
            out_hop = self._process_aligned_hop(
                np.concatenate((prev_mic, mic[:need])),
                np.concatenate((prev_ref, ref[:need])),
            )
            out[:need] = out_hop[g:]
            pos += need
            self._grid_pos = 0

        while pos + N <= total:
            out[pos : pos + N] = self._process_aligned_hop(mic[pos : pos + N], ref[pos : pos + N])
            pos += N

        rem = total - pos
        if rem > 0:
            out[pos:] = self._process_partial_tail(mic[pos:], ref[pos:])
            self._grid_pos = rem
        return out


class ServerAEC:
    """
    Real-time CPU-based Acoustic Echo Canceller & Noise Suppressor for Server Audio mode.

    Uses Google WebRTC's native C++ EchoCanceller (AEC3 + HPF) and standalone
    NoiseSuppressor via `pywebrtc-audio` with exact 10ms framing FIFOs and causal
    polyphase resampling for non-native sample rates, falling back to a pure-NumPy
    PBFDAF+RES implementation if WebRTC bindings are absent.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        echo_cancellation: bool = True,
        noise_suppression: bool = False,
        ns_level: int = 2,
        stream_delay_ms: int = 0,
        causal_lead_ms: int = 10,
        max_delay_ms: float = 300.0,
        tail_ms: float = 64.0,
    ):
        self.sample_rate = max(8000, int(sample_rate))
        self.echo_cancellation = bool(echo_cancellation)
        self.noise_suppression = bool(noise_suppression)
        self.ns_level = int(ns_level)

        if _HAS_WEBRTC:
            self._backend = "webrtc"
            self._engine = _WebRtcAecEngine(
                sample_rate=self.sample_rate,
                echo_cancellation=self.echo_cancellation,
                noise_suppression=self.noise_suppression,
                ns_level=self.ns_level,
                stream_delay_ms=stream_delay_ms,
                causal_lead_ms=causal_lead_ms,
            )
        else:
            self._backend = "numpy"
            self._engine = _NumpyFallbackAecEngine(
                sample_rate=self.sample_rate,
                max_delay_ms=max_delay_ms,
                tail_ms=tail_ms,
            )

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def bulk_delay_hops(self) -> int:
        return getattr(self._engine, "bulk_delay_hops", 0)

    def set_noise_suppression(self, enabled: bool, ns_level: Optional[int] = None) -> None:
        self.noise_suppression = bool(enabled)
        if ns_level is not None:
            self.ns_level = int(np.clip(int(ns_level), 0, 3))
        if hasattr(self._engine, "set_noise_suppression"):
            self._engine.set_noise_suppression(self.noise_suppression, ns_level=self.ns_level)

    def reset(self) -> None:
        self._engine.reset()

    def process(self, mic_chunk: np.ndarray, ref_chunk: Optional[np.ndarray] = None) -> np.ndarray:
        return self._engine.process(mic_chunk, ref_chunk)
