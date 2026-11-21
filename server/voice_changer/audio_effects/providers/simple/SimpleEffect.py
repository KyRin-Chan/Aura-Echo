import math
import logging
from typing import Dict, Any, Optional, Tuple

import numpy as np
import scipy.signal
import torch

from ...AudioEffect import AudioEffect, AudioChannel

logger = logging.getLogger(__name__)


class SimpleEffect(AudioEffect):
    """Simple audio effect implementation using stateful NumPy/SciPy DSP operations."""

    def __init__(self, effect_type: str, channel: AudioChannel, order: int = 0):
        super().__init__(effect_type, channel, order)
        self._sos_cache: Dict[Tuple[Any, ...], np.ndarray] = {}
        self._freqs_cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._init_effect()

    def _init_effect(self):
        """Initialize internal state for the effect."""
        if self.effect_type == "lowpass":
            self._lowpass_zi = None
        elif self.effect_type == "highpass":
            self._highpass_zi = None
        elif self.effect_type == "delay":
            self._delay_buffer_np: Optional[np.ndarray] = None
            self._delay_index = 0
        elif self.effect_type == "exciter":
            self._exciter_hp_zi = None
            self._exciter_clean_zi = None
        elif self.effect_type == "deesser":
            self._deesser_lp_zi = None
            self._deesser_hp_zi = None
            self._deesser_env_state = None
            self._deesser_filt_state = None
        elif self.effect_type == "resonanceSuppressor":
            self._res_n_fft = 1024
            self._res_hop = 512
            win_hann = 0.5 * (
                1.0 - np.cos(2.0 * np.pi * np.arange(self._res_n_fft, dtype=np.float32) / self._res_n_fft)
            )
            self._res_win = np.sqrt(win_hann).astype(np.float32)
            self._res_in_buf = np.zeros(self._res_hop, dtype=np.float32)
            self._res_ola_tail = np.zeros(self._res_hop, dtype=np.float32)
            self._res_out_buf = np.zeros(self._res_hop, dtype=np.float32)
            self._res_prev_gains = None
        elif self.effect_type == "noiseExpander":
            self._expander_env = 0.0
        elif self.effect_type == "autoTune":
            pass

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        super().set_parameters(parameters)
        # Preserve runtime DSP state across parameter updates; clear coefficient cache
        self._sos_cache.clear()

    def _get_butter_sos(self, tag: str, cutoff: float, btype: str, sample_rate: int) -> np.ndarray:
        key = (tag, round(float(cutoff), 2), btype, int(sample_rate))
        sos = self._sos_cache.get(key)
        if sos is None:
            sos = scipy.signal.butter(2, cutoff, btype, fs=sample_rate, output="sos")
            self._sos_cache[key] = sos
        return sos

    def process_numpy(self, audio_np: np.ndarray, sample_rate: int) -> np.ndarray:
        if not self.enabled or audio_np.size == 0:
            return audio_np

        arr = np.ascontiguousarray(audio_np, dtype=np.float32)
        try:
            if self.effect_type == "gain":
                return self._apply_gain_np(arr)
            elif self.effect_type == "lowpass":
                return self._apply_lowpass_np(arr, sample_rate)
            elif self.effect_type == "highpass":
                return self._apply_highpass_np(arr, sample_rate)
            elif self.effect_type == "delay":
                return self._apply_delay_np(arr, sample_rate)
            elif self.effect_type == "exciter":
                return self._apply_exciter_np(arr, sample_rate)
            elif self.effect_type == "deesser":
                return self._apply_deesser_np(arr, sample_rate)
            elif self.effect_type == "resonanceSuppressor":
                return self._apply_resonance_suppressor_np(arr, sample_rate)
            elif self.effect_type == "noiseExpander":
                return self._apply_noise_expander_np(arr, sample_rate)
            elif self.effect_type == "autoTune":
                return arr
            else:
                return arr
        except Exception as e:
            logger.error(f"Error in {self.effect_type} effect: {e}")
            return audio_np

    def process(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if not self.enabled:
            return audio
        if self.effect_type == "gain":
            gain_db = self.parameters.get("gain", 0.0)
            return audio * (10.0 ** (gain_db / 20.0))
        if self.effect_type == "autoTune":
            return audio

        audio_np = audio.detach().cpu().numpy()
        processed = self.process_numpy(audio_np, sample_rate)
        return torch.as_tensor(processed, dtype=audio.dtype, device=audio.device)

    def _apply_gain_np(self, audio_np: np.ndarray) -> np.ndarray:
        gain_db = self.parameters.get("gain", 0.0)
        gain_linear = np.float32(10.0 ** (gain_db / 20.0))
        return audio_np * gain_linear

    def _apply_lowpass_np(self, audio_np: np.ndarray, sample_rate: int) -> np.ndarray:
        cutoff = min(max(float(self.parameters.get("cutoff", 1000.0)), 20.0), sample_rate * 0.48)
        sos = self._get_butter_sos("lp", cutoff, "lowpass", sample_rate)

        if self._lowpass_zi is None or self._lowpass_zi.shape[0] != sos.shape[0]:
            self._lowpass_zi = scipy.signal.sosfilt_zi(sos) * 0.0

        processed, self._lowpass_zi = scipy.signal.sosfilt(sos, audio_np, zi=self._lowpass_zi)
        processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)
        return processed.astype(np.float32, copy=False)

    def _apply_highpass_np(self, audio_np: np.ndarray, sample_rate: int) -> np.ndarray:
        cutoff = min(max(float(self.parameters.get("cutoff", 1000.0)), 20.0), sample_rate * 0.48)
        sos = self._get_butter_sos("hp", cutoff, "highpass", sample_rate)

        if self._highpass_zi is None or self._highpass_zi.shape[0] != sos.shape[0]:
            self._highpass_zi = scipy.signal.sosfilt_zi(sos) * 0.0

        processed, self._highpass_zi = scipy.signal.sosfilt(sos, audio_np, zi=self._highpass_zi)
        processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)
        return processed.astype(np.float32, copy=False)

    def _apply_delay_np(self, audio_np: np.ndarray, sample_rate: int) -> np.ndarray:
        delay_time = float(self.parameters.get("delayTime", 300.0)) / 1000.0
        feedback = float(self.parameters.get("feedback", 0.3))
        wet_level = float(self.parameters.get("wetLevel", 0.3))

        delay_samples = max(1, int(delay_time * sample_rate))
        if self._delay_buffer_np is None or len(self._delay_buffer_np) != delay_samples:
            self._delay_buffer_np = np.zeros(delay_samples, dtype=np.float32)
            self._delay_index = 0

        buf = self._delay_buffer_np
        n = len(audio_np)
        output = np.empty(n, dtype=np.float32)
        pos = 0

        # Vectorized chunked ring-buffer read/write (supports any delay_samples >= 1)
        while pos < n:
            step = min(n - pos, delay_samples)
            idx = self._delay_index
            first = min(step, delay_samples - idx)

            delayed = np.empty(step, dtype=np.float32)
            delayed[:first] = buf[idx : idx + first]
            if first < step:
                delayed[first:] = buf[: step - first]

            seg = audio_np[pos : pos + step]
            output[pos : pos + step] = seg + wet_level * delayed
            fb_write = seg + feedback * delayed

            buf[idx : idx + first] = fb_write[:first]
            if first < step:
                buf[: step - first] = fb_write[first:]

            self._delay_index = (idx + step) % delay_samples
            pos += step

        return output

    def _apply_exciter_np(self, audio_np: np.ndarray, sample_rate: int) -> np.ndarray:
        frequency = min(max(float(self.parameters.get("frequency", 3000.0)), 20.0), sample_rate * 0.48)
        drive = float(self.parameters.get("drive", 1.5))
        mix = float(self.parameters.get("mix", 0.15))

        sos_hp = self._get_butter_sos("exc_hp", frequency, "highpass", sample_rate)
        if self._exciter_hp_zi is None or self._exciter_hp_zi.shape[0] != sos_hp.shape[0]:
            self._exciter_hp_zi = scipy.signal.sosfilt_zi(sos_hp) * 0.0

        high_band, self._exciter_hp_zi = scipy.signal.sosfilt(sos_hp, audio_np, zi=self._exciter_hp_zi)

        scaled = high_band * drive
        harmonics = np.where(scaled > 0, np.tanh(scaled), 0.6 * np.tanh(scaled * 1.66))

        clean_freq = min(frequency * 1.1, sample_rate * 0.45)
        sos_clean = self._get_butter_sos("exc_clean", clean_freq, "highpass", sample_rate)
        if self._exciter_clean_zi is None or self._exciter_clean_zi.shape[0] != sos_clean.shape[0]:
            self._exciter_clean_zi = scipy.signal.sosfilt_zi(sos_clean) * 0.0

        clean_harmonics, self._exciter_clean_zi = scipy.signal.sosfilt(
            sos_clean, harmonics, zi=self._exciter_clean_zi
        )

        processed = audio_np + mix * clean_harmonics
        processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)
        return processed.astype(np.float32, copy=False)

    def _apply_deesser_np(self, audio_np: np.ndarray, sample_rate: int) -> np.ndarray:
        cutoff = min(max(float(self.parameters.get("cutoff", 5000.0)), 20.0), sample_rate * 0.48)
        threshold_db = float(self.parameters.get("threshold", -30.0))
        ratio = float(self.parameters.get("ratio", 4.0))

        sos_hp = self._get_butter_sos("deess_hp", cutoff, "highpass", sample_rate)
        if self._deesser_hp_zi is None or self._deesser_hp_zi.shape[0] != sos_hp.shape[0]:
            self._deesser_hp_zi = scipy.signal.sosfilt_zi(sos_hp) * 0.0

        high_band, self._deesser_hp_zi = scipy.signal.sosfilt(sos_hp, audio_np, zi=self._deesser_hp_zi)

        tc = 0.015
        g = math.exp(-1.0 / (sample_rate * tc))
        b_env = [1.0 - g]
        a_env = [1.0, -g]

        if self._deesser_env_state is None:
            self._deesser_env_state = np.zeros(1)

        rectified = np.abs(high_band)
        envelope, self._deesser_env_state = scipy.signal.lfilter(
            b_env, a_env, rectified, zi=self._deesser_env_state
        )

        envelope_db = 20.0 * np.log10(envelope + 1e-6)
        gain_reduction_db = np.zeros_like(envelope_db)
        over_threshold = envelope_db > threshold_db
        if np.any(over_threshold):
            gain_reduction_db[over_threshold] = (threshold_db - envelope_db[over_threshold]) * (
                1.0 - 1.0 / ratio
            )

        min_gain_db = max(float(np.min(gain_reduction_db)), -20.0)

        w0 = 2.0 * math.pi * cutoff / sample_rate
        q_factor = 1.2
        alpha = math.sin(w0) / (2.0 * q_factor)
        a_val = 10.0 ** (min_gain_db / 40.0)
        cos_w0 = math.cos(w0)

        b0 = 1.0 + alpha * a_val
        b1 = -2.0 * cos_w0
        b2 = 1.0 - alpha * a_val
        a0 = 1.0 + alpha / a_val
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha / a_val

        b = [b0 / a0, b1 / a0, b2 / a0]
        a = [1.0, a1 / a0, a2 / a0]

        if self._deesser_filt_state is None:
            self._deesser_filt_state = np.zeros(2)

        processed, self._deesser_filt_state = scipy.signal.lfilter(
            b, a, audio_np, zi=self._deesser_filt_state
        )
        processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)
        return processed.astype(np.float32, copy=False)

    def _apply_resonance_suppressor_np(self, audio_np: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply Dynamic Resonance Suppressor using stateful 50%-overlap Hann-windowed OLA."""
        threshold = float(self.parameters.get("threshold", 6.0))
        amount = float(self.parameters.get("amount", 0.6))
        freq_range = self.parameters.get("frequencyRange", "mid-high")

        n_in = len(audio_np)
        n_fft = self._res_n_fft
        hop = self._res_hop
        win = self._res_win

        freq_key = (n_fft, int(sample_rate))
        freqs = self._freqs_cache.get(freq_key)
        if freqs is None:
            freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
            self._freqs_cache[freq_key] = freqs

        n_freqs = len(freqs)
        if freq_range == "mid-high":
            mask = (freqs >= 1000.0) & (freqs <= 8000.0)
        elif freq_range == "high":
            mask = freqs >= 4000.0
        else:
            mask = np.ones(n_freqs, dtype=bool)

        ma_kernel = np.ones(25, dtype=np.float32) / 25.0
        pad_width = 12

        self._res_in_buf = np.concatenate([self._res_in_buf, audio_np])
        new_out_chunks = []

        while len(self._res_in_buf) >= n_fft:
            frame = self._res_in_buf[:n_fft] * win
            self._res_in_buf = self._res_in_buf[hop:]

            fft_coeffs = np.fft.rfft(frame)
            mag = np.abs(fft_coeffs)

            padded = np.pad(mag, pad_width, mode="edge")
            smoothed = np.convolve(padded, ma_kernel, mode="valid")
            smoothed = np.maximum(smoothed, 1e-5)

            ratio = mag / smoothed
            gains = np.ones(n_freqs, dtype=np.float32)
            exceeded = (ratio > threshold) & mask
            if np.any(exceeded):
                reduction = threshold / ratio[exceeded]
                gains[exceeded] = 1.0 - amount * (1.0 - reduction)

            if self._res_prev_gains is None or len(self._res_prev_gains) != n_freqs:
                self._res_prev_gains = gains
            else:
                gains = 0.6 * gains + 0.4 * self._res_prev_gains
                self._res_prev_gains = gains

            fft_coeffs *= gains
            synth = np.fft.irfft(fft_coeffs, n=n_fft).astype(np.float32) * win

            hop_out = self._res_ola_tail + synth[:hop]
            self._res_ola_tail = synth[hop:].copy()
            new_out_chunks.append(hop_out)

        if new_out_chunks:
            self._res_out_buf = np.concatenate([self._res_out_buf] + new_out_chunks)

        out = self._res_out_buf[:n_in].copy()
        self._res_out_buf = self._res_out_buf[n_in:]
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    def _apply_noise_expander_np(self, audio_np: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply smooth downward expander using 1ms sub-block envelope tracking + linear interpolation."""
        threshold_db = float(self.parameters.get("threshold", -35.0))
        ratio = float(self.parameters.get("ratio", 2.0))
        attack_ms = max(float(self.parameters.get("attack", 5.0)), 0.1)
        release_ms = max(float(self.parameters.get("release", 150.0)), 1.0)

        n = len(audio_np)
        sub_len = max(1, int(round(sample_rate * 0.001)))
        abs_x = np.abs(audio_np)

        full_subs = n // sub_len
        sub_peaks = []
        if full_subs > 0:
            sub_peaks.append(abs_x[: full_subs * sub_len].reshape(full_subs, sub_len).max(axis=1))
        if full_subs * sub_len < n:
            sub_peaks.append(np.array([abs_x[full_subs * sub_len :].max()], dtype=np.float32))
        peaks = np.concatenate(sub_peaks) if len(sub_peaks) > 1 else sub_peaks[0]

        dt_sub = sub_len / float(sample_rate)
        g_attack = math.exp(-dt_sub / (attack_ms / 1000.0))
        g_release = math.exp(-dt_sub / (release_ms / 1000.0))

        curr_env = float(getattr(self, "_expander_env", 0.0))
        n_subs = len(peaks)
        env_knots = np.empty(n_subs + 1, dtype=np.float32)
        env_knots[0] = curr_env
        for i in range(n_subs):
            val = float(peaks[i])
            if val > curr_env:
                curr_env = g_attack * curr_env + (1.0 - g_attack) * val
            else:
                curr_env = g_release * curr_env + (1.0 - g_release) * val
            env_knots[i + 1] = curr_env
        self._expander_env = curr_env

        # Build sample positions for knot interpolation so sample 0 starts from previous block's tail
        knot_x = np.empty(n_subs + 1, dtype=np.float32)
        knot_x[0] = 0.0
        for i in range(full_subs):
            knot_x[i + 1] = float((i + 1) * sub_len)
        if full_subs * sub_len < n:
            knot_x[-1] = float(n)

        sample_x = np.arange(n, dtype=np.float32)
        envelope = np.interp(sample_x, knot_x, env_knots).astype(np.float32)

        envelope_db = 20.0 * np.log10(envelope + 1e-6)
        gain_db = np.zeros_like(envelope_db)
        below = envelope_db < threshold_db
        if np.any(below):
            gain_db[below] = (envelope_db[below] - threshold_db) * (ratio - 1.0)
        gain_db = np.maximum(gain_db, -60.0)

        gain_linear = (10.0 ** (gain_db / 20.0)).astype(np.float32)
        processed = audio_np * gain_linear
        processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)
        return processed.astype(np.float32, copy=False)
