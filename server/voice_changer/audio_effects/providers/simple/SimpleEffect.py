import torch
import numpy as np
from typing import Dict, Any
from ...AudioEffect import AudioEffect, AudioChannel
import logging

logger = logging.getLogger(__name__)


class SimpleEffect(AudioEffect):
    """Simple audio effect implementation using basic DSP operations"""
    
    def __init__(self, effect_type: str, channel: AudioChannel, order: int = 0):
        super().__init__(effect_type, channel, order)
        self._init_effect()
    
    def _init_effect(self):
        """Initialize internal state for the effect"""
        if self.effect_type == "lowpass":
            self._lowpass_zi = None
        elif self.effect_type == "highpass":
            self._highpass_zi = None
        elif self.effect_type == "delay":
            max_delay_samples = int(48000 * 2)  # 2 seconds at 48kHz
            self._delay_buffer = torch.zeros(max_delay_samples)
            self._delay_index = 0
        elif self.effect_type == "exciter":
            self._exciter_hp_zi = None
            self._exciter_clean_zi = None
        elif self.effect_type == "deesser":
            self._deesser_lp_zi = None
            self._deesser_hp_zi = None
            self._deesser_env = 0.0
        elif self.effect_type == "resonanceSuppressor":
            pass
        elif self.effect_type == "noiseExpander":
            self._expander_env = 0.0

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        super().set_parameters(parameters)
        self._init_effect()  # Reinitialize if needed
    
    def process(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if not self.enabled:
            return audio
        
        try:
            if self.effect_type == "gain":
                return self._apply_gain(audio)
            elif self.effect_type == "lowpass":
                return self._apply_lowpass(audio, sample_rate)
            elif self.effect_type == "highpass":
                return self._apply_highpass(audio, sample_rate)
            elif self.effect_type == "delay":
                return self._apply_delay(audio, sample_rate)
            elif self.effect_type == "exciter":
                return self._apply_exciter(audio, sample_rate)
            elif self.effect_type == "deesser":
                return self._apply_deesser(audio, sample_rate)
            elif self.effect_type == "resonanceSuppressor":
                return self._apply_resonance_suppressor(audio, sample_rate)
            elif self.effect_type == "noiseExpander":
                return self._apply_noise_expander(audio, sample_rate)
            else:
                return audio
        except Exception as e:
            logger.error(f"Error in {self.effect_type} effect: {e}")
            return audio
    
    def _apply_gain(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply simple gain adjustment"""
        gain_db = self.parameters.get("gain", 0.0)
        gain_linear = 10 ** (gain_db / 20.0)
        return audio * gain_linear
    
    def _apply_lowpass(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Apply optimized Butterworth low-pass filter using SciPy"""
        cutoff = self.parameters.get("cutoff", 1000.0)
        cutoff = min(max(cutoff, 20.0), sample_rate * 0.48)
        import scipy.signal
        
        audio_np = audio.detach().cpu().numpy().copy().astype(np.float32)
        sos = scipy.signal.butter(2, cutoff, 'lowpass', fs=sample_rate, output='sos')
        
        if self._lowpass_zi is None or self._lowpass_zi.shape[0] != sos.shape[0]:
            self._lowpass_zi = scipy.signal.sosfilt_zi(sos) * 0.0
            
        processed, self._lowpass_zi = scipy.signal.sosfilt(sos, audio_np, zi=self._lowpass_zi)
        processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.tensor(processed, dtype=audio.dtype, device=audio.device)
    
    def _apply_highpass(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Apply optimized Butterworth high-pass filter using SciPy"""
        cutoff = self.parameters.get("cutoff", 1000.0)
        cutoff = min(max(cutoff, 20.0), sample_rate * 0.48)
        import scipy.signal
        
        audio_np = audio.detach().cpu().numpy().copy().astype(np.float32)
        sos = scipy.signal.butter(2, cutoff, 'highpass', fs=sample_rate, output='sos')
        
        if self._highpass_zi is None or self._highpass_zi.shape[0] != sos.shape[0]:
            self._highpass_zi = scipy.signal.sosfilt_zi(sos) * 0.0
            
        processed, self._highpass_zi = scipy.signal.sosfilt(sos, audio_np, zi=self._highpass_zi)
        processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.tensor(processed, dtype=audio.dtype, device=audio.device)
    
    def _apply_delay(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Apply simple delay effect (Vectorized and corrected)"""
        delay_time = self.parameters.get("delayTime", 300.0) / 1000.0  # Convert ms to seconds
        feedback = self.parameters.get("feedback", 0.3)
        wet_level = self.parameters.get("wetLevel", 0.3)
        
        delay_samples = int(delay_time * sample_rate)
        delay_samples = max(1, delay_samples)
        
        # Ensure delay buffer is allocated with the correct parameter size on the same device
        if not hasattr(self, "_delay_buffer") or self._delay_buffer is None or len(self._delay_buffer) != delay_samples:
            self._delay_buffer = torch.zeros(delay_samples, device=audio.device, dtype=audio.dtype)
            self._delay_index = 0
            
        if self._delay_buffer.device != audio.device:
            self._delay_buffer = self._delay_buffer.to(audio.device)
            
        N = len(audio)
        output = audio.clone()
        
        # Fast path: Vectorized block processing
        if delay_samples >= N:
            indices = (self._delay_index + torch.arange(N, device=audio.device)) % delay_samples
            delayed_samples = self._delay_buffer[indices]
            
            output = audio + wet_level * delayed_samples
            self._delay_buffer[indices] = audio + feedback * delayed_samples
            self._delay_index = (self._delay_index + N) % delay_samples
        else:
            # Fallback for extremely short delays (less than block size)
            for i in range(N):
                delayed_sample = self._delay_buffer[self._delay_index]
                output[i] = audio[i] + wet_level * delayed_sample
                self._delay_buffer[self._delay_index] = audio[i] + feedback * delayed_sample
                self._delay_index = (self._delay_index + 1) % delay_samples
                
        return output

    def _apply_exciter(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Apply Harmonic Exciter using asymmetric warm tube-like distortion on high band"""
        frequency = self.parameters.get("frequency", 3000.0)
        drive = self.parameters.get("drive", 1.5)
        mix = self.parameters.get("mix", 0.15)
        
        # Clamp cutoff to safe range [20, sample_rate * 0.48]
        frequency = min(max(frequency, 20.0), sample_rate * 0.48)
        
        audio_np = audio.detach().cpu().numpy().copy().astype(np.float32)
        import scipy.signal
        
        # 1. Highpass filter to isolate high frequencies
        sos_hp = scipy.signal.butter(2, frequency, 'highpass', fs=sample_rate, output='sos')
        if self._exciter_hp_zi is None or self._exciter_hp_zi.shape[0] != sos_hp.shape[0]:
            self._exciter_hp_zi = scipy.signal.sosfilt_zi(sos_hp) * 0.0
            
        high_band, self._exciter_hp_zi = scipy.signal.sosfilt(sos_hp, audio_np, zi=self._exciter_hp_zi)
        
        # 2. Asymmetric saturation: emulates vacuum tube warm saturation (adds both even and odd harmonics)
        scaled = high_band * drive
        harmonics = np.where(scaled > 0, np.tanh(scaled), 0.6 * np.tanh(scaled * 1.66))
        
        # 3. Secondary highpass filter to clean up low-frequency modulation harmonics
        clean_freq = min(frequency * 1.1, sample_rate * 0.45)
        sos_clean = scipy.signal.butter(2, clean_freq, 'highpass', fs=sample_rate, output='sos')
        if self._exciter_clean_zi is None or self._exciter_clean_zi.shape[0] != sos_clean.shape[0]:
            self._exciter_clean_zi = scipy.signal.sosfilt_zi(sos_clean) * 0.0
            
        clean_harmonics, self._exciter_clean_zi = scipy.signal.sosfilt(sos_clean, harmonics, zi=self._exciter_clean_zi)
        
        # 4. Mix back
        processed = audio_np + mix * clean_harmonics
        processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.tensor(processed, dtype=audio.dtype, device=audio.device)

    def _apply_deesser(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Apply Dynamic Peaking EQ De-esser (only ducks the sibilance frequency, preserving air)"""
        cutoff = self.parameters.get("cutoff", 5000.0)
        threshold_db = self.parameters.get("threshold", -30.0)
        ratio = self.parameters.get("ratio", 4.0)
        
        # Clamp cutoff to safe range [20, sample_rate * 0.48]
        cutoff = min(max(cutoff, 20.0), sample_rate * 0.48)
        
        audio_np = audio.detach().cpu().numpy().copy().astype(np.float32)
        import scipy.signal
        
        # 1. Highpass filter to isolate sibilant band
        sos_hp = scipy.signal.butter(2, cutoff, 'highpass', fs=sample_rate, output='sos')
        if not hasattr(self, "_deesser_hp_zi") or self._deesser_hp_zi is None or self._deesser_hp_zi.shape[0] != sos_hp.shape[0]:
            self._deesser_hp_zi = scipy.signal.sosfilt_zi(sos_hp) * 0.0
            
        high_band, self._deesser_hp_zi = scipy.signal.sosfilt(sos_hp, audio_np, zi=self._deesser_hp_zi)
        
        # 2. Vectorized envelope follower on high band (extremely fast 15ms tracker using compiled C lfilter)
        tc = 0.015
        g = np.exp(-1.0 / (sample_rate * tc))
        b_env = [1.0 - g]
        a_env = [1.0, -g]
        
        if not hasattr(self, "_deesser_env_state") or self._deesser_env_state is None:
            self._deesser_env_state = np.zeros(1)
            
        rectified = np.abs(high_band)
        envelope, self._deesser_env_state = scipy.signal.lfilter(b_env, a_env, rectified, zi=self._deesser_env_state)
        
        # 3. Calculate dynamic gain reduction
        envelope_db = 20 * np.log10(envelope + 1e-6)
        gain_reduction_db = np.zeros_like(envelope_db)
        over_threshold = envelope_db > threshold_db
        gain_reduction_db[over_threshold] = (threshold_db - envelope_db[over_threshold]) * (1.0 - 1.0 / ratio)
        
        # Target attenuation is the peak reduction required in this block, capped at -20dB
        min_gain_db = np.min(gain_reduction_db)
        min_gain_db = max(min_gain_db, -20.0)
        
        # 4. Apply Dynamic Peaking EQ (always run biquad to maintain state continuity and avoid clicks)
        w0 = 2 * np.pi * cutoff / sample_rate
        q_factor = 1.2
        alpha = np.sin(w0) / (2.0 * q_factor)
        a_val = 10 ** (min_gain_db / 40.0)  # min_gain_db is negative (cut) or 0 (flat)
        
        b0 = 1.0 + alpha * a_val
        b1 = -2.0 * np.cos(w0)
        b2 = 1.0 - alpha * a_val
        a0 = 1.0 + alpha / a_val
        a1 = -2.0 * np.cos(w0)
        a2 = 1.0 - alpha / a_val
        
        b = [b0 / a0, b1 / a0, b2 / a0]
        a = [1.0, a1 / a0, a2 / a0]
        
        if not hasattr(self, "_deesser_filt_state") or self._deesser_filt_state is None:
            self._deesser_filt_state = np.zeros(2)
            
        processed, self._deesser_filt_state = scipy.signal.lfilter(b, a, audio_np, zi=self._deesser_filt_state)
        processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.tensor(processed, dtype=audio.dtype, device=audio.device)

    def _apply_resonance_suppressor(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Apply Dynamic Resonance Suppressor with temporal spectral gain smoothing to eliminate clicks"""
        threshold = self.parameters.get("threshold", 6.0)
        amount = self.parameters.get("amount", 0.6)
        freq_range = self.parameters.get("frequencyRange", "mid-high")
        
        audio_np = audio.detach().cpu().numpy().copy().astype(np.float32)
        n_samples = len(audio_np)
        
        if n_samples < 64:
            return audio
            
        # 1. FFT
        fft_coeffs = np.fft.rfft(audio_np)
        mag = np.abs(fft_coeffs)
        
        # 2. Smooth magnitude spectrum using NumPy convolution (dependency-free moving average)
        window_size = 25
        pad_width = window_size // 2
        padded = np.pad(mag, pad_width, mode='edge')
        smoothed = np.convolve(padded, np.ones(window_size) / window_size, mode='valid')
        smoothed = np.maximum(smoothed, 1e-5) # Prevent division by zero
        
        # 3. Detect resonance peaks (ratio)
        ratio = mag / smoothed
        
        n_freqs = len(fft_coeffs)
        freqs = np.fft.rfftfreq(n_samples, d=1.0/sample_rate)
        
        # Limit frequency range of suppression
        mask = np.ones(n_freqs, dtype=bool)
        if freq_range == "mid-high":
            mask = (freqs >= 1000.0) & (freqs <= 8000.0)
        elif freq_range == "high":
            mask = freqs >= 4000.0
            
        # 4. Calculate dynamic gains for all bins
        gains = np.ones(n_freqs, dtype=np.float32)
        exceeded = (ratio > threshold) & mask
        if np.any(exceeded):
            reduction = threshold / ratio[exceeded]
            gains[exceeded] = 1.0 - amount * (1.0 - reduction)
            
        # Temporal smoothing of the spectral gain mask to prevent block boundary click artifacts
        if not hasattr(self, "_res_prev_gains") or self._res_prev_gains is None or len(self._res_prev_gains) != n_freqs:
            self._res_prev_gains = gains
        else:
            smoothed_gains = 0.6 * gains + 0.4 * self._res_prev_gains
            self._res_prev_gains = smoothed_gains
            gains = smoothed_gains
            
        fft_coeffs *= gains
        
        # 5. Inverse FFT
        processed = np.fft.irfft(fft_coeffs, n=n_samples)
        processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.tensor(processed, dtype=audio.dtype, device=audio.device)

    def _apply_noise_expander(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Apply smooth downward expander for natural noise reduction"""
        threshold_db = self.parameters.get("threshold", -35.0)
        ratio = self.parameters.get("ratio", 2.0)
        attack_ms = self.parameters.get("attack", 5.0)
        release_ms = self.parameters.get("release", 150.0)
        
        audio_np = audio.detach().cpu().numpy().copy().astype(np.float32)
        
        # Envelope follower coefficients
        g_attack = np.exp(-1.0 / (sample_rate * (attack_ms / 1000.0)))
        g_release = np.exp(-1.0 / (sample_rate * (release_ms / 1000.0)))
        
        rectified = np.abs(audio_np)
        envelope = np.zeros_like(audio_np)
        
        curr_env = getattr(self, "_expander_env", 0.0)
        for i in range(len(rectified)):
            val = rectified[i]
            if val > curr_env:
                curr_env = g_attack * curr_env + (1.0 - g_attack) * val
            else:
                curr_env = g_release * curr_env + (1.0 - g_release) * val
            envelope[i] = curr_env
        self._expander_env = curr_env
        
        # Convert envelope to dB
        envelope_db = 20 * np.log10(envelope + 1e-6)
        
        # Downward expansion formula
        gain_db = np.zeros_like(envelope_db)
        below_threshold = envelope_db < threshold_db
        gain_db[below_threshold] = (envelope_db[below_threshold] - threshold_db) * (ratio - 1.0)
        
        # Clamp gain to a minimum floor of -60dB (no infinite silence artifacts)
        gain_db = np.maximum(gain_db, -60.0)
        
        gain_linear = 10 ** (gain_db / 20.0)
        processed = audio_np * gain_linear
        processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.tensor(processed, dtype=audio.dtype, device=audio.device)