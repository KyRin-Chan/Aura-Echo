import torch
import numpy as np
from typing import Dict, Any
from ...AudioEffect import AudioEffect, AudioChannel
import logging

logger = logging.getLogger(__name__)

try:
    from pedalboard import (
        Reverb, Compressor, Delay, LowpassFilter, Pedalboard,
        LowShelfFilter, HighShelfFilter, PeakFilter,
        Chorus, Distortion, NoiseGate, Gain, HighpassFilter,
        Bitcrush, Clipping, Limiter, Invert, LadderFilter,
        Convolution, MP3Compressor, GSMFullRateCompressor
    )
    PEDALBOARD_AVAILABLE = True
except ImportError:
    PEDALBOARD_AVAILABLE = False
    logger.warning("Pedalboard not available, pedalboard effects will be disabled")


def _generate_synthetic_ir(ir_type: str, room_size: float, damping: float, sample_rate: int = 48000) -> np.ndarray:
    """
    Generates a synthetic impulse response NumPy array.
    """
    # 1. Determine duration (rt60) based on type and room_size
    if ir_type == "hall":
        rt60 = 1.5 + room_size * 2.5  # 1.5 to 4.0 seconds
    elif ir_type == "room":
        rt60 = 0.2 + room_size * 0.8  # 0.2 to 1.0 seconds
    elif ir_type == "plate":
        rt60 = 0.8 + room_size * 1.7  # 0.8 to 2.5 seconds
    elif ir_type == "spring":
        rt60 = 1.0 + room_size * 1.5  # 1.0 to 2.5 seconds
    elif ir_type == "cathedral":
        rt60 = 3.0 + room_size * 5.0  # 3.0 to 8.0 seconds
    else:
        rt60 = 1.5
        
    num_samples = int(rt60 * sample_rate)
    t = np.linspace(0, rt60, num_samples, endpoint=False)
    noise = np.random.normal(0, 1.0, num_samples)
    
    # Create exponential decay
    decay_rate = 6.91 / rt60
    
    import scipy.signal
    try:
        # Multi-band damping: split into low, mid, high bands
        # and apply faster decay to higher frequencies
        sos_low = scipy.signal.butter(2, 500, 'lp', fs=sample_rate, output='sos')
        low_band = scipy.signal.sosfilt(sos_low, noise)
        
        sos_mid = scipy.signal.butter(2, [500, 3000], 'bp', fs=sample_rate, output='sos')
        mid_band = scipy.signal.sosfilt(sos_mid, noise)
        
        sos_high = scipy.signal.butter(2, 3000, 'hp', fs=sample_rate, output='sos')
        high_band = scipy.signal.sosfilt(sos_high, noise)
        
        decay_low = decay_rate
        decay_mid = decay_rate * (1.0 + damping)
        decay_high = decay_rate * (1.0 + damping * 3.0)
        
        low_env = np.exp(-decay_low * t)
        mid_env = np.exp(-decay_mid * t)
        high_env = np.exp(-decay_high * t)
        
        ir = (low_band * low_env) + (mid_band * mid_env) + (high_band * high_env)
    except Exception as e:
        logger.warning(f"Failed to use scipy filtering for IR generation, falling back to simple envelope: {e}")
        envelope = np.exp(-decay_rate * (1.0 + damping) * t)
        ir = noise * envelope
        
    # Apply fade-in for initial diffusion simulation
    fade_in_samples = int(0.015 * sample_rate)  # 15ms
    if fade_in_samples < num_samples:
        fade_in = np.linspace(0.0, 1.0, fade_in_samples)
        ir[:fade_in_samples] *= fade_in
        
    # Spring reverb special comb reflections
    if ir_type == "spring":
        for delay_ms in [25, 55, 85]:
            idx = int((delay_ms / 1000.0) * sample_rate)
            if idx < num_samples:
                ir[idx:] += ir[:-idx] * 0.4
                
    # Normalize to prevent clipping
    max_val = np.max(np.abs(ir))
    if max_val > 1e-6:
        ir = ir / max_val * 0.7
        
    return ir.astype(np.float32)


class PedalboardEffect(AudioEffect):
    """Pedalboard-specific audio effect implementation"""
    
    def __init__(self, effect_type: str, channel: AudioChannel, order: int = 0):
        super().__init__(effect_type, channel, order)
        self._effect = None
        self._init_effect()
    
    def _init_effect(self):
        """Initialize the pedalboard effect"""
        if not PEDALBOARD_AVAILABLE:
            return
        
        try:
            if self.effect_type == "reverb":
                self._effect = Reverb(
                    room_size=self.parameters.get("roomSize", 0.5),
                    damping=self.parameters.get("damping", 0.3),
                    wet_level=self.parameters.get("wetDryMix", 0.3),
                    dry_level=1.0 - self.parameters.get("wetDryMix", 0.3)
                )
            elif self.effect_type == "compressor":
                self._effect = Compressor(
                    threshold_db=self.parameters.get("threshold", -20.0),
                    ratio=self.parameters.get("ratio", 4.0),
                    attack_ms=self.parameters.get("attack", 10.0),
                    release_ms=self.parameters.get("release", 100.0)
                )
            elif self.effect_type == "echo":
                delay = Delay(
                    delay_seconds=self.parameters.get("delayTime", 300.0) / 1000.0,
                    feedback=self.parameters.get("feedback", 0.4),
                    mix=self.parameters.get("wetLevel", 0.3)
                )
                high_cut = LowpassFilter(cutoff_frequency_hz=self.parameters.get("highCut", 8000.0))
                self._effect = Pedalboard([delay, high_cut])
            
            elif self.effect_type == "equalizer":
                eq_filters = []
                if abs(self.parameters.get("low", 0.0)) > 0.1:
                    eq_filters.append(LowShelfFilter(cutoff_frequency_hz=80, gain_db=self.parameters["low"]))
                if abs(self.parameters.get("lowMid", 0.0)) > 0.1:
                    eq_filters.append(PeakFilter(cutoff_frequency_hz=320, gain_db=self.parameters["lowMid"]))
                if abs(self.parameters.get("mid", 0.0)) > 0.1:
                    eq_filters.append(PeakFilter(cutoff_frequency_hz=1250, gain_db=self.parameters["mid"]))
                if abs(self.parameters.get("highMid", 0.0)) > 0.1:
                    eq_filters.append(PeakFilter(cutoff_frequency_hz=5000, gain_db=self.parameters["highMid"]))
                if abs(self.parameters.get("high", 0.0)) > 0.1:
                    eq_filters.append(HighShelfFilter(cutoff_frequency_hz=12500, gain_db=self.parameters["high"]))
                self._effect = Pedalboard(eq_filters) if eq_filters else None
            
            elif self.effect_type == "chorus":
                self._effect = Chorus(
                    rate_hz=self.parameters.get("rate", 1.5),
                    depth=self.parameters.get("depth", 0.3),
                    mix=self.parameters.get("mix", 0.5)
                )
            elif self.effect_type == "distortion":
                self._effect = Distortion(drive_db=self.parameters.get("drive", 0.3) * 20)
            
            elif self.effect_type == "noiseGate":
                self._effect = NoiseGate(
                    threshold_db=self.parameters.get("threshold", -30.0),
                    ratio=self.parameters.get("ratio", 10.0),
                    attack_ms=self.parameters.get("attack", 5.0),
                    release_ms=self.parameters.get("release", 100.0)
                )
            elif self.effect_type == "gain":
                self._effect = Gain(gain_db=self.parameters.get("gain", 0.0))
            
            elif self.effect_type == "lowpass":
                self._effect = LowpassFilter(cutoff_frequency_hz=self.parameters.get("cutoff", 8000.0))
            elif self.effect_type == "highpass":
                self._effect = HighpassFilter(cutoff_frequency_hz=self.parameters.get("cutoff", 100.0))
            
            elif self.effect_type == "bitcrush":
                self._effect = Bitcrush(bit_depth=self.parameters.get("bitDepth", 8))
            elif self.effect_type == "clipping":
                self._effect = Clipping(threshold_db=self.parameters.get("threshold", -6.0))
            elif self.effect_type == "limiter":
                self._effect = Limiter(
                    threshold_db=self.parameters.get("threshold", -3.0),
                    release_ms=self.parameters.get("release", 50.0)
                )
            elif self.effect_type == "invert":
                self._effect = Invert()
            
            elif self.effect_type == "ladderFilter":
                mode_map = {
                    "LPF12": LadderFilter.Mode.LPF12,
                    "LPF24": LadderFilter.Mode.LPF24,
                    "HPF12": LadderFilter.Mode.HPF12,
                    "HPF24": LadderFilter.Mode.HPF24,
                    "BPF12": LadderFilter.Mode.BPF12,
                    "BPF24": LadderFilter.Mode.BPF24
                }
                mode = mode_map.get(self.parameters.get("mode", "LPF12"), LadderFilter.Mode.LPF12)
                self._effect = LadderFilter(
                    mode=mode,
                    cutoff_hz=self.parameters.get("cutoff", 1000.0),
                    resonance=self.parameters.get("resonance", 0.7),
                    drive=self.parameters.get("drive", 1.0)
                )
            
            elif self.effect_type == "peakFilter":
                self._effect = PeakFilter(
                    cutoff_frequency_hz=self.parameters.get("cutoff", 1000.0),
                    gain_db=self.parameters.get("gain", 0.0),
                    q=self.parameters.get("q", 1.0)
                )
            
            elif self.effect_type == "highShelfFilter":
                self._effect = HighShelfFilter(
                    cutoff_frequency_hz=self.parameters.get("cutoff", 8000.0),
                    gain_db=self.parameters.get("gain", 0.0),
                    q=self.parameters.get("q", 0.7)
                )
            
            elif self.effect_type == "lowShelfFilter":
                self._effect = LowShelfFilter(
                    cutoff_frequency_hz=self.parameters.get("cutoff", 200.0),
                    gain_db=self.parameters.get("gain", 0.0),
                    q=self.parameters.get("q", 0.7)
                )
            
            elif self.effect_type == "convolution":
                ir_type = self.parameters.get("impulseResponse", "hall")
                room_size = self.parameters.get("roomSize", 0.5)
                damping = self.parameters.get("damping", 0.3)
                mix = self.parameters.get("mix", 0.3)
                
                # Generate synthetic impulse response NumPy array
                ir = _generate_synthetic_ir(ir_type, room_size, damping, sample_rate=48000)
                
                # Spotify's Pedalboard supports passing a NumPy float32 array directly
                self._effect = Convolution(
                    ir,
                    sample_rate=48000,
                    mix=mix
                )
            
            elif self.effect_type == "mp3Compressor":
                self._effect = MP3Compressor(vbr_quality=self.parameters.get("vbrQuality", 2))
            elif self.effect_type == "gsmCompressor":
                self._effect = GSMFullRateCompressor()
                
        except Exception as e:
            logger.error(f"Failed to initialize {self.effect_type} effect: {e}")
            self._effect = None
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        super().set_parameters(parameters)
        self._init_effect()  # Reinitialize with new parameters
    
    def process(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if not self.enabled or not PEDALBOARD_AVAILABLE or self._effect is None:
            return audio
        
        try:
            # Convert to numpy
            audio_np = audio.detach().cpu().numpy().copy().astype(np.float32)
            
            # Pedalboard expects 2D array (channels, samples)
            if audio_np.ndim == 1:
                audio_np = audio_np.reshape(1, -1)
            
            # Apply effect
            processed = self._effect(audio_np, sample_rate)
            
            # Convert back to 1D if needed
            if processed.ndim == 2 and processed.shape[0] == 1:
                processed = processed[0]
            
            # Convert back to torch tensor
            return torch.tensor(processed.copy(), dtype=audio.dtype, device=audio.device)
            
        except Exception as e:
            logger.error(f"Error in {self.effect_type} effect: {e}")
            return audio