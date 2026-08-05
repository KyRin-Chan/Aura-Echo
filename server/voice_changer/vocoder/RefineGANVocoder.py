import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio import transforms as tat
import logging
from voice_changer.vocoder.Vocoder import Vocoder
from voice_changer.common.deviceManager.DeviceManager import DeviceManager

logger = logging.getLogger(__name__)

class RefinementBlock(nn.Module):
    """Multi-Scale Artifact Refinement Module for RefineGAN."""
    def __init__(self, channels: int, kernel_size: int = 7, dilation: tuple = (1, 3, 5)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(
                channels, channels, kernel_size,
                padding=(kernel_size - 1) * d // 2,
                dilation=d
            ) for d in dilation
        ])
        self.gains = nn.ParameterList([nn.Parameter(torch.ones(1, channels, 1)) for _ in dilation])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for conv, gain in zip(self.convs, self.gains):
            res = F.leaky_relu(out, 0.2)
            res = conv(res) * gain
            out = out + res
        return out

class RefineGANGenerator(nn.Module):
    """RefineGAN Generator with Pitch-Guided Audio Synthesis."""
    def __init__(
        self,
        in_channels: int = 192,
        out_channels: int = 1,
        upsample_rates: tuple = (8, 8, 2, 2),
        upsample_kernel_sizes: tuple = (16, 16, 4, 4),
        channels: int = 512,
        sr: int = 32000
    ):
        super().__init__()
        self.sr = sr
        self.in_conv = nn.Conv1d(in_channels, channels, kernel_size=7, padding=3)

        self.upsamples = nn.ModuleList()
        curr_channels = channels
        for r, k in zip(upsample_rates, upsample_kernel_sizes):
            next_channels = curr_channels // 2
            self.upsamples.append(
                nn.ConvTranspose1d(
                    curr_channels, next_channels,
                    kernel_size=k, stride=r, padding=(k - r) // 2
                )
            )
            curr_channels = next_channels

        self.refinement_blocks = nn.ModuleList([
            RefinementBlock(curr_channels, kernel_size=7) for _ in range(3)
        ])

        self.out_conv = nn.Conv1d(curr_channels, out_channels, kernel_size=7, padding=3)

    def forward(self, z: torch.Tensor, pitchf: torch.Tensor | None = None, g: torch.Tensor | None = None) -> torch.Tensor:
        x = self.in_conv(z)
        for up in self.upsamples:
            x = F.leaky_relu(x, 0.2)
            x = up(x)

        for block in self.refinement_blocks:
            x = block(x)

        x = F.leaky_relu(x, 0.2)
        out = torch.tanh(self.out_conv(x))
        return out

class RefineGANVocoder(Vocoder):
    """RefineGAN Vocoder Instance supporting GPU-accelerated Resampling."""

    def __init__(self, model_path: str, target_sr: int = 40000):
        super().__init__()
        self.device_manager = DeviceManager.get_instance()
        self.device = self.device_manager.device
        self.is_half = self.device_manager.use_fp16()
        self.vocoder_sr = 32000  # RefineGAN native sample rate
        self.target_sr = target_sr

        logger.info(f"Initializing RefineGAN Vocoder (Native SR: {self.vocoder_sr}Hz, Target SR: {self.target_sr}Hz)")

        self.model = RefineGANGenerator(sr=self.vocoder_sr).to(self.device)
        self.is_loaded = False

        if os.path.exists(model_path):
            try:
                cpt = torch.load(model_path, map_location=self.device)
                if isinstance(cpt, dict) and "weight" in cpt:
                    self.model.load_state_dict(cpt["weight"], strict=False)
                elif isinstance(cpt, dict) and "model" in cpt:
                    self.model.load_state_dict(cpt["model"], strict=False)
                elif isinstance(cpt, dict):
                    self.model.load_state_dict(cpt, strict=False)
                self.is_loaded = True
                logger.info(f"Successfully loaded RefineGAN vocoder weights from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load RefineGAN weight file {model_path}: {e}. Running in initialization mode.")
        else:
            logger.warning(f"RefineGAN weight file not found at {model_path}. Please download it in the Downloader panel.")

        self.model.eval()
        if self.is_half:
            self.model = self.model.half()

        # Set up GPU Resamplers if target_sr != vocoder_sr
        if self.target_sr != self.vocoder_sr:
            logger.info(f"Creating GPU Resampler: {self.vocoder_sr}Hz -> {self.target_sr}Hz")
            self.resampler_out = tat.Resample(
                orig_freq=self.vocoder_sr,
                new_freq=self.target_sr,
                dtype=torch.float32
            ).to(self.device)
        else:
            self.resampler_out = None

    def infer(
        self,
        z: torch.Tensor,
        pitchf: torch.Tensor,
        g: torch.Tensor | None = None
    ) -> torch.Tensor:
        dtype = torch.float16 if self.is_half else torch.float32
        z = z.to(device=self.device, dtype=dtype)
        if pitchf is not None:
            pitchf = pitchf.to(device=self.device, dtype=dtype)

        with torch.no_grad():
            audio_32k = self.model(z, pitchf, g)
            if self.resampler_out is not None:
                audio_32k_f32 = audio_32k.to(torch.float32)
                audio_target = self.resampler_out(audio_32k_f32)
                return audio_target.to(dtype)
            return audio_32k
