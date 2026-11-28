import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

from voice_changer.common.SafetensorsUtils import load_model
from voice_changer.pitch_extractor.PitchExtractor import decode_salience_to_f0

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
N_CLASS = 360


def init_layer(layer: nn.Module) -> None:
    """Initialize convolutional/linear layer weights with Xavier uniform."""
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias") and layer.bias is not None:
        layer.bias.data.fill_(0.0)


def init_bn(bn: nn.BatchNorm2d) -> None:
    """Initialize BatchNorm2d parameters to identity defaults."""
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)
    bn.running_mean.data.fill_(0.0)
    bn.running_var.data.fill_(1.0)


class _PatchRearrange(nn.Module):
    """Pure-PyTorch equivalent of einops Rearrange('b c (w p1) (h p2) -> b (w h) (p1 p2 c)')."""

    def __init__(self, p1: int, p2: int):
        super().__init__()
        self.p1 = p1
        self.p2 = p2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h_total, w_total = x.shape
        w = h_total // self.p1
        h = w_total // self.p2
        return (
            x.reshape(b, c, w, self.p1, h, self.p2)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(b, w * h, self.p1 * self.p2 * c)
        )


class BiGRU(nn.Module):
    """Patch-based bidirectional GRU for temporal pitch sequence refinement."""

    def __init__(self, patch_size: tuple[int, int], channels: int, depth: int):
        super().__init__()
        patch_width, patch_height = patch_size
        patch_dim = channels * patch_height * patch_width
        self.to_patch_embedding = nn.Sequential(_PatchRearrange(patch_width, patch_height))
        self.gru = nn.GRU(
            patch_dim,
            patch_dim // 2,
            num_layers=depth,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.to_patch_embedding(x)
        try:
            return self.gru(x)[0]
        except Exception:
            torch.backends.cudnn.enabled = False
            return self.gru(x)[0]


class ResConvBlock(nn.Module):
    """Pre-activation residual convolution block (BatchNorm -> PReLU -> Conv2d) x2."""

    def __init__(self, in_planes: int, out_planes: int):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.01)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.01)
        self.act1 = nn.PReLU()
        self.act2 = nn.PReLU()
        self.conv1 = nn.Conv2d(in_planes, out_planes, (3, 3), padding=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(out_planes, out_planes, (3, 3), padding=(1, 1), bias=False)

        self.is_shortcut = in_planes != out_planes
        if self.is_shortcut:
            self.shortcut = nn.Conv2d(in_planes, out_planes, (1, 1))

        self.init_weights()

    def init_weights(self) -> None:
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_layer(self.conv1)
        init_layer(self.conv2)
        if self.is_shortcut:
            init_layer(self.shortcut)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.act1(self.bn1(x)))
        out = self.conv2(self.act2(self.bn2(out)))
        if self.is_shortcut:
            return self.shortcut(x) + out
        return out + x


class ResEncoderBlock(nn.Module):
    """Residual encoder block with optional spatial pooling."""

    def __init__(self, in_channels: int, out_channels: int, n_blocks: int, kernel_size):
        super().__init__()
        self.conv = nn.ModuleList([ResConvBlock(in_channels, out_channels)])
        for _ in range(n_blocks - 1):
            self.conv.append(ResConvBlock(out_channels, out_channels))
        self.pool = nn.MaxPool2d(kernel_size) if kernel_size is not None else None

    def forward(self, x: torch.Tensor):
        for layer in self.conv:
            x = layer(x)
        if self.pool is not None:
            return x, self.pool(x)
        return x


class Encoder(nn.Module):
    """6-stage UNet encoder downsampling along frequency bins."""

    def __init__(self, in_channels: int, n_blocks: int):
        super().__init__()
        channels = [
            (in_channels, 32),
            (32, 64),
            (64, 128),
            (128, 256),
            (256, 384),
            (384, 384),
        ]
        self.en_blocks = nn.ModuleList(
            [ResEncoderBlock(c_in, c_out, n_blocks, (1, 2)) for c_in, c_out in channels]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        concat_tensors: list[torch.Tensor] = []
        for layer in self.en_blocks:
            skip, x = layer(x)
            concat_tensors.append(skip)
        return x, concat_tensors


class LatentBlocks(nn.Module):
    """Bottleneck residual blocks without spatial downsampling."""

    def __init__(self, n_blocks: int, latent_layers: int):
        super().__init__()
        self.latent_blocks = nn.ModuleList(
            [ResEncoderBlock(384, 384, n_blocks, None) for _ in range(latent_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.latent_blocks:
            x = layer(x)
        return x


class ResDecoderBlock(nn.Module):
    """Pre-activation residual decoder block with ConvTranspose2d upsampling."""

    def __init__(self, in_channels: int, out_channels: int, n_blocks: int, stride: tuple[int, int]):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(
            in_channels, out_channels, stride, stride, (0, 0), bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels, momentum=0.01)
        self.conv = nn.ModuleList([ResConvBlock(out_channels * 2, out_channels)])
        for _ in range(n_blocks - 1):
            self.conv.append(ResConvBlock(out_channels, out_channels))
        self.init_weights()

    def init_weights(self) -> None:
        init_bn(self.bn1)
        init_layer(self.conv1)

    def forward(self, x: torch.Tensor, concat: torch.Tensor) -> torch.Tensor:
        x = self.conv1(F.relu_(self.bn1(x)))
        x = torch.cat((x, concat), dim=1)
        for layer in self.conv:
            x = layer(x)
        return x


class Decoder(nn.Module):
    """6-stage UNet decoder mirroring Encoder skip connections."""

    def __init__(self, n_blocks: int):
        super().__init__()
        channels = [
            (384, 384),
            (384, 384),
            (384, 256),
            (256, 128),
            (128, 64),
            (64, 32),
        ]
        self.de_blocks = nn.ModuleList(
            [ResDecoderBlock(c_in, c_out, n_blocks, (1, 2)) for c_in, c_out in channels]
        )

    def forward(self, x: torch.Tensor, concat_tensors: list[torch.Tensor]) -> torch.Tensor:
        for i, layer in enumerate(self.de_blocks):
            x = layer(x, concat_tensors[-1 - i])
        return x


class PE_Decoder(nn.Module):
    """Pitch estimation decoder producing [B, T, n_class] salience probabilities."""

    def __init__(
        self,
        n_blocks: int,
        seq_layers: int = 1,
        window_length: int = 1024,
        n_class: int = N_CLASS,
    ):
        super().__init__()
        self.de_blocks = Decoder(n_blocks)
        self.after_conv1 = ResEncoderBlock(32, 32, n_blocks, None)
        self.after_conv2 = nn.Conv2d(32, 1, (1, 1))
        self.fc = nn.Sequential(
            BiGRU((1, window_length // 2), 1, seq_layers),
            nn.Linear(window_length // 2, n_class),
            nn.Sigmoid(),
        )
        init_layer(self.after_conv2)

    def forward(self, x: torch.Tensor, concat_tensors: list[torch.Tensor]) -> torch.Tensor:
        x = self.after_conv2(self.after_conv1(self.de_blocks(x, concat_tensors)))
        return self.fc(x)


class SVS_Decoder(nn.Module):
    """Singing voice separation decoder producing spectrogram mask features."""

    def __init__(self, in_channels: int, n_blocks: int):
        super().__init__()
        self.de_blocks = Decoder(n_blocks)
        self.after_conv1 = ResEncoderBlock(32, 32, n_blocks, None)
        self.after_conv2 = nn.Conv2d(32, in_channels * 4, (1, 1))
        init_layer(self.after_conv2)

    def forward(self, x: torch.Tensor, concat_tensors: list[torch.Tensor]) -> torch.Tensor:
        return self.after_conv2(self.after_conv1(self.de_blocks(x, concat_tensors)))


class DJCMM(nn.Module):
    """Deep Joint Cascade Model for Singing Voice Separation and Vocal Pitch Estimation."""

    def __init__(
        self,
        in_channels: int = 1,
        n_blocks: int = 1,
        latent_layers: int = 1,
        svs: bool = False,
        window_length: int = 1024,
        n_class: int = N_CLASS,
    ):
        super().__init__()
        self.bn = nn.BatchNorm2d(window_length // 2 + 1, momentum=0.01)
        self.pe_encoder = Encoder(in_channels, n_blocks)
        self.pe_latent = LatentBlocks(n_blocks, latent_layers)
        self.pe_decoder = PE_Decoder(n_blocks, window_length=window_length, n_class=n_class)
        self.svs = svs
        if svs:
            self.svs_encoder = Encoder(in_channels, n_blocks)
            self.svs_latent = LatentBlocks(n_blocks, latent_layers)
            self.svs_decoder = SVS_Decoder(in_channels, n_blocks)
        init_bn(self.bn)

    def spec(self, x: torch.Tensor, spec_m: torch.Tensor) -> torch.Tensor:
        bs, c, time_steps, freqs_steps = x.shape
        x = x.reshape(bs, c // 4, 4, time_steps, freqs_steps)
        mask_spec = x[:, :, 0, :, :].sigmoid()
        linear_spec = x[:, :, 3, :, :]
        return (spec_m.detach() * mask_spec + linear_spec).relu()

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        x = self.bn(spec.transpose(1, 3)).transpose(1, 3)[..., :-1]
        if self.svs:
            x, concat_tensors = self.svs_encoder(x)
            x = self.svs_decoder(self.svs_latent(x), concat_tensors)
            x = self.spec(F.pad(x, pad=(0, 1)), spec)[..., :-1]
        x, concat_tensors = self.pe_encoder(x)
        return self.pe_decoder(self.pe_latent(x), concat_tensors)


class Spectrogram(nn.Module):
    """Magnitude STFT spectrogram extractor returning shape [B, C, T, F]."""

    def __init__(
        self,
        hop_length: int = 160,
        win_length: int = 1024,
        n_fft: int | None = None,
        clamp: float = 1e-10,
    ):
        super().__init__()
        self.n_fft = win_length if n_fft is None else n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.clamp = clamp
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

    def forward(self, audio: torch.Tensor, center: bool = True) -> torch.Tensor:
        bs, c, segment_samples = audio.shape
        flat = audio.reshape(bs * c, segment_samples)
        fft = torch.stft(
            flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=center,
            pad_mode="reflect",
            return_complex=True,
        )
        mag = fft.abs().transpose(1, 2).clamp(min=self.clamp)
        return mag.reshape(bs, c, mag.shape[1], mag.shape[2])


def pad_audio_segments(audio: torch.Tensor, seg_len: int) -> torch.Tensor:
    """Pad and slice 1D audio into overlapping windows of length seg_len."""
    hop = seg_len // 2
    audio_len = audio.shape[-1]
    left_pad = seg_len // 4
    right_pad = (
        ((audio_len + seg_len - 1) // seg_len + 1) * seg_len + hop
        - audio_len
        - left_pad
    )
    return (
        F.pad(audio, (left_pad, right_pad))
        .unfold(0, seg_len, hop)
        .unsqueeze(1)
        .contiguous()
    )


class DJCM:
    """PyTorch DJCM pitch extraction wrapper supporting real-time and long-form audio."""

    def __init__(
        self,
        model_path: str,
        is_half: bool,
        device: torch.device,
        dtype: torch.dtype | None = None,
        svs: bool = False,
        segment_len: float = 5.12,
        batch_size: int = 1,
    ):
        self.device = device
        self.target_dtype = (
            dtype if dtype is not None else (torch.float16 if is_half else torch.float32)
        )
        window_length = 2048 if svs else 1024

        model = DJCMM(1, 1, 1, svs=svs, window_length=window_length, n_class=N_CLASS)
        if model_path.endswith(".safetensors"):
            with safe_open(
                model_path, "pt", device=str(device) if device.type == "cuda" else "cpu"
            ) as cpt:
                load_model(model, cpt, strict=False)
        else:
            cpt = torch.load(
                model_path,
                map_location=device if device.type == "cuda" else "cpu",
                weights_only=True,
            )
            if isinstance(cpt, dict) and "state_dict" in cpt:
                cpt = cpt["state_dict"]
            model.load_state_dict(cpt, strict=False)

        model = model.eval().to(device)
        if self.target_dtype != torch.float32:
            model = model.to(dtype=self.target_dtype)
        self.model = model

        self.spec_extractor = Spectrogram(
            hop_length=SAMPLE_RATE // 100, win_length=window_length
        ).to(device)
        self.seg_len = int(segment_len * SAMPLE_RATE)
        self.seg_frames = self.seg_len // (SAMPLE_RATE // 100)
        self.batch_size = batch_size

        self.idx = torch.arange(N_CLASS, device=device)[None, None, :]
        self.idx_cents = self.idx * 20 + 1997.3794084376191

    def pad_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Pad and slice 1D audio into overlapping windows of length seg_len."""
        return pad_audio_segments(audio, self.seg_len)

    @torch.inference_mode()
    def infer_salience(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute [1, n_frames, 360] salience tensor for a 1D audio tensor."""
        if audio.ndim > 1:
            audio = audio.reshape(-1)
        n_frames = audio.shape[-1] // (SAMPLE_RATE // 100) + 1

        # Direct single-pass for real-time audio chunks (<= 5.12s); chunked stitching for longer inputs.
        if audio.shape[-1] <= self.seg_len:
            spec = self.spec_extractor(audio.unsqueeze(0).unsqueeze(0).float())
            hidden = self.model(spec.to(dtype=self.target_dtype))
            return hidden[:, :n_frames, :]

        segments = self.pad_audio(audio.float())
        hidden_chunks = []
        for i in range(0, len(segments), self.batch_size):
            spec = self.spec_extractor(segments[i : i + self.batch_size])
            hidden_chunks.append(self.model(spec.to(dtype=self.target_dtype)))
        all_segs = torch.cat(hidden_chunks, dim=0)

        start = self.seg_frames // 4
        end = int(self.seg_frames * 0.75)
        stitched = torch.cat([seg[start:end] for seg in all_segs], dim=0)[:n_frames]
        return stitched.unsqueeze(0)

    def decode_with_confidence(
        self, hidden: torch.Tensor, threshold: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return decode_salience_to_f0(hidden, threshold, self.idx, self.idx_cents)

    @torch.inference_mode()
    def infer_from_audio_t_with_confidence(
        self, audio: torch.Tensor, threshold: float = 0.03
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.infer_salience(audio)
        return self.decode_with_confidence(hidden, threshold)

    @torch.inference_mode()
    def infer_from_audio_t(
        self, audio: torch.Tensor, threshold: float = 0.03
    ) -> torch.Tensor:
        f0, _ = self.infer_from_audio_t_with_confidence(audio, threshold)
        return f0
