import torch
from torchfcpe.models_infer import spawn_model
from torchfcpe.tools import (
    DotDict
)


class InferCFNaiveMelPE(torch.nn.Module):
    """Infer CFNaiveMelPE
    Args:
        args (DotDict): Config.
        state_dict (dict): Model state dict.
    """

    def __init__(self, args, state_dict):
        super().__init__()
        self.model = spawn_model(args)
        self.model.load_state_dict(state_dict)
        if args.is_half:
            self.model = self.model.half()
        self.model.eval()

    @torch.no_grad()
    def forward(self,
                mel: torch.Tensor,
                decoder_mode: str = 'local_argmax',
                threshold: float = 0.006
                ) -> torch.Tensor:
        """Infer
        Args:
            mel (torch.Tensor): Input mel spectrogram.
            decoder_mode (str): Decoder type. Default: "local_argmax", support "argmax" or "local_argmax".
            threshold (float): Threshold to mask. Default: 0.006.
        return: f0 (torch.Tensor): f0 Hz, shape (B, (n_sample//hop_size + 1), 1).
        """
        return self.model.infer(mel, decoder=decoder_mode, threshold=threshold)  # (B, T, 1)

    @torch.no_grad()
    def infer_with_confidence(
        self,
        mel: torch.Tensor,
        threshold: float = 0.006,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Infer F0 and per-frame peak confidence in a single forward pass."""
        latent = self.model(mel)  # [B, T, 360]
        confident, _ = torch.max(latent, dim=-1, keepdim=True)  # [B, T, 1]
        cents = self.model.latent2cents_local_decoder(latent, threshold=threshold, mask=False)
        f0 = self.model.cent_to_f0(cents)
        f0 = f0 * (confident > threshold)
        return f0.squeeze(-1), confident.squeeze(-1)  # ([B, T], [B, T])


def spawn_infer_model_from_pt(pt_path: str, is_half: bool = False, device: torch.device = torch.device('cpu'), bundled_model: bool = False) -> InferCFNaiveMelPE:
    """
    Spawn infer model from pt file
    Args:
        pt_path (str): Path to pt file.
        device (str): Device. Default: None.
        bundled_model (bool): Whether this model is bundled model, only used in spawn_bundled_infer_model.
    """
    ckpt = torch.load(pt_path, map_location='cpu')
    if bundled_model:
        ckpt['config_dict']['model']['conv_dropout'] = 0.0
        ckpt['config_dict']['model']['atten_dropout'] = 0.0
    args = DotDict(ckpt['config_dict'])
    args.is_half = is_half
    if (args.is_onnx is not None) and (args.is_onnx is True):
        raise ValueError(f'  [ERROR] spawn_infer_model_from_pt: this model is an onnx model.')

    if args.model.type == 'CFNaiveMelPE':
        infer_model = InferCFNaiveMelPE(args, ckpt['model']).to(device).eval()
    else:
        raise ValueError(f'  [ERROR] args.model.type is {args.model.type}, but only support CFNaiveMelPE')

    return infer_model
