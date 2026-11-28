import torch

from const import PitchExtractorType
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
from voice_changer.common.rmvpe.rmvpe import RMVPE
from voice_changer.pitch_extractor.PitchExtractor import (
    PitchExtractor,
    calibrate_voicing_confidence,
)


class HPARMVPEPitchExtractor(PitchExtractor):
    """PyTorch pitch extractor using the YOLOv13-based HPA-RMVPE architecture."""

    def __init__(self, file: str):
        super().__init__()
        self.type: PitchExtractorType = "hpa_rmvpe"

        device_manager = DeviceManager.get_instance()
        self.rmvpe = RMVPE(
            model_path=file,
            is_half=device_manager.use_fp16(),
            use_jit_compile=False,
            device=device_manager.device,
            dtype=device_manager.get_torch_dtype(),
            hpa=True,
        )

    def extract_with_confidence(
        self,
        audio: torch.Tensor,
        sr: int,
        window: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        f0, raw_conf = self.rmvpe.infer_from_audio_t_with_confidence(audio, threshold=0.032)
        f0 = f0.squeeze()
        raw_conf = raw_conf.squeeze().float()
        return f0, calibrate_voicing_confidence(f0, raw_conf)

    def extract(
        self,
        audio: torch.Tensor,
        sr: int,
        window: int,
    ) -> torch.Tensor:
        return self.rmvpe.infer_from_audio_t(audio, threshold=0.032).squeeze()
