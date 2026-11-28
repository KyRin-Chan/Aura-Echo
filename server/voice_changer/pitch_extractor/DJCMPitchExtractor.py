import torch

from const import PitchExtractorType
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
from voice_changer.common.djcm.djcm import DJCM
from voice_changer.pitch_extractor.PitchExtractor import (
    PitchExtractor,
    calibrate_voicing_confidence,
)


class DJCMPitchExtractor(PitchExtractor):
    """PyTorch pitch extractor using DJCM (Deep Joint Cascade Model)."""

    def __init__(self, file: str):
        super().__init__()
        self.type: PitchExtractorType = "djcm"

        device_manager = DeviceManager.get_instance()
        self.djcm = DJCM(
            model_path=file,
            is_half=device_manager.use_fp16(),
            device=device_manager.device,
            dtype=device_manager.get_torch_dtype(),
            svs=False,
        )

    def extract_with_confidence(
        self,
        audio: torch.Tensor,
        sr: int,
        window: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        f0, raw_conf = self.djcm.infer_from_audio_t_with_confidence(audio, threshold=0.03)
        f0 = f0.squeeze()
        raw_conf = raw_conf.squeeze().float()
        return f0, calibrate_voicing_confidence(f0, raw_conf)

    def extract(
        self,
        audio: torch.Tensor,
        sr: int,
        window: int,
    ) -> torch.Tensor:
        return self.djcm.infer_from_audio_t(audio, threshold=0.03).squeeze()
