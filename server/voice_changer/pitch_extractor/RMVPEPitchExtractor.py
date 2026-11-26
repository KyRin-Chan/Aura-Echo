from const import PitchExtractorType
from voice_changer.pitch_extractor.PitchExtractor import PitchExtractor
from voice_changer.common.rmvpe.rmvpe import RMVPE
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
import torch


class RMVPEPitchExtractor(PitchExtractor):

    def __init__(self, file: str):
        super().__init__()
        self.type: PitchExtractorType = "rmvpe"

        device_manager = DeviceManager.get_instance()
        self.rmvpe = RMVPE(
            model_path=file,
            is_half=device_manager.use_fp16(),
            use_jit_compile=device_manager.use_jit_compile(),
            device=device_manager.device,
            dtype=device_manager.get_torch_dtype(),
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
        above = torch.clamp(raw_conf - 0.020, min=0.0)
        cal_conf = torch.where(f0 > 0, above / (above + 0.08 + 1e-6), torch.zeros_like(raw_conf))
        return f0, cal_conf

    def extract(
        self,
        audio: torch.Tensor,
        sr: int,
        window: int,
    ) -> torch.Tensor:
        return self.rmvpe.infer_from_audio_t(audio, threshold=0.032).squeeze()