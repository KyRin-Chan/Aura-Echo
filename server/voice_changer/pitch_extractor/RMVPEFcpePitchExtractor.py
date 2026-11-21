import math
from typing import Optional
import torch
import torch.nn.functional as F

from const import PitchExtractorType
from voice_changer.common.FCPE import spawn_infer_model_from_pt
from voice_changer.common.MelExtractorFcpe import Wav2MelModule
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
from voice_changer.common.rmvpe.rmvpe import RMVPE
from voice_changer.pitch_extractor.PitchExtractor import PitchExtractor


class RMVPEFcpePitchExtractor(PitchExtractor):
    """Hybrid pitch extractor combining RMVPE and FCPE via calibrated confidence fusion and causal continuity arbitration."""

    def __init__(self, rmvpe_file: str, fcpe_file: str):
        super().__init__()
        self.type: PitchExtractorType = "rmvpe_fcpe"

        device_manager = DeviceManager.get_instance()
        self.device = device_manager.device
        self.rmvpe_threshold = 0.045
        self.fcpe_threshold = 0.006

        self.rmvpe = RMVPE(
            model_path=rmvpe_file,
            is_half=device_manager.use_fp16(),
            use_jit_compile=device_manager.use_jit_compile(),
            device=self.device,
        )

        # NOTE: FCPE doesn't behave correctly in FP16 mode.
        self.fcpe = spawn_infer_model_from_pt(
            fcpe_file, False, self.device, bundled_model=True
        )
        self.fcpe_mel_extractor = Wav2MelModule(
            sr=16000,
            n_mels=128,
            n_fft=1024,
            win_size=1024,
            hop_length=160,
            fmin=0,
            fmax=8000,
            clip_val=1e-05,
            is_half=False,
        ).to(self.device)

        self._fcpe_stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        )

    @staticmethod
    def _calibrate_confidence(
        conf: torch.Tensor, threshold: float, scale_k: float
    ) -> torch.Tensor:
        """Map raw model confidence above threshold onto a calibrated [0, 1] scale."""
        above = torch.clamp(conf - threshold, min=0.0)
        return above / (above + scale_k + 1e-6)

    @classmethod
    def fuse_predictions(
        cls,
        f0_rmvpe: torch.Tensor,
        conf_rmvpe: torch.Tensor,
        f0_fcpe: torch.Tensor,
        conf_fcpe: torch.Tensor,
        rmvpe_threshold: float = 0.045,
        fcpe_threshold: float = 0.006,
    ) -> torch.Tensor:
        """Fuse RMVPE and FCPE F0/confidence predictions without averaging octave divergences."""
        f1 = f0_rmvpe.reshape(-1).float()
        c1 = conf_rmvpe.reshape(-1).float()
        f2 = f0_fcpe.reshape(-1).float()
        c2 = conf_fcpe.reshape(-1).float()

        T = f1.shape[0]
        if T == 0:
            return f1

        if f2.shape[0] > T:
            f2 = f2[:T]
            c2 = c2[:T]
        elif f2.shape[0] < T:
            pad_len = T - f2.shape[0]
            if f2.numel() > 0:
                f2 = F.pad(f2.unsqueeze(0), (0, pad_len), mode="replicate").squeeze(0)
                c2 = F.pad(c2.unsqueeze(0), (0, pad_len), mode="replicate").squeeze(0)
            else:
                f2 = torch.zeros_like(f1)
                c2 = torch.zeros_like(c1)

        # Calibrate both models onto a comparable [0, 1] scale
        s1 = cls._calibrate_confidence(c1, rmvpe_threshold, 0.08)
        s2 = cls._calibrate_confidence(c2, fcpe_threshold, 0.035)

        # Gently boost FCPE on the rightmost 4 frames (40ms) where RMVPE's backward BiGRU lacks future context
        tail_len = min(4, T)
        if tail_len > 0:
            ramp = torch.tensor(
                [1.05, 1.12, 1.20, 1.30][-tail_len:], dtype=s2.dtype, device=s2.device
            )
            s2 = s2.clone()
            s2[-tail_len:] = torch.clamp(s2[-tail_len:] * ramp, max=1.0)

        v1 = f1 > 0.0
        v2 = f2 > 0.0
        both_voiced = v1 & v2

        cent1 = torch.where(
            v1, 1200.0 * torch.log2(torch.clamp(f1, min=10.0) / 10.0), torch.zeros_like(f1)
        )
        cent2 = torch.where(
            v2, 1200.0 * torch.log2(torch.clamp(f2, min=10.0) / 10.0), torch.zeros_like(f2)
        )
        diff_cents = torch.abs(cent1 - cent2)

        w_sum = torch.clamp(s1 + s2, min=1e-6)
        cent_fused = (s1 * cent1 + s2 * cent2) / w_sum

        need_arb = (both_voiced & (diff_cents > 50.0)) | (v1 ^ v2)
        if not bool(torch.any(need_arb).item()):
            return torch.where(
                both_voiced,
                10.0 * torch.pow(2.0, cent_fused / 1200.0),
                torch.zeros_like(f1),
            )

        cent_out = torch.where(both_voiced, cent_fused, torch.where(v1, cent1, cent2))
        voiced_out = v1 | v2

        v1_l = v1.tolist()
        v2_l = v2.tolist()
        both_l = both_voiced.tolist()
        cent1_l = cent1.tolist()
        cent2_l = cent2.tolist()
        diff_l = diff_cents.tolist()
        s1_l = s1.tolist()
        s2_l = s2.tolist()
        cent_out_l = cent_out.tolist()
        voiced_out_l = voiced_out.tolist()

        prev_cent = 0.0
        unvoiced_run = 0

        for t in range(T):
            if not v1_l[t] and not v2_l[t]:
                unvoiced_run += 1
                if unvoiced_run > 3:
                    prev_cent = 0.0
                continue

            if both_l[t]:
                if diff_l[t] <= 50.0:
                    c_val = cent_out_l[t]
                else:
                    # Large divergence (e.g. octave error): never average; hard-switch using causal continuity
                    if prev_cent > 0.0:
                        p1 = math.exp(-abs(cent1_l[t] - prev_cent) / 250.0)
                        p2 = math.exp(-abs(cent2_l[t] - prev_cent) / 250.0)
                        score1 = s1_l[t] * (0.35 + 0.65 * p1)
                        score2 = s2_l[t] * (0.35 + 0.65 * p2)
                    else:
                        score1 = s1_l[t]
                        score2 = s2_l[t]
                    c_val = cent1_l[t] if score1 >= score2 else cent2_l[t]
                    cent_out_l[t] = c_val
                prev_cent = c_val
                unvoiced_run = 0
                continue

            # Single-voiced frame (V/UV boundary or breathy tail): complement only if continuous or high confidence
            c_act = cent1_l[t] if v1_l[t] else cent2_l[t]
            s_act = s1_l[t] if v1_l[t] else s2_l[t]
            if prev_cent > 0.0 and abs(c_act - prev_cent) <= 150.0:
                accept = s_act >= 0.22
            else:
                accept = s_act >= 0.55

            if accept:
                cent_out_l[t] = c_act
                voiced_out_l[t] = True
                prev_cent = c_act
                unvoiced_run = 0
            else:
                voiced_out_l[t] = False
                unvoiced_run += 1
                if unvoiced_run > 3:
                    prev_cent = 0.0

        cent_final = torch.tensor(cent_out_l, dtype=torch.float32, device=f1.device)
        voiced_final = torch.tensor(voiced_out_l, dtype=torch.bool, device=f1.device)
        return torch.where(
            voiced_final,
            10.0 * torch.pow(2.0, cent_final / 1200.0),
            torch.zeros_like(f1),
        )

    def extract(
        self,
        audio: torch.Tensor,
        sr: int,
        window: int,
    ) -> torch.Tensor:
        if audio.device.type == "cuda" and self._fcpe_stream is not None:
            cur_stream = torch.cuda.current_stream(audio.device)
            self._fcpe_stream.wait_stream(cur_stream)
            audio.record_stream(self._fcpe_stream)

            with torch.cuda.stream(self._fcpe_stream):
                mel_fcpe = self.fcpe_mel_extractor(audio.unsqueeze(0).float())
                f0_fcpe, conf_fcpe = self.fcpe.infer_with_confidence(
                    mel_fcpe, threshold=self.fcpe_threshold
                )
                f0_fcpe.record_stream(cur_stream)
                conf_fcpe.record_stream(cur_stream)

            f0_rmvpe, conf_rmvpe = self.rmvpe.infer_from_audio_t_with_confidence(
                audio, threshold=self.rmvpe_threshold
            )
            cur_stream.wait_stream(self._fcpe_stream)
        else:
            f0_rmvpe, conf_rmvpe = self.rmvpe.infer_from_audio_t_with_confidence(
                audio, threshold=self.rmvpe_threshold
            )
            mel_fcpe = self.fcpe_mel_extractor(audio.unsqueeze(0).float())
            f0_fcpe, conf_fcpe = self.fcpe.infer_with_confidence(
                mel_fcpe, threshold=self.fcpe_threshold
            )

        f0_out = self.fuse_predictions(
            f0_rmvpe,
            conf_rmvpe,
            f0_fcpe,
            conf_fcpe,
            rmvpe_threshold=self.rmvpe_threshold,
            fcpe_threshold=self.fcpe_threshold,
        )
        return f0_out.to(dtype=audio.dtype)
