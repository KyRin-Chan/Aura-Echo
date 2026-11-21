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
        self.rmvpe_threshold = 0.032
        self.fcpe_threshold = 0.006
        self.rmvpe_sustain_threshold = 0.020
        self.fcpe_sustain_threshold = 0.004
        self._last_voiced_cent: float = 0.0

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

    def reset(self) -> None:
        """Reset cross-chunk F0 continuity state."""
        self._last_voiced_cent = 0.0

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
        rmvpe_threshold: float = 0.032,
        fcpe_threshold: float = 0.006,
        rmvpe_sustain_threshold: float = 0.020,
        fcpe_sustain_threshold: float = 0.004,
        initial_prev_cent: float = 0.0,
    ) -> torch.Tensor:
        """Fuse RMVPE and FCPE F0/confidence predictions with hysteresis and causal continuity."""
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

        # Calibrate relative to sustain thresholds so sustain-band frames retain positive weights
        s1 = cls._calibrate_confidence(c1, rmvpe_sustain_threshold, 0.08)
        s2 = cls._calibrate_confidence(c2, fcpe_sustain_threshold, 0.035)

        # Gently boost FCPE on the rightmost 4 frames (40ms) where RMVPE's backward BiGRU lacks future context
        tail_len = min(4, T)
        if tail_len > 0:
            ramp = torch.tensor(
                [1.05, 1.12, 1.20, 1.30][-tail_len:], dtype=s2.dtype, device=s2.device
            )
            s2 = s2.clone()
            s2[-tail_len:] = torch.clamp(s2[-tail_len:] * ramp, max=1.0)

        v1_high = (f1 > 0.0) & (c1 >= rmvpe_threshold)
        v2_high = (f2 > 0.0) & (c2 >= fcpe_threshold)
        v1_low = (f1 > 0.0) & (c1 >= rmvpe_sustain_threshold)
        v2_low = (f2 > 0.0) & (c2 >= fcpe_sustain_threshold)
        both_high = v1_high & v2_high

        cent1 = torch.where(
            v1_low, 1200.0 * torch.log2(torch.clamp(f1, min=10.0) / 10.0), torch.zeros_like(f1)
        )
        cent2 = torch.where(
            v2_low, 1200.0 * torch.log2(torch.clamp(f2, min=10.0) / 10.0), torch.zeros_like(f2)
        )
        diff_cents = torch.abs(cent1 - cent2)

        w_sum = torch.clamp(s1 + s2, min=1e-6)
        cent_fused = (s1 * cent1 + s2 * cent2) / w_sum

        need_arb = (both_high & (diff_cents > 50.0)) | (v1_low ^ v2_low) | ((v1_low | v2_low) & ~both_high)
        if not bool(torch.any(need_arb).item()):
            return torch.where(
                both_high,
                10.0 * torch.pow(2.0, cent_fused / 1200.0),
                torch.zeros_like(f1),
            )

        cent_out = torch.where(v1_low & v2_low, cent_fused, torch.where(v1_low, cent1, cent2))
        voiced_out = torch.zeros_like(v1_low)

        v1_high_l = v1_high.tolist()
        v2_high_l = v2_high.tolist()
        v1_low_l = v1_low.tolist()
        v2_low_l = v2_low.tolist()
        cent1_l = cent1.tolist()
        cent2_l = cent2.tolist()
        diff_l = diff_cents.tolist()
        s1_l = s1.tolist()
        s2_l = s2.tolist()
        cent_fused_l = cent_fused.tolist()
        cent_out_l = cent_out.tolist()
        voiced_out_l = voiced_out.tolist()

        prev_cent = float(initial_prev_cent) if initial_prev_cent > 0.0 else 0.0
        unvoiced_run = 0 if prev_cent > 0.0 else 4

        for t in range(T):
            if v1_high_l[t] and v2_high_l[t]:
                if diff_l[t] <= 50.0:
                    c_val = cent_fused_l[t]
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
                voiced_out_l[t] = True
                prev_cent = c_val
                unvoiced_run = 0
                continue

            if v1_high_l[t] or v2_high_l[t]:
                if v1_low_l[t] and v2_low_l[t] and diff_l[t] <= 50.0:
                    c_val = cent_fused_l[t]
                    cent_out_l[t] = c_val
                    voiced_out_l[t] = True
                    prev_cent = c_val
                    unvoiced_run = 0
                else:
                    c_act = cent1_l[t] if v1_high_l[t] else cent2_l[t]
                    s_act = s1_l[t] if v1_high_l[t] else s2_l[t]
                    if prev_cent > 0.0 and abs(c_act - prev_cent) <= 180.0:
                        accept = s_act >= 0.18
                    else:
                        accept = s_act >= 0.45

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
                continue

            if v1_low_l[t] or v2_low_l[t]:
                # Hysteresis sustain band: allow keeping voice alive when continuous with active pitch
                accepted = False
                c_cand = 0.0
                if prev_cent > 0.0 and unvoiced_run <= 2:
                    if v1_low_l[t] and v2_low_l[t] and diff_l[t] <= 60.0:
                        c_cand = cent_fused_l[t]
                        accepted = abs(c_cand - prev_cent) <= 160.0
                    else:
                        if v1_low_l[t] and v2_low_l[t]:
                            if abs(cent1_l[t] - prev_cent) <= abs(cent2_l[t] - prev_cent):
                                c_cand, s_cand = cent1_l[t], s1_l[t]
                            else:
                                c_cand, s_cand = cent2_l[t], s2_l[t]
                        elif v1_low_l[t]:
                            c_cand, s_cand = cent1_l[t], s1_l[t]
                        else:
                            c_cand, s_cand = cent2_l[t], s2_l[t]
                        accepted = abs(c_cand - prev_cent) <= 140.0 and s_cand >= 0.08

                if accepted:
                    cent_out_l[t] = c_cand
                    voiced_out_l[t] = True
                    prev_cent = c_cand
                    unvoiced_run = 0
                else:
                    voiced_out_l[t] = False
                    unvoiced_run += 1
                    if unvoiced_run > 3:
                        prev_cent = 0.0
                continue

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
                    mel_fcpe, threshold=self.fcpe_sustain_threshold
                )
                f0_fcpe.record_stream(cur_stream)
                conf_fcpe.record_stream(cur_stream)

            f0_rmvpe, conf_rmvpe = self.rmvpe.infer_from_audio_t_with_confidence(
                audio, threshold=self.rmvpe_sustain_threshold
            )
            cur_stream.wait_stream(self._fcpe_stream)
        else:
            f0_rmvpe, conf_rmvpe = self.rmvpe.infer_from_audio_t_with_confidence(
                audio, threshold=self.rmvpe_sustain_threshold
            )
            mel_fcpe = self.fcpe_mel_extractor(audio.unsqueeze(0).float())
            f0_fcpe, conf_fcpe = self.fcpe.infer_with_confidence(
                mel_fcpe, threshold=self.fcpe_sustain_threshold
            )

        f0_out = self.fuse_predictions(
            f0_rmvpe,
            conf_rmvpe,
            f0_fcpe,
            conf_fcpe,
            rmvpe_threshold=self.rmvpe_threshold,
            fcpe_threshold=self.fcpe_threshold,
            rmvpe_sustain_threshold=self.rmvpe_sustain_threshold,
            fcpe_sustain_threshold=self.fcpe_sustain_threshold,
            initial_prev_cent=self._last_voiced_cent,
        )

        if f0_out.numel() > 0:
            tail_l = f0_out[-min(4, f0_out.shape[0]):].tolist()
            last_hz = 0.0
            for val in reversed(tail_l):
                if val > 0.0:
                    last_hz = float(val)
                    break
            self._last_voiced_cent = (
                1200.0 * math.log2(max(last_hz, 10.0) / 10.0) if last_hz > 0.0 else 0.0
            )
        else:
            self._last_voiced_cent = 0.0

        return f0_out.to(dtype=audio.dtype)
