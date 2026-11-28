import math
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F

from const import PitchExtractorType
from voice_changer.common.FCPE import spawn_infer_model_from_pt
from voice_changer.common.MelExtractorFcpe import Wav2MelModule
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
from voice_changer.common.rmvpe.rmvpe import RMVPE
from voice_changer.pitch_extractor.PitchExtractor import PitchExtractor

_TAIL_RAMP_NP = np.array([1.05, 1.12, 1.20, 1.30], dtype=np.float32)


class RMVPEFcpePitchExtractor(PitchExtractor):
    """Hybrid pitch extractor combining RMVPE/HPA-RMVPE and FCPE via calibrated confidence fusion and causal continuity arbitration."""

    def __init__(self, rmvpe_file: str, fcpe_file: str, use_hpa: bool = False):
        super().__init__()
        self.type: PitchExtractorType = "hpa_rmvpe_fcpe" if use_hpa else "rmvpe_fcpe"

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
            use_jit_compile=False if use_hpa else device_manager.use_jit_compile(),
            device=self.device,
            dtype=device_manager.get_torch_dtype(),
            hpa=use_hpa,
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

    @staticmethod
    def _calibrate_confidence_np(
        conf: np.ndarray, threshold: float, scale_k: float
    ) -> np.ndarray:
        """NumPy equivalent of _calibrate_confidence operating on CPU buffers."""
        above = np.maximum(conf - np.float32(threshold), np.float32(0.0))
        return above / (above + np.float32(scale_k + 1e-6))

    @classmethod
    def _fuse_predictions_with_conf_np(
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
        hop_frames: Optional[int] = None,
        tail_frames: int = 0,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """Fuse RMVPE and FCPE F0/confidence predictions using a single batched D2H copy."""
        f1 = f0_rmvpe.reshape(-1).float()
        c1 = conf_rmvpe.reshape(-1).float()
        f2 = f0_fcpe.reshape(-1).float()
        c2 = conf_fcpe.reshape(-1).float()

        T = f1.shape[0]
        if T == 0:
            empty = np.empty(0, dtype=np.float32)
            return empty, 0.0, empty

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

        # Pack all 4 aligned 1D tensors into a single D2H transfer (eliminates per-tensor syncs)
        packed = torch.stack((f1, c1, f2, c2), dim=0).detach().cpu().numpy()
        f1_np, c1_np, f2_np, c2_np = packed[0], packed[1], packed[2], packed[3]

        # Calibrate relative to sustain thresholds so sustain-band frames retain positive weights
        s1 = cls._calibrate_confidence_np(c1_np, rmvpe_sustain_threshold, 0.08)
        s2 = cls._calibrate_confidence_np(c2_np, fcpe_sustain_threshold, 0.035)

        # Gently boost FCPE on the rightmost 4 frames (40ms) where RMVPE's backward BiGRU lacks future context
        tail_len = min(4, T)
        if tail_len > 0:
            s2 = s2.copy()
            s2[-tail_len:] = np.minimum(s2[-tail_len:] * _TAIL_RAMP_NP[-tail_len:], np.float32(1.0))

        v1_high = (f1_np > 0.0) & (c1_np >= rmvpe_threshold)
        v2_high = (f2_np > 0.0) & (c2_np >= fcpe_threshold)
        v1_low = (f1_np > 0.0) & (c1_np >= rmvpe_sustain_threshold)
        v2_low = (f2_np > 0.0) & (c2_np >= fcpe_sustain_threshold)
        both_high = v1_high & v2_high

        cent1 = np.where(
            v1_low,
            np.float32(1200.0) * np.log2(np.maximum(f1_np, np.float32(10.0)) / np.float32(10.0)),
            np.float32(0.0),
        )
        cent2 = np.where(
            v2_low,
            np.float32(1200.0) * np.log2(np.maximum(f2_np, np.float32(10.0)) / np.float32(10.0)),
            np.float32(0.0),
        )
        diff_cents = np.abs(cent1 - cent2)

        w_sum = np.maximum(s1 + s2, np.float32(1e-6))
        cent_fused = (s1 * cent1 + s2 * cent2) / w_sum

        # Only apply previous-chunk tail cent at t=0 when the entire input is contiguous new frames.
        # When hop_frames < T, t=0 lies ~200-300ms in the past; let the prefix propagate real past pitch forward.
        hist_end = (T - hop_frames) if (hop_frames is not None and 0 < hop_frames < T) else 0
        seed_at_start = float(initial_prev_cent) if (hist_end == 0 and initial_prev_cent > 0.0) else 0.0

        need_arb = (both_high & (diff_cents > 50.0)) | (v1_low ^ v2_low) | ((v1_low | v2_low) & ~both_high)
        if not bool(np.any(need_arb)):
            f0_np = np.where(
                both_high,
                np.float32(10.0) * np.power(np.float32(2.0), cent_fused / np.float32(1200.0)),
                np.float32(0.0),
            ).astype(np.float32, copy=False)
        else:
            cent_out = np.where(v1_low & v2_low, cent_fused, np.where(v1_low, cent1, cent2))

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
            voiced_out_l = [False] * T

            prev_cent = seed_at_start
            unvoiced_run = 0 if prev_cent > 0.0 else 4

            for t in range(T):
                if t == hist_end and hist_end > 0 and hist_end <= 2 and prev_cent == 0.0 and initial_prev_cent > 0.0:
                    prev_cent = float(initial_prev_cent)
                    unvoiced_run = 0

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
                        is_tail = t >= T - 4
                        max_cont_cents = 260.0 if (is_tail and v2_high_l[t]) else 220.0
                        if prev_cent > 0.0 and unvoiced_run <= 2:
                            if abs(c_act - prev_cent) <= max_cont_cents:
                                accept = True
                            else:
                                accept = s_act >= 0.45
                        else:
                            accept = s_act >= 0.30

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
                        if v1_low_l[t] and v2_low_l[t] and diff_l[t] <= 80.0:
                            c_cand = cent_fused_l[t]
                            accepted = abs(c_cand - prev_cent) <= 200.0
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
                            accepted = abs(c_cand - prev_cent) <= 180.0 and s_cand >= 0.035

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

            cent_final = np.asarray(cent_out_l, dtype=np.float32)
            voiced_final = np.asarray(voiced_out_l, dtype=bool)
            f0_np = np.where(
                voiced_final,
                np.float32(10.0) * np.power(np.float32(2.0), cent_final / np.float32(1200.0)),
                np.float32(0.0),
            ).astype(np.float32, copy=False)

        conf_np = np.where(
            f0_np > np.float32(0.0),
            np.clip(np.maximum(s1, s2), np.float32(0.0), np.float32(1.0)),
            np.float32(0.0),
        ).astype(np.float32, copy=False)

        commit_end = T - max(0, int(tail_frames))
        if commit_end <= 0:
            commit_end = T
        commit_slice = f0_np[max(0, commit_end - 4) : commit_end]
        last_hz = 0.0
        for val in reversed(commit_slice.tolist()):
            if val > 0.0:
                last_hz = float(val)
                break
        last_voiced_cent = (
            1200.0 * math.log2(max(last_hz, 10.0) / 10.0) if last_hz > 0.0 else 0.0
        )
        return f0_np, last_voiced_cent, conf_np

    @classmethod
    def _fuse_predictions_np(
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
        hop_frames: Optional[int] = None,
        tail_frames: int = 0,
    ) -> tuple[np.ndarray, float]:
        f0_np, last_voiced_cent, _ = cls._fuse_predictions_with_conf_np(
            f0_rmvpe,
            conf_rmvpe,
            f0_fcpe,
            conf_fcpe,
            rmvpe_threshold=rmvpe_threshold,
            fcpe_threshold=fcpe_threshold,
            rmvpe_sustain_threshold=rmvpe_sustain_threshold,
            fcpe_sustain_threshold=fcpe_sustain_threshold,
            initial_prev_cent=initial_prev_cent,
            hop_frames=hop_frames,
            tail_frames=tail_frames,
        )
        return f0_np, last_voiced_cent

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
        hop_frames: Optional[int] = None,
        tail_frames: int = 0,
    ) -> torch.Tensor:
        """Fuse RMVPE and FCPE F0/confidence predictions with hysteresis and causal continuity."""
        f0_np, _, _ = cls._fuse_predictions_with_conf_np(
            f0_rmvpe,
            conf_rmvpe,
            f0_fcpe,
            conf_fcpe,
            rmvpe_threshold=rmvpe_threshold,
            fcpe_threshold=fcpe_threshold,
            rmvpe_sustain_threshold=rmvpe_sustain_threshold,
            fcpe_sustain_threshold=fcpe_sustain_threshold,
            initial_prev_cent=initial_prev_cent,
            hop_frames=hop_frames,
            tail_frames=tail_frames,
        )
        if f0_np.size == 0:
            return torch.empty(0, dtype=torch.float32, device=f0_rmvpe.device)
        return torch.from_numpy(f0_np).to(device=f0_rmvpe.device)

    def extract_with_confidence(
        self,
        audio: torch.Tensor,
        sr: int,
        window: int,
        hop_frames: Optional[int] = None,
        tail_frames: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

        f0_np, last_voiced_cent, conf_np = self._fuse_predictions_with_conf_np(
            f0_rmvpe,
            conf_rmvpe,
            f0_fcpe,
            conf_fcpe,
            rmvpe_threshold=self.rmvpe_threshold,
            fcpe_threshold=self.fcpe_threshold,
            rmvpe_sustain_threshold=self.rmvpe_sustain_threshold,
            fcpe_sustain_threshold=self.fcpe_sustain_threshold,
            initial_prev_cent=self._last_voiced_cent,
            hop_frames=hop_frames,
            tail_frames=tail_frames,
        )
        self._last_voiced_cent = last_voiced_cent

        if f0_np.size == 0:
            empty = torch.empty(0, dtype=audio.dtype, device=audio.device)
            return empty, empty
        return (
            torch.from_numpy(f0_np).to(device=audio.device, dtype=audio.dtype),
            torch.from_numpy(conf_np).to(device=audio.device, dtype=torch.float32),
        )

    def extract(
        self,
        audio: torch.Tensor,
        sr: int,
        window: int,
        hop_frames: Optional[int] = None,
        tail_frames: int = 0,
    ) -> torch.Tensor:
        f0, _ = self.extract_with_confidence(
            audio, sr, window, hop_frames=hop_frames, tail_frames=tail_frames
        )
        return f0
