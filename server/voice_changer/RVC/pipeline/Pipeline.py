from concurrent.futures import ThreadPoolExecutor
import logging
import math
import sys
from typing import Optional

from faiss import IndexIVFFlat
import faiss.contrib.torch_utils
import torch
import torch.nn.functional as F
from torchaudio import transforms as tat

from const import F0_MEL_MIN, F0_MEL_MAX, HUBERT_SAMPLE_RATE, WINDOW_SIZE
from voice_changer.audio_effects.AudioEffectsManager import AudioEffectsManager
from voice_changer.common.AudioDSP import map_f0_distribution
from voice_changer.common.TorchUtils import circular_write
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
from voice_changer.embedder.Embedder import Embedder
from voice_changer.pitch_extractor.PitchExtractor import PitchExtractor
from voice_changer.RVC.inferencer.Inferencer import Inferencer
from voice_changer.utils.SystemPriority import register_current_thread_mmcss
from voice_changer.utils.Timer import Timer2

logger = logging.getLogger(__name__)

_AUTO_TUNE_MIDI_MAP = {
    "G4": 67, "G#4": 68, "A4": 69, "A#4": 70, "B4": 71,
    "C5": 72, "C#5": 73, "D5": 74, "D#5": 75, "E5": 76,
    "F5": 77, "F#5": 78, "G5": 79, "A5": 81, "C6": 84,
}


class Pipeline:
    embedder: Embedder
    inferencer: Inferencer
    pitchExtractor: PitchExtractor

    index: IndexIVFFlat | None
    index_reconstruct: torch.Tensor | None

    model_sr: int
    device: torch.device
    audio_effects_manager: Optional[AudioEffectsManager]

    def __init__(
        self,
        embedder: Embedder,
        inferencer: Inferencer,
        pitchExtractor: PitchExtractor,
        index: IndexIVFFlat | None,
        index_reconstruct: torch.Tensor | None,
        use_f0: bool,
        model_sr: int,
        embChannels: int,
        audio_effects_manager: Optional[AudioEffectsManager] = None,
    ):
        self.embedder = embedder
        self.inferencer = inferencer
        self.pitchExtractor = pitchExtractor
        logger.info("GENERATE INFERENCER" + str(self.inferencer))
        logger.info("GENERATE EMBEDDER" + str(self.embedder))
        logger.info("GENERATE PITCH EXTRACTOR" + str(self.pitchExtractor))

        self.device_manager = DeviceManager.get_instance()
        self.device = self.device_manager.device
        self.is_half = self.device_manager.use_fp16()
        self.dtype = self.device_manager.get_torch_dtype()

        self.index = index
        self.index_reconstruct = index_reconstruct
        self.use_index = index is not None and self.index_reconstruct is not None
        self.use_gpu_index = sys.platform == 'linux' and '+cu' in torch.__version__ and self.device.type == 'cuda'
        self.use_f0 = use_f0

        self.model_sr = model_sr
        self.model_window = model_sr // 100

        # GPU-resident FAISS replacement for Windows/MPS when faiss-gpu is unavailable
        self.use_torch_gpu_index = False
        self._gpu_index_norms: Optional[torch.Tensor] = None
        if self.use_index and not self.use_gpu_index and self.device.type in ('cuda', 'mps'):
            try:
                self.index_reconstruct = self.index_reconstruct.to(
                    device=self.device, dtype=self.dtype
                ).contiguous()
                ntotal = self.index_reconstruct.shape[0]
                norms = torch.empty(ntotal, dtype=torch.float32, device=self.device)
                chunk_sz = 65536
                for start in range(0, ntotal, chunk_sz):
                    end = min(start + chunk_sz, ntotal)
                    norms[start:end] = torch.sum(self.index_reconstruct[start:end].float() ** 2, dim=1)
                self._gpu_index_norms = norms
                self.use_torch_gpu_index = True
                logger.info(
                    f"Enabled zero-copy PyTorch GPU index search ({ntotal} vectors, dtype={self.dtype})."
                )
            except Exception as e:
                logger.warning(f"Failed to prepare PyTorch GPU index search, falling back to CPU FAISS: {e}")
                self.use_torch_gpu_index = False
                self._gpu_index_norms = None

        self.settings = None
        self.last_f0_smoothed: Optional[float] = None

        # Share external AudioEffectsManager when provided (used for autoTune parameters)
        if audio_effects_manager is not None:
            self.audio_effects_manager = audio_effects_manager
        else:
            try:
                self.audio_effects_manager = AudioEffectsManager()
            except Exception as e:
                logger.error(f"Failed to initialize AudioEffectsManager: {e}")
                self.audio_effects_manager = None

        # Persistent worker thread for Pitch || Embedder concurrency
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="PipelineEmbedder")

        # Persistent device streams from DeviceManager
        self._pitch_stream = self.device_manager.pitch_stream
        self._embedder_stream = self.device_manager.embedder_stream

        # Cached small scalar tensors & formant resamplers
        self._cached_plen_val: Optional[int] = None
        self._cached_plen_tensor: Optional[torch.Tensor] = None
        self._cached_sid_val: Optional[int] = None
        self._cached_sid_tensor: Optional[torch.Tensor] = None
        self._formant_resamplers: dict[tuple[int, int], tat.Resample] = {}

    def __del__(self):
        executor = getattr(self, "_executor", None)
        if executor is not None:
            try:
                executor.shutdown(wait=False)
            except Exception:
                pass

    def getPipelineInfo(self):
        inferencerInfo = self.inferencer.getInferencerInfo() if self.inferencer else {}
        embedderInfo = self.embedder.get_embedder_info()
        pitchExtractorInfo = self.pitchExtractor.getPitchExtractorInfo()
        return {"inferencer": inferencerInfo, "embedder": embedderInfo, "pitchExtractor": pitchExtractorInfo}

    def setPitchExtractor(self, pitchExtractor: PitchExtractor):
        self.pitchExtractor = pitchExtractor

    def configure_audio_effects(self, settings: dict) -> None:
        """Configure audio effects from settings."""
        if self.audio_effects_manager is None:
            return
        from voice_changer.audio_effects.AudioEffectsConfig import AudioEffectsConfig
        AudioEffectsConfig.configure_effects_from_settings(self.audio_effects_manager, settings)

    def _get_active_autotune(self):
        if self.audio_effects_manager is None:
            return None
        for effect in self.audio_effects_manager.input_effects:
            if effect.effect_type == "autoTune" and effect.is_enabled():
                return effect
        return None

    def _apply_autotune_tail(
        self, f0: torch.Tensor, auto_tune_effect, new_frames: int
    ) -> torch.Tensor:
        if new_frames <= 0 or f0.numel() == 0:
            return f0

        ceiling_str = auto_tune_effect.parameters.get("pitchCeiling", "Off")
        retune_speed = float(auto_tune_effect.parameters.get("retuneSpeed", 35.0))
        ceiling_midi = _AUTO_TUNE_MIDI_MAP.get(ceiling_str, 0)

        # Vectorized chromatic semitone snapping on device
        f0_tail = f0[-new_frames:].float()
        voiced_mask = f0_tail > 0
        midi = torch.where(
            voiced_mask,
            69.0 + 12.0 * torch.log2(torch.clamp(f0_tail, min=1e-3) / 440.0),
            f0_tail,
        )
        midi_snapped = torch.round(midi)
        if ceiling_midi > 0:
            midi_snapped = torch.clamp(midi_snapped, max=float(ceiling_midi))
        target_hz = torch.where(
            voiced_mask,
            440.0 * torch.pow(2.0, (midi_snapped - 69.0) / 12.0),
            torch.zeros_like(f0_tail),
        )

        # Sequential causal exponential smoothing over newly arrived frames only
        alpha = math.exp(-10.0 / max(1.0, retune_speed))
        inv_alpha = 1.0 - alpha
        voiced_list = voiced_mask.tolist()
        target_list = target_hz.tolist()

        last_val = self.last_f0_smoothed
        smoothed_list = [0.0] * len(target_list)
        for t in range(len(target_list)):
            if not voiced_list[t]:
                smoothed_list[t] = 0.0
                continue
            tgt = target_list[t]
            if t == 0:
                val = tgt if (last_val is None or last_val <= 0.0) else (alpha * last_val + inv_alpha * tgt)
            else:
                val = tgt if not voiced_list[t - 1] else (alpha * smoothed_list[t - 1] + inv_alpha * tgt)
            smoothed_list[t] = val
            last_val = val

        self.last_f0_smoothed = last_val
        f0_out = f0.clone()
        f0_out[-new_frames:] = torch.tensor(smoothed_list, dtype=f0.dtype, device=f0.device)
        return f0_out

    @staticmethod
    def _stabilize_f0(
        f0: torch.Tensor,
        max_gap_frames: int = 3,
        max_gap_semitones: float = 4.0,
        pitch_mode: str = "speech",
    ) -> torch.Tensor:
        """Bridge short interior V-UV-V dropouts via log-linear interpolation and apply spike-safe voiced filtering."""
        T = f0.shape[0]
        if T < 3:
            return f0

        is_singing = str(pitch_mode).strip().lower() == "singing"
        eff_gaps = min(2, max_gap_frames) if is_singing else max_gap_frames
        eff_semi = min(3.0, max_gap_semitones) if is_singing else max_gap_semitones

        x = f0.float().clone()
        orig = x.clone()
        max_ratio = 2.0 ** (eff_semi / 12.0)
        inv_max_ratio = 1.0 / max_ratio

        # Step A: Bridge short V-UV-V gaps (1..3 frames) using immutable `orig` masks
        if eff_gaps >= 1 and T >= 3:
            l1, m1, r1 = orig[:-2], orig[1:-1], orig[2:]
            ratio1 = r1 / torch.clamp(l1, min=1e-6)
            mask1 = (l1 > 0) & (m1 == 0) & (r1 > 0) & (ratio1 >= inv_max_ratio) & (ratio1 <= max_ratio)
            interp1 = torch.sqrt(torch.clamp(l1 * r1, min=1e-12))
            x[1:-1] = torch.where(mask1, interp1, x[1:-1])

        if eff_gaps >= 2 and T >= 4:
            l2, m2a, m2b, r2 = orig[:-3], orig[1:-2], orig[2:-1], orig[3:]
            ratio2 = r2 / torch.clamp(l2, min=1e-6)
            mask2 = (l2 > 0) & (m2a == 0) & (m2b == 0) & (r2 > 0) & (ratio2 >= inv_max_ratio) & (ratio2 <= max_ratio)
            log_l2 = torch.log(torch.clamp(l2, min=1e-6))
            log_r2 = torch.log(torch.clamp(r2, min=1e-6))
            x[1:-2] = torch.where(mask2, torch.exp((2.0 / 3.0) * log_l2 + (1.0 / 3.0) * log_r2), x[1:-2])
            x[2:-1] = torch.where(mask2, torch.exp((1.0 / 3.0) * log_l2 + (2.0 / 3.0) * log_r2), x[2:-1])

        if eff_gaps >= 3 and T >= 5:
            l3, m3a, m3b, m3c, r3 = orig[:-4], orig[1:-3], orig[2:-2], orig[3:-1], orig[4:]
            ratio3 = r3 / torch.clamp(l3, min=1e-6)
            mask3 = (
                (l3 > 0)
                & (m3a == 0)
                & (m3b == 0)
                & (m3c == 0)
                & (r3 > 0)
                & (ratio3 >= inv_max_ratio)
                & (ratio3 <= max_ratio)
            )
            log_l3 = torch.log(torch.clamp(l3, min=1e-6))
            log_r3 = torch.log(torch.clamp(r3, min=1e-6))
            x[1:-3] = torch.where(mask3, torch.exp(0.75 * log_l3 + 0.25 * log_r3), x[1:-3])
            x[2:-2] = torch.where(mask3, torch.exp(0.50 * log_l3 + 0.50 * log_r3), x[2:-2])
            x[3:-1] = torch.where(mask3, torch.exp(0.25 * log_l3 + 0.75 * log_r3), x[3:-1])

        # Step B: In speech mode, apply 3-frame median filter on fully-voiced windows;
        # in singing mode, preserve natural vibrato peaks and only suppress extreme >3 st single-frame spikes.
        left, mid, right = x[:-2], x[1:-1], x[2:]
        all_voiced = (left > 0) & (mid > 0) & (right > 0)
        med = torch.median(torch.stack([left, mid, right], dim=0), dim=0).values
        if is_singing:
            spike_r = 2.0 ** (3.0 / 12.0)
            neigh_r = 2.0 ** (2.0 / 12.0)
            l_safe = torch.clamp(left, min=1e-6)
            m_safe = torch.clamp(mid, min=1e-6)
            r_safe = torch.clamp(right, min=1e-6)
            is_spike = (
                all_voiced
                & ((m_safe / l_safe > spike_r) | (l_safe / m_safe > spike_r))
                & ((m_safe / r_safe > spike_r) | (r_safe / m_safe > spike_r))
                & (r_safe / l_safe <= neigh_r)
                & (l_safe / r_safe <= neigh_r)
            )
            apply_mask = is_spike
        else:
            apply_mask = all_voiced
        out = x.clone()
        out[1:-1] = torch.where(apply_mask, med, mid)
        return out.to(dtype=f0.dtype)

    def extract_pitch(
        self,
        audio: torch.Tensor,
        pitch: torch.Tensor | None,
        pitchf: torch.Tensor | None,
        f0_up_key: float,
        pitch_hop_frames: Optional[int] = None,
        pitch_scratch: Optional[torch.Tensor] = None,
        pitchf_scratch: Optional[torch.Tensor] = None,
        pitch_dist_cfg: Optional[dict] = None,
        voicing_conf: Optional[torch.Tensor] = None,
        voicing_conf_scratch: Optional[torch.Tensor] = None,
        tail_frames: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if getattr(self.pitchExtractor, "type", "") in ("rmvpe_fcpe", "hpa_rmvpe_fcpe") and hasattr(
            self.pitchExtractor, "extract_with_confidence"
        ):
            f0, raw_conf = self.pitchExtractor.extract_with_confidence(
                audio,
                HUBERT_SAMPLE_RATE,
                WINDOW_SIZE,
                hop_frames=pitch_hop_frames,
                tail_frames=tail_frames,
            )
        elif hasattr(self.pitchExtractor, "extract_with_confidence"):
            f0, raw_conf = self.pitchExtractor.extract_with_confidence(
                audio,
                HUBERT_SAMPLE_RATE,
                WINDOW_SIZE,
            )
        else:
            f0 = self.pitchExtractor.extract(
                audio,
                HUBERT_SAMPLE_RATE,
                WINDOW_SIZE,
            )
            raw_conf = (f0 > 0).float()

        pitch_mode = str(pitch_dist_cfg.get("pitch_mode", "speech")) if pitch_dist_cfg is not None else "speech"

        if (
            pitch_dist_cfg is not None
            and float(pitch_dist_cfg.get("src_mean", 0.0)) > 0.0
            and float(pitch_dist_cfg.get("tgt_mean", 0.0)) > 0.0
        ):
            f0 = self._stabilize_f0(f0, pitch_mode=pitch_mode)
            f0 = map_f0_distribution(
                f0,
                src_log2_mean=float(pitch_dist_cfg["src_mean"]),
                src_log2_std=float(pitch_dist_cfg.get("src_std", 0.0)),
                tgt_log2_mean=float(pitch_dist_cfg["tgt_mean"]),
                tgt_log2_std=float(pitch_dist_cfg.get("tgt_std", 0.0)),
                tgt_f0_p10=float(pitch_dist_cfg.get("tgt_p10", 0.0)),
                tgt_f0_p90=float(pitch_dist_cfg.get("tgt_p90", 0.0)),
                pitch_scale=float(pitch_dist_cfg.get("pitch_scale", 1.0)),
                pitch_offset_semitones=float(pitch_dist_cfg.get("pitch_offset", 0.0)),
                pitch_mode=pitch_mode,
            )
        else:
            f0 = f0 * (2 ** (f0_up_key / 12))
            f0 = self._stabilize_f0(f0, pitch_mode=pitch_mode)

        raw_conf = raw_conf.reshape(-1).float()
        if raw_conf.shape[0] != f0.shape[0]:
            raw_conf = (f0 > 0).float()
        else:
            raw_conf = torch.where(
                f0 > 0,
                torch.clamp(raw_conf, min=0.35, max=1.0),
                torch.zeros_like(raw_conf),
            )

        auto_tune_effect = self._get_active_autotune()
        hop_frames = (
            min(pitch_hop_frames, f0.shape[0])
            if (pitch_hop_frames is not None and pitch_hop_frames > 0)
            else f0.shape[0]
        )
        if auto_tune_effect is not None:
            f0 = self._apply_autotune_tail(f0, auto_tune_effect, hop_frames)

        f0_mel = 1127.0 * torch.log(1.0 + f0 / 700.0)
        f0_mel = torch.clip(
            (f0_mel - F0_MEL_MIN) * 254 / (F0_MEL_MAX - F0_MEL_MIN) + 1,
            1,
            255,
            out=f0_mel,
        )
        f0_coarse = torch.round(f0_mel, out=f0_mel).long()
        f0_coarse[f0 == 0] = 0

        if pitch is not None and pitchf is not None:
            if pitch_hop_frames is not None:
                # Shift buffers by exact hop of new audio frames to stay aligned with convert_buffer
                hop = min(max(0, pitch_hop_frames), pitchf.shape[0])
                if hop > 0:
                    keep = pitchf.shape[0] - hop
                    if keep > 0:
                        if pitch_scratch is not None and pitch_scratch.shape == pitch.shape:
                            pitch_scratch[:keep].copy_(pitch[hop:])
                            pitch[:keep].copy_(pitch_scratch[:keep])
                        else:
                            pitch[:keep] = pitch[hop:].detach().clone()
                        if pitchf_scratch is not None and pitchf_scratch.shape == pitchf.shape:
                            pitchf_scratch[:keep].copy_(pitchf[hop:])
                            pitchf[:keep].copy_(pitchf_scratch[:keep])
                        else:
                            pitchf[:keep] = pitchf[hop:].detach().clone()
                        if voicing_conf is not None:
                            if voicing_conf_scratch is not None and voicing_conf_scratch.shape == voicing_conf.shape:
                                voicing_conf_scratch[:keep].copy_(voicing_conf[hop:])
                                voicing_conf[:keep].copy_(voicing_conf_scratch[:keep])
                            else:
                                voicing_conf[:keep] = voicing_conf[hop:].detach().clone()

                    # When autoTune is active, only write the causally smoothed tail so prior frames stay smooth
                    write_len = hop if auto_tune_effect is not None else min(f0.shape[0], pitchf.shape[0])
                    if write_len > 0:
                        pitch[-write_len:] = f0_coarse[-write_len:].to(dtype=pitch.dtype)
                        pitchf[-write_len:] = f0[-write_len:].to(dtype=pitchf.dtype)
                        if voicing_conf is not None:
                            voicing_conf[-write_len:] = raw_conf[-write_len:].to(dtype=voicing_conf.dtype)
            else:
                circular_write(f0_coarse, pitch, scratch=pitch_scratch)
                circular_write(f0, pitchf, scratch=pitchf_scratch)
                if voicing_conf is not None:
                    circular_write(raw_conf, voicing_conf, scratch=voicing_conf_scratch)
        else:
            pitch = f0_coarse
            pitchf = f0
            voicing_conf = raw_conf

        if voicing_conf is None:
            voicing_conf = (pitchf > 0).float()

        return pitch.unsqueeze(0), pitchf.unsqueeze(0), voicing_conf.unsqueeze(0)

    def _search_index_with_gate(
        self, audio: torch.Tensor, top_k: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_f32 = audio.float()
        q_norms = torch.sum(q_f32 ** 2, dim=1, keepdim=True)
        if self.use_torch_gpu_index and self._gpu_index_norms is not None:
            q_match = audio.to(dtype=self.index_reconstruct.dtype)
            ntotal = self.index_reconstruct.shape[0]
            k_eff = min(top_k, ntotal)

            if ntotal <= 131072:
                dots = torch.matmul(q_match, self.index_reconstruct.t()).float()
                dist2 = (q_norms + self._gpu_index_norms.unsqueeze(0) - 2.0 * dots).clamp_min_(1e-8)
                score, ix = torch.topk(dist2, k=k_eff, dim=1, largest=False)
            else:
                chunk_sz = 65536
                cand_scores = []
                cand_indices = []
                for start in range(0, ntotal, chunk_sz):
                    end = min(start + chunk_sz, ntotal)
                    dots = torch.matmul(q_match, self.index_reconstruct[start:end].t()).float()
                    dist2 = (
                        q_norms + self._gpu_index_norms[start:end].unsqueeze(0) - 2.0 * dots
                    ).clamp_min_(1e-8)
                    sc, idx = torch.topk(dist2, k=min(k_eff, end - start), dim=1, largest=False)
                    cand_scores.append(sc)
                    cand_indices.append(idx + start)
                all_scores = torch.cat(cand_scores, dim=1)
                all_indices = torch.cat(cand_indices, dim=1)
                score, pos = torch.topk(all_scores, k=k_eff, dim=1, largest=False)
                ix = torch.gather(all_indices, 1, pos)

            rel_dist = score[:, 0] / (q_norms.squeeze(1) + 1e-6)
            gate = torch.clamp((1.2 - rel_dist) / 0.6, 0.0, 1.0).unsqueeze(-1)
            if k_eff == 1:
                return self.index_reconstruct[ix.squeeze(1)], gate
            weight = torch.square(1.0 / (score + 1e-6))
            weight /= weight.sum(dim=1, keepdim=True)
            return torch.sum(self.index_reconstruct[ix].float() * weight.unsqueeze(2), dim=1), gate

        # CPU / Linux faiss-gpu fallback
        query = audio if self.use_gpu_index else audio.detach().cpu()
        if top_k == 1:
            score, ix = self.index.search(query, 1)
            score = score.to(self.device).float()
            ix = ix.to(self.device)
            rel_dist = score[:, 0] / (q_norms.squeeze(1) + 1e-6)
            gate = torch.clamp((1.2 - rel_dist) / 0.6, 0.0, 1.0).unsqueeze(-1)
            return self.index_reconstruct[ix.squeeze(1)], gate

        score, ix = self.index.search(query, k=top_k)
        score = score.to(self.device).float()
        ix = ix.to(self.device)
        rel_dist = score[:, 0] / (q_norms.squeeze(1) + 1e-6)
        gate = torch.clamp((1.2 - rel_dist) / 0.6, 0.0, 1.0).unsqueeze(-1)
        weight = torch.square(1.0 / (score + 1e-6))
        weight /= weight.sum(dim=1, keepdim=True)
        return torch.sum(self.index_reconstruct[ix].float() * weight.unsqueeze(2), dim=1), gate

    def _search_index(self, audio: torch.Tensor, top_k: int = 1) -> torch.Tensor:
        retrieved, _ = self._search_index_with_gate(audio, top_k=top_k)
        return retrieved

    def _upscale(self, feats: torch.Tensor) -> torch.Tensor:
        return torch.stack([feats, feats], dim=2).flatten(1, 2)

    def _get_cached_plen(self, audio_feats_len: int) -> torch.Tensor:
        if (
            self._cached_plen_tensor is None
            or self._cached_plen_val != audio_feats_len
            or self._cached_plen_tensor.device != self.device
        ):
            self._cached_plen_val = audio_feats_len
            self._cached_plen_tensor = torch.tensor([audio_feats_len], device=self.device, dtype=torch.int64)
        return self._cached_plen_tensor

    def _get_cached_sid(self, sid: int) -> torch.Tensor:
        if (
            self._cached_sid_tensor is None
            or self._cached_sid_val != sid
            or self._cached_sid_tensor.device != self.device
        ):
            self._cached_sid_val = sid
            self._cached_sid_tensor = torch.tensor([sid], device=self.device, dtype=torch.int64)
        return self._cached_sid_tensor

    def exec(
        self,
        sid: int,
        audio: torch.Tensor,
        pitch: torch.Tensor | None,
        pitchf: torch.Tensor | None,
        f0_up_key: float,
        index_rate: float,
        audio_feats_len: int,
        silence_front: int,
        embOutputLayer: int,
        useFinalProj: bool,
        skip_head: int,
        return_length: int,
        protect: float = 0.5,
        pitch_hop_frames: Optional[int] = None,
        pitch_scratch: Optional[torch.Tensor] = None,
        pitchf_scratch: Optional[torch.Tensor] = None,
        formant_shift: float = 0.0,
        pitch_dist_cfg: Optional[dict] = None,
        voicing_conf: Optional[torch.Tensor] = None,
        voicing_conf_scratch: Optional[torch.Tensor] = None,
        tail_frames: int = 0,
    ) -> torch.Tensor:
        with Timer2("Pipeline-Exec", False) as t:
            assert audio.dim() == 1, audio.dim()

            standalone_active = False
            if not getattr(self.inferencer, "is_refinegan", False):
                try:
                    from voice_changer.vocoder.VocoderManager import VocoderManager

                    cand = VocoderManager.get_instance().get_active_vocoder()
                    standalone_active = cand is not None and getattr(cand, "is_loaded", False)
                except Exception:
                    standalone_active = False

            effective_formant_shift = 0.0 if standalone_active else float(formant_shift)
            formant_factor = 2.0 ** (effective_formant_shift / 12.0)
            scaled_window = max(1, int(math.floor(formant_factor * self.model_window)))
            if scaled_window != self.model_window and return_length > 0:
                formant_length = max(1, int(math.ceil(return_length * formant_factor)))
                effective_f0_up_key = f0_up_key - effective_formant_shift
            else:
                formant_length = return_length
                effective_f0_up_key = f0_up_key
            t.record("pre-process")

            # Pitch extraction and feature extraction in parallel via persistent executor
            audio_emb = audio.to(dtype=self.dtype).view(1, -1)
            if self.use_f0:
                def _run_feats_worker():
                    try:
                        register_current_thread_mmcss("high")
                    except Exception:
                        pass
                    if self._embedder_stream is not None:
                        with torch.cuda.stream(self._embedder_stream):
                            return self.embedder.extract_features(
                                audio_emb, embOutputLayer, useFinalProj
                            )
                    return self.embedder.extract_features(
                        audio_emb, embOutputLayer, useFinalProj
                    )

                if self._embedder_stream is not None:
                    self._embedder_stream.wait_stream(torch.cuda.current_stream(self.device))
                fut_feats = self._executor.submit(_run_feats_worker)

                pitch_ok = False
                try:
                    if self._pitch_stream is not None:
                        cur = torch.cuda.current_stream(self.device)
                        self._pitch_stream.wait_stream(cur)
                        with torch.cuda.stream(self._pitch_stream):
                            pitch, pitchf, voicing_conf = self.extract_pitch(
                                audio[silence_front:],
                                pitch,
                                pitchf,
                                effective_f0_up_key,
                                pitch_hop_frames=pitch_hop_frames,
                                pitch_scratch=pitch_scratch,
                                pitchf_scratch=pitchf_scratch,
                                pitch_dist_cfg=pitch_dist_cfg,
                                voicing_conf=voicing_conf,
                                voicing_conf_scratch=voicing_conf_scratch,
                                tail_frames=tail_frames,
                            )
                        cur.wait_stream(self._pitch_stream)
                    else:
                        pitch, pitchf, voicing_conf = self.extract_pitch(
                            audio[silence_front:],
                            pitch,
                            pitchf,
                            effective_f0_up_key,
                            pitch_hop_frames=pitch_hop_frames,
                            pitch_scratch=pitch_scratch,
                            pitchf_scratch=pitchf_scratch,
                            pitch_dist_cfg=pitch_dist_cfg,
                            voicing_conf=voicing_conf,
                            voicing_conf_scratch=voicing_conf_scratch,
                            tail_frames=tail_frames,
                        )
                    pitch_ok = True
                finally:
                    if pitch_ok:
                        feats = fut_feats.result()
                        if self._embedder_stream is not None:
                            torch.cuda.current_stream(self.device).wait_stream(self._embedder_stream)
                    else:
                        try:
                            fut_feats.result()
                        except Exception as feats_err:
                            logger.debug(f"Embedder worker also failed during pitch error cleanup: {feats_err}")
                        if self._embedder_stream is not None:
                            try:
                                torch.cuda.current_stream(self.device).wait_stream(self._embedder_stream)
                            except Exception:
                                pass
            else:
                pitch, pitchf, voicing_conf = None, None, None
                feats = self.embedder.extract_features(audio_emb, embOutputLayer, useFinalProj)

            feats = torch.cat((feats, feats[:, -1:, :]), 1)
            t.record("extract-pitch-and-feats")

            # Index feature retrieval & protect blending (single upscale)
            is_active_index = self.use_index and index_rate > 0
            use_protect = protect < 0.5

            if is_active_index:
                skip_offset = skip_head // 2
                feats_orig_slice = feats[0, skip_offset:]
                index_audio, index_gate = self._search_index_with_gate(feats_orig_slice.float(), 8)
                delta_raw = (index_audio.to(dtype=feats.dtype) - feats_orig_slice) * index_gate.to(dtype=feats.dtype)
                if delta_raw.shape[0] >= 3:
                    d_pad = F.pad(
                        delta_raw.transpose(0, 1).unsqueeze(0), (1, 1), mode="replicate"
                    ).squeeze(0).transpose(0, 1)
                    delta_raw = 0.15 * d_pad[:-2] + 0.70 * d_pad[1:-1] + 0.15 * d_pad[2:]

                if self.use_f0 and use_protect:
                    delta = delta_raw.unsqueeze(0)
                    feats = self._upscale(feats)[:, :audio_feats_len, :]
                    pitch = pitch[:, -audio_feats_len:]
                    pitchf = pitchf[:, -audio_feats_len:]
                    voicing_conf = voicing_conf[:, -audio_feats_len:]

                    up_start = skip_offset * 2
                    rem = audio_feats_len - up_start
                    if rem > 0:
                        delta_up = self._upscale(delta)[:, :rem, :]
                        conf_slice = voicing_conf[:, up_start:audio_feats_len].clamp(0.0, 1.0)
                        pitchff = (protect + (1.0 - protect) * conf_slice).to(dtype=feats.dtype).unsqueeze(-1)
                        feats[:, up_start:audio_feats_len, :] = (
                            feats[:, up_start:audio_feats_len, :]
                            + delta_up * (index_rate * pitchff)
                        )
                else:
                    feats[0, skip_offset:] = feats_orig_slice + delta_raw * index_rate
                    feats = self._upscale(feats)[:, :audio_feats_len, :]
                    if self.use_f0:
                        pitch = pitch[:, -audio_feats_len:]
                        pitchf = pitchf[:, -audio_feats_len:]
            else:
                feats = self._upscale(feats)[:, :audio_feats_len, :]
                if self.use_f0:
                    pitch = pitch[:, -audio_feats_len:]
                    pitchf = pitchf[:, -audio_feats_len:]

            p_len = self._get_cached_plen(audio_feats_len)
            sid_t = self._get_cached_sid(sid)
            pitchf_infer = (
                pitchf * (formant_length / float(return_length))
                if (self.use_f0 and pitchf is not None and formant_length != return_length and return_length > 0)
                else pitchf
            )
            t.record("mid-precess")

            out_audio = self.inferencer.infer(
                feats, p_len, pitch, pitchf_infer, sid_t, skip_head, return_length, formant_length
            ).float()
            t.record("infer")

            uses_standalone = getattr(self.inferencer, "uses_standalone_vocoder", lambda: False)()
            if not uses_standalone and scaled_window != self.model_window and return_length > 0 and out_audio.numel() > 0:
                resampler_key = (scaled_window, self.model_window)
                resampler = self._formant_resamplers.get(resampler_key)
                if resampler is None:
                    resampler = tat.Resample(
                        orig_freq=scaled_window,
                        new_freq=self.model_window,
                        lowpass_filter_width=16,
                        rolloff=0.94,
                        resampling_method="sinc_interp_kaiser",
                        dtype=torch.float32,
                    ).to(self.device)
                    self._formant_resamplers[resampler_key] = resampler
                slice_len = min(out_audio.shape[-1], return_length * scaled_window)
                out_audio = resampler(out_audio[:slice_len])
                target_len = return_length * self.model_window
                if out_audio.shape[-1] > target_len:
                    out_audio = out_audio[:target_len]
                elif out_audio.shape[-1] < target_len:
                    out_audio = F.pad(out_audio, (0, target_len - out_audio.shape[-1]))

        return out_audio
