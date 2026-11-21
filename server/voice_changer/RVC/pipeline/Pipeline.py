from concurrent.futures import ThreadPoolExecutor
import logging
import math
import sys
from typing import Optional

from faiss import IndexIVFFlat
import faiss.contrib.torch_utils
import torch

from const import F0_MEL_MIN, F0_MEL_MAX, HUBERT_SAMPLE_RATE, WINDOW_SIZE
from voice_changer.audio_effects.AudioEffectsManager import AudioEffectsManager
from voice_changer.common.TorchUtils import circular_write
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
from voice_changer.embedder.Embedder import Embedder
from voice_changer.pitch_extractor.PitchExtractor import PitchExtractor
from voice_changer.RVC.inferencer.Inferencer import Inferencer
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
        self.dtype = torch.float16 if self.is_half else torch.float32

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

        # Cached small scalar tensors
        self._cached_plen_val: Optional[int] = None
        self._cached_plen_tensor: Optional[torch.Tensor] = None
        self._cached_sid_val: Optional[int] = None
        self._cached_sid_tensor: Optional[torch.Tensor] = None

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

    def extract_pitch(
        self,
        audio: torch.Tensor,
        pitch: torch.Tensor | None,
        pitchf: torch.Tensor | None,
        f0_up_key: float,
        pitch_hop_frames: Optional[int] = None,
        pitch_scratch: Optional[torch.Tensor] = None,
        pitchf_scratch: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        f0 = self.pitchExtractor.extract(
            audio,
            HUBERT_SAMPLE_RATE,
            WINDOW_SIZE,
        )
        f0 = f0 * (2 ** (f0_up_key / 12))

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

                    # When autoTune is active, only write the causally smoothed tail so prior frames stay smooth
                    write_len = hop if auto_tune_effect is not None else min(f0.shape[0], pitchf.shape[0])
                    if write_len > 0:
                        pitch[-write_len:] = f0_coarse[-write_len:].to(dtype=pitch.dtype)
                        pitchf[-write_len:] = f0[-write_len:].to(dtype=pitchf.dtype)
            else:
                circular_write(f0_coarse, pitch, scratch=pitch_scratch)
                circular_write(f0, pitchf, scratch=pitchf_scratch)
        else:
            pitch = f0_coarse
            pitchf = f0

        return pitch.unsqueeze(0), pitchf.unsqueeze(0)

    def _search_index(self, audio: torch.Tensor, top_k: int = 1) -> torch.Tensor:
        if self.use_torch_gpu_index and self._gpu_index_norms is not None:
            q_match = audio.to(dtype=self.index_reconstruct.dtype)
            q_norms = torch.sum(audio.float() ** 2, dim=1, keepdim=True)
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

            if k_eff == 1:
                return self.index_reconstruct[ix.squeeze(1)]
            weight = torch.square(1.0 / (score + 1e-6))
            weight /= weight.sum(dim=1, keepdim=True)
            return torch.sum(self.index_reconstruct[ix].float() * weight.unsqueeze(2), dim=1)

        # CPU / Linux faiss-gpu fallback
        query = audio if self.use_gpu_index else audio.detach().cpu()
        if top_k == 1:
            _, ix = self.index.search(query, 1)
            ix = ix.to(self.device)
            return self.index_reconstruct[ix.squeeze(1)]

        score, ix = self.index.search(query, k=top_k)
        score = score.to(self.device)
        ix = ix.to(self.device)
        weight = torch.square(1.0 / (score + 1e-6))
        weight /= weight.sum(dim=1, keepdim=True)
        return torch.sum(self.index_reconstruct[ix] * weight.unsqueeze(2), dim=1)

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
    ) -> torch.Tensor:
        with Timer2("Pipeline-Exec", False) as t:
            assert audio.dim() == 1, audio.dim()

            formant_length = return_length
            t.record("pre-process")

            # Pitch extraction and feature extraction in parallel via persistent executor
            if self.use_f0:
                def _run_feats_worker():
                    if self._embedder_stream is not None:
                        with torch.cuda.stream(self._embedder_stream):
                            return self.embedder.extract_features(
                                audio.view(1, -1), embOutputLayer, useFinalProj
                            )
                    return self.embedder.extract_features(
                        audio.view(1, -1), embOutputLayer, useFinalProj
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
                            pitch, pitchf = self.extract_pitch(
                                audio[silence_front:],
                                pitch,
                                pitchf,
                                f0_up_key,
                                pitch_hop_frames=pitch_hop_frames,
                                pitch_scratch=pitch_scratch,
                                pitchf_scratch=pitchf_scratch,
                            )
                        cur.wait_stream(self._pitch_stream)
                    else:
                        pitch, pitchf = self.extract_pitch(
                            audio[silence_front:],
                            pitch,
                            pitchf,
                            f0_up_key,
                            pitch_hop_frames=pitch_hop_frames,
                            pitch_scratch=pitch_scratch,
                            pitchf_scratch=pitchf_scratch,
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
                pitch, pitchf = None, None
                feats = self.embedder.extract_features(audio.view(1, -1), embOutputLayer, useFinalProj)

            feats = torch.cat((feats, feats[:, -1:, :]), 1)
            t.record("extract-pitch-and-feats")

            # Index feature retrieval & protect blending (single upscale)
            is_active_index = self.use_index and index_rate > 0
            use_protect = protect < 0.5

            if is_active_index:
                skip_offset = skip_head // 2
                feats_orig_slice = feats[0, skip_offset:]
                index_audio = self._search_index(feats_orig_slice.float(), 8).to(dtype=feats.dtype)

                if self.use_f0 and use_protect:
                    delta = (index_audio - feats_orig_slice).unsqueeze(0)
                    feats = self._upscale(feats)[:, :audio_feats_len, :]
                    pitch = pitch[:, -audio_feats_len:]
                    pitchf = pitchf[:, -audio_feats_len:]

                    up_start = skip_offset * 2
                    rem = audio_feats_len - up_start
                    if rem > 0:
                        delta_up = self._upscale(delta)[:, :rem, :]
                        pitchff = torch.where(
                            pitchf[:, up_start:audio_feats_len] > 0, 1.0, protect
                        ).to(dtype=feats.dtype).unsqueeze(-1)
                        feats[:, up_start:audio_feats_len, :] = (
                            feats[:, up_start:audio_feats_len, :]
                            + delta_up * (index_rate * pitchff)
                        )
                else:
                    feats[0, skip_offset:] = (
                        index_audio * index_rate + feats_orig_slice * (1.0 - index_rate)
                    )
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
            t.record("mid-precess")

            out_audio = self.inferencer.infer(
                feats, p_len, pitch, pitchf, sid_t, skip_head, return_length, formant_length
            ).float()
            t.record("infer")

        return out_audio
