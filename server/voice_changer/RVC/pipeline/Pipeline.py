from faiss import IndexIVFFlat
import faiss.contrib.torch_utils

from onnx import TensorProto
from onnx.helper import (
    make_model, make_node, make_graph,
    make_tensor_value_info, make_opsetid
)

import numpy as np
import sys
import torch
import torch.nn.functional as F
import onnxruntime
from torchaudio import transforms as tat
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
import logging

from const import HUBERT_SAMPLE_RATE, WINDOW_SIZE
from voice_changer.common.TorchUtils import circular_write
from voice_changer.embedder.Embedder import Embedder
from voice_changer.RVC.inferencer.Inferencer import Inferencer

from voice_changer.pitch_extractor.PitchExtractor import PitchExtractor
from voice_changer.utils.Timer import Timer2
from const import F0_MEL_MIN, F0_MEL_MAX
from voice_changer.audio_effects.AudioEffectsManager import AudioEffectsManager
from voice_changer.audio_effects.AudioEffectsConfig import AudioEffectsConfig

logger = logging.getLogger(__name__)


class Pipeline:
    embedder: Embedder
    inferencer: Inferencer
    pitchExtractor: PitchExtractor

    index: IndexIVFFlat | None
    index_reconstruct: torch.Tensor | None
    # feature: Any | None

    model_sr: int
    device: torch.device
    isHalf: bool
    audio_effects_manager: AudioEffectsManager

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

        self.index = index
        self.index_reconstruct: torch.Tensor | None = index_reconstruct
        self.use_index = index is not None and self.index_reconstruct is not None
        self.use_gpu_index = sys.platform == 'linux' and '+cu' in torch.__version__ and self.device.type == 'cuda'
        self.use_f0 = use_f0

        self.model_sr = model_sr
        self.model_window = model_sr // 100

        self.dtype = torch.float16 if self.is_half else torch.float32

        self.settings = None
        self._formant_filter_pre_tensor = None
        self._formant_filter_post_tensor = None
        self._formant_filter_cache_key = None
        
        # Initialize Audio Effects Manager with provider system
        try:
            self.audio_effects_manager = AudioEffectsManager()
            logger.info("Audio Effects Manager initialized with provider system")
        except Exception as e:
            logger.error(f"Failed to initialize AudioEffectsManager: {e}")
            self.audio_effects_manager = None



    def getPipelineInfo(self):
        inferencerInfo = self.inferencer.getInferencerInfo() if self.inferencer else {}
        embedderInfo = self.embedder.get_embedder_info()
        pitchExtractorInfo = self.pitchExtractor.getPitchExtractorInfo()
        return {"inferencer": inferencerInfo, "embedder": embedderInfo, "pitchExtractor": pitchExtractorInfo}

    def setPitchExtractor(self, pitchExtractor: PitchExtractor):
        self.pitchExtractor = pitchExtractor
    
    def configure_audio_effects(self, settings: dict) -> None:
        """Configure audio effects from frontend settings"""
        if self.audio_effects_manager is None:
            logger.warning("AudioEffectsManager not available, skipping configuration")
            return
            
        success = AudioEffectsConfig.configure_effects_from_settings(
            self.audio_effects_manager, 
            settings
        )
        if success:
            logger.debug("Audio effects configured successfully")
        else:
            logger.warning("Failed to configure audio effects")

    def extract_pitch(self, audio: torch.Tensor, pitch: torch.Tensor | None, pitchf: torch.Tensor | None, f0_up_key: float) -> tuple[torch.Tensor, torch.Tensor]:
        f0 = self.pitchExtractor.extract(
            audio,
            HUBERT_SAMPLE_RATE,
            WINDOW_SIZE,
        )
        f0 *= 2 ** (f0_up_key / 12)

        f0_mel = 1127.0 * torch.log(1.0 + f0 / 700.0)
        f0_mel = torch.clip(
            (f0_mel - F0_MEL_MIN) * 254 / (F0_MEL_MAX - F0_MEL_MIN) + 1,
            1,
            255,
            out=f0_mel
        )
        f0_coarse = torch.round(f0_mel, out=f0_mel).long()

        if pitch is not None and pitchf is not None:
            circular_write(f0_coarse, pitch)
            circular_write(f0, pitchf)
        else:
            pitch = f0_coarse
            pitchf = f0

        return pitch.unsqueeze(0), pitchf.unsqueeze(0)

    def _search_index(self, audio: torch.Tensor, top_k: int = 1):
        if top_k == 1:
            _, ix = self.index.search(audio if self.use_gpu_index else audio.detach().cpu(), 1)
            ix = ix.to(self.device)
            return self.index_reconstruct[ix.squeeze()]

        score, ix = self.index.search(audio if self.use_gpu_index else audio.detach().cpu(), k=top_k)
        score, ix = (
            score.to(self.device),
            ix.to(self.device),
        )
        weight = torch.square(1 / score)
        weight /= weight.sum(dim=1, keepdim=True)
        return torch.sum(self.index_reconstruct[ix] * weight.unsqueeze(2), dim=1)

    def _upscale(self, feats: torch.Tensor) -> torch.Tensor:
        # Vectorized nearest-neighbor upscaling (scale_factor=2) using native PyTorch operations.
        # This is fully device-agnostic, works on all devices, and avoids GPU-CPU roundtrips.
        return torch.stack([feats, feats], dim=2).flatten(1, 2)

    def exec(
        self,
        sid: int,
        audio: torch.Tensor,  # torch.tensor [n]
        pitch: torch.Tensor | None,  # torch.tensor [m]
        pitchf: torch.Tensor | None,  # torch.tensor [m]
        f0_up_key: float,
        index_rate: float,
        audio_feats_len: int,
        silence_front: int,
        embOutputLayer: int,
        useFinalProj: bool,
        skip_head: int,
        return_length: int,
        protect: float = 0.5,
    ) -> torch.Tensor:
        with Timer2("Pipeline-Exec", False) as t:  # NOQA
            # 16000のサンプリングレートで入ってきている。以降この世界は16000で処理。
            assert audio.dim() == 1, audio.dim()

            # Precompute formant filters if settings are available and changed
            if self.settings is not None:
                active = getattr(self.settings, "formantProfileActive", False)
                target_env = getattr(self.settings, "formantProfileTargetEnvelope", "[]")
                target_sr = getattr(self.settings, "formantProfileTargetSr", 0)
                input_env = getattr(self.settings, "formantProfileInputEnvelope", "[]")
                input_sr = getattr(self.settings, "formantProfileInputSr", 0)
                strength = getattr(self.settings, "formantProfileStrength", 0.35)
                device = self.device
                model_sr = self.model_sr

                cache_key = (active, target_env, target_sr, input_env, input_sr, strength, model_sr, str(device))
                if cache_key != self._formant_filter_cache_key:
                    self._formant_filter_cache_key = cache_key
                    if active and target_env and input_env and target_sr > 0 and input_sr > 0:
                        try:
                            import json
                            import numpy as np
                            tgt_env = np.array(json.loads(target_env), dtype=np.float32)
                            in_env = np.array(json.loads(input_env), dtype=np.float32)
                            
                            if len(tgt_env) > 0 and len(in_env) > 0:
                                n_fft_out = 2048
                                f_out = np.linspace(0, model_sr / 2, n_fft_out // 2 + 1)
                                
                                # 1. Post-filter (target envelope coloring)
                                f_tgt = np.linspace(0, target_sr / 2, len(tgt_env))
                                log_env_tgt_out = np.interp(f_out, f_tgt, tgt_env)
                                log_H_post = strength * log_env_tgt_out
                                H_post = np.exp(log_H_post)
                                self._formant_filter_post_tensor = torch.tensor(H_post, dtype=torch.float32, device=device)
                                
                                # 2. Pre-filter (input envelope inverse filtering / whitening)
                                input_audio_sr = 16000
                                f_in_out = np.linspace(0, input_audio_sr / 2, n_fft_out // 2 + 1)
                                f_in = np.linspace(0, input_sr / 2, len(in_env))
                                log_env_in_out = np.interp(f_in_out, f_in, in_env)
                                
                                # Use "Soft-whitening" with beta = 0.5 * strength (typically 0.17 to 0.35)
                                # to avoid over-flattening the spectrum and protect Hubert's phonetic comprehension.
                                beta = 0.5 * strength
                                log_H_pre = -beta * log_env_in_out
                                log_H_pre = np.clip(log_H_pre, -1.5, 1.5)
                                H_pre = np.exp(log_H_pre)
                                self._formant_filter_pre_tensor = torch.tensor(H_pre, dtype=torch.float32, device=device)
                                
                                logger.info(f"Formant profile pre- and post-filters precomputed successfully")
                            else:
                                self._formant_filter_pre_tensor = None
                                self._formant_filter_post_tensor = None
                        except Exception as ex:
                            logger.error(f"Error precomputing formant filters: {ex}")
                            self._formant_filter_pre_tensor = None
                            self._formant_filter_post_tensor = None
                    else:
                          self._formant_filter_pre_tensor = None
                          self._formant_filter_post_tensor = None

            # Apply pre-whitening filter to input audio if active
            if self._formant_filter_pre_tensor is not None:
                try:
                    n_fft = 2048
                    hop_length = 256  # Smaller hop length (87.5% overlap) to prevent time-smearing & window boundary artifacts
                    audio_float = audio.to(device=self.device, dtype=torch.float32)
                    if audio_float.shape[0] >= n_fft:
                        window = torch.hann_window(n_fft, device=self.device)
                        S = torch.stft(
                            audio_float,
                            n_fft=n_fft,
                            hop_length=hop_length,
                            window=window,
                            return_complex=True
                        )
                        S_filtered = S * self._formant_filter_pre_tensor.unsqueeze(-1)
                        audio_filtered = torch.istft(
                            S_filtered,
                            n_fft=n_fft,
                            hop_length=hop_length,
                            window=window,
                            length=audio_float.shape[0]
                        )
                        audio = audio_filtered.to(device=audio.device, dtype=audio.dtype)
                except Exception as ex:
                    logger.error(f"Error applying pre-whitening filter: {ex}")

            formant_length = return_length
            t.record("pre-process")

            # Audio Effects vor Voice Conversion anwenden
            if self.audio_effects_manager is not None:
                audio = self.audio_effects_manager.process_input_chain(audio, sample_rate=16000)
            t.record("input-effects")

            # Pitch extraction and feature extraction in parallel (releasing GIL in ONNX/PyTorch)
            if self.use_f0:
                import threading

                pitch_res = [None, None]
                feats_res = [None]
                exceptions = []

                def _run_pitch():
                    try:
                        pitch_res[0], pitch_res[1] = self.extract_pitch(
                            audio[silence_front:], pitch, pitchf, f0_up_key
                        )
                    except Exception as e:
                        exceptions.append(e)

                def _run_feats():
                    try:
                        feats_res[0] = self.embedder.extract_features(audio.view(1, -1), embOutputLayer, useFinalProj)
                    except Exception as e:
                        exceptions.append(e)

                t_pitch = threading.Thread(target=_run_pitch)
                t_feats = threading.Thread(target=_run_feats)

                t_pitch.start()
                t_feats.start()

                t_pitch.join()
                t_feats.join()

                if exceptions:
                    raise exceptions[0]

                pitch, pitchf = pitch_res[0], pitch_res[1]
                feats = feats_res[0]
            else:
                pitch, pitchf = None, None
                feats = self.embedder.extract_features(audio.view(1, -1), embOutputLayer, useFinalProj)

            feats = torch.cat((feats, feats[:, -1:, :]), 1)
            t.record("extract-pitch-and-feats")

            # Index - feature抽出
            is_active_index = self.use_index and index_rate > 0
            use_protect = protect < 0.5
            if self.use_f0 and is_active_index and use_protect:
                feats_orig = feats.detach().clone()

            if is_active_index:
                skip_offset = skip_head // 2
                index_audio = feats[0][skip_offset :]

                # TODO: kは調整できるようにする
                index_audio = self._search_index(index_audio.float(), 8).unsqueeze(0)
                if self.is_half:
                    index_audio = index_audio.half()

                # Recover silent front
                feats[0][skip_offset :] = index_audio * index_rate + feats[0][skip_offset :] * (1 - index_rate)

            feats = self._upscale(feats)[:, :audio_feats_len, :]
            if self.use_f0:
                pitch = pitch[:, -audio_feats_len:]
                pitchf = pitchf[:, -audio_feats_len:]
                # pitchの推定が上手くいかない(pitchf=0)場合、検索前の特徴を混ぜる
                # pitchffの作り方の疑問はあるが、本家通りなので、このまま使うことにする。
                # https://github.com/w-okada/voice-changer/pull/276#issuecomment-1571336929
                if is_active_index and use_protect:
                    # FIXME: Another interpolate on feats is a big performance hit.
                    feats_orig = self._upscale(feats_orig)[:, :audio_feats_len, :]
                    pitchff = pitchf.detach().clone()
                    pitchff[pitchf > 0] = 1
                    pitchff[pitchf < 1] = protect
                    pitchff = pitchff.unsqueeze(-1)
                    feats = feats * pitchff + feats_orig * (1 - pitchff)

            p_len = torch.tensor([audio_feats_len], device=self.device, dtype=torch.int64)

            sid = torch.tensor([sid], device=self.device, dtype=torch.int64)
            t.record("mid-precess")
            # 推論実行
            out_audio = self.inferencer.infer(feats, p_len, pitch, pitchf, sid, skip_head, return_length, formant_length).float()
            t.record("infer")
            
            # Apply formant profile warp filter if active
            if self._formant_filter_post_tensor is not None:
                try:
                    n_fft = 2048
                    hop_length = 256  # Smaller hop length (87.5% overlap) to prevent time-smearing & window boundary artifacts
                    
                    # Ensure out_audio is float32 and on the correct device
                    out_audio_float = out_audio.to(device=self.device, dtype=torch.float32)
                    
                    # STFT requires input length >= n_fft to avoid padding issues or errors
                    if out_audio_float.shape[0] >= n_fft:
                        window = torch.hann_window(n_fft, device=self.device)
                        S = torch.stft(
                            out_audio_float,
                            n_fft=n_fft,
                            hop_length=hop_length,
                            window=window,
                            return_complex=True
                        )
                        
                        # Pointwise scaling of magnitude spectrum
                        S_filtered = S * self._formant_filter_post_tensor.unsqueeze(-1)
                        
                        # Reconstruct via ISTFT
                        out_audio_filtered = torch.istft(
                            S_filtered,
                            n_fft=n_fft,
                            hop_length=hop_length,
                            window=window,
                            length=out_audio_float.shape[0]
                        )
                        out_audio = out_audio_filtered.to(device=out_audio.device, dtype=out_audio.dtype)
                except Exception as ex:
                    logger.error(f"Error applying formant profile filter: {ex}")
            
            # Audio Effects nach Voice Conversion anwenden
            if self.audio_effects_manager is not None:
                out_audio = self.audio_effects_manager.process_output_chain(out_audio, sample_rate=self.model_sr)
            t.record("output-effects")
        return out_audio
