import numpy as np
from time import time
from typing import Union
from msgspec import msgpack
import asyncio

from fastapi import APIRouter, Request, Form, UploadFile
from fastapi.responses import Response, PlainTextResponse, JSONResponse
from fastapi.encoders import jsonable_encoder
from const import get_edition, get_version
from voice_changer.VoiceChangerManager import VoiceChangerManager
from webserver.restapi.mods.FileUploader import upload_file

from const import UPLOAD_DIR

import logging
logger = logging.getLogger(__name__)


class MMVC_Rest_VoiceChanger:
    def __init__(self, voiceChangerManager: VoiceChangerManager):
        self.voiceChangerManager = voiceChangerManager
        self.router = APIRouter()
        self.router.add_api_route("/test", self.test, methods=["POST"])
        self.router.add_api_route("/edition", self.edition, methods=["GET"])
        self.router.add_api_route("/version", self.version, methods=["GET"])
        self.router.add_api_route("/info", self.get_info, methods=["GET"])
        self.router.add_api_route("/update_settings", self.post_update_settings, methods=["POST"])
        self.router.add_api_route("/upload_file", self.post_upload_file, methods=["POST"])
        self.router.add_api_route("/analyze_voice", self.post_analyze_voice, methods=["POST"])

    def edition(self):
        return PlainTextResponse(get_edition())

    def version(self):
        return PlainTextResponse(get_version())

    async def test(self, req: Request):
        recv_timestamp = round(time() * 1000)
        try:
            data = await req.body()
            ts, voice = msgpack.decode(data)

            unpackedData = np.frombuffer(voice, dtype=np.int16).astype(np.float32) / 32768

            out_audio, vol, perf, err = await asyncio.to_thread(self.voiceChangerManager.change_voice, unpackedData)
            out_audio = np.nan_to_num(out_audio)
            out_audio = np.clip(out_audio, -1.0, 1.0)
            out_audio = (out_audio * 32767).astype(np.int16).tobytes()

            if err is not None:
                error_code, error_message = err
                return Response(
                    content=msgpack.encode({
                        "error": True,
                        "details": {
                            "code": error_code,
                            "message": error_message,
                        },
                    }),
                    headers={'Content-Type': 'application/octet-stream'},
                )

            ping = recv_timestamp - ts
            send_timestamp = round(time() * 1000)
            return Response(
                content=msgpack.encode({
                    "error": False,
                    "audio": out_audio,
                    "perf": perf,
                    "vol": vol,
                    "ping": ping,
                    "sendTimestamp": send_timestamp,
                }),
                headers={'Content-Type': 'application/octet-stream'},
            )

        except Exception as e:
            logger.exception(e)
            return Response(
                content=msgpack.encode({
                    "error": True,
                    "timestamp": 0,
                    "details": {
                        "code": "GENERIC_REST_SERVER_ERROR",
                        "message": "Check command line for more details.",
                    },
                }),
                headers={'Content-Type': 'application/octet-stream'},
            )

    def get_info(self):
        try:
            info = self.voiceChangerManager.get_info()
            json_compatible_item_data = jsonable_encoder(info)
            return JSONResponse(content=json_compatible_item_data)
        except Exception as e:
            logger.exception(e)

    def post_update_settings(self, key: str = Form(...), val: Union[int, str, float] = Form(...)):
        try:
            info = self.voiceChangerManager.update_settings(key, val)
            json_compatible_item_data = jsonable_encoder(info)
            return JSONResponse(content=json_compatible_item_data)
        except Exception as e:
            logger.exception(e)

    # Uploads a file to the upload_dir
    def post_upload_file(self, file: UploadFile, filename: str = Form(...)):
        try:
            res = upload_file(UPLOAD_DIR, file, filename)
            json_compatible_item_data = jsonable_encoder(res)
            return JSONResponse(content=json_compatible_item_data)
        except Exception as e:
            logger.exception(e)

    async def post_analyze_voice(self, target_file: UploadFile, input_file: UploadFile = None):
        import uuid
        import os
        from const import TMP_DIR
        import librosa
        import numpy as np

        target_ext = os.path.splitext(target_file.filename)[1] or ".wav"
        target_path = os.path.join(TMP_DIR, f"target_{uuid.uuid4().hex}{target_ext}")
        input_path = None
        
        try:
            # Save uploaded target file temporarily
            with open(target_path, "wb") as f:
                f.write(await target_file.read())
                
            # Load target audio using librosa
            y_tgt, sr_tgt = librosa.load(target_path, sr=None)
            f0_tgt, cent_tgt, env_tgt = self._analyze_audio(y_tgt, sr_tgt)
            
            if input_file is not None:
                input_ext = os.path.splitext(input_file.filename)[1] or ".wav"
                input_path = os.path.join(TMP_DIR, f"input_{uuid.uuid4().hex}{input_ext}")
                with open(input_path, "wb") as f:
                    f.write(await input_file.read())
                y_in, sr_in = librosa.load(input_path, sr=None)
                f0_in, cent_in, env_in = self._analyze_audio(y_in, sr_in)
                
                recommended_pitch = 0.0
                if f0_tgt > 0 and f0_in > 0:
                    recommended_pitch = 12 * np.log2(f0_tgt / f0_in)
                    recommended_pitch = round(recommended_pitch * 2) / 2
                    
                recommended_formant = 0.0
                if cent_tgt > 0 and cent_in > 0:
                    leakage_coefficient = 0.35
                    recommended_formant = leakage_coefficient * 12 * np.log2(cent_tgt / cent_in)
                    recommended_formant = round(recommended_formant, 2)
                    
                return JSONResponse({
                    "success": True,
                    "target_f0": round(float(f0_tgt), 1),
                    "input_f0": round(float(f0_in), 1),
                    "target_centroid": round(float(cent_tgt), 1),
                    "input_centroid": round(float(cent_in), 1),
                    "recommended_pitch": recommended_pitch,
                    "recommended_formant_shift": recommended_formant,
                    "target_envelope": env_tgt.tolist(),
                    "target_sr": int(sr_tgt),
                    "input_envelope": env_in.tolist(),
                    "input_sr": int(sr_in)
                })
            else:
                return JSONResponse({
                    "success": True,
                    "target_f0": round(float(f0_tgt), 1),
                    "target_centroid": round(float(cent_tgt), 1),
                    "target_envelope": env_tgt.tolist(),
                    "target_sr": int(sr_tgt)
                })
            
        except Exception as e:
            logger.exception(e)
            error_msg = str(e)
            if "NoBackendError" in type(e).__name__ or "NoBackendError" in error_msg:
                error_msg = "No audio decoding backend found (FFmpeg is required to load compressed formats like .m4a on Windows). Please install FFmpeg and add it to your system PATH, or convert your audio file to .wav format before uploading."
            return JSONResponse({
                "success": False,
                "error": error_msg
            }, status_code=500)
            
        finally:
            # Cleanup temporary files
            if os.path.exists(target_path):
                try:
                    os.remove(target_path)
                except Exception as ex:
                    logger.warning(f"Failed to remove temp file {target_path}: {ex}")
            if input_path and os.path.exists(input_path):
                try:
                    os.remove(input_path)
                except Exception as ex:
                    logger.warning(f"Failed to remove temp file {input_path}: {ex}")

    def _analyze_audio(self, y, sr):
        import librosa
        import numpy as np
        
        # 1. Pitch F0 extraction (using YIN)
        f0 = librosa.yin(y, fmin=50, fmax=800, sr=sr)
        rms = librosa.feature.rms(y=y)
        threshold = 0.05 * np.max(rms) if np.max(rms) > 0 else 1e-4
        voiced_frames = rms[0] > threshold
        voiced_f0 = f0[voiced_frames] if np.any(voiced_frames) else f0
        voiced_f0 = voiced_f0[voiced_f0 > 0]
        
        f0_median = np.median(voiced_f0) if len(voiced_f0) > 0 else 0.0
        
        # 2. Cepstral Envelope Centroid extraction (pitch-decoupled formant proxy)
        # Compute STFT magnitude spectrogram
        S = np.abs(librosa.stft(y))
        log_S = np.log(S + 1e-8)
        
        # Compute real inverse FFT along frequency bins (axis 0) to get cepstrum
        cepstrum = np.fft.irfft(log_S, axis=0)
        n_fft_samples = cepstrum.shape[0]
        
        # Keep only the lower quefrency coefficients (first 20 coefficients) representing vocal tract shape (formants),
        # and zero out the high quefrency components representing fundamental pitch harmonics.
        quefrency_cutoff = 20
        if n_fft_samples > 2 * quefrency_cutoff:
            cepstrum[quefrency_cutoff : -quefrency_cutoff, :] = 0
            
        # Compute forward real FFT back to frequency domain to get the smoothed log envelope
        # Take the real part of the envelope to keep it real-valued and avoid imaginary warnings/errors
        log_envelope = np.fft.rfft(cepstrum, axis=0).real
        # Handle shape mismatch if any
        if log_envelope.shape[0] > log_S.shape[0]:
            log_envelope = log_envelope[:log_S.shape[0], :]
        elif log_envelope.shape[0] < log_S.shape[0]:
            pass
        envelope = np.exp(log_envelope)
        
        # Spectral Centroid of the smoothed envelope
        frequencies = librosa.fft_frequencies(sr=sr, n_fft=2 * (S.shape[0] - 1))
        
        # Compute centroid for each frame using the pitch-independent envelope
        sum_envelope = np.sum(envelope, axis=0)
        sum_envelope[sum_envelope == 0] = 1e-8
        centroid_frames = np.sum(envelope * frequencies[:, np.newaxis], axis=0) / sum_envelope
        
        voiced_cent = centroid_frames[voiced_frames] if np.any(voiced_frames) else centroid_frames
        cent_median = np.median(voiced_cent) if len(voiced_cent) > 0 else 0.0
        
        # Compute mean log envelope across voiced frames
        voiced_log_envelope = log_envelope[:, voiced_frames] if np.any(voiced_frames) else log_envelope
        mean_log_envelope = np.mean(voiced_log_envelope, axis=1)
        
        return f0_median, cent_median, mean_log_envelope
