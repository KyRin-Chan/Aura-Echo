import numpy as np
from time import time
from typing import Union
from msgspec import msgpack

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

            out_audio, vol, perf, err = self.voiceChangerManager.change_voice(unpackedData)
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

    async def post_analyze_voice(self, target_file: UploadFile, input_file: UploadFile):
        import uuid
        import os
        from const import TMP_DIR
        import librosa
        import numpy as np

        target_ext = os.path.splitext(target_file.filename)[1] or ".wav"
        input_ext = os.path.splitext(input_file.filename)[1] or ".wav"
        
        target_path = os.path.join(TMP_DIR, f"target_{uuid.uuid4().hex}{target_ext}")
        input_path = os.path.join(TMP_DIR, f"input_{uuid.uuid4().hex}{input_ext}")
        
        try:
            # Save uploaded files temporarily
            with open(target_path, "wb") as f:
                f.write(await target_file.read())
            with open(input_path, "wb") as f:
                f.write(await input_file.read())
                
            # Load audio using librosa
            y_tgt, sr_tgt = librosa.load(target_path, sr=None)
            y_in, sr_in = librosa.load(input_path, sr=None)
            
            f0_tgt, cent_tgt = self._analyze_audio(y_tgt, sr_tgt)
            f0_in, cent_in = self._analyze_audio(y_in, sr_in)
            
            recommended_pitch = 0.0
            if f0_tgt > 0 and f0_in > 0:
                recommended_pitch = 12 * np.log2(f0_tgt / f0_in)
                # Round to nearest 0.5 semitone
                recommended_pitch = round(recommended_pitch * 2) / 2
                
            recommended_formant = 0.0
            if cent_tgt > 0 and cent_in > 0:
                recommended_formant = 12 * np.log2(cent_tgt / cent_in)
                recommended_formant = round(recommended_formant, 2)
                
            return JSONResponse({
                "success": True,
                "target_f0": round(float(f0_tgt), 1),
                "input_f0": round(float(f0_in), 1),
                "target_centroid": round(float(cent_tgt), 1),
                "input_centroid": round(float(cent_in), 1),
                "recommended_pitch": recommended_pitch,
                "recommended_formant_shift": recommended_formant
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
            if os.path.exists(input_path):
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
        
        # 2. Spectral Centroid extraction
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        voiced_cent = cent[0][voiced_frames] if np.any(voiced_frames) else cent[0]
        cent_median = np.median(voiced_cent) if len(voiced_cent) > 0 else 0.0
        
        return f0_median, cent_median
