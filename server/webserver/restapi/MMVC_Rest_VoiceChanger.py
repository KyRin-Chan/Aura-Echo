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
from voice_changer.FormantProfileManager import FormantProfileManager
from voice_changer.common.AudioDSP import (
    analyze_voice_acoustics,
    compute_residual_match_eq_curve,
)
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
        self.router.add_api_route("/calibrate_match_eq", self.post_calibrate_match_eq, methods=["POST"])

        # Formant Profile & Binding API routes
        self.router.add_api_route("/formant_profiles", self.get_formant_profiles, methods=["GET"])
        self.router.add_api_route("/formant_profiles", self.post_formant_profile, methods=["POST"])
        self.router.add_api_route("/formant_profiles/{profile_id}", self.delete_formant_profile, methods=["DELETE"])
        self.router.add_api_route("/formant_bindings", self.get_formant_bindings, methods=["GET"])
        self.router.add_api_route("/formant_bindings", self.post_formant_binding, methods=["POST"])
        self.router.add_api_route("/formant_bindings/{slot_index}", self.delete_formant_binding, methods=["DELETE"])

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

    def _load_and_analyze_file(self, path: str):
        import librosa

        y, sr = librosa.load(path, sr=None, mono=True)
        res = analyze_voice_acoustics(y, sr)
        return y, int(sr), res

    async def post_analyze_voice(self, target_file: UploadFile, input_file: UploadFile = None):
        import uuid
        import os
        from const import TMP_DIR

        os.makedirs(TMP_DIR, exist_ok=True)
        target_ext = os.path.splitext(target_file.filename or "")[1] or ".wav"
        target_path = os.path.join(TMP_DIR, f"target_{uuid.uuid4().hex}{target_ext}")
        input_path = None

        try:
            with open(target_path, "wb") as f:
                f.write(await target_file.read())

            _, sr_tgt, tgt_res = await asyncio.to_thread(self._load_and_analyze_file, target_path)
            f0_tgt = tgt_res["f0"]
            cent_tgt = tgt_res["centroid"]
            env_tgt = tgt_res["envelope"]

            if input_file is not None:
                input_ext = os.path.splitext(input_file.filename or "")[1] or ".wav"
                input_path = os.path.join(TMP_DIR, f"input_{uuid.uuid4().hex}{input_ext}")
                with open(input_path, "wb") as f:
                    f.write(await input_file.read())

                _, sr_in, in_res = await asyncio.to_thread(self._load_and_analyze_file, input_path)
                f0_in = in_res["f0"]
                cent_in = in_res["centroid"]
                env_in = in_res["envelope"]

                recommended_pitch = 0.0
                if f0_tgt > 0 and f0_in > 0:
                    recommended_pitch = 12.0 * float(np.log2(f0_tgt / f0_in))
                    recommended_pitch = round(recommended_pitch * 2.0) / 2.0

                recommended_formant = 0.0
                if cent_tgt > 0 and cent_in > 0:
                    leakage_coefficient = 0.35
                    raw_shift = leakage_coefficient * 12.0 * float(np.log2(cent_tgt / cent_in))
                    recommended_formant = round(round(float(np.clip(raw_shift, -3.0, 3.0)) * 20.0) / 20.0, 2)

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
                    "target_voiced_ratio": round(float(tgt_res["voiced_ratio"]), 3),
                    "input_envelope": env_in.tolist(),
                    "input_sr": int(sr_in),
                    "input_voiced_ratio": round(float(in_res["voiced_ratio"]), 3),
                })
            else:
                return JSONResponse({
                    "success": True,
                    "target_f0": round(float(f0_tgt), 1),
                    "target_centroid": round(float(cent_tgt), 1),
                    "target_envelope": env_tgt.tolist(),
                    "target_sr": int(sr_tgt),
                    "target_voiced_ratio": round(float(tgt_res["voiced_ratio"]), 3),
                })

        except Exception as e:
            logger.exception(e)
            error_msg = str(e)
            if "NoBackendError" in type(e).__name__ or "NoBackendError" in error_msg:
                error_msg = "No audio decoding backend found (FFmpeg is required to load compressed formats like .m4a on Windows). Please install FFmpeg and add it to your system PATH, or convert your audio file to .wav format before uploading."
            return JSONResponse({
                "success": False,
                "error": error_msg,
            }, status_code=500)

        finally:
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

    async def post_calibrate_match_eq(
        self,
        target_file: UploadFile,
        input_file: UploadFile,
        run_rvc: bool = Form(True),
    ):
        import uuid
        import os
        from const import TMP_DIR
        import librosa

        os.makedirs(TMP_DIR, exist_ok=True)
        target_ext = os.path.splitext(target_file.filename or "")[1] or ".wav"
        input_ext = os.path.splitext(input_file.filename or "")[1] or ".wav"
        target_path = os.path.join(TMP_DIR, f"cal_tgt_{uuid.uuid4().hex}{target_ext}")
        input_path = os.path.join(TMP_DIR, f"cal_in_{uuid.uuid4().hex}{input_ext}")

        try:
            with open(target_path, "wb") as f:
                f.write(await target_file.read())
            with open(input_path, "wb") as f:
                f.write(await input_file.read())

            _, sr_tgt, tgt_res = await asyncio.to_thread(self._load_and_analyze_file, target_path)
            y_in, sr_in = await asyncio.to_thread(lambda p: librosa.load(p, sr=None, mono=True), input_path)
            dry_in_res = await asyncio.to_thread(analyze_voice_acoustics, y_in, int(sr_in))

            pitch_dist_cfg = None
            if dry_in_res.get("f0_log2_mean", 0.0) > 0.0 and tgt_res.get("f0_log2_mean", 0.0) > 0.0:
                pitch_dist_cfg = {
                    "src_mean": float(dry_in_res["f0_log2_mean"]),
                    "src_std": float(dry_in_res.get("f0_log2_std", 0.0)),
                    "tgt_mean": float(tgt_res["f0_log2_mean"]),
                    "tgt_std": float(tgt_res.get("f0_log2_std", 0.0)),
                    "tgt_p10": float(tgt_res.get("f0_p10", 0.0)),
                    "tgt_p90": float(tgt_res.get("f0_p90", 0.0)),
                    "pitch_scale": 1.0,
                    "pitch_offset": 0.0,
                }

            if run_rvc:
                vcmodel = getattr(self.voiceChangerManager.vc, "vcmodel", None)
                if (
                    vcmodel is None
                    or not hasattr(vcmodel, "convert_offline")
                    or getattr(vcmodel, "pipeline", None) is None
                ):
                    return JSONResponse({
                        "success": False,
                        "error": "No active RVC model loaded. Please select an RVC model first, or disable RVC conversion to supply an already-converted sample.",
                    }, status_code=400)

                def _run_offline():
                    with self.voiceChangerManager.device_manager.lock:
                        return vcmodel.convert_offline(
                            y_in,
                            int(sr_in),
                            pitch_dist_cfg=pitch_dist_cfg,
                            formant_shift_override=0.0,
                        )

                y_base, sr_base = await asyncio.to_thread(_run_offline)
                if y_base is None or y_base.size == 0:
                    return JSONResponse({
                        "success": False,
                        "error": "Offline RVC conversion returned empty audio.",
                    }, status_code=400)
            else:
                y_base, sr_base = y_in, int(sr_in)

            base_res = await asyncio.to_thread(analyze_voice_acoustics, y_base, int(sr_base))

            settings = self.voiceChangerManager.settings
            post_sr = int(getattr(settings, "outputSampleRate", 48000) or 48000)
            _, _, rms_residual_db, max_correction_db = compute_residual_match_eq_curve(
                tgt_res["envelope"],
                int(tgt_res["sr"]),
                base_res["envelope"],
                int(base_res["sr"]),
                post_sr=post_sr,
            )

            return JSONResponse({
                "success": True,
                "calibrated_with_rvc": bool(run_rvc),
                "model_slot_index": int(getattr(settings, "modelSlotIndex", -1)),
                "calibrated_tran": float(getattr(settings, "tran", 0.0)),
                "calibrated_formant_shift": float(getattr(settings, "formantShift", 0.0)),
                "target_f0": round(float(tgt_res["f0"]), 1),
                "target_f0_log2_mean": float(tgt_res.get("f0_log2_mean", 0.0)),
                "target_f0_log2_std": float(tgt_res.get("f0_log2_std", 0.0)),
                "target_f0_p10": round(float(tgt_res.get("f0_p10", 0.0)), 1),
                "target_f0_p90": round(float(tgt_res.get("f0_p90", 0.0)), 1),
                "target_centroid": round(float(tgt_res["centroid"]), 1),
                "target_envelope": tgt_res["envelope"].tolist(),
                "target_sr": int(tgt_res["sr"]),
                "input_f0_log2_mean": float(dry_in_res.get("f0_log2_mean", 0.0)),
                "input_f0_log2_std": float(dry_in_res.get("f0_log2_std", 0.0)),
                "input_f0_p10": round(float(dry_in_res.get("f0_p10", 0.0)), 1),
                "input_f0_p90": round(float(dry_in_res.get("f0_p90", 0.0)), 1),
                "baseline_f0": round(float(base_res["f0"]), 1),
                "baseline_centroid": round(float(base_res["centroid"]), 1),
                "baseline_envelope": base_res["envelope"].tolist(),
                "baseline_sr": int(base_res["sr"]),
                "rms_residual_db": round(float(rms_residual_db), 2),
                "max_correction_db": round(float(max_correction_db), 2),
            })

        except Exception as e:
            logger.exception(e)
            error_msg = str(e)
            if "NoBackendError" in type(e).__name__ or "NoBackendError" in error_msg:
                error_msg = "No audio decoding backend found (FFmpeg is required to load compressed formats like .m4a on Windows). Please install FFmpeg and add it to your system PATH, or convert your audio file to .wav format before uploading."
            return JSONResponse({
                "success": False,
                "error": error_msg,
            }, status_code=500)

        finally:
            for p in (target_path, input_path):
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception as ex:
                        logger.warning(f"Failed to remove temp file {p}: {ex}")

    def _analyze_audio(self, y, sr):
        res = analyze_voice_acoustics(y, sr)
        return res["f0"], res["centroid"], res["envelope"]

    # Formant Profile & Binding API handlers
    async def get_formant_profiles(self):
        profiles = FormantProfileManager.get_instance().get_all_profiles()
        return JSONResponse(profiles)

    async def post_formant_profile(self, req: Request):
        try:
            data = await req.json()
            saved = FormantProfileManager.get_instance().save_profile(data)
            return JSONResponse(saved)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    async def delete_formant_profile(self, profile_id: str):
        success = FormantProfileManager.get_instance().delete_profile(profile_id)
        return JSONResponse({"success": success})

    async def get_formant_bindings(self):
        bindings = FormantProfileManager.get_instance().get_all_bindings()
        return JSONResponse(bindings)

    async def post_formant_binding(self, req: Request):
        try:
            data = await req.json()
            slot_index = data.get("slotIndex")
            if slot_index is None:
                return JSONResponse({"error": "slotIndex required"}, status_code=400)
            saved = FormantProfileManager.get_instance().save_binding(slot_index, data)
            return JSONResponse(saved)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    async def delete_formant_binding(self, slot_index: str):
        success = FormantProfileManager.get_instance().delete_binding(slot_index)
        return JSONResponse({"success": success})
