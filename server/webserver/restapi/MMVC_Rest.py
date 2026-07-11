import logging
import os

from webserver.restapi.mods.TrustedOrigin import TrustedOriginMiddleware
from fastapi import FastAPI, Request, Response, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.routing import APIRoute
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from typing import Callable
from voice_changer.VoiceChangerManager import VoiceChangerManager

from webserver.restapi.MMVC_Rest_Sounds import MMVC_Rest_Sounds
from webserver.restapi.MMVC_Rest_VoiceChanger import MMVC_Rest_VoiceChanger
from webserver.restapi.MMVC_Rest_Models import MMVC_Rest_Models
from webserver.restapi.MMVC_Rest_PretrainDownloader import MMVC_Rest_PretrainDownloader
from settings import get_settings
from const import TMP_DIR

logger = logging.getLogger(__name__)

class ValidationErrorLoggingRoute(APIRoute):
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            try:
                return await original_route_handler(request)
            except RequestValidationError as e:  # type: ignore
                logger.exception(e)
                body = await request.body()
                detail = {"errors": e.errors(), "body": body.decode()}
                raise HTTPException(status_code=422, detail=detail)

        return custom_route_handler

class MMVC_Rest:
    _instance = None

    @classmethod
    def get_instance(cls, voiceChangerManager: VoiceChangerManager):
        if cls._instance is None:
            logger.info("Initializing...")
            settings = get_settings()
            app_fastapi = FastAPI()
            app_fastapi.router.route_class = ValidationErrorLoggingRoute
            app_fastapi.add_middleware(
                TrustedOriginMiddleware,
                allowed_origins=settings.allowed_origins,
                port=settings.port
            )

            app_fastapi.mount("/tmp", StaticFiles(directory=TMP_DIR), name="static")

            app_fastapi.mount(
                "/model_dir",
                StaticFiles(directory=settings.model_dir),
                name="static",
            )
            app_fastapi.mount(
                "/sound_dir",
                StaticFiles(directory=settings.sound_dir),
                name="static",
            )

            restVoiceChanger = MMVC_Rest_VoiceChanger(voiceChangerManager)
            app_fastapi.include_router(restVoiceChanger.router)
            
            modelsApi = MMVC_Rest_Models(voiceChangerManager)
            app_fastapi.include_router(modelsApi.router)

            soundsApi = MMVC_Rest_Sounds(voiceChangerManager)
            app_fastapi.include_router(soundsApi.router)

            pretrainDownloader = MMVC_Rest_PretrainDownloader()
            app_fastapi.include_router(pretrainDownloader.router)

            # Raw high-performance WebSocket route for binary audio streaming
            import struct
            from time import time
            import numpy as np

            @app_fastapi.websocket("/ws/voice")
            async def websocket_voice(websocket: WebSocket):
                await websocket.accept()
                try:
                    while True:
                        data = await websocket.receive_bytes()
                        recv_timestamp = round(time() * 1000)
                        if len(data) < 8:
                            continue
                        
                        ts = struct.unpack("<q", data[:8])[0]
                        raw_audio = data[8:]
                        input_audio = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768

                        out_audio, vol, perf, err = voiceChangerManager.change_voice(input_audio)
                        if err is not None:
                            error_code, error_message = err
                            error_msg = f"{error_code}: {error_message}".encode("utf-8")
                            header = struct.pack("<qiffffBB", 0, 0, 0.0, 0.0, 0.0, 0.0, 1, 0)
                            await websocket.send_bytes(header + error_msg)
                        else:
                            ping = recv_timestamp - ts
                            out_audio = np.nan_to_num(out_audio)
                            out_audio = np.clip(out_audio, -1.0, 1.0)
                            out_audio = (out_audio * 32767).astype(np.int16).tobytes()
                            send_timestamp = round(time() * 1000)
                            header = struct.pack("<qiffffBB", send_timestamp, ping, vol, float(perf[0]), float(perf[1]), float(perf[2]), 0, 0)
                            await websocket.send_bytes(header + out_audio)
                except WebSocketDisconnect:
                    logger.debug("WebSocket client disconnected from /ws/voice")
                except Exception as e:
                    logger.exception(f"Error in websocket_voice: {e}")

            cls._instance = app_fastapi
            logger.info("Initialized.")
            return cls._instance

        return cls._instance