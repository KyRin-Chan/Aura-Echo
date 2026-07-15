import json
import numpy as np
import socketio
from time import time
from voice_changer.VoiceChangerManager import VoiceChangerManager
from voice_changer.utils.PacketLossConcealment import PacketLossConcealment

import asyncio

import logging
logger = logging.getLogger(__name__)

class MMVC_Namespace(socketio.AsyncNamespace):
    sid: str | None = None

    async def emitTo(self, vol, perf, err):
        if err is not None:
            error_code, error_message = err
            await self.emit("error", [error_code, error_message], to=self.sid)
        else:
            await self.emit("server_stats", [vol, perf], to=self.sid)

    def emit_coroutine(self, vol, perf, err):
        if self.sid:
            asyncio.run(self.emitTo(vol, perf, err))

    async def emitProgress(self, step: int, total: int, label: str):
        if self.sid:
            await self.emit("model_loading_progress", {"step": step, "total": total, "label": label}, to=self.sid)

    def emit_progress_coroutine(self, step: int, total: int, label: str):
        if self.sid:
            asyncio.run(self.emitProgress(step, total, label))

    def __init__(self, namespace: str, voiceChangerManager: VoiceChangerManager):
        super().__init__(namespace)
        self.voiceChangerManager = voiceChangerManager
        self.min_diffs = {}
        self.stale_stats = {}
        self.plc_buffers = {}
        # self.voiceChangerManager.voiceChanger.emitTo = self.emit_coroutine
        self.voiceChangerManager.setEmitTo(self.emit_coroutine)
        self.voiceChangerManager.setProgressEmitTo(self.emit_progress_coroutine)

    @classmethod
    def get_instance(cls, voiceChangerManager: VoiceChangerManager):
        if not hasattr(cls, "_instance"):
            cls._instance = cls("/test", voiceChangerManager)
        return cls._instance

    def on_connect(self, sid, environ, ext):
        self.sid = sid
        logger.info(f"Connected SID: {sid}")
        self.min_diffs[sid] = None
        self.stale_stats[sid] = {
            "count": 0,
            "ages": [],
            "last_log_time": 0.0
        }

    async def on_request_message(self, sid, msg):
        recv_timestamp = round(time() * 1000)

        ts, data = msg
        # Receive and send int16 instead of float32 to reduce bandwidth requirement over websocket
        input_audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768

        # Align clock offsets
        diff = recv_timestamp - ts
        min_diff = self.min_diffs.get(sid)
        if min_diff is None or diff < min_diff:
            self.min_diffs[sid] = diff
            min_diff = diff
        relative_age = diff - min_diff

        stats = self.stale_stats.setdefault(sid, {"count": 0, "ages": [], "last_log_time": 0.0})

        if relative_age > 150:
            # Packet is too stale, skip inference to catch up
            stats["count"] += 1
            stats["ages"].append(relative_age)
            current_time = time()
            if stats["last_log_time"] == 0.0:
                stats["last_log_time"] = current_time
            elif current_time - stats["last_log_time"] >= 3.0:
                avg_age = sum(stats["ages"]) // len(stats["ages"])
                logger.warning(f"[SIO] Stale packets detected: {stats['count']} packet(s) skipped in the last {current_time - stats['last_log_time']:.1f}s (avg relative age: {avg_age}ms). Skipping inference.")
                stats["count"] = 0
                stats["ages"] = []
                stats["last_log_time"] = current_time

            # Use Packet Loss Concealment to extrapolate missing audio instead of hard silence
            plc = self.plc_buffers.setdefault(sid, PacketLossConcealment())
            out_audio = plc.conceal(len(input_audio))
            vol = float(np.sqrt(np.square(out_audio).mean(dtype=np.float32))) if len(out_audio) > 0 else 0.0
            perf = [0.0, 0.0, 0.0]
            err = None
        else:
            if stats["count"] > 0:
                current_time = time()
                duration = current_time - stats["last_log_time"]
                avg_age = sum(stats["ages"]) // len(stats["ages"])
                logger.warning(f"[SIO] Stale packets detected: {stats['count']} packet(s) skipped over {duration:.1f}s (avg relative age: {avg_age}ms). Skipping inference.")
                stats["count"] = 0
                stats["ages"] = []
                stats["last_log_time"] = 0.0

            out_audio, vol, perf, err = await asyncio.to_thread(self.voiceChangerManager.change_voice, input_audio)
            if err is None:
                plc = self.plc_buffers.setdefault(sid, PacketLossConcealment())
                plc.update(out_audio)

        if err is not None:
            error_code, error_message = err
            await self.emit("error", [error_code, error_message], to=sid)
        else:
            ping = recv_timestamp - ts
            out_audio = np.nan_to_num(out_audio)  # Convert NaNs and infs to numbers
            out_audio = np.clip(out_audio, -1.0, 1.0)  # Ensure values are within valid range
            out_audio = (out_audio * 32767).astype(np.int16).tobytes()
            send_timestamp = round(time() * 1000)
            await self.emit("response", [send_timestamp, out_audio, ping, vol, perf], to=sid)

    def on_disconnect(self, sid):
        self.sid = None
        logger.info(f"Disconnected SID: {sid}")
        if sid in self.min_diffs:
            del self.min_diffs[sid]
        if sid in self.stale_stats:
            stats = self.stale_stats[sid]
            if stats["count"] > 0:
                current_time = time()
                duration = current_time - stats["last_log_time"]
                avg_age = sum(stats["ages"]) // len(stats["ages"])
                logger.warning(f"[SIO] Stale packets detected for SID {sid}: {stats['count']} packet(s) skipped over {duration:.1f}s (avg relative age: {avg_age}ms). Skipping inference.")
            del self.stale_stats[sid]
        if sid in self.plc_buffers:
            del self.plc_buffers[sid]
