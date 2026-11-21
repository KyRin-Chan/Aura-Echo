import asyncio
import logging
from time import time

import numpy as np
import socketio

from voice_changer.VoiceChangerManager import VoiceChangerManager
from voice_changer.utils.PacketLossConcealment import ClockDriftTracker

logger = logging.getLogger(__name__)


class MMVC_Namespace(socketio.AsyncNamespace):
    async def emitTo(self, vol, perf, err):
        if not self.sids:
            return
        for sid in list(self.sids):
            try:
                if err is not None:
                    error_code, error_message = err
                    await self.emit("error", [error_code, error_message], to=sid)
                else:
                    await self.emit("server_stats", [vol, perf], to=sid)
            except Exception as e:
                logger.debug(f"Failed to emit stats to SID {sid}: {e}")

    def emit_coroutine(self, vol, perf, err):
        if self.sids and self._loop is not None and not self._loop.is_closed():
            try:
                asyncio.run_coroutine_threadsafe(self.emitTo(vol, perf, err), self._loop)
            except Exception as e:
                logger.debug(f"Failed to schedule emitTo: {e}")

    async def emitProgress(self, step: int, total: int, label: str):
        if not self.sids:
            return
        for sid in list(self.sids):
            try:
                await self.emit("model_loading_progress", {"step": step, "total": total, "label": label}, to=sid)
            except Exception as e:
                logger.debug(f"Failed to emit progress to SID {sid}: {e}")

    def emit_progress_coroutine(self, step: int, total: int, label: str):
        if self.sids and self._loop is not None and not self._loop.is_closed():
            try:
                asyncio.run_coroutine_threadsafe(self.emitProgress(step, total, label), self._loop)
            except Exception as e:
                logger.debug(f"Failed to schedule emitProgress: {e}")

    def __init__(self, namespace: str, voiceChangerManager: VoiceChangerManager):
        super().__init__(namespace)
        self.voiceChangerManager = voiceChangerManager
        self._loop: asyncio.AbstractEventLoop | None = None
        self.sids: set[str] = set()
        self.clock_trackers: dict[str, ClockDriftTracker] = {}
        self.stale_stats = {}
        self.voiceChangerManager.setEmitTo(self.emit_coroutine)
        self.voiceChangerManager.setProgressEmitTo(self.emit_progress_coroutine)

    @classmethod
    def get_instance(cls, voiceChangerManager: VoiceChangerManager):
        if not hasattr(cls, "_instance"):
            cls._instance = cls("/test", voiceChangerManager)
        return cls._instance

    def on_connect(self, sid, environ, ext):
        self.sids.add(sid)
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
        logger.info(f"Connected SID: {sid}")
        self.clock_trackers[sid] = ClockDriftTracker(window_sec=60.0)
        self.stale_stats[sid] = {
            "count": 0,
            "ages": [],
            "last_log_time": 0.0
        }

    async def on_request_message(self, sid, msg):
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        recv_timestamp = round(time() * 1000)

        ts, data = msg
        input_audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

        # Align clock offsets with 60s sliding window to resist long-term clock drift
        diff = recv_timestamp - ts
        tracker = self.clock_trackers.setdefault(sid, ClockDriftTracker(window_sec=60.0))
        relative_age = tracker.update(diff)

        stats = self.stale_stats.setdefault(sid, {"count": 0, "ages": [], "last_log_time": 0.0})

        if relative_age > 150:
            stats["count"] += 1
            stats["ages"].append(relative_age)
            current_time = time()
            if stats["last_log_time"] == 0.0:
                stats["last_log_time"] = current_time
            elif current_time - stats["last_log_time"] >= 3.0:
                avg_age = sum(stats["ages"]) // len(stats["ages"]) if stats["ages"] else 0
                logger.warning(f"[SIO] Stale packets detected: {stats['count']} packet(s) skipped in the last {current_time - stats['last_log_time']:.1f}s (avg relative age: {avg_age}ms). Skipping inference.")
                stats["count"] = 0
                stats["ages"] = []
                stats["last_log_time"] = current_time

            out_audio, vol, perf, err = await asyncio.to_thread(
                self.voiceChangerManager.skip_voice, input_audio
            )
        else:
            if stats["count"] > 0:
                current_time = time()
                duration = (current_time - stats["last_log_time"]) if stats["last_log_time"] > 0.0 else 0.0
                avg_age = sum(stats["ages"]) // len(stats["ages"]) if stats["ages"] else 0
                logger.warning(f"[SIO] Stale packets detected: {stats['count']} packet(s) skipped over {duration:.1f}s (avg relative age: {avg_age}ms). Skipping inference.")
                stats["count"] = 0
                stats["ages"] = []
                stats["last_log_time"] = 0.0

            out_audio, vol, perf, err = await asyncio.to_thread(
                self.voiceChangerManager.change_voice, input_audio
            )

        if err is not None:
            error_code, error_message = err
            await self.emit("error", [error_code, error_message], to=sid)
        else:
            ping = recv_timestamp - ts
            out_audio = np.nan_to_num(out_audio)
            out_audio = np.round(np.clip(out_audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
            send_timestamp = round(time() * 1000)
            await self.emit("response", [send_timestamp, out_audio, ping, vol, perf], to=sid)

    def on_disconnect(self, sid):
        self.sids.discard(sid)
        logger.info(f"Disconnected SID: {sid}")
        self.clock_trackers.pop(sid, None)
        if sid in self.stale_stats:
            stats = self.stale_stats[sid]
            if stats["count"] > 0:
                current_time = time()
                duration = (current_time - stats["last_log_time"]) if stats["last_log_time"] > 0.0 else 0.0
                avg_age = sum(stats["ages"]) // len(stats["ages"]) if stats["ages"] else 0
                logger.warning(f"[SIO] Stale packets detected for SID {sid}: {stats['count']} packet(s) skipped over {duration:.1f}s (avg relative age: {avg_age}ms). Skipping inference.")
            del self.stale_stats[sid]
