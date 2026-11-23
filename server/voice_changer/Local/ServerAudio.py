import logging
import sys
import threading
from time import monotonic
from typing import Optional, Protocol, Union

import numpy as np
import sounddevice as sd

from const import SERVER_DEVICE_SAMPLE_RATES
from voice_changer.Local.AudioDeviceList import (
    PORTAUDIO_LOCK,
    checkSamplingRate,
    list_audio_device,
    reinitialize_portaudio,
)
from voice_changer.Local.WasapiCommCapture import WasapiCommCapture
from voice_changer.common.ServerAEC import ServerAEC
from voice_changer.common.ServerDFN3 import ServerDFN3
from voice_changer.utils.SystemPriority import register_current_thread_mmcss
from voice_changer.utils.VoiceChangerModel import AudioInOutFloat
from voice_changer.VoiceChangerSettings import VoiceChangerSettings

logger = logging.getLogger(__name__)

ERR_SAMPLE_RATE_NOT_SUPPORTED = """Specified sample rate is not supported by all selected audio devices.
Available sample rates:
  [Input]: %s
  [Output]: %s
  [Monitor]: %s"""
ERR_GENERIC_SERVER_AUDIO_ERROR = "A server audio error occurred."


class ServerAudioCallbacks(Protocol):
    def on_audio(self, unpackedData: AudioInOutFloat) -> tuple[AudioInOutFloat, float, list[Union[int, float]], tuple | None]:
        ...

    def on_skip_audio(self, unpackedData: AudioInOutFloat) -> tuple[AudioInOutFloat, float, list[Union[int, float]], tuple | None]:
        ...

    def emit_to(self, volume: float, performance: list[float], err: tuple[str, str] | None):
        ...


class Float32RingBuffer:
    """
    Pre-allocated thread-safe 1D float32 ring buffer with zero runtime allocations
    during reads/writes in real-time audio callbacks.
    """

    def __init__(self, capacity: int):
        self._cap = max(1, int(capacity))
        self._buf = np.zeros(self._cap, dtype=np.float32)
        self._r = 0
        self._w = 0
        self._count = 0
        self._lock = threading.Lock()

    def clear(self) -> None:
        with self._lock:
            self._r = 0
            self._w = 0
            self._count = 0

    def available(self) -> int:
        with self._lock:
            return self._count

    def write(self, data: np.ndarray) -> None:
        arr = np.asarray(data, dtype=np.float32).reshape(-1)
        n = int(arr.shape[0])
        if n == 0:
            return

        with self._lock:
            if n >= self._cap:
                self._buf[:] = arr[-self._cap :]
                self._r = 0
                self._w = 0
                self._count = self._cap
                return

            if self._count + n > self._cap:
                overflow = (self._count + n) - self._cap
                self._r = (self._r + overflow) % self._cap
                self._count = self._cap - n

            first = min(n, self._cap - self._w)
            self._buf[self._w : self._w + first] = arr[:first]
            if first < n:
                self._buf[: n - first] = arr[first:]
            self._w = (self._w + n) % self._cap
            self._count += n

    def read_into(self, dst: np.ndarray) -> int:
        n = int(dst.shape[0])
        if n == 0:
            return 0
        with self._lock:
            to_read = min(n, self._count)
            if to_read == 0:
                return 0
            first = min(to_read, self._cap - self._r)
            dst[:first] = self._buf[self._r : self._r + first]
            if first < to_read:
                dst[first:to_read] = self._buf[: to_read - first]
            self._r = (self._r + to_read) % self._cap
            self._count -= to_read
            return to_read

    def read_exact_into(self, dst: np.ndarray) -> bool:
        n = int(dst.shape[0])
        with self._lock:
            if self._count < n:
                return False
            first = min(n, self._cap - self._r)
            dst[:first] = self._buf[self._r : self._r + first]
            if first < n:
                dst[first:] = self._buf[: n - first]
            self._r = (self._r + n) % self._cap
            self._count -= n
            return True

    def drop_oldest(self, n: int) -> int:
        if n <= 0:
            return 0
        with self._lock:
            drop = min(n, self._count)
            self._r = (self._r + drop) % self._cap
            self._count -= drop
            return drop

    def drop_with_crossfade(self, drop_n: int, xfade_buf: np.ndarray) -> int:
        """Drop `drop_n` oldest samples while recording the first `len(xfade_buf)` dropped
        samples into `xfade_buf` so the caller can blend seamlessly."""
        xfade_len = int(xfade_buf.shape[0])
        with self._lock:
            drop = min(drop_n, self._count)
            if drop <= 0:
                return 0
            copy_len = min(xfade_len, drop)
            first = min(copy_len, self._cap - self._r)
            xfade_buf[:first] = self._buf[self._r : self._r + first]
            if first < copy_len:
                xfade_buf[first:copy_len] = self._buf[: copy_len - first]
            self._r = (self._r + drop) % self._cap
            self._count -= drop
            return copy_len


class ServerAudio:
    def __init__(self, callbacks: ServerAudioCallbacks, settings: VoiceChangerSettings):
        self.settings = settings
        self.callbacks = callbacks
        self.serverAudioInputDevices = None
        self.serverAudioOutputDevices = None
        self.performance = [0, 0, 0]

        self.stream = None
        self.monitor = None
        self.running = False
        self._state_lock = threading.RLock()

        self._block_frame = 2048
        self._out_sr = 48000
        self._mon_sr = 48000
        self._has_monitor = False

        self.in_ring: Optional[Float32RingBuffer] = None
        self.out_ring: Optional[Float32RingBuffer] = None
        self.mon_ring: Optional[Float32RingBuffer] = None

        self._in_event = threading.Event()
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        # Preallocated scratch arrays for callbacks & worker
        self._cb_out_buf = np.zeros(16384, dtype=np.float32)
        self._mon_out_buf = np.zeros(16384, dtype=np.float32)
        self._out_xfade_scratch = np.zeros(128, dtype=np.float32)
        self._mon_xfade_scratch = np.zeros(128, dtype=np.float32)
        self._worker_in_buf = np.zeros(2048, dtype=np.float32)
        self._worker_skip_buf = np.zeros(2048, dtype=np.float32)
        self._aec_lock = threading.Lock()
        self._comm_capture: Optional[WasapiCommCapture] = None
        self._ns_processor: Optional[ServerAEC] = None
        self._dfn3: Optional[ServerDFN3] = None

        self._fade_in_64 = np.linspace(0.0, 1.0, 64, dtype=np.float32)
        self._fade_out_128 = np.linspace(1.0, 0.0, 128, dtype=np.float32)
        self._xfade_alpha_128 = np.linspace(0.0, 1.0, 128, dtype=np.float32)
        self._out_underrun = True
        self._out_last_sample = 0.0
        self._out_has_started = False
        self._out_target_cushion = 0
        self._out_min_cushion = 0
        self._out_cushion_step = 2048
        self._out_max_cushion = 4096
        self._comm_active = False
        self._out_stable_callbacks = 0
        self._out_decay_threshold = 234
        self._out_trim_pending = False
        self._out_underrun_count = 0
        self._last_logged_underruns = 0
        self._dropped_input_chunks = 0
        self._last_diag_log_ts = 0.0
        self._last_cb_frames = 0
        self._cb_in_overflows = 0
        self._cb_out_underflows = 0
        self._last_logged_in_overflows = 0
        self._last_logged_out_underflows = 0
        self._max_infer_ms = 0.0
        self._max_worker_gap_ms = 0.0
        self._last_worker_iter_ts = 0.0
        self._mon_underrun = True
        self._mon_last_sample = 0.0

    def getServerInputAudioDevice(self, index: int):
        if index is None or index < 0:
            return None
        audioinput, _ = list_audio_device()
        serverAudioDevice = [x for x in audioinput if x.index == index]
        return serverAudioDevice[0] if len(serverAudioDevice) > 0 else None

    def getServerOutputAudioDevice(self, index: int):
        if index is None or index < 0:
            return None
        _, audiooutput = list_audio_device()
        serverAudioDevice = [x for x in audiooutput if x.index == index]
        return serverAudioDevice[0] if len(serverAudioDevice) > 0 else None

    ###########################################
    # Inference Worker Section
    ###########################################

    def _sync_audio_processing_state(self, active: bool) -> None:
        if not active:
            with self._aec_lock:
                self._ns_processor = None
                self._dfn3 = None
            return

        want_ns1 = bool(getattr(self.settings, 'noiseSuppression', False))
        want_ns2 = bool(getattr(self.settings, 'noiseSuppression2', False))
        ns2_strength = str(getattr(self.settings, 'noiseSuppression2Strength', 'medium'))

        with self._aec_lock:
            comm_cap = self._comm_capture
            cur_ns = self._ns_processor
            cur_dfn3 = self._dfn3

        if comm_cap is not None and comm_cap.is_running:
            comm_cap.set_deep_noise_suppression(want_ns1)

        win_apo_handles_ns1 = (
            comm_cap is not None
            and comm_cap.is_running
            and (comm_cap.os_aec_active or comm_cap.os_ns_active)
        )
        need_webrtc_ns1 = want_ns1 and not win_apo_handles_ns1

        if need_webrtc_ns1:
            if (
                cur_ns is None
                or cur_ns.sample_rate != self._out_sr
                or cur_ns.echo_cancellation
                or not cur_ns.noise_suppression
            ):
                new_ns = ServerAEC(
                    sample_rate=self._out_sr,
                    echo_cancellation=False,
                    noise_suppression=True,
                )
            else:
                new_ns = cur_ns
        else:
            new_ns = None

        if want_ns2:
            if cur_dfn3 is None or cur_dfn3.sample_rate != self._out_sr:
                new_dfn3 = ServerDFN3(sample_rate=self._out_sr, strength=ns2_strength)
            else:
                cur_dfn3.set_strength(ns2_strength)
                new_dfn3 = cur_dfn3
        else:
            new_dfn3 = None

        with self._aec_lock:
            self._ns_processor = new_ns
            self._dfn3 = new_dfn3

        logger.info(
            f"[ServerAudio] Input processing state: win_comm_aec={comm_cap is not None and comm_cap.is_running}, "
            f"os_aec={comm_cap.os_aec_active if comm_cap is not None else False}, "
            f"ns1={'os_apo' if (want_ns1 and win_apo_handles_ns1) else ('webrtc' if new_ns is not None else 'off')}, "
            f"ns2={f'dfn3({new_dfn3.strength})' if (new_dfn3 is not None and new_dfn3.available) else ('unavailable' if want_ns2 else 'off')}, "
            f"sr={self._out_sr}"
        )

    def _apply_input_gain(self, mic_buf: np.ndarray) -> None:
        in_gain = float(self.settings.serverInputAudioGain)
        if in_gain != 1.0:
            mic_buf *= np.float32(in_gain)

    def _apply_input_processing(self, mic_buf: np.ndarray) -> None:
        with self._aec_lock:
            ns_proc = self._ns_processor
            dfn3 = self._dfn3
            if ns_proc is not None:
                mic_buf[:] = ns_proc.process(mic_buf, None)
            if dfn3 is not None and dfn3.available:
                mic_buf[:] = dfn3.process(mic_buf)
        self._apply_input_gain(mic_buf)

    def _on_comm_input_frames(self, mono: np.ndarray) -> None:
        in_ring = self.in_ring
        if in_ring is not None and mono.size > 0:
            in_ring.write(mono)
            self._in_event.set()

    def _maybe_log_diagnostics(self, now: float) -> None:
        if now - self._last_diag_log_ts < 3.0:
            return
        cur_underruns = self._out_underrun_count
        new_underruns = cur_underruns - self._last_logged_underruns
        dropped = self._dropped_input_chunks
        cur_in_ovf = self._cb_in_overflows
        new_in_ovf = cur_in_ovf - self._last_logged_in_overflows
        cur_out_udf = self._cb_out_underflows
        new_out_udf = cur_out_udf - self._last_logged_out_underflows

        has_event = new_underruns > 0 or dropped > 0 or new_in_ovf > 0 or new_out_udf > 0
        if has_event:
            cushion = self._out_target_cushion
            cushion_ms = (cushion / float(max(1, self._out_sr))) * 1000.0
            in_avail = self.in_ring.available() if self.in_ring is not None else 0
            out_avail = self.out_ring.available() if self.out_ring is not None else 0
            logger.warning(
                f"[ServerAudio] Performance fluctuation detected: "
                f"{new_underruns} output underrun(s), {dropped} stale input chunk(s) skipped; "
                f"adaptive output cushion={cushion} samples ({cushion_ms:.1f}ms), "
                f"max_infer={self._max_infer_ms:.1f}ms, max_worker_gap={self._max_worker_gap_ms:.1f}ms, "
                f"cb_frames={self._last_cb_frames}/{self._block_frame}, pa_ovf={new_in_ovf}, "
                f"pa_udf={new_out_udf}, in_avail={in_avail}, out_avail={out_avail}, "
                f"out_underrun={self._out_underrun}."
            )
            self._last_logged_underruns = cur_underruns
            self._last_logged_in_overflows = cur_in_ovf
            self._last_logged_out_underflows = cur_out_udf
            self._dropped_input_chunks = 0

        if has_event or self._last_diag_log_ts > 0.0:
            self._max_infer_ms = 0.0
            self._max_worker_gap_ms = 0.0
            self._last_diag_log_ts = now

    def _inference_worker_loop(self) -> None:
        try:
            register_current_thread_mmcss()
        except Exception:
            pass

        block_frame = self._block_frame
        last_emit_ts = 0.0

        while not self._stop_event.is_set():
            self._in_event.wait(timeout=0.05)
            self._in_event.clear()

            while not self._stop_event.is_set():
                in_ring = self.in_ring
                if in_ring is None:
                    break

                avail = in_ring.available()
                if avail < block_frame:
                    break

                iter_ts = monotonic()
                if self._last_worker_iter_ts > 0.0:
                    self._max_worker_gap_ms = max(
                        self._max_worker_gap_ms, (iter_ts - self._last_worker_iter_ts) * 1000.0
                    )
                self._last_worker_iter_ts = iter_ts

                # Keep latency bounded if producer outpaces GPU inference while preserving context continuity
                if avail > block_frame * 2:
                    drop_chunks = (avail - block_frame) // block_frame
                    skip_cb = getattr(self.callbacks, "on_skip_audio", None)
                    for _ in range(drop_chunks):
                        if self._stop_event.is_set():
                            break
                        if not in_ring.read_exact_into(self._worker_skip_buf):
                            break
                        self._apply_input_processing(self._worker_skip_buf)
                        if skip_cb is not None:
                            try:
                                skip_cb(self._worker_skip_buf)
                            except Exception as skip_err:
                                logger.debug(f"on_skip_audio error: {skip_err}")
                        self._dropped_input_chunks += 1
                    if self._stop_event.is_set():
                        break

                if not in_ring.read_exact_into(self._worker_in_buf):
                    break

                self._apply_input_processing(self._worker_in_buf)

                try:
                    out_wav, vol, perf, err = self.callbacks.on_audio(self._worker_in_buf)
                    if self._stop_event.is_set():
                        break
                    self.performance = perf
                    infer_ms = float(perf[1]) * 1000.0 if (perf and len(perf) > 1) else 0.0
                    self._max_infer_ms = max(self._max_infer_ms, infer_ms)

                    if out_wav is not None and out_wav.size > 0:
                        if self.out_ring is not None:
                            self.out_ring.write(out_wav)
                        if self._has_monitor and self.mon_ring is not None:
                            self.mon_ring.write(out_wav)

                    now = monotonic()
                    if err is not None or (now - last_emit_ts >= 0.05):
                        self.callbacks.emit_to(vol, self.performance, err)
                        last_emit_ts = now
                except Exception as e:
                    if self._stop_event.is_set():
                        break
                    logger.exception(e)
                    self.callbacks.emit_to(
                        0,
                        self.performance,
                        ('ERR_GENERIC_SERVER_AUDIO_ERROR', ERR_GENERIC_SERVER_AUDIO_ERROR),
                    )
                self._maybe_log_diagnostics(monotonic())

    ###########################################
    # Callback Section (Lock-free / Zero-Alloc)
    ###########################################

    def _fill_underrun_tail(self, buf: np.ndarray, got: int, frames: int, last_sample: float) -> None:
        tail_start = buf[got - 1] if got > 0 else last_sample
        fade_len = min(128, frames - got)
        if fade_len > 0 and abs(tail_start) > 1e-6:
            buf[got : got + fade_len] = tail_start * self._fade_out_128[:fade_len]
            if got + fade_len < frames:
                buf[got + fade_len : frames] = 0.0
        else:
            buf[got:frames] = 0.0

    def _fill_output_buffer(self, outdata: np.ndarray, frames: int) -> None:
        if frames > self._cb_out_buf.shape[0]:
            self._cb_out_buf = np.zeros(frames, dtype=np.float32)

        buf = self._cb_out_buf[:frames]
        out_ring = self.out_ring
        avail = out_ring.available() if out_ring is not None else 0

        if self._out_underrun:
            needed = frames + self._out_target_cushion
            if out_ring is None or avail < needed:
                buf.fill(0.0)
                self._out_last_sample = 0.0
            else:
                got = out_ring.read_into(buf)
                if got == frames:
                    fi_len = min(64, frames)
                    buf[:fi_len] *= self._fade_in_64[:fi_len]
                    self._out_underrun = False
                    self._out_has_started = True
                    self._out_stable_callbacks = 0
                    self._out_trim_pending = False
                    self._out_last_sample = float(buf[frames - 1])
                else:
                    if got > 0:
                        fi_len = min(64, got)
                        buf[:fi_len] *= self._fade_in_64[:fi_len]
                    self._fill_underrun_tail(buf, got, frames, self._out_last_sample)
                    self._out_last_sample = 0.0
        else:
            target_avail = frames + self._out_target_cushion
            high_water = target_avail + max(frames * 2, self._block_frame * 2, int(self._out_sr * 0.08))
            xfade_copied = 0
            if out_ring is not None and (
                avail > high_water or (self._out_trim_pending and avail > target_avail)
            ):
                drop_n = avail - target_avail
                xfade_copied = out_ring.drop_with_crossfade(drop_n, self._out_xfade_scratch)
            self._out_trim_pending = False

            got = out_ring.read_into(buf) if out_ring is not None else 0
            if xfade_copied > 0 and got > 0:
                blend_len = min(xfade_copied, got, 128)
                alpha = (
                    self._xfade_alpha_128
                    if blend_len == 128
                    else np.linspace(0.0, 1.0, blend_len, dtype=np.float32)
                )
                buf[:blend_len] = (
                    buf[:blend_len] * alpha + self._out_xfade_scratch[:blend_len] * (1.0 - alpha)
                )

            if got == frames:
                self._out_last_sample = float(buf[frames - 1])
                self._out_stable_callbacks += 1
                min_cushion = max(
                    self._out_min_cushion,
                    self._out_cushion_step if self._out_underrun_count >= 2 else 0,
                )
                if (
                    self._out_target_cushion > min_cushion
                    and self._out_stable_callbacks >= self._out_decay_threshold
                    and float(np.max(np.abs(buf))) < 1e-3
                ):
                    self._out_target_cushion = max(
                        min_cushion,
                        self._out_target_cushion - self._out_cushion_step,
                    )
                    self._out_stable_callbacks = 0
                    self._out_trim_pending = True
            else:
                self._fill_underrun_tail(buf, got, frames, self._out_last_sample)
                self._out_underrun = True
                self._out_last_sample = 0.0
                self._out_stable_callbacks = 0
                self._out_trim_pending = False
                if self._out_has_started:
                    self._out_underrun_count += 1
                    self._out_target_cushion = min(
                        self._out_max_cushion,
                        self._out_target_cushion + self._out_cushion_step,
                    )

        outdata[:] = buf[:, None] * np.float32(self.settings.serverOutputAudioGain)

    def audio_stream_callback(self, indata: np.ndarray, outdata: np.ndarray, frames, times, status):
        try:
            self._last_cb_frames = int(frames)
            if status:
                if getattr(status, "input_overflow", False):
                    self._cb_in_overflows += 1
                if getattr(status, "output_underflow", False):
                    self._cb_out_underflows += 1

            # Fast mono mixdown without librosa overhead (gain is applied post-denoising in worker)
            mono = indata[:, 0] if indata.shape[1] == 1 else indata.mean(axis=1)
            if self.in_ring is not None:
                self.in_ring.write(mono)
                self._in_event.set()

            self._fill_output_buffer(outdata, frames)
        except Exception as e:
            outdata.fill(0.0)
            logger.exception(e)

    def audio_output_only_callback(self, outdata: np.ndarray, frames, times, status):
        try:
            self._last_cb_frames = int(frames)
            if status and getattr(status, "output_underflow", False):
                self._cb_out_underflows += 1
            self._fill_output_buffer(outdata, frames)
        except Exception as e:
            outdata.fill(0.0)
            logger.exception(e)

    def audio_monitor_callback(self, outdata: np.ndarray, frames, times, status):
        try:
            if frames > self._mon_out_buf.shape[0]:
                self._mon_out_buf = np.zeros(frames, dtype=np.float32)
            buf = self._mon_out_buf[:frames]

            mon_ring = self.mon_ring
            if mon_ring is None:
                outdata.fill(0.0)
                return

            # Protect against clock drift between main output and monitor audio devices
            high_water = max(self._block_frame * 3, int(self._mon_sr * 0.12))
            target_level = max(self._block_frame, int(self._mon_sr * 0.04))
            avail = mon_ring.available()
            xfade_copied = 0
            if avail > high_water:
                drop_n = avail - target_level
                xfade_copied = mon_ring.drop_with_crossfade(drop_n, self._mon_xfade_scratch)

            got = mon_ring.read_into(buf)
            if xfade_copied > 0 and got > 0:
                blend_len = min(xfade_copied, got, 128)
                alpha = np.linspace(0.0, 1.0, blend_len, dtype=np.float32)
                buf[:blend_len] = (
                    buf[:blend_len] * alpha + self._mon_xfade_scratch[:blend_len] * (1.0 - alpha)
                )

            if got == frames:
                if self._mon_underrun:
                    fi_len = min(64, frames)
                    buf[:fi_len] *= self._fade_in_64[:fi_len]
                    self._mon_underrun = False
                self._mon_last_sample = float(buf[frames - 1])
            else:
                if got > 0 and self._mon_underrun:
                    fi_len = min(64, got)
                    buf[:fi_len] *= self._fade_in_64[:fi_len]
                self._fill_underrun_tail(buf, got, frames, self._mon_last_sample)
                self._mon_underrun = True
                self._mon_last_sample = 0.0

            outdata[:] = buf[:, None] * np.float32(self.settings.serverMonitorAudioGain)
        except Exception as e:
            outdata.fill(0.0)
            logger.exception(e)

    ###########################################
    # Main Loop Section
    ###########################################

    def _start_worker_and_rings(
        self,
        block_frame: int,
        has_monitor: bool,
        sample_rate: Optional[int] = None,
        mon_sample_rate: Optional[int] = None,
        input_device_name: Optional[str] = None,
        use_win_comm_aec: bool = False,
    ) -> bool:
        self._block_frame = block_frame
        self._has_monitor = has_monitor
        self._out_sr = int(
            sample_rate
            if sample_rate is not None
            else (getattr(self.settings, 'serverOutputAudioSampleRate', 48000) or 48000)
        )
        self._mon_sr = int(
            mon_sample_rate
            if mon_sample_rate is not None
            else (getattr(self.settings, 'serverMonitorAudioSampleRate', 48000) or 48000)
        )

        if self._worker_in_buf.shape[0] != block_frame:
            self._worker_in_buf = np.zeros(block_frame, dtype=np.float32)
        if self._worker_skip_buf.shape[0] != block_frame:
            self._worker_skip_buf = np.zeros(block_frame, dtype=np.float32)

        self.in_ring = Float32RingBuffer(max(block_frame * 8, 32768))
        self.out_ring = Float32RingBuffer(max(block_frame * 8, 32768))
        self.mon_ring = Float32RingBuffer(max(block_frame * 12, int(self._mon_sr * 0.5))) if has_monitor else None

        comm_active = False
        if use_win_comm_aec:
            comm_cap = WasapiCommCapture(
                device_name=input_device_name,
                sample_rate=self._out_sr,
                on_audio=self._on_comm_input_frames,
                enable_deep_ns=bool(getattr(self.settings, 'noiseSuppression', False)),
            )
            if comm_cap.start():
                with self._aec_lock:
                    self._comm_capture = comm_cap
                comm_active = True
            else:
                logger.warning(
                    "[ServerAudio] Failed to start Windows Communications AEC capture; "
                    "falling back to standard duplex stream."
                )

        self._sync_audio_processing_state(active=True)

        self._comm_active = comm_active
        self._out_underrun = True
        self._out_last_sample = 0.0
        self._out_has_started = False
        self._out_min_cushion = block_frame if comm_active else 0
        self._out_target_cushion = self._out_min_cushion
        self._out_cushion_step = block_frame
        self._out_max_cushion = block_frame * 2
        self._out_stable_callbacks = 0
        self._out_decay_threshold = max(1, int(round(10.0 * self._out_sr / max(1, block_frame))))
        self._out_trim_pending = False
        self._out_underrun_count = 0
        self._last_logged_underruns = 0
        self._dropped_input_chunks = 0
        self._last_diag_log_ts = 0.0
        self._last_cb_frames = 0
        self._cb_in_overflows = 0
        self._cb_out_underflows = 0
        self._last_logged_in_overflows = 0
        self._last_logged_out_underflows = 0
        self._max_infer_ms = 0.0
        self._max_worker_gap_ms = 0.0
        self._last_worker_iter_ts = 0.0
        self._mon_underrun = True
        self._mon_last_sample = 0.0

        self._stop_event.clear()
        self._in_event.clear()
        self._worker_thread = threading.Thread(
            target=self._inference_worker_loop,
            name="ServerAudioWorker",
            daemon=True,
        )
        self._worker_thread.start()
        return comm_active

    def _open_primary_stream(
        self,
        block_frame: int,
        inputMaxChannel: int,
        outputMaxChannel: int,
        inputExtraSetting,
        outputExtraSetting,
        in_dev: int,
        out_dev: int,
        sr: int,
        comm_active: bool,
    ):
        if comm_active:
            return sd.OutputStream(
                callback=self.audio_output_only_callback,
                latency='low',
                dtype="float32",
                device=out_dev,
                blocksize=block_frame,
                samplerate=sr,
                channels=outputMaxChannel,
                extra_settings=outputExtraSetting,
            )
        return sd.Stream(
            callback=self.audio_stream_callback,
            latency='low',
            dtype="float32",
            device=(in_dev, out_dev),
            blocksize=block_frame,
            samplerate=sr,
            channels=(inputMaxChannel, outputMaxChannel),
            extra_settings=(inputExtraSetting, outputExtraSetting),
        )

    def run_no_monitor(
        self,
        block_frame: int,
        inputMaxChannel: int,
        outputMaxChannel: int,
        inputExtraSetting,
        outputExtraSetting,
        inputDeviceId: Optional[int] = None,
        outputDeviceId: Optional[int] = None,
        sampleRate: Optional[int] = None,
        inputDeviceName: Optional[str] = None,
        useWinCommAec: bool = False,
    ):
        in_dev = self.settings.serverInputDeviceId if inputDeviceId is None else inputDeviceId
        out_dev = self.settings.serverOutputDeviceId if outputDeviceId is None else outputDeviceId
        sr = self.settings.serverInputAudioSampleRate if sampleRate is None else sampleRate
        comm_active = self._start_worker_and_rings(
            block_frame,
            has_monitor=False,
            sample_rate=sr,
            input_device_name=inputDeviceName,
            use_win_comm_aec=useWinCommAec,
        )
        with PORTAUDIO_LOCK:
            self.stream = self._open_primary_stream(
                block_frame,
                inputMaxChannel,
                outputMaxChannel,
                inputExtraSetting,
                outputExtraSetting,
                in_dev,
                out_dev,
                sr,
                comm_active,
            )
            self.stream.start()

    def run_with_monitor(
        self,
        block_frame: int,
        inputMaxChannel: int,
        outputMaxChannel: int,
        monitorMaxChannel: int,
        inputExtraSetting,
        outputExtraSetting,
        monitorExtraSetting,
        inputDeviceId: Optional[int] = None,
        outputDeviceId: Optional[int] = None,
        monitorDeviceId: Optional[int] = None,
        sampleRate: Optional[int] = None,
        inputDeviceName: Optional[str] = None,
        useWinCommAec: bool = False,
    ):
        in_dev = self.settings.serverInputDeviceId if inputDeviceId is None else inputDeviceId
        out_dev = self.settings.serverOutputDeviceId if outputDeviceId is None else outputDeviceId
        mon_dev = self.settings.serverMonitorDeviceId if monitorDeviceId is None else monitorDeviceId
        sr = self.settings.serverInputAudioSampleRate if sampleRate is None else sampleRate
        mon_sr = self.settings.serverMonitorAudioSampleRate if sampleRate is None else sampleRate
        comm_active = self._start_worker_and_rings(
            block_frame,
            has_monitor=True,
            sample_rate=sr,
            mon_sample_rate=mon_sr,
            input_device_name=inputDeviceName,
            use_win_comm_aec=useWinCommAec,
        )
        with PORTAUDIO_LOCK:
            self.stream = self._open_primary_stream(
                block_frame,
                inputMaxChannel,
                outputMaxChannel,
                inputExtraSetting,
                outputExtraSetting,
                in_dev,
                out_dev,
                sr,
                comm_active,
            )
            self.monitor = sd.OutputStream(
                callback=self.audio_monitor_callback,
                latency='low',
                dtype="float32",
                device=mon_dev,
                blocksize=block_frame,
                samplerate=mon_sr,
                channels=monitorMaxChannel,
                extra_settings=monitorExtraSetting
            )
            self.stream.start()
            self.monitor.start()

    def stop(self):
        with self._state_lock:
            self.running = False
            self.settings.serverAudioStated = 0
            self._stop_event.set()
            self._in_event.set()

            with self._aec_lock:
                old_comm = self._comm_capture
                self._comm_capture = None
            if old_comm is not None:
                try:
                    old_comm.stop()
                except Exception:
                    pass

            if self.stream is not None or self.monitor is not None:
                with PORTAUDIO_LOCK:
                    if self.stream is not None:
                        try:
                            self.stream.close()
                        finally:
                            self.stream = None
                    if self.monitor is not None:
                        try:
                            self.monitor.close()
                        finally:
                            self.monitor = None

            if self._worker_thread is not None and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=2.0)
            self._worker_thread = None

            self._sync_audio_processing_state(active=False)

            if self.in_ring is not None:
                self.in_ring.clear()
            if self.out_ring is not None:
                self.out_ring.clear()
            if self.mon_ring is not None:
                self.mon_ring.clear()

    ###########################################
    # Start Section
    ###########################################
    def start(self):
        with self._state_lock:
            self.stop()

            input_dev_id = int(self.settings.serverInputDeviceId)
            output_dev_id = int(self.settings.serverOutputDeviceId)
            monitor_dev_id = int(self.settings.serverMonitorDeviceId)

            if input_dev_id < 0 or output_dev_id < 0:
                logger.warning("Input or output device is not selected.")
                return

            with PORTAUDIO_LOCK:
                reinitialize_portaudio()

                serverInputAudioDevice = self.getServerInputAudioDevice(input_dev_id)
                serverOutputAudioDevice = self.getServerOutputAudioDevice(output_dev_id)
                serverMonitorAudioDevice = self.getServerOutputAudioDevice(monitor_dev_id)

                if serverInputAudioDevice is None or serverOutputAudioDevice is None:
                    logger.error("Input or output device is not selected.")
                    self.callbacks.emit_to(0, self.performance, ('ERR_GENERIC_SERVER_AUDIO_ERROR', ERR_GENERIC_SERVER_AUDIO_ERROR))
                    return

                wasapiExclusiveMode = bool(self.settings.exclusiveMode)

                inputChannels = serverInputAudioDevice.maxInputChannels
                inputExtraSetting = None
                if serverInputAudioDevice and "WASAPI" in serverInputAudioDevice.hostAPI:
                    inputExtraSetting = sd.WasapiSettings(exclusive=wasapiExclusiveMode, auto_convert=not wasapiExclusiveMode)
                elif serverInputAudioDevice and "ASIO" in serverInputAudioDevice.hostAPI and self.settings.asioInputChannel != -1:
                    inputExtraSetting = sd.AsioSettings(channel_selectors=[self.settings.asioInputChannel])
                    inputChannels = 1

                outputChannels = serverOutputAudioDevice.maxOutputChannels
                outputExtraSetting = None
                if serverOutputAudioDevice and "WASAPI" in serverOutputAudioDevice.hostAPI:
                    outputExtraSetting = sd.WasapiSettings(exclusive=wasapiExclusiveMode, auto_convert=not wasapiExclusiveMode)
                elif serverOutputAudioDevice and "ASIO" in serverOutputAudioDevice.hostAPI and self.settings.asioOutputChannel != -1:
                    outputExtraSetting = sd.AsioSettings(channel_selectors=[self.settings.asioOutputChannel])
                    outputChannels = 1

                monitorExtraSetting = None
                if serverMonitorAudioDevice and "WASAPI" in serverMonitorAudioDevice.hostAPI:
                    monitorExtraSetting = sd.WasapiSettings(exclusive=wasapiExclusiveMode, auto_convert=not wasapiExclusiveMode)

                logger.info("Devices:")
                logger.info(f"  [Input]: {serverInputAudioDevice} {inputExtraSetting}")
                logger.info(f"  [Output]: {serverOutputAudioDevice}, {outputExtraSetting}")
                logger.info(f"  [Monitor]: {serverMonitorAudioDevice}, {monitorExtraSetting}")

                sample_rate = int(self.settings.serverAudioSampleRate)
                self.settings.serverInputAudioSampleRate = sample_rate
                self.settings.serverOutputAudioSampleRate = sample_rate
                self.settings.serverMonitorAudioSampleRate = sample_rate

                if "WASAPI" not in serverInputAudioDevice.hostAPI and not wasapiExclusiveMode:
                    inputAudioSampleRateAvailable = checkSamplingRate(input_dev_id, sample_rate, "input")
                    outputAudioSampleRateAvailable = checkSamplingRate(output_dev_id, sample_rate, "output")
                    monitorAudioSampleRateAvailable = checkSamplingRate(monitor_dev_id, sample_rate, "output") if serverMonitorAudioDevice else True

                    logger.info("Sample Rate:")
                    logger.info(f"  [Input]: {sample_rate} -> {inputAudioSampleRateAvailable}")
                    logger.info(f"  [Output]: {sample_rate} -> {outputAudioSampleRateAvailable}")
                    if serverMonitorAudioDevice is not None:
                        logger.info(f"  [Monitor]: {sample_rate} -> {monitorAudioSampleRateAvailable}")

                    if not inputAudioSampleRateAvailable or not outputAudioSampleRateAvailable or not monitorAudioSampleRateAvailable:
                        logger.info("Checking Available Sample Rate:")
                        availableInputSampleRate = []
                        availableOutputSampleRate = []
                        availableMonitorSampleRate = []
                        for sr in SERVER_DEVICE_SAMPLE_RATES:
                            if checkSamplingRate(input_dev_id, sr, "input"):
                                availableInputSampleRate.append(sr)
                            if checkSamplingRate(output_dev_id, sr, "output"):
                                availableOutputSampleRate.append(sr)
                            if serverMonitorAudioDevice is not None:
                                if checkSamplingRate(monitor_dev_id, sr, "output"):
                                    availableMonitorSampleRate.append(sr)
                        err = ERR_SAMPLE_RATE_NOT_SUPPORTED % (availableInputSampleRate, availableOutputSampleRate, availableMonitorSampleRate)
                        self.callbacks.emit_to(
                            0,
                            self.performance,
                            ('ERR_SAMPLE_RATE_NOT_SUPPORTED', err)
                        )
                        logger.error(err)
                        return

                block_frame = max(
                    128,
                    int(round((self.settings.serverReadChunkSize * 128 / 48000.0) * sample_rate)),
                )

                want_echo_cancel = bool(getattr(self.settings, 'echoCancel', False))
                use_win_aec = (
                    sys.platform == "win32"
                    and want_echo_cancel
                    and "ASIO" not in serverInputAudioDevice.hostAPI
                    and not wasapiExclusiveMode
                )
                if want_echo_cancel and not use_win_aec:
                    logger.warning(
                        "[ServerAudio] Windows native AEC requires Windows shared mode (non-ASIO, non-exclusive); "
                        "echo cancellation disabled for current driver/mode."
                    )

                try:
                    if serverMonitorAudioDevice is None:
                        self.run_no_monitor(
                            block_frame,
                            inputChannels,
                            outputChannels,
                            inputExtraSetting,
                            outputExtraSetting,
                            inputDeviceId=input_dev_id,
                            outputDeviceId=output_dev_id,
                            sampleRate=sample_rate,
                            inputDeviceName=serverInputAudioDevice.name,
                            useWinCommAec=use_win_aec,
                        )
                    else:
                        self.run_with_monitor(
                            block_frame,
                            inputChannels,
                            outputChannels,
                            serverMonitorAudioDevice.maxOutputChannels,
                            inputExtraSetting,
                            outputExtraSetting,
                            monitorExtraSetting,
                            inputDeviceId=input_dev_id,
                            outputDeviceId=output_dev_id,
                            monitorDeviceId=monitor_dev_id,
                            sampleRate=sample_rate,
                            inputDeviceName=serverInputAudioDevice.name,
                            useWinCommAec=use_win_aec,
                        )
                    self.running = True
                    self.settings.serverAudioStated = 1
                except Exception as e:
                    self.stop()
                    self.callbacks.emit_to(0, self.performance, ('ERR_GENERIC_SERVER_AUDIO_ERROR', ERR_GENERIC_SERVER_AUDIO_ERROR))
                    logger.exception(e)

    ###########################################
    # Info Section
    ###########################################
    def get_info(self):
        data = {}
        if (
            self.running
            and self.serverAudioInputDevices is not None
            and self.serverAudioOutputDevices is not None
        ):
            data["serverAudioInputDevices"] = self.serverAudioInputDevices
            data["serverAudioOutputDevices"] = self.serverAudioOutputDevices
            return data
        try:
            audioinput, audiooutput = list_audio_device()
            self.serverAudioInputDevices = audioinput
            self.serverAudioOutputDevices = audiooutput
        except Exception as e:
            self.callbacks.emit_to(0, self.performance, ('ERR_GENERIC_SERVER_AUDIO_ERROR', ERR_GENERIC_SERVER_AUDIO_ERROR))
            logger.exception(e)

        data["serverAudioInputDevices"] = self.serverAudioInputDevices
        data["serverAudioOutputDevices"] = self.serverAudioOutputDevices
        return data

    def update_settings(self, key: str, val, old_val):
        with self._state_lock:
            if key == 'serverAudioStated':
                if val:
                    self.start()
                else:
                    self.stop()
                return
            if key == 'enableServerAudio' and not val:
                self.stop()
                return
            if key in {'noiseSuppression', 'noiseSuppression2', 'noiseSuppression2Strength'}:
                if self.running:
                    self._sync_audio_processing_state(active=True)
                return
            if self.running and key in {'echoCancel', 'exclusiveMode', 'serverInputDeviceId', 'serverOutputDeviceId', 'serverMonitorDeviceId', 'serverReadChunkSize', 'serverAudioSampleRate', 'asioInputChannel', 'asioOutputChannel'}:
                if key in {'serverInputDeviceId', 'serverOutputDeviceId'} and (val is None or int(val) < 0):
                    self.stop()
                else:
                    self.start()
