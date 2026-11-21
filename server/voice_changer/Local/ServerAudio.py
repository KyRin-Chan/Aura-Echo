import logging
import threading
from time import monotonic
from typing import Optional, Protocol, Union

import numpy as np
import sounddevice as sd

from const import SERVER_DEVICE_SAMPLE_RATES
from voice_changer.Local.AudioDeviceList import checkSamplingRate, list_audio_device
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

        self._block_frame = 2048
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
        self._mon_xfade_scratch = np.zeros(128, dtype=np.float32)
        self._worker_in_buf = np.zeros(2048, dtype=np.float32)

        self._fade_in_64 = np.linspace(0.0, 1.0, 64, dtype=np.float32)
        self._fade_out_128 = np.linspace(1.0, 0.0, 128, dtype=np.float32)
        self._out_underrun = True
        self._out_last_sample = 0.0
        self._mon_underrun = True
        self._mon_last_sample = 0.0

    def getServerInputAudioDevice(self, index: int):
        audioinput, _ = list_audio_device()
        serverAudioDevice = [x for x in audioinput if x.index == index]
        return serverAudioDevice[0] if len(serverAudioDevice) > 0 else None

    def getServerOutputAudioDevice(self, index: int):
        _, audiooutput = list_audio_device()
        serverAudioDevice = [x for x in audiooutput if x.index == index]
        return serverAudioDevice[0] if len(serverAudioDevice) > 0 else None

    ###########################################
    # Inference Worker Section
    ###########################################

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

                # Keep latency bounded if producer outpaces GPU inference
                if avail > block_frame * 2:
                    drop_chunks = (avail - block_frame) // block_frame
                    if drop_chunks > 0:
                        in_ring.drop_oldest(drop_chunks * block_frame)

                if not in_ring.read_exact_into(self._worker_in_buf):
                    break

                try:
                    out_wav, vol, perf, err = self.callbacks.on_audio(self._worker_in_buf)
                    self.performance = perf

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
                    logger.exception(e)
                    self.callbacks.emit_to(
                        0,
                        self.performance,
                        ('ERR_GENERIC_SERVER_AUDIO_ERROR', ERR_GENERIC_SERVER_AUDIO_ERROR),
                    )

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

    def audio_stream_callback(self, indata: np.ndarray, outdata: np.ndarray, frames, times, status):
        try:
            # Fast mono mixdown without librosa overhead
            mono = indata[:, 0] if indata.shape[1] == 1 else indata.mean(axis=1)
            in_gain = float(self.settings.serverInputAudioGain)
            if in_gain != 1.0:
                mono = mono * np.float32(in_gain)

            if self.in_ring is not None:
                self.in_ring.write(mono)
                self._in_event.set()

            if frames > self._cb_out_buf.shape[0]:
                self._cb_out_buf = np.zeros(frames, dtype=np.float32)

            buf = self._cb_out_buf[:frames]
            got = self.out_ring.read_into(buf) if self.out_ring is not None else 0

            if got == frames:
                if self._out_underrun:
                    fi_len = min(64, frames)
                    buf[:fi_len] *= self._fade_in_64[:fi_len]
                    self._out_underrun = False
                self._out_last_sample = float(buf[frames - 1])
            else:
                if got > 0 and self._out_underrun:
                    fi_len = min(64, got)
                    buf[:fi_len] *= self._fade_in_64[:fi_len]
                self._fill_underrun_tail(buf, got, frames, self._out_last_sample)
                self._out_underrun = True
                self._out_last_sample = 0.0

            outdata[:] = buf[:, None] * np.float32(self.settings.serverOutputAudioGain)
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

    def _start_worker_and_rings(self, block_frame: int, has_monitor: bool) -> None:
        self._block_frame = block_frame
        self._has_monitor = has_monitor
        self._mon_sr = int(getattr(self.settings, 'serverMonitorAudioSampleRate', 48000) or 48000)

        if self._worker_in_buf.shape[0] != block_frame:
            self._worker_in_buf = np.zeros(block_frame, dtype=np.float32)

        self.in_ring = Float32RingBuffer(max(block_frame * 8, 32768))
        self.out_ring = Float32RingBuffer(max(block_frame * 8, 32768))
        self.mon_ring = Float32RingBuffer(max(block_frame * 12, int(self._mon_sr * 0.5))) if has_monitor else None

        self._out_underrun = True
        self._out_last_sample = 0.0
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

    def run_no_monitor(self, block_frame: int, inputMaxChannel: int, outputMaxChannel: int, inputExtraSetting, outputExtraSetting):
        self._start_worker_and_rings(block_frame, has_monitor=False)
        self.stream = sd.Stream(
            callback=self.audio_stream_callback,
            latency='low',
            dtype="float32",
            device=(self.settings.serverInputDeviceId, self.settings.serverOutputDeviceId),
            blocksize=block_frame,
            samplerate=self.settings.serverInputAudioSampleRate,
            channels=(inputMaxChannel, outputMaxChannel),
            extra_settings=(inputExtraSetting, outputExtraSetting)
        )
        self.stream.start()

    def run_with_monitor(self, block_frame: int, inputMaxChannel: int, outputMaxChannel: int, monitorMaxChannel: int, inputExtraSetting, outputExtraSetting, monitorExtraSetting):
        self._start_worker_and_rings(block_frame, has_monitor=True)
        self.stream = sd.Stream(
            callback=self.audio_stream_callback,
            latency='low',
            dtype="float32",
            device=(self.settings.serverInputDeviceId, self.settings.serverOutputDeviceId),
            blocksize=block_frame,
            samplerate=self.settings.serverInputAudioSampleRate,
            channels=(inputMaxChannel, outputMaxChannel),
            extra_settings=(inputExtraSetting, outputExtraSetting)
        )
        self.monitor = sd.OutputStream(
            callback=self.audio_monitor_callback,
            latency='low',
            dtype="float32",
            device=self.settings.serverMonitorDeviceId,
            blocksize=block_frame,
            samplerate=self.settings.serverMonitorAudioSampleRate,
            channels=monitorMaxChannel,
            extra_settings=monitorExtraSetting
        )
        self.stream.start()
        self.monitor.start()

    def stop(self):
        self.running = False
        self._stop_event.set()
        self._in_event.set()

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
        self.stop()

        sd._terminate()
        sd._initialize()

        serverInputAudioDevice = self.getServerInputAudioDevice(self.settings.serverInputDeviceId)
        serverOutputAudioDevice = self.getServerOutputAudioDevice(self.settings.serverOutputDeviceId)
        serverMonitorAudioDevice = self.getServerOutputAudioDevice(self.settings.serverMonitorDeviceId)

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

        self.settings.serverInputAudioSampleRate = self.settings.serverAudioSampleRate
        self.settings.serverOutputAudioSampleRate = self.settings.serverAudioSampleRate
        self.settings.serverMonitorAudioSampleRate = self.settings.serverAudioSampleRate

        if "WASAPI" not in serverInputAudioDevice.hostAPI and not wasapiExclusiveMode:
            inputAudioSampleRateAvailable = checkSamplingRate(self.settings.serverInputDeviceId, self.settings.serverInputAudioSampleRate, "input")
            outputAudioSampleRateAvailable = checkSamplingRate(self.settings.serverOutputDeviceId, self.settings.serverOutputAudioSampleRate, "output")
            monitorAudioSampleRateAvailable = checkSamplingRate(self.settings.serverMonitorDeviceId, self.settings.serverMonitorAudioSampleRate, "output") if serverMonitorAudioDevice else True

            logger.info("Sample Rate:")
            logger.info(f"  [Input]: {self.settings.serverInputAudioSampleRate} -> {inputAudioSampleRateAvailable}")
            logger.info(f"  [Output]: {self.settings.serverOutputAudioSampleRate} -> {outputAudioSampleRateAvailable}")
            if serverMonitorAudioDevice is not None:
                logger.info(f"  [Monitor]: {self.settings.serverMonitorAudioSampleRate} -> {monitorAudioSampleRateAvailable}")

            if not inputAudioSampleRateAvailable or not outputAudioSampleRateAvailable or not monitorAudioSampleRateAvailable:
                logger.info("Checking Available Sample Rate:")
                availableInputSampleRate = []
                availableOutputSampleRate = []
                availableMonitorSampleRate = []
                for sr in SERVER_DEVICE_SAMPLE_RATES:
                    if checkSamplingRate(self.settings.serverInputDeviceId, sr, "input"):
                        availableInputSampleRate.append(sr)
                    if checkSamplingRate(self.settings.serverOutputDeviceId, sr, "output"):
                        availableOutputSampleRate.append(sr)
                    if serverMonitorAudioDevice is not None:
                        if checkSamplingRate(self.settings.serverMonitorDeviceId, sr, "output"):
                            availableMonitorSampleRate.append(sr)
                err = ERR_SAMPLE_RATE_NOT_SUPPORTED % (availableInputSampleRate, availableOutputSampleRate, availableMonitorSampleRate)
                self.callbacks.emit_to(
                    0,
                    self.performance,
                    ('ERR_SAMPLE_RATE_NOT_SUPPORTED', err)
                )
                logger.error(err)
                return

        block_frame = int((self.settings.serverReadChunkSize * 128 / 48000) * self.settings.serverInputAudioSampleRate)

        try:
            if serverMonitorAudioDevice is None:
                self.run_no_monitor(block_frame, inputChannels, outputChannels, inputExtraSetting, outputExtraSetting)
            else:
                self.run_with_monitor(block_frame, inputChannels, outputChannels, serverMonitorAudioDevice.maxOutputChannels, inputExtraSetting, outputExtraSetting, monitorExtraSetting)
            self.running = True
        except Exception as e:
            self.stop()
            self.callbacks.emit_to(0, self.performance, ('ERR_GENERIC_SERVER_AUDIO_ERROR', ERR_GENERIC_SERVER_AUDIO_ERROR))
            logger.exception(e)

    ###########################################
    # Info Section
    ###########################################
    def get_info(self):
        data = {}
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
        if key == 'serverAudioStated':
            if val:
                self.start()
            else:
                self.stop()
        if self.running and key in {'serverInputDeviceId', 'serverOutputDeviceId', 'serverMonitorDeviceId', 'serverReadChunkSize', 'serverAudioSampleRate', 'asioInputChannel', 'asioOutputChannel'}:
            self.start()
