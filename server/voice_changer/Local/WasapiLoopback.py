import ctypes
from ctypes import wintypes
import logging
import sys
import threading
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Windows COM / WASAPI constants
COINIT_MULTITHREADED = 0x0
RPC_E_CHANGED_MODE = -2147417850  # 0x80010106
S_OK = 0
S_FALSE = 1

CLSCTX_ALL = 23
E_RENDER = 0
E_CONSOLE = 0

AUDCLNT_SHAREMODE_SHARED = 0
AUDCLNT_STREAMFLAGS_LOOPBACK = 0x00020000
AUDCLNT_BUFFERFLAGS_SILENT = 0x2
AUDCLNT_E_DEVICE_INVALIDATED = -2004287484  # 0x88890004

WAVE_FORMAT_PCM = 0x0001
WAVE_FORMAT_IEEE_FLOAT = 0x0003
WAVE_FORMAT_EXTENSIBLE = 0xFFFE


class GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", wintypes.DWORD),
        ("Data2", wintypes.WORD),
        ("Data3", wintypes.WORD),
        ("Data4", ctypes.c_ubyte * 8),
    ]

    @classmethod
    def from_str(cls, guid_str: str) -> "GUID":
        s = guid_str.strip("{}")
        parts = s.split("-")
        d1 = int(parts[0], 16)
        d2 = int(parts[1], 16)
        d3 = int(parts[2], 16)
        d4_hex = parts[3] + parts[4]
        d4 = (ctypes.c_ubyte * 8)(*[int(d4_hex[i : i + 2], 16) for i in range(0, 16, 2)])
        return cls(d1, d2, d3, d4)

    def matches(self, other: "GUID") -> bool:
        return (
            self.Data1 == other.Data1
            and self.Data2 == other.Data2
            and self.Data3 == other.Data3
            and bytes(self.Data4) == bytes(other.Data4)
        )


class WAVEFORMATEX(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("wFormatTag", wintypes.WORD),
        ("nChannels", wintypes.WORD),
        ("nSamplesPerSec", wintypes.DWORD),
        ("nAvgBytesPerSec", wintypes.DWORD),
        ("nBlockAlign", wintypes.WORD),
        ("wBitsPerSample", wintypes.WORD),
        ("cbSize", wintypes.WORD),
    ]


class WAVEFORMATEXTENSIBLE(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("Format", WAVEFORMATEX),
        ("wValidBitsPerSample", wintypes.WORD),
        ("dwChannelMask", wintypes.DWORD),
        ("SubFormat", GUID),
    ]


CLSID_MMDEVICE_ENUMERATOR = GUID.from_str("BCDE0395-E52F-467C-8E3D-C4579291692E")
IID_IMMDEVICE_ENUMERATOR = GUID.from_str("A95664D2-9614-4F35-A746-DE8DB63617E6")
IID_IAUDIO_CLIENT = GUID.from_str("1CB9AD4C-DBFA-4C32-B178-C2F568A703B2")
IID_IAUDIO_CAPTURE_CLIENT = GUID.from_str("C8ADBD64-E71E-48A0-A4DE-185C395CD317")
KSDATAFORMAT_SUBTYPE_PCM = GUID.from_str("00000001-0000-0010-8000-00AA00389B71")
KSDATAFORMAT_SUBTYPE_IEEE_FLOAT = GUID.from_str("00000003-0000-0010-8000-00AA00389B71")


def _com_method(ptr: ctypes.c_void_p, vtbl_index: int, restype, *argtypes):
    vtbl = ctypes.cast(ptr, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    fn_addr = vtbl[vtbl_index]
    proto = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)
    fn = ctypes.cast(ctypes.c_void_p(fn_addr), proto)
    return lambda *args: fn(ptr, *args)


def _com_release(ptr: Optional[ctypes.c_void_p]) -> None:
    if ptr is not None and ptr.value:
        try:
            _com_method(ptr, 2, ctypes.c_ulong)()
        except Exception:
            pass
        ptr.value = None


def _decode_mix_format(pwfx_addr: int) -> tuple[int, int, int, bool]:
    """Return (sample_rate, channels, bits_per_sample, is_float) from WAVEFORMATEX*."""
    wfx = WAVEFORMATEX.from_address(pwfx_addr)
    tag = int(wfx.wFormatTag)
    channels = max(1, int(wfx.nChannels))
    sample_rate = max(1, int(wfx.nSamplesPerSec))
    bits = int(wfx.wBitsPerSample)

    if tag == WAVE_FORMAT_IEEE_FLOAT:
        is_float = True
    elif tag == WAVE_FORMAT_PCM:
        is_float = False
    elif tag == WAVE_FORMAT_EXTENSIBLE and wfx.cbSize >= 22:
        wfx_ext = WAVEFORMATEXTENSIBLE.from_address(pwfx_addr)
        if wfx_ext.SubFormat.matches(KSDATAFORMAT_SUBTYPE_IEEE_FLOAT):
            is_float = True
        elif wfx_ext.SubFormat.matches(KSDATAFORMAT_SUBTYPE_PCM):
            is_float = False
        else:
            raise ValueError(f"Unsupported WAVEFORMATEXTENSIBLE SubFormat")
    else:
        raise ValueError(f"Unsupported WAVEFORMATEX format tag: 0x{tag:04X}")

    return sample_rate, channels, bits, is_float


def _pcm_bytes_to_mono_float32(
    data_ptr: int,
    num_frames: int,
    channels: int,
    bits: int,
    is_float: bool,
) -> np.ndarray:
    if num_frames <= 0 or not data_ptr:
        return np.empty(0, dtype=np.float32)

    bytes_per_sample = max(1, bits // 8)
    total_samples = num_frames * channels
    total_bytes = total_samples * bytes_per_sample
    raw_buf = (ctypes.c_ubyte * total_bytes).from_address(data_ptr)

    if is_float and bits == 32:
        arr = np.frombuffer(raw_buf, dtype=np.float32).copy()
    elif not is_float and bits == 16:
        arr = np.frombuffer(raw_buf, dtype=np.int16).astype(np.float32) / 32768.0
    elif not is_float and bits == 32:
        arr = np.frombuffer(raw_buf, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif not is_float and bits == 24:
        u8 = np.frombuffer(raw_buf, dtype=np.uint8).reshape(total_samples, 3)
        signed = (
            u8[:, 0].astype(np.int32)
            | (u8[:, 1].astype(np.int32) << 8)
            | (u8[:, 2].astype(np.int32) << 16)
        )
        signed = np.where(signed & 0x800000, signed - 0x1000000, signed)
        arr = signed.astype(np.float32) / 8388608.0
    else:
        return np.zeros(num_frames, dtype=np.float32)

    if channels > 1:
        return arr.reshape(num_frames, channels).mean(axis=1, dtype=np.float32)
    return arr.reshape(-1)


class WasapiLoopbackCapture:
    """
    Captures the Windows default playback device (eRender / eConsole) in WASAPI
    shared loopback mode and exposes mono float32 reference audio at `sample_rate`.
    """

    def __init__(self, sample_rate: int = 48000, buffer_seconds: float = 0.5):
        self.sample_rate = max(8000, int(sample_rate))
        self._buffer_cap = max(8192, int(round(self.sample_rate * buffer_seconds)))
        self._ring = None
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_ok = False
        self._running = False
        self._underrun_debt = 0

        # Stateful linear resampler state when native mix rate != target sample_rate
        self._resample_phase = 0.0
        self._resample_prev = 0.0
        self._resample_has_prev = False

    def _ensure_ring(self):
        if self._ring is None:
            from voice_changer.Local.ServerAudio import Float32RingBuffer

            self._ring = Float32RingBuffer(self._buffer_cap)
        return self._ring

    @property
    def is_running(self) -> bool:
        return self._running and self._thread is not None and self._thread.is_alive()

    def start(self, timeout: float = 2.0) -> bool:
        if sys.platform != "win32":
            logger.warning("[WasapiLoopback] Default playback loopback is only supported on Windows.")
            return False

        self.stop()
        ring = self._ensure_ring()
        ring.clear()
        self._underrun_debt = 0
        self._resample_phase = 0.0
        self._resample_prev = 0.0
        self._resample_has_prev = False

        self._stop_event.clear()
        self._ready_event.clear()
        self._start_ok = False

        self._thread = threading.Thread(
            target=self._capture_loop,
            name="WasapiLoopbackCapture",
            daemon=True,
        )
        self._thread.start()
        self._ready_event.wait(timeout=timeout)
        self._running = self._start_ok
        return self._start_ok

    def stop(self) -> None:
        self._stop_event.set()
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        if self._ring is not None:
            self._ring.clear()
        self._underrun_debt = 0

    def _resample_to_target(self, mono: np.ndarray, src_sr: int) -> np.ndarray:
        if mono.size == 0 or src_sr == self.sample_rate:
            return mono
        ratio = float(src_sr) / float(self.sample_rate)
        if not self._resample_has_prev:
            self._resample_prev = float(mono[0])
            self._resample_has_prev = True
            self._resample_phase = 0.0

        ext = np.concatenate(([np.float32(self._resample_prev)], mono))
        n_src = mono.shape[0]
        # Available fractional indices in `ext` go from 0.0 up to n_src
        # Sample positions start at (1.0 + self._resample_phase) - 1.0? Wait:
        # `ext[0]` is previous chunk's last sample (index -1), `ext[1:]` is [0..n_src-1].
        # Phase is relative to `mono[0]` (index 1.0 in `ext`).
        start_pos = self._resample_phase
        if start_pos >= n_src:
            self._resample_phase = start_pos - n_src
            self._resample_prev = float(mono[-1])
            return np.empty(0, dtype=np.float32)

        count = int(np.floor((n_src - 1e-9 - start_pos) / ratio)) + 1
        if count <= 0:
            self._resample_prev = float(mono[-1])
            return np.empty(0, dtype=np.float32)

        pos = start_pos + np.arange(count, dtype=np.float64) * ratio
        ext_pos = pos + 1.0
        idx0 = np.clip(np.floor(ext_pos).astype(np.int64), 0, n_src)
        idx1 = np.minimum(idx0 + 1, n_src)
        frac = (ext_pos - idx0).astype(np.float32)
        out = ext[idx0] * (1.0 - frac) + ext[idx1] * frac

        next_pos = start_pos + count * ratio
        self._resample_phase = next_pos - n_src
        self._resample_prev = float(mono[-1])
        return out.astype(np.float32, copy=False)

    def read_chunk(self, n_samples: int, out_buf: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Read `n_samples` mono float32 reference samples aligned to real time.
        Drops excess stale samples beyond a small cushion and tracks underrun debt
        so timing stays synchronized across clock drift or bursty WASAPI packets.
        """
        n = max(0, int(n_samples))
        if out_buf is None or out_buf.shape[0] != n:
            out_buf = np.zeros(n, dtype=np.float32)
        if n == 0:
            return out_buf

        ring = self._ring
        if ring is None or not self._running:
            out_buf.fill(0.0)
            return out_buf

        # Reconcile any previous underrun debt before reading new samples
        if self._underrun_debt > 0:
            avail = ring.available()
            if avail > 0:
                dropped = ring.drop_oldest(min(avail, self._underrun_debt))
                self._underrun_debt -= dropped

        # Prevent stale backlog buildup from clock drift or startup skew
        target_cushion = min(n, max(256, int(self.sample_rate * 0.015)))
        max_backlog = n + max(target_cushion * 2, int(self.sample_rate * 0.04))
        avail = ring.available()
        if avail > max_backlog:
            ring.drop_oldest(avail - (n + target_cushion))

        got = ring.read_into(out_buf)
        if got < n:
            out_buf[got:n] = 0.0
            missing = n - got
            # Cap debt so prolonged silence doesn't discard audio indefinitely
            self._underrun_debt = min(
                self._underrun_debt + missing,
                max(n * 2, int(self.sample_rate * 0.1)),
            )
        return out_buf

    def _capture_loop(self) -> None:
        ole32 = ctypes.windll.ole32
        ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, wintypes.DWORD]
        ole32.CoInitializeEx.restype = ctypes.c_long
        ole32.CoCreateInstance.argtypes = [
            ctypes.POINTER(GUID),
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(GUID),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        ole32.CoCreateInstance.restype = ctypes.c_long
        ole32.CoTaskMemFree.argtypes = [ctypes.c_void_p]
        ole32.CoTaskMemFree.restype = None

        hr_co = ole32.CoInitializeEx(None, COINIT_MULTITHREADED)
        need_co_uninit = hr_co in (S_OK, S_FALSE)
        if hr_co not in (S_OK, S_FALSE, RPC_E_CHANGED_MODE):
            logger.error(f"[WasapiLoopback] CoInitializeEx failed: 0x{hr_co & 0xFFFFFFFF:08X}")
            self._ready_event.set()
            return

        ring = self._ensure_ring()
        signaled_ready = False

        try:
            while not self._stop_event.is_set():
                enumerator = ctypes.c_void_p()
                device = ctypes.c_void_p()
                audio_client = ctypes.c_void_p()
                capture_client = ctypes.c_void_p()
                pwfx = ctypes.c_void_p()
                started = False

                try:
                    hr = ole32.CoCreateInstance(
                        ctypes.byref(CLSID_MMDEVICE_ENUMERATOR),
                        None,
                        CLSCTX_ALL,
                        ctypes.byref(IID_IMMDEVICE_ENUMERATOR),
                        ctypes.byref(enumerator),
                    )
                    if hr < 0:
                        raise OSError(f"CoCreateInstance(MMDeviceEnumerator) failed: 0x{hr & 0xFFFFFFFF:08X}")

                    # IMMDeviceEnumerator::GetDefaultAudioEndpoint (vtbl index 4)
                    hr = _com_method(
                        enumerator, 4, ctypes.c_long, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)
                    )(E_RENDER, E_CONSOLE, ctypes.byref(device))
                    if hr < 0 or not device.value:
                        raise OSError(f"GetDefaultAudioEndpoint(eRender, eConsole) failed: 0x{hr & 0xFFFFFFFF:08X}")

                    # IMMDevice::Activate (vtbl index 3)
                    hr = _com_method(
                        device,
                        3,
                        ctypes.c_long,
                        ctypes.POINTER(GUID),
                        wintypes.DWORD,
                        ctypes.c_void_p,
                        ctypes.POINTER(ctypes.c_void_p),
                    )(ctypes.byref(IID_IAUDIO_CLIENT), CLSCTX_ALL, None, ctypes.byref(audio_client))
                    if hr < 0 or not audio_client.value:
                        raise OSError(f"IMMDevice::Activate(IAudioClient) failed: 0x{hr & 0xFFFFFFFF:08X}")

                    # IAudioClient::GetMixFormat (vtbl index 8)
                    hr = _com_method(
                        audio_client, 8, ctypes.c_long, ctypes.POINTER(ctypes.c_void_p)
                    )(ctypes.byref(pwfx))
                    if hr < 0 or not pwfx.value:
                        raise OSError(f"IAudioClient::GetMixFormat failed: 0x{hr & 0xFFFFFFFF:08X}")

                    mix_sr, channels, bits, is_float = _decode_mix_format(pwfx.value)

                    # IAudioClient::Initialize (vtbl index 3), 100ms shared loopback buffer
                    hns_buffer_duration = 1_000_000  # 100ms in 100-ns units
                    hr = _com_method(
                        audio_client,
                        3,
                        ctypes.c_long,
                        ctypes.c_int,
                        wintypes.DWORD,
                        ctypes.c_int64,
                        ctypes.c_int64,
                        ctypes.c_void_p,
                        ctypes.c_void_p,
                    )(
                        AUDCLNT_SHAREMODE_SHARED,
                        AUDCLNT_STREAMFLAGS_LOOPBACK,
                        hns_buffer_duration,
                        0,
                        pwfx.value,
                        None,
                    )
                    if hr < 0:
                        raise OSError(f"IAudioClient::Initialize(LOOPBACK) failed: 0x{hr & 0xFFFFFFFF:08X}")

                    # IAudioClient::GetService (vtbl index 14)
                    hr = _com_method(
                        audio_client,
                        14,
                        ctypes.c_long,
                        ctypes.POINTER(GUID),
                        ctypes.POINTER(ctypes.c_void_p),
                    )(ctypes.byref(IID_IAUDIO_CAPTURE_CLIENT), ctypes.byref(capture_client))
                    if hr < 0 or not capture_client.value:
                        raise OSError(f"IAudioClient::GetService(IAudioCaptureClient) failed: 0x{hr & 0xFFFFFFFF:08X}")

                    # IAudioClient::Start (vtbl index 10)
                    hr = _com_method(audio_client, 10, ctypes.c_long)()
                    if hr < 0:
                        raise OSError(f"IAudioClient::Start failed: 0x{hr & 0xFFFFFFFF:08X}")
                    started = True

                    logger.info(
                        f"[WasapiLoopback] Capturing default playback device "
                        f"({mix_sr} Hz, {channels} ch, {bits}-bit {'float' if is_float else 'int'}) "
                        f"-> target {self.sample_rate} Hz mono"
                    )

                    if not signaled_ready:
                        self._start_ok = True
                        signaled_ready = True
                        self._ready_event.set()

                    get_next_packet_size = _com_method(
                        capture_client, 5, ctypes.c_long, ctypes.POINTER(wintypes.UINT)
                    )
                    get_buffer = _com_method(
                        capture_client,
                        3,
                        ctypes.c_long,
                        ctypes.POINTER(ctypes.c_void_p),
                        ctypes.POINTER(wintypes.UINT),
                        ctypes.POINTER(wintypes.DWORD),
                        ctypes.POINTER(ctypes.c_uint64),
                        ctypes.POINTER(ctypes.c_uint64),
                    )
                    release_buffer = _com_method(
                        capture_client, 4, ctypes.c_long, wintypes.UINT
                    )

                    packet_frames = wintypes.UINT()
                    data_ptr = ctypes.c_void_p()
                    frames_read = wintypes.UINT()
                    flags = wintypes.DWORD()
                    dev_pos = ctypes.c_uint64()
                    qpc_pos = ctypes.c_uint64()

                    last_activity_ts = time.monotonic()
                    device_invalidated = False

                    while not self._stop_event.is_set():
                        got_any_packet = False
                        while not self._stop_event.is_set():
                            hr = get_next_packet_size(ctypes.byref(packet_frames))
                            if hr == AUDCLNT_E_DEVICE_INVALIDATED:
                                device_invalidated = True
                                break
                            if hr < 0:
                                raise OSError(f"GetNextPacketSize failed: 0x{hr & 0xFFFFFFFF:08X}")
                            if packet_frames.value == 0:
                                break

                            hr = get_buffer(
                                ctypes.byref(data_ptr),
                                ctypes.byref(frames_read),
                                ctypes.byref(flags),
                                ctypes.byref(dev_pos),
                                ctypes.byref(qpc_pos),
                            )
                            if hr == AUDCLNT_E_DEVICE_INVALIDATED:
                                device_invalidated = True
                                break
                            if hr < 0:
                                raise OSError(f"GetBuffer failed: 0x{hr & 0xFFFFFFFF:08X}")

                            n_frames = int(frames_read.value)
                            buf_flags = int(flags.value)
                            if n_frames > 0:
                                if (buf_flags & AUDCLNT_BUFFERFLAGS_SILENT) != 0:
                                    mono = np.zeros(n_frames, dtype=np.float32)
                                else:
                                    mono = _pcm_bytes_to_mono_float32(
                                        data_ptr.value or 0,
                                        n_frames,
                                        channels,
                                        bits,
                                        is_float,
                                    )
                            else:
                                mono = np.empty(0, dtype=np.float32)

                            release_buffer(frames_read)
                            got_any_packet = True

                            if mono.size > 0:
                                resampled = self._resample_to_target(mono, mix_sr)
                                if resampled.size > 0:
                                    ring.write(resampled)

                        if device_invalidated:
                            logger.info("[WasapiLoopback] Default playback device changed/invalidated, reconnecting...")
                            break

                        now = time.monotonic()
                        if got_any_packet:
                            last_activity_ts = now
                        else:
                            # When playback is idle, WASAPI stops emitting loopback packets.
                            # Emit silent frames to keep the reference timeline aligned.
                            idle_elapsed = now - last_activity_ts
                            if idle_elapsed >= 0.015:
                                silent_n = min(
                                    int(round(idle_elapsed * self.sample_rate)),
                                    self._buffer_cap // 4,
                                )
                                if silent_n > 0:
                                    ring.write(np.zeros(silent_n, dtype=np.float32))
                                last_activity_ts = now

                        time.sleep(0.004)

                    if not device_invalidated:
                        break

                except Exception as e:
                    logger.warning(f"[WasapiLoopback] Capture error: {e}")
                    if not signaled_ready:
                        self._start_ok = False
                        signaled_ready = True
                        self._ready_event.set()
                    break
                finally:
                    if started and audio_client.value:
                        try:
                            _com_method(audio_client, 11, ctypes.c_long)()
                        except Exception:
                            pass
                    if pwfx.value:
                        ole32.CoTaskMemFree(pwfx.value)
                        pwfx.value = None
                    _com_release(capture_client)
                    _com_release(audio_client)
                    _com_release(device)
                    _com_release(enumerator)

                if self._stop_event.is_set():
                    break
                time.sleep(0.1)
        finally:
            if not signaled_ready:
                self._ready_event.set()
            if need_co_uninit:
                ole32.CoUninitialize()
            self._running = False
