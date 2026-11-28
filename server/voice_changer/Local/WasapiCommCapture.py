import ctypes
from ctypes import wintypes
import logging
import sys
import threading
import time
from typing import Callable, Optional

import numpy as np

from voice_changer.common.ServerAEC import NumpyCausalPolyphaseResampler
from voice_changer.utils.SystemPriority import (
    register_current_thread_mmcss,
    revert_current_thread_mmcss,
)
from voice_changer.Local.WasapiLoopback import (
    AUDCLNT_BUFFERFLAGS_SILENT,
    AUDCLNT_E_DEVICE_INVALIDATED,
    AUDCLNT_SHAREMODE_SHARED,
    CLSID_MMDEVICE_ENUMERATOR,
    CLSCTX_ALL,
    COINIT_MULTITHREADED,
    GUID,
    IID_IAUDIO_CAPTURE_CLIENT,
    IID_IMMDEVICE_ENUMERATOR,
    RPC_E_CHANGED_MODE,
    S_FALSE,
    S_OK,
    _com_method,
    _com_release,
    _decode_mix_format,
    _get_device_id,
    _pcm_bytes_to_mono_float32,
)

logger = logging.getLogger(__name__)

E_CAPTURE = 1
E_CONSOLE = 0
DEVICE_STATE_ACTIVE = 1
STGM_READ = 0

AUDIO_CATEGORY_COMMUNICATIONS = 3
AUDCLNT_STREAMOPTIONS_NONE = 0

IID_IAUDIO_CLIENT2 = GUID.from_str("726778CD-F60A-4EDA-82DE-E47610CD78AA")
IID_IAEC_CONTROL = GUID.from_str("F4AE25B5-AAA3-437D-B6B3-DBBE2D0E9549")
IID_IAUDIO_EFFECTS_MANAGER = GUID.from_str("4460B3AE-4B44-4527-8676-7548A8ACD260")

GUID_EFFECT_AEC = GUID.from_str("6f64adbe-8211-11e2-8c70-2c27d7f001fa")
GUID_EFFECT_NS = GUID.from_str("6f64adbf-8211-11e2-8c70-2c27d7f001fa")
GUID_EFFECT_AGC = GUID.from_str("6f64adc0-8211-11e2-8c70-2c27d7f001fa")
GUID_EFFECT_BEAMFORMING = GUID.from_str("6f64adc1-8211-11e2-8c70-2c27d7f001fa")
GUID_EFFECT_DEEP_NS = GUID.from_str("6f64add0-8211-11e2-8c70-2c27d7f001fa")

_EFFECT_NAMES = {
    "6f64adbe-8211-11e2-8c70-2c27d7f001fa": "AEC",
    "6f64adbf-8211-11e2-8c70-2c27d7f001fa": "NS",
    "6f64adc0-8211-11e2-8c70-2c27d7f001fa": "AGC",
    "6f64adc1-8211-11e2-8c70-2c27d7f001fa": "BEAMFORMING",
    "6f64add0-8211-11e2-8c70-2c27d7f001fa": "DEEP_NS",
}


class PROPERTYKEY(ctypes.Structure):
    _fields_ = [
        ("fmtid", GUID),
        ("pid", wintypes.DWORD),
    ]


class PROPVARIANT(ctypes.Structure):
    _fields_ = [
        ("vt", wintypes.WORD),
        ("wReserved1", wintypes.WORD),
        ("wReserved2", wintypes.WORD),
        ("wReserved3", wintypes.WORD),
        ("pwszVal", ctypes.c_void_p),
        ("pad", ctypes.c_void_p),
    ]


class AudioClientProperties(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.UINT),
        ("bIsOffload", wintypes.BOOL),
        ("eCategory", ctypes.c_int),
        ("Options", ctypes.c_int),
    ]


class AUDIO_EFFECT(ctypes.Structure):
    _fields_ = [
        ("id", GUID),
        ("canSetState", wintypes.BOOL),
        ("state", ctypes.c_int),  # 0=Off, 1=On
    ]


PKEY_Device_FriendlyName = PROPERTYKEY(
    GUID.from_str("A45C254E-DF1C-4EFD-8020-67D146A850E0"), 14
)


def _guid_to_str(g: GUID) -> str:
    b = bytes(g.Data4)
    return (
        f"{g.Data1:08x}-{g.Data2:04x}-{g.Data3:04x}-"
        f"{b[0]:02x}{b[1]:02x}-" + "".join(f"{x:02x}" for x in b[2:])
    )


def _get_device_friendly_name(device_ptr: ctypes.c_void_p, ole32) -> str:
    if not device_ptr or not device_ptr.value:
        return ""
    store = ctypes.c_void_p()
    pv = PROPVARIANT()
    try:
        # IMMDevice::OpenPropertyStore (vtbl index 4)
        hr = _com_method(
            device_ptr, 4, ctypes.c_long, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p)
        )(STGM_READ, ctypes.byref(store))
        if hr < 0 or not store.value:
            return ""
        # IPropertyStore::GetValue (vtbl index 5)
        hr = _com_method(
            store, 5, ctypes.c_long, ctypes.POINTER(PROPERTYKEY), ctypes.POINTER(PROPVARIANT)
        )(ctypes.byref(PKEY_Device_FriendlyName), ctypes.byref(pv))
        if hr >= 0 and pv.vt == 31 and pv.pwszVal:
            return ctypes.wstring_at(pv.pwszVal)
    except Exception:
        pass
    finally:
        try:
            ole32.PropVariantClear(ctypes.byref(pv))
        except Exception:
            pass
        _com_release(store)
    return ""


def _resolve_capture_device(
    enumerator: ctypes.c_void_p,
    ole32,
    target_name: Optional[str],
) -> tuple[ctypes.c_void_p, str]:
    target = (target_name or "").strip()
    if target:
        coll = ctypes.c_void_p()
        candidates: list[tuple[ctypes.c_void_p, str]] = []
        try:
            # IMMDeviceEnumerator::EnumAudioEndpoints (vtbl index 3)
            hr = _com_method(
                enumerator,
                3,
                ctypes.c_long,
                ctypes.c_int,
                wintypes.DWORD,
                ctypes.POINTER(ctypes.c_void_p),
            )(E_CAPTURE, DEVICE_STATE_ACTIVE, ctypes.byref(coll))
            if hr >= 0 and coll.value:
                cnt = wintypes.UINT()
                hr = _com_method(coll, 3, ctypes.c_long, ctypes.POINTER(wintypes.UINT))(
                    ctypes.byref(cnt)
                )
                if hr >= 0:
                    item_fn = _com_method(
                        coll, 4, ctypes.c_long, wintypes.UINT, ctypes.POINTER(ctypes.c_void_p)
                    )
                    for idx in range(int(cnt.value)):
                        dev_i = ctypes.c_void_p()
                        if item_fn(idx, ctypes.byref(dev_i)) >= 0 and dev_i.value:
                            name_i = _get_device_friendly_name(dev_i, ole32)
                            candidates.append((dev_i, name_i))
        finally:
            _com_release(coll)

        chosen_idx: Optional[int] = None
        target_lower = target.lower()
        for idx, (_, name_i) in enumerate(candidates):
            if name_i == target:
                chosen_idx = idx
                break
        if chosen_idx is None:
            for idx, (_, name_i) in enumerate(candidates):
                if name_i.lower() == target_lower:
                    chosen_idx = idx
                    break
        if chosen_idx is None:
            for idx, (_, name_i) in enumerate(candidates):
                nl = name_i.lower()
                if nl and (nl.startswith(target_lower) or target_lower.startswith(nl)):
                    chosen_idx = idx
                    break

        if chosen_idx is not None:
            chosen_dev, chosen_name = candidates[chosen_idx]
            for idx, (dev_i, _) in enumerate(candidates):
                if idx != chosen_idx:
                    _com_release(dev_i)
            return chosen_dev, chosen_name

        for dev_i, _ in candidates:
            _com_release(dev_i)
        logger.warning(
            f"[WasapiComm] Device '{target}' not matched in active WASAPI endpoints; falling back to default microphone."
        )

    default_dev = ctypes.c_void_p()
    hr = _com_method(
        enumerator,
        4,
        ctypes.c_long,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
    )(E_CAPTURE, E_CONSOLE, ctypes.byref(default_dev))
    if hr < 0 or not default_dev.value:
        raise OSError(f"GetDefaultAudioEndpoint(eCapture, eConsole) failed: 0x{hr & 0xFFFFFFFF:08X}")
    return default_dev, _get_device_friendly_name(default_dev, ole32)


class WasapiCommCapture:
    """
    Captures a Windows microphone via WASAPI shared mode tagged as
    `AudioCategory_Communications` so Windows 11 Voice Clarity / hardware
    Communications APO performs clock-locked OS-level Acoustic Echo Cancellation.
    """

    def __init__(
        self,
        device_name: Optional[str],
        sample_rate: int,
        on_audio: Callable[[np.ndarray], None],
        enable_deep_ns: bool = False,
    ):
        self.device_name = device_name
        self.sample_rate = max(8000, int(sample_rate))
        self._on_audio = on_audio
        self._desired_deep_ns = bool(enable_deep_ns)

        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_ok = False
        self._running = False

        self._os_aec_active = False
        self._os_aec_control = False
        self._os_ns_active = False
        self._os_deep_ns_supported = False
        self._active_effects: list[str] = []

        self._resampler: Optional[NumpyCausalPolyphaseResampler] = None
        self._resampler_src_sr: int = 0

    @property
    def is_running(self) -> bool:
        return self._running and self._thread is not None and self._thread.is_alive()

    @property
    def os_aec_active(self) -> bool:
        return self._os_aec_active

    @property
    def os_aec_control(self) -> bool:
        return self._os_aec_control

    @property
    def os_ns_active(self) -> bool:
        return self._os_ns_active

    @property
    def os_deep_ns_supported(self) -> bool:
        return self._os_deep_ns_supported

    @property
    def active_effects(self) -> list[str]:
        return list(self._active_effects)

    def set_deep_noise_suppression(self, enabled: bool) -> None:
        self._desired_deep_ns = bool(enabled)

    def start(self, timeout: float = 2.0) -> bool:
        if sys.platform != "win32":
            logger.warning("[WasapiComm] Windows Communications AEC capture is only supported on Windows.")
            return False

        self.stop()
        self._os_aec_active = False
        self._os_aec_control = False
        self._os_ns_active = False
        self._os_deep_ns_supported = False
        self._active_effects = []
        self._resampler = None
        self._resampler_src_sr = 0

        self._stop_event.clear()
        self._ready_event.clear()
        self._start_ok = False

        self._thread = threading.Thread(
            target=self._capture_loop,
            name="WasapiCommCapture",
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
        if self._resampler is not None:
            self._resampler.reset()

    def _resample_to_target(self, mono: np.ndarray, src_sr: int) -> np.ndarray:
        if mono.size == 0 or src_sr == self.sample_rate:
            return mono
        if self._resampler is None or self._resampler_src_sr != src_sr:
            self._resampler = NumpyCausalPolyphaseResampler(src_sr, self.sample_rate)
            self._resampler_src_sr = src_sr
        return self._resampler.process(mono)

    def _capture_loop(self) -> None:
        try:
            register_current_thread_mmcss("critical")
        except Exception:
            pass
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
        ole32.PropVariantClear.argtypes = [ctypes.POINTER(PROPVARIANT)]
        ole32.PropVariantClear.restype = ctypes.c_long

        hr_co = ole32.CoInitializeEx(None, COINIT_MULTITHREADED)
        need_co_uninit = hr_co in (S_OK, S_FALSE)
        if hr_co not in (S_OK, S_FALSE, RPC_E_CHANGED_MODE):
            logger.error(f"[WasapiComm] CoInitializeEx failed: 0x{hr_co & 0xFFFFFFFF:08X}")
            self._ready_event.set()
            try:
                revert_current_thread_mmcss()
            except Exception:
                pass
            return

        winmm = None
        timer_period_set = False
        try:
            winmm = ctypes.windll.winmm
            if winmm.timeBeginPeriod(1) == 0:
                timer_period_set = True
        except Exception:
            winmm = None

        signaled_ready = False

        try:
            while not self._stop_event.is_set():
                enumerator = ctypes.c_void_p()
                device = ctypes.c_void_p()
                audio_client = ctypes.c_void_p()
                capture_client = ctypes.c_void_p()
                aec_control = ctypes.c_void_p()
                fx_mgr = ctypes.c_void_p()
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

                    device, friendly_name = _resolve_capture_device(enumerator, ole32, self.device_name)
                    current_dev_id = _get_device_id(device, ole32)

                    # IMMDevice::Activate(IID_IAudioClient2) (vtbl index 3)
                    hr = _com_method(
                        device,
                        3,
                        ctypes.c_long,
                        ctypes.POINTER(GUID),
                        wintypes.DWORD,
                        ctypes.c_void_p,
                        ctypes.POINTER(ctypes.c_void_p),
                    )(ctypes.byref(IID_IAUDIO_CLIENT2), CLSCTX_ALL, None, ctypes.byref(audio_client))
                    if hr < 0 or not audio_client.value:
                        raise OSError(f"IMMDevice::Activate(IAudioClient2) failed: 0x{hr & 0xFFFFFFFF:08X}")

                    # IAudioClient2::SetClientProperties (vtbl index 16) -> AudioCategory_Communications
                    props = AudioClientProperties(
                        ctypes.sizeof(AudioClientProperties),
                        False,
                        AUDIO_CATEGORY_COMMUNICATIONS,
                        AUDCLNT_STREAMOPTIONS_NONE,
                    )
                    hr_prop = _com_method(
                        audio_client, 16, ctypes.c_long, ctypes.POINTER(AudioClientProperties)
                    )(ctypes.byref(props))
                    if hr_prop < 0:
                        logger.warning(f"[WasapiComm] SetClientProperties(Communications) failed: 0x{hr_prop & 0xFFFFFFFF:08X}")

                    # IAudioClient::GetMixFormat (vtbl index 8)
                    hr = _com_method(
                        audio_client, 8, ctypes.c_long, ctypes.POINTER(ctypes.c_void_p)
                    )(ctypes.byref(pwfx))
                    if hr < 0 or not pwfx.value:
                        raise OSError(f"IAudioClient::GetMixFormat failed: 0x{hr & 0xFFFFFFFF:08X}")

                    mix_sr, channels, bits, is_float = _decode_mix_format(pwfx.value)
                    hns_buffer_duration = 1_000_000  # 100ms in 100-ns units

                    # IAudioClient::Initialize (vtbl index 3)
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
                    )(AUDCLNT_SHAREMODE_SHARED, 0, hns_buffer_duration, 0, pwfx.value, None)
                    if hr < 0:
                        raise OSError(f"IAudioClient::Initialize(CAPTURE) failed: 0x{hr & 0xFFFFFFFF:08X}")

                    # Query IAcousticEchoCancellationControl (optional; Windows 11 22H2+)
                    aec_ctrl_ok = False
                    hr_aec = _com_method(
                        audio_client,
                        14,
                        ctypes.c_long,
                        ctypes.POINTER(GUID),
                        ctypes.POINTER(ctypes.c_void_p),
                    )(ctypes.byref(IID_IAEC_CONTROL), ctypes.byref(aec_control))
                    if hr_aec >= 0 and aec_control.value:
                        hr_set = _com_method(
                            aec_control, 3, ctypes.c_long, wintypes.LPCWSTR
                        )(None)
                        aec_ctrl_ok = hr_set >= 0
                    self._os_aec_control = aec_ctrl_ok

                    # Query IAudioEffectsManager to inspect active Communications APO effects
                    os_aec = False
                    os_ns = False
                    deep_ns_settable = False
                    fx_names: list[str] = []
                    hr_fx = _com_method(
                        audio_client,
                        14,
                        ctypes.c_long,
                        ctypes.POINTER(GUID),
                        ctypes.POINTER(ctypes.c_void_p),
                    )(ctypes.byref(IID_IAUDIO_EFFECTS_MANAGER), ctypes.byref(fx_mgr))
                    if hr_fx >= 0 and fx_mgr.value:
                        fx_ptr = ctypes.POINTER(AUDIO_EFFECT)()
                        fx_cnt = wintypes.UINT()
                        hr_get_fx = _com_method(
                            fx_mgr,
                            5,
                            ctypes.c_long,
                            ctypes.POINTER(ctypes.POINTER(AUDIO_EFFECT)),
                            ctypes.POINTER(wintypes.UINT),
                        )(ctypes.byref(fx_ptr), ctypes.byref(fx_cnt))
                        if hr_get_fx >= 0 and fx_ptr:
                            try:
                                for idx in range(int(fx_cnt.value)):
                                    eff = fx_ptr[idx]
                                    g_str = _guid_to_str(eff.id)
                                    label = _EFFECT_NAMES.get(g_str, g_str)
                                    if eff.state == 1:
                                        fx_names.append(label)
                                    if eff.id.matches(GUID_EFFECT_AEC) and eff.state == 1:
                                        os_aec = True
                                    if (eff.id.matches(GUID_EFFECT_NS) or eff.id.matches(GUID_EFFECT_DEEP_NS)) and eff.state == 1:
                                        os_ns = True
                                    if eff.id.matches(GUID_EFFECT_DEEP_NS) and bool(eff.canSetState):
                                        deep_ns_settable = True
                            finally:
                                ole32.CoTaskMemFree(fx_ptr)

                    self._os_aec_active = os_aec or aec_ctrl_ok
                    self._os_ns_active = os_ns
                    self._os_deep_ns_supported = deep_ns_settable
                    self._active_effects = fx_names

                    # IAudioClient::GetService(IAudioCaptureClient) (vtbl index 14)
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

                    set_effect_state = None
                    applied_deep_ns = self._desired_deep_ns
                    if deep_ns_settable and fx_mgr.value:
                        set_effect_state = _com_method(
                            fx_mgr, 6, ctypes.c_long, GUID, ctypes.c_int
                        )
                        try:
                            set_effect_state(GUID_EFFECT_DEEP_NS, 1 if applied_deep_ns else 0)
                        except Exception:
                            pass

                    if self._os_aec_active:
                        logger.info(
                            f"[WasapiComm] Capturing '{friendly_name or 'Default Mic'}' "
                            f"({mix_sr} Hz, {channels} ch, {bits}-bit {'float' if is_float else 'int'}) "
                            f"-> {self.sample_rate} Hz mono | os_aec=True, aec_ctrl={aec_ctrl_ok}, "
                            f"effects={fx_names}"
                        )
                    else:
                        logger.warning(
                            f"[WasapiComm] Capturing '{friendly_name or 'Default Mic'}' "
                            f"({mix_sr} Hz, {channels} ch) without Windows AEC APO "
                            f"(device may be virtual or Windows Audio Enhancements are disabled)."
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
                    get_default_endpoint = _com_method(
                        enumerator,
                        4,
                        ctypes.c_long,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.POINTER(ctypes.c_void_p),
                    )

                    packet_frames = wintypes.UINT()
                    data_ptr = ctypes.c_void_p()
                    frames_read = wintypes.UINT()
                    flags = wintypes.DWORD()
                    dev_pos = ctypes.c_uint64()
                    qpc_pos = ctypes.c_uint64()

                    last_dev_check_ts = time.monotonic()
                    device_invalidated = False
                    track_default_dev = not (self.device_name or "").strip()

                    while not self._stop_event.is_set():
                        if set_effect_state is not None:
                            desired = self._desired_deep_ns
                            if desired != applied_deep_ns:
                                applied_deep_ns = desired
                                try:
                                    set_effect_state(GUID_EFFECT_DEEP_NS, 1 if desired else 0)
                                except Exception:
                                    pass

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

                            if mono.size > 0:
                                resampled = self._resample_to_target(mono, mix_sr)
                                if resampled.size > 0:
                                    self._on_audio(resampled)

                        if device_invalidated:
                            logger.info("[WasapiComm] Capture device invalidated, reconnecting...")
                            break

                        now = time.monotonic()
                        if track_default_dev and (now - last_dev_check_ts >= 0.5):
                            last_dev_check_ts = now
                            if current_dev_id is not None:
                                probe_dev = ctypes.c_void_p()
                                try:
                                    hr_p = get_default_endpoint(
                                        E_CAPTURE, E_CONSOLE, ctypes.byref(probe_dev)
                                    )
                                    if hr_p >= 0 and probe_dev.value:
                                        probe_id = _get_device_id(probe_dev, ole32)
                                        if probe_id is not None and probe_id != current_dev_id:
                                            logger.info("[WasapiComm] Default capture device switched, reconnecting...")
                                            device_invalidated = True
                                            break
                                except Exception:
                                    pass
                                finally:
                                    _com_release(probe_dev)

                        time.sleep(0.002)

                    if not device_invalidated:
                        break

                except Exception as e:
                    logger.warning(f"[WasapiComm] Capture error: {e}")
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
                    _com_release(aec_control)
                    _com_release(fx_mgr)
                    _com_release(audio_client)
                    _com_release(device)
                    _com_release(enumerator)

                if self._stop_event.is_set():
                    break
                if self._resampler is not None:
                    self._resampler.reset()
                time.sleep(0.05)
        finally:
            if timer_period_set and winmm is not None:
                try:
                    winmm.timeEndPeriod(1)
                except Exception:
                    pass
            if not signaled_ready:
                self._ready_event.set()
            if need_co_uninit:
                ole32.CoUninitialize()
            try:
                revert_current_thread_mmcss()
            except Exception:
                pass
            self._running = False
