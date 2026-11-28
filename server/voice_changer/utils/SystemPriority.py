import ctypes
from ctypes import wintypes
import logging
import os
import threading

logger = logging.getLogger(__name__)

_thread_local = threading.local()
_dll_lock = threading.Lock()
_process_priority_set = False
_kernel32 = None
_avrt = None
_avrt_load_attempted = False

_HIGH_PRIORITY_CLASS = 0x00000080
_PROCESS_POWER_THROTTLING = 4
_PROCESS_POWER_THROTTLING_CURRENT_VERSION = 1
_PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1
_PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION = 0x4

_AVRT_PRIORITY_HIGH = 1
_AVRT_PRIORITY_CRITICAL = 2
_THREAD_PRIORITY_NORMAL = 0
_THREAD_PRIORITY_HIGHEST = 2
_THREAD_PRIORITY_TIME_CRITICAL = 15
_ERROR_THREAD_ALREADY_IN_TASK = 1552

_PRIORITY_TIERS = {
    "critical": (_AVRT_PRIORITY_CRITICAL, _THREAD_PRIORITY_TIME_CRITICAL),
    "high": (_AVRT_PRIORITY_HIGH, _THREAD_PRIORITY_HIGHEST),
}


class _PROCESS_POWER_THROTTLING_STATE(ctypes.Structure):
    _fields_ = [
        ("Version", wintypes.ULONG),
        ("ControlMask", wintypes.ULONG),
        ("StateMask", wintypes.ULONG),
    ]


def _get_kernel32():
    global _kernel32
    if _kernel32 is not None:
        return _kernel32
    with _dll_lock:
        if _kernel32 is None:
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.GetCurrentProcess.argtypes = []
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            k32.SetPriorityClass.restype = wintypes.BOOL
            k32.SetProcessInformation.argtypes = [
                wintypes.HANDLE,
                ctypes.c_int,
                ctypes.c_void_p,
                wintypes.DWORD,
            ]
            k32.SetProcessInformation.restype = wintypes.BOOL
            k32.GetCurrentThreadId.argtypes = []
            k32.GetCurrentThreadId.restype = wintypes.DWORD
            k32.GetCurrentThread.argtypes = []
            k32.GetCurrentThread.restype = wintypes.HANDLE
            k32.SetThreadPriority.argtypes = [wintypes.HANDLE, ctypes.c_int]
            k32.SetThreadPriority.restype = wintypes.BOOL
            _kernel32 = k32
    return _kernel32


def _get_avrt():
    global _avrt, _avrt_load_attempted
    if _avrt is not None or _avrt_load_attempted:
        return _avrt
    with _dll_lock:
        if not _avrt_load_attempted:
            _avrt_load_attempted = True
            try:
                avrt = ctypes.WinDLL("avrt", use_last_error=True)
                avrt.AvSetMmThreadCharacteristicsW.argtypes = [
                    wintypes.LPCWSTR,
                    ctypes.POINTER(wintypes.DWORD),
                ]
                avrt.AvSetMmThreadCharacteristicsW.restype = wintypes.HANDLE
                avrt.AvSetMmThreadPriority.argtypes = [wintypes.HANDLE, ctypes.c_int]
                avrt.AvSetMmThreadPriority.restype = wintypes.BOOL
                avrt.AvRevertMmThreadCharacteristics.argtypes = [wintypes.HANDLE]
                avrt.AvRevertMmThreadCharacteristics.restype = wintypes.BOOL
                _avrt = avrt
            except Exception as e:
                logger.debug(f"Could not load avrt.dll: {e}")
                _avrt = None
    return _avrt


def _disable_power_throttling(k32, process_handle) -> None:
    """Opt out of Windows 11 EcoQoS / background throttling and timer-resolution throttling."""
    full_mask = (
        _PROCESS_POWER_THROTTLING_EXECUTION_SPEED
        | _PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION
    )
    for mask, label in (
        (full_mask, "EXECUTION_SPEED|IGNORE_TIMER_RESOLUTION"),
        (_PROCESS_POWER_THROTTLING_EXECUTION_SPEED, "EXECUTION_SPEED"),
    ):
        state = _PROCESS_POWER_THROTTLING_STATE(
            Version=_PROCESS_POWER_THROTTLING_CURRENT_VERSION,
            ControlMask=mask,
            StateMask=0,
        )
        if k32.SetProcessInformation(
            process_handle,
            _PROCESS_POWER_THROTTLING,
            ctypes.byref(state),
            ctypes.sizeof(state),
        ):
            logger.info(f"Disabled Windows process power throttling ({label})")
            return
    err = ctypes.get_last_error()
    logger.debug(f"SetProcessInformation(ProcessPowerThrottling) unsupported or failed (GetLastError: {err})")


def set_high_process_priority():
    """Elevates the current process priority to HIGH and disables power/timer throttling on Windows."""
    global _process_priority_set
    if os.name != "nt":
        return
    with _dll_lock:
        if _process_priority_set:
            return
        _process_priority_set = True

    try:
        k32 = _get_kernel32()
        process = k32.GetCurrentProcess()
        if k32.SetPriorityClass(process, _HIGH_PRIORITY_CLASS):
            logger.info("Successfully elevated Windows process priority class to HIGH_PRIORITY_CLASS")
        else:
            err = ctypes.get_last_error()
            logger.warning(
                f"Failed to elevate process priority class (GetLastError: {err}). "
                "Might lack administrator/sufficient privileges."
            )

        _disable_power_throttling(k32, process)
    except Exception as e:
        logger.error(f"Error setting process priority class: {e}")


def register_current_thread_mmcss(priority: str = "high"):
    """Register the calling thread to Windows MMCSS 'Pro Audio' with tiered priority.

    Args:
        priority: 'critical' for real-time audio I/O capture/loopback threads,
            'high' for inference and feature extraction worker threads.
    """
    if os.name != "nt":
        return
    if getattr(_thread_local, "priority_optimized", False):
        return

    _thread_local.priority_optimized = True
    _thread_local.mmcss_registered = False
    _thread_local.thread_priority_set = False
    _thread_local.mmcss_handle = None

    avrt_priority, win32_priority = _PRIORITY_TIERS.get(
        priority, _PRIORITY_TIERS["high"]
    )

    try:
        k32 = _get_kernel32()
        tid = k32.GetCurrentThreadId()
        thread_name = threading.current_thread().name

        avrt = _get_avrt()
        if avrt is not None:
            try:
                task_index = wintypes.DWORD(0)
                handle = avrt.AvSetMmThreadCharacteristicsW(
                    "Pro Audio", ctypes.byref(task_index)
                )
                if handle:
                    if not avrt.AvSetMmThreadPriority(handle, avrt_priority):
                        logger.debug(
                            f"AvSetMmThreadPriority({avrt_priority}) returned False "
                            f"(GetLastError: {ctypes.get_last_error()})"
                        )
                    _thread_local.mmcss_registered = True
                    _thread_local.mmcss_handle = handle
                    logger.info(
                        f"Registered thread {thread_name} (TID {tid}) to Windows MMCSS 'Pro Audio' "
                        f"(tier={priority})"
                    )
                    return

                err = ctypes.get_last_error()
                if err == _ERROR_THREAD_ALREADY_IN_TASK:
                    _thread_local.mmcss_registered = True
                    _thread_local.mmcss_handle = None
                    logger.info(
                        f"Thread {thread_name} (TID {tid}) is already registered to MMCSS "
                        "(ERROR_THREAD_ALREADY_IN_TASK)"
                    )
                    return
                logger.warning(
                    f"AvSetMmThreadCharacteristicsW returned null for 'Pro Audio' "
                    f"(GetLastError: {err}). Falling back to SetThreadPriority."
                )
            except Exception as mmcss_ex:
                logger.warning(f"MMCSS registration failed: {mmcss_ex}. Falling back to SetThreadPriority.")

        thread_handle = k32.GetCurrentThread()
        if k32.SetThreadPriority(thread_handle, win32_priority):
            _thread_local.thread_priority_set = True
            logger.info(
                f"Elevated thread {thread_name} (TID {tid}) priority to {win32_priority} "
                f"(tier={priority})"
            )
        else:
            err = ctypes.get_last_error()
            logger.warning(f"SetThreadPriority({win32_priority}) failed with error: {err}")
    except Exception as e:
        logger.warning(f"Failed to optimize thread scheduling priority: {e}")


def revert_current_thread_mmcss():
    """Revert the calling thread from MMCSS scheduling or custom Win32 thread priority."""
    if os.name != "nt":
        return

    thread_name = threading.current_thread().name

    if getattr(_thread_local, "mmcss_registered", False):
        try:
            avrt = _get_avrt()
            handle = getattr(_thread_local, "mmcss_handle", None)
            if avrt is not None and handle:
                if avrt.AvRevertMmThreadCharacteristics(handle):
                    logger.info(f"Reverted thread {thread_name} from Windows MMCSS")
        except Exception as e:
            logger.warning(f"Failed to revert thread from Windows MMCSS: {e}")
        finally:
            _thread_local.mmcss_registered = False
            _thread_local.mmcss_handle = None

    if getattr(_thread_local, "thread_priority_set", False):
        try:
            k32 = _get_kernel32()
            thread_handle = k32.GetCurrentThread()
            if k32.SetThreadPriority(thread_handle, _THREAD_PRIORITY_NORMAL):
                logger.info(f"Reverted thread {thread_name} priority to THREAD_PRIORITY_NORMAL (0)")
        except Exception as e:
            logger.warning(f"Failed to revert thread priority: {e}")
        finally:
            _thread_local.thread_priority_set = False

    _thread_local.priority_optimized = False
