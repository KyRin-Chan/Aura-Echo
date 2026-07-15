import os
import logging
import threading
import ctypes
from ctypes import wintypes

logger = logging.getLogger(__name__)

_thread_local = threading.local()

def set_high_process_priority():
    """Elevates the current process priority to HIGH on Windows."""
    if os.name != 'nt':
        return
    try:
        # GetCurrentProcess returns a pseudo-handle for the current process
        kernel32 = ctypes.windll.kernel32
        process = kernel32.GetCurrentProcess()
        # HIGH_PRIORITY_CLASS is 0x00000080
        HIGH_PRIORITY_CLASS = 0x00000080
        success = kernel32.SetPriorityClass(process, HIGH_PRIORITY_CLASS)
        if success:
            logger.info("Successfully elevated Windows process priority class to HIGH_PRIORITY_CLASS")
        else:
            logger.warning("Failed to elevate process priority class. Might lack administrator/sufficient privileges.")
    except Exception as e:
        logger.error(f"Error setting process priority class: {e}")

def register_current_thread_mmcss():
    """Registers the calling thread to the Windows MMCSS Pro Audio task class,
    with a robust fallback to SetThreadPriority if MMCSS fails or is disabled."""
    if os.name != 'nt':
        return
    if getattr(_thread_local, 'mmcss_registered', False) or getattr(_thread_local, 'thread_priority_set', False):
        return
    try:
        kernel32 = ctypes.windll.kernel32
        tid = kernel32.GetCurrentThreadId()

        # Try MMCSS first
        try:
            avrt = ctypes.WinDLL("Avrt.dll")
            avrt.AvSetMmThreadCharacteristicsW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)]
            avrt.AvSetMmThreadCharacteristicsW.restype = wintypes.HANDLE
            
            task_index = wintypes.DWORD(0)
            handle = avrt.AvSetMmThreadCharacteristicsW("Pro Audio", ctypes.byref(task_index))
            if handle:
                _thread_local.mmcss_registered = True
                _thread_local.mmcss_handle = handle
                logger.info(f"Registered thread {threading.current_thread().name} (TID {tid}) to Windows MMCSS 'Pro Audio'")
                return
            else:
                err = kernel32.GetLastError()
                logger.warning(f"AvSetMmThreadCharacteristicsW returned null for 'Pro Audio' (GetLastError: {err}). Falling back to SetThreadPriority.")
        except Exception as mmcss_ex:
            logger.warning(f"MMCSS registration failed: {mmcss_ex}. Falling back to SetThreadPriority.")

        # Fallback to SetThreadPriority
        thread_handle = kernel32.GetCurrentThread()
        # THREAD_PRIORITY_TIME_CRITICAL is 15
        THREAD_PRIORITY_TIME_CRITICAL = 15
        success = kernel32.SetThreadPriority(thread_handle, THREAD_PRIORITY_TIME_CRITICAL)
        if success:
            _thread_local.mmcss_registered = False
            _thread_local.thread_priority_set = True
            logger.info(f"Successfully elevated thread {threading.current_thread().name} (TID {tid}) priority to THREAD_PRIORITY_TIME_CRITICAL (15)")
        else:
            err = kernel32.GetLastError()
            logger.warning(f"SetThreadPriority failed with error: {err}")
    except Exception as e:
        logger.warning(f"Failed to optimize thread scheduling priority: {e}")

def revert_current_thread_mmcss():
    """Reverts the calling thread from MMCSS scheduling or custom thread priority."""
    if os.name != 'nt':
        return
    
    # 1. Revert MMCSS if active
    if getattr(_thread_local, 'mmcss_registered', False):
        try:
            avrt = ctypes.WinDLL("Avrt.dll")
            avrt.AvRevertMmThreadCharacteristics.argtypes = [wintypes.HANDLE]
            avrt.AvRevertMmThreadCharacteristics.restype = wintypes.BOOL
            
            success = avrt.AvRevertMmThreadCharacteristics(_thread_local.mmcss_handle)
            if success:
                _thread_local.mmcss_registered = False
                _thread_local.mmcss_handle = None
                logger.info(f"Reverted thread {threading.current_thread().name} from Windows MMCSS")
        except Exception as e:
            logger.warning(f"Failed to revert thread from Windows MMCSS: {e}")

    # 2. Revert SetThreadPriority if active
    if getattr(_thread_local, 'thread_priority_set', False):
        try:
            kernel32 = ctypes.windll.kernel32
            thread_handle = kernel32.GetCurrentThread()
            # THREAD_PRIORITY_NORMAL is 0
            THREAD_PRIORITY_NORMAL = 0
            success = kernel32.SetThreadPriority(thread_handle, THREAD_PRIORITY_NORMAL)
            if success:
                _thread_local.thread_priority_set = False
                logger.info(f"Reverted thread {threading.current_thread().name} priority to THREAD_PRIORITY_NORMAL (0)")
        except Exception as e:
            logger.warning(f"Failed to revert thread priority: {e}")
