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
    """Registers the calling thread to the Windows MMCSS Pro Audio task class."""
    if os.name != 'nt':
        return
    if getattr(_thread_local, 'mmcss_registered', False):
        return
    try:
        avrt = ctypes.WinDLL("Avrt.dll")
        kernel32 = ctypes.windll.kernel32
        
        # Set function signatures
        avrt.AvSetMmThreadCharacteristicsW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)]
        avrt.AvSetMmThreadCharacteristicsW.restype = wintypes.HANDLE
        
        task_index = wintypes.DWORD(0)
        # Register to "Pro Audio" task class
        handle = avrt.AvSetMmThreadCharacteristicsW("Pro Audio", ctypes.byref(task_index))
        if handle:
            _thread_local.mmcss_registered = True
            _thread_local.mmcss_handle = handle
            tid = kernel32.GetCurrentThreadId()
            logger.info(f"Registered thread {threading.current_thread().name} (TID {tid}) to Windows MMCSS 'Pro Audio'")
        else:
            logger.warning("AvSetMmThreadCharacteristicsW returned null handle for MMCSS 'Pro Audio'")
    except Exception as e:
        logger.warning(f"Failed to register thread to Windows MMCSS: {e}")

def revert_current_thread_mmcss():
    """Reverts the calling thread from MMCSS scheduling."""
    if os.name != 'nt':
        return
    if not getattr(_thread_local, 'mmcss_registered', False):
        return
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
