import os
import site
import sys
import torch
import onnxruntime
import re
import threading
from typing import TypedDict, Literal, Optional
from enum import IntFlag
from const import TMP_DIR, TENSORRT_LIBS_DIR

try:
    import torch_directml
except ImportError:
    import voice_changer.common.deviceManager.DummyDML as torch_directml

import logging
logger = logging.getLogger(__name__)

class CoreMLFlag(IntFlag):
    USE_CPU_ONLY = 0x001
    ENABLE_ON_SUBGRAPH = 0x002
    ONLY_ENABLE_DEVICE_WITH_ANE = 0x004
    ONLY_ALLOW_STATIC_INPUT_SHAPES = 0x008
    CREATE_MLPROGRAM = 0x010

class DevicePresentation(TypedDict):
    id: int
    name: str
    memory: int
    backend: Literal['cpu', 'cuda', 'directml', 'mps']


class DeviceManager(object):
    _instance = None

    @classmethod
    def get_instance(cls):
        # TODO: Dictionary of device manager and client sessions (?)
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.device = torch.device('cpu')
        self.cuda_enabled = torch.cuda.is_available()
        self.mps_enabled: bool = (
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        )
        self.dml_enabled: bool = torch_directml.is_available()
        self.fp16_available = False
        self.force_fp32 = False
        self.disable_jit = False
        self.use_tensorrt = False
        self.lock = threading.RLock()
        self.pitch_stream: Optional[torch.cuda.Stream] = None
        self.embedder_stream: Optional[torch.cuda.Stream] = None
        self.infer_stream: Optional[torch.cuda.Stream] = None
        self._trt_dll_handles = []
        self._registered_dll_dirs = set()
        logger.info('Initialized DeviceManager. Backend statuses:')
        logger.info(f'* DirectML: {self.dml_enabled}, device count: {torch_directml.device_count()}')
        logger.info(f'* CUDA: {self.cuda_enabled}, device count: {torch.cuda.device_count()}')
        logger.info(f'* MPS: {self.mps_enabled}')

    def initialize(self, device_id: int, force_fp32: bool, disable_jit: bool, use_tensorrt: bool = False):
        self.force_fp32 = force_fp32
        self.disable_jit = disable_jit
        self.use_tensorrt = bool(use_tensorrt)
        self.set_device(device_id)

    def set_device(self, id: int):
        if self.mps_enabled:
            torch.mps.empty_cache()
        elif self.cuda_enabled:
            torch.cuda.empty_cache()
        # Try to resolve requested device; fall back to CPU when none available
        try:
            device, metadata = self._get_device(id)
        except Exception:
            logger.warning(f"Requested device id {id} not available. Falling back to CPU.")
            device, metadata = (torch.device("cpu"), {"id": -1, "name": "CPU", 'backend': 'cpu'})
        self.device = device
        self.device_metadata = metadata
        self.fp16_available = self.is_fp16_available()
        logger.info(f'Switched to {metadata["name"]} ({device}). FP16 support: {self.fp16_available}')

        # Enable TF32 and create high-priority persistent CUDA streams on CUDA devices
        self.pitch_stream = None
        self.embedder_stream = None
        self.infer_stream = None
        try:
            if self.device.type == 'cuda':
                if hasattr(torch.backends, 'cudnn'):
                    torch.backends.cudnn.benchmark = True
                major, _ = torch.cuda.get_device_capability(self.device)
                if major >= 8:
                    # Allow TensorFloat-32 for float32 matmul and cuDNN conv
                    torch.backends.cuda.matmul.allow_tf32 = True
                    if hasattr(torch.backends, 'cudnn'):
                        torch.backends.cudnn.allow_tf32 = True
                    # Use highest matmul precision (lets PyTorch pick TF32 where applicable)
                    if hasattr(torch, 'set_float32_matmul_precision'):
                        torch.set_float32_matmul_precision('high')
                    logger.info('Enabled TF32 (TensorFloat-32) and cuDNN benchmark for CUDA matmul/conv on this GPU.')
                else:
                    logger.info('Enabled cuDNN benchmark for CUDA conv on this GPU.')

                _, greatest_p = torch.cuda.Stream.priority_range()
                self.pitch_stream = torch.cuda.Stream(device=self.device, priority=greatest_p)
                self.embedder_stream = torch.cuda.Stream(device=self.device, priority=greatest_p)
                self.infer_stream = torch.cuda.Stream(device=self.device, priority=greatest_p)
                logger.info(f'Initialized persistent high-priority CUDA streams (priority={greatest_p}).')

                onnx_providers, _ = self.get_onnx_execution_provider()
                logger.info(f'Selected ONNX Runtime execution providers: {onnx_providers}')
        except Exception as e:
            logger.debug(f'Unable to configure CUDA TF32/streams: {e}')

    def use_fp16(self):
        return self.fp16_available and not self.force_fp32

    def use_jit_compile(self):
        # FIXME: DirectML backend seems to have issues with JIT. Disable it for now.
        return self.device_metadata['backend'] != 'directml' and not self.disable_jit

    # TODO: This function should also accept backend type
    def _get_device(self, dev_id: int) -> tuple[torch.device, DevicePresentation]:
        if dev_id == -1:
            if self.mps_enabled:
                return (torch.device('mps'), { "id": -1, "name": "MPS", 'backend': 'mps' })
            else:
                return (torch.device("cpu"), { "id": -1, "name": "CPU", 'backend': 'cpu' })

        if self.cuda_enabled:
            name = torch.cuda.get_device_name(dev_id)
            memory = torch.cuda.get_device_properties(dev_id).total_memory
            return (torch.device("cuda", index=dev_id), {"id": dev_id, "name": f"{dev_id}: {name} (CUDA)", "memory": memory, 'backend': 'cuda'})
        elif self.dml_enabled:
            name = torch_directml.device_name(dev_id)
            return (torch.device(torch_directml.device(dev_id)), {"id": dev_id, "name": f"{dev_id}: {name} (DirectML)", "memory": 0, 'backend': 'directml'})
        # If no GPU/accelerator backend is available, default to CPU
        return (torch.device("cpu"), {"id": -1, "name": "CPU", 'backend': 'cpu'})

    @staticmethod
    def list_devices() -> list[DevicePresentation]:
        devCount = torch.cuda.device_count()
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            devices = [{ "id": -1, "name": "MPS", 'backend': 'mps' }]
        else:
            devices = [{ "id": -1, "name": "CPU", 'backend': 'cpu' }]
        for id in range(devCount):
            name = torch.cuda.get_device_name(id)
            memory = torch.cuda.get_device_properties(id).total_memory
            device = {"id": id, "name": f"{id}: {name} (CUDA)", "memory": memory, 'backend': 'cuda'}
            devices.append(device)
        devCount = torch_directml.device_count()
        for id in range(devCount):
            name = torch_directml.device_name(id)
            device = {"id": id, "name": f"{id}: {name} (DirectML)", "memory": 0, 'backend': 'directml'}
            devices.append(device)
        return devices

    def _find_tensorrt_lib_dirs(self) -> list[str]:
        candidates = [TENSORRT_LIBS_DIR]
        try:
            candidates.extend(site.getsitepackages())
        except Exception:
            pass
        try:
            user_site = site.getusersitepackages()
            if user_site:
                candidates.append(user_site)
        except Exception:
            pass
        candidates.extend([
            os.path.join(sys.prefix, "Lib", "site-packages"),
            getattr(sys, "_MEIPASS", None),
            os.path.join(os.path.dirname(sys.executable), "_internal"),
            os.path.dirname(sys.executable),
        ])

        trt_dirs = []
        seen = set()
        managed_norm = os.path.normcase(os.path.abspath(TENSORRT_LIBS_DIR))
        for base in candidates:
            if not base:
                continue
            base_norm = os.path.normcase(os.path.abspath(base))
            paths = (base,) if base_norm == managed_norm else (os.path.join(base, "tensorrt_libs"), base)
            for path in paths:
                norm = os.path.normcase(os.path.abspath(path))
                if norm in seen or not os.path.isdir(path):
                    continue
                if os.path.exists(os.path.join(path, ".pending_delete")):
                    continue
                seen.add(norm)
                trt_dirs.append(path)
        return trt_dirs

    @staticmethod
    def _dir_has_nvinfer_dll(directory: str) -> bool:
        try:
            if os.path.exists(os.path.join(directory, ".pending_delete")):
                return False
            entries = {
                file.lower()
                for file in os.listdir(directory)
                if os.path.isfile(os.path.join(directory, file))
            }
            managed_norm = os.path.normcase(os.path.abspath(TENSORRT_LIBS_DIR))
            dir_norm = os.path.normcase(os.path.abspath(directory))
            if dir_norm == managed_norm:
                required = ("nvinfer_10.dll", "nvinfer_plugin_10.dll", "nvonnxparser_10.dll")
                return all(dll in entries for dll in required)
            return any(name.startswith("nvinfer") and name.endswith(".dll") for name in entries)
        except Exception:
            return False

    def is_tensorrt_available(self) -> bool:
        return self._is_tensorrt_available()

    def refresh_tensorrt_libs(self) -> bool:
        self._register_tensorrt_libs()
        return self._is_tensorrt_available()

    def is_tensorrt_supported(self) -> bool:
        return (
            sys.platform == "win32"
            and self.cuda_enabled
            and ("TensorrtExecutionProvider" in onnxruntime.get_available_providers())
        )

    def _register_tensorrt_libs(self) -> None:
        current_path = os.environ.get("PATH", "")
        path_entries = {
            os.path.normcase(os.path.abspath(p))
            for p in current_path.split(os.pathsep)
            if p
        }
        for directory in self._find_tensorrt_lib_dirs():
            if not self._dir_has_nvinfer_dll(directory):
                continue
            norm = os.path.normcase(os.path.abspath(directory))
            if norm not in path_entries:
                os.environ["PATH"] = directory + os.pathsep + os.environ.get("PATH", "")
                path_entries.add(norm)
            if hasattr(os, "add_dll_directory") and norm not in self._registered_dll_dirs:
                try:
                    self._trt_dll_handles.append(os.add_dll_directory(directory))
                    self._registered_dll_dirs.add(norm)
                except Exception as e:
                    logger.debug(f"Failed to register TensorRT DLL directory {directory}: {e}")

    def _is_tensorrt_available(self):
        self._register_tensorrt_libs()
        search_dirs = os.environ.get("PATH", "").split(os.pathsep)
        search_dirs.extend(self._find_tensorrt_lib_dirs())
        search_dirs.extend([
            os.path.join(os.environ.get("SystemRoot", "C:\\Windows"), "System32"),
            os.path.join(os.environ.get("SystemRoot", "C:\\Windows"), "SysWOW64")
        ])
        for directory in search_dirs:
            if not directory or not os.path.isdir(directory):
                continue
            if self._dir_has_nvinfer_dll(directory):
                return True
        return False

    def get_onnx_execution_provider(self, cuda_stream: Optional[torch.cuda.Stream] = None):
        cpu_settings = {
            "intra_op_num_threads": 8,
            "execution_mode": onnxruntime.ExecutionMode.ORT_PARALLEL,
            "inter_op_num_threads": 8,
        }
        availableProviders = onnxruntime.get_available_providers()
        if "TensorrtExecutionProvider" in availableProviders and not self._is_tensorrt_available():
            availableProviders = [p for p in availableProviders if p != "TensorrtExecutionProvider"]

        if self.device.type == 'cuda' and "ROCMExecutionProvider" in availableProviders:
            return ["ROCMExecutionProvider", "CPUExecutionProvider"], [{"device_id": self.device.index}, cpu_settings]
        elif self.device.type == 'cuda':
            providers = []
            options = []
            if self.use_tensorrt and "TensorrtExecutionProvider" in availableProviders:
                trt_cache_dir = os.path.join(TMP_DIR, "trt_cache")
                os.makedirs(trt_cache_dir, exist_ok=True)
                providers.append("TensorrtExecutionProvider")
                options.append({
                    "device_id": self.device.index,
                    "trt_fp16_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": trt_cache_dir,
                    "trt_timing_cache_enable": True,
                    "trt_timing_cache_path": trt_cache_dir,
                    "trt_builder_optimization_level": 1,
                })
            if "CUDAExecutionProvider" in availableProviders:
                providers.append("CUDAExecutionProvider")
                cuda_opts = {
                    "device_id": self.device.index,
                    "cudnn_conv_algo_search": "HEURISTIC",
                    "do_copy_in_default_stream": True,
                }
                if cuda_stream is not None:
                    cuda_opts["user_compute_stream"] = str(cuda_stream.cuda_stream)
                options.append(cuda_opts)
            if len(providers) > 0:
                providers.append("CPUExecutionProvider")
                options.append(cpu_settings)
                return providers, options
        elif self.device.type == 'privateuseone' and "DmlExecutionProvider" in availableProviders:
            return ["DmlExecutionProvider", "CPUExecutionProvider"], [{"device_id": self.device.index}, cpu_settings]
        elif 'CoreMLExecutionProvider' in availableProviders:
            coreml_flags = CoreMLFlag.ONLY_ENABLE_DEVICE_WITH_ANE
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"], [{'coreml_flags': coreml_flags}, cpu_settings]
        else:
            return ["CPUExecutionProvider"], [cpu_settings]

    def set_disable_jit(self, disable_jit: bool):
        if self.mps_enabled:
            torch.mps.empty_cache()
        elif self.cuda_enabled:
            torch.cuda.empty_cache()
        self.disable_jit = disable_jit

    def set_force_fp32(self, force_fp32: bool):
        if self.mps_enabled:
            torch.mps.empty_cache()
        elif self.cuda_enabled:
            torch.cuda.empty_cache()
        self.force_fp32 = force_fp32

    def set_use_tensorrt(self, use_tensorrt: bool):
        if self.mps_enabled:
            torch.mps.empty_cache()
        elif self.cuda_enabled:
            torch.cuda.empty_cache()
        self.use_tensorrt = bool(use_tensorrt)
        if self.device.type == 'cuda':
            onnx_providers, _ = self.get_onnx_execution_provider()
            logger.info(f'Selected ONNX Runtime execution providers: {onnx_providers}')

    def is_int8_avalable(self):
        if self.device.type == 'cpu':
            return True
        # TODO: Need information on INT8 support on GPUs.
        return False

    def is_fp16_available(self):
        # TODO: Maybe need to add bfloat16 support?
        # FP16 is not supported on CPU
        if self.device.type == 'cpu':
            return False

        device_name_uppercase = self.device_metadata['name'].upper()
        # TODO: Need information and filtering for Radeon and Intel GPUs
        # All Radeon GPUs starting from GCN 1 (Radeon HD 7000 series and later) reportedly have 2:1 FP16 performance
        # Intel UHD Graphics 600 and later reportedly have 2:1 FP16 performance
        # All Intel Arc GPUs reportedly have 2:1 FP16 performance or better
        ignored_nvidia_gpu = re.search(r'((GTX|RTX|TESLA|QUADRO) (V100|[789]\d{2}|1[06]\d{2}|P40|TITAN)|MX\d{3}|\d{3}MX)', device_name_uppercase)
        if ignored_nvidia_gpu is not None:
            return False

        # FIXME: Apparently FP16 does not work well for Intel iGPUs in DirectML backend.
        # Causes problems with Intel UHD on 10th Gen Intel CPU.
        ignored_intel_gpu = 'INTEL' in device_name_uppercase and 'ARC' not in device_name_uppercase
        if ignored_intel_gpu:
            return False

        if self.device.type == 'cuda':
            major, _ = torch.cuda.get_device_capability(self.device)
            if major < 7:  # コンピューティング機能が7以上の場合half precisionが使えるとされている（が例外がある？T500とか）
                return False

        return True
