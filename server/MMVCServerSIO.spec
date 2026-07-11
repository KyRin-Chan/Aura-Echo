# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_all, collect_dynamic_libs, collect_submodules
import sys
import os.path
import site

sys.setrecursionlimit(sys.getrecursionlimit() * 5)

backend = os.environ.get('BACKEND', 'cuda')

with open('edition.txt', 'w') as f:
    f.write('Nvidia-CUDA')

datas = [('../client/modern-gui/dist', './dist'), ('./edition.txt', '.')]

if 'BUILD_NAME' in os.environ:
  with open('version.txt', 'w') as f:
      f.write(os.environ['BUILD_NAME'])
  datas += [('./version.txt', '.')]

binaries = []
hiddenimports = ['app', 'torchgen']

# Collect TensorRT dynamic libraries and data for RTX 5080 / Blackwell optimization
datas += collect_data_files('onnxscript', include_py_files=True)
try:
    trt_ret = collect_all('tensorrt')
    datas += trt_ret[0]
    binaries += trt_ret[1]
    hiddenimports += trt_ret[2]
except Exception as e:
    print(f"Warning: Could not collect tensorrt: {e}")

hiddenimports += collect_submodules('scipy') # Fix "ModuleNotFoundError: No module named 'scipy._lib.*'"

try:
    tmp_ret = collect_all('onnxruntime') # Fix "ModuleNotFoundError: No module named 'onnxruntime.transformers.*'"
    datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
except Exception as e:
    print(f"Warning: Could not collect onnxruntime: {e}")

excludes_list = [
    'torch.utils.tensorboard',
    'tkinter', 'tcl', 'tk',
    'matplotlib', 'IPython', 'ipykernel', 'notebook', 'jinja2',
]

a = Analysis(
    ['client.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=['./pyinstaller-hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes_list,
    noarchive=False,
)

# Filter out unnecessary compile-time and unused files
is_trt_build = True
def filter_unwanted(collected_list):
    filtered = []
    for item in collected_list:
        dest, source, *extra = item
        src_lower = source.replace('\\', '/').lower()
        
        # Exclude C/C++ header files and static library files
        if src_lower.endswith('.lib') or src_lower.endswith('.a') or src_lower.endswith('.h') or src_lower.endswith('.hpp'):
            continue
        
        # Exclude test suites, documentation, and legacy Caffe2 code
        if any(x in src_lower for x in ['/test/', '/tests/', 'caffe2', 'torch/include', 'torch/share', 'torch/bin']):
            continue
        
        # Exclude non-audio NVIDIA CUDA libraries (e.g. nvjpeg is not used for voice processing)
        if 'nvjpeg' in src_lower:
            continue
            
        # Exclude NCCL (multi-GPU training only) on Linux
        if 'libnccl' in src_lower:
            continue
            
        # Exclude ONNX Runtime TensorRT provider in non-TRT builds
        if not is_trt_build and 'onnxruntime_providers_tensorrt' in src_lower:
            continue
            
        filtered.append(item)
    return filtered

a.datas = filter_unwanted(a.datas)
a.binaries = filter_unwanted(a.binaries)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MMVCServerSIO',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="./vc_64.ico",
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MMVCServerSIO',
)
