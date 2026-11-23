# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_all, collect_dynamic_libs, collect_submodules
import sys
import os.path
import site
import subprocess

sys.setrecursionlimit(sys.getrecursionlimit() * 5)

backend = os.environ.get('BACKEND', 'cuda')

with open('edition.txt', 'w', encoding='utf-8') as f:
    f.write('Nvidia-CUDA')

def _resolve_spec_version():
    env_ver = (os.environ.get('APP_VERSION') or os.environ.get('BUILD_NAME') or '').strip()
    if env_ver:
        return env_ver
    repo_root = os.path.abspath('..')
    version_path = os.path.join(repo_root, 'VERSION')
    base = '1.0.0'
    if os.path.exists(version_path):
        with open(version_path, 'r', encoding='utf-8') as f:
            base = f.read().strip() or base
    try:
        count = subprocess.check_output(
            ['git', 'rev-list', '--count', 'HEAD'],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        ).strip()
        short_hash = subprocess.check_output(
            ['git', 'rev-parse', '--short=7', 'HEAD'],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        ).strip()
        if count and short_hash:
            return f"v{base}-b{count}-{short_hash}"
        if short_hash:
            return f"v{base}-{short_hash}"
    except Exception:
        pass
    return f"v{base}"

with open('version.txt', 'w', encoding='utf-8') as f:
    f.write(_resolve_spec_version())

datas = [('../client/modern-gui/dist', './dist'), ('./edition.txt', '.'), ('./version.txt', '.')]

binaries = []
hiddenimports = ['app', 'torchgen']

# Collect onnxscript data files (TensorRT runtime DLLs are downloaded on demand into tensorrt_libs/)
datas += collect_data_files('onnxscript', include_py_files=True)

hiddenimports += collect_submodules('scipy') # Fix "ModuleNotFoundError: No module named 'scipy._lib.*'"

try:
    tmp_ret = collect_all('onnxruntime') # Fix "ModuleNotFoundError: No module named 'onnxruntime.transformers.*'"
    datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
except Exception as e:
    print(f"Warning: Could not collect onnxruntime: {e}")

try:
    webrtc_ret = collect_all('pywebrtc_audio')
    datas += webrtc_ret[0]
    binaries += webrtc_ret[1]
    hiddenimports += webrtc_ret[2]
    for sp in site.getsitepackages():
        libs_dir = os.path.join(sp, 'pywebrtc_audio.libs')
        if os.path.isdir(libs_dir):
            for fn in os.listdir(libs_dir):
                if fn.lower().endswith('.dll'):
                    binaries.append((os.path.join(libs_dir, fn), 'pywebrtc_audio.libs'))
            break
except Exception as e:
    print(f"Warning: Could not collect pywebrtc_audio: {e}")

try:
    df_ret = collect_all('df')
    datas += df_ret[0]
    binaries += df_ret[1]
    hiddenimports += df_ret[2]
    hiddenimports.append('libdf')
except Exception as e:
    print(f"Warning: Could not collect df: {e}")

excludes_list = [
    'torch.utils.tensorboard',
    'tkinter', 'tcl', 'tk',
    'matplotlib', 'IPython', 'ipykernel', 'notebook', 'jinja2',
    'tensorrt', 'tensorrt_libs', 'tensorrt_cu12', 'tensorrt_cu12_libs', 'tensorrt_cu12_bindings',
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

        # Exclude bundled TensorRT runtime DLLs (downloaded on demand into tensorrt_libs/)
        if (
            'tensorrt_libs' in src_lower
            or ('nvinfer' in src_lower and 'onnxruntime_providers_tensorrt' not in src_lower)
            or 'nvonnxparser' in src_lower
        ):
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
    name='AuraEcho',
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
    name='AuraEcho',
)
