# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_all, collect_dynamic_libs, collect_submodules
import sys
import os.path
import site

sys.setrecursionlimit(sys.getrecursionlimit() * 5)

backend = os.environ.get('BACKEND', 'cuda')

with open('edition.txt', 'w') as f:
    f.write('NVIDIA-CUDA-RTX5080-Optimized')

datas = [('../client/modern-gui/dist', './dist'), ('./edition.txt', '.')]

if 'BUILD_NAME' in os.environ:
  with open('version.txt', 'w') as f:
      f.write(os.environ['BUILD_NAME'])
  datas += [('./version.txt', '.')]
datas += collect_data_files('onnxscript', include_py_files=True)

binaries = []

# Collect TensorRT dynamic libraries and data for RTX 5080 / Blackwell optimization
try:
    trt_ret = collect_all('tensorrt')
    datas += trt_ret[0]
    binaries += trt_ret[1]
    hiddenimports_trt = trt_ret[2]
except Exception as e:
    print(f"Warning: Could not collect tensorrt: {e}")
    hiddenimports_trt = []

hiddenimports = ['app'] + hiddenimports_trt
hiddenimports += collect_submodules('scipy') # Fix "ModuleNotFoundError: No module named 'scipy._lib.*'"

tmp_ret = collect_all('onnxruntime') # Fix "ModuleNotFoundError: No module named 'onnxruntime.transformers.*'"
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

a = Analysis(
    ['client.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=['./pyinstaller-hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
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
