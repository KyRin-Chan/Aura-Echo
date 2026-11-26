from enum import Enum
import functools
import os
import subprocess
import sys

# Prevent Intel OpenMP / LLVM OpenMP duplicate runtime warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import tempfile
from typing import Literal, TypeAlias
import numpy as np


VoiceChangerType: TypeAlias = Literal[
    "RVC",
]

ROOT_PATH = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.realpath(sys.argv[0]))

# Files
LOG_FILE = os.path.join(ROOT_PATH, 'vcclient.log')
DOTENV_FILE = os.path.join(ROOT_PATH, '.env')
STORED_SETTING_FILE = os.path.join(ROOT_PATH, 'stored_setting.json')
ASSETS_FILE = os.path.join(ROOT_PATH, 'assets.json')
PTH_MERGED_FILENAME = "merged.pth"
EDITION_FILE = os.path.join(sys._MEIPASS, "edition.txt") if hasattr(sys, "_MEIPASS") else 'edition.txt'
VERSION_FILE = os.path.join(sys._MEIPASS, "version.txt") if hasattr(sys, "_MEIPASS") else 'version.txt'

# Directories
tmpdir = tempfile.TemporaryDirectory()
SSL_KEY_DIR = os.path.join(ROOT_PATH, "keys")
UPLOAD_DIR = os.path.join(tmpdir.name, "upload_dir") if hasattr(sys, "_MEIPASS") else "upload_dir"
TMP_DIR = os.path.join(tmpdir.name, "tmp_dir") if hasattr(sys, "_MEIPASS") else "tmp_dir"
FRONTEND_DIR = os.path.join(sys._MEIPASS, "dist") if hasattr(sys, "_MEIPASS") else "../client/modern-gui/dist"
FORMANT_PROFILES_DIR = os.path.join(ROOT_PATH, "formant_profiles")
TENSORRT_LIBS_DIR = os.path.join(ROOT_PATH, "tensorrt_libs")

# Voice Changer
EmbedderType: TypeAlias = Literal["hubert_base", "contentvec", "spin_base"]
SERVER_DEVICE_SAMPLE_RATES = [16000, 32000, 44100, 48000, 96000, 192000]
HUBERT_SAMPLE_RATE = 16000
WINDOW_SIZE = HUBERT_SAMPLE_RATE // 100


class EnumInferenceTypes(Enum):
    pyTorchRVC = "pyTorchRVC"
    pyTorchRVCNono = "pyTorchRVCNono"
    pyTorchRVCv2 = "pyTorchRVCv2"
    pyTorchRVCv2Nono = "pyTorchRVCv2Nono"
    pyTorchWebUI = "pyTorchWebUI"
    pyTorchWebUINono = "pyTorchWebUINono"
    onnxRVC = "onnxRVC"
    onnxRVCNono = "onnxRVCNono"

# F0 Extractor
F0_MIN = 50
F0_MAX = 1100
TRACKER_F0_MAX = 1600
F0_MEL_MIN = 1127 * np.log(1 + F0_MIN / 700)
F0_MEL_MAX = 1127 * np.log(1 + F0_MAX / 700)

PitchExtractorType: TypeAlias = Literal[
    "crepe_full",
    "crepe_tiny",
    "crepe_full_onnx",
    "crepe_tiny_onnx",
    "rmvpe",
    "rmvpe_onnx",
    "fcpe",
    "fcpe_onnx",
    "rmvpe_fcpe",
]

ServerAudioDeviceType: TypeAlias = Literal["audioinput", "audiooutput"]

def get_edition():
    if not os.path.exists(EDITION_FILE):
        return 'Nvidia-CUDA'
    with open(EDITION_FILE, 'r', encoding='utf-8') as f:
        return f.read().strip() or 'Nvidia-CUDA'


@functools.lru_cache(maxsize=1)
def _resolve_version() -> str:
    if os.path.exists(VERSION_FILE):
        with open(VERSION_FILE, 'r', encoding='utf-8') as f:
            ver = f.read().strip()
            if ver:
                return ver

    env_ver = (os.environ.get('APP_VERSION') or os.environ.get('BUILD_NAME') or '').strip()
    if env_ver:
        return env_ver

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    version_path = os.path.join(repo_root, 'VERSION')
    base = '1.0.0'
    if os.path.exists(version_path):
        with open(version_path, 'r', encoding='utf-8') as f:
            base = f.read().strip() or base

    count = ''
    short_hash = ''
    try:
        count = subprocess.check_output(
            ['git', 'rev-list', '--count', 'HEAD'],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        ).strip()
    except Exception:
        pass
    try:
        short_hash = subprocess.check_output(
            ['git', 'rev-parse', '--short=7', 'HEAD'],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        ).strip()
    except Exception:
        pass

    if count and short_hash:
        return f"v{base}-b{count}-{short_hash}"
    if short_hash:
        return f"v{base}-{short_hash}"
    return f"v{base}"


def get_version():
    return _resolve_version()

MAX_SLOT_NUM = 500
