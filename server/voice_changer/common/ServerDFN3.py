import logging
import math
import os
import urllib.request
import warnings
import zipfile
from typing import Optional

import numpy as np
import torch

from const import ROOT_PATH
from voice_changer.common.ServerAEC import NumpyCausalPolyphaseResampler

logger = logging.getLogger(__name__)

DFN3_URL = "https://github.com/Rikorose/DeepFilterNet/raw/main/models/DeepFilterNet3.zip"

try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*AudioMetaData.*", category=UserWarning)
        from df.enhance import enhance as _df_enhance, init_df as _df_init_df

    _HAS_DF = True
except Exception as _df_err:  # pragma: no cover
    _df_enhance = None
    _df_init_df = None
    _HAS_DF = False
    logger.warning(f"[ServerDFN3] deepfilternet unavailable: {_df_err}")


def _resolve_dfn3_model_dir(explicit_dir: Optional[str] = None) -> str:
    if explicit_dir:
        return explicit_dir
    root_candidate = os.path.join(ROOT_PATH, "pretrain", "denoiser", "DeepFilterNet3")
    if os.path.isfile(os.path.join(root_candidate, "config.ini")):
        return root_candidate
    repo_candidate = os.path.normpath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "pretrain",
            "denoiser",
            "DeepFilterNet3",
        )
    )
    if os.path.isfile(os.path.join(repo_candidate, "config.ini")):
        return repo_candidate
    return root_candidate


def _is_dfn3_dir_ready(model_dir: str) -> bool:
    return os.path.isfile(os.path.join(model_dir, "config.ini")) and os.path.isdir(
        os.path.join(model_dir, "checkpoints")
    )


def _ensure_dfn3_model_dir(model_dir: str, auto_download: bool = True) -> bool:
    if _is_dfn3_dir_ready(model_dir):
        return True
    if not auto_download:
        return False

    parent_dir = os.path.dirname(os.path.abspath(model_dir))
    tmp_zip = os.path.join(parent_dir, "DeepFilterNet3.zip.tmp")
    try:
        os.makedirs(parent_dir, exist_ok=True)
        logger.info(f"[ServerDFN3] Downloading DeepFilterNet3 weights from {DFN3_URL}...")
        req = urllib.request.Request(DFN3_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp, open(tmp_zip, "wb") as out_f:
            while chunk := resp.read(65536):
                out_f.write(chunk)
        with zipfile.ZipFile(tmp_zip, "r") as zf:
            zf.extractall(parent_dir)
        os.remove(tmp_zip)
        if not _is_dfn3_dir_ready(model_dir):
            raise RuntimeError(f"Extracted archive did not produce valid model at {model_dir}")
        logger.info(f"[ServerDFN3] Downloaded DeepFilterNet3 to {model_dir}")
        return True
    except Exception as e:
        logger.warning(f"[ServerDFN3] Failed to download DeepFilterNet3: {e}")
        try:
            if os.path.exists(tmp_zip):
                os.remove(tmp_zip)
        except Exception:
            pass
        return False


class ServerDFN3:
    """
    Real-time streaming neural noise suppressor for Server Audio mode
    powered by official DeepFilterNet3 (`df.enhance.init_df` / `enhance`).

    Runs at 48 kHz internally in memory on float32 PyTorch tensors with
    hop-aligned causal context + lookahead framing, and causal polyphase
    resampling for non-48 kHz host sample rates (e.g. 44.1 kHz / 16 kHz).
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        model_dir: Optional[str] = None,
        auto_download: bool = True,
    ):
        self.sample_rate = max(8000, int(sample_rate))
        self._proc_sr = 48000
        self._hop = 480
        self.model_dir = _resolve_dfn3_model_dir(model_dir)
        self.available = False
        self._model = None
        self._df_state = None

        if self.sample_rate != self._proc_sr:
            self._resampler_in: Optional[NumpyCausalPolyphaseResampler] = NumpyCausalPolyphaseResampler(
                self.sample_rate, self._proc_sr
            )
            self._resampler_out: Optional[NumpyCausalPolyphaseResampler] = NumpyCausalPolyphaseResampler(
                self._proc_sr, self.sample_rate
            )
        else:
            self._resampler_in = None
            self._resampler_out = None

        if _HAS_DF and _ensure_dfn3_model_dir(self.model_dir, auto_download=auto_download):
            try:
                self._model, self._df_state, _ = _df_init_df(
                    model_base_dir=self.model_dir,
                    post_filter=False,
                    log_level="ERROR",
                    log_file=None,
                    config_allow_defaults=True,
                )
                self._model.eval()
                self._hop = max(1, int(self._df_state.hop_size()))
                self.available = True
                dev = next(self._model.parameters()).device
                logger.info(
                    f"[ServerDFN3] Initialized streaming denoiser (sr={self.sample_rate}, device={dev}, model={self.model_dir})"
                )
            except Exception as e:
                logger.warning(f"[ServerDFN3] Failed to initialize DeepFilterNet3: {e}")
                self._model = None
                self._df_state = None
                self.available = False

        self._lookahead = 2 * self._hop
        self._ctx_len = 12 * self._hop
        self._init_out_pad = int(
            math.ceil((self._hop + self._lookahead) * self.sample_rate / float(self._proc_sr))
        ) + (64 if self.sample_rate != self._proc_sr else 0)

        self.reset()

    def reset(self) -> None:
        if self._resampler_in is not None:
            self._resampler_in.reset()
        if self._resampler_out is not None:
            self._resampler_out.reset()
        if self._df_state is not None:
            self._df_state.reset()
        self._in_fifo = np.zeros(self._lookahead, dtype=np.float32)
        self._hist = np.zeros(self._ctx_len, dtype=np.float32)
        self._out_fifo = np.zeros(self._init_out_pad, dtype=np.float32)

    def process(self, mic_chunk: np.ndarray) -> np.ndarray:
        mic = np.ascontiguousarray(mic_chunk, dtype=np.float32).reshape(-1)
        total = int(mic.shape[0])
        if total == 0 or not self.available or self._model is None or self._df_state is None:
            return mic

        sig_48k = self._resampler_in.process(mic) if self._resampler_in is not None else mic
        if sig_48k.size > 0:
            self._in_fifo = np.concatenate((self._in_fifo, sig_48k))

        hop = self._hop
        avail_new = int(self._in_fifo.shape[0]) - self._lookahead
        n_hops = avail_new // hop
        if n_hops > 0:
            step = n_hops * hop
            win = np.ascontiguousarray(
                np.concatenate((self._hist, self._in_fifo[: step + self._lookahead])),
                dtype=np.float32,
            )
            audio_t = torch.from_numpy(win).unsqueeze(0)
            self._df_state.reset()
            enhanced_t = _df_enhance(self._model, self._df_state, audio_t)
            enhanced_np = enhanced_t.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
            block_48k = enhanced_np[self._ctx_len : self._ctx_len + step]

            self._hist = np.concatenate((self._hist, self._in_fifo[:step]))[-self._ctx_len :]
            self._in_fifo = self._in_fifo[step:]

            block_out = (
                self._resampler_out.process(block_48k)
                if self._resampler_out is not None
                else block_48k
            )
            if block_out.size > 0:
                self._out_fifo = np.concatenate((self._out_fifo, block_out))

        if self._out_fifo.shape[0] >= total:
            out = self._out_fifo[:total]
            self._out_fifo = self._out_fifo[total:]
            return out

        out = np.zeros(total, dtype=np.float32)
        have = int(self._out_fifo.shape[0])
        if have > 0:
            out[:have] = self._out_fifo
            self._out_fifo = np.empty(0, dtype=np.float32)
        return out
