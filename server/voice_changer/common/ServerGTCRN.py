import logging
import math
import os
import urllib.request
from typing import Optional

import numpy as np
from xxhash import xxh128

from const import ROOT_PATH
from voice_changer.common.ServerAEC import NumpyCausalPolyphaseResampler

logger = logging.getLogger(__name__)

try:
    from downloader.PretrainList import denoisers as _PRETRAIN_DENOISERS

    _GTCRN_INFO = _PRETRAIN_DENOISERS.get("gtcrn", {})
except Exception:
    _GTCRN_INFO = {}

_FALLBACK_GTCRN_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speech-enhancement-models/gtcrn_simple.onnx"
)
_FALLBACK_GTCRN_HASH = "e3dc178af6575c64e9de070ce0888fe8"

GTCRN_URL: str = _GTCRN_INFO.get("url", _FALLBACK_GTCRN_URL)
GTCRN_HASH: str = _GTCRN_INFO.get("hash", _FALLBACK_GTCRN_HASH)

try:
    import onnxruntime as _ort

    _HAS_ORT = True
except Exception as _ort_err:  # pragma: no cover
    _ort = None
    _HAS_ORT = False
    logger.warning(f"[ServerGTCRN] onnxruntime unavailable: {_ort_err}")


def _resolve_gtcrn_model_path(explicit_path: Optional[str] = None) -> str:
    if explicit_path:
        return explicit_path
    root_candidate = os.path.join(ROOT_PATH, "pretrain", "denoiser", "gtcrn_simple.onnx")
    if os.path.exists(root_candidate):
        return root_candidate
    repo_candidate = os.path.normpath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "pretrain",
            "denoiser",
            "gtcrn_simple.onnx",
        )
    )
    if os.path.exists(repo_candidate):
        return repo_candidate
    return root_candidate


def _verify_file_xxh128(path: str, expected_hash: str) -> bool:
    if not expected_hash:
        return True
    h = xxh128()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest() == expected_hash


def _ensure_gtcrn_model_file(path: str, auto_download: bool = True) -> bool:
    if os.path.exists(path):
        return True
    if not auto_download:
        return False

    tmp_path = f"{path}.tmp"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        logger.info(f"[ServerGTCRN] Downloading gtcrn_simple.onnx from {GTCRN_URL}...")
        req = urllib.request.Request(GTCRN_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp, open(tmp_path, "wb") as out_f:
            while chunk := resp.read(65536):
                out_f.write(chunk)
        if GTCRN_HASH and not _verify_file_xxh128(tmp_path, GTCRN_HASH):
            raise ValueError("GTCRN ONNX hash verification failed")
        os.replace(tmp_path, path)
        logger.info(f"[ServerGTCRN] Downloaded gtcrn_simple.onnx to {path}")
        return True
    except Exception as e:
        logger.warning(f"[ServerGTCRN] Failed to download gtcrn_simple.onnx: {e}")
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return False


class ServerGTCRN:
    """
    Real-time CPU streaming neural noise suppressor for Server Audio mode
    powered by GTCRN (`gtcrn_simple.onnx`) via ONNX Runtime.

    Runs at 16 kHz internally (512-pt STFT, 256-pt hop, periodic sqrt-Hann
    overlap-add) with causal polyphase resampling for arbitrary host rates
    such as 44.1 kHz / 48 kHz / 96 kHz.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        model_path: Optional[str] = None,
        auto_download: bool = True,
    ):
        self.sample_rate = max(8000, int(sample_rate))
        self._proc_sr = 16000
        self._n_fft = 512
        self._hop = 256
        self.model_path = _resolve_gtcrn_model_path(model_path)
        self.available = False
        self._session = None

        hann = np.hanning(self._n_fft + 1)[:-1].astype(np.float32)
        self._win = np.sqrt(hann).astype(np.float32)

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

        self._init_out_pad = int(
            math.ceil(self._hop * self.sample_rate / float(self._proc_sr))
        ) + (64 if self.sample_rate != self._proc_sr else 0)

        self._mix_name = "mix"
        self._enh_name = "enh"
        self._cache_specs: list[tuple[str, tuple[int, ...], str]] = []
        self._caches: dict[str, np.ndarray] = {}
        self._output_names: list[str] = []
        self._mix_buf = np.zeros((1, self._n_fft // 2 + 1, 1, 2), dtype=np.float32)

        if _HAS_ORT and _ensure_gtcrn_model_file(self.model_path, auto_download=auto_download):
            try:
                opts = _ort.SessionOptions()
                opts.intra_op_num_threads = 1
                opts.inter_op_num_threads = 1
                opts.graph_optimization_level = _ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self._session = _ort.InferenceSession(
                    self.model_path, sess_options=opts, providers=["CPUExecutionProvider"]
                )
                inputs = self._session.get_inputs()
                outputs = self._session.get_outputs()
                self._output_names = [o.name for o in outputs]
                self._mix_name = inputs[0].name
                self._enh_name = outputs[0].name

                out_name_set = set(self._output_names)
                for idx, inp in enumerate(inputs[1:], start=1):
                    shape = tuple(int(d) if isinstance(d, int) and d > 0 else 1 for d in inp.shape)
                    out_cand = f"{inp.name}_out"
                    if out_cand not in out_name_set and idx < len(self._output_names):
                        out_cand = self._output_names[idx]
                    self._cache_specs.append((inp.name, shape, out_cand))

                self.available = True
                logger.info(
                    f"[ServerGTCRN] Initialized CPU streaming denoiser (sr={self.sample_rate}, model={self.model_path})"
                )
            except Exception as e:
                logger.warning(f"[ServerGTCRN] Failed to initialize ONNX session: {e}")
                self._session = None
                self.available = False

        self.reset()

    def reset(self) -> None:
        if self._resampler_in is not None:
            self._resampler_in.reset()
        if self._resampler_out is not None:
            self._resampler_out.reset()
        self._in_fifo = np.empty(0, dtype=np.float32)
        self._prev_in_hop = np.zeros(self._hop, dtype=np.float32)
        self._ola_buf = np.zeros(self._hop, dtype=np.float32)
        self._out_fifo = np.zeros(self._init_out_pad, dtype=np.float32)
        self._caches = {
            cname: np.zeros(cshape, dtype=np.float32)
            for cname, cshape, _ in self._cache_specs
        }

    def _run_hop(self, curr_hop: np.ndarray) -> np.ndarray:
        frame_512 = np.concatenate((self._prev_in_hop, curr_hop))
        spec = np.fft.rfft(frame_512 * self._win, n=self._n_fft)
        self._mix_buf[0, :, 0, 0] = spec.real
        self._mix_buf[0, :, 0, 1] = spec.imag

        feeds = {self._mix_name: self._mix_buf}
        feeds.update(self._caches)
        outputs = self._session.run(self._output_names, feeds)
        out_map = dict(zip(self._output_names, outputs))

        enh = out_map[self._enh_name]
        for cname, _, out_cname in self._cache_specs:
            if out_cname in out_map:
                self._caches[cname] = out_map[out_cname]

        enh_c = enh[0, :, 0, 0] + 1j * enh[0, :, 0, 1]
        time_512 = np.fft.irfft(enh_c, n=self._n_fft).astype(np.float32, copy=False) * self._win
        out_hop = self._ola_buf + time_512[: self._hop]
        self._ola_buf[:] = time_512[self._hop :]
        self._prev_in_hop[:] = curr_hop
        return out_hop

    def process(self, mic_chunk: np.ndarray) -> np.ndarray:
        mic = np.ascontiguousarray(mic_chunk, dtype=np.float32).reshape(-1)
        total = int(mic.shape[0])
        if total == 0 or not self.available or self._session is None:
            return mic

        sig_16k = self._resampler_in.process(mic) if self._resampler_in is not None else mic
        if sig_16k.size > 0:
            self._in_fifo = np.concatenate((self._in_fifo, sig_16k))

        hop = self._hop
        n_hops = int(self._in_fifo.shape[0]) // hop
        if n_hops > 0:
            produced = [
                self._run_hop(np.ascontiguousarray(self._in_fifo[i * hop : (i + 1) * hop], dtype=np.float32))
                for i in range(n_hops)
            ]
            self._in_fifo = self._in_fifo[n_hops * hop :]
            block_16k = np.concatenate(produced)
            block_out = (
                self._resampler_out.process(block_16k)
                if self._resampler_out is not None
                else block_16k
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
