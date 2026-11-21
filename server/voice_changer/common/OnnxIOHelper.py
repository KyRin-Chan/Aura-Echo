from contextlib import contextmanager
import logging
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import onnxruntime
import torch

logger = logging.getLogger(__name__)

_TORCH_TO_NP_DTYPE = {
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.int64: np.int64,
    torch.int32: np.int32,
    torch.int16: np.int16,
    torch.uint8: np.uint8,
    torch.bool: np.bool_,
}


def ort_value_to_torch(out_val: Any, device: torch.device) -> torch.Tensor:
    """Convert an ONNX Runtime OrtValue to a PyTorch tensor on `device`."""
    from torch.utils.dlpack import from_dlpack

    if hasattr(out_val, "to_dlpack"):
        return from_dlpack(out_val.to_dlpack())
    if hasattr(out_val, "_ortvalue") and hasattr(out_val._ortvalue, "to_dlpack"):
        return from_dlpack(out_val._ortvalue.to_dlpack())
    return torch.from_numpy(out_val.numpy()).to(device)


class OnnxCudaIOBufferCache:
    """
    Caches pre-allocated PyTorch CUDA output buffers keyed by (input_signature, output_names)
    so ONNX Runtime IOBinding writes directly into pre-allocated GPU memory without
    per-frame allocation or DLPack/CPU copies.
    """

    def __init__(self, max_entries: int = 8):
        self.max_entries = max_entries
        self._cache: Dict[Tuple[Any, Tuple[str, ...]], Dict[str, Tuple[torch.Tensor, Any]]] = {}

    def clear(self) -> None:
        self._cache.clear()

    def run(
        self,
        session: onnxruntime.InferenceSession,
        binding: onnxruntime.IOBinding,
        output_names: Sequence[str],
        input_sig: Any,
        device: torch.device,
        clone_outputs: bool = True,
    ) -> List[torch.Tensor]:
        out_tuple = tuple(output_names)
        key = (input_sig, out_tuple)
        cached = self._cache.get(key)

        if cached is not None:
            try:
                for name in out_tuple:
                    buf, np_dtype = cached[name]
                    binding.bind_output(
                        name,
                        device_type="cuda",
                        device_id=device.index,
                        element_type=np_dtype,
                        shape=tuple(buf.shape),
                        buffer_ptr=buf.data_ptr(),
                    )
                session.run_with_iobinding(binding)
                if clone_outputs:
                    return [cached[name][0].clone() for name in out_tuple]
                return [cached[name][0] for name in out_tuple]
            except Exception as e:
                logger.debug(f"Pre-allocated IOBinding fallback to dynamic binding: {e}")
                self._cache.pop(key, None)

        # Dynamic bind once to probe output shapes & dtypes
        for name in out_tuple:
            binding.bind_output(name, device_type="cuda", device_id=device.index)

        session.run_with_iobinding(binding)
        ort_outputs = binding.get_outputs()
        tensors = [ort_value_to_torch(v, device) for v in ort_outputs]

        # Populate cache if all outputs have known numpy dtypes and non-empty static shapes
        entry: Dict[str, Tuple[torch.Tensor, Any]] = {}
        can_cache = True
        for name, t in zip(out_tuple, tensors):
            np_dtype = _TORCH_TO_NP_DTYPE.get(t.dtype)
            if np_dtype is None or t.numel() == 0:
                can_cache = False
                break
            buf = t.contiguous().clone()
            entry[name] = (buf, np_dtype)

        if can_cache:
            if len(self._cache) >= self.max_entries:
                oldest_key = next(iter(self._cache))
                self._cache.pop(oldest_key, None)
            self._cache[key] = entry
            if clone_outputs:
                return [entry[name][0].clone() for name in out_tuple]
            return [entry[name][0] for name in out_tuple]

        return tensors


def create_ort_session_with_stream(
    model_or_path: Union[str, Any],
    sess_options: onnxruntime.SessionOptions,
    device_manager: Any,
    cuda_stream: Optional[torch.cuda.Stream] = None,
) -> Tuple[onnxruntime.InferenceSession, bool]:
    """
    Create an ONNX Runtime InferenceSession, attaching `cuda_stream` to CUDAExecutionProvider
    when supported and safely falling back if `user_compute_stream` is unsupported.

    Returns:
        (session, bound_to_user_stream)
    """
    model_data = model_or_path if isinstance(model_or_path, str) else model_or_path.SerializeToString()

    providers, provider_options = device_manager.get_onnx_execution_provider(cuda_stream=cuda_stream)
    has_user_stream = any(
        isinstance(opt, dict) and "user_compute_stream" in opt for opt in provider_options
    )

    try:
        sess = onnxruntime.InferenceSession(
            model_data,
            sess_options=sess_options,
            providers=providers,
            provider_options=provider_options,
        )
        active_providers = sess.get_providers()
        bound = (
            cuda_stream is not None
            and has_user_stream
            and len(active_providers) > 0
            and active_providers[0] == "CUDAExecutionProvider"
        )
        logger.info(
            f"Initialized ORT InferenceSession (active_providers={active_providers}, custom_stream_bound={bound})"
        )
        return sess, bound
    except Exception as e:
        if has_user_stream:
            logger.warning(
                f"ORT failed with user_compute_stream ({e}); retrying without custom stream binding."
            )
            providers, provider_options = device_manager.get_onnx_execution_provider(cuda_stream=None)
            sess = onnxruntime.InferenceSession(
                model_data,
                sess_options=sess_options,
                providers=providers,
                provider_options=provider_options,
            )
            logger.info(
                f"Initialized ORT InferenceSession after retry (active_providers={sess.get_providers()}, custom_stream_bound=False)"
            )
            return sess, False
        raise


@contextmanager
def ort_cuda_context(
    device: torch.device,
    bound_to_user_stream: bool,
    cuda_stream: Optional[torch.cuda.Stream] = None,
) -> Iterator[None]:
    """
    Context manager guaranteeing stream ordering between PyTorch kernels and ONNX Runtime IOBinding.
    When ORT is bound to `cuda_stream`, executes PyTorch pre/post ops (e.g. buffer clone) on the same
    stream so no CPU blocking is needed; otherwise synchronizes the current PyTorch stream first.
    """
    if device.type != "cuda":
        yield
        return

    if bound_to_user_stream and cuda_stream is not None:
        cur = torch.cuda.current_stream(device)
        if cur.cuda_stream != cuda_stream.cuda_stream:
            cuda_stream.wait_stream(cur)
            try:
                with torch.cuda.stream(cuda_stream):
                    yield
            finally:
                cur.wait_stream(cuda_stream)
        else:
            yield
    else:
        torch.cuda.current_stream(device).synchronize()
        yield
