import logging
from typing import Any, Callable, Dict, Optional, Set, Tuple

import torch

logger = logging.getLogger(__name__)


def circular_write(
    new_data: torch.Tensor,
    target: torch.Tensor,
    scratch: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Ensure 1-D tensor with matching device/dtype without unnecessary copies when already aligned
    new_data = torch.as_tensor(new_data, device=target.device, dtype=target.dtype).reshape(-1)
    if target.dim() != 1:
        target = target.reshape(-1)

    # Determine the amount we can actually write
    write_size = min(new_data.shape[0], target.shape[0])
    if write_size == 0:
        return target

    # Use the last part of new_data if it's longer than the buffer
    src = new_data[-write_size:]

    # Shift existing buffer content left by write_size and append new data at the end
    keep_len = target.shape[0] - write_size
    if keep_len > 0:
        if (
            scratch is not None
            and scratch.shape == target.shape
            and scratch.device == target.device
            and scratch.dtype == target.dtype
        ):
            scratch[:keep_len].copy_(target[write_size:])
            target[:keep_len].copy_(scratch[:keep_len])
        else:
            target[:keep_len] = target[write_size:].detach().clone()
    # Write the new samples into the tail
    target[-write_size:] = src
    return target


def _clone_tensor_tree(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    if isinstance(obj, tuple):
        return tuple(_clone_tensor_tree(x) for x in obj)
    if isinstance(obj, list):
        return [_clone_tensor_tree(x) for x in obj]
    return obj


class StaticCUDAGraphRunner:
    """
    Captures and replays a single static-shape CUDA Graph for the latest tensor/scalar signature.
    Automatically warms up before capture and gracefully falls back to eager execution on any error.
    """

    def __init__(self, fn: Callable[..., Any], warmup_steps: int = 2, name: str = "CUDAGraph"):
        self.fn = fn
        self.warmup_steps = max(1, warmup_steps)
        self.name = name
        self._active_key: Optional[Tuple[Any, ...]] = None
        self._warmup_count: int = 0
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._static_inputs: Optional[Tuple[torch.Tensor, ...]] = None
        self._static_outputs: Any = None
        self._disabled_keys: Set[Tuple[Any, ...]] = set()

    def reset(self) -> None:
        self._active_key = None
        self._warmup_count = 0
        self._graph = None
        self._static_inputs = None
        self._static_outputs = None

    @torch.no_grad()
    def run(
        self,
        tensor_args: Tuple[torch.Tensor, ...],
        scalar_args: Tuple[Any, ...] = (),
    ) -> Any:
        if len(tensor_args) == 0 or any(t.device.type != "cuda" for t in tensor_args):
            return self.fn(*tensor_args, *scalar_args)

        key = (
            tuple((tuple(t.shape), str(t.dtype), t.device.index) for t in tensor_args),
            scalar_args,
        )
        if key in self._disabled_keys:
            return self.fn(*tensor_args, *scalar_args)

        if key != self._active_key:
            self.reset()
            self._active_key = key

        # Replay captured graph
        if self._graph is not None and self._static_inputs is not None:
            try:
                for s_in, t in zip(self._static_inputs, tensor_args):
                    s_in.copy_(t)
                self._graph.replay()
                return _clone_tensor_tree(self._static_outputs)
            except Exception as e:
                logger.warning(f"[{self.name}] CUDA graph replay failed ({e}); disabling graph for shape.")
                self.reset()
                self._disabled_keys.add(key)
                return self.fn(*tensor_args, *scalar_args)

        # Warmup steps
        if self._warmup_count < self.warmup_steps:
            self._warmup_count += 1
            return self.fn(*tensor_args, *scalar_args)

        # Capture CUDA graph
        dev = tensor_args[0].device
        try:
            static_inputs = tuple(t.detach().contiguous().clone() for t in tensor_args)
            capture_stream = torch.cuda.Stream(device=dev)
            cur_stream = torch.cuda.current_stream(dev)
            capture_stream.wait_stream(cur_stream)
            with torch.cuda.stream(capture_stream):
                # Warmup once on the capture stream to initialize stream-local allocator/cuDNN state
                _ = self.fn(*static_inputs, *scalar_args)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    static_outputs = self.fn(*static_inputs, *scalar_args)
            cur_stream.wait_stream(capture_stream)

            self._static_inputs = static_inputs
            self._static_outputs = static_outputs
            self._graph = graph

            # Execute captured graph immediately so outputs reflect current tensor_args
            for s_in, t in zip(self._static_inputs, tensor_args):
                s_in.copy_(t)
            self._graph.replay()
            logger.info(f"[{self.name}] Captured CUDA Graph for key {key[0]}")
            return _clone_tensor_tree(self._static_outputs)
        except Exception as e:
            logger.warning(f"[{self.name}] CUDA graph capture failed ({e}); falling back to eager.")
            self.reset()
            self._disabled_keys.add(key)
            return self.fn(*tensor_args, *scalar_args)
