from typing import Any, Callable, Optional, Tuple

import torch


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


class StaticCUDAGraphRunner:
    """
    Compatibility pass-through runner (CUDA Graph capture disabled to prevent
    PyTorch RNG and CUDACachingAllocator corruption with dynamic TorchScript models).
    """

    def __init__(self, fn: Callable[..., Any], warmup_steps: int = 2, name: str = "CUDAGraph"):
        self.fn = fn
        self.warmup_steps = max(1, warmup_steps)
        self.name = name

    def reset(self) -> None:
        pass

    @torch.no_grad()
    def run(
        self,
        tensor_args: Tuple[torch.Tensor, ...],
        scalar_args: Tuple[Any, ...] = (),
    ) -> Any:
        return self.fn(*tensor_args, *scalar_args)
