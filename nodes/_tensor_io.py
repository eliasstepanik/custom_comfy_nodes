"""torch <-> numpy helpers shared by the tilemap nodes.

The leading underscore keeps this module out of the pack's auto-discovery
(see the root ``__init__.py``); it defines no node classes.

ComfyUI conventions:
  * IMAGE: float32 tensor [B, H, W, C] in 0..1 (C = 3, rarely 4), CPU.
  * MASK:  float32 tensor [B, H, W] (or [H, W]) in 0..1.

Rules for every helper here: inputs are never mutated, NaN is treated as 0,
+inf as 1 and -inf as 0, values are clipped to 0..1 and rounded half-up
(``floor(x * 255 + 0.5)``). Returned numpy arrays are C-contiguous uint8.
The reverse direction divides by 255, so uint8 -> tensor -> uint8 is exact
for all 256 values.
"""

from typing import List, Sequence, Union

import numpy as np
import torch

_ImageInput = Union[torch.Tensor, Sequence[torch.Tensor]]


def _to_uint8(x: np.ndarray) -> np.ndarray:
    """Float array in nominal 0..1 -> contiguous uint8 (nan/inf-safe, rounds half-up)."""
    x = np.nan_to_num(x.astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=0.0)
    return np.ascontiguousarray((np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))


def _tensor_to_np(t: torch.Tensor, what: str) -> np.ndarray:
    """Detached CPU float32 numpy copy of a floating-point tensor (input untouched)."""
    if not isinstance(t, torch.Tensor):
        raise ValueError(f"{what} must be a torch.Tensor, got {type(t).__name__}")
    if not t.is_floating_point():
        raise ValueError(f"{what} must be a floating-point tensor, got dtype {t.dtype}")
    return t.detach().to(device="cpu", dtype=torch.float32).numpy()


def image_list_to_np(
    t: _ImageInput, channels: int = 3, first_only: bool = False
) -> List[np.ndarray]:
    """Convert IMAGE tensor(s) to a list of uint8 ``HxWxchannels`` arrays.

    Args:
        t: one IMAGE tensor [B, H, W, C] or a list/tuple of them (as delivered
            to a node with ``INPUT_IS_LIST = True``).
        channels: 3 (RGB) or 4 (RGBA). A 4-channel input is accepted for
            ``channels=3`` (alpha dropped); a 3-channel input is rejected for
            ``channels=4``.
        first_only: use only batch element 0 of each tensor.

    Returns:
        One array per image (all batch elements in order, or element 0 only),
        each C-contiguous uint8.

    Raises:
        ValueError: on a non-tensor / non-float input, ndim != 4, an empty
            batch or spatial size, or an incompatible channel count.
    """
    if channels not in (3, 4):
        raise ValueError(f"channels must be 3 or 4, got {channels}")
    tensors = [t] if isinstance(t, torch.Tensor) else list(t)
    if not tensors:
        raise ValueError("image list is empty")
    out: List[np.ndarray] = []
    for i, tensor in enumerate(tensors):
        arr = _tensor_to_np(tensor, f"image[{i}]")
        if arr.ndim != 4:
            raise ValueError(f"image[{i}] must have shape [B,H,W,C], got {tuple(arr.shape)}")
        b, h, w, c = arr.shape
        if b < 1 or h < 1 or w < 1:
            raise ValueError(f"image[{i}] is empty: shape {tuple(arr.shape)}")
        if c != channels and not (c == 4 and channels == 3):
            raise ValueError(f"image[{i}] has {c} channels, expected {channels}")
        for j in range(1 if first_only else b):
            out.append(_to_uint8(arr[j, :, :, :channels]))
    return out


def np_to_image(arrs: Union[np.ndarray, Sequence[np.ndarray]]) -> torch.Tensor:
    """Stack uint8 ``HxWxC`` arrays (C = 3 or 4) into a float32 IMAGE [B, H, W, C].

    Args:
        arrs: one ``HxWxC`` / ``BxHxWxC`` uint8 array, or a non-empty sequence of
            ``HxWxC`` uint8 arrays that all share the same shape.

    Returns:
        Contiguous float32 tensor, values ``uint8 / 255``.

    Raises:
        ValueError: on wrong dtype, wrong ndim, channel count not 3/4,
            mismatched shapes or an empty input.
    """
    if isinstance(arrs, np.ndarray):
        if arrs.ndim not in (3, 4):
            raise ValueError(f"array must have shape [H,W,C] or [B,H,W,C], got {arrs.shape}")
        items = [arrs] if arrs.ndim == 3 else list(arrs)
    else:
        items = list(arrs)
    if not items:
        raise ValueError("no arrays given")
    first = items[0]
    for i, a in enumerate(items):
        if not isinstance(a, np.ndarray) or a.dtype != np.uint8:
            raise ValueError(f"array[{i}] must be a uint8 numpy array")
        if a.ndim != 3 or a.shape[2] not in (3, 4):
            raise ValueError(f"array[{i}] must have shape [H,W,3|4], got {a.shape}")
        if a.shape != first.shape:
            raise ValueError(f"array[{i}] shape {a.shape} differs from array[0] shape {first.shape}")
        if a.shape[0] < 1 or a.shape[1] < 1:
            raise ValueError(f"array[{i}] is empty: shape {a.shape}")
    stacked = np.stack(items, axis=0).astype(np.float32) / np.float32(255.0)
    return torch.from_numpy(np.ascontiguousarray(stacked))


def alpha_to_mask(a: np.ndarray) -> torch.Tensor:
    """uint8 alpha ``HxW`` (0..255) -> float32 MASK [1, H, W] = ``a / 255`` (1 = opaque).

    Raises:
        ValueError: if ``a`` is not a 2-D uint8 numpy array or is empty.
    """
    if not isinstance(a, np.ndarray) or a.dtype != np.uint8:
        raise ValueError("alpha must be a uint8 numpy array")
    if a.ndim != 2 or a.shape[0] < 1 or a.shape[1] < 1:
        raise ValueError(f"alpha must have non-empty shape [H,W], got {a.shape}")
    out = a.astype(np.float32) / np.float32(255.0)
    return torch.from_numpy(np.ascontiguousarray(out)).unsqueeze(0)


def mask_to_alpha(m: torch.Tensor) -> np.ndarray:
    """MASK [H, W] or [B, H, W] (0..1) -> contiguous uint8 ``HxW`` (batch element 0).

    Raises:
        ValueError: on a non-float / non-tensor input, ndim not 2 or 3, or empty.
    """
    arr = _tensor_to_np(m, "mask")
    if arr.ndim == 3:
        arr = arr[0] if arr.shape[0] >= 1 else arr
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError(f"mask must have non-empty shape [H,W] or [B,H,W], got {tuple(m.shape)}")
    return _to_uint8(arr)
