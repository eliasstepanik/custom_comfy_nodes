"""Wang corner transition tiles between two seamless ground tiles.

Pure numpy, deterministic, no I/O, no global state. ``wang_tiles`` builds the
16 corner-mask tiles that blend ground type A into ground type B at the exact
``n x n`` tile size. Every output pixel is copied from ``a_rgba`` or ``b_rgba``
(never interpolated), so the result is palette-exact and fully opaque.

Corner order is ``(ne, se, sw, nw)``; a bit of 1 means the corner belongs to
terrain B, 0 to terrain A. The tile with corners ``(ne, se, sw, nw)`` has mask
``ne | se << 1 | sw << 2 | nw << 3`` and the result list is sorted by mask, so
mask 0 is pure A and mask 15 is pure B.

The decision field of a tile is the bilinear blend of the four corner bits plus
a small periodic noise term. Noise is a sum of three sinusoids with integer
frequencies (one period per tile) and seed-derived phases, so the field is a
continuous function that repeats every tile: two neighbouring tiles that agree
on the corner bits of their shared edge have the same field value on that edge.
"""

import math
from typing import List, Tuple

import numpy as np

from . import pixelize

__all__ = ["wang_tiles"]

_NOISE_AMP = 0.18  # < 0.5, so uniform-corner tiles stay pure A / pure B
_NOISE_FREQS = ((1, 2), (2, -1), (3, 1))  # integer (fx, fy): period n in both axes
_SEED_MAX = 2**64 - 1


def _int(value: object, name: str, lo: int, hi: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}")
    if not lo <= int(value) <= hi:
        raise ValueError(f"{name} must be in {lo}..{hi}, got {int(value)}")
    return int(value)


def _size(n: object) -> int:
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise ValueError(f"n must be an integer, got {type(n).__name__}")
    if int(n) not in pixelize.TILE_SIZES:
        raise ValueError(f"n must be one of {pixelize.TILE_SIZES}, got {int(n)}")
    return int(n)


def _ground(a: object, name: str, n: int) -> np.ndarray:
    if not isinstance(a, np.ndarray):
        raise ValueError(f"{name} must be a numpy array, got {type(a).__name__}")
    if a.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8, got {a.dtype}")
    if a.shape != (n, n, 4):
        raise ValueError(f"{name} must have shape ({n}, {n}, 4), got {a.shape}")
    if not (a[..., 3] == 255).all():
        raise ValueError(f"{name} must be fully opaque (alpha 255 everywhere)")
    return a


def _phases(seed: int) -> List[float]:
    rng = np.random.default_rng(seed)
    return [float(p) for p in rng.uniform(0.0, 2.0 * math.pi, len(_NOISE_FREQS))]


def _field(corners: Tuple[int, int, int, int], u: np.ndarray, v: np.ndarray,
           phases: List[float]) -> np.ndarray:
    """Decision field at tile coordinates ``u, v`` in [0, 1] (v grows southwards)."""
    ne, se, sw, nw = corners
    bilinear = (nw * (1 - u) * (1 - v) + ne * u * (1 - v)
                + sw * (1 - u) * v + se * u * v)
    noise = sum(np.cos(2.0 * math.pi * (fx * u + fy * v) + ph)
                for (fx, fy), ph in zip(_NOISE_FREQS, phases)) / len(_NOISE_FREQS)
    return bilinear + _NOISE_AMP * noise


def wang_tiles(a_rgba: np.ndarray, b_rgba: np.ndarray, n: int,
               seed: int) -> List[Tuple[Tuple[int, int, int, int], np.ndarray]]:
    """Return the 16 Wang corner tiles blending ground A into ground B.

    ``a_rgba`` / ``b_rgba`` are fully opaque ``uint8 (n, n, 4)`` seamless ground
    tiles (variant 0) sharing one palette; ``n`` must be in
    ``pixelize.TILE_SIZES``; ``seed`` is an int in ``0..2**64-1``. The result is
    ``[(corners, rgba), ...]`` sorted by mask, ``corners = (ne, se, sw, nw)``
    with 1 = terrain B. A pixel takes B where the field is ``>= 0.5``, else A.
    Inputs are not modified; each returned tile is a new ``uint8 (n, n, 4)`` array.
    """
    n = _size(n)
    a = _ground(a_rgba, "a_rgba", n)
    b = _ground(b_rgba, "b_rgba", n)
    seed = _int(seed, "seed", 0, _SEED_MAX)
    phases = _phases(seed)
    centres = (np.arange(n, dtype=np.float64) + 0.5) / n
    u, v = np.meshgrid(centres, centres)  # u: column (x), v: row (y)
    tiles = []
    for mask in range(16):
        corners = (mask & 1, (mask >> 1) & 1, (mask >> 2) & 1, (mask >> 3) & 1)
        take_b = (_field(corners, u, v, phases) >= 0.5)[..., None]
        tiles.append((corners, np.where(take_b, b, a).astype(np.uint8)))
    return tiles
