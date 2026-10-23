"""Pixel-perfect tile reduction for the tilemap generator.

Pure numpy + PIL, deterministic, no I/O, no global state. Every array is
``uint8 HxWx3`` (RGB "work" images) unless stated otherwise. A tile of
``cells_w x cells_h`` cells and tile size ``n`` is first resampled to the
fixed work grid ``(cells_w*n*8, cells_h*n*8)``; each 8x8 work block later
becomes exactly one output pixel, so output sizes are exact by construction,
whatever size the generator image had.

Alpha grids are boolean arrays of shape ``(work_h // 8, work_w // 8)``;
``True`` means opaque (alpha 255), ``False`` means transparent (alpha 0).
All functions validate their arguments and raise ``ValueError`` with a
user-readable message; inputs are never mutated.
"""

import math
from collections import deque
from typing import Sequence

import numpy as np
from PIL import Image

__all__ = [
    "WORK_SCALE",
    "TILE_SIZES",
    "to_work",
    "make_seamless",
    "key_alpha",
    "build_palette",
    "index_image",
    "block_reduce",
]

WORK_SCALE = 8
TILE_SIZES = (8, 16, 24, 32, 48, 64)
_CELLS_MAX = 16  # sanity cap on footprint cells per side (plans use <= 4)
_PALETTE_SAMPLE_MAX = 262144
_INDEX_CHUNK = 4_000_000  # max (pixels x palette) float32 cells per chunk
_BG_FRAC = 0.25  # block with this share of near-background pixels is background


def _int(value: object, name: str, lo: int, hi: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}")
    if not lo <= int(value) <= hi:
        raise ValueError(f"{name} must be in {lo}..{hi}, got {int(value)}")
    return int(value)


def _number(value: object, name: str, lo: float, hi: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a number, got {type(value).__name__}")
    v = float(value)
    if not math.isfinite(v) or not lo <= v <= hi:
        raise ValueError(f"{name} must be a finite number in {lo}..{hi}, got {value!r}")
    return v


def _rgb(a: object, name: str, block_aligned: bool = False) -> np.ndarray:
    if not isinstance(a, np.ndarray):
        raise ValueError(f"{name} must be a numpy array, got {type(a).__name__}")
    if a.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8, got {a.dtype}")
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"{name} must have shape HxWx3, got {a.shape}")
    if a.shape[0] < 1 or a.shape[1] < 1:
        raise ValueError(f"{name} must not be empty, got {a.shape}")
    if block_aligned and (a.shape[0] % WORK_SCALE or a.shape[1] % WORK_SCALE):
        raise ValueError(f"{name} sides must be multiples of {WORK_SCALE}, got {a.shape[:2]}")
    return a


def _grid(a: object, name: str, work_hw: tuple) -> np.ndarray:
    expected = (work_hw[0] // WORK_SCALE, work_hw[1] // WORK_SCALE)
    if not isinstance(a, np.ndarray) or a.dtype != np.bool_ or a.shape != expected:
        got = (a.dtype, a.shape) if isinstance(a, np.ndarray) else type(a).__name__
        raise ValueError(f"{name} must be a bool array of shape {expected}, got {got}")
    return a


def _palette(p: object) -> np.ndarray:
    if not isinstance(p, np.ndarray) or p.dtype != np.uint8 or p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("palette must be a uint8 array of shape (k, 3)")
    if not 1 <= p.shape[0] <= 256:
        raise ValueError(f"palette must hold 1..256 colours, got {p.shape[0]}")
    return p


def to_work(rgb: np.ndarray, cells_w: int, cells_h: int, n: int) -> np.ndarray:
    """Center-crop ``rgb`` to aspect ``cells_w:cells_h``, LANCZOS-resize to
    exactly ``(cells_w*n*8, cells_h*n*8)`` and return ``uint8 HxWx3``."""
    _rgb(rgb, "rgb")
    cells_w = _int(cells_w, "cells_w", 1, _CELLS_MAX)
    cells_h = _int(cells_h, "cells_h", 1, _CELLS_MAX)
    n = _int(n, "n", 1, max(TILE_SIZES))
    if n not in TILE_SIZES:
        raise ValueError(f"n must be one of {TILE_SIZES}, got {n!r}")
    h, w = rgb.shape[:2]
    if w * cells_h > h * cells_w:  # too wide: crop columns
        cw = max(1, h * cells_w // cells_h)
        x0 = (w - cw) // 2
        crop = rgb[:, x0 : x0 + cw]
    else:  # too tall (or equal): crop rows
        ch = max(1, w * cells_h // cells_w)
        y0 = (h - ch) // 2
        crop = rgb[y0 : y0 + ch]
    size = (cells_w * n * WORK_SCALE, cells_h * n * WORK_SCALE)
    img = Image.fromarray(np.ascontiguousarray(crop))
    return np.array(img.resize(size, Image.Resampling.LANCZOS), dtype=np.uint8)


def _ramp(side: int, span: float) -> np.ndarray:
    """0 on the first/last pixel, rising linearly to 1 over ``span`` pixels."""
    i = np.arange(side)
    d = np.minimum(i, side - 1 - i).astype(np.float32)
    return np.clip(d / np.float32(span), 0.0, 1.0)


def make_seamless(work: np.ndarray, blend: float = 0.25) -> np.ndarray:
    """Blend ``work`` against its half-shifted wrap copy so it tiles seamlessly.

    ``out = m*work + (1-m)*roll(work)`` with ``m`` zero on every border and
    one beyond ``blend*side`` from it. Border pixels equal the rolled copy,
    which is wrap-continuous. ``blend`` 0 returns an unchanged copy.
    """
    _rgb(work, "work")
    blend = _number(blend, "blend", 0.0, 0.5)
    if blend == 0.0:
        return work.copy()
    h, w = work.shape[:2]
    rolled = np.roll(work, (h // 2, w // 2), (0, 1)).astype(np.float32)
    mx = _ramp(w, blend * w)[None, :]
    my = _ramp(h, blend * h)[:, None]
    m = np.minimum(my, mx)[..., None]
    out = m * work.astype(np.float32) + (1.0 - m) * rolled
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def key_alpha(work: np.ndarray, n_cells_grid: Sequence[int], tol: int = 40) -> np.ndarray:
    """Cut the border-connected background of an object tile.

    ``n_cells_grid`` is the block-grid shape ``(rows, cols)`` =
    ``(work_h // 8, work_w // 8)``. The background colour is the per-channel
    median of the 1-px work border. Blocks whose mean max-channel distance to
    it is below ``tol`` and that are 4-connected to such a border block are
    transparent. One extra ring is peeled: a block with at least 25% of its
    pixels within ``tol`` of the background that touches (4-neighbourhood) the
    flooded area or the image border is transparent too. Returns the bool
    alpha grid (``True`` = opaque); the cut is block-aligned by construction.
    """
    _rgb(work, "work", block_aligned=True)
    tol = _number(tol, "tol", 0, 255)
    gh, gw = work.shape[0] // WORK_SCALE, work.shape[1] // WORK_SCALE
    try:
        shape = tuple(_int(v, "n_cells_grid entry", 1, 1 << 20) for v in n_cells_grid)
    except TypeError:
        raise ValueError("n_cells_grid must be a (rows, cols) pair of integers") from None
    if shape != (gh, gw):
        raise ValueError(f"n_cells_grid must be {(gh, gw)} for work {work.shape[:2]}, got {shape}")
    w16 = work.astype(np.int16)
    border = np.concatenate([w16[0], w16[-1], w16[:, 0], w16[:, -1]])
    bg = np.median(border, axis=0).astype(np.int16)
    dist = np.abs(w16 - bg).max(axis=2).astype(np.float32)
    blocks = dist.reshape(gh, WORK_SCALE, gw, WORK_SCALE)
    # Flood only through blocks whose mean distance is below tol (KB rule), so
    # thin dark outlines stop it. Mostly-background blocks (>= _BG_FRAC near-bg
    # pixels) are peeled afterwards, one ring only, to remove the fringe.
    cand = (blocks.mean(axis=(1, 3)) < tol).tolist()
    near = (blocks < tol).mean(axis=(1, 3)) >= _BG_FRAC
    seen = [[False] * gw for _ in range(gh)]
    queue: deque = deque()
    for r in range(gh):
        for c in range(gw):
            if (r in (0, gh - 1) or c in (0, gw - 1)) and cand[r][c]:
                seen[r][c] = True
                queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < gh and 0 <= nc < gw and cand[nr][nc] and not seen[nr][nc]:
                seen[nr][nc] = True
                queue.append((nr, nc))
    flooded = np.array(seen, dtype=np.bool_)
    pad = np.pad(flooded, 1)
    touch = pad[:-2, 1:-1] | pad[2:, 1:-1] | pad[1:-1, :-2] | pad[1:-1, 2:]
    touch[0] = touch[-1] = True
    touch[:, 0] = touch[:, -1] = True
    return ~(flooded | (near & touch))


def build_palette(works: Sequence[np.ndarray], alphas: Sequence[np.ndarray], k: int) -> np.ndarray:
    """Shared palette of at most ``k`` colours over all opaque work pixels.

    Opaque pixels are concatenated, subsampled with a deterministic stride to
    at most 262144 and median-cut quantised without dithering. Returns the
    unique used colours as ``(k', 3) uint8``, ``k' <= k``, sorted.
    """
    k = _int(k, "k", 1, 256)
    if isinstance(works, np.ndarray) or isinstance(alphas, np.ndarray) or len(works) != len(alphas):
        raise ValueError("works and alphas must be sequences of equal length")
    if len(works) == 0:
        raise ValueError("build_palette needs at least one tile")
    parts = []
    for i, (work, alpha) in enumerate(zip(works, alphas)):
        _rgb(work, f"works[{i}]", block_aligned=True)
        _grid(alpha, f"alphas[{i}]", work.shape[:2])
        mask = np.repeat(np.repeat(alpha, WORK_SCALE, axis=0), WORK_SCALE, axis=1)
        parts.append(work[mask])
    pixels = np.concatenate(parts)
    if pixels.shape[0] == 0:
        raise ValueError("build_palette found no opaque pixels (every tile fully transparent)")
    if pixels.shape[0] > _PALETTE_SAMPLE_MAX:
        pixels = pixels[:: -(-pixels.shape[0] // _PALETTE_SAMPLE_MAX)]
    img = Image.fromarray(np.ascontiguousarray(pixels.reshape(1, -1, 3)))
    quant = img.quantize(colors=k, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    table = np.array(quant.getpalette() or [], dtype=np.uint8).reshape(-1, 3)
    used = np.unique(np.asarray(quant))
    return np.unique(table[used], axis=0)


def index_image(work: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Nearest palette index per pixel (squared RGB distance, lowest index on
    ties) as ``int16 HxW``. Processed in row chunks to bound memory."""
    _rgb(work, "work")
    palette = _palette(palette)
    h, w = work.shape[:2]
    pal = palette.astype(np.float32)
    psq = (pal * pal).sum(axis=1)
    rows = max(1, _INDEX_CHUNK // (w * palette.shape[0]))
    out = np.empty((h, w), dtype=np.int16)
    for y in range(0, h, rows):
        px = work[y : y + rows].reshape(-1, 3).astype(np.float32)
        # |p-c|^2 = |p|^2 - 2 p.c + |c|^2; |p|^2 is constant per pixel. All
        # terms are integers < 2^24, so float32 is exact and ties are real.
        score = psq[None, :] - 2.0 * (px @ pal.T)
        out[y : y + rows] = score.argmin(axis=1).reshape(-1, w)
    return out


def block_reduce(idx: np.ndarray, alpha_grid: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Reduce each 8x8 index block to one RGBA pixel.

    The most frequent palette index wins (tie -> lowest index). Opaque blocks
    become ``(palette colour, 255)``, transparent blocks ``(0, 0, 0, 0)``.
    Returns ``uint8 (H/8, W/8, 4)``: every pixel is a palette colour or fully
    transparent.
    """
    palette = _palette(palette)
    if not isinstance(idx, np.ndarray) or idx.dtype.kind not in "iu" or idx.ndim != 2:
        raise ValueError("idx must be a 2-D integer array")
    h, w = idx.shape
    if h < 1 or w < 1 or h % WORK_SCALE or w % WORK_SCALE:
        raise ValueError(f"idx sides must be non-zero multiples of {WORK_SCALE}, got {idx.shape}")
    _grid(alpha_grid, "alpha_grid", idx.shape)
    k = palette.shape[0]
    if int(idx.min()) < 0 or int(idx.max()) >= k:
        raise ValueError(f"idx values must be in 0..{k - 1}")
    gh, gw = h // WORK_SCALE, w // WORK_SCALE
    blocks = idx.reshape(gh, WORK_SCALE, gw, WORK_SCALE).transpose(0, 2, 1, 3)
    blocks = blocks.reshape(gh * gw, WORK_SCALE * WORK_SCALE).astype(np.int64)
    flat = blocks + np.arange(gh * gw, dtype=np.int64)[:, None] * k
    counts = np.bincount(flat.ravel(), minlength=gh * gw * k).reshape(gh * gw, k)
    mode = counts.argmax(axis=1)  # argmax returns the first (lowest) index on ties
    out = np.zeros((gh, gw, 4), dtype=np.uint8)
    opaque = alpha_grid.reshape(-1)
    rgba = out.reshape(-1, 4)
    rgba[opaque, :3] = palette[mode[opaque]]
    rgba[opaque, 3] = 255
    return out
