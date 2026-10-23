"""Palette extension for the tilemap ``extend`` mode.

Pure numpy, deterministic, no I/O, no global state. ``extend_palette`` keeps
an existing global palette byte-for-byte and appends at most ``k_extra``
colours fitted only to the *residual* opaque pixels of new tiles: pixels whose
nearest base colour is farther than ``_EXTEND_MIN_DIST`` (Euclidean distance in
plain RGB, no colour-space conversion). Pixels already covered by the base
never influence the result, so existing tiles (which only ever used base
entries) stay exact and two successive extends are reproducible.

Lives beside ``pixelize`` (not inside it) because of the 300-line rule; it
imports helpers from ``pixelize`` and ``pixelize`` must never import it.
Validation follows ``pixelize.build_palette``: ``ValueError`` with a message,
no silent clamping, inputs never mutated.
"""

from typing import Sequence

import numpy as np

from .pixelize import WORK_SCALE, _grid, _int, _median_cut, _palette, _rgb

__all__ = ["extend_palette"]

_EXTEND_MIN_DIST = 24  # RGB Euclidean distance; a pixel farther than this from every base colour is residual
_K_EXTRA_MAX = 16  # extra_colours widget range 0..16
_PALETTE_MAX = 256
_DIST_CHUNK = 4_000_000  # max (colours x palette) int32 cells per chunk


def _nearest_sq(colours: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Exact squared RGB distance from each ``(m, 3)`` colour to its nearest ``base`` colour."""
    pal = base.astype(np.int32)
    out = np.empty(colours.shape[0], dtype=np.int32)
    rows = max(1, _DIST_CHUNK // pal.shape[0])
    for y in range(0, colours.shape[0], rows):
        diff = colours[y : y + rows, None, :].astype(np.int32) - pal[None, :, :]
        out[y : y + rows] = (diff * diff).sum(axis=2).min(axis=1)
    return out


def extend_palette(
    base: np.ndarray, works: Sequence[np.ndarray], alphas: Sequence[np.ndarray], k_extra: int
) -> np.ndarray:
    """Extend ``base`` ``(n, 3) uint8`` by at most ``k_extra`` colours (0..16).

    ``works``/``alphas`` are the new tiles (``uint8 HxWx3`` multiples of 8 and
    matching bool alpha grids, as for ``build_palette``). Only opaque pixels
    whose nearest base colour is farther than 24 (exactly 24 counts as covered)
    are median-cut to at most ``k_extra`` colours. A candidate is dropped when
    it lies within 24 of a base colour (this includes equality) or repeats an
    earlier candidate, so the result never holds duplicates that were not
    already in ``base``. Returns a new ``(n + k', 3) uint8`` array,
    ``k' <= k_extra``, ``n + k' <= 256``; rows ``0..n-1`` equal ``base`` and
    appended rows are sorted. ``k_extra == 0`` or no residual returns a copy
    of ``base``.
    """
    base = _palette(base)
    k_extra = _int(k_extra, "k_extra", 0, _K_EXTRA_MAX)
    if base.shape[0] + k_extra > _PALETTE_MAX:
        raise ValueError(f"palette of {base.shape[0]} + k_extra {k_extra} exceeds {_PALETTE_MAX} colours")
    if isinstance(works, np.ndarray) or isinstance(alphas, np.ndarray) or len(works) != len(alphas):
        raise ValueError("works and alphas must be sequences of equal length")
    if len(works) == 0:
        raise ValueError("extend_palette needs at least one tile")
    parts = []
    for i, (work, alpha) in enumerate(zip(works, alphas)):
        _rgb(work, f"works[{i}]", block_aligned=True)
        _grid(alpha, f"alphas[{i}]", work.shape[:2])
        mask = np.repeat(np.repeat(alpha, WORK_SCALE, axis=0), WORK_SCALE, axis=1)
        parts.append(work[mask])
    pixels = np.concatenate(parts)
    out = base.copy()
    if k_extra == 0 or pixels.shape[0] == 0:
        return out
    # Distances are computed once per distinct colour, then mapped back so the
    # median cut still sees every residual pixel (frequency-weighted).
    packed = (pixels[:, 0].astype(np.uint32) << 16) | (pixels[:, 1].astype(np.uint32) << 8) | pixels[:, 2]
    uniq, inverse = np.unique(packed, return_inverse=True)
    colours = np.stack([(uniq >> 16) & 255, (uniq >> 8) & 255, uniq & 255], axis=1).astype(np.uint8)
    residual_colour = _nearest_sq(colours, base) > _EXTEND_MIN_DIST**2
    residual = pixels[residual_colour[inverse.reshape(-1)]]
    if residual.shape[0] == 0:
        return out
    cand = _median_cut(residual, k_extra)  # sorted unique rows, so the kept subset stays sorted and unique
    cand = cand[_nearest_sq(cand, base) > _EXTEND_MIN_DIST**2]
    return np.concatenate([out, cand])
