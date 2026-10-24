"""Heuristic tile review: the deterministic floor under the vision review (``ccn-review/1``).

Flags measurable defects of generated tiles without any LLM: ``ground_not_texture``, ``not_tileable``
(ground tiles) and ``not_cut_out``, ``wrong_content``, ``fringe`` (object / overlay tiles). numpy + stdlib
only (plus ``pixelize.WORK_SCALE`` and ``plan_schema.CATEGORIES``); no torch, no I/O, no randomness.

Inputs are the *work* images of ``pixelize`` (``uint8 HxWx3``, both sides a multiple of ``WORK_SCALE``) and
the bool alpha grids of ``pixelize.key_alpha`` (shape ``(H // 8, W // 8)``, ``True`` = opaque). Which work
image is judged is the caller's choice; a ground work image that already went through
``pixelize.make_seamless`` is tileable by construction, so pass the raw one to judge the Flux output.

Every issue is a dict ``{"tile", "kind", "severity", "fix_hint", "score"}``. ``tile`` is the job key
``"<tile_id>#<variant>"``; ``kind`` and ``severity`` follow the review contract; ``score`` is the measured
statistic that was compared with the threshold (see the table in ``heuristic_issues``). Issues carry
``medium`` severity, ``ground_not_texture`` ``high``; the merge with vision issues dedupes by ``(tile, kind)``.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np

from .pixelize import WORK_SCALE
from .plan_schema import CATEGORIES

__all__ = [
    "GROUND_COLOUR_COVER", "GROUND_BORDER_COVER", "TILEABLE_RATIO", "CUTOUT_MAX_COVERAGE",
    "WRONG_CONTENT_MIN_COVERAGE", "FRINGE_EDGE_FRACTION", "FRINGE_PERCENTILE",
    "LIGHT_LUMA", "COLOUR_BIN_SHIFT", "COLOUR_TOLERANCE", "FIX_HINTS", "heuristic_issues",
]

# Thresholds from the review-loop architecture (section 5).
GROUND_COLOUR_COVER = 0.35  # one light colour covers >= this share of the work image ...
GROUND_BORDER_COVER = 0.60  # ... and >= this share of its 1 px border ring -> ground_not_texture
TILEABLE_RATIO = 2.0  # wrapped-edge mean |d| > this x interior neighbour mean |d| -> not_tileable
CUTOUT_MAX_COVERAGE = 0.90  # alpha coverage > this -> not_cut_out
WRONG_CONTENT_MIN_COVERAGE = 0.03  # alpha coverage < this -> wrong_content
FRINGE_EDGE_FRACTION = 0.15  # >= this share of edge pixels brighter than the percentile -> fringe
FRINGE_PERCENTILE = 90  # ... of the opaque pixel brightness of the tile
# Implementation parameters the architecture leaves open ("one light colour"); tune against real Flux output.
LIGHT_LUMA = 160.0  # a colour is light when its Rec.601 luma is >= this (0..255)
COLOUR_BIN_SHIFT = 5  # histogram bins of 2**5 levels per channel pick the dominant colour
COLOUR_TOLERANCE = 24  # a pixel is "that colour" when every channel is within this of the bin median

FIX_HINTS = {
    "ground_not_texture": "fill the whole frame edge to edge with the repeating ground texture, no flat background",
    "not_tileable": "make the texture continuous across all four edges so it repeats without a visible seam",
    "not_cut_out": "draw only the object on a plain flat background, leaving empty space around it",
    "wrong_content": "show the described object large and clearly, centred in the frame",
    "fringe": "keep the object outline clean with no bright halo or light edge pixels",
}
_LUMA = np.array([0.299, 0.587, 0.114])
_WORK_PIXELS_MAX = (4 * 64 * WORK_SCALE) ** 2  # 4 cells of 64 px at the work scale, per side


def _issue(key: str, kind: str, score: float) -> dict:
    severity = "high" if kind == "ground_not_texture" else "medium"
    return {"tile": key, "kind": kind, "severity": severity, "fix_hint": FIX_HINTS[kind], "score": float(score)}


def _ground_not_texture(key: str, work: np.ndarray) -> Optional[dict]:
    """Dominant light colour covering the image and its border ring (flat blob-on-cream ground)."""
    flat = work.reshape(-1, 3)
    bins = (flat >> COLOUR_BIN_SHIFT).astype(np.int32)
    side = 256 >> COLOUR_BIN_SHIFT
    packed = (bins[:, 0] * side + bins[:, 1]) * side + bins[:, 2]
    mode = int(np.argmax(np.bincount(packed)))
    centre = np.median(flat[packed == mode].astype(np.float64), axis=0)
    if float(centre @ _LUMA) < LIGHT_LUMA:
        return None
    ring = np.concatenate([work[0], work[-1], work[1:-1, 0], work[1:-1, -1]])

    def share(px: np.ndarray) -> float:
        return float((np.abs(px.astype(np.float64) - centre).max(axis=-1) <= COLOUR_TOLERANCE).mean())

    cover, border = share(flat), share(ring)
    if cover >= GROUND_COLOUR_COVER and border >= GROUND_BORDER_COVER:
        return _issue(key, "ground_not_texture", cover)
    return None


def _not_tileable(key: str, work: np.ndarray) -> Optional[dict]:
    """Wrapped left/right and top/bottom edge difference versus the ordinary neighbour difference."""
    w = work.astype(np.int16)
    wrap = np.concatenate([np.abs(w[:, 0] - w[:, -1]).ravel(), np.abs(w[0] - w[-1]).ravel()])
    dx, dy = np.abs(w[:, 1:] - w[:, :-1]), np.abs(w[1:] - w[:-1])
    interior = (float(dx.sum()) + float(dy.sum())) / (dx.size + dy.size)
    if interior == 0.0:  # constant image: wrapped edges are equal too, nothing to compare
        return None
    ratio = float(wrap.mean()) / interior
    return _issue(key, "not_tileable", ratio) if ratio > TILEABLE_RATIO else None


def _coverage_issues(key: str, alpha: np.ndarray) -> list:
    coverage = float(alpha.mean())
    if coverage > CUTOUT_MAX_COVERAGE:
        return [_issue(key, "not_cut_out", coverage)]
    if coverage < WRONG_CONTENT_MIN_COVERAGE:
        return [_issue(key, "wrong_content", coverage)]
    return []


def _fringe(key: str, work: np.ndarray, alpha: np.ndarray) -> Optional[dict]:
    """Bright opaque pixels touching transparency (grid border is not transparency)."""
    rows, cols = alpha.shape
    blocks = work.reshape(rows, WORK_SCALE, cols, WORK_SCALE, 3).astype(np.float64).mean(axis=(1, 3))
    luma = blocks @ _LUMA
    pad = np.pad(alpha, 1, constant_values=True)
    touches = ~pad[:-2, 1:-1] | ~pad[2:, 1:-1] | ~pad[1:-1, :-2] | ~pad[1:-1, 2:]
    edge = alpha & touches
    if not edge.any():
        return None
    bright = luma[edge] > np.percentile(luma[alpha], FRINGE_PERCENTILE)
    frac = float(bright.mean())
    return _issue(key, "fringe", frac) if frac >= FRINGE_EDGE_FRACTION else None


def _check_work(key: str, work: object) -> np.ndarray:
    if not isinstance(work, np.ndarray) or work.dtype != np.uint8 or work.ndim != 3 or work.shape[2] != 3:
        raise ValueError(f"cells_work[{key!r}] must be a uint8 RGB array of shape HxWx3")
    h, w = work.shape[:2]
    if h < 2 * WORK_SCALE or w < 2 * WORK_SCALE or h % WORK_SCALE or w % WORK_SCALE:
        raise ValueError(f"cells_work[{key!r}] sides must be multiples of {WORK_SCALE} and at least {2 * WORK_SCALE}, got {w}x{h}")
    if h * w > _WORK_PIXELS_MAX:
        raise ValueError(f"cells_work[{key!r}] is too large ({w}x{h} px)")
    return work


def _check_alpha(key: str, alpha: object, work: np.ndarray) -> np.ndarray:
    want = (work.shape[0] // WORK_SCALE, work.shape[1] // WORK_SCALE)
    if not isinstance(alpha, np.ndarray) or alpha.dtype != np.bool_ or alpha.shape != want:
        raise ValueError(f"alphas[{key!r}] must be a bool array of shape {want} (rows, cols of the work image / {WORK_SCALE})")
    return alpha


def _job_keys(jobs: Sequence) -> list:
    if isinstance(jobs, (str, bytes, Mapping)) or not isinstance(jobs, Sequence):
        raise ValueError("jobs must be a list of job mappings")
    keys = []
    for i, job in enumerate(jobs):
        if not isinstance(job, Mapping) or any(k not in job for k in ("tile_id", "category", "variant")):
            raise ValueError(f"job {i} must be a mapping with keys tile_id, category, variant")
        if not isinstance(job["tile_id"], str) or job["category"] not in CATEGORIES:
            raise ValueError(f"job {i} needs a text tile_id and a category from {list(CATEGORIES)}")
        if type(job["variant"]) is not int or job["variant"] < 0:
            raise ValueError(f"job {i} variant must be a non-negative integer")
        keys.append(f"{job['tile_id']}#{job['variant']}")
    if len(set(keys)) != len(keys):
        raise ValueError("jobs contain duplicate tile keys")
    return keys


def heuristic_issues(cells_work: Mapping, alphas: Mapping, jobs: Sequence) -> list:
    """Return the heuristic issues of every job, in job order (then detector order below).

    ``cells_work``: job key -> ``uint8 HxWx3`` work image (every job needs one, no extra keys).
    ``alphas``: job key -> bool alpha grid for ``object`` / ``overlay`` jobs (required there, shape
    ``(H // 8, W // 8)``); ``ground`` jobs need none (absent or ``None``). ``jobs``: mappings with
    ``tile_id``, ``category`` (ground / object / overlay) and ``variant``. Invalid input raises ``ValueError``.

    Detectors, thresholds (named constants above) and the ``score`` each one reports:

    * ground ``ground_not_texture`` (high): the dominant colour is light (luma >= ``LIGHT_LUMA``) and covers
      >= ``GROUND_COLOUR_COVER`` of the image and >= ``GROUND_BORDER_COVER`` of the border ring; score = image share.
    * ground ``not_tileable``: mean |left - right column| and |top - bottom row| > ``TILEABLE_RATIO`` x the mean
      |neighbour difference| of the image; score = that ratio.
    * object / overlay ``not_cut_out``: alpha coverage > ``CUTOUT_MAX_COVERAGE``; score = coverage.
    * object / overlay ``wrong_content``: alpha coverage < ``WRONG_CONTENT_MIN_COVERAGE``; score = coverage.
    * object / overlay ``fringe``: >= ``FRINGE_EDGE_FRACTION`` of the opaque pixels 4-adjacent to a transparent
      one are brighter (block-mean luma) than the ``FRINGE_PERCENTILE`` percentile of the opaque pixels; score = that share.
    """
    if not isinstance(cells_work, Mapping) or not isinstance(alphas, Mapping):
        raise ValueError("cells_work and alphas must be mappings keyed by tile key")
    keys = _job_keys(jobs)
    for name, mapping in (("cells_work", cells_work), ("alphas", alphas)):
        extra = sorted(str(k) for k in mapping if k not in keys)
        if extra:
            raise ValueError(f"{name} has keys that are not in jobs: {extra[:5]}")
    issues = []
    for job, key in zip(jobs, keys):
        if key not in cells_work:
            raise ValueError(f"cells_work has no image for {key!r}")
        work = _check_work(key, cells_work[key])
        if job["category"] == "ground":
            if alphas.get(key) is not None:
                _check_alpha(key, alphas[key], work)
            found = [_ground_not_texture(key, work), _not_tileable(key, work)]
        else:
            if alphas.get(key) is None:
                raise ValueError(f"alphas has no grid for {job['category']} {key!r}")
            alpha = _check_alpha(key, alphas[key], work)
            found = _coverage_issues(key, alpha) + [_fringe(key, work, alpha)]
        issues.extend(f for f in found if f is not None)
    return issues
