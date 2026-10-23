"""Pixelize a whole tile job list against one shared palette, and derive Wang transition tiles.

Extracted from ``nodes/tilemap_assemble.py`` (``_snap_tiles``, ``_wang_items``, ``_wang_id``,
``_preview_cells``) so the review loop and the assemble node run the exact same code. numpy + PIL only
(through ``pixelize``/``transitions``); no torch, no ComfyUI imports, no IO, deterministic.

Images are ``uint8 HxWx3`` numpy arrays (the node converts its IMAGE tensors first). Two palette paths:

* ``palette_mode = "keep"`` (default): quantise to the ``palette`` argument when given, else fit a new one.
* ``palette_mode = "refit"``: always fit a new palette over all non-base works (``palette`` must be None).

``base_cells`` maps job keys (``"<tile_id>#<variant>"``) to final ``uint8 RGBA`` cells (``(ch*n, cw*n, 4)``)
of an earlier tileset. Those jobs are passed through as copies: their image is ignored, never keyed, made
seamless or re-quantised, and they take no part in palette fitting. They require a fixed ``palette``.
"""

from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

import numpy as np

from . import pixelize
from . import transitions as wang

PALETTE_MODES = ("keep", "refit")
SNAP_DEFAULTS = {
    "palette_size": 24, "seam_blend": 0.25, "key_tolerance": 40, "transitions": True,
    "palette_mode": "keep", "extra_colours": 0,
}
_SEED_LIMIT = 2**64
_IMAGE_PIXELS_MAX = 2048 * 2048  # per Flux image (jobs are <= 1024 px a side); larger is rejected
_JOB_KEYS = ("tile_id", "category", "variant", "footprint")


def _int_in(value: object, name: str, lo: int, hi: int) -> int:
    """Strict integer: a real int (no bool/float) inside ``lo..hi``."""
    if type(value) is not int or not lo <= value <= hi:
        raise ValueError(f"{name} must be an integer from {lo} to {hi}")
    return value


def _float_in(value: object, name: str, lo: float, hi: float) -> float:
    """Strict float: int or float (no bool), finite, inside ``lo..hi``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number from {lo} to {hi}")
    if not math.isfinite(value) or not lo <= value <= hi:
        raise ValueError(f"{name} must be a number from {lo} to {hi}")
    return float(value)


def check_snap_params(snap_params: Optional[Mapping]) -> dict:
    """Validate ``snap_params`` (the ``job_json.snap`` block) and return a full dict; unknown keys are errors."""
    if snap_params is None:
        snap_params = {}
    if not isinstance(snap_params, Mapping):
        raise ValueError("snap_params must be a mapping")
    unknown = sorted(str(k) for k in snap_params if k not in SNAP_DEFAULTS)
    if unknown:
        raise ValueError(f"snap_params has unknown keys {unknown}; allowed: {sorted(SNAP_DEFAULTS)}")
    p = {**SNAP_DEFAULTS, **snap_params}
    _int_in(p["palette_size"], "palette_size", 1, 256)
    _float_in(p["seam_blend"], "seam_blend", 0.0, 0.5)
    _int_in(p["key_tolerance"], "key_tolerance", 0, 255)
    _int_in(p["extra_colours"], "extra_colours", 0, 16)
    if not isinstance(p["transitions"], bool):
        raise ValueError("transitions must be a boolean")
    if p["palette_mode"] not in PALETTE_MODES:
        raise ValueError(f"palette_mode must be one of {list(PALETTE_MODES)}, got {p['palette_mode']!r}")
    return p


def _check_palette(palette: object) -> np.ndarray:
    if not isinstance(palette, np.ndarray) or palette.dtype != np.uint8 or palette.ndim != 2 or palette.shape[1] != 3:
        raise ValueError("palette must be a uint8 array of shape (k, 3)")
    if not 1 <= palette.shape[0] <= 256:
        raise ValueError(f"palette must hold 1..256 colours, got {palette.shape[0]}")
    return palette


def _job_key(job: object, i: int) -> str:
    if not isinstance(job, Mapping) or any(k not in job for k in _JOB_KEYS):
        raise ValueError(f"job {i} must be a mapping with keys {list(_JOB_KEYS)}")
    if not isinstance(job["category"], str) or not isinstance(job["tile_id"], str):
        raise ValueError(f"job {i} tile_id and category must be text")
    return f"{job['tile_id']}#{job['variant']}"


def _check_image(img: object, i: int) -> np.ndarray:
    """Job image ``i`` as ``uint8 HxWx3``; size is checked before any conversion."""
    if not isinstance(img, np.ndarray) or img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"image {i} must be a uint8 RGB array of shape HxWx3")
    if img.shape[0] * img.shape[1] > _IMAGE_PIXELS_MAX:
        raise ValueError(f"image {i} is too large ({img.shape[1]}x{img.shape[0]} px)")
    return img


def _check_base(base_cells: Mapping, keys: list, jobs: list, size: int, palette: Optional[np.ndarray]) -> None:
    if palette is None:
        raise ValueError("base_cells need a fixed palette; pass palette= (palette_mode 'keep')")
    unknown = sorted(str(k) for k in base_cells if k not in keys)
    if unknown:
        raise ValueError(f"base_cells has keys that are not in jobs: {unknown[:5]}")
    for job, key in zip(jobs, keys):
        cell = base_cells.get(key)
        if cell is None:
            continue
        cw, ch = job["footprint"]
        want = (ch * size, cw * size, 4)
        if not isinstance(cell, np.ndarray) or cell.dtype != np.uint8 or cell.shape != want:
            raise ValueError(f"base_cells[{key!r}] must be a uint8 RGBA array of shape {want}")


def snap_tiles(
    images: Sequence, jobs: Sequence, size: int, snap_params: Optional[Mapping] = None,
    palette: Optional[np.ndarray] = None, base_cells: Optional[Mapping] = None,
) -> tuple:
    """Pixelize every job image; return ``(cell_images by key, palette)``.

    ``images[i]`` is the ``uint8 HxWx3`` Flux image of ``jobs[i]`` (ignored for keys in ``base_cells``).
    Ground tiles are made seamless (``seam_blend``) and fully opaque, other categories get a keyed alpha
    (``key_tolerance``). ``snap_params`` is a ``job_json.snap`` style mapping (see ``SNAP_DEFAULTS``).
    Without ``palette`` (or with ``palette_mode='refit'``) one palette of ``palette_size`` colours is fitted
    over all non-base works; with ``palette`` that palette is used unchanged and returned as given.
    """
    p = check_snap_params(snap_params)
    if len(images) != len(jobs):
        raise ValueError(f"got {len(images)} images for {len(jobs)} jobs")
    if len(jobs) == 0:
        raise ValueError("snap_tiles needs at least one job")
    keys = [_job_key(job, i) for i, job in enumerate(jobs)]
    if len(set(keys)) != len(keys):
        raise ValueError("jobs contain duplicate tile keys")
    if palette is not None:
        palette = _check_palette(palette)
        if p["palette_mode"] == "refit":
            raise ValueError("palette_mode 'refit' fits a new palette; do not pass palette=")
    base_cells = {} if base_cells is None else base_cells
    if base_cells:
        _check_base(base_cells, keys, jobs, size, palette)
    works, alphas = {}, {}
    for i, (job, img) in enumerate(zip(jobs, images)):
        if keys[i] in base_cells:
            continue
        rgb = _check_image(img, i)
        cw, ch = job["footprint"]
        work = pixelize.to_work(rgb, cw, ch, size)
        del rgb
        grid = (work.shape[0] // pixelize.WORK_SCALE, work.shape[1] // pixelize.WORK_SCALE)
        if job["category"] == "ground":
            work = pixelize.make_seamless(work, p["seam_blend"])
            alpha = np.ones(grid, bool)
        else:
            alpha = pixelize.key_alpha(work, grid, p["key_tolerance"])
        works[i], alphas[i] = work, alpha
    if palette is None:
        palette = pixelize.build_palette(list(works.values()), list(alphas.values()), p["palette_size"])
    cells = {}
    for i, key in enumerate(keys):
        if key in base_cells:
            cells[key] = np.array(base_cells[key], dtype=np.uint8, copy=True)
            continue
        idx = pixelize.index_image(works[i], palette)
        cells[key] = pixelize.block_reduce(idx, alphas[i], palette)
        works[i] = alphas[i] = None  # free the big intermediates as we go
    return cells, palette


def wang_id(a: str, b: str, taken: set) -> str:
    """Return ``a_b``; raise if a plan tile or an earlier Wang pair already owns that id."""
    wid = f"{a}_{b}"
    if wid in taken:
        raise ValueError(f"transition {a}+{b}: Wang tile id {wid!r} collides with another tile id; rename a tile")
    taken.add(wid)
    return wid


def wang_items(plan: Mapping, cells: Mapping, seed: int, n: int) -> tuple:
    """Return ``(items, cell images)`` for the 16 Wang tiles of every plan transition.

    ``cells`` must hold ``"<a>#0"`` and ``"<b>#0"`` of every transition pair; pair ``k`` is seeded with
    ``(seed + k) % 2**64``. Item keys are ``"<a>_<b>#<mask>"``.
    """
    if not isinstance(plan, Mapping) or not isinstance(plan.get("tiles"), list) or not isinstance(plan.get("transitions"), list):
        raise ValueError("plan must be a mapping with 'tiles' and 'transitions' lists")
    _int_in(seed, "seed", 0, _SEED_LIMIT - 1)
    taken = {t["id"] for t in plan["tiles"]}
    items, images = [], {}
    for pair_no, (a, b) in enumerate(plan["transitions"]):
        missing = [k for k in (f"{a}#0", f"{b}#0") if k not in cells]
        if missing:
            raise ValueError(f"transition {a}+{b}: no cell image for {missing}")
        wid = wang_id(a, b, taken)
        tiles = wang.wang_tiles(cells[f"{a}#0"], cells[f"{b}#0"], n, (seed + pair_no) % _SEED_LIMIT)
        for mask, (corners, rgba) in enumerate(tiles):
            key = f"{wid}#{mask}"
            wang_info = {"a": a, "b": b, "corners": list(corners)}
            items.append(
                {
                    "key": key, "cells_w": 1, "cells_h": 1, "id": wid, "name": f"{a} to {b} {mask}",
                    "category": "ground", "variant": mask, "wang": wang_info,
                }
            )
            images[key] = rgba
    return items, images


def preview_cells(placements: Sequence, cell_images: Mapping, n: int) -> list:
    """All n x n cells (RGB) in placement order, row-major inside a footprint."""
    out = []
    for p in placements:
        if p["key"] not in cell_images:
            raise ValueError(f"no cell image for placement {p['key']!r}")
        img = cell_images[p["key"]]
        for r in range(p["cells_h"]):
            for c in range(p["cells_w"]):
                out.append(img[r * n : (r + 1) * n, c * n : (c + 1) * n, :3])
    return out
