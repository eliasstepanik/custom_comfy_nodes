"""``CCN_TilemapAssemble``: turn the Flux images of a tile job list into one pixel-exact RGBA atlas.

Inputs are the per-job images (a list, one tensor per job, in job order) and the ``job_json`` string
from ``CCN_TilemapExpand``. Per job the image is cropped/resized to the work size, ground tiles are made
seamless, objects/overlays get a keyed alpha, then one shared palette is built over all tiles and every
tile is snapped to it and block-reduced to ``tile_size`` pixels per cell. Wang corner transitions are
generated for the plan's ground pairs, everything is packed and verified before it is returned.

``job_json`` arrives through a STRING socket, so it is untrusted: it is parsed strictly, the plan is
re-validated and the job list must equal the one the plan expands to. Errors never echo raw payload text.
No network, no file IO.
"""

from __future__ import annotations

import json
import math

import numpy as np
import torch

from ..tilemap_core import atlas, pixelize, plan_prompt, plan_schema
from ..tilemap_core import expand as tile_expand
from ..tilemap_core import transitions as wang
from ._tensor_io import alpha_to_mask, image_list_to_np, np_to_image

CATEGORY = "custom_comfy_nodes/tilemap"
_JOB_JSON_MAX = 400_000
_SEED_LIMIT = 2**64
_JOB_KEYS = ("tile_id", "category", "variant", "footprint")
_WANG_VARIANTS = 16
_IMAGE_PIXELS_MAX = 2048 * 2048  # per Flux image (jobs are <= 1024 px a side); larger is rejected
_FIX = "reconnect job_json from CCN_TilemapExpand"


def _one(value: object, name: str) -> object:
    """INPUT_IS_LIST delivers every input as a list; scalar widgets are element 0."""
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"{name} is missing")
        return value[0]
    return value


def _int_in(value: object, name: str, lo: int, hi: int) -> int:
    """Strict widget integer: a real int (no bool/float) inside ``lo..hi``."""
    if type(value) is not int or not lo <= value <= hi:
        raise ValueError(f"{name} must be an integer from {lo} to {hi}")
    return value


def _float_in(value: object, name: str, lo: float, hi: float) -> float:
    """Strict widget float: int or float (no bool), finite, inside ``lo..hi``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number from {lo} to {hi}")
    if not math.isfinite(value) or not lo <= value <= hi:
        raise ValueError(f"{name} must be a number from {lo} to {hi}")
    return float(value)


def _no_dup(pairs: list) -> dict:
    out = dict(pairs)
    if len(out) != len(pairs):
        raise ValueError("duplicate key")
    return out


def _parse_int(text: str) -> int:
    if len(text) > 20:
        raise ValueError("integer too long")
    return int(text)


def _reject_constant(_name: str) -> None:
    raise ValueError("non-finite number")


def _load_json(text: object) -> dict:
    if not isinstance(text, str):
        raise ValueError(f"job_json must be text, got {type(text).__name__}; {_FIX}")
    if len(text) > _JOB_JSON_MAX:
        raise ValueError(f"job_json is too long ({len(text)} chars, limit {_JOB_JSON_MAX}); {_FIX}")
    try:
        data = json.loads(text, object_pairs_hook=_no_dup, parse_int=_parse_int, parse_constant=_reject_constant)
    except (ValueError, RecursionError):
        raise ValueError(f"job_json is not valid JSON; {_FIX}") from None
    if not isinstance(data, dict):
        raise ValueError(f"job_json must be a JSON object; {_FIX}")
    return data


def _signature(jobs: list) -> str:
    """Canonical text of the fields that decide tile identity; JSON keeps True != 1 and 1.0 != 1."""
    rows = [[j.get(k) if isinstance(j, dict) else None for k in _JOB_KEYS] for j in jobs]
    return json.dumps(rows, sort_keys=True)


def _parse_jobs(text: object) -> tuple:
    """Return ``(tile_size, plan, jobs, base_seed)`` from an untrusted ``job_json``; ``jobs`` is re-derived."""
    data = _load_json(text)
    size = data.get("tile_size")
    if type(size) is not int or size not in tile_expand.TILE_SIZES:
        raise ValueError(f"job_json tile_size must be one of {list(tile_expand.TILE_SIZES)}; {_FIX}")
    view = data.get("view")
    if not isinstance(view, str) or view not in plan_prompt.VIEWS:
        raise ValueError(f"job_json view must be one of {list(plan_prompt.VIEWS)}; {_FIX}")
    try:
        plan = plan_schema.validate_plan(data.get("plan"), plan_schema.MAX_JOBS)
    except plan_schema.PlanError:
        raise ValueError(f"job_json plan failed validation; {_FIX}") from None
    expected = tile_expand.expand_jobs(plan, size, view, "", 0)
    jobs = data.get("jobs")
    try:
        matches = isinstance(jobs, list) and _signature(jobs) == _signature(expected)
    except RecursionError:
        matches = False
    if not matches:
        raise ValueError(f"job_json jobs do not match its plan; {_FIX}")
    seed = jobs[0].get("seed")
    if type(seed) is not int or not 0 <= seed < _SEED_LIMIT:
        raise ValueError(f"job_json job seed must be an integer from 0 to {_SEED_LIMIT - 1}; {_FIX}")
    return size, view, plan, expected, seed


def _wang_id(a: str, b: str, taken: set) -> str:
    """Return ``a_b``; raise if a plan tile or an earlier Wang pair already owns that id."""
    wid = f"{a}_{b}"
    if wid in taken:
        raise ValueError(f"transition {a}+{b}: Wang tile id {wid!r} collides with another tile id; rename a tile")
    taken.add(wid)
    return wid


def _image_rgb(img: object, i: int) -> np.ndarray:
    """Batch element 0 of one IMAGE tensor as uint8 RGB; size is checked before any conversion."""
    if not isinstance(img, torch.Tensor) or img.ndim != 4:
        raise ValueError(f"image {i} must be an IMAGE tensor [B,H,W,C]")
    if img.shape[1] * img.shape[2] > _IMAGE_PIXELS_MAX:
        raise ValueError(f"image {i} is too large ({img.shape[2]}x{img.shape[1]} px)")
    try:
        return image_list_to_np(img[:1], 3, True)[0]
    except ValueError as exc:
        raise ValueError(f"image {i}: {exc}") from None


def _snap_tiles(images: list, jobs: list, size: int, palette_size: int, blend: float, tol: int) -> tuple:
    """Pixelize every job image; return ``(cell_images by key, palette)``."""
    works, alphas = [], []
    for i, (job, img) in enumerate(zip(jobs, images)):
        rgb = _image_rgb(img, i)
        cw, ch = job["footprint"]
        work = pixelize.to_work(rgb, cw, ch, size)
        del rgb
        grid = (work.shape[0] // pixelize.WORK_SCALE, work.shape[1] // pixelize.WORK_SCALE)
        if job["category"] == "ground":
            work = pixelize.make_seamless(work, blend)
            alpha = np.ones(grid, bool)
        else:
            alpha = pixelize.key_alpha(work, grid, tol)
        works.append(work)
        alphas.append(alpha)
    palette = pixelize.build_palette(works, alphas, palette_size)
    cells = {}
    for i, job in enumerate(jobs):
        idx = pixelize.index_image(works[i], palette)
        cells[f"{job['tile_id']}#{job['variant']}"] = pixelize.block_reduce(idx, alphas[i], palette)
        works[i] = alphas[i] = None  # free the big intermediates as we go
    return cells, palette


def _wang_items(plan: dict, cells: dict, seed: int, n: int) -> tuple:
    """Return ``(items, cell images)`` for the 16 Wang tiles of every plan transition."""
    taken = {t["id"] for t in plan["tiles"]}
    items, images = [], {}
    for pair_no, (a, b) in enumerate(plan["transitions"]):
        wid = _wang_id(a, b, taken)
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


def _preview_cells(placements: list, cell_images: dict, n: int) -> list:
    """All n x n cells (RGB) in placement order, row-major inside a footprint."""
    out = []
    for p in placements:
        img = cell_images[p["key"]]
        for r in range(p["cells_h"]):
            for c in range(p["cells_w"]):
                out.append(img[r * n : (r + 1) * n, c * n : (c + 1) * n, :3])
    return out


class CCN_TilemapAssemble:
    """Assemble Flux tile images + job_json into a pixel-exact RGBA tile atlas with metadata."""

    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "images": ("IMAGE",),
                "job_json": ("STRING", {"forceInput": True}),
                "palette_size": ("INT", {"default": 24, "min": 4, "max": 64, "step": 1}),
                "columns": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1}),
                "spacing": ("INT", {"default": 0, "min": 0, "max": 16, "step": 1}),
                "seam_blend": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 0.5, "step": 0.05}),
                "key_tolerance": ("INT", {"default": 40, "min": 0, "max": 255, "step": 1}),
                "transitions": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("atlas", "alpha", "tiles", "metadata")
    FUNCTION = "assemble"
    CATEGORY = CATEGORY

    def assemble(
        self, images: list, job_json: list, palette_size: list, columns: list, spacing: list,
        seam_blend: list, key_tolerance: list, transitions: list,
    ) -> tuple:
        psize = _int_in(_one(palette_size, "palette_size"), "palette_size", 4, 64)
        columns = _int_in(_one(columns, "columns"), "columns", 1, 64)
        spacing = _int_in(_one(spacing, "spacing"), "spacing", 0, 16)
        blend = _float_in(_one(seam_blend, "seam_blend"), "seam_blend", 0.0, 0.5)
        tol = _int_in(_one(key_tolerance, "key_tolerance"), "key_tolerance", 0, 255)
        with_wang = _one(transitions, "transitions")
        if not isinstance(with_wang, bool):
            raise ValueError("transitions must be a boolean")
        size, view, plan, jobs, seed = _parse_jobs(_one(job_json, "job_json"))
        images = list(images) if isinstance(images, (list, tuple)) else [images]
        if len(images) != len(jobs):
            raise ValueError(f"got {len(images)} images for {len(jobs)} jobs; connect one image per job in job order")
        cells, palette = _snap_tiles(images, jobs, size, psize, blend, tol)
        items = [
            {
                "key": f"{j['tile_id']}#{j['variant']}", "cells_w": j["footprint"][0], "cells_h": j["footprint"][1],
                "id": j["tile_id"], "name": j["name"], "category": j["category"], "variant": j["variant"],
                "wang": None,
            }
            for j in jobs
        ]
        wang_count = 0
        if with_wang:
            wang_items, wang_images = _wang_items(plan, cells, seed, size)
            items += wang_items
            cells.update(wang_images)
            wang_count = len(plan["transitions"]) * _WANG_VARIANTS
        placements = atlas.pack(items, columns)
        rgba = atlas.assemble(placements, cells, size, columns, spacing)
        meta = atlas.build_metadata(placements, size, columns, spacing, palette, view, "", seed, plan["theme"])
        if len(meta["tiles"]) != len(jobs) + wang_count:
            raise RuntimeError(f"atlas has {len(meta['tiles'])} tiles, expected {len(jobs) + wang_count}")
        atlas.verify_atlas(rgba, meta, palette, psize)
        previews = _preview_cells(placements, cells, size)
        return (
            np_to_image([rgba]),
            alpha_to_mask(rgba[..., 3]),
            np_to_image(previews),
            json.dumps(meta, sort_keys=True, separators=(",", ":")),
        )


NODE_CLASS_MAPPINGS = {"CCN_TilemapAssemble": CCN_TilemapAssemble}
NODE_DISPLAY_NAME_MAPPINGS = {"CCN_TilemapAssemble": "Tilemap Assemble (pixel-perfect atlas)"}
