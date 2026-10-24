"""``CCN_TilemapAssemble``: turn the Flux images of a tile job list into one pixel-exact RGBA atlas.

Thin wrapper over ``tilemap_core.snap`` and ``tilemap_core.atlas`` (metadata ``ccn-tilemap/2``).
``job_json.snap`` overrides the widgets. With ``job_json.base`` plus ``base_atlas``/``base_alpha`` the base
tiles without an image stay byte-identical, the palette stays fixed (plus ``snap.extra_colours``) and new
tiles are appended; ``spacing`` must equal the base atlas spacing. ``job_json`` is untrusted: parsed
strictly, plan re-validated, job list must equal the plan's (minus kept base tiles). No network/file IO.
"""

from __future__ import annotations

import json
import math
import re

import numpy as np
import torch

from ..tilemap_core import atlas, palette_extend, pixelize, plan_prompt, plan_schema, tileset_io
from ..tilemap_core import expand as tile_expand
from ..tilemap_core import snap as tsnap
from ._tensor_io import alpha_to_mask, image_list_to_np, mask_to_alpha, np_to_image

CATEGORY = "custom_comfy_nodes/tilemap"
_JOB_JSON_MAX = 400_000
_SEED_LIMIT = 2**64
_JOB_KEYS = ("tile_id", "category", "variant", "footprint")
_IMAGE_PIXELS_MAX = 2048 * 2048  # per Flux image (jobs are <= 1024 px a side); larger is rejected
_HEX = re.compile(r"#[0-9a-f]{6}")
_FIX = "reconnect job_json from CCN_TilemapExpand"


def _one(value: object, name: str) -> object:
    """INPUT_IS_LIST delivers every input as a list; scalar widgets are element 0."""
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"{name} is missing")
        return value[0]
    return value


def _num(value: object, name: str, lo: float, hi: float, real: bool = False) -> float:
    """Strict widget number: a real int (or finite int/float when ``real``), never a bool, inside ``lo..hi``."""
    ok = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) if real else type(value) is int
    if not ok or not lo <= value <= hi:
        raise ValueError(f"{name} must be {'a number' if real else 'an integer'} from {lo} to {hi}")
    return float(value) if real else value


def _load_json(text: object) -> dict:
    if not isinstance(text, str):
        raise ValueError(f"job_json must be text, got {type(text).__name__}; {_FIX}")
    if len(text) > _JOB_JSON_MAX:
        raise ValueError(f"job_json is too long ({len(text)} chars, limit {_JOB_JSON_MAX}); {_FIX}")
    try:
        data = json.loads(text, object_pairs_hook=tileset_io._no_dup, parse_int=tileset_io._int_text, parse_constant=tileset_io._reject_constant)
    except (ValueError, RecursionError):
        raise ValueError(f"job_json is not valid JSON; {_FIX}") from None
    if not isinstance(data, dict):
        raise ValueError(f"job_json must be a JSON object; {_FIX}")
    return data


def _signature(jobs: list) -> str:
    """Canonical text of the fields that decide tile identity; JSON keeps True != 1 and 1.0 != 1."""
    rows = [[j.get(k) if isinstance(j, dict) else None for k in _JOB_KEYS] for j in jobs]
    return json.dumps(rows, sort_keys=True)


def _key(j: object) -> str:
    return f"{j.get('tile_id')}#{j.get('variant')}" if isinstance(j, dict) else ""


def _item(j: dict) -> dict:
    return {"key": _key(j), "cells_w": j["footprint"][0], "cells_h": j["footprint"][1], "id": j["tile_id"],
            "name": j["name"], "category": j["category"], "variant": j["variant"], "wang": None}


def _check_base(base: object) -> object:
    """``job_json.base`` is ``None`` or exactly ``{path, keys, palette, snap}`` with well-formed values."""
    ok = base is None or isinstance(base, dict) and set(base) == {"path", "keys", "palette", "snap"}
    if ok and base:
        path, keys, pal = base["path"], base["keys"], base["palette"]
        ok = isinstance(path, str) and 1 <= len(path) <= 256 and isinstance(base["snap"], dict)
        ok = ok and isinstance(keys, list) and 1 <= len(keys) <= 512 and all(isinstance(k, str) for k in keys)
        ok = ok and len(set(keys)) == len(keys) and isinstance(pal, list) and 1 <= len(pal) <= 256
        ok = ok and all(isinstance(h, str) and _HEX.fullmatch(h) for h in pal)
    if not ok:
        raise ValueError(f"job_json base is malformed; {_FIX}")
    return base


def _parse_jobs(text: object) -> dict:
    """Parse an untrusted ``job_json``; ``jobs`` are the plan's re-derived jobs that need an image."""
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
    base = _check_base(data.get("base"))
    style, cell_px, model = data.get("style_prefix", ""), data.get("cell_px", 512), data.get("model", "")
    snap, review = data.get("snap", {}), data.get("review")
    if not isinstance(style, str) or type(cell_px) is not int or not isinstance(model, str) or not isinstance(snap, dict):
        raise ValueError(f"job_json style_prefix, cell_px, model or snap is invalid; {_FIX}")
    if review is not None and not isinstance(review, dict):
        raise ValueError(f"job_json review must be an object; {_FIX}")
    expected = tile_expand.expand_jobs(plan, size, view, "", 0)
    rows = data.get("jobs")
    got = {_key(r) for r in rows} if isinstance(rows, list) else set()
    kept = set(base["keys"]) if base else set()
    try:
        matches = isinstance(rows, list) and _signature(rows) == _signature([j for j in expected if _key(j) in got])
    except RecursionError:
        matches = False
    if not matches or any(_key(j) not in got and _key(j) not in kept for j in expected):
        raise ValueError(f"job_json jobs do not match its plan; {_FIX}")
    seed = data.get("seed")
    if seed is None and rows and type(rows[0].get("seed")) is int and type(rows[0].get("index", 0)) is int:
        seed = rows[0]["seed"] - rows[0].get("index", 0)  # job seed = base_seed + index (also after filter_new)
    if type(seed) is not int or not 0 <= seed < _SEED_LIMIT:
        raise ValueError(f"job_json job seed must be an integer from 0 to {_SEED_LIMIT - 1}; {_FIX}")
    full = tile_expand.expand_jobs(plan, size, view, style, seed, cell_px)
    return {"size": size, "view": view, "plan": plan, "base": base, "snap": snap, "review": review, "seed": seed,
            "style": style, "cell_px": cell_px, "model": model, "rows": {_key(r): r for r in rows},
            "want": {_key(j): j for j in expected}, "full": {_key(j): j for j in full},
            "jobs": [j for j in expected if _key(j) in got]}


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


def _works(rgbs: list, jobs: list, size: int, p: dict) -> tuple:
    """Work images and alpha grids of the new tiles, built as ``snap_tiles`` builds them (palette extension)."""
    works, alphas = [], []
    for rgb, job in zip(rgbs, jobs):
        work = pixelize.to_work(rgb, *job["footprint"], size)
        grid = (work.shape[0] // pixelize.WORK_SCALE, work.shape[1] // pixelize.WORK_SCALE)
        if job["category"] == "ground":
            work, alpha = pixelize.make_seamless(work, p["seam_blend"]), np.ones(grid, bool)
        else:
            alpha = pixelize.key_alpha(work, grid, p["key_tolerance"])
        works.append(work)
        alphas.append(alpha)
    return works, alphas


def _base_cells(P: dict, atlas_img: object, mask: object, spacing: int) -> tuple:
    """Slice the base atlas by replaying the pack of ``base.keys``; return ``(items, cells, columns)``."""
    n, plan = P["size"], P["plan"]
    descs = {k: _item(j) for k, j in P["want"].items()}
    dummy = {f"{t['id']}#0": np.full((n, n, 4), 255, np.uint8) for t in plan["tiles"]}  # Wang keys/shapes only (opaque ground)
    descs.update({w["key"]: w for w in tsnap.wang_items(plan, dummy, P["seed"], n)[0]})
    unknown = [k for k in P["base"]["keys"] if k not in descs]
    if unknown:
        raise ValueError(f"job_json base keys are not in its plan: {unknown[:5]}; {_FIX}")
    rgb = image_list_to_np(_one(atlas_img, "base_atlas")[:1], 3, True)[0]
    alpha = mask_to_alpha(_one(mask, "base_alpha"))
    if alpha.shape != rgb.shape[:2]:
        raise ValueError("base_alpha must have the same width and height as base_atlas")
    rgba, step = np.dstack([rgb, alpha]), n + spacing
    cols, rem = divmod(rgba.shape[1] + spacing, step)
    if rem or cols < 1:
        raise ValueError("base_atlas width does not fit tile_size and spacing")
    items = [descs[k] for k in P["base"]["keys"]]
    placed = atlas.pack(items, cols)
    cells = {}
    for p in placed:
        img = np.zeros((p["cells_h"] * n, p["cells_w"] * n, 4), np.uint8)
        for j in range(p["cells_h"]):
            for i in range(p["cells_w"]):
                y, x = (p["row"] + j) * step, (p["col"] + i) * step
                img[j * n : (j + 1) * n, i * n : (i + 1) * n] = rgba[y : y + n, x : x + n]
        cells[p["key"]] = img
    if not np.array_equal(atlas.assemble(placed, cells, n, cols, spacing), rgba) or any(c[..., 3].min() < 255 for k, c in cells.items() if descs[k]["category"] == "ground"):
        raise ValueError("base_atlas does not match the job_json base keys (layout or stray pixels)")
    return items, cells, cols


def _provenance(P: dict, items: list, new: set, bkeys: set) -> None:
    """Add prompt/seed/attempt/source/job_index to every item (Wang tiles: no prompt, the transition seed)."""
    trans = [list(t) for t in P["plan"]["transitions"]]
    for it in items:
        key, w = it["key"], it["wang"]
        r = {} if w else {**P["full"][key], **P["rows"].get(key, {})}
        it.update(prompt=r.get("prompt"), attempt=r.get("attempt", 0),
                  seed=(P["seed"] + trans.index([w["a"], w["b"]])) % _SEED_LIMIT if w else r["seed"],
                  job_index=None if w else P["want"][key]["index"],
                  source=("regenerated" if key in bkeys else "generated") if key in new else "kept")


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
            },
            "optional": {"base_atlas": ("IMAGE",), "base_alpha": ("MASK",)},
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("atlas", "alpha", "tiles", "metadata")
    FUNCTION = "assemble"
    CATEGORY = CATEGORY

    def assemble(
        self, images, job_json, palette_size, columns, spacing, seam_blend, key_tolerance, transitions,
        base_atlas=None, base_alpha=None,
    ) -> tuple:
        widgets = {"palette_size": _num(_one(palette_size, "palette_size"), "palette_size", 4, 64),
                   "seam_blend": _num(_one(seam_blend, "seam_blend"), "seam_blend", 0.0, 0.5, True),
                   "key_tolerance": _num(_one(key_tolerance, "key_tolerance"), "key_tolerance", 0, 255),
                   "transitions": _one(transitions, "transitions")}
        columns, spacing = _num(_one(columns, "columns"), "columns", 1, 64), _num(_one(spacing, "spacing"), "spacing", 0, 16)
        if not isinstance(widgets["transitions"], bool):
            raise ValueError("transitions must be a boolean")
        P = _parse_jobs(_one(job_json, "job_json"))
        base, size, plan, seed, jobs = P["base"], P["size"], P["plan"], P["seed"], P["jobs"]
        given = (base_atlas is not None) + (base_alpha is not None)
        if given == 1 or (given == 2) != (base is not None):
            raise ValueError("connect base_atlas and base_alpha together, and only when job_json has a base")
        p = tsnap.check_snap_params({**widgets, **(base["snap"] if base else {}), **P["snap"]})
        images = list(images) if isinstance(images, (list, tuple)) else [images]
        if len(images) != len(jobs):
            raise ValueError(f"got {len(images)} images for {len(jobs)} jobs; connect one image per job in job order")
        rgbs = [_image_rgb(img, i) for i, img in enumerate(images)]
        keep, got = p["palette_mode"] == "keep", {_key(j) for j in jobs}
        bitems, kept, bcols, bkeys, palette = [], {}, None, set(), None
        if base:
            bitems, bcells, bcols = _base_cells(P, base_atlas, base_alpha, spacing)
            bkeys = set(base["keys"])
            if keep:
                kept = {k: c for k, c in bcells.items() if k not in got}
                palette = np.array([[int(h[i : i + 2], 16) for i in (1, 3, 5)] for h in base["palette"]], np.uint8)
            elif any(k not in got for k in P["want"]):
                raise ValueError("palette_mode 'refit' needs an image for every tile of the plan")
        if jobs and palette is not None and p["extra_colours"]:
            palette = palette_extend.extend_palette(palette, *_works(rgbs, jobs, size, p), p["extra_colours"])
        cells, palette = tsnap.snap_tiles(rgbs, jobs, size, p, palette) if jobs else ({}, palette)
        cells.update(kept)
        adds = [j for j in jobs if _key(j) not in bkeys]
        if base:
            adds.sort(key=lambda j: (plan_schema.CATEGORIES.index(j["category"]), j["tile_id"], j["variant"]))
        items, new, wadds = bitems + [_item(j) for j in adds], set(got), []
        if p["transitions"]:
            witems, wimgs = tsnap.wang_items(plan, cells, seed, size)
            for w in witems:
                if w["key"] in kept and not {f"{w['wang']['a']}#0", f"{w['wang']['b']}#0"} & got:
                    continue  # pixels of an untouched base set are reused
                cells[w["key"]] = wimgs[w["key"]]
                new.add(w["key"])
                wadds += [w] if w["key"] not in bkeys else []
        items = [it for it in items + wadds if it["key"] in cells]
        _provenance(P, items, new, bkeys)
        placements = atlas.pack(items, columns)
        rgba = atlas.assemble(placements, cells, size, columns, spacing)
        extra = {"style_prefix": P["style"], "cell_px": P["cell_px"], "plan": plan, "snap": p}
        extra.update({} if P["review"] is None else {"review": P["review"]})
        if base:
            mode = "restyle" if not keep else "extend" if (adds or wadds) else "improve"
            extra["lineage"] = {"parent": base["path"], "mode": mode, "generation": 1, **({"repacked": True} if columns != bcols else {}),
                                **({} if keep else {"palette_refit": True})}
        meta = atlas.build_metadata(placements, size, columns, spacing, palette, P["view"], P["model"], seed, plan["theme"], extra)
        if len(meta["tiles"]) != len(items):
            raise RuntimeError(f"atlas has {len(meta['tiles'])} tiles, expected {len(items)}")
        atlas.verify_atlas(rgba, meta, palette, 256 if base else p["palette_size"])
        previews = tsnap.preview_cells(placements, cells, size)
        return (np_to_image([rgba]), alpha_to_mask(rgba[..., 3]), np_to_image(previews),
                json.dumps(meta, sort_keys=True, separators=(",", ":")))


NODE_CLASS_MAPPINGS = {"CCN_TilemapAssemble": CCN_TilemapAssemble}
NODE_DISPLAY_NAME_MAPPINGS = {"CCN_TilemapAssemble": "Tilemap Assemble (pixel-perfect atlas)"}
