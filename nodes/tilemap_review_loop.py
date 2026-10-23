"""``CCN_TilemapReviewLoop``: review the first-pass tile images in a test scene and regenerate the bad ones.

Per round (at most ``max_rounds``): snap every raw image (``tilemap_core.snap``), render a test scene and a
labelled tile grid, review them (heuristics always, plus a vision verdict through the LLM proxy unless
``review_mode`` is ``heuristic``), then regenerate the worst tiles (``regen_plan`` + ``render_jobs``). The
loop stops when the review passes, when no tile can be regenerated, or after ``max_rounds``. Per tile the
best reviewed version is kept (``regen_plan.best_version``); a tile regenerated in the last round is never
reviewed again and is logged ``unreviewed``. ``max_rounds = 0`` returns the first-pass images (and ``job_json``)
unchanged and only renders the scene. LLM calls (review, one repair per review, LLM scene layout) are capped
at ``2 * max_rounds``; a vision failure falls back to the heuristics (``on_review_error = keep``) or raises
(``strict``). Widgets are validated strictly (``ValueError``), nothing is clamped. ComfyUI/torch imports are lazy;
interrupts propagate. The review log holds verdict counters only: no images, request bodies or keys.
"""


from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace

import numpy as np
from PIL import Image

from ..tilemap_core import (heuristics, llm_client, pixelize, regen_plan, review, sanitize, scene_layout,
                            scene_render, snap)
from . import _loop_state as ls
from .tilemap_regen import render_jobs

CATEGORY = "custom_comfy_nodes/tilemap"
_ID_MAX = 32  # review.py id slug limit


def _layout(source: str, llm: ls.LlmBudget, catalogue: dict, view: str, seed: int) -> tuple:
    """``(scene, layout_source, problem)``; an LLM layout that fails falls back to the auto layout."""
    problem = ""
    if source == "llm":
        try:
            scene, src, problem = scene_layout.parse_scene(
                llm.ask(scene_layout.build_scene_messages(catalogue, view), ls.LAYOUT_TOKENS), catalogue, view, seed)
            if src == "llm":
                return scene, "llm", ""
        except llm_client.LLMProxyError as exc:
            problem = sanitize.clean_text(str(exc), 200)
    return scene_layout.auto_layout(catalogue, view, seed)[0], "auto" if source == "auto" else "auto_fallback", problem


def _scene(cur: dict, jobs: list, plan: dict, size: int, seed: int, snap_p: dict, wang: bool, scene: dict) -> tuple:
    """Snap ``cur`` and render the scene; return ``(cells, rows, rgba)``. ``rows`` are atlas-metadata-shaped tiles."""
    cells, _ = snap.snap_tiles([cur[ls.job_key(j)] for j in jobs], jobs, size, snap_p)
    rows = [{"id": j["tile_id"], "variant": j["variant"], "category": j["category"], "footprint": j["footprint"],
             "wang": None} for j in jobs]
    if wang:
        items, images = snap.wang_items(plan, cells, seed, size)
        cells.update(images)
        rows += [{"id": it["id"], "variant": it["variant"], "category": "ground", "footprint": [1, 1], "wang": it["wang"]}
                 for it in items]
    rgba, _warnings = scene_render.render_scene(scene, cells, {"tiles": rows}, size)
    return cells, rows, rgba


def _grid(rows: list, cells: dict, size: int) -> np.ndarray:
    """One labelled picture per tile row (transparent pixels grey), padded to the largest footprint."""
    h, w = (max(cells[f"{r['id']}#{r['variant']}"].shape[a] for r in rows) for a in (0, 1))
    pics = []
    for r in rows:
        cell = cells[f"{r['id']}#{r['variant']}"]
        pic = np.full((h, w, 3), 128, np.uint8)
        pic[: cell.shape[0], : cell.shape[1]] = np.where(cell[..., 3:] > 0, cell[..., :3], 128)
        pics.append(pic)
    cols = max(1, min(8, review.IMAGE_MAX_SIDE // (w + 2)))
    return scene_render.label_grid(pics, min(cols, 16), 2, "gray", True)


def _fit(img: np.ndarray) -> np.ndarray:
    """Box-downscale to the review image limit when needed."""
    side = max(img.shape[:2])
    if side <= review.IMAGE_MAX_SIDE:
        return img
    f = review.IMAGE_MAX_SIDE / side
    return np.asarray(Image.fromarray(img).resize((max(1, int(img.shape[1] * f)), max(1, int(img.shape[0] * f))), Image.BOX))


def _heuristics(cur: dict, jobs: list, size: int, tol: int) -> list:
    works, alphas = {}, {}
    for job in jobs:
        key, (cw, ch) = ls.job_key(job), job["footprint"]
        works[key] = work = pixelize.to_work(cur[key], cw, ch, size)
        if job["category"] != "ground":
            grid = (work.shape[0] // pixelize.WORK_SCALE, work.shape[1] // pixelize.WORK_SCALE)
            alphas[key] = pixelize.key_alpha(work, grid, tol)
    return heuristics.heuristic_issues(works, alphas, jobs)


def _verdict(llm: ls.LlmBudget, messages: list, known: frozenset, pass_score: int) -> dict:
    reply = llm.ask(messages)
    try:
        return review.parse_verdict(reply, known, pass_score)
    except review.ReviewError as exc:
        return review.parse_verdict(llm.ask(review.build_review_repair_messages(messages, reply, str(exc))), known, pass_score)


def _review(o: SimpleNamespace, llm: ls.LlmBudget, cur: dict, jobs: list, rows: list, cells: dict, rgba: np.ndarray,
            size: int, view: str, tol: int) -> tuple:
    """Return ``(issues, verdict or None, info)``: heuristics always, vision unless ``heuristic`` mode."""
    found, verdict, info = _heuristics(cur, jobs, size, tol), None, {"review_mode": "heuristic"}
    if o.review_mode != "heuristic":
        shown = [r for r in rows if len(r["id"]) <= _ID_MAX]
        cat = [{"id": r["id"], "variant": r["variant"], "category": r["category"], "footprint": r["footprint"],
                "tileable": r["category"] == "ground", "index": i + 1} for i, r in enumerate(shown)]
        try:
            msgs = review.build_review_messages(cat, _fit(rgba[..., :3]), _fit(_grid(shown, cells, size)), view)
            verdict = _verdict(llm, msgs, review.tile_refs(cat), o.pass_score)
            info["review_mode"] = o.review_mode
        except (llm_client.LLMProxyError, review.ReviewError) as exc:
            reason = sanitize.clean_text(str(exc), 200)
            if o.on_review_error == "strict":
                raise ValueError(f"vision review failed (on_review_error=strict): {reason}") from None
            info["fallback"] = reason
    return review.merge_issues(verdict["issues"] if verdict else [], found), verdict, info


def review_loop(model: object, clip: object, vae: object, text: str, images: list | None, o: SimpleNamespace,
                chat=None, render=None) -> tuple:
    """Run the loop; return ``(images, job_json, scene rgb, review_log)``.

    ``images``: uint8 ``HxWx3`` arrays in job order, or ``None`` to render the first pass with ``render``
    (default ``render_jobs``). Untouched tiles come back as the very same objects. ``chat`` defaults to
    ``llm_client.chat``. ``o`` carries the validated widgets (see ``_loop_state.parse_options``).
    """
    chat, render = chat or llm_client.chat, render or render_jobs
    data, size, view, plan, jobs, seed = ls.parse_job_json(text)
    keys = [ls.job_key(j) for j in jobs]
    base = dict(zip(keys, jobs))
    if images is None:
        cur = dict(zip(keys, (ls.quantize(i) for i in render(model, clip, vae, jobs, o.params, o.negative))))
    elif len(images) != len(jobs):
        raise ValueError(f"got {len(images)} images for {len(jobs)} jobs; connect one image per job in job order")
    else:
        cur = dict(zip(keys, images))
    given = data.get("snap") or {}
    if not isinstance(given, Mapping):
        raise ValueError("job_json snap must be an object")
    snap_p = snap.check_snap_params({**o.snap, **given})
    trans = plan["transitions"] if o.transitions else []
    llm = ls.LlmBudget(chat, o.review_model, 2 * o.max_rounds)
    catalogue = {"tiles": [{k: t[k] for k in ("id", "category", "variants", "footprint") if k in t} for t in plan["tiles"]],
                 "transitions": trans}
    scene, layout_src, layout_problem = _layout(o.layout_source, llm, catalogue, view, seed)
    attempts, best, last, rounds, rgba, stop = {}, {}, [], [], None, "no_rounds"
    for rnd in range(1, o.max_rounds + 1):
        cells, rows, rgba = _scene(cur, jobs, plan, size, seed, snap_p, bool(trans), scene)
        issues, verdict, info = _review(o, llm, cur, jobs, rows, cells, rgba, size, view, snap_p["key_tolerance"])
        weights = regen_plan.key_weights(issues, base, trans) if issues else {}
        reverted = []
        for key in last:  # keep-best: a regenerated tile stays only when its weight is strictly lower
            win, replaced = regen_plan.best_version(best.get(key), (cur[key], weights.get(key, 0)))
            if not replaced:
                cur[key] = win[0]
                reverted.append(key)
            best[key] = win
        for key in keys:
            best.setdefault(key, (cur[key], weights.get(key, 0)))
        entry = ls.round_entry(rnd, info, issues, verdict, reverted)
        rounds.append(entry)
        if ls.is_passed(verdict, issues, o.pass_score) and not reverted:
            stop, last = "passed", []
            break
        new_jobs, entry["skipped"] = regen_plan.plan_regen(issues, base, attempts, o.max_regen, o.max_attempts, rnd,
                                                           transitions=trans) if issues else ([], [])
        if not new_jobs:
            stop, last = "no_jobs", []
            break
        fresh = render(model, clip, vae, new_jobs, o.params, o.negative)
        attempts = regen_plan.record_attempts(attempts, new_jobs)
        snap_p = regen_plan.snap_for_issues(issues, snap_p)
        for job, img in zip(new_jobs, fresh):
            cur[ls.job_key(job)] = ls.quantize(img)
        last = [ls.job_key(j) for j in new_jobs]
        entry["regen"] = sorted(last)
        stop = "max_rounds"
    if rgba is None:
        rgba = _scene(cur, jobs, plan, size, seed, snap_p, bool(trans), scene)[2]
    out = ls.build_job_json(text, data, snap_p, len(rounds), attempts, stop, o.max_rounds)
    log = ls.build_log(stop, llm.calls, layout_src, layout_problem, last, rounds)
    return [cur[k] for k in keys], out, np.ascontiguousarray(rgba[..., :3]), log


def _combo(attr: str, default: str) -> list:
    try:  # ComfyUI's own sampler / scheduler names; a plain list when ComfyUI is not importable
        from comfy import samplers

        return list(getattr(samplers.KSampler, attr))
    except ImportError:
        return [default]


class CCN_TilemapReviewLoop:
    """Review the first-pass tiles in a test scene and regenerate the worst ones (bounded rounds)."""

    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "model": ("MODEL",), "clip": ("CLIP",), "vae": ("VAE",),
                "job_json": ("STRING", {"forceInput": True}),
                "review_mode": (list(ls.REVIEW_MODES), {"default": ls.REVIEW_MODES[0]}),
                "review_model": (list(llm_client.MODELS), {"default": llm_client.MODELS[0]}),
                "max_rounds": ("INT", {"default": 3, "min": 0, "max": regen_plan.MAX_ROUND, "step": 1}),
                "max_regen_per_round": ("INT", {"default": 8, "min": 1, "max": 32, "step": 1}),
                "max_attempts": ("INT", {"default": 3, "min": 1, "max": 5, "step": 1}),
                "pass_score": ("INT", {"default": review.PASS_SCORE_DEFAULT, "min": 0, "max": 100, "step": 1}),
                "layout_source": (list(ls.LAYOUT_SOURCES), {"default": "auto"}),
                "palette_size": ("INT", {"default": 24, "min": 4, "max": 64, "step": 1}),
                "seam_blend": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 0.5, "step": 0.05}),
                "key_tolerance": ("INT", {"default": 40, "min": 0, "max": 255, "step": 1}),
                "transitions": ("BOOLEAN", {"default": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler": (_combo("SAMPLERS", "euler"), {"default": "euler"}),
                "scheduler": (_combo("SCHEDULERS", "simple"), {"default": "simple"}),
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
                "negative": ("STRING", {"multiline": True, "default": ""}),
                "on_review_error": (list(ls.ON_ERROR), {"default": "keep"}),
            },
            "optional": {"images": ("IMAGE",), "base_atlas": ("IMAGE",), "base_alpha": ("MASK",)},
        }

    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("images", "job_json", "scene", "review_log")
    OUTPUT_IS_LIST = (True, False, False, False)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, **kw: list) -> tuple:
        ls.reject_base_inputs(kw)
        o = ls.parse_options(kw)
        from ._tensor_io import image_list_to_np, np_to_image  # torch is imported here, not at module import

        given = kw.get("images")
        arrays = None if given is None else image_list_to_np(given, 3, True)
        out, text, scene, log = review_loop(ls.one(kw["model"], "model"), ls.one(kw["clip"], "clip"), ls.one(kw["vae"], "vae"),
                                            ls.one(kw["job_json"], "job_json"), arrays, o)
        return [np_to_image([a]) for a in out], text, np_to_image([scene]), log


NODE_CLASS_MAPPINGS = {"CCN_TilemapReviewLoop": CCN_TilemapReviewLoop}
NODE_DISPLAY_NAME_MAPPINGS = {"CCN_TilemapReviewLoop": "Tilemap Review Loop (review and regenerate)"}