"""Tilemap planning nodes: ``CCN_TilemapPlan`` (LLM tile plan) and ``CCN_TilemapExpand`` (Flux job lists).

``CCN_TilemapPlan`` asks the llm-proxy for a tile plan, validates it with the user's ``max_tiles`` and
caches it on disk. ``CCN_TilemapExpand`` turns the plan into per-job lists (prompt, width, height,
seed) that core ComfyUI nodes consume once per element, plus one ``job_json`` for the assemble node.

The API key never passes through these nodes: ``llm_client.chat`` loads it itself and redacts it from
its errors. No import-time IO or network.
"""

from __future__ import annotations

import json
import os
import tempfile

from ..tilemap_core import expand as tile_expand
from ..tilemap_core import llm_client, plan_cache, plan_prompt, plan_schema

try:  # ComfyUI provides folder_paths; a plain test harness does not
    import folder_paths
except ImportError:
    folder_paths = None

CATEGORY = "custom_comfy_nodes/tilemap"
CACHE_DIRNAME = "ccn_tilemap_cache"
MAX_TILES_RANGE = (1, 48)
LLM_MAX_TOKENS = 8000
LLM_TEMPERATURE = 0.4
LLM_TIMEOUT_S = 180
_ERR_CHARS = 300
_SEED_MAX = 2**64 - 1
_DEFAULT_MODEL = "claude-sonnet-5"


def _cache_dir() -> str:
    """Return the plan cache dir: ``<ComfyUI user dir>/ccn_tilemap_cache`` (temp dir without ComfyUI)."""
    getter = getattr(folder_paths, "get_user_directory", None)
    base = getter() if getter is not None else tempfile.gettempdir()
    return os.path.join(base, CACHE_DIRNAME)


def _int_in(value: object, name: str, lo: int, hi: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        raise ValueError(f"{name} must be an integer from {lo} to {hi}, got {value!r}")
    return value


def _tile_size(value: object) -> int:
    """Return ``value`` as a plain int from ``TILE_SIZES``; combo widgets may deliver text."""
    if isinstance(value, str) and value.strip().isascii() and value.strip().isdigit():
        value = int(value.strip())
    if type(value) is not int or value not in tile_expand.TILE_SIZES:
        raise ValueError(f"tile_size must be one of {list(tile_expand.TILE_SIZES)}, got {value!r}")
    return value


def _ask_plan(messages: list, model: str, max_tiles: int) -> dict:
    """Ask the LLM; on a schema failure ask exactly once more with the error text, then raise."""

    def call(msgs: list) -> str:
        return llm_client.chat(
            msgs,
            model,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            timeout=LLM_TIMEOUT_S,
        )

    reply = call(messages)
    try:
        return plan_schema.parse_and_validate(reply, max_tiles)
    except plan_schema.PlanError as first:
        first_msg = str(first)
    reply2 = call(plan_prompt.build_repair_messages(messages, reply, first_msg))
    try:
        return plan_schema.parse_and_validate(reply2, max_tiles)
    except plan_schema.PlanError as second:
        raise plan_schema.PlanError(
            "the LLM did not return a valid tile plan after one repair attempt. "
            f"First error: {first_msg[:_ERR_CHARS]} Second error: {str(second)[:_ERR_CHARS]}"
        ) from None


def _summary(plan: dict, tile_size: int, view: str, model: str, source: str) -> str:
    """Return a short human-readable plan overview."""
    tiles = plan["tiles"]
    lines = [
        f"Theme: {plan['theme']}",
        f"{len(tiles)} tiles, {sum(t['variants'] for t in tiles)} Flux jobs | {tile_size}px | "
        f"{view} | {model} | plan from {source}",
    ]
    for cat in plan_schema.CATEGORIES:
        names = [t["id"] + (f" x{t['variants']}" if t["variants"] > 1 else "") for t in tiles if t["category"] == cat]
        if names:
            lines.append(f"{cat}: {', '.join(names)}")
    if plan["transitions"]:
        lines.append("transitions: " + ", ".join(f"{a}<->{b}" for a, b in plan["transitions"]))
    return "\n".join(lines)


class CCN_TilemapPlan:
    """Plan a tilemap with the llm-proxy: validated JSON plan, cached on disk by (model, prompt, view, max_tiles)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "view": (list(plan_prompt.VIEWS), {"default": "top-down"}),
                "tile_size": (list(tile_expand.TILE_SIZES), {"default": 16}),
                "max_tiles": (
                    "INT",
                    {"default": 24, "min": MAX_TILES_RANGE[0], "max": MAX_TILES_RANGE[1], "step": 1},
                ),
                "model": (list(llm_client.MODELS), {"default": _DEFAULT_MODEL}),
                "model_override": ("STRING", {"default": ""}),
                "refresh_plan": ("BOOLEAN", {"default": False}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, refresh_plan=False, **_kwargs):
        # NaN never equals the previous value, so a forced refresh re-runs even with unchanged inputs.
        return float("nan") if refresh_plan else ""

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("plan_json", "summary")
    FUNCTION = "plan"
    CATEGORY = CATEGORY

    def plan(self, prompt, view, tile_size, max_tiles, model, model_override="", refresh_plan=False):
        size = _tile_size(tile_size)
        if view not in plan_prompt.VIEWS:
            raise ValueError(f"view must be one of {list(plan_prompt.VIEWS)}, got {view!r}")
        max_tiles = _int_in(max_tiles, "max_tiles", *MAX_TILES_RANGE)
        name = llm_client.validate_model(model_override.strip() or model)
        messages = plan_prompt.build_messages(prompt, view, max_tiles)  # also rejects an empty prompt
        key = plan_cache.cache_key(name, prompt, view, max_tiles)
        directory = _cache_dir()
        plan = None if refresh_plan else plan_cache.load(directory, key, max_tiles)
        source = "cache"
        if plan is None:
            source = "LLM"
            plan = _ask_plan(messages, name, max_tiles)
            try:
                plan_cache.store(directory, key, plan, max_tiles)
            except OSError:  # cache is an optimisation; an unwritable dir must not fail the run
                print("[custom_comfy_nodes] plan cache not writable, continuing without cache")
        else:
            print("[custom_comfy_nodes] plan cache hit")
        out = dict(plan)
        out["meta"] = {"tile_size": size, "view": view, "model": name}
        text = json.dumps(out, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return (text, _summary(plan, size, view, name, source))


class CCN_TilemapExpand:
    """Expand a tile plan into one Flux job per tile variant (list outputs) plus job_json for the assembler."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan_json": ("STRING", {"forceInput": True}),
                "style_prefix": ("STRING", {"default": "GRPZA, umempart, pixel art"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": _SEED_MAX}),
                "cell_px": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 64}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("prompt", "width", "height", "seed", "job_json")
    OUTPUT_IS_LIST = (True, True, True, True, False)
    FUNCTION = "expand"
    CATEGORY = CATEGORY

    def expand(self, plan_json, style_prefix, seed, cell_px):
        data = plan_schema.extract_json(plan_json)
        meta = data.get("meta")
        if not isinstance(meta, dict):
            raise ValueError("plan_json has no 'meta' block: connect the output of CCN_TilemapPlan")
        size = _tile_size(meta.get("tile_size"))
        view = meta.get("view")
        if not isinstance(view, str) or view not in plan_prompt.VIEWS:
            raise ValueError(f"plan_json meta.view must be one of {list(plan_prompt.VIEWS)}")
        # validate_plan drops unknown keys (meta) and caps variants at MAX_JOBS
        plan = plan_schema.validate_plan(data, plan_schema.MAX_JOBS)
        jobs = tile_expand.expand_jobs(plan, size, view, style_prefix, seed, cell_px)
        job_json = json.dumps(
            {"tile_size": size, "view": view, "plan": plan, "jobs": jobs},
            separators=(",", ":"),
        )
        return (
            [j["prompt"] for j in jobs],
            [j["width"] for j in jobs],
            [j["height"] for j in jobs],
            [j["seed"] for j in jobs],
            job_json,
        )


NODE_CLASS_MAPPINGS = {"CCN_TilemapPlan": CCN_TilemapPlan, "CCN_TilemapExpand": CCN_TilemapExpand}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CCN_TilemapPlan": "Tilemap Plan (LLM)",
    "CCN_TilemapExpand": "Tilemap Expand (Flux jobs)",
}
