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
LLM_TIMEOUT_S = 180
_ERR_CHARS = 300
_REPLY_CHARS = 200
_SEED_MAX = 2**64 - 1
_DEFAULT_MODEL = "claude-sonnet-5"
PLAN_MODES = ("new", "extend")
_BASE_KEYS = {"path", "keys", "palette", "snap"}


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


def _reply_excerpt(reply: str) -> str:
    """Return ``reply`` on one line (non-printable chars and whitespace runs -> one space), max 200 chars."""
    flat = "".join(ch if ch.isprintable() else " " for ch in reply)
    return " ".join(flat.split())[:_REPLY_CHARS].rstrip()


def _ask_plan(messages: list, model: str, max_tiles: int, existing: dict | None = None):
    """Ask the LLM; on a schema failure ask exactly once more with the error text, then raise.

    Only ``PlanError`` triggers the repair; ``LLMProxyError`` from ``chat`` propagates at once.
    ``temperature`` is not sent: several models (claude-sonnet-5, opus-5) reject it. With ``existing`` the
    result is ``(merged_plan, new_keys)`` from ``validate_extension``.
    """

    def parse(text: str):
        if existing is None:
            return plan_schema.parse_and_validate(text, max_tiles)
        return plan_schema.validate_extension(existing, plan_schema.extract_json(text), max_tiles)

    def call(msgs: list) -> str:
        return llm_client.chat(msgs, model, max_tokens=LLM_MAX_TOKENS, timeout=LLM_TIMEOUT_S)

    reply = call(messages)
    try:
        return parse(reply)
    except plan_schema.PlanError as first:
        first_msg = str(first)
    reply2 = call(plan_prompt.build_repair_messages(messages, reply, first_msg))
    try:
        return parse(reply2)
    except plan_schema.PlanError as second:
        raise plan_schema.PlanError(
            "the LLM did not return a valid tile plan after one repair attempt. "
            f"First error: {first_msg[:_ERR_CHARS]} Second error: {str(second)[:_ERR_CHARS]} "
            f"Reply start: {_reply_excerpt(reply2)}"
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


def _check_base(base: object) -> dict:
    """Return ``base`` (a ``job_json.base`` block: path, keys, palette, snap) or raise ``ValueError``."""
    if not isinstance(base, dict) or set(base) != _BASE_KEYS or not isinstance(base["keys"], list):
        raise ValueError(f"base must be an object with exactly the keys {sorted(_BASE_KEYS)} and a keys list")
    return base


def _existing(text: object, size: int, view: str) -> tuple:
    """Return ``(plan, base or None)`` from ``existing_plan``: a ``CCN_LoadTilemap`` job_json or a bare plan."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("plan_mode extend needs existing_plan: connect CCN_LoadTilemap job_json")
    data = plan_schema.extract_json(text)
    inner = data.get("plan")
    if not isinstance(inner, dict):
        return plan_schema.validate_plan(data, plan_schema.MAX_JOBS), None
    for name, want in (("tile_size", size), ("view", view)):
        if name in data and data[name] != want:
            raise ValueError(f"existing_plan {name} is {data[name]!r}, the node is set to {want!r}")
    return plan_schema.validate_plan(inner, plan_schema.MAX_JOBS), _check_base(data["base"]) if "base" in data else None


def _keys(plan: dict) -> list:
    """Return the ``id#variant`` keys of ``plan`` in job order (category, tile, variant)."""
    return [
        f"{t['id']}#{v}" for c in plan_schema.CATEGORIES for t in plan["tiles"] if t["category"] == c for v in range(t["variants"])
    ]


def _extension_jobs(jobs: list, ext: object) -> tuple:
    """Keep only the jobs of the extension block ``plan_json.meta.extend``; return ``(new jobs, base or None)``."""
    if not isinstance(ext, dict) or set(ext) - {"new_keys", "base"} or not isinstance(ext.get("new_keys"), list):
        raise ValueError("plan_json meta.extend must be an object with new_keys (list) and optional base")
    new = set(ext["new_keys"])
    base = _check_base(ext["base"]) if "base" in ext else None
    if base is not None:
        held = set(base["keys"])
        rest = {f"{j['tile_id']}#{j['variant']}" for j in jobs} - new
        if held & new or rest - held:
            raise ValueError("base.keys must hold every plan tile key that is not in new_keys, and none of new_keys")
    return tile_expand.filter_new(jobs, ext["new_keys"]), base


class CCN_TilemapPlan:
    """Plan a tilemap with the llm-proxy: validated JSON plan, cached on disk by (model, prompt, view, max_tiles).

    ``plan_mode`` extend returns the plan merged with an LLM addition to ``existing_plan`` (a ``CCN_LoadTilemap``
    job_json); ``meta.extend`` carries the new keys and the loaded ``base`` for ``CCN_TilemapExpand``."""

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
            },
            "optional": {
                "plan_mode": (list(PLAN_MODES), {"default": "new"}),
                "existing_plan": ("STRING", {"forceInput": True, "default": ""}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, refresh_plan=False, **_kwargs):
        # NaN never equals the previous value, so a forced refresh re-runs even with unchanged inputs.
        return float("nan") if refresh_plan else ""

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("plan_json", "summary", "model_used")
    FUNCTION = "plan"
    CATEGORY = CATEGORY

    def plan(self, prompt, view, tile_size, max_tiles, model, model_override="", refresh_plan=False, plan_mode="new", existing_plan=""):
        size = _tile_size(tile_size)
        if view not in plan_prompt.VIEWS:
            raise ValueError(f"view must be one of {list(plan_prompt.VIEWS)}, got {view!r}")
        max_tiles = _int_in(max_tiles, "max_tiles", *MAX_TILES_RANGE)
        name = llm_client.validate_model(model_override.strip() or model)
        if plan_mode not in PLAN_MODES:
            raise ValueError(f"plan_mode must be one of {list(PLAN_MODES)}, got {plan_mode!r}")
        existing = base = new_keys = None
        if plan_mode == "extend":
            existing, base = _existing(existing_plan, size, view)
            if len(existing["tiles"]) > max_tiles:
                raise ValueError(f"max_tiles {max_tiles} is below the {len(existing['tiles'])} existing tiles")
            messages = plan_prompt.build_extend_messages(prompt, view, existing)
        elif isinstance(existing_plan, str) and existing_plan.strip():
            raise ValueError("existing_plan is only used with plan_mode extend")
        else:
            messages = plan_prompt.build_messages(prompt, view, max_tiles)  # also rejects an empty prompt
        ehash = "" if existing is None else plan_cache.plan_hash(existing)
        key = plan_cache.cache_key(name, prompt, view, max_tiles, ehash)
        directory = _cache_dir()
        plan = None if refresh_plan else plan_cache.load(directory, key, max_tiles)
        if plan is not None and existing is not None:
            old = set(_keys(existing))
            new_keys = [k for k in _keys(plan) if k not in old]
            if not new_keys or not old <= set(_keys(plan)):  # a cache file that adds nothing or drops base tiles is a miss
                plan = None
        source = "cache"
        if plan is None:
            source = "LLM"
            plan = _ask_plan(messages, name, max_tiles, existing)
            if existing is not None:
                plan, new_keys = plan
            try:
                plan_cache.store(directory, key, plan, max_tiles)
            except OSError:  # cache is an optimisation; an unwritable dir must not fail the run
                print("[custom_comfy_nodes] plan cache not writable, continuing without cache")
        else:
            print("[custom_comfy_nodes] plan cache hit")
        out = dict(plan)
        out["meta"] = {"tile_size": size, "view": view, "model": name}
        summary = _summary(plan, size, view, name, source)
        if existing is not None:
            out["meta"]["extend"] = {"new_keys": new_keys, **({} if base is None else {"base": base})}
            summary += f"\nextension: {len(new_keys)} new Flux jobs"
        text = json.dumps(out, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return (text, summary, name)


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
            },
            "optional": {"model": ("STRING", {"forceInput": True, "default": ""})},
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("prompt", "width", "height", "seed", "job_json")
    OUTPUT_IS_LIST = (True, True, True, True, False)
    FUNCTION = "expand"
    CATEGORY = CATEGORY

    def expand(self, plan_json, style_prefix, seed, cell_px, model=""):
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
        if not isinstance(model, str):
            raise ValueError(f"model must be text, got {type(model).__name__}")
        jobs = tile_expand.expand_jobs(plan, size, view, style_prefix, seed, cell_px)
        base = None
        if "extend" in meta:  # the plan is the merged one: keep only the new jobs, re-indexed after the base
            jobs, base = _extension_jobs(jobs, meta["extend"])
        job = {"tile_size": size, "view": view, "plan": plan, "jobs": jobs, **({} if base is None else {"base": base})}
        if model.strip():
            job["model"] = llm_client.validate_model(model.strip())
        job_json = json.dumps(job, separators=(",", ":"))
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
