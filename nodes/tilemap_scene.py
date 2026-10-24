"""Scene node: ``CCN_TilemapScene`` renders a ``ccn-scene/1`` test scene from a tileset, with no review loop.

Lets a user look at any assembled or saved tileset (architecture-review-loop section 8). All logic lives in
``tilemap_core`` (``scene_layout`` for the layout, ``scene_render`` for the pixels, ``tileset_io`` for the
atlas); this module validates widgets, adapts the IMAGE tensors and builds the output text.

Tile input (``tiles``, an IMAGE list) has two accepted forms, both RGBA (4 channels, alpha only 0 or 255):

* the **atlas** (one image exactly ``width x height`` of the metadata, e.g. ``CCN_TilemapAssemble.atlas`` or
  ``CCN_LoadTilemap.atlas``): re-encoded and checked by ``tileset_io.load_tileset`` (``strict=False``: pixels
  off the palette are only a warning), then sliced;
* the **tile list** (one image per metadata tile, in metadata order, each ``footprint_h*n x footprint_w*n``,
  e.g. ``CCN_LoadTilemap.tiles``).

Layout priority: a non-empty ``scene_json`` is validated and rendered as given (a bad one raises
``ValueError``, it is never replaced silently); else ``layout_source`` = ``auto`` (deterministic) or ``llm``
(one proxy call, any failure falls back to the auto layout and is reported in ``layout_log``).
Every failure is a ``ValueError``; widgets are validated strictly, nothing is clamped. The torch import is
lazy, so the module imports without ComfyUI.
"""

from __future__ import annotations

import io
import json
import math
from collections.abc import Mapping

from PIL import Image

from ..tilemap_core import llm_client, sanitize, scene_layout, scene_render, tileset_io
from ..tilemap_core.atlas_meta import read_format
from ..tilemap_core.plan_prompt import VIEWS

CATEGORY = "custom_comfy_nodes/tilemap"
LAYOUT_SOURCES = ("auto", "llm")
SEED_MAX = 2**64 - 1
UPSCALE_DEFAULT = 4
SCENE_JSON_MAX = 100_000  # chars; a 32x32 scene is ~15k
LLM_MODEL = "claude-sonnet-5"  # the KB node surface has no model input; same default as CCN_TilemapPlan
LLM_MAX_TOKENS = 8000
LLM_TIMEOUT_S = 120
LOG_LINES_MAX = 20


def _one(value: object, name: str) -> object:
    """INPUT_IS_LIST delivers every input as a list; scalar widgets are element 0."""
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"{name} must be a single value, got {len(value)}")
        return value[0]
    return value


def _int_in(value: object, name: str, lo: int, hi: int) -> int:
    """Strict widget integer: a real int (no bool/float) inside ``lo..hi``."""
    if type(value) is not int or not lo <= value <= hi:
        raise ValueError(f"{name} must be an integer from {lo} to {hi}")
    return value


def _catalogue(meta: Mapping) -> dict:
    """Plan-shaped catalogue (``scene_layout``) from atlas metadata; wang records become transition pairs."""
    tiles, pairs = {}, []
    for i, t in enumerate(meta.get("tiles") or []):
        if not isinstance(t, Mapping) or not isinstance(t.get("id"), str):
            raise ValueError(f"metadata tiles[{i}] must be an object with a string id")
        wang = t.get("wang")
        if isinstance(wang, Mapping):
            pair = [wang.get("a"), wang.get("b")]
            if pair not in pairs:
                pairs.append(pair)
            continue
        entry = tiles.setdefault(t["id"], {"id": t["id"], "category": t.get("category"), "variants": 0, "footprint": t.get("footprint")})
        entry["variants"] += 1
    return {"tiles": list(tiles.values()), "transitions": pairs}


def _atlas_cells(rgba, text: str) -> tuple:
    """Atlas image -> ``(meta, cells, warnings)`` through the shared loader (verifies atlas against metadata)."""
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    ts = tileset_io.load_tileset(buf.getvalue(), text, strict=False)
    return ts.meta, ts.cells, ts.warnings


def _list_cells(arrs: list, meta: dict) -> dict:
    """One RGBA image per metadata tile (metadata order) -> ``{"<id>#<variant>": image}``; shapes checked."""
    tiles, n = meta.get("tiles"), meta.get("tile_size")
    if not isinstance(tiles, list) or not tiles or len(arrs) != len(tiles):
        raise ValueError(f"got {len(arrs)} tile images for {len(tiles) if isinstance(tiles, list) else 0} metadata tiles; "
                         "connect the atlas image, or one image per tile in metadata order")
    cells = {}
    for i, (t, img) in enumerate(zip(tiles, arrs)):
        key, fp = tileset_io._tile_key(t), t.get("footprint")
        if not (isinstance(fp, list) and len(fp) == 2 and all(type(v) is int and v >= 1 for v in fp)) or type(n) is not int:
            raise ValueError(f"metadata tiles[{i}] needs an integer footprint [w, h] and the metadata a tile_size")
        if key in cells or img.shape != (fp[1] * n, fp[0] * n, 4):
            raise ValueError(f"tile image {i} for {key[:60]!r} must be {fp[0] * n}x{fp[1] * n} px RGBA (duplicate keys are refused)")
        cells[key] = img
    return cells


def _load(tiles: object, text: str) -> tuple:
    from ._tensor_io import image_list_to_np  # torch is imported here, not at module import

    arrs = image_list_to_np(tiles, channels=4)
    meta = tileset_io._parse_meta(text)
    read_format(meta)
    if len(arrs) == 1 and arrs[0].shape[:2] == (meta.get("height"), meta.get("width")):
        return _atlas_cells(arrs[0], text)
    return meta, _list_cells(arrs, meta), ()


def _layout(source: str, given: str, catalogue: dict, view: str, seed: int) -> tuple:
    """Return ``(scene, layout_source, problem, unplaced)``; ``layout_source`` is given|auto|llm|auto_fallback."""
    if given:
        try:
            obj = json.loads(given)
        except (ValueError, RecursionError):
            raise ValueError("scene_json is not valid JSON") from None
        return scene_layout.validate_scene(obj, catalogue), "given", "", []
    if source == "llm":
        try:
            reply = llm_client.chat(scene_layout.build_scene_messages(catalogue, view), LLM_MODEL,
                                    max_tokens=LLM_MAX_TOKENS, timeout=LLM_TIMEOUT_S)
        except llm_client.LLMProxyError as exc:
            problem = sanitize.clean_text(str(exc), 300)
        else:
            scene, src, problem = scene_layout.parse_scene(reply, catalogue, view, seed)
            if src == "llm":
                return scene, src, "", []
        return scene_layout.auto_layout(catalogue, view, seed)[0], "auto_fallback", problem, []
    scene, unplaced = scene_layout.auto_layout(catalogue, view, seed)
    return scene, "auto", "", unplaced


def _log(source: str, problem: str, unplaced: list, warnings: list) -> str:
    lines = [f"layout_source: {source}"]
    if problem:
        lines.append(f"layout problem: {problem}")
    if unplaced:
        lines.append(f"unplaced: {', '.join(unplaced)}")
    lines += [f"warning: {w}" for w in warnings[:LOG_LINES_MAX]]
    if len(warnings) > LOG_LINES_MAX:
        lines.append(f"... {len(warnings) - LOG_LINES_MAX} more warnings")
    return "\n".join(lines)


class CCN_TilemapScene:
    """Render a test scene (ground, transitions, objects, overlays) from a tileset atlas or tile list."""

    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "tiles": ("IMAGE",),
                "metadata": ("STRING", {"forceInput": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": SEED_MAX, "step": 1}),
                "layout_source": (list(LAYOUT_SOURCES), {"default": "auto"}),
                "upscale": ("INT", {"default": UPSCALE_DEFAULT, "min": 1, "max": scene_render.UPSCALE_MAX, "step": 1}),
            },
            "optional": {"scene_json": ("STRING", {"multiline": True, "default": ""})},
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("scene", "scene_json", "layout_log")
    FUNCTION = "render"
    CATEGORY = CATEGORY

    def render(self, tiles: list, metadata: list, seed: list, layout_source: list, upscale: list, scene_json: list = None) -> tuple:
        seed = _int_in(_one(seed, "seed"), "seed", 0, SEED_MAX)
        scale = _int_in(_one(upscale, "upscale"), "upscale", 1, scene_render.UPSCALE_MAX)
        source = _one(layout_source, "layout_source")
        if source not in LAYOUT_SOURCES:
            raise ValueError(f"layout_source must be one of {list(LAYOUT_SOURCES)}")
        given = "" if scene_json is None else _one(scene_json, "scene_json")
        if not isinstance(given, str) or len(given) > SCENE_JSON_MAX:
            raise ValueError(f"scene_json must be text of at most {SCENE_JSON_MAX} characters")
        text = _one(metadata, "metadata")
        if not isinstance(text, str):
            raise ValueError("metadata must be text")
        from ._tensor_io import np_to_image  # torch is imported here, not at module import

        meta, cells, tile_warnings = _load(tiles, text)
        view = meta.get("view")
        if view not in VIEWS:
            raise ValueError(f"metadata view must be one of {sorted(VIEWS)}")
        catalogue = _catalogue(meta)
        scene, src, problem, unplaced = _layout(source, given.strip(), catalogue, view, seed)
        rgba, warnings = scene_render.render_scene(scene, cells, meta, meta["tile_size"], scale)
        return (
            np_to_image([rgba[..., :3]]),
            json.dumps(scene, sort_keys=True, separators=(",", ":"), allow_nan=False),
            _log(src, problem, unplaced, list(tile_warnings) + warnings),
        )


NODE_CLASS_MAPPINGS = {"CCN_TilemapScene": CCN_TilemapScene}
NODE_DISPLAY_NAME_MAPPINGS = {"CCN_TilemapScene": "Tilemap Scene Preview"}
