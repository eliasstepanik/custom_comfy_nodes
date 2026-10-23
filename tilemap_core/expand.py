"""Expand a validated tile plan into one Flux job per tile variant.

Pure stdlib, deterministic, no I/O. ``expand_jobs`` re-validates the plan with
``plan_schema.validate_plan`` (so a plan edited after validation cannot sneak
bad values into a prompt) and never mutates its input.
"""

from . import sanitize
from .plan_prompt import VIEWS
from .plan_schema import CATEGORIES, validate_plan

__all__ = ["TILE_SIZES", "CATEGORY_SUFFIX", "expand_jobs"]

TILE_SIZES = (8, 16, 24, 32, 48, 64)

#: category -> trailing prompt clause that steers Flux towards a usable tile.
CATEGORY_SUFFIX = {
    "ground": "seamless tileable texture filling the whole frame, flat even lighting, "
    "no objects, no border, no shadow vignette",
    "object": "single isolated object centered, entire object visible, plain white background",
    "overlay": "small scattered detail sprite centered, plain white background",
}

SEED_LIMIT = 2**64
_STYLE_MAX = 200
_STEP = 16
_MIN_SIDE = 256
_CELL_PX_RANGE = (256, 1024)
_MAX_SIDE_RANGE = (_MIN_SIDE, 4096)


def _int_arg(value: object, name: str, lo: int, hi: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        raise ValueError(f"{name} must be an integer from {lo} to {hi}, got {value!r}")
    return value


def _job_size(cells_w: int, cells_h: int, cell_px: int, max_side: int) -> tuple:
    """Return (width, height): cell pixels scaled uniformly to ``max_side``, floored to 16, min 256."""
    width, height = cells_w * cell_px, cells_h * cell_px
    longest = max(width, height)
    if longest > max_side:  # integer math: same result on every platform
        width, height = width * max_side // longest, height * max_side // longest
    return (
        max(_MIN_SIDE, width // _STEP * _STEP),
        max(_MIN_SIDE, height // _STEP * _STEP),
    )


def expand_jobs(
    plan: dict,
    tile_size: int,
    view: str,
    style_prefix: str,
    base_seed: int,
    cell_px: int = 512,
    max_side: int = 1024,
) -> list:
    """Return the Flux job list for ``plan``: one job per tile variant (<= ``MAX_JOBS``).

    Order: ground tiles, objects, overlays (plan order inside each category),
    variants of one tile consecutive. Each job is
    ``{index, tile_id, name, category, variant, footprint:[w, h], tileable,
    prompt, width, height, seed}`` where ``variant`` counts from 0, ``seed`` is
    ``(base_seed + index) % 2**64`` and the prompt is
    ``"<style>, <VIEWS[view]>, <description>, <CATEGORY_SUFFIX[category]>"``
    (an empty style is omitted). ``width``/``height`` are ``footprint * cell_px``
    scaled uniformly down so the longer side is <= ``max_side``, each side
    floored to a multiple of 16 and at least 256.

    Raises ``ValueError`` for a ``tile_size`` outside ``TILE_SIZES``, an unknown
    ``view``, a non-integer / out-of-range ``base_seed`` (0..2**64-1),
    ``cell_px`` (256..1024) or ``max_side`` (256..4096); ``PlanError`` (a
    ``ValueError``) for an invalid plan or a plan with more than ``MAX_JOBS``
    variants in total (enforced by ``validate_plan``).
    """
    if type(tile_size) is not int or tile_size not in TILE_SIZES:
        raise ValueError(f"tile_size must be one of {list(TILE_SIZES)}, got {tile_size!r}")
    if not isinstance(view, str) or view not in VIEWS:
        raise ValueError(f"view must be one of {list(VIEWS)}, got {view!r}")
    base_seed = _int_arg(base_seed, "base_seed", 0, SEED_LIMIT - 1)
    cell_px = _int_arg(cell_px, "cell_px", *_CELL_PX_RANGE)
    max_side = _int_arg(max_side, "max_side", *_MAX_SIDE_RANGE)
    style = sanitize.clean_text(style_prefix, _STYLE_MAX)
    raw_tiles = plan.get("tiles") if isinstance(plan, dict) else None
    count = len(raw_tiles) if isinstance(raw_tiles, list) and raw_tiles else 1
    checked = validate_plan(plan, count)
    view_phrase = VIEWS[view]
    jobs: list = []
    for category in CATEGORIES:
        for tile in checked["tiles"]:
            if tile["category"] != category:
                continue
            cells_w, cells_h = tile["footprint"]
            width, height = _job_size(cells_w, cells_h, cell_px, max_side)
            parts = [style, view_phrase, tile["description"], CATEGORY_SUFFIX[category]]
            prompt = ", ".join(p for p in parts if p)
            for variant in range(tile["variants"]):
                index = len(jobs)
                jobs.append(
                    {
                        "index": index,
                        "tile_id": tile["id"],
                        "name": tile["name"],
                        "category": category,
                        "variant": variant,
                        "footprint": [cells_w, cells_h],
                        "tileable": category == "ground",
                        "prompt": prompt,
                        "width": width,
                        "height": height,
                        "seed": (base_seed + index) % SEED_LIMIT,
                    }
                )
    return jobs
