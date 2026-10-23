"""Defect-to-fix table for the review loop: what a regenerated tile changes per defect kind.

Pure stdlib, deterministic, no I/O, no randomness. Prompt text goes through ``sanitize.clean_text``.

``DEFECT_FIXES[kind] = {"prompt": str, "negative": str, "snap": dict}`` has one entry for every ``kind`` of
the ``ccn-review/1`` contract (``DEFECT_KINDS``). What each column does to a regeneration:

* ``prompt`` - positive corrective phrase appended to the tile's prompt (``amend_job``). This is the only
  column that changes pixels: the shipped sampler runs ``cfg 1.0``, so Flux ignores negative conditioning.
* ``negative`` - **inert** while ``cfg == 1.0``. Kept for a future ``cfg > 1`` route (``negative_text``).
* ``snap`` - overrides for the ``job_json.snap`` block (keys of ``snap.SNAP_DEFAULTS``) applied to the
  whole atlas, not per tile (``snap_overrides``). Only ``fringe`` has one: a higher ``key_tolerance``
  keys more of the light background out, which removes bright edge pixels.
* seed shift - not stored here: ``regen_plan`` sets ``seed = base_job.seed + attempt_no * 10007``.
* steps / cfg / sampler - never changed by a defect kind; they stay the loop node's widgets.

The heuristic review (``heuristics.FIX_HINTS``) emits a subset of these kinds (``ground_not_texture``,
``not_tileable``, ``not_cut_out``, ``wrong_content``, ``fringe``); the vision review may emit all of them.
"""

import copy
from typing import Iterable, Mapping

from . import sanitize

__all__ = [
    "DEFECT_KINDS", "DEFECT_FIXES", "PROMPT_MAX", "HINT_MAX",
    "amendment", "negative_text", "snap_overrides", "amend_job",
]

#: kinds of the ``ccn-review/1`` contract, in the canonical order used for deterministic output.
DEFECT_KINDS = (
    "style_mismatch", "seam", "fringe", "wrong_content", "not_cut_out", "not_tileable",
    "wrong_scale", "transition", "ground_not_texture", "other",
)
PROMPT_MAX = 2000  # amended prompt length cap (characters, via clean_text)
HINT_MAX = 200  # review ``fix_hint`` cap; longer hints are rejected, not cut

DEFECT_FIXES = {
    "style_mismatch": {
        "prompt": "strictly the same pixel art style, flat limited colour palette and hard outlines as the rest of the set",
        "negative": "photo, realistic, 3d render, smooth gradients, blurry, soft shading",
        "snap": {},
    },
    "seam": {
        "prompt": "uniform texture without any visible seam, line or brightness step across the tile",
        "negative": "seam, visible edge, line, border, frame",
        "snap": {},
    },
    "fringe": {
        "prompt": "clean crisp outline with no bright halo, no glow and no light edge pixels around the object",
        "negative": "halo, glow, bright outline, white edge, anti-aliased fringe",
        "snap": {"key_tolerance": 60},
    },
    "wrong_content": {
        "prompt": "the described subject large and clearly recognisable, centred and filling most of the frame",
        "negative": "empty frame, tiny subject, unrelated objects, text",
        "snap": {},
    },
    "not_cut_out": {
        "prompt": "only the single object on a plain flat white background with empty space all around it",
        "negative": "background scenery, ground, floor, shadow on background, cropped object",
        "snap": {},
    },
    "not_tileable": {
        "prompt": "texture continuous across all four edges so it repeats without a visible seam, no centred subject",
        "negative": "vignette, border, frame, centred object, edge shading, gradient across the tile",
        "snap": {},
    },
    "wrong_scale": {
        "prompt": "correct scale, with details sized to fit a small game tile consistent with the other tiles",
        "negative": "huge details, tiny details, mismatched scale, close-up macro",
        "snap": {},
    },
    "transition": {
        "prompt": "ground texture filling the whole frame edge to edge, continuous across all four edges, "
        "no flat background and no isolated blobs",
        "negative": "blob, cut-out shape, flat background, border, frame, vignette",
        "snap": {},
    },
    "ground_not_texture": {
        "prompt": "texture filling the whole frame edge to edge with a repeating ground pattern, "
        "no flat background colour, no isolated blob or cut-out shape",
        "negative": "blob, cut-out shape, flat cream background, isolated object, border, vignette",
        "snap": {},
    },
    "other": {
        "prompt": "clean, simple, game-ready pixel art tile",
        "negative": "artifacts, noise, text, watermark",
        "snap": {},
    },
}


def _check_kinds(kinds: object) -> list:
    """Return ``kinds`` deduplicated in canonical order; non-empty, known kinds only."""
    if isinstance(kinds, (str, bytes, Mapping)) or not isinstance(kinds, Iterable):
        raise ValueError("kinds must be a list, tuple or set of kind names")
    items = list(kinds)
    if not items:
        raise ValueError("kinds must not be empty")
    for k in items:
        if type(k) is not str or k not in DEFECT_FIXES:
            raise ValueError(f"unknown defect kind {k!r}; allowed: {list(DEFECT_KINDS)}")
    return [k for k in DEFECT_KINDS if k in items]


def _check_hint(hint: object) -> str:
    """Return ``hint`` sanitised; over ``HINT_MAX`` characters after cleaning is an error."""
    if not isinstance(hint, str):
        raise ValueError(f"hint must be a string, got {type(hint).__name__}")
    cleaned = sanitize.clean_text(hint, len(hint))
    if len(cleaned) > HINT_MAX:
        raise ValueError(f"hint is too long ({len(cleaned)} chars, limit {HINT_MAX})")
    return cleaned


def amendment(kinds: Iterable, hint: str = "") -> str:
    """Return the positive prompt tail for ``kinds``: phrases in canonical order, then the sanitised ``hint``.

    Raises ``ValueError`` for an empty / unknown ``kinds`` or a non-string / over-long ``hint``.
    """
    ordered = _check_kinds(kinds)
    parts = [DEFECT_FIXES[k]["prompt"] for k in ordered]
    cleaned = _check_hint(hint)
    if cleaned:
        parts.append(cleaned)
    return sanitize.clean_text(", ".join(parts), PROMPT_MAX)


def negative_text(kinds: Iterable) -> str:
    """Return the negative prompt for ``kinds`` (canonical order). Inert at ``cfg 1.0``; see module doc."""
    ordered = _check_kinds(kinds)
    return sanitize.clean_text(", ".join(DEFECT_FIXES[k]["negative"] for k in ordered), PROMPT_MAX)


def snap_overrides(kinds: Iterable) -> dict:
    """Return the merged ``snap`` overrides for ``kinds``; on a shared key the larger value wins."""
    merged: dict = {}
    for k in _check_kinds(kinds):
        for name, value in DEFECT_FIXES[k]["snap"].items():
            merged[name] = max(merged[name], value) if name in merged else value
    return merged


def amend_job(job: Mapping, kinds: Iterable, hint: str = "") -> dict:
    """Return a deep copy of ``job`` whose ``prompt`` is ``"<prompt>, <amendment(kinds, hint)>"``.

    The tail is never cut: when the result would exceed ``PROMPT_MAX`` characters the original prompt is
    shortened instead. Nothing else in ``job`` changes (the seed shift belongs to ``regen_plan``). Raises
    ``ValueError`` if ``job`` is not a mapping with a string ``prompt``, or on bad ``kinds`` / ``hint``.
    """
    if not isinstance(job, Mapping) or not isinstance(job.get("prompt"), str):
        raise ValueError("job must be a mapping with a string 'prompt'")
    tail = amendment(kinds, hint)
    base = sanitize.clean_text(job["prompt"], PROMPT_MAX)
    room = PROMPT_MAX - len(tail) - 2
    out = copy.deepcopy(dict(job))
    out["prompt"] = (base[:room].rstrip() + ", " if base[:room].strip() else "") + tail
    return out
