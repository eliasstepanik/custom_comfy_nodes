"""Extraction and strict validation of the LLM tile plan.

Pure stdlib, deterministic, no I/O. The LLM reply is untrusted: it is size
capped, parsed with ``json.loads`` machinery only (never ``eval``), rejected on
duplicate keys, ``NaN``/``Infinity``, oversized integers and deep nesting, and
then validated field by field. ``validate_plan`` builds a fresh normalised dict
(unknown keys dropped, ids slugged, texts cleaned); it never returns parts of
its input. Every failure raises ``PlanError`` whose text names the field path
and the rule, so ``plan_prompt.build_repair_messages`` can feed it back to the
model.
"""

import json
from typing import Any

from .sanitize import clean_text, slug

__all__ = [
    "PlanError",
    "MAX_RAW_CHARS",
    "MAX_JOBS",
    "CATEGORIES",
    "extract_json",
    "validate_plan",
    "parse_and_validate",
    "validate_extension",
]

MAX_RAW_CHARS = 65536
MAX_JOBS = 64
CATEGORIES = ("ground", "object", "overlay")

_ID_MAX = 32
_THEME_MAX = 80
_NAME_MAX = 40
_DESC_MIN, _DESC_MAX = 10, 300
_VARIANTS_MAX = 4
_SIDE_MAX = 4
_AREA_MAX = 8
_TRANSITIONS_MAX = 4
_INT_DIGITS_MAX = 18  # JSON ints longer than this are never valid plan values


class PlanError(ValueError):
    """The LLM output is not a valid tile plan; the text is repair-friendly."""


def _show(value: Any) -> str:
    """Short, safe rendering of an untrusted value for error messages."""
    if isinstance(value, str):
        return repr(value[:40])
    if isinstance(value, int) and not isinstance(value, bool):
        return repr(value) if value.bit_length() <= 64 else "a huge integer"
    if value is None or isinstance(value, (bool, float)):
        return repr(value)
    return f"a {type(value).__name__}"


def _object_pairs(pairs: list) -> dict:
    out: dict = {}
    for key, value in pairs:
        if key in out:
            raise PlanError(f"duplicate key {_show(key)} in JSON object")
        out[key] = value
    return out


def _parse_int(text: str) -> int:
    if len(text) > _INT_DIGITS_MAX:
        raise PlanError("JSON contains an integer with too many digits")
    return int(text)


def _parse_constant(name: str) -> Any:
    raise PlanError(f"JSON contains {name}, which is not allowed")


_DECODER = json.JSONDecoder(
    object_pairs_hook=_object_pairs,
    parse_int=_parse_int,
    parse_constant=_parse_constant,
)


def extract_json(text: str) -> dict:
    """Return the JSON object embedded in an LLM reply.

    Accepts a bare object, one wrapped in code fences, or one surrounded by
    prose or a BOM: the span from the first ``{`` to the last ``}`` is parsed,
    and if that fails because of trailing text, the first complete object
    starting at the first ``{`` is used. Raises ``PlanError`` on non-string
    input, text over ``MAX_RAW_CHARS``, no object, invalid JSON, duplicate keys,
    ``NaN``/``Infinity``, integers over 18 digits and nesting too deep to parse.
    """
    if not isinstance(text, str):
        raise PlanError(f"reply must be text, got {type(text).__name__}")
    if len(text) > MAX_RAW_CHARS:
        raise PlanError(f"reply is too long ({len(text)} chars, limit {MAX_RAW_CHARS})")
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise PlanError("no JSON object found: reply with one JSON object starting with '{'")
    try:
        try:
            return _DECODER.decode(text[start : end + 1])
        except json.JSONDecodeError as first:
            try:
                return _DECODER.raw_decode(text, start)[0]
            except json.JSONDecodeError:
                raise PlanError(
                    f"invalid JSON ({first.msg}), line {first.lineno} column {first.colno}"
                ) from None
    except RecursionError:
        raise PlanError("JSON is nested too deeply") from None


def _int_in(value: Any, lo: int, hi: int, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlanError(f"{path}: must be an integer from {lo} to {hi}, got {_show(value)}")
    if not lo <= value <= hi:
        raise PlanError(f"{path}: must be from {lo} to {hi}, got {_show(value)}")
    return value


def _text(value: Any, path: str, max_len: int, min_len: int = 1) -> str:
    if not isinstance(value, str):
        raise PlanError(f"{path}: must be a string, got {_show(value)}")
    out = clean_text(value, max_len)
    if len(out) < min_len:
        raise PlanError(f"{path}: needs at least {min_len} usable characters")
    return out


def _ident(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise PlanError(f"{path}: must be a string identifier, got {_show(value)}")
    try:
        return slug(value, _ID_MAX)
    except ValueError:
        raise PlanError(
            f"{path}: {_show(value)} has no letters or digits; use lowercase letters, "
            "digits and underscore"
        ) from None


def _footprint(raw: Any, category: str, path: str) -> list:
    if raw is None:
        return [1, 1]
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise PlanError(f"{path}: must be a list [w, h], got {_show(raw)}")
    w = _int_in(raw[0], 1, _SIDE_MAX, f"{path}[0]")
    h = _int_in(raw[1], 1, _SIDE_MAX, f"{path}[1]")
    if w * h > _AREA_MAX:
        raise PlanError(f"{path}: w*h must be at most {_AREA_MAX}, got {w}*{h}")
    if category != "object" and (w, h) != (1, 1):
        raise PlanError(f"{path}: only category 'object' may be larger than [1, 1]")
    return [w, h]


def _tile(raw: Any, path: str) -> dict:
    if not isinstance(raw, dict):
        raise PlanError(f"{path}: must be an object, got {_show(raw)}")
    for key in ("id", "name", "category", "description"):
        if key not in raw:
            raise PlanError(f"{path}.{key}: missing")
    category = raw["category"]
    if not isinstance(category, str) or category not in CATEGORIES:
        raise PlanError(
            f"{path}.category: must be one of {', '.join(CATEGORIES)}, got {_show(category)}"
        )
    variants = raw.get("variants")
    return {
        "id": _ident(raw["id"], f"{path}.id"),
        "name": _text(raw["name"], f"{path}.name", _NAME_MAX),
        "category": category,
        "description": _text(raw["description"], f"{path}.description", _DESC_MAX, _DESC_MIN),
        "variants": 1 if variants is None else _int_in(variants, 1, _VARIANTS_MAX, f"{path}.variants"),
        "footprint": _footprint(raw.get("footprint"), category, f"{path}.footprint"),
    }


def _transitions(raw: Any, grounds: set, all_ids: set) -> list:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise PlanError(f"transitions: must be a list of [id_a, id_b] pairs, got {_show(raw)}")
    if len(raw) > _TRANSITIONS_MAX:
        raise PlanError(f"transitions: at most {_TRANSITIONS_MAX} pairs, got {len(raw)}")
    out: list = []
    seen: set = set()
    for i, pair in enumerate(raw):
        path = f"transitions[{i}]"
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise PlanError(f"{path}: must be a pair [id_a, id_b], got {_show(pair)}")
        a, b = _ident(pair[0], f"{path}[0]"), _ident(pair[1], f"{path}[1]")
        for ident in (a, b):
            if ident not in all_ids:
                raise PlanError(f"{path}: unknown tile id {_show(ident)}")
            if ident not in grounds:
                raise PlanError(f"{path}: tile {_show(ident)} is not a ground tile")
        if a == b:
            raise PlanError(f"{path}: needs two different ids, got {_show(a)} twice")
        key = frozenset((a, b))
        if key in seen:
            raise PlanError(f"{path}: duplicate pair {_show(a)}, {_show(b)}")
        seen.add(key)
        out.append([a, b])
    return out


def validate_plan(obj: dict, max_tiles: int) -> dict:
    """Validate ``obj`` against the tile plan schema and return a normalised copy.

    Unknown keys are dropped, ids are slugged (max 32) before the uniqueness
    check, texts go through ``clean_text``, ``variants`` defaults to 1 and
    ``footprint`` to ``[1, 1]``. Rules: 1..``max_tiles`` tiles, sum of
    variants <= ``MAX_JOBS``, ``variants`` 1..4, footprint sides 1..4 with
    w*h <= 8 and only ``object`` tiles above [1, 1], at most 4 distinct
    unordered transition pairs of existing ground ids. Booleans are not
    integers. Raises ``PlanError`` naming the field path; ``ValueError`` if
    ``max_tiles`` is not an integer >= 1.
    """
    if isinstance(max_tiles, bool) or not isinstance(max_tiles, int) or max_tiles < 1:
        raise ValueError("max_tiles must be an integer >= 1")
    if not isinstance(obj, dict):
        raise PlanError(f"plan must be a JSON object, got {_show(obj)}")
    if "theme" not in obj:
        raise PlanError("theme: missing")
    theme = _text(obj["theme"], "theme", _THEME_MAX)
    raw_tiles = obj.get("tiles")
    if not isinstance(raw_tiles, list):
        raise PlanError(f"tiles: must be a list of tile objects, got {_show(raw_tiles)}")
    if not 1 <= len(raw_tiles) <= max_tiles:
        raise PlanError(f"tiles: need 1 to {max_tiles} tiles, got {len(raw_tiles)}")
    tiles: list = []
    ids: set = set()
    for i, raw in enumerate(raw_tiles):
        tile = _tile(raw, f"tiles[{i}]")
        if tile["id"] in ids:
            raise PlanError(f"tiles[{i}].id: duplicate id {_show(tile['id'])}; ids must be unique")
        ids.add(tile["id"])
        tiles.append(tile)
    jobs = sum(t["variants"] for t in tiles)
    if jobs > MAX_JOBS:
        raise PlanError(f"tiles: sum of variants is {jobs}, limit {MAX_JOBS}; reduce tiles or variants")
    grounds = {t["id"] for t in tiles if t["category"] == "ground"}
    return {
        "theme": theme,
        "tiles": tiles,
        "transitions": _transitions(obj.get("transitions"), grounds, ids),
    }


def parse_and_validate(text: str, max_tiles: int) -> dict:
    """``extract_json`` then ``validate_plan``: raw LLM reply -> normalised plan."""
    return validate_plan(extract_json(text), max_tiles)


def validate_extension(existing: dict, addition: dict, max_tiles: int) -> tuple:
    """Check extension ``addition`` against ``existing`` (same theme; a known id may only raise ``variants``;
    no repeated transition pair; merged plan valid for ``max_tiles``); return ``(merged_plan, new "id#variant" keys)``."""
    if isinstance(max_tiles, bool) or not isinstance(max_tiles, int) or max_tiles < 1:
        raise ValueError("max_tiles must be an integer >= 1")
    base = validate_plan(existing, MAX_JOBS)
    if not isinstance(addition, dict):
        raise PlanError(f"extension must be a JSON object, got {_show(addition)}")
    if "theme" not in addition or _text(addition["theme"], "theme", _THEME_MAX) != base["theme"]:
        raise PlanError(f"theme: must equal the existing theme {_show(base['theme'])}")
    raw = addition.get("tiles") or []
    if not isinstance(raw, list):
        raise PlanError(f"tiles: must be a list of tile objects, got {_show(raw)}")
    tiles = [dict(t) for t in base["tiles"]]
    known, old, seen = {t["id"]: t for t in tiles}, {t["id"]: t["variants"] for t in tiles}, set()
    for i, item in enumerate(raw):
        tile, path = _tile(item, f"tiles[{i}]"), f"tiles[{i}]"
        prev, tid = known.get(tile["id"]), tile["id"]
        if tid in seen:
            raise PlanError(f"{path}.id: duplicate id {_show(tid)}; ids must be unique")
        seen.add(tid)
        if prev is None:
            known[tid] = tile
            tiles.append(tile)
            continue
        if (tile["category"], tile["name"]) != (prev["category"], prev["name"]) or (
            item.get("footprint") is not None and tile["footprint"] != prev["footprint"]
        ):
            raise PlanError(f"{path}: cannot redefine category, name or footprint of {_show(tid)}")
        if tile["variants"] <= prev["variants"]:
            raise PlanError(f"{path}.variants: {_show(tid)} has {prev['variants']}; give a larger total")
        prev["variants"] = tile["variants"]
    grounds = {t["id"] for t in tiles if t["category"] == "ground"}
    pairs = _transitions(addition.get("transitions"), grounds, set(known))
    have = {frozenset(p) for p in base["transitions"]}
    for i, pair in enumerate(pairs):
        if frozenset(pair) in have:
            raise PlanError(f"transitions[{i}]: pair {_show(pair[0])}, {_show(pair[1])} already exists")
    if not (seen or pairs):
        raise PlanError("extension adds nothing: give new tiles, more variants or new transitions")
    merged = validate_plan({"theme": base["theme"], "tiles": tiles, "transitions": base["transitions"] + pairs}, max_tiles)
    return merged, [f"{t['id']}#{v}" for c in CATEGORIES for t in merged["tiles"] if t["category"] == c
                    for v in range(old.get(t["id"], 0), t["variants"])]
