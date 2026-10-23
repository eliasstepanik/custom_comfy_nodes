"""``ccn-tilemap/2`` metadata helpers (format check, top-level extras, per-tile provenance).

Pure stdlib, no I/O. ``atlas.py`` stays under the 300-line limit by delegating the
``/2`` rules here: ``read_format`` accepts the ``/1`` and ``/2`` format strings,
``check_extra`` validates the optional top-level blocks that ``/2`` adds to the
``/1`` keys (allowlist, no collision with ``/1`` keys, JSON-safe values), and
``tile_provenance`` validates the optional per-tile keys (``prompt, seed,
attempt, source, job_index, key``). Everything raises ``ValueError`` with a
readable message and never mutates its arguments.
"""

import math
import numbers
from typing import Mapping

__all__ = ["FORMAT", "FORMATS", "EXTRA_KEYS", "read_format", "check_extra", "tile_provenance"]

FORMAT = "ccn-tilemap/2"
FORMATS = ("ccn-tilemap/1", FORMAT)
EXTRA_KEYS = ("style_prefix", "cell_px", "plan", "snap", "sampler", "lineage", "review")  # architecture-tileset-reuse section 3
_V1_KEYS = ("format", "tile_size", "columns", "rows", "spacing", "width", "height", "view", "model", "seed", "theme", "palette", "tiles")
_DICT_KEYS = ("plan", "snap", "sampler", "lineage", "review")  # blocks that must be JSON objects
_SOURCES = ("generated", "kept", "regenerated")
_DEPTH_MAX = 8
_INT_MAX = 2**64  # magnitude cap for JSON ints inside extras
_CELL_PX = (1, 4096)
_PROMPT_MAX = 4000
_KEY_MAX = 200  # same limit as atlas._entry
_SMALL_MAX = 2**31 - 1  # attempt / job_index upper bound


def _show(value: object) -> str:
    """Short printable rendering of an untrusted value for error messages (no control chars)."""
    return repr(value[:60]) if isinstance(value, str) else f"<{type(value).__name__}>"  # repr escapes control chars


def read_format(meta: object) -> str:
    """Return ``meta["format"]`` if it is one of ``FORMATS``; ``ValueError`` otherwise."""
    fmt = meta.get("format") if isinstance(meta, Mapping) else None
    if not isinstance(fmt, str) or fmt not in FORMATS:
        raise ValueError(f"metadata format must be one of {FORMATS}, got {_show(fmt)}")
    return fmt


def _plain(value: object, path: str, depth: int) -> object:
    """Validate JSON-safe ``value`` (exact scalar types, finite floats, depth cap); return a fresh copy."""
    if depth > _DEPTH_MAX:
        raise ValueError(f"extra {path} is nested deeper than {_DEPTH_MAX} levels")
    if value is None or type(value) in (bool, str):
        return value
    if type(value) is int:
        if abs(value) > _INT_MAX:
            raise ValueError(f"extra {path} integer is out of range")
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"extra {path} must be a finite number")
        return value
    if isinstance(value, list):
        return [_plain(v, f"{path}[{i}]", depth + 1) for i, v in enumerate(value)]
    if isinstance(value, dict):
        for k in value:
            if type(k) is not str:
                raise ValueError(f"extra {path} keys must be strings, got {_show(k)}")
        return {k: _plain(v, f"{path}.{k[:40]}", depth + 1) for k, v in value.items()}
    raise ValueError(f"extra {path} must be JSON-serialisable (dict, list, str, int, float, bool, None), got {type(value).__name__}")


def check_extra(extra: object) -> dict:
    """Validate the optional top-level ``/2`` blocks and return a deep copy (``None`` -> ``{}``).

    Allowed keys are exactly ``EXTRA_KEYS``: ``style_prefix`` (str), ``cell_px``
    (int 1..4096, not bool), and ``plan, snap, sampler, lineage, review`` (JSON
    objects of dict/list/str/int/float/bool/None, finite numbers, depth <= 8).
    Keys of the ``/1`` layout and unknown keys raise ``ValueError``.
    """
    if extra is None:
        return {}
    if not isinstance(extra, dict):
        raise ValueError(f"extra must be a dict or None, got {type(extra).__name__}")
    for k in extra:
        if type(k) is not str:
            raise ValueError(f"extra keys must be strings, got {_show(k)}")
        if k in _V1_KEYS:
            raise ValueError(f"extra key {_show(k)} collides with a ccn-tilemap/1 key")
        if k not in EXTRA_KEYS:
            raise ValueError(f"extra key {_show(k)} is not allowed, expected one of {EXTRA_KEYS}")
    out = {k: _plain(v, k, 0) for k, v in extra.items()}
    if "style_prefix" in out and not isinstance(out["style_prefix"], str):
        raise ValueError("extra style_prefix must be a string")
    if "cell_px" in out and not (type(out["cell_px"]) is int and _CELL_PX[0] <= out["cell_px"] <= _CELL_PX[1]):
        raise ValueError(f"extra cell_px must be an integer in {_CELL_PX[0]}..{_CELL_PX[1]}")
    for k in _DICT_KEYS:
        if k in out and not isinstance(out[k], dict):
            raise ValueError(f"extra {k} must be an object")
    return out


def _count(value: object, name: str, lo: int, hi: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer, got {_show(value)}")
    if not lo <= int(value) <= hi:
        raise ValueError(f"{name} must be in {lo}..{hi}")
    return int(value)


def tile_provenance(p: Mapping, key: str) -> dict:
    """Return the new optional per-tile metadata keys that item/placement ``p`` carries.

    Candidates: ``prompt`` (str or ``None`` for Wang tiles), ``seed`` (int
    0..2**64-1), ``attempt`` (int >= 0), ``source`` (``generated|kept|regenerated``),
    ``job_index`` (int >= 0 or ``None``). ``key`` (the placement key) is added only
    when at least one of those is carried; a ``key`` carried by ``p`` must equal it.
    A tile carrying none of them yields ``{}``. Invalid values raise ``ValueError``.
    """
    if not isinstance(p, Mapping):
        raise ValueError("tile item must be a dict")
    if not isinstance(key, str) or not 1 <= len(key) <= _KEY_MAX:
        raise ValueError(f"key must be a non-empty string of at most {_KEY_MAX} characters")
    if "key" in p and p["key"] != key:
        raise ValueError(f"tile key {_show(p['key'])} differs from placement key {_show(key)}")
    name = f"tile {key[:60]!r}"
    out: dict = {}
    if "prompt" in p:
        prompt = p["prompt"]
        if prompt is not None and (not isinstance(prompt, str) or len(prompt) > _PROMPT_MAX):
            raise ValueError(f"{name} prompt must be a string of at most {_PROMPT_MAX} characters or None")
        out["prompt"] = prompt
    if "seed" in p:
        out["seed"] = _count(p["seed"], f"{name} seed", 0, 2**64 - 1)
    if "attempt" in p:
        out["attempt"] = _count(p["attempt"], f"{name} attempt", 0, _SMALL_MAX)
    if "source" in p:
        if not isinstance(p["source"], str) or p["source"] not in _SOURCES:
            raise ValueError(f"{name} source must be one of {_SOURCES}, got {_show(p['source'])}")
        out["source"] = p["source"]
    if "job_index" in p:
        out["job_index"] = None if p["job_index"] is None else _count(p["job_index"], f"{name} job_index", 0, _SMALL_MAX)
    if out:
        out["key"] = key
    return out
