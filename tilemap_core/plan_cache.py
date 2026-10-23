"""Disk cache for validated tile plans.

Key: sha256 over canonical JSON of ``(SCHEMA_VERSION, model, prompt, view,
max_tiles)`` plus, only in extend mode, the hash of the existing plan. Seed,
tile size and temperature are not part of the key. Files
are named ``<64 hex chars>.json`` inside the cache directory the caller passes
in, so no input text ever reaches a path component. A plan is validated before
it is stored and again when it is loaded; the file on disk is never trusted.
Corrupt, oversized, unreadable, symlinked or no-longer-valid files are plain
cache misses. Writes go to a temp file in the same directory, are fsynced and
then moved with ``os.replace``, so a reader sees the old or the new file, never
a partial one.

Only the plan is stored. Nothing here sees, logs or stores the API key.
Stdlib only; no import-time IO, no global state.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from typing import Optional, Union

from .plan_prompt import SCHEMA_VERSION
from .plan_schema import MAX_JOBS, MAX_RAW_CHARS, PlanError, extract_json, validate_plan

__all__ = ["cache_key", "load", "plan_hash", "store"]

PathLike = Union[str, "os.PathLike[str]"]

_KEY_RE = re.compile(r"[0-9a-f]{64}")
# A stored plan is at most MAX_RAW_CHARS characters (4 UTF-8 bytes each, worst case).
_MAX_FILE_BYTES = MAX_RAW_CHARS * 4


def plan_hash(plan: dict) -> str:
    """Return the 64-char lowercase hex sha256 of a plan's canonical JSON.

    This is the value callers pass as ``existing_hash`` in extend mode. Key order
    does not matter. Raises ``ValueError`` when ``plan`` is not a dict or holds
    values JSON cannot represent.
    """
    if not isinstance(plan, dict):
        raise ValueError(f"plan must be a dict, got {type(plan).__name__}")
    try:
        canon = json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError(f"plan is not JSON-serialisable: {exc}") from None
    return hashlib.sha256(canon.encode("ascii")).hexdigest()


def cache_key(model: str, prompt: str, view: str, max_tiles: int, existing_hash: str = "") -> str:
    """Return the 64-char lowercase hex sha256 key for these plan inputs.

    ``existing_hash`` is the ``plan_hash`` of the plan an extend request builds
    on. The empty default adds nothing to the hashed text, so keys of plain
    (non-extend) requests are identical to those made before the parameter
    existed; a non-empty hash is a further key component.

    Deterministic across runs and platforms. Raises ``ValueError`` when a text
    argument is not a ``str``, ``max_tiles`` is not an ``int`` or
    ``existing_hash`` is neither empty nor 64 lowercase hex characters.
    """
    for name, value in (("model", model), ("prompt", prompt), ("view", view)):
        if not isinstance(value, str):
            raise ValueError(f"{name} must be text, got {type(value).__name__}")
    if isinstance(max_tiles, bool) or not isinstance(max_tiles, int):
        raise ValueError(f"max_tiles must be an integer, got {type(max_tiles).__name__}")
    if not isinstance(existing_hash, str):
        raise ValueError(f"existing_hash must be text, got {type(existing_hash).__name__}")
    if existing_hash and _KEY_RE.fullmatch(existing_hash) is None:
        raise ValueError("existing_hash must be empty or a 64-character lowercase hex digest (use plan_hash())")
    parts = {
        "max_tiles": max_tiles,
        "model": model,
        "prompt": prompt,
        "schema_version": SCHEMA_VERSION,
        "view": view,
    }
    if existing_hash:
        parts["existing_hash"] = existing_hash
    canon = json.dumps(parts, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canon.encode("ascii")).hexdigest()


def _path(cache_dir: PathLike, key: str) -> str:
    """Return the file path for ``key``; reject anything but a 64-char hex digest."""
    if not isinstance(key, str) or _KEY_RE.fullmatch(key) is None:
        raise ValueError("cache key must be a 64-character lowercase hex digest (use cache_key())")
    return os.path.join(os.fspath(cache_dir), key + ".json")


def _read_capped(path: str) -> Optional[bytes]:
    """Read a regular, non-symlink file of at most ``_MAX_FILE_BYTES``; else None."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        return None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return None
        with os.fdopen(fd, "rb") as fh:
            fd = -1
            data = fh.read(_MAX_FILE_BYTES + 1)
    except OSError:
        return None
    finally:
        if fd != -1:
            os.close(fd)
    return data if len(data) <= _MAX_FILE_BYTES else None


def load(cache_dir: PathLike, key: str, max_tiles: int = MAX_JOBS) -> Optional[dict]:
    """Return the cached plan for ``key`` or ``None`` on any miss.

    The file is parsed with the hardened ``extract_json`` and re-validated with
    ``validate_plan(plan, max_tiles)``; the returned dict is the normalised
    copy, never the raw file content. ``max_tiles`` defaults to the loosest
    bound (``MAX_JOBS``); callers pass the value that went into the key.
    A missing, unreadable, oversized, corrupt or invalid file is a miss and
    never raises. A malformed ``key`` raises ``ValueError``.
    """
    data = _read_capped(_path(cache_dir, key))
    if data is None:
        return None
    try:
        return validate_plan(extract_json(data.decode("utf-8")), max_tiles)
    except (UnicodeDecodeError, PlanError, ValueError, RecursionError):
        return None


def store(cache_dir: PathLike, key: str, plan: dict, max_tiles: int = MAX_JOBS) -> None:
    """Validate ``plan`` and write it atomically to ``<cache_dir>/<key>.json``.

    Creates ``cache_dir`` if needed. Raises ``PlanError`` when the plan is not
    valid, ``ValueError`` for a malformed ``key`` or a plan too large to be
    loaded again, and ``OSError`` when the write fails (callers may treat that
    as non-fatal). A failed write removes its temp file and leaves any
    existing cache file untouched.
    """
    target = _path(cache_dir, key)
    text = json.dumps(
        validate_plan(plan, max_tiles), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    if len(text) > MAX_RAW_CHARS:
        raise ValueError(f"plan is {len(text)} characters, cache limit {MAX_RAW_CHARS}")
    directory = os.path.dirname(target)
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
