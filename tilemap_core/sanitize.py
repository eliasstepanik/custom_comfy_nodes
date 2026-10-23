"""Text and path sanitizers for the tilemap generator.

Pure stdlib, deterministic, no I/O. All functions treat their input as
untrusted (LLM output, user widgets) and raise ``ValueError`` with a
user-readable message on invalid arguments.
"""

import re
import unicodedata

__all__ = ["slug", "clean_text", "sanitize_prefix"]

_NON_SLUG = re.compile(r"[^a-z0-9_]+")
_NON_SEGMENT = re.compile(r"[^A-Za-z0-9_-]+")
_BANNED_CHARS = frozenset("{}|<>[]\\`")
_PREFIX_ROOT = "tilemap"
_PREFIX_MAX_SEGMENTS = 3
_PREFIX_SEGMENT_MAX = 64


def _require_str(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string, got {type(value).__name__}")
    return value


def _ascii_fold(s: str) -> str:
    """Decompose unicode and drop everything that is not ASCII."""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


def slug(s: str, max_len: int = 32) -> str:
    """Return a lowercase ``[a-z0-9_]`` identifier derived from ``s``.

    Non-ASCII letters are folded to ASCII where possible (``é`` -> ``e``);
    every other run of characters collapses to a single ``_``. Leading and
    trailing ``_`` are stripped. Raises ``ValueError`` if the result is empty
    or ``max_len`` is below 1.
    """
    _require_str(s, "slug input")
    if max_len < 1:
        raise ValueError("slug max_len must be >= 1")
    out = _NON_SLUG.sub("_", _ascii_fold(s).lower()).strip("_")
    out = out[:max_len].strip("_")
    if not out:
        raise ValueError(f"cannot build an identifier from {s[:40]!r}")
    return out


def clean_text(s: str, max_len: int) -> str:
    """Return ``s`` made safe for prompts and metadata.

    Applies NFKC normalisation, replaces any whitespace with a space, drops
    control/format/surrogate/private/unassigned characters (this removes NUL
    bytes and bidi overrides) and the characters ``{}|<>[]\\`` , collapses
    whitespace, trims, and truncates to ``max_len`` characters. May return an
    empty string. Raises ``ValueError`` if ``max_len`` is negative.
    """
    _require_str(s, "text")
    if max_len < 0:
        raise ValueError("clean_text max_len must be >= 0")
    kept = []
    for ch in unicodedata.normalize("NFKC", s):
        if ch.isspace():
            kept.append(" ")
        elif ch in _BANNED_CHARS or unicodedata.category(ch)[0] == "C":
            continue
        else:
            kept.append(ch)
    return " ".join("".join(kept).split())[:max_len].rstrip()


def sanitize_prefix(p: str) -> str:
    """Return a safe relative save prefix that always lives under ``tilemap/``.

    The input is split on ``/`` and ``\\``; empty, ``.`` and ``..`` segments
    are dropped; every other segment is ASCII-folded, characters outside
    ``[A-Za-z0-9_-]`` become ``_`` and the segment is cut to 64 characters.
    ``tilemap/`` is prepended unless the first segment already is ``tilemap``,
    the result has at most 3 segments, and a bare ``tilemap`` (or empty input)
    becomes ``tilemap/tilemap``. Result is never absolute and has no ``..``.
    """
    _require_str(p, "prefix")
    segments = []
    for raw in re.split(r"[/\\]", p):
        raw = raw.strip()
        if raw in ("", ".", ".."):
            continue
        seg = _NON_SEGMENT.sub("_", _ascii_fold(raw))[:_PREFIX_SEGMENT_MAX]
        if seg:
            segments.append(seg)
    if not segments or segments[0] != _PREFIX_ROOT:
        segments.insert(0, _PREFIX_ROOT)
    if len(segments) == 1:
        segments.append(_PREFIX_ROOT)
    return "/".join(segments[:_PREFIX_MAX_SEGMENTS])
