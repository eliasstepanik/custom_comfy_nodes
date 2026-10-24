"""Load a saved ``ccn-tilemap/1|2`` tileset (PNG + JSON sidecar) back into arrays; torch-free, no ComfyUI.

``read_pair`` is the only function that touches the disk: it reads ``<stem>.png`` and ``<stem>.json`` strictly
inside a caller-supplied root (the ComfyUI output or input dir) and refuses absolute paths, ``..``, odd
characters, symlinks (every directory component and both files) and anything resolving outside the root.
``load_tileset`` takes the raw bytes/text, caps sizes, decodes the PNG (text chunks are never read), verifies
the atlas against the metadata (``atlas.verify_atlas``) and slices every tile into ``cells``. All errors are
``ValueError``; nothing is clamped, guessed or partially returned. Deterministic.
"""

import io
import json
import os
import re
import stat
from dataclasses import dataclass
from typing import Mapping

import numpy as np
from PIL import Image

from . import atlas, expand, plan_schema, snap
from .atlas_meta import FORMAT, read_format
from .pixelize import TILE_SIZES

__all__ = ["PNG_MAX_BYTES", "MAX_METADATA_CHARS", "Tileset", "read_pair", "load_tileset", "slice_cells", "jobs_from_metadata", "base_block"]

PNG_MAX_BYTES = 64 * 1024 * 1024
MAX_METADATA_CHARS = 1_048_576  # same cap as nodes/tilemap_save.py
PIXELS_MAX = 1 << 26  # atlas pixels, as atlas._PIXELS_MAX (8192 x 8192)
_SEGMENT = re.compile(r"[A-Za-z0-9_-]{1,64}")
_HEX = re.compile(r"#[0-9a-f]{6}")
_SEGMENTS_MAX = 3
_PATH_MAX = 256
_CELLS_MAX = 8


@dataclass(frozen=True)
class Tileset:
    """A verified, loaded tileset. ``cells`` is in metadata index order; ``plan``/``jobs_by_key`` are empty unless ``reproducible``."""

    fmt: str
    meta: dict
    rgba: np.ndarray  # (H, W, 4) uint8
    cells: dict  # "<id>#<variant>" -> (footprint_h * n, footprint_w * n, 4) uint8 copy
    palette: np.ndarray  # (k, 3) uint8
    plan: dict
    jobs_by_key: dict
    snap: dict  # validated snap parameters (defaults filled in)
    reproducible: bool
    palette_exact: bool  # False only for strict=False loads that found off-palette pixels
    warnings: tuple


def _segments(path: object, subdir: str) -> tuple:
    """Return ``(dirs, stem)`` of a clean relative path; the ``.png``/``.json`` suffix is optional."""
    if not isinstance(path, str) or not 1 <= len(path) <= _PATH_MAX:
        raise ValueError(f"path must be a text of 1..{_PATH_MAX} characters")
    for suffix in (".png", ".json"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    parts = path.split("/")
    if not all(_SEGMENT.fullmatch(p) for p in parts):
        raise ValueError("path must be relative, with segments of letters, digits, '_' or '-' only (no '..', no backslash, no empty segment)")
    if subdir and parts[0] != subdir:
        parts.insert(0, subdir)
    if len(parts) > _SEGMENTS_MAX or (subdir and len(parts) < 2):
        raise ValueError(f"path needs a file name and at most {_SEGMENTS_MAX} segments" + (f" inside '{subdir}/'" if subdir else ""))
    return tuple(parts[:-1]), parts[-1]


def _read_regular(folder: str, name: str, cap: int) -> bytes:
    """Read a regular, non-symlink file of at most ``cap`` bytes; ``ValueError`` otherwise."""
    path = os.path.join(folder, name)
    try:
        if not stat.S_ISREG(os.lstat(path).st_mode):
            raise ValueError(f"{name} is not a regular file (symlinks are refused)")
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0))
    except FileNotFoundError:
        raise ValueError(f"{name} was not found") from None
    except OSError:
        raise ValueError(f"{name} cannot be opened") from None
    with os.fdopen(fd, "rb") as fh:
        st = os.fstat(fh.fileno())
        if not stat.S_ISREG(st.st_mode) or st.st_size > cap:
            raise ValueError(f"{name} is not a regular file of at most {cap} bytes")
        data = fh.read(cap + 1)
    if len(data) > cap:
        raise ValueError(f"{name} is larger than {cap} bytes")
    return data


def read_pair(root: str, path: str, subdir: str = "tilemap") -> tuple:
    """Return ``(png_bytes, json_text, rel)`` for ``<root>/<path>.png`` + ``.json``; ``rel`` is the cleaned stem path.

    ``root`` is the already trusted ComfyUI output or input directory. With ``subdir`` (default ``tilemap``,
    as the Save node) the path is forced below ``root/subdir``; pass ``""`` for no forced sub folder. Each
    directory component must be a real directory, both files regular non-symlinks inside the root, the PNG
    at most ``PNG_MAX_BYTES``, the sidecar at most ``MAX_METADATA_CHARS``. A missing sidecar is an error.
    """
    if not isinstance(root, str) or not root or not os.path.isdir(root):
        raise ValueError("root must be an existing directory")
    if not isinstance(subdir, str) or (subdir and not _SEGMENT.fullmatch(subdir)):
        raise ValueError("subdir must be '' or a single clean segment")
    dirs, stem = _segments(path, subdir)
    top = os.path.realpath(root)
    cur = top
    for part in dirs:
        cur = os.path.join(cur, part)
        try:
            is_dir = stat.S_ISDIR(os.lstat(cur).st_mode)  # lstat: a symlinked directory is not accepted
        except OSError:
            raise ValueError("tileset folder was not found") from None
        if not is_dir:
            raise ValueError("refusing to read through a symlink or a non-directory")
    if os.path.commonpath([os.path.realpath(cur), top]) != top:
        raise ValueError("path resolves outside the root directory")
    png = _read_regular(cur, stem + ".png", PNG_MAX_BYTES)
    try:
        raw = _read_regular(cur, stem + ".json", MAX_METADATA_CHARS)
    except ValueError as exc:
        raise ValueError(f"metadata sidecar: {exc} (it must sit next to the PNG with the same name)") from None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("metadata sidecar is not valid UTF-8") from None
    return png, text, "/".join(dirs + (stem,))


def _no_dup(pairs: list) -> dict:
    out = dict(pairs)
    if len(out) != len(pairs):
        raise ValueError("duplicate key")
    return out


def _int_text(text: str) -> int:
    if len(text) > 20:
        raise ValueError("integer too long")
    return int(text)


def _reject_constant(_name: str) -> None:
    raise ValueError("non-finite number")


def _parse_meta(text: object) -> dict:
    if not isinstance(text, str) or len(text) > MAX_METADATA_CHARS:
        raise ValueError(f"metadata must be text of at most {MAX_METADATA_CHARS} characters")
    try:
        meta = json.loads(text, object_pairs_hook=_no_dup, parse_int=_int_text, parse_constant=_reject_constant)
    except (ValueError, RecursionError):
        raise ValueError("metadata is not valid JSON (duplicate keys, huge integers and NaN are refused)") from None
    if not isinstance(meta, dict):
        raise ValueError("metadata must be a JSON object")
    return meta


def _decode_png(data: object) -> np.ndarray:
    """PNG bytes -> ``(H, W, 4)`` uint8; size is checked from the header before any pixel is decoded."""
    if not isinstance(data, (bytes, bytearray)) or not 0 < len(data) <= PNG_MAX_BYTES:
        raise ValueError(f"PNG must be 1..{PNG_MAX_BYTES} bytes")
    try:
        with Image.open(io.BytesIO(data)) as img:
            if img.format != "PNG" or img.mode != "RGBA" or getattr(img, "n_frames", 1) != 1:
                raise ValueError("atlas must be a single-frame RGBA PNG")
            if img.width * img.height > PIXELS_MAX:
                raise ValueError(f"atlas PNG is larger than {PIXELS_MAX} pixels")
            img.verify()
        with Image.open(io.BytesIO(data)) as img:  # verify() leaves the image unusable: reopen to decode
            return np.array(img, dtype=np.uint8)  # text chunks (img.info) are never read
    except (OSError, SyntaxError, EOFError, Image.DecompressionBombError):
        raise ValueError("atlas is not a valid PNG image") from None


def _palette(meta: Mapping) -> np.ndarray:
    hexes = meta.get("palette")
    if not isinstance(hexes, list) or not 1 <= len(hexes) <= 256 or not all(isinstance(h, str) and _HEX.fullmatch(h) for h in hexes):
        raise ValueError("metadata palette must be a list of 1..256 colours like '#rrggbb'")
    return np.array([[int(h[i : i + 2], 16) for i in (1, 3, 5)] for h in hexes], dtype=np.uint8)


def _tile_key(t: Mapping) -> str:
    tid, variant = t.get("id"), t.get("variant")
    if not isinstance(tid, str) or type(variant) is not int:
        raise ValueError("metadata tile needs a string id and an integer variant")
    key = f"{tid}#{variant}"
    if "key" in t and t["key"] != key:
        raise ValueError(f"tile key {str(t['key'])[:60]!r} does not match id/variant {key!r}")
    return key


def slice_cells(rgba: np.ndarray, meta: Mapping) -> dict:
    """Return ``{"<id>#<variant>": uint8 copy (ch*n, cw*n, 4)}`` in metadata index order (inverse of ``atlas.assemble``).

    Every cell rectangle must be an exact ``n x n`` block inside ``rgba``; a bad rectangle raises ``ValueError``.
    """
    if not isinstance(rgba, np.ndarray) or rgba.dtype != np.uint8 or rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("atlas must be a uint8 array of shape HxWx4")
    meta = meta if isinstance(meta, Mapping) else {}
    n, tiles = meta.get("tile_size"), meta.get("tiles")
    if type(n) is not int or n not in TILE_SIZES or not isinstance(tiles, list) or not tiles:
        raise ValueError(f"metadata needs tile_size in {TILE_SIZES} and a non-empty tiles list")
    out: dict = {}
    for i, t in enumerate(tiles):
        if not isinstance(t, Mapping):
            raise ValueError(f"metadata tiles[{i}] must be an object")
        key, fp, cells = _tile_key(t), t.get("footprint"), t.get("cells")
        ok = isinstance(fp, list) and len(fp) == 2 and all(type(v) is int and 1 <= v <= _CELLS_MAX for v in fp)
        if not ok or not isinstance(cells, list) or len(cells) != fp[0] * fp[1] or key in out:
            raise ValueError(f"tile {key[:60]!r}: bad footprint, cell list or duplicate key")
        cw, ch = fp
        img = np.zeros((ch * n, cw * n, 4), dtype=np.uint8)
        for j, cell in enumerate(cells):
            x, y = (cell.get("x"), cell.get("y")) if isinstance(cell, Mapping) else (None, None)
            if type(x) is not int or type(y) is not int or x < 0 or y < 0 or x + n > rgba.shape[1] or y + n > rgba.shape[0]:
                raise ValueError(f"tile {key[:60]!r}: cell {j} is not an {n}x{n} block inside the atlas")
            r, c = divmod(j, cw)
            img[r * n : (r + 1) * n, c * n : (c + 1) * n] = rgba[y : y + n, x : x + n]
        out[key] = img
    return out


def jobs_from_metadata(meta: Mapping) -> dict:
    """Rebuild the ``job_json`` dict (``tile_size, view, plan, jobs, seed, style_prefix, cell_px, snap, model``) of a ``/2`` file.

    Jobs come from ``expand.expand_jobs`` on the embedded (re-validated) plan, then ``prompt`` and ``seed`` are
    replaced by the values stored per tile (the ones actually used). ``max_side`` is not stored, so widths and
    heights assume the default. ``ValueError`` for ``/1`` metadata or a missing plan / style_prefix / cell_px /
    tile record / prompt / seed.
    """
    if read_format(meta) != FORMAT:
        raise ValueError(f"jobs can only be rebuilt from {FORMAT} metadata")
    missing = [k for k in ("plan", "style_prefix", "cell_px") if k not in meta]
    if missing:
        raise ValueError(f"metadata has no {missing}: the tileset is not reproducible")
    plan = plan_schema.validate_plan(meta["plan"], plan_schema.MAX_JOBS)
    jobs = expand.expand_jobs(plan, meta.get("tile_size"), meta.get("view"), meta["style_prefix"], meta.get("seed"), meta["cell_px"])
    records = {_tile_key(t): t for t in meta.get("tiles", []) if isinstance(t, Mapping)}
    out = []
    for job in jobs:
        key = f"{job['tile_id']}#{job['variant']}"
        rec = records.get(key)
        prompt, seed = (rec.get("prompt"), rec.get("seed")) if rec else (None, None)
        if rec is None or not isinstance(prompt, str) or not prompt or type(seed) is not int or not 0 <= seed < 2**64:
            raise ValueError(f"metadata has no usable record (prompt, seed) for job {key[:60]!r}")
        out.append({**job, "prompt": prompt, "seed": seed})
    model = meta.get("model")
    return {
        "tile_size": meta["tile_size"], "view": meta["view"], "plan": plan, "jobs": out, "seed": meta["seed"],
        "style_prefix": meta["style_prefix"], "cell_px": meta["cell_px"], "snap": snap.check_snap_params(meta.get("snap")),
        "model": model if isinstance(model, str) else "",
    }  # fmt: skip


def load_tileset(png_bytes: bytes, json_text: str, strict: bool = True) -> Tileset:
    """Decode, verify and slice a saved tileset (``ccn-tilemap/1`` or ``/2``).

    ``strict=True`` applies every ``atlas.verify_atlas`` invariant. ``strict=False`` only downgrades
    "opaque pixel colour not in the palette" to a warning (``palette_exact=False``); size, alpha, ground,
    overlap and rectangle checks stay fatal. ``/1`` (or ``/2`` without a plan) loads with empty ``plan`` /
    ``jobs_by_key`` and ``reproducible=False``; a plan that is present must rebuild its jobs or the load fails.
    """
    meta = _parse_meta(json_text)
    fmt = read_format(meta)
    rgba = _decode_png(png_bytes)
    palette = _palette(meta)
    warnings: list = []
    check = rgba
    if not strict:
        bad = (rgba[..., 3] == 255) & ~np.isin(atlas._pack32(rgba), atlas._pack32(palette))
        if bad.any():
            check = rgba.copy()
            check[bad, :3] = palette[0]  # verify everything else; the off-palette pixels are reported below
            warnings.append(f"{int(bad.sum())} opaque pixels use colours outside the palette (palette_mode forced to refit)")
    atlas.verify_atlas(check, meta, palette, palette_size=len(palette))
    cells = slice_cells(rgba, meta)
    reproducible = fmt == FORMAT and "plan" in meta
    job_json = jobs_from_metadata(meta) if reproducible else None
    return Tileset(
        fmt=fmt, meta=meta, rgba=rgba, cells=cells, palette=palette,
        plan=job_json["plan"] if job_json else {},
        jobs_by_key={f"{j['tile_id']}#{j['variant']}": j for j in job_json["jobs"]} if job_json else {},
        snap=snap.check_snap_params(meta.get("snap")) if fmt == FORMAT else snap.check_snap_params(None),
        reproducible=reproducible, palette_exact=not warnings, warnings=tuple(warnings),
    )  # fmt: skip


def base_block(ts: Tileset, path: str) -> dict:
    """Return ``job_json["base"]``: ``{path, keys, palette, snap}`` naming the keys that already have final pixels.

    ``path`` is the stem path from ``read_pair``. ``palette_exact=False`` forces ``snap.palette_mode = "refit"``."""
    if not isinstance(ts, Tileset):
        raise ValueError("ts must be a Tileset")
    dirs, stem = _segments(path, "")
    snap_block = dict(ts.snap)
    if not ts.palette_exact:
        snap_block["palette_mode"] = "refit"
    return {"path": "/".join(dirs + (stem,)), "keys": list(ts.cells), "palette": list(ts.meta["palette"]), "snap": snap_block}
