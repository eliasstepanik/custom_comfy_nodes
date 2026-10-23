"""Atlas packing, assembly, metadata and invariant checks for the tilemap generator.

Pure numpy + stdlib, deterministic, no I/O. The atlas is a fixed grid of
``columns`` cells of ``n x n`` pixels, ``spacing`` transparent pixels apart; a
tile covers ``cells_w x cells_h`` cells (single tile = 1x1). ``pack`` places
items first-fit row-major, ``assemble`` copies their RGBA cells, ``build_metadata``
describes the result (``format`` ``ccn-tilemap/1``), ``verify_atlas`` re-derives
every invariant from the pixels. Arguments are validated (``ValueError`` with a
readable message) and never mutated.
"""

import copy
from typing import Mapping, Sequence

import numpy as np

from . import sanitize
from .pixelize import TILE_SIZES
from .plan_schema import CATEGORIES

__all__ = ["FORMAT", "pack", "assemble", "build_metadata", "verify_atlas"]

FORMAT = "ccn-tilemap/1"
_COLUMNS = (1, 64)
_SPACING = (0, 16)
_CELLS_MAX = 8  # footprint cells per side (plans use <= 4)
_ITEMS_MAX = 512
_PIXELS_MAX = 1 << 26  # 8192 x 8192
_CARRY = ("id", "name", "category", "variant", "wang")  # descriptor keys pack passes through


def _int(value: object, name: str, lo: int, hi: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}")
    if not lo <= int(value) <= hi:
        raise ValueError(f"{name} must be in {lo}..{hi}, got {int(value)}")
    return int(value)


def _geometry(n: object, columns: object, spacing: object) -> tuple:
    n = _int(n, "n", 1, max(TILE_SIZES))
    if n not in TILE_SIZES:
        raise ValueError(f"n must be one of {TILE_SIZES}, got {n}")
    return n, _int(columns, "columns", *_COLUMNS), _int(spacing, "spacing", *_SPACING)


def _dims(rows: int, columns: int, n: int, spacing: int) -> tuple:
    h, w = rows * n + (rows - 1) * spacing, columns * n + (columns - 1) * spacing
    if h * w > _PIXELS_MAX:
        raise ValueError(f"atlas would be {w}x{h} px: too large, use more columns or fewer tiles")
    return h, w


def _seq(value: object, name: str) -> Sequence:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a list of dicts")
    if not 1 <= len(value) <= _ITEMS_MAX:
        raise ValueError(f"{name} must hold 1..{_ITEMS_MAX} entries, got {len(value)}")
    return value


def _entry(item: object, i: int, seen: set, columns: int) -> tuple:
    """Validate key/cells_w/cells_h of one item; return ``(key, cells_w, cells_h)``."""
    key = item.get("key") if isinstance(item, Mapping) else None
    if not isinstance(key, str) or not 1 <= len(key) <= 200:
        raise ValueError(f"entry {i}: must be a dict with a non-empty string key of at most 200 characters")
    if key in seen:
        raise ValueError(f"duplicate key {key[:60]!r}")
    seen.add(key)
    cw = _int(item.get("cells_w"), f"tile {key[:60]!r} cells_w", 1, _CELLS_MAX)
    ch = _int(item.get("cells_h"), f"tile {key[:60]!r} cells_h", 1, _CELLS_MAX)
    if cw > columns:
        raise ValueError(f"tile {key[:60]!r} is {cw} cells wide: columns must be at least {cw}")
    return key, cw, ch


def pack(items: Sequence[Mapping], columns: int) -> list:
    """Place ``items`` first-fit, row-major, on a ``columns``-wide occupancy grid.

    Item: ``{key, cells_w, cells_h}`` (``key`` unique), optionally ``id, name,
    category, variant, wang`` for ``build_metadata``. Returns new dicts in input
    order with ``col``/``row`` added; footprints stay contiguous. A tile wider
    than ``columns`` raises ``ValueError`` naming it and the minimum columns.
    """
    columns = _int(columns, "columns", *_COLUMNS)
    grid: set = set()  # occupied (row, col) cells
    seen: set = set()
    out = []
    for i, item in enumerate(_seq(items, "items")):
        key, cw, ch = _entry(item, i, seen, columns)
        row, col = 0, None
        while col is None:  # a row below all placed tiles always fits
            free = (c for c in range(columns - cw + 1) if not any((r, k) in grid for r in range(row, row + ch) for k in range(c, c + cw)))
            col = next(free, None)
            row += col is None
        grid.update((r, k) for r in range(row, row + ch) for k in range(col, col + cw))
        placed = {k: copy.deepcopy(item[k]) for k in _CARRY if k in item}
        out.append({**placed, "key": key, "cells_w": cw, "cells_h": ch, "col": col, "row": row})
    return out


def _check_placements(placements: object, columns: int) -> int:
    """Validate placements (in bounds, unique keys, no overlap); return the row count."""
    claimed: dict = {}
    seen: set = set()
    rows = 0
    for i, p in enumerate(_seq(placements, "placements")):
        key, cw, ch = _entry(p, i, seen, columns)
        col = _int(p.get("col"), f"tile {key[:60]!r} col", 0, columns - cw)
        row = _int(p.get("row"), f"tile {key[:60]!r} row", 0, _ITEMS_MAX * _CELLS_MAX)
        for cell in ((r, c) for r in range(row, row + ch) for c in range(col, col + cw)):
            if cell in claimed:
                raise ValueError(f"tiles {claimed[cell][:60]!r} and {key[:60]!r} overlap at {cell}")
            claimed[cell] = key
        rows = max(rows, row + ch)
    return rows


def assemble(placements: Sequence[Mapping], cell_images: Mapping, n: int, columns: int, spacing: int) -> np.ndarray:
    """Build the RGBA atlas ``(rows*n + (rows-1)*spacing, columns*n + (columns-1)*spacing, 4)``.

    ``cell_images`` maps each placement key to a ``uint8`` RGBA array of shape
    ``(cells_h*n, cells_w*n, 4)``, split into ``n x n`` cells at
    ``(col*(n+spacing), row*(n+spacing))``. Empty cells and gaps stay ``(0,0,0,0)``.
    """
    n, columns, spacing = _geometry(n, columns, spacing)
    height, width = _dims(_check_placements(placements, columns), columns, n, spacing)
    if not isinstance(cell_images, Mapping) or set(cell_images) != {p["key"] for p in placements}:
        raise ValueError("cell_images must be a dict with exactly one image per placement key")
    atlas = np.zeros((height, width, 4), dtype=np.uint8)
    step = n + spacing
    for p in placements:
        cw, ch = int(p["cells_w"]), int(p["cells_h"])
        img, want = cell_images[p["key"]], (ch * n, cw * n, 4)
        if not isinstance(img, np.ndarray) or img.dtype != np.uint8 or img.shape != want:
            got = (img.dtype, img.shape) if isinstance(img, np.ndarray) else type(img).__name__
            raise ValueError(f"image for tile {p['key'][:60]!r} must be uint8 of shape {want}, got {got}")
        for j in range(ch):
            for i in range(cw):
                y, x = (int(p["row"]) + j) * step, (int(p["col"]) + i) * step
                atlas[y : y + n, x : x + n] = img[j * n : (j + 1) * n, i * n : (i + 1) * n]
    return atlas


def _hex_palette(palette: object) -> list:
    if not isinstance(palette, np.ndarray) or palette.dtype != np.uint8 or palette.ndim != 2:
        raise ValueError("palette must be a uint8 array of shape (k, 3)")
    if palette.shape[1] != 3 or not 1 <= palette.shape[0] <= 256:
        raise ValueError(f"palette must have shape (1..256, 3), got {palette.shape}")
    return ["#%02x%02x%02x" % tuple(c) for c in palette.tolist()]


def _ident(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or sanitize.slug(value, 80) != value:
        raise ValueError(f"{name} must be a lowercase [a-z0-9_] identifier of at most 80 characters")
    return value


def _wang(raw: object, key: str) -> object:
    """Validate and copy a Wang descriptor ``{a, b, corners}`` (or ``None``)."""
    if raw is None:
        return None
    corners = raw.get("corners") if isinstance(raw, Mapping) else None
    if not (isinstance(corners, (list, tuple)) and len(corners) == 4 and all(type(c) is int and c in (0, 1) for c in corners)):
        raise ValueError(f"tile {key[:60]!r} wang must be {{a, b, corners: [ne, se, sw, nw] of 0/1}}")
    a, b = (_ident(raw.get(k), f"tile {key[:60]!r} wang.{k}") for k in "ab")
    return {"a": a, "b": b, "corners": list(corners)}


def _cells(col: int, row: int, cw: int, ch: int, step: int) -> list:
    return [{"col": col + i, "row": row + j, "x": (col + i) * step, "y": (row + j) * step} for j in range(ch) for i in range(cw)]


def _tile_record(index: int, p: Mapping, n: int, spacing: int) -> dict:
    key = p["key"]
    tid = _ident(p.get("id"), f"tile {key[:60]!r} id")
    variant = _int(p.get("variant"), f"tile {key[:60]!r} variant", 0, 255)
    if p.get("category") not in CATEGORIES:
        raise ValueError(f"tile {key[:60]!r} category must be one of {CATEGORIES}, got {p.get('category')!r}")
    wang = _wang(p.get("wang"), key)
    cw, ch, col, row, step = int(p["cells_w"]), int(p["cells_h"]), int(p["col"]), int(p["row"]), n + spacing
    return {
        "index": index, "id": tid, "name": sanitize.clean_text(p.get("name", tid), 40) or tid,
        "category": p["category"], "variant": variant, "tileable": p["category"] == "ground" and wang is None,
        "footprint": [cw, ch], "x": col * step, "y": row * step,
        "w": cw * n if spacing == 0 else n, "h": ch * n if spacing == 0 else n,
        "cells": _cells(col, row, cw, ch, step), "wang": wang,
    }  # fmt: skip


def build_metadata(
    placements: Sequence[Mapping], n: int, columns: int, spacing: int, palette: np.ndarray,
    view: str = "", model: str = "", seed: int = 0, theme: str = "",
) -> dict:  # fmt: skip
    """Return the metadata dict (``format`` ``ccn-tilemap/1``) for packed ``placements``.

    Placement needs ``id`` (slug), ``variant`` (0..255), ``category``; optional
    ``name`` (``clean_text``, default id) and ``wang`` (``None`` or ``{a, b,
    corners}``). ``tileable`` is derived (ground, not Wang). With ``spacing`` 0,
    ``x, y, w, h`` cover the whole footprint, otherwise cell 0; ``cells`` lists
    every cell. ``view``/``model``/``theme`` are cleaned; result is JSON-safe.
    """
    n, columns, spacing = _geometry(n, columns, spacing)
    rows = _check_placements(placements, columns)
    height, width = _dims(rows, columns, n, spacing)
    tiles = [_tile_record(i, p, n, spacing) for i, p in enumerate(placements)]
    if len({(t["id"], t["variant"]) for t in tiles}) != len(tiles):
        raise ValueError("tile id/variant pairs must be unique")
    return {
        "format": FORMAT, "tile_size": n, "columns": columns, "rows": rows, "spacing": spacing,
        "width": width, "height": height, "view": sanitize.clean_text(view, 40),
        "model": sanitize.clean_text(model, 100), "seed": _int(seed, "seed", 0, 2**64 - 1),
        "theme": sanitize.clean_text(theme, 80), "palette": _hex_palette(palette), "tiles": tiles,
    }  # fmt: skip


def _check_tiles(meta: Mapping, n: int, spacing: int, columns: int, rows: int) -> list:
    """Check each metadata tile against the grid; return ``(id, category, [(x, y)])`` per tile."""
    tiles = meta.get("tiles")
    if not isinstance(tiles, list) or not tiles:
        raise ValueError("metadata tiles must be a non-empty list")
    step, claimed, ids, out = n + spacing, {}, set(), []
    for i, t in enumerate(tiles):
        if not isinstance(t, Mapping):
            raise ValueError(f"metadata tiles[{i}] must be an object")
        tid = _ident(t.get("id"), f"tiles[{i}].id")
        variant = _int(t.get("variant"), f"tile {tid!r} variant", 0, 255)
        fp, cells = t.get("footprint"), t.get("cells")
        if t.get("index") != i or t.get("category") not in CATEGORIES or (tid, variant) in ids:
            raise ValueError(f"tile {tid!r}/{variant}: bad index, category or duplicate id/variant")
        ids.add((tid, variant))
        if not (isinstance(fp, list) and len(fp) == 2 and isinstance(cells, list) and cells and isinstance(cells[0], Mapping)):
            raise ValueError(f"tile {tid!r}: footprint must be [w, h] with a cells list")
        cw, ch = (_int(v, f"tile {tid!r} footprint", 1, _CELLS_MAX) for v in fp)
        col = _int(cells[0].get("col"), f"tile {tid!r} col", 0, columns - cw)
        row = _int(cells[0].get("row"), f"tile {tid!r} row", 0, rows - ch)
        w, h = (cw * n, ch * n) if spacing == 0 else (n, n)
        if cells != _cells(col, row, cw, ch, step) or [t.get(k) for k in "xywh"] != [col * step, row * step, w, h]:
            raise ValueError(f"tile {tid!r}: x/y/w/h/cells do not match footprint and grid position")
        for cell in cells:
            if (cell["col"], cell["row"]) in claimed:
                raise ValueError(f"tiles {claimed[(cell['col'], cell['row'])]!r} and {tid!r} overlap")
            claimed[(cell["col"], cell["row"])] = tid
        out.append((tid, t["category"], [(c["x"], c["y"]) for c in cells]))
    return out


def _pack32(a: np.ndarray) -> np.ndarray:
    return (a[..., 0].astype(np.uint32) << 16) | (a[..., 1].astype(np.uint32) << 8) | a[..., 2]


def verify_atlas(rgba: np.ndarray, meta: Mapping, palette: np.ndarray, palette_size: int = 256) -> None:
    """Raise ``ValueError`` on the first broken invariant (returns ``None`` otherwise).

    Checks: shape ``(rows*n + (rows-1)*spacing, columns*n + (columns-1)*spacing, 4)``
    equals the metadata size; every metadata cell is exactly ``n x n`` on the
    ``n + spacing`` grid inside the image, no two tiles overlap; alpha is only 0
    or 255; opaque RGB is in ``palette`` (<= ``palette_size`` colours, equal to
    ``meta["palette"]``); ground cells are opaque; everything outside the claimed
    cells (empty cells, gaps) is transparent; ``(id, variant)`` is unique.
    """
    if not isinstance(rgba, np.ndarray) or rgba.dtype != np.uint8 or rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("atlas must be a uint8 array of shape HxWx4")
    if not isinstance(meta, Mapping) or meta.get("format") != FORMAT:
        raise ValueError(f"metadata format must be {FORMAT!r}")
    n, columns, spacing = _geometry(meta.get("tile_size"), meta.get("columns"), meta.get("spacing"))
    rows = _int(meta.get("rows"), "rows", 1, _ITEMS_MAX * _CELLS_MAX)
    height, width = _dims(rows, columns, n, spacing)
    if (meta.get("width"), meta.get("height")) != (width, height) or rgba.shape != (height, width, 4):
        raise ValueError(f"atlas must be exactly {width}x{height} px, got {rgba.shape[1]}x{rgba.shape[0]}")
    hexes = _hex_palette(palette)
    if len(hexes) > _int(palette_size, "palette_size", 1, 256) or meta.get("palette") != hexes:
        raise ValueError("palette exceeds palette_size or differs from the metadata palette")
    alpha = rgba[..., 3]
    if np.any((alpha != 0) & (alpha != 255)):
        raise ValueError("atlas alpha must be only 0 or 255")
    opaque = alpha == 255
    if not np.isin(_pack32(rgba)[opaque], _pack32(palette)).all():
        raise ValueError("atlas holds opaque colours that are not in the palette")
    owned = np.zeros((height, width), dtype=np.bool_)
    for tid, category, positions in _check_tiles(meta, n, spacing, columns, rows):
        for x, y in positions:
            if x + n > width or y + n > height:
                raise ValueError(f"tile {tid!r}: cell at ({x}, {y}) is not a full {n}x{n} block")
            owned[y : y + n, x : x + n] = True
            if category == "ground" and not opaque[y : y + n, x : x + n].all():
                raise ValueError(f"ground tile {tid!r} has transparent pixels")
    if np.any(alpha[~owned]):
        raise ValueError("atlas has visible pixels outside the cells claimed by metadata")
