"""Render a ``ccn-scene/1`` test scene from tile cells, and draw the labelled tile grid.

numpy + PIL + stdlib only (PIL for the grid labels), deterministic, no I/O, no torch.
Images are ``uint8`` numpy arrays. The *catalogue* of ``pick_ground`` / ``render_scene`` is the
atlas metadata (``atlas.build_metadata``): tile records ``{id, variant, category, footprint,
wang}``. Wang records (``wang = {a, b, corners}``, id ``<a>_<b>``, ``variant`` = mask) follow the
corner bit order of ``transitions.wang_tiles``: mask = ``ne | se << 1 | sw << 2 | nw << 3``, 1 = ``b``.
``cells`` maps ``"<id>#<variant>"`` to the RGBA cell image ``(footprint_h * n, footprint_w * n, 4)``
(alpha only 0 or 255). Bad input raises ``ValueError``; nothing is clamped or printed.
"""

import zlib
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import scene_layout
from .pixelize import TILE_SIZES

__all__ = ["UPSCALE_MAX", "PIXELS_MAX", "pick_ground", "render_scene", "label_grid"]

UPSCALE_MAX = 16
PIXELS_MAX = 1 << 26  # output pixels (8192 x 8192), as the atlas
_GRID_BG = {"black": (0, 0, 0), "white": (255, 255, 255), "gray": (128, 128, 128)}


def _int(value: Any, name: str, lo: int, hi: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}")
    if not lo <= int(value) <= hi:
        raise ValueError(f"{name} must be in {lo}..{hi}, got {int(value)}")
    return int(value)


def _hash(col: int, row: int, seed: int) -> int:
    return zlib.crc32(f"{seed}:{col}:{row}".encode())


class _Index:
    """Lookup tables over the metadata tile records."""

    def __init__(self, meta: Any) -> None:
        tiles = meta.get("tiles") if isinstance(meta, Mapping) else None
        if not isinstance(tiles, list) or not tiles:
            raise ValueError("catalogue (atlas metadata) needs a non-empty 'tiles' list")
        self.plain: dict = {}  # ground id -> sorted variants
        self.wang: dict = {}  # (a, b) -> (wang id, {mask})
        self.items: dict = {}  # object/overlay id -> (category, (fw, fh), sorted variants)
        for i, t in enumerate(tiles):
            if not isinstance(t, Mapping) or not isinstance(t.get("id"), str):
                raise ValueError(f"tiles[{i}] must be an object with a string id")
            tid, variant, wang = t["id"], _int(t.get("variant"), f"tiles[{i}].variant", 0, 255), t.get("wang")
            if t.get("category") == "ground" and wang is None:
                self.plain.setdefault(tid, []).append(variant)
            elif t.get("category") == "ground":
                self._add_wang(tid, variant, wang)
            elif t.get("category") in ("object", "overlay"):
                fw, fh = (_int(v, f"tile {tid!r} footprint", 1, 8) for v in t.get("footprint", (0, 0)))
                cat, fp, vs = self.items.setdefault(tid, (t["category"], (fw, fh), []))
                if (cat, fp) != (t["category"], (fw, fh)):
                    raise ValueError(f"tile {tid!r}: variants disagree on category or footprint")
                vs.append(variant)
        for vs in self.plain.values():
            vs.sort()
        self.items = {k: (c, fp, sorted(vs)) for k, (c, fp, vs) in self.items.items()}

    def _add_wang(self, tid: str, variant: int, wang: Any) -> None:
        a, b = (wang.get(k) for k in "ab") if isinstance(wang, Mapping) else (None, None)
        want = [(variant >> bit) & 1 for bit in range(4)]
        if not (isinstance(a, str) and isinstance(b, str)) or list(wang.get("corners", ())) != want:
            raise ValueError(f"tile {tid!r}/{variant}: wang corners must be the bits of the mask in order ne, se, sw, nw")
        wid, masks = self.wang.setdefault((a, b), (tid, set()))
        if wid != tid:
            raise ValueError(f"two wang sets for the pair {a!r}+{b!r}: {wid!r} and {tid!r}")
        masks.add(variant)

    def plain_variants(self, gid: str) -> list:
        if gid not in self.plain:
            raise ValueError(f"no plain ground tile for id {gid!r} in the catalogue")
        return self.plain[gid]


def _pick(idx: _Index, corners: Sequence, col: int, row: int, seed: int) -> Optional[tuple]:
    if any(c is None for c in corners):
        return None
    ids = sorted(set(corners))
    if len(ids) == 1:
        vs = idx.plain_variants(ids[0])
        return ids[0], vs[_hash(col, row, seed) % len(vs)], None
    if len(ids) == 2:
        for a, b in ((ids[0], ids[1]), (ids[1], ids[0])):  # the set (b, a) inverts the mask
            if (a, b) in idx.wang:
                wid, masks = idx.wang[(a, b)]
                mask = scene_layout.corner_mask(*(int(c == b) for c in corners))
                if mask not in masks:
                    raise ValueError(f"wang set {wid!r} has no tile for mask {mask}")
                return wid, mask, None
    count = Counter(corners)
    gid = min(count, key=lambda g: (-count[g], g))
    vs = idx.plain_variants(gid)
    why = "three or more ground ids" if len(ids) > 2 else "no transition set for the pair"
    warn = f"cell ({col}, {row}): corners {list(corners)}: {why}; drawn as majority ground {gid!r}"
    return gid, vs[_hash(col, row, seed) % len(vs)], warn


def pick_ground(corners: Sequence, catalogue: Any, col: int, row: int, seed: int) -> Optional[tuple]:
    """Ground tile of cell ``(col, row)`` for its corners ``(ne, se, sw, nw)``.

    Returns ``(tile_id, variant, warning)``; ``None`` when a corner is ``None`` (no ground, shows the
    background). Four equal ids: plain tile, variant ``crc32(seed:col:row) % variants``. Two ids with
    a wang set ``(a, b)`` (tried in sorted order, then ``(b, a)``, mask inverted): ``(wang id, mask)``.
    Else the majority id (ties: smallest id) as plain tile plus a warning string, otherwise ``None``.
    """
    if not isinstance(corners, Sequence) or len(corners) != 4 or not all(c is None or isinstance(c, str) for c in corners):
        raise ValueError("corners must be four ground ids or None (ne, se, sw, nw)")
    return _pick(_Index(catalogue), corners, _int(col, "col", 0, 1 << 31), _int(row, "row", 0, 1 << 31), _int(seed, "seed", 0, 2**64 - 1))


def _cell(cells: Mapping, tid: str, variant: int, shape: tuple) -> np.ndarray:
    key = f"{tid}#{variant}"
    img = cells.get(key)
    if not isinstance(img, np.ndarray) or img.dtype != np.uint8 or img.shape != shape:
        raise ValueError(f"cells[{key!r}] must be a uint8 array of shape {shape}")
    if ((img[..., 3] != 0) & (img[..., 3] != 255)).any():
        raise ValueError(f"cells[{key!r}] alpha must be only 0 or 255")
    return img


def _blit(canvas: np.ndarray, img: np.ndarray, col: int, row: int, n: int) -> None:
    h, w = img.shape[:2]
    region = canvas[row * n : row * n + h, col * n : col * n + w]
    solid = img[..., 3] == 255
    region[solid] = img[solid]


def _check_scene(scene: Any, idx: _Index) -> tuple:
    """Validate what the renderer reads; return ``(width, height, seed, background rgb)``."""
    if not isinstance(scene, Mapping) or scene.get("format") != scene_layout.FORMAT:
        raise ValueError(f"scene must be a {scene_layout.FORMAT!r} object")
    w = _int(scene.get("width"), "width", scene_layout.DIM_MIN, scene_layout.DIM_MAX)
    h = _int(scene.get("height"), "height", scene_layout.DIM_MIN, scene_layout.DIM_MAX)
    seed = _int(scene.get("seed", 0), "seed", 0, 2**64 - 1)
    terrain = scene.get("terrain")
    if not isinstance(terrain, list) or len(terrain) != h + 1:
        raise ValueError(f"terrain must be a list of height+1 = {h + 1} rows")
    for r, line in enumerate(terrain):
        if not isinstance(line, list) or len(line) != w + 1:
            raise ValueError(f"terrain[{r}] must be a list of width+1 = {w + 1} entries")
        for c, v in enumerate(line):
            if v is not None and not (isinstance(v, str) and v in idx.plain):
                raise ValueError(f"terrain[{r}][{c}] must be a ground id with a plain tile, or null, got {v!r}")
    bg = scene.get("background", scene_layout.BACKGROUND)
    if not isinstance(bg, str) or len(bg) != 7 or bg[0] != "#":
        raise ValueError(f"background must be a #rrggbb colour, got {bg!r}")
    try:
        rgb = tuple(int(bg[i : i + 2], 16) for i in (1, 3, 5))
    except ValueError:
        raise ValueError(f"background must be a #rrggbb colour, got {bg!r}") from None
    return w, h, seed, rgb


def _place(scene: Mapping, key: str, category: str, idx: _Index, cells: Mapping, canvas: np.ndarray, n: int, seed: int) -> None:
    w, h = canvas.shape[1] // n, canvas.shape[0] // n
    items = scene.get(key) or []
    if not isinstance(items, list):
        raise ValueError(f"{key} must be a list")
    for i, it in enumerate(items):
        tid = it.get("id") if isinstance(it, Mapping) else None
        if tid not in idx.items or idx.items[tid][0] != category:
            raise ValueError(f"{key}[{i}].id must be an existing {category} tile id, got {tid!r}")
        _, (fw, fh), variants = idx.items[tid]
        col, row = _int(it.get("col"), f"{key}[{i}].col", 0, w - fw), _int(it.get("row"), f"{key}[{i}].row", 0, h - fh)
        variant = it.get("variant")
        if variant is None:
            variant = variants[_hash(col, row, seed) % len(variants)]
        elif _int(variant, f"{key}[{i}].variant", 0, 255) not in variants:
            raise ValueError(f"{key}[{i}].variant {variant} does not exist for tile {tid!r}")
        _blit(canvas, _cell(cells, tid, variant, (fh * n, fw * n, 4)), col, row, n)


def render_scene(scene: Mapping, cells: Mapping, meta: Mapping, n: int, upscale: int = 1) -> tuple:
    """Render ``scene`` to ``(rgba, warnings)``: ``uint8`` ``(height*n*upscale, width*n*upscale, 4)``.

    Order: background fill (opaque), ground per cell (``pick_ground``), objects, overlays; a source pixel
    with alpha 255 replaces, alpha 0 leaves the pixel. ``upscale`` repeats pixels (nearest neighbour).
    ``warnings`` lists every cell drawn as majority ground, in row-major order. Output larger than
    ``PIXELS_MAX`` pixels raises ``ValueError``; inputs are never mutated.
    """
    n = _int(n, "n", 1, max(TILE_SIZES))
    if n not in TILE_SIZES:
        raise ValueError(f"n must be one of {TILE_SIZES}, got {n}")
    upscale = _int(upscale, "upscale", 1, UPSCALE_MAX)
    if not isinstance(cells, Mapping):
        raise ValueError("cells must be a dict of '<id>#<variant>' -> RGBA arrays")
    idx = _Index(meta)
    w, h, seed, bg = _check_scene(scene, idx)
    if w * h * (n * upscale) ** 2 > PIXELS_MAX:
        raise ValueError(f"scene would be {w * n * upscale}x{h * n * upscale} px: too large, lower upscale")
    canvas = np.empty((h * n, w * n, 4), dtype=np.uint8)
    canvas[...] = (*bg, 255)
    warnings = []
    for row in range(h):
        for col in range(w):
            picked = _pick(idx, scene_layout.cell_corners(scene["terrain"], col, row), col, row, seed)
            if picked is not None:
                tid, variant, warn = picked
                _blit(canvas, _cell(cells, tid, variant, (n, n, 4)), col, row, n)
                warnings += [warn] if warn else []
    _place(scene, "objects", "object", idx, cells, canvas, n, seed)
    _place(scene, "overlays", "overlay", idx, cells, canvas, n, seed)
    return np.repeat(np.repeat(canvas, upscale, axis=0), upscale, axis=1), warnings


def label_grid(images: Sequence, columns: int, padding: int, background: str, show_index: bool) -> np.ndarray:
    """Tile ``images`` (``uint8`` ``HxWx3|4``, all one shape) into a ``uint8`` RGB grid.

    Cell ``i`` sits at ``padding + (i % columns) * (w + padding)`` / ``padding + (i // columns) * (h + padding)``;
    with ``show_index`` its 1-based number is drawn in the top-left corner (white on black, PIL default font).
    Same drawing as the ``CCN_ImageGridLabeled`` node: size ``columns*w + (columns+1)*padding`` by
    ``rows*h + (rows+1)*padding`` with ``rows = ceil(count / columns)``.
    """
    columns, padding = _int(columns, "columns", 1, 16), _int(padding, "padding", 0, 256)
    if background not in _GRID_BG or not isinstance(show_index, bool):
        raise ValueError(f"background must be one of {sorted(_GRID_BG)} and show_index a bool")
    imgs = [images[i] for i in range(len(images))] if isinstance(images, (np.ndarray, list, tuple)) else None
    if not imgs:
        raise ValueError("images must be a non-empty list or batch of uint8 arrays")
    h, w = imgs[0].shape[:2] if isinstance(imgs[0], np.ndarray) and imgs[0].ndim == 3 else (0, 0)
    for i, im in enumerate(imgs):
        if not isinstance(im, np.ndarray) or im.dtype != np.uint8 or im.ndim != 3 or im.shape[2] not in (3, 4) or im.shape[:2] != (h, w) or not h:
            raise ValueError(f"images[{i}] must be a uint8 array of shape ({h}, {w}, 3|4), the shape of images[0]")
    rows = -(-len(imgs) // columns)
    grid = Image.new("RGB", (columns * w + (columns + 1) * padding, rows * h + (rows + 1) * padding), _GRID_BG[background])
    draw, font = ImageDraw.Draw(grid), ImageFont.load_default()
    for i, im in enumerate(imgs):
        x, y = padding + (i % columns) * (w + padding), padding + (i // columns) * (h + padding)
        grid.paste(Image.fromarray(np.ascontiguousarray(im[:, :, :3])), (x, y))
        if show_index:
            label = str(i + 1)
            right, bottom = draw.textbbox((x + 2, y + 2), label, font=font)[2:]
            draw.rectangle((x, y, right + 2, bottom + 2), fill=(0, 0, 0))
            draw.text((x + 2, y + 2), label, fill=(255, 255, 255), font=font)
    return np.asarray(grid)
