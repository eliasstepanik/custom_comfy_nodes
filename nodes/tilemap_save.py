"""Tilemap save node: ``CCN_SaveTilemap`` writes the atlas as RGBA PNG plus its metadata JSON.

Files go ONLY below ``<ComfyUI output>/tilemap/``. The prefix is reduced by ``sanitize_prefix``, every
directory component must be a real directory (symlinks are refused, no ``realpath`` escape), names are
``<stem>_NNNNN_.png`` / ``.json`` and are claimed with ``O_CREAT|O_EXCL`` so nothing is ever overwritten.
All validation and encoding happens in memory first; data is written to a temp file in the target dir and
moved with ``os.replace``, and any failure removes every file this call created. No import-time IO.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import stat
import tempfile

import numpy as np
import torch
from PIL import Image, PngImagePlugin

from ..tilemap_core import atlas as tile_atlas
from ..tilemap_core import sanitize
from ._tensor_io import image_list_to_np, mask_to_alpha

try:  # ComfyUI provides these; a plain test harness does not
    import folder_paths
except ImportError:
    folder_paths = None
try:
    from comfy.cli_args import args
except ImportError:
    args = None

CATEGORY = "custom_comfy_nodes/tilemap"
ROOT_DIR = "tilemap"
MAX_METADATA_CHARS = 1_048_576
_HEX = re.compile(r"^#[0-9a-f]{6}$")
_TEXT_KEY = re.compile(r"^[A-Za-z0-9_-]{1,79}$")
_CLAIM_TRIES = 1000


def _output_root() -> str:
    """Real path of the ComfyUI output dir (a temp dir when running without ComfyUI)."""
    getter = getattr(folder_paths, "get_output_directory", None)
    base = getter() if getter is not None else os.path.join(tempfile.gettempdir(), "ccn_tilemap_output")
    return os.path.realpath(base)


def _rgba(atlas: torch.Tensor, alpha: torch.Tensor) -> np.ndarray:
    """Combine the IMAGE (RGB or RGBA) and the MASK (1 = opaque) into one uint8 HxWx4 array."""
    if not isinstance(atlas, torch.Tensor) or atlas.ndim != 4 or atlas.shape[0] != 1:
        raise ValueError("atlas must be a single IMAGE of shape [1,H,W,C]")
    if not isinstance(alpha, torch.Tensor) or alpha.ndim not in (2, 3) or (alpha.ndim == 3 and alpha.shape[0] != 1):
        raise ValueError("alpha must be a single MASK of shape [1,H,W]")
    img = image_list_to_np(atlas, channels=4 if atlas.shape[-1] == 4 else 3)[0]
    a = mask_to_alpha(alpha)
    if a.shape != img.shape[:2]:
        raise ValueError(f"alpha is {a.shape[1]}x{a.shape[0]} but atlas is {img.shape[1]}x{img.shape[0]}")
    if img.shape[2] == 4 and not np.array_equal(img[..., 3], a):
        raise ValueError("atlas alpha channel differs from the alpha mask")
    return np.ascontiguousarray(np.dstack([img[..., :3], a]))


def _parse_metadata(text: str) -> dict:
    if not isinstance(text, str):
        raise ValueError("metadata must be a string")
    if len(text) > MAX_METADATA_CHARS:
        raise ValueError(f"metadata is larger than {MAX_METADATA_CHARS} characters")
    try:
        meta = json.loads(text)
    except (ValueError, RecursionError):
        raise ValueError("metadata is not valid JSON") from None
    if not isinstance(meta, dict):
        raise ValueError("metadata must be a JSON object")
    return meta


def _validate(rgba: np.ndarray, meta: dict) -> None:
    """Check the atlas against the metadata (palette format, then the full ``verify_atlas`` invariants)."""
    hexes = meta.get("palette")
    if not isinstance(hexes, list) or not 1 <= len(hexes) <= 256:
        raise ValueError("metadata palette must be a list of 1..256 colours")
    if not all(isinstance(h, str) and _HEX.match(h) for h in hexes):
        raise ValueError("metadata palette entries must look like '#rrggbb'")
    palette = np.array([[int(h[i : i + 2], 16) for i in (1, 3, 5)] for h in hexes], dtype=np.uint8)
    try:
        tile_atlas.verify_atlas(rgba, meta, palette, palette_size=len(hexes))
    except ValueError as exc:
        raise ValueError(f"metadata does not match the atlas: {str(exc)[:200]}") from None


def _pnginfo(prompt: object, extra_pnginfo: object):
    """Workflow text chunks like ComfyUI's SaveImage (none with ``--disable-metadata``)."""
    if getattr(args, "disable_metadata", False):
        return None
    info = PngImagePlugin.PngInfo()
    if prompt is not None:
        info.add_text("prompt", json.dumps(prompt, default=str))
    if isinstance(extra_pnginfo, dict):
        for key, value in extra_pnginfo.items():
            if isinstance(key, str) and _TEXT_KEY.match(key):
                info.add_text(key, json.dumps(value, default=str))
    return info


def _encode_png(rgba: np.ndarray, info) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG", pnginfo=info, compress_level=6)
    return buf.getvalue()


def _target_dir(root: str, parts: list) -> str:
    """Create/validate ``root/parts...``: every component a real directory, final path inside ``root/tilemap``."""
    cur = root
    for part in parts:
        cur = os.path.join(cur, part)
        with contextlib.suppress(FileExistsError):
            os.mkdir(cur, 0o755)
        if not stat.S_ISDIR(os.lstat(cur).st_mode):  # lstat: a symlink is not a directory here
            raise ValueError("refusing to save through a symlink or a non-directory")
    base = os.path.join(root, ROOT_DIR)
    if os.path.commonpath([os.path.realpath(cur), base]) != base:
        raise ValueError("save path resolves outside the output tilemap directory")
    return cur


def _claim(path: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    os.close(os.open(path, flags, 0o644))


def _claim_names(folder: str, stem: str) -> tuple:
    """Reserve a fresh ``(png_path, json_path)`` pair with the next free counter."""
    pattern = re.compile(rf"^{re.escape(stem)}_(\d{{5,}})_\.(?:png|json)$")
    used = [int(m.group(1)) for m in map(pattern.match, os.listdir(folder)) if m]
    counter = max(used, default=0) + 1
    for _ in range(_CLAIM_TRIES):
        png = os.path.join(folder, f"{stem}_{counter:05d}_.png")
        js = os.path.join(folder, f"{stem}_{counter:05d}_.json")
        counter += 1
        try:
            _claim(png)
        except FileExistsError:
            continue
        try:
            _claim(js)
        except FileExistsError:
            os.unlink(png)
            continue
        return png, js
    raise RuntimeError("could not reserve a free output file name")


def _write_atomic(path: str, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".ccn_", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


class CCN_SaveTilemap:
    """Save the tilemap atlas (RGBA PNG) and its metadata (JSON) under ``output/tilemap/``."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "atlas": ("IMAGE",),
                "alpha": ("MASK",),
                "metadata": ("STRING", {"forceInput": True}),
                "filename_prefix": ("STRING", {"default": "tilemap/tilemap"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    def save(self, atlas, alpha, metadata, filename_prefix, prompt=None, extra_pnginfo=None):
        rgba = _rgba(atlas, alpha)
        meta = _parse_metadata(metadata)
        _validate(rgba, meta)
        parts = sanitize.sanitize_prefix(filename_prefix).split("/")
        png_data = _encode_png(rgba, _pnginfo(prompt, extra_pnginfo))
        json_data = json.dumps(meta, indent=1, allow_nan=False).encode("ascii")
        folder = _target_dir(_output_root(), parts[:-1])
        png_path, json_path = _claim_names(folder, parts[-1])
        try:
            _write_atomic(json_path, json_data)
            _write_atomic(png_path, png_data)
        except BaseException:
            for path in (png_path, json_path):
                with contextlib.suppress(OSError):
                    os.unlink(path)
            raise
        shown = {"filename": os.path.basename(png_path), "subfolder": "/".join(parts[:-1]), "type": "output"}
        return {"ui": {"images": [shown]}}


NODE_CLASS_MAPPINGS = {"CCN_SaveTilemap": CCN_SaveTilemap}
NODE_DISPLAY_NAME_MAPPINGS = {"CCN_SaveTilemap": "Save Tilemap (PNG + JSON)"}
