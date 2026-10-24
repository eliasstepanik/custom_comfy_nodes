"""Tilemap load node: ``CCN_LoadTilemap`` reads a saved ``ccn-tilemap/1|2`` atlas (PNG + JSON sidecar) back in.

The node is thin. Everything that touches the disk or parses untrusted data lives in
``tilemap_core.tileset_io``: ``read_pair`` confines the path to ``<root>/tilemap/`` (``output``) or the input
dir (``input``), refuses absolute paths, ``..``, symlinks and oversize files; ``load_tileset`` decodes the PNG,
verifies it against the metadata (``atlas.verify_atlas``) and slices the tiles. This module only resolves the
trusted root through ComfyUI's ``folder_paths`` (imported lazily; no import-time IO, and torch is imported
only when the node runs) and converts the result to ComfyUI types. All failures are ``ValueError``.
"""

from __future__ import annotations

import hashlib
import json

from ..tilemap_core import tileset_io

CATEGORY = "custom_comfy_nodes/tilemap"
ROOT_DIR = "tilemap"  # same forced sub folder as nodes/tilemap_save.py
_SOURCES = ("output", "input")


def _root(source: object) -> tuple:
    """Return ``(trusted_root_dir, forced_subdir)``; ``output`` is forced below ``tilemap/`` like the Save node."""
    if source not in _SOURCES:
        raise ValueError(f"source must be one of {list(_SOURCES)}")
    try:
        import folder_paths  # ComfyUI only; imported lazily so this module imports anywhere
    except ImportError:
        raise ValueError("ComfyUI folder_paths is not available") from None
    if source == "output":
        return folder_paths.get_output_directory(), ROOT_DIR
    return folder_paths.get_input_directory(), ""


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _job_json(ts: tileset_io.Tileset, rel: str, strict: bool) -> str:
    """``job_json`` (+ ``base`` block) of a reproducible tileset; ``""`` for ``/1`` or plan-less files."""
    if not ts.reproducible:
        return ""
    job = tileset_io.jobs_from_metadata(ts.meta)
    job["base"] = tileset_io.base_block(ts, rel)
    if not strict:  # architecture-tileset-reuse section 2: strict=False forces palette_mode = refit
        job["snap"] = {**job["snap"], "palette_mode": "refit"}
        job["base"]["snap"] = {**job["base"]["snap"], "palette_mode": "refit"}
    return _json(job)


def _summary(ts: tileset_io.Tileset, rel: str) -> str:
    h, w = ts.rgba.shape[:2]
    lines = [
        f"{ts.fmt} {rel}: {w}x{h} px, tile_size {ts.meta.get('tile_size')}, {len(ts.cells)} tiles, palette {len(ts.palette)} colours",
        f"reproducible: {'true' if ts.reproducible else 'false'}",
    ]
    if not ts.reproducible:
        lines.append("no plan/prompts stored: improve, extend and restyle need a ccn-tilemap/2 file with a plan")
    lines += [f"warning: {w}" for w in ts.warnings]
    return "\n".join(lines)


class CCN_LoadTilemap:
    """Load a saved tilemap atlas (PNG + JSON sidecar) from the ComfyUI output or input folder."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": "tilemap/tilemap_00001_"}),
                "source": (list(_SOURCES), {"default": "output"}),
                "strict": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("atlas", "alpha", "tiles", "metadata", "job_json", "summary")
    OUTPUT_IS_LIST = (False, False, True, False, False, False)
    FUNCTION = "load"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, path, source, strict):
        """Re-run when the files change on disk; an unreadable pair always re-runs so ``load`` reports the error."""
        try:
            root, subdir = _root(source)
            png, text, _ = tileset_io.read_pair(root, path, subdir)
        except ValueError:
            return float("nan")
        return hashlib.sha256(png + b"\0" + text.encode("utf-8")).hexdigest()

    def load(self, path, source, strict):
        if not isinstance(strict, bool):
            raise ValueError("strict must be a boolean")
        from ._tensor_io import alpha_to_mask, np_to_image  # torch is imported here, not at module import

        root, subdir = _root(source)
        png, text, rel = tileset_io.read_pair(root, path, subdir)
        ts = tileset_io.load_tileset(png, text, strict=strict)
        return (
            np_to_image([ts.rgba]),
            alpha_to_mask(ts.rgba[..., 3]),
            [np_to_image([cell]) for cell in ts.cells.values()],
            _json(ts.meta),
            _job_json(ts, rel, strict),
            _summary(ts, rel),
        )


NODE_CLASS_MAPPINGS = {"CCN_LoadTilemap": CCN_LoadTilemap}
NODE_DISPLAY_NAME_MAPPINGS = {"CCN_LoadTilemap": "Load Tilemap (PNG + JSON)"}
