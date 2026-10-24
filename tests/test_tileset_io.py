"""Tests for tilemap_core.tileset_io: ``read_pair`` (confinement, caps, symlinks) and ``jobs_from_metadata``.

Companion: tests/test_tileset_io_more.py (``load_tileset``, ``slice_cells``, ``base_block``); it imports
the fixture builder ``make_fixture``, ``edited``, ``IoCase`` from here. Fixtures are built with the real
``atlas.pack/assemble/build_metadata`` + PIL; all files live in ``tempfile`` directories.

Residual gaps (not tested): the ``commonpath`` escape check in ``read_pair`` (unreachable while every
directory component is lstat-verified as a real directory); the ``OSError`` branch of ``_read_regular`` for
unreadable files (tests run as a user that may bypass permissions); the post-open ``fstat`` regular-file
re-check (a race against file replacement); ``O_NOFOLLOW``/``O_CLOEXEC`` flag availability on non-POSIX;
huge PNG decode-bomb behaviour beyond the header pixel cap.
"""

import copy
import io
import json
import os
import unittest
from dataclasses import dataclass
from unittest import mock

import numpy as np
from PIL import Image

from tests import _support
from tilemap_core import atlas, expand, plan_schema, snap
from tilemap_core import tileset_io as T

PAL = [(10, 20, 30), (200, 100, 50), (90, 180, 60)]
STYLE = "pixel art"
KEYS = ["g0#0", "g1#0", "o0#0", "v0#0"]


@dataclass
class Fixture:
    n: int
    spacing: int
    columns: int
    plan: dict
    jobs: list
    placements: list
    images: dict
    rgba: np.ndarray
    meta: dict
    pal: np.ndarray

    @property
    def png(self) -> bytes:
        return png_of(self.rgba)

    @property
    def text(self) -> str:
        return json.dumps(self.meta)


def png_of(rgba: np.ndarray, **save) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, "PNG", **save)
    return buf.getvalue()


def make_fixture(n: int = 8, spacing: int = 0, columns: int = 4, pal=PAL, seed: int = 100) -> Fixture:
    """Real saved-style tileset: ground g0/g1 (1x1), object o0 (2x2, top-left quarter transparent), overlay v0."""
    plan = _support.valid_plan()
    plan["tiles"][2]["footprint"] = [2, 2]
    jobs = expand.expand_jobs(plan, n, "top-down", STYLE, seed, 512)
    items, images = [], {}
    for j in jobs:
        cw, ch = j["footprint"]
        key = f"{j['tile_id']}#{j['variant']}"
        rows = [np.hstack([_support.opaque_rgba_tile(n, j["index"] * 31 + r * 7 + c, pal) for c in range(cw)]) for r in range(ch)]
        img = np.vstack(rows)
        if j["category"] != "ground":
            img[: n // 2, : n // 2] = 0
        images[key] = img
        items.append({"key": key, "cells_w": cw, "cells_h": ch, "id": j["tile_id"], "name": j["name"], "category": j["category"],
                      "variant": j["variant"], "prompt": j["prompt"], "seed": j["seed"], "attempt": 0, "source": "generated",
                      "job_index": j["index"]})  # fmt: skip
    placements = atlas.pack(items, columns)
    palette = np.array(pal, dtype=np.uint8)
    rgba = atlas.assemble(placements, images, n, columns, spacing)
    extra = {"style_prefix": STYLE, "cell_px": 512, "plan": plan, "snap": {"palette_size": 3}}
    meta = atlas.build_metadata(placements, n, columns, spacing, palette, view="top-down", model="m", seed=seed, theme="forest", extra=extra)
    return Fixture(n, spacing, columns, plan, jobs, placements, images, rgba, meta, palette)


def edited(fx, fn) -> str:
    """Metadata text of ``fx`` after ``fn(meta)`` mutated a deep copy."""
    meta = copy.deepcopy(fx.meta)
    fn(meta)
    return json.dumps(meta)


class IoCase(unittest.TestCase):
    def rejects(self, fn, *args, contains=None, exact=True, **kw):
        """Assert ``fn`` raises ``ValueError`` (exactly that type unless ``exact=False``) whose message holds ``contains``."""
        with self.assertRaises(ValueError) as cm:
            fn(*args, **kw)
        if exact:
            self.assertIs(type(cm.exception), ValueError)
        if contains is not None:
            self.assertIn(contains, str(cm.exception))
        return cm.exception

    def put(self, root, rel, data):
        path = os.path.join(root, *rel.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(data if isinstance(data, bytes) else data.encode("utf-8"))
        return path


class TestReadPair(IoCase):
    def setUp(self):
        self.fx = make_fixture()

    def store(self, root, rel="tilemap/ts"):
        self.put(root, rel + ".png", self.fx.png)
        self.put(root, rel + ".json", self.fx.text)

    def test_reads_pair_and_returns_clean_rel(self):
        with _support.temp_dir() as root:
            self.store(root)
            for given in ("ts", "tilemap/ts", "ts.png", "ts.json", "tilemap/ts.png", "tilemap/ts.json"):
                png, text, rel = T.read_pair(root, given)
                self.assertEqual((png, text, rel), (self.fx.png, self.fx.text, "tilemap/ts"), given)

    def test_nested_folder_and_segment_limit(self):
        with _support.temp_dir() as root:
            self.store(root, "tilemap/a/ts")
            self.assertEqual(T.read_pair(root, "a/ts")[2], "tilemap/a/ts")
            self.rejects(T.read_pair, root, "a/b/ts", contains="at most 3 segments")  # becomes tilemap/a/b/ts: 4 parts
            self.rejects(T.read_pair, root, "tilemap/a/b/ts", contains="at most 3 segments")
            self.rejects(T.read_pair, root, "tilemap", contains="file name")

    def test_empty_subdir_means_no_forced_folder(self):
        with _support.temp_dir() as root:
            self.store(root, "ts")
            self.store(root, "x/y/ts")
            for given in ("ts", "x/y/ts"):
                self.assertEqual(T.read_pair(root, given, subdir="")[2], given)
            self.rejects(T.read_pair, root, "a/x/y/ts", subdir="", contains="at most 3 segments")
            self.rejects(T.read_pair, root, "ts")  # default subdir: tilemap/ts does not exist

    def test_bad_paths(self):
        with _support.temp_dir() as root:
            self.store(root)
            for bad in (None, 5, b"ts", ["ts"], "", "../ts", "/ts", "tilemap/../ts", "a//ts", "a\\ts", "a b", "a/./ts", "\u00e4", "ts.png.png", "ts/", ".png", "a" * 65):
                self.rejects(T.read_pair, root, bad)
            self.rejects(T.read_pair, root, "a" * 300, contains="1..256")
            self.assertNotIn("path must be a text", str(self.rejects(T.read_pair, root, "a/" * 128)))  # 256 chars: length ok
            self.rejects(T.read_pair, root, "a" * 64 + "/" + "b" * 64 + "/" + "c" * 64)  # 3 + forced tilemap = 4 segments

    def test_segment_length_boundary(self):
        with _support.temp_dir() as root:
            self.store(root, "tilemap/" + "s" * 64)
            self.assertEqual(T.read_pair(root, "s" * 64)[2], "tilemap/" + "s" * 64)
            self.rejects(T.read_pair, root, "s" * 65)

    def test_bad_root_and_subdir(self):
        with _support.temp_dir() as root:
            self.store(root)
            for bad_root in (None, 5, "", os.path.join(root, "missing"), os.path.join(root, "tilemap", "ts.png"), b"/tmp"):
                self.rejects(T.read_pair, bad_root, "ts", contains="root must be an existing directory")
            for bad_sub in (None, 5, "a/b", "..", "a b", "x" * 65):
                self.rejects(T.read_pair, root, "ts", subdir=bad_sub, contains="subdir")

    def test_missing_files_are_value_errors_not_oserror(self):
        with _support.temp_dir() as root:
            self.rejects(T.read_pair, root, "ts", contains="tileset folder was not found")
            os.makedirs(os.path.join(root, "tilemap"))
            self.rejects(T.read_pair, root, "ts", contains="ts.png was not found")
            self.put(root, "tilemap/ts.png", self.fx.png)
            self.rejects(T.read_pair, root, "ts", contains="metadata sidecar")

    def test_sidecar_must_be_utf8(self):
        with _support.temp_dir() as root:
            self.put(root, "tilemap/ts.png", self.fx.png)
            self.put(root, "tilemap/ts.json", b"{\xff\xfe}")
            self.rejects(T.read_pair, root, "ts", contains="UTF-8")

    def test_size_caps_are_inclusive(self):
        with _support.temp_dir() as root:
            self.put(root, "tilemap/ts.png", b"p" * 10)
            self.put(root, "tilemap/ts.json", b"j" * 6)
            with mock.patch.object(T, "PNG_MAX_BYTES", 10), mock.patch.object(T, "MAX_METADATA_CHARS", 6):
                self.assertEqual(T.read_pair(root, "ts")[:2], (b"p" * 10, "j" * 6))
            with mock.patch.object(T, "PNG_MAX_BYTES", 9):
                self.rejects(T.read_pair, root, "ts", contains="ts.png")
            with mock.patch.object(T, "MAX_METADATA_CHARS", 5):
                self.rejects(T.read_pair, root, "ts", contains="metadata sidecar")

    def test_module_constants(self):
        self.assertEqual(T.PNG_MAX_BYTES, 64 * 1024 * 1024)
        self.assertEqual(T.MAX_METADATA_CHARS, 1_048_576)
        self.assertEqual(T.PIXELS_MAX, 8192 * 8192)
        self.assertEqual(sorted(T.__all__), sorted(["PNG_MAX_BYTES", "MAX_METADATA_CHARS", "Tileset", "read_pair", "load_tileset", "slice_cells", "jobs_from_metadata", "base_block"]))
        self.assertTrue(all(hasattr(T, name) for name in T.__all__))

    def test_symlinks_and_special_files_refused(self):
        with _support.temp_dir() as root, _support.temp_dir() as outside:
            self.store(outside, "real/ts")
            self.store(root, "tilemap/ok")
            os.symlink(os.path.join(outside, "real"), os.path.join(root, "tilemap", "linkdir"))  # directory symlink
            self.rejects(T.read_pair, root, "linkdir/ts", contains="symlink")
            os.symlink(os.path.join(outside, "real", "ts.png"), os.path.join(root, "tilemap", "lp.png"))
            self.put(root, "tilemap/lp.json", self.fx.text)
            self.rejects(T.read_pair, root, "lp", contains="symlinks are refused")
            self.put(root, "tilemap/lj.png", self.fx.png)
            os.symlink(os.path.join(outside, "real", "ts.json"), os.path.join(root, "tilemap", "lj.json"))
            self.rejects(T.read_pair, root, "lj", contains="symlinks are refused")
            os.symlink(os.path.join(root, "tilemap"), os.path.join(root, "alias"))  # symlink inside root
            self.rejects(T.read_pair, root, "alias/ok", subdir="", contains="symlink")
            os.makedirs(os.path.join(root, "tilemap", "d.png"))  # directory with the PNG name
            self.put(root, "tilemap/d.json", self.fx.text)
            self.rejects(T.read_pair, root, "d", contains="not a regular file")
            os.mkfifo(os.path.join(root, "tilemap", "f.png"))  # must be refused without blocking on open
            self.put(root, "tilemap/f.json", self.fx.text)
            self.rejects(T.read_pair, root, "f", contains="not a regular file")
            self.assertEqual(T.read_pair(root, "ok")[2], "tilemap/ok")  # untouched neighbours still load

    def test_symlink_swapped_in_after_lstat_is_refused_by_open(self):
        with _support.temp_dir() as root, _support.temp_dir() as outside:
            self.store(outside, "real/ts")
            self.store(root, "tilemap/ts")
            os.remove(os.path.join(root, "tilemap", "ts.png"))
            os.symlink(os.path.join(outside, "real", "ts.png"), os.path.join(root, "tilemap", "ts.png"))
            regular = os.lstat(os.path.join(outside, "real", "ts.json"))  # lstat says "regular file"; the real entry is a symlink
            with mock.patch.object(T.os, "lstat", side_effect=lambda p: regular if p.endswith("ts.png") else os.stat(p)):
                self.rejects(T.read_pair, root, "ts", contains="cannot be opened")

    def test_root_may_itself_be_a_symlink_and_dir_is_not_modified(self):
        with _support.temp_dir() as real, _support.temp_dir() as holder:
            self.store(real)
            link = os.path.join(holder, "rootlink")
            os.symlink(real, link)
            before = sorted(os.listdir(os.path.join(real, "tilemap")))
            self.assertEqual(T.read_pair(link, "ts")[2], "tilemap/ts")
            self.assertEqual(sorted(os.listdir(os.path.join(real, "tilemap"))), before)


class TestJobsFromMetadata(IoCase):
    def setUp(self):
        self.fx = make_fixture()
        self.meta = copy.deepcopy(self.fx.meta)

    def test_rebuilds_job_json(self):
        out = T.jobs_from_metadata(self.meta)
        self.assertEqual(self.meta, self.fx.meta)  # input not mutated
        self.assertEqual(sorted(out), sorted(["tile_size", "view", "plan", "jobs", "seed", "style_prefix", "cell_px", "snap", "model"]))
        self.assertEqual(out["jobs"], self.fx.jobs)
        self.assertEqual((out["tile_size"], out["view"], out["seed"], out["style_prefix"], out["cell_px"]), (8, "top-down", 100, "pixel art", 512))
        self.assertEqual((out["model"], out["plan"]), ("m", plan_schema.validate_plan(self.fx.plan, 4)))
        self.assertEqual(out["snap"], snap.check_snap_params({"palette_size": 3}))
        self.assertEqual(out, T.jobs_from_metadata(copy.deepcopy(self.meta)))

    def test_stored_prompt_and_seed_replace_generated(self):
        self.meta["tiles"][0].update(prompt="p!", seed=2**64 - 1)
        job = T.jobs_from_metadata(self.meta)["jobs"][0]
        self.assertEqual((job["prompt"], job["seed"], job["tile_id"]), ("p!", 2**64 - 1, "g0"))

    def test_model_and_snap_defaults(self):
        del self.meta["snap"]
        self.meta["model"] = 5
        out = T.jobs_from_metadata(self.meta)
        self.assertEqual((out["model"], out["snap"]), ("", snap.check_snap_params(None)))

    def test_junk_tile_entries_and_extra_records_are_ignored(self):
        self.meta["tiles"] += ["junk", {"id": "other", "variant": 0}]
        self.assertEqual(T.jobs_from_metadata(self.meta)["jobs"], self.fx.jobs)

    def test_rejects_wrong_format_and_missing_blocks(self):
        for bad in (None, [], "x", {}, {"format": "ccn-tilemap/1"}, {**self.meta, "format": "ccn-tilemap/3"}):
            self.rejects(T.jobs_from_metadata, bad, contains="ccn-tilemap/")
        self.rejects(T.jobs_from_metadata, {**self.meta, "format": "ccn-tilemap/1"}, contains="can only be rebuilt from ccn-tilemap/2")
        for key in ("plan", "style_prefix", "cell_px"):
            self.rejects(T.jobs_from_metadata, {k: v for k, v in self.meta.items() if k != key}, contains=key)
        self.rejects(T.jobs_from_metadata, {k: v for k, v in self.meta.items() if k not in ("plan", "cell_px")}, contains="not reproducible")

    def test_invalid_plan_and_expand_arguments(self):
        for plan in (None, [], {}, {"theme": "t", "tiles": []}):
            self.rejects(T.jobs_from_metadata, {**self.meta, "plan": plan}, exact=False)
        for key, bad in (("tile_size", 9), ("tile_size", None), ("view", "iso"), ("view", None), ("seed", None), ("seed", -1), ("seed", 2**64),
                         ("cell_px", 255), ("cell_px", 1025), ("cell_px", True), ("cell_px", "512")):
            self.rejects(T.jobs_from_metadata, {**self.meta, key: bad}, exact=False)
        for key, ok in (("cell_px", 256), ("cell_px", 1024), ("seed", 0), ("seed", 2**64 - 1)):
            self.assertEqual(len(T.jobs_from_metadata({**self.meta, key: ok})["jobs"]), 4)

    def test_rejects_unusable_records(self):
        base = self.meta["tiles"]
        for field, bad in (("prompt", ""), ("prompt", None), ("prompt", 5), ("seed", -1), ("seed", 2**64), ("seed", True), ("seed", 1.0), ("seed", "1")):
            meta = copy.deepcopy(self.meta)
            meta["tiles"][2][field] = bad
            self.rejects(T.jobs_from_metadata, meta, contains="o0#0")
        meta = copy.deepcopy(self.meta)
        meta["tiles"][1]["seed"] = 0
        self.assertEqual(T.jobs_from_metadata(meta)["jobs"][1]["seed"], 0)
        for i, name in enumerate(KEYS):
            self.rejects(T.jobs_from_metadata, {**self.meta, "tiles": base[:i] + base[i + 1 :]}, contains=name)
        self.rejects(T.jobs_from_metadata, {**self.meta, "tiles": [{**base[0], "id": 5}]}, contains="string id")
