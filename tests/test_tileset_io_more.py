"""Tests for tilemap_core.tileset_io: ``load_tileset``, ``slice_cells``, ``base_block``.

Fixtures and helpers come from tests/test_tileset_io.py (``make_fixture``, ``IoCase``, ``png_of``, ``edited``);
``read_pair`` and ``jobs_from_metadata`` are tested there. Real ``atlas`` + PIL build the inputs; no disk access.

Residual gaps (not tested): PNG decompression-bomb handling beyond the header pixel cap
(``Image.DecompressionBombError``); ``RecursionError`` from deep JSON only on the default recursion limit;
``warnings`` of a non-strict load only ever carries the off-palette message (no other kind exists).
"""

import copy
import dataclasses
import io
from unittest import mock

import numpy as np
from PIL import Image, PngImagePlugin

from tests.test_tileset_io import KEYS, PAL, IoCase, edited, make_fixture, png_of
from tilemap_core import plan_schema, snap
from tilemap_core import tileset_io as T
from tilemap_core.pixelize import TILE_SIZES

_UNSET = object()


class TestLoadRoundTrip(IoCase):
    def test_round_trip_all_tile_sizes_and_spacings(self):
        for n in TILE_SIZES:
            for spacing in (0, 1, 3):
                fx = make_fixture(n=n, spacing=spacing)
                ts = T.load_tileset(fx.png, fx.text)
                self.assertEqual((ts.fmt, ts.reproducible, ts.palette_exact, ts.warnings), ("ccn-tilemap/2", True, True, ()), (n, spacing))
                self.assertTrue(np.array_equal(ts.rgba, fx.rgba))
                self.assertEqual((ts.rgba.dtype, ts.palette.dtype, list(ts.cells)), (np.uint8, np.uint8, KEYS))
                for key, img in fx.images.items():
                    self.assertTrue(np.array_equal(ts.cells[key], img), (n, spacing, key))
                self.assertEqual(ts.cells["o0#0"].shape, (2 * n, 2 * n, 4))
                self.assertTrue(np.array_equal(ts.palette, fx.pal))
                self.assertEqual(ts.meta, fx.meta)

    def test_reproducibility_data(self):
        fx = make_fixture()
        ts = T.load_tileset(fx.png, fx.text)
        self.assertEqual(ts.plan, plan_schema.validate_plan(fx.plan, 4))
        self.assertEqual((list(ts.jobs_by_key), list(ts.jobs_by_key.values())), (KEYS, fx.jobs))
        self.assertEqual(ts.snap, snap.check_snap_params({"palette_size": 3}))

    def test_deterministic_frozen_and_inputs_untouched(self):
        fx = make_fixture()
        png, text = fx.png, fx.text
        a, b = T.load_tileset(png, text), T.load_tileset(bytearray(png), text)
        self.assertEqual((a.meta, a.plan, a.jobs_by_key, a.snap), (b.meta, b.plan, b.jobs_by_key, b.snap))
        self.assertTrue(np.array_equal(a.rgba, b.rgba))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            a.fmt = "x"
        a.cells["g0#0"][:] = 7  # cells and rgba are independent copies
        a.rgba[:] = 9
        self.assertTrue(np.array_equal(T.load_tileset(png, text).cells["g0#0"], fx.images["g0#0"]))

    def test_png_text_chunks_are_ignored(self):
        fx = make_fixture()
        info = PngImagePlugin.PngInfo()
        info.add_text("prompt", '{"evil": true}')
        ts = T.load_tileset(png_of(fx.rgba, pnginfo=info), fx.text)
        self.assertEqual((ts.meta, ts.fmt), (fx.meta, "ccn-tilemap/2"))

    def test_palette_size_boundaries(self):
        for pal in ([(1, 2, 3)], [(i, i * 7 % 256, i * 13 % 256) for i in range(256)]):
            fx = make_fixture(pal=pal)
            self.assertEqual(len(T.load_tileset(fx.png, fx.text).palette), len(pal))
        fx = make_fixture()
        self.rejects(T.load_tileset, fx.png, edited(fx, lambda m: m.update(palette=["#%06x" % i for i in range(257)])), contains="1..256")

    def test_seed_boundary_and_stored_values_win(self):
        fx = make_fixture(seed=2**64 - 1)  # seeds wrap modulo 2**64
        ts = T.load_tileset(fx.png, fx.text)
        self.assertEqual([j["seed"] for j in ts.jobs_by_key.values()], [2**64 - 1, 0, 1, 2])
        job = T.load_tileset(fx.png, edited(fx, lambda m: m["tiles"][1].update(prompt="custom", seed=123))).jobs_by_key["g1#0"]
        self.assertEqual((job["prompt"], job["seed"]), ("custom", 123))

    def test_format_one_and_plan_less_load_not_reproducible(self):
        fx = make_fixture()
        ts = T.load_tileset(fx.png, edited(fx, lambda m: m.update(format="ccn-tilemap/1", snap={"bogus": 1})))
        self.assertEqual((ts.fmt, ts.reproducible, ts.plan, ts.jobs_by_key), ("ccn-tilemap/1", False, {}, {}))
        self.assertEqual((ts.snap, list(ts.cells)), (snap.check_snap_params(None), KEYS))  # /1: embedded snap is ignored
        ts = T.load_tileset(fx.png, edited(fx, lambda m: m.pop("plan")))
        self.assertEqual((ts.fmt, ts.reproducible, ts.plan, ts.jobs_by_key, ts.snap["palette_size"]), ("ccn-tilemap/2", False, {}, {}, 3))
        self.rejects(T.load_tileset, fx.png, edited(fx, lambda m: m.update(snap={"bogus": 1})), contains="unknown keys")
        for fmt in ("ccn-tilemap/3", None):
            self.rejects(T.load_tileset, fx.png, edited(fx, lambda m: m.update(format=fmt)), contains="format")
        self.rejects(T.load_tileset, fx.png, edited(fx, lambda m: m.pop("format")), contains="format")

    def test_plan_present_but_unreproducible_fails_whole_load(self):
        fx = make_fixture()
        for key in ("style_prefix", "cell_px"):
            self.rejects(T.load_tileset, fx.png, edited(fx, lambda m: m.pop(key)), contains=key)
        self.rejects(T.load_tileset, fx.png, edited(fx, lambda m: m["tiles"][0].pop("prompt")), contains="g0#0")


class TestLoadRejects(IoCase):
    def setUp(self):
        self.fx = make_fixture()

    def load(self, png=_UNSET, text=_UNSET, **kw):
        return T.load_tileset(self.fx.png if png is _UNSET else png, self.fx.text if text is _UNSET else text, **kw)

    def test_metadata_text_errors(self):
        for bad in (None, 5, b"{}", [], "", "{", "[]", "5", "null", '{"a": 1, "a": 2}', '{"a": NaN}', '{"a": Infinity}', '{"a": 123456789012345678901}', "[" * 100000):
            self.rejects(self.load, text=bad)
        self.rejects(self.load, text='{"a": 1, "a": 2}', contains="duplicate keys")
        self.rejects(self.load, text="[]", contains="JSON object")
        size = len(self.fx.text)
        with mock.patch.object(T, "MAX_METADATA_CHARS", size):
            self.assertEqual(self.load().fmt, "ccn-tilemap/2")
        with mock.patch.object(T, "MAX_METADATA_CHARS", size - 1):
            self.rejects(self.load, contains="at most")

    def test_integer_text_length_boundary(self):
        text = edited(self.fx, lambda m: m.update(extra="@@"))
        self.assertEqual(self.load(text=text.replace('"@@"', "1" * 20)).meta["extra"], int("1" * 20))
        self.rejects(self.load, text=text.replace('"@@"', "1" * 21), contains="not valid JSON")

    def test_png_errors(self):
        rgba, png = self.fx.rgba, self.fx.png
        for bad in (None, "png", 5, b"", bytearray(), b"not a png", png[: len(png) // 2]):
            self.rejects(self.load, png=bad)
        self.rejects(self.load, png=b"", contains="1..")
        self.rejects(self.load, png=png_of(rgba[..., :3]), contains="single-frame RGBA")
        other, tif, anim = rgba.copy(), io.BytesIO(), io.BytesIO()
        other[0, 0, :3] = PAL[1]
        Image.fromarray(rgba).save(tif, "TIFF")
        Image.fromarray(rgba).save(anim, "PNG", save_all=True, append_images=[Image.fromarray(other)])
        for data in (tif.getvalue(), anim.getvalue()):
            self.rejects(self.load, png=data, contains="single-frame RGBA")
        for name, size, delta in (("PNG_MAX_BYTES", len(png), 1), ("PIXELS_MAX", rgba.shape[0] * rgba.shape[1], 1)):
            with mock.patch.object(T, name, size):
                self.assertEqual(self.load().fmt, "ccn-tilemap/2")
            with mock.patch.object(T, name, size - delta):
                self.rejects(self.load, contains="PNG must be" if name == "PNG_MAX_BYTES" else "larger than")

    def test_palette_errors(self):
        for bad in (None, "#000000", [], {}, ["#000000", 5], ["#FFFFFF"], ["#fff"], ["000000"], ["#00000g"], ["#0000000"], [None]):
            self.rejects(self.load, text=edited(self.fx, lambda m: m.update(palette=bad)), contains="1..256")
        self.rejects(self.load, text=edited(self.fx, lambda m: m.pop("palette")), contains="1..256")

    def test_verify_invariants_are_fatal_in_both_modes(self):
        n, rgba = self.fx.n, self.fx.rgba
        half, hole, stray = rgba.copy(), rgba.copy(), rgba.copy()
        half[n, 0, 3] = 128  # inside v0, transparent quarter
        hole[0, 0, 3] = 0  # ground g0
        stray[n, n] = (*PAL[0], 255)  # row 1, col 1 belongs to no tile
        cases = {"exactly": rgba[:-1], "alpha must be only": half, "ground tile": hole, "outside the cells": stray}
        for want, arr in cases.items():
            for strict in (True, False):
                self.rejects(self.load, png=png_of(arr), strict=strict, contains=want)

    def test_metadata_structure_errors(self):
        muts = [
            lambda m: m.update(tiles=[]), lambda m: m.update(rows=3), lambda m: m.update(tile_size=9), lambda m: m.update(columns=0),
            lambda m: m.update(spacing=17), lambda m: m["tiles"][1].update(id="g0"), lambda m: m["tiles"][2].update(footprint=[3, 2]),
            lambda m: m["tiles"][0].update(variant=True), lambda m: m["tiles"][0].update(key="zzz#0"),
        ]
        for fn in muts:
            for strict in (True, False):
                self.rejects(self.load, text=edited(self.fx, fn), strict=strict)

    def test_strict_flag_off_palette_pixels(self):
        bad = self.fx.rgba.copy()
        bad[0, 0, :3] = (1, 1, 1)  # opaque ground pixels off palette
        bad[1, 1, :3] = (2, 2, 2)
        png = png_of(bad)
        self.rejects(self.load, png=png, contains="not in the palette")
        ts = self.load(png=png, strict=False)
        self.assertEqual((ts.palette_exact, len(ts.warnings), type(ts.warnings)), (False, 1, tuple))
        self.assertIn("2 opaque pixels", ts.warnings[0])
        self.assertIn("refit", ts.warnings[0])
        self.assertTrue(np.array_equal(ts.rgba, bad))  # original pixels kept, not the repaired copy
        self.assertEqual(tuple(ts.cells["g0#0"][0, 0, :3]), (1, 1, 1))
        self.assertEqual(T.base_block(ts, "tilemap/ts")["snap"]["palette_mode"], "refit")
        clean = self.load(strict=False)
        self.assertEqual((clean.palette_exact, clean.warnings), (True, ()))
        bad[1, 1, :3] = PAL[0]
        self.assertIn("1 opaque pixels", self.load(png=png_of(bad), strict=False).warnings[0])

    def test_colour_under_zero_alpha_is_not_checked(self):
        px = self.fx.rgba.copy()
        px[self.fx.n, 0, :3] = (77, 88, 99)  # transparent pixel of v0: any rgb is fine
        self.assertTrue(self.load(png=png_of(px)).palette_exact)


def meta_for(n=8, tiles=None):
    """Minimal metadata for ``slice_cells`` (checks only tile_size and the tile records)."""
    return {"tile_size": n, "tiles": [rec()] if tiles is None else tiles}


def rec(tid="a", variant=0, fp=(1, 1), cells=((0, 0),), **extra):
    return {"id": tid, "variant": variant, "footprint": list(fp), "cells": [{"x": x, "y": y} for x, y in cells], **extra}


class TestSliceCells(IoCase):
    def setUp(self):
        self.rgba = np.random.default_rng(5).integers(0, 256, size=(16, 24, 4), dtype=np.uint8)

    def test_footprint_cells_are_row_major(self):
        cells = [(x, y) for y in (8, 0) for x in (16, 0, 8)]  # scrambled grid positions, j -> (j // 3, j % 3)
        img = T.slice_cells(self.rgba, meta_for(tiles=[rec("b", 3, (3, 2), cells)]))["b#3"]
        self.assertEqual(img.shape, (16, 24, 4))
        for j, (x, y) in enumerate(cells):
            r, c = divmod(j, 3)
            self.assertTrue(np.array_equal(img[r * 8 : r * 8 + 8, c * 8 : c * 8 + 8], self.rgba[y : y + 8, x : x + 8]), j)

    def test_order_copy_and_no_mutation(self):
        before = self.rgba.copy()
        meta = meta_for(tiles=[rec("z", 1, cells=[(8, 8)]), rec("a", 0, key="a#0"), rec("m", 2, cells=[(16, 0)])])
        snapshot = copy.deepcopy(meta)
        out = T.slice_cells(self.rgba, meta)
        self.assertEqual((list(out), out["a#0"].dtype), (["z#1", "a#0", "m#2"], np.uint8))
        out["z#1"][:] = 0
        self.assertTrue(np.array_equal(self.rgba, before))
        self.assertEqual(meta, snapshot)

    def test_bounds_are_exact(self):
        edge = T.slice_cells(self.rgba, meta_for(tiles=[rec(cells=[(16, 8)])]))  # last full block of the 24x16 atlas
        self.assertTrue(np.array_equal(edge["a#0"], self.rgba[8:, 16:]))
        for xy in ((17, 8), (16, 9), (-1, 0), (0, -1), (24, 0), (0, 16)):
            self.rejects(T.slice_cells, self.rgba, meta_for(tiles=[rec(cells=[xy])]), contains="inside the atlas")

    def test_bad_cell_values(self):
        for bad in ({"x": 0.0, "y": 0}, {"x": True, "y": 0}, {"x": "0", "y": 0}, {"x": 0}, {"y": 0}, {}, None, 5, [0, 0]):
            self.rejects(T.slice_cells, self.rgba, meta_for(tiles=[{**rec(), "cells": [bad]}]), contains="cell 0")

    def test_bad_atlas_argument(self):
        for bad in (None, [[0]], self.rgba.astype(np.int32), self.rgba[..., :3], self.rgba[..., 0], self.rgba[None]):
            self.rejects(T.slice_cells, bad, meta_for(), contains="uint8 array of shape HxWx4")

    def test_bad_tile_size_and_tile_list(self):
        for n in (None, 0, 7, 9, 128, 8.0, True, "8"):
            self.rejects(T.slice_cells, self.rgba, {**meta_for(), "tile_size": n}, contains="tile_size")
        for n in TILE_SIZES:
            self.assertEqual(list(T.slice_cells(np.zeros((n, n, 4), dtype=np.uint8), meta_for(n))), ["a#0"])
        for tiles in (None, [], {}, "x", ()):
            self.rejects(T.slice_cells, self.rgba, {"tile_size": 8, "tiles": tiles}, contains="non-empty tiles list")
        for meta in (None, [], "x", {}):
            self.rejects(T.slice_cells, self.rgba, meta, contains="tile_size")

    def test_bad_tile_records(self):
        raw = {"id": "a", "variant": 0, "footprint": [1, 1], "cells": [{"x": 0, "y": 0}]}
        for tiles in ([5], [rec(tid=5)], [{k: v for k, v in raw.items() if k != "id"}], [rec(variant=True)], [rec(variant=1.0)], [rec(fp=(1,))],
                      [rec(fp=(0, 1), cells=[])], [rec(fp=(9, 1), cells=[(0, 0)] * 9)], [{**raw, "footprint": [True, 1]}], [{**raw, "footprint": "11"}],
                      [rec(fp=(2, 1), cells=[(0, 0)])], [{**raw, "cells": "x"}]):
            self.rejects(T.slice_cells, self.rgba, meta_for(tiles=tiles))
        self.rejects(T.slice_cells, self.rgba, meta_for(tiles=[rec(), rec()]), contains="duplicate key")
        self.rejects(T.slice_cells, self.rgba, meta_for(tiles=[rec(key="b#0")]), contains="does not match")

    def test_footprint_side_limit_of_eight(self):
        out = T.slice_cells(np.zeros((8, 64, 4), dtype=np.uint8), meta_for(tiles=[rec(fp=(8, 1), cells=[(8 * i, 0) for i in range(8)])]))
        self.assertEqual(out["a#0"].shape, (8, 64, 4))


class TestBaseBlock(IoCase):
    def setUp(self):
        self.fx = make_fixture()
        self.ts = T.load_tileset(self.fx.png, self.fx.text)

    def test_block_contents_and_independence(self):
        before = copy.deepcopy(self.ts.meta)
        block = T.base_block(self.ts, "tilemap/ts.png")
        self.assertEqual((block["path"], block["keys"], block["palette"], block["snap"]), ("tilemap/ts", KEYS, self.fx.meta["palette"], self.ts.snap))
        block["keys"].append("x")
        block["palette"].append("#000000")
        block["snap"]["palette_size"] = 99
        self.assertEqual((self.ts.meta, self.ts.snap["palette_size"], list(self.ts.cells)), (before, 3, KEYS))

    def test_keys_keep_metadata_order(self):
        ts = dataclasses.replace(self.ts, cells=dict(reversed(list(self.ts.cells.items()))))
        self.assertEqual(T.base_block(ts, "ts")["keys"], KEYS[::-1])

    def test_palette_mode_follows_exactness(self):
        self.assertEqual(T.base_block(self.ts, "ts")["snap"]["palette_mode"], self.ts.snap["palette_mode"])
        bad = self.fx.rgba.copy()
        bad[0, 0, :3] = (1, 1, 1)
        inexact = T.load_tileset(png_of(bad), self.fx.text, strict=False)
        self.assertEqual(T.base_block(inexact, "ts")["snap"]["palette_mode"], "refit")
        self.assertNotEqual(self.ts.snap["palette_mode"], "refit")  # exact set keeps the stored mode
        self.assertNotEqual(inexact.snap["palette_mode"], "refit")  # the Tileset itself is never rewritten

    def test_path_is_cleaned_not_forced_into_a_subdir(self):
        for given, rel in (("ts", "ts"), ("ts.png", "ts"), ("a/b/ts.json", "a/b/ts"), ("tilemap/ts", "tilemap/ts")):
            self.assertEqual(T.base_block(self.ts, given)["path"], rel, given)
        for bad in (None, 5, "", "a/b/c/d", "../ts", "/ts", "a//b", "a\\b", "a b", "x" * 65, "ts.png.png"):
            self.rejects(T.base_block, self.ts, bad)

    def test_requires_tileset(self):
        for bad in (None, {}, self.ts.meta, "ts"):
            self.rejects(T.base_block, bad, "ts", contains="Tileset")
