"""Tests for tilemap_core.atlas beyond tests/test_atlas_transitions.py (T0): limits, field-level metadata, verify_atlas.

Reuses the T0 helpers (fixture, item, ground, px, tw, at, ...); atlas_meta is covered by test_atlas_meta.py.

Residual gaps (not tested): the ``x + n > width`` guard inside verify_atlas (unreachable: cells are re-derived from
the grid and bounds-checked first); the exact wording of every error message; ``_PIXELS_MAX`` inside ``verify_atlas``
(shares ``_dims`` with ``assemble``, tested there); behaviour for items that are Mapping but not dict subclasses is
only sampled (MappingProxyType), not every Mapping type.
"""

import copy
import json
import unittest
from types import MappingProxyType

import numpy as np

from tests.test_atlas_transitions import PAL, PA, PB, SIZES, VIEW, Base, at, fixture, ground, item, px, tw
from tilemap_core import atlas, transitions


def cell(key, w=1, h=1, **kw):
    return {"key": key, "cells_w": w, "cells_h": h, **kw}


def one(n=8, cat="ground", pal=PAL, **kw):
    """Single 1x1 item: (placements, images, rgba atlas 2 columns, metadata)."""
    pl = atlas.pack([item("g", cat, 0, **kw)], 2)
    imgs = {"g#0": ground(n, 1, PA)}
    return pl, imgs, atlas.assemble(pl, imgs, n, 2, 0), atlas.build_metadata(pl, n, 2, 0, pal)


class PackExtraTests(Base):
    def test_footprint_and_column_boundaries(self):
        out = atlas.pack([cell("a", 8, 8), cell("b", 8, 1)], 8)
        self.assertEqual([(p["col"], p["row"]) for p in out], [(0, 0), (0, 8)])
        self.assertEqual(atlas.pack([cell("a", 1, 8), cell("b", 1, 8), cell("c", 1, 1)], 2)[2]["row"], 8)
        self.rejects(atlas.pack, [([cell("a", 9, 1)], 64), ([cell("a", 1, 9)], 64), ([cell("a", 8, 1)], 7)])
        self.assertEqual([p["col"] for p in atlas.pack([cell(str(i)) for i in range(5)], 1)], [0] * 5)
        self.assertEqual([p["row"] for p in atlas.pack([cell(str(i)) for i in range(5)], 1)], list(range(5)))

    def test_tall_tile_is_backfilled_beside_and_wide_waits(self):
        out = atlas.pack([cell("t", 1, 3), cell("w", 3, 1), cell("s"), cell("s2")], 3)
        self.assertEqual([(p["key"], p["col"], p["row"]) for p in out], [("t", 0, 0), ("w", 0, 3), ("s", 1, 0), ("s2", 2, 0)])

    def test_key_length_types_and_item_shapes(self):
        for k in ("k", "k" * 200):
            self.assertEqual(atlas.pack([cell(k)], 1)[0]["key"], k)
        self.rejects(atlas.pack, [([cell("k" * 201)], 1), ([cell(None)], 1), ([cell(5)], 1), ([cell(b"k")], 1)])
        self.rejects(atlas.pack, [(v, 2) for v in (None, "ab", b"ab", {"key": "k"}, 5, [None], ["k"], [[("key", "k")]], iter([cell("k")]))])
        self.assertEqual(len(atlas.pack((cell("a"), cell("b")), 2)), 2)  # any Sequence
        self.assertEqual(atlas.pack([MappingProxyType(cell("a"))], 2)[0]["key"], "a")  # any Mapping item
        self.rejects(atlas.pack, [([cell("a", w=np.float64(1))], 2), ([cell("a", h=True)], 2), ([cell("a", w=None)], 2), ([{"key": "a"}], 2)])
        self.assertEqual(atlas.pack([cell("a", np.int64(2), np.int8(1))], 2)[0]["cells_w"], 2)

    def test_output_carries_only_known_keys_and_ignores_incoming_position(self):
        src = cell("a", id="g", name="N", category="ground", variant=2, wang=None, prompt="p", seed=1, attempt=0, source="kept", job_index=4,
                   junk=1, col=7, row=9, tileable=True)  # fmt: skip
        out = atlas.pack([src], 3)[0]
        self.assertEqual(set(out), {"key", "cells_w", "cells_h", "col", "row", "id", "name", "category", "variant", "wang", "prompt", "seed",
                                    "attempt", "source", "job_index"})  # fmt: skip
        self.assertEqual((out["col"], out["row"]), (0, 0))
        self.assertIsNot(out, src)
        self.assertEqual(atlas.pack([cell("a")], 1)[0], {"key": "a", "cells_w": 1, "cells_h": 1, "col": 0, "row": 0})
        self.assertIs(type(atlas.pack([cell("a", np.int64(1))], np.int64(2))[0]["cells_w"]), int)

    def test_error_messages_name_tile(self):
        with self.assertRaisesRegex(ValueError, "duplicate key 'a'"):
            atlas.pack([cell("a"), cell("a")], 2)
        with self.assertRaisesRegex(ValueError, "entry 1"):
            atlas.pack([cell("a"), 5], 2)
        with self.assertRaisesRegex(ValueError, "x.*cells_w"):
            atlas.pack([cell("x", 0)], 2)


class AssembleExtraTests(Base):
    def test_spacing_gap_pixels_and_cell_origin(self):
        for n, sp in ((8, 1), (16, 16), (64, 3)):
            img = np.full((n, n, 4), 255, np.uint8)
            pl = [{**cell("a"), "col": 1, "row": 1}]
            out = atlas.assemble(pl, {"a": img}, n, 2, sp)
            st = n + sp
            self.assertEqual(out.shape, (2 * n + sp, 2 * n + sp, 4))
            self.assertTrue((out[st : st + n, st : st + n] == 255).all())
            self.assertEqual(int((out != 0).sum()), n * n * 4)
            self.assertTrue(out[:st].sum() == 0 and out[:, :st].sum() == 0)

    def test_multi_cell_footprint_split_and_row_order(self):
        n = 8
        big = np.arange(2 * n * 3 * n * 4, dtype=np.uint32).reshape(2 * n, 3 * n, 4).astype(np.uint8)
        small = np.full((n, n, 4), 9, np.uint8)
        pl = [{**cell("s"), "col": 0, "row": 0}, {**cell("b", 3, 2), "col": 1, "row": 0}]
        out = atlas.assemble(pl, {"b": big, "s": small}, n, 4, 0)  # image dict order differs from placement order
        self.assertEqual(out.shape, (2 * n, 4 * n, 4))
        self.assertTrue(np.array_equal(out[:, n:], big) and np.array_equal(out[:n, :n], small) and not out[n:, :n].any())
        sp = atlas.assemble(pl, {"b": big, "s": small}, n, 4, 2)
        self.assertTrue(np.array_equal(sp[n + 2 : 2 * n + 2, 2 * (n + 2) : 2 * (n + 2) + n], big[n:, n : 2 * n]))
        self.assertFalse(sp[n : n + 2].any() and sp[:, n : n + 2].any())

    def test_placement_validation(self):
        img = {"k": np.zeros((8, 8, 4), np.uint8)}
        good = {**cell("k"), "col": 0, "row": 0}
        self.assertEqual(atlas.assemble([good], img, 8, 1, 0).shape, (8, 8, 4))
        bad = [{**good, "col": 1}, {**good, "col": -1}, {**good, "row": -1}, {**good, "col": "0"}, {**good, "row": 0.0}, {**good, "col": True},
               {**good, "col": None}, {k: v for k, v in good.items() if k != "row"}, {**good, "cells_w": 2}]  # fmt: skip
        self.rejects(atlas.assemble, [([b], img, 8, 1, 0) for b in bad])
        self.rejects(atlas.assemble, [(v, img, 8, 1, 0) for v in ([], None, "k", {"k": good}, [None], [good, good])])
        self.rejects(atlas.assemble, [([good], v, 8, 1, 0) for v in (None, [("k", img["k"])], [img["k"]])])
        self.assertEqual(atlas.assemble(((good),), MappingProxyType(img), 8, 1, 0).shape, (8, 8, 4))

    def test_image_validation_and_no_alias(self):
        good = {**cell("k", 2, 1), "col": 0, "row": 0}
        mk = lambda a: atlas.assemble([good], {"k": a}, 8, 2, 0)  # noqa: E731
        ok = np.ones((8, 16, 4), np.uint8)
        ok.flags.writeable = False
        self.assertEqual(mk(ok).shape, (8, 16, 4))
        self.rejects(mk, [(a,) for a in (ok[:, :, :3], ok[:, :8], ok.T, ok.astype(np.uint16), ok.astype(bool), ok[None], ok.tolist(), None, "x")])
        out = mk(ok)
        out[:] = 7
        self.assertEqual(int(ok.max()), 1)

    def test_rows_follow_tallest_placement_and_gap_rows_stay_transparent(self):
        img = {"a": np.full((8, 8, 4), 5, np.uint8)}
        out = atlas.assemble([{**cell("a"), "col": 0, "row": 3}], img, 8, 1, 0)
        self.assertEqual(out.shape, (32, 8, 4))
        self.assertTrue(not out[:24].any() and (out[24:] == 5).all())

    def test_pixel_cap_and_geometry_boundaries(self):
        img = {"a": np.zeros((8, 8, 4), np.uint8)}
        pl = [{**cell("a"), "col": 0, "row": 0}]
        for n in SIZES:
            self.assertEqual(atlas.assemble(pl, {"a": np.zeros((n, n, 4), np.uint8)}, n, 1, 0).shape, (n, n, 4))
        for g in ((0, 1, 0), (-8, 1, 0), (8, 0, 0), (8, 1, 17), (8, 65, 0), (80, 1, 0), (8, "1", 0), (8, 1, None)):
            self.rejects(atlas.assemble, [(pl, img, *g)])
        self.assertEqual(atlas.assemble(pl, img, 8, 64, 16).shape, (8, 64 * 8 + 63 * 16, 4))


class MetadataExtraTests(Base):
    def test_palette_hex_and_sizes(self):
        pal = np.array([[0, 0, 0], [255, 255, 255], [1, 2, 10], [171, 205, 239]], np.uint8)
        pl = atlas.pack([item("g", "ground", 0)], 1)
        self.assertEqual(atlas.build_metadata(pl, 8, 1, 0, pal)["palette"], ["#000000", "#ffffff", "#01020a", "#abcdef"])
        self.assertEqual(len(atlas.build_metadata(pl, 8, 1, 0, pal[:1])["palette"]), 1)
        self.assertEqual(len(atlas.build_metadata(pl, 8, 1, 0, np.zeros((256, 3), np.uint8))["palette"]), 256)
        for p in (pal[:0], np.zeros((257, 3), np.uint8), pal[:, :2], np.zeros((4, 4), np.uint8), pal[0], pal[None], pal.astype(int), pal.tolist(), None):
            self.rejects(atlas.build_metadata, [(pl, 8, 1, 0, p)])

    def test_tile_fields_spacing_zero_and_positive(self):
        pl = atlas.pack([item("a", "ground", 0), item("t", "object", 0, 2, 3), item("o", "overlay", 2)], 3)
        for n in (8, 24, 64):
            m0, m3 = (atlas.build_metadata(pl, n, 3, sp, PAL) for sp in (0, 3))
            self.assertEqual((m0["rows"], m0["width"], m0["height"]), (3, 3 * n, 3 * n))
            self.assertEqual((m3["width"], m3["height"]), (3 * n + 6, 3 * n + 6))
            t0, t3 = m0["tiles"][1], m3["tiles"][1]
            self.assertEqual((t0["x"], t0["y"], t0["w"], t0["h"]), (n, 0, 2 * n, 3 * n))
            self.assertEqual((t3["x"], t3["y"], t3["w"], t3["h"]), (n + 3, 0, n, n))
            self.assertEqual([(c["col"], c["row"], c["x"], c["y"]) for c in t3["cells"]],
                             [(1 + i, j, (1 + i) * (n + 3), j * (n + 3)) for j in range(3) for i in range(2)])  # fmt: skip
            self.assertEqual([t["index"] for t in m0["tiles"]], [0, 1, 2])
            self.assertEqual([t["category"] for t in m0["tiles"]], ["ground", "object", "overlay"])
            self.assertEqual([t["variant"] for t in m0["tiles"]], [0, 0, 2])

    def test_defaults_names_and_view_trimming(self):
        pl = atlas.pack([cell("k", id="g", category="ground", variant=0)], 1)
        m = atlas.build_metadata(pl, 8, 1, 0, PAL)
        self.assertEqual((m["view"], m["model"], m["seed"], m["theme"], m["tiles"][0]["name"], m["tiles"][0]["wang"]), ("", "", 0, "", "g", None))
        m = atlas.build_metadata(pl, 8, 1, 0, PAL, "  top   down  ", "m" * 300, 5, "t" * 300)
        self.assertEqual((m["view"], len(m["model"]), len(m["theme"]), m["seed"]), ("top down", 100, 80, 5))
        self.rejects(atlas.build_metadata, [(pl, 8, 1, 0, PAL, v) for v in (None, 5, b"x")])

    def test_identifier_and_variant_boundaries(self):
        base = atlas.pack([item("g", "ground", 0)], 1)[0]
        for tid, v in (("a", 0), ("a1_b", 255), ("x" * 80, 0), ("0", 1)):
            self.assertEqual(atlas.build_metadata([{**base, "id": tid, "variant": v}], 8, 1, 0, PAL)["tiles"][0]["id"], tid)
        for cat in ("ground", "object", "overlay"):
            self.assertEqual(atlas.build_metadata([{**base, "category": cat}], 8, 1, 0, PAL)["tiles"][0]["tileable"], cat == "ground")
        self.rejects(atlas.build_metadata, [([{**base, "id": "x" * 81}], 8, 1, 0, PAL), ([{**base, "variant": 256}], 8, 1, 0, PAL),
                                            ([{**base, "category": None}], 8, 1, 0, PAL), ([{k: v for k, v in base.items() if k != "id"}], 8, 1, 0, PAL)])  # fmt: skip

    def test_wang_descriptor_copy_and_tileable(self):
        w = {"a": "g0", "b": "g1", "corners": (1, 0, 1, 0), "junk": 1}
        pl = [{**atlas.pack([item("g", "ground", 5)], 1)[0], "wang": w}]
        t = atlas.build_metadata(pl, 8, 1, 0, PAL)["tiles"][0]
        self.assertEqual((t["wang"], t["tileable"]), ({"a": "g0", "b": "g1", "corners": [1, 0, 1, 0]}, False))
        t["wang"]["corners"][0] = 0
        self.assertEqual(w["corners"], (1, 0, 1, 0))
        for bad in ({"a": "g0", "b": "g1", "corners": [0, 0, 0, 2]}, {"a": "g0", "b": "g1", "corners": [0, 0, 0, 0, 0]},
                    {"a": "g0", "b": "g1", "corners": np.array([0, 0, 0, 0])}, {"a": "G0", "b": "g1", "corners": [0] * 4}, {"corners": [0] * 4}, []):  # fmt: skip
            self.rejects(atlas.build_metadata, [([{**pl[0], "wang": bad}], 8, 1, 0, PAL)])

    def test_json_round_trip_all_16_wang_masks_verifies(self):
        for n in (8, 64):
            for sp in (0, 5):
                _, pl, _, img, meta = fixture(n, sp) if n == 8 else (None,) * 5
                if meta is None:
                    g0, g1 = ground(n, 1), ground(n, 4, PB)
                    its = [item("g0", "ground", 0), item("g1", "ground", 0)] + [
                        item("w", "ground", m, wang={"a": "g0", "b": "g1", "corners": list(c)}) for m, (c, _) in enumerate(transitions.wang_tiles(g0, g1, n, 2))]  # fmt: skip
                    imgs = {"g0#0": g0, "g1#0": g1, **{f"w#{m}": t for m, (_, t) in enumerate(transitions.wang_tiles(g0, g1, n, 2))}}
                    pl = atlas.pack(its, 6)
                    meta = atlas.build_metadata(pl, n, 6, sp, PAL, *VIEW)
                    img = atlas.assemble(pl, imgs, n, 6, sp)
                back = json.loads(json.dumps(meta))
                self.assertIsNone(atlas.verify_atlas(img, back, PAL, 6))
                wt = [t for t in back["tiles"] if t["wang"]]
                self.assertEqual(sorted(t["variant"] for t in wt), list(range(16)))
                self.assertEqual({tuple(t["wang"]["corners"]) for t in wt}, {(m & 1, m >> 1 & 1, m >> 2 & 1, m >> 3 & 1) for m in range(16)})
                self.assertEqual(len({(t["x"], t["y"]) for t in back["tiles"]}), len(back["tiles"]))

    def test_item_count_limits_and_determinism(self):
        mk = lambda k: atlas.pack([item(f"t{i}", "ground", 0) for i in range(k)], 64)  # noqa: E731
        self.assertEqual(atlas.build_metadata(mk(512), 8, 64, 0, PAL)["rows"], 8)
        self.rejects(atlas.build_metadata, [([], 8, 64, 0, PAL), (mk(1) * 513, 8, 64, 0, PAL), ("x", 8, 64, 0, PAL), (None, 8, 64, 0, PAL)])
        pl = mk(3)
        self.assertEqual(atlas.build_metadata(pl, 8, 64, 0, PAL), atlas.build_metadata(copy.deepcopy(pl), 8, 64, 0, PAL))
        self.rejects(atlas.build_metadata, [(pl[:1] + [{**pl[1], "col": 0, "row": 0}], 8, 64, 0, PAL)])  # overlap


class VerifyExtraTests(Base):
    def test_uniform_palette_binary_alpha_and_v1_format(self):
        pal = np.array([[7, 8, 9]], np.uint8)
        pl = atlas.pack([item("g", "ground", 0), item("o", "object", 0)], 3)
        solid = np.zeros((8, 8, 4), np.uint8)
        solid[...] = (7, 8, 9, 255)
        ring = solid.copy()
        ring[:4] = 0  # object with transparent half
        imgs = {"g#0": solid, "o#0": ring}
        img, meta = atlas.assemble(pl, imgs, 8, 3, 0), atlas.build_metadata(pl, 8, 3, 0, pal)
        self.assertIsNone(atlas.verify_atlas(img, meta, pal, 1))
        self.assertIsNone(atlas.verify_atlas(img, {**meta, "format": "ccn-tilemap/1"}, pal))
        self.rejects(atlas.verify_atlas, [(img, {**meta, "format": f}, pal) for f in ("ccn-tilemap/3", None, "")])
        for a in (1, 127, 254):
            self.rejects(atlas.verify_atlas, [(px(img, 1, 1, 7, 8, 9, a), meta, pal)])
        self.rejects(atlas.verify_atlas, [(px(img, 1, 1, 7, 8, 10, 255), meta, pal)])

    def test_ground_must_be_opaque_but_objects_may_not(self):
        pl, imgs, img, meta = one(8)
        self.assertIsNone(atlas.verify_atlas(img, meta, PAL))
        self.rejects(atlas.verify_atlas, [(px(img, 7, 7, 0, 0, 0, 0), meta, PAL), (px(img, 0, 0, 0, 0, 0, 0), meta, PAL)])
        pl, imgs, img, meta = one(8, "overlay")
        clear = img.copy()
        clear[:4] = 0
        self.assertIsNone(atlas.verify_atlas(clear, meta, PAL))

    def test_palette_size_bounds_and_types(self):
        _, _, img, meta = one(8)
        for ps in (6, 256):
            self.assertIsNone(atlas.verify_atlas(img, meta, PAL, ps))
        self.rejects(atlas.verify_atlas, [(img, meta, PAL, ps) for ps in (0, -1, 257, 5, True, 6.0, "6", None)])
        self.assertIsNone(atlas.verify_atlas(img, meta, PAL, np.int64(6)))

    def test_rejects_bad_array_and_meta_shapes(self):
        _, _, img, meta = one(8)
        self.rejects(atlas.verify_atlas, [(v, meta, PAL) for v in (None, img.tolist(), img[0], img[None], img.astype(np.int32), img[..., :2])])
        self.rejects(atlas.verify_atlas, [(img, v, PAL) for v in (None, [], "x", {}, {"format": "ccn-tilemap/2"})])
        for k in ("tile_size", "columns", "rows", "spacing", "width", "height", "palette", "tiles"):
            self.rejects(atlas.verify_atlas, [(img, {a: b for a, b in meta.items() if a != k}, PAL)])
        for k, v in (("tile_size", 9), ("tile_size", 8.0), ("columns", 0), ("columns", 65), ("rows", 0), ("spacing", 17), ("spacing", True), ("height", None)):
            self.rejects(atlas.verify_atlas, [(img, {**meta, k: v}, PAL)])
        self.rejects(atlas.verify_atlas, [(img, meta, v) for v in (None, PAL.tolist(), PAL[:0], PAL[:, :2], PAL.astype(np.int16))])

    def test_rejects_tile_record_corruption(self):
        _, _, img, meta = one(8)
        bad = [tw(meta, 0, id="Bad"), tw(meta, 0, variant=256), tw(meta, 0, variant=None), tw(meta, 0, index=1), tw(meta, 0, index=None),
               tw(meta, 0, category="wall"), tw(meta, 0, footprint=[1]), tw(meta, 0, footprint=(1, 1)), tw(meta, 0, footprint=[0, 1]),
               tw(meta, 0, footprint=[9, 1]), tw(meta, 0, cells=[]), tw(meta, 0, cells=None), tw(meta, 0, cells=[5]), tw(meta, 0, y=8),
               tw(meta, 0, w=9), tw(meta, 0, h=7), tw(meta, 0, x=None), {**meta, "tiles": None}, {**meta, "tiles": [5]}, {**meta, "tiles": "x"},
               at(meta, 0, 2), at(meta, 0, -1)]  # fmt: skip
        self.rejects(lambda m: atlas.verify_atlas(img, m, PAL), [(m,) for m in bad])
        self.assertIsNone(atlas.verify_atlas(img, tw(meta, 0, name="renamed", junk=1), PAL))  # unchecked fields are ignored

    def test_spacing_gap_and_empty_cell_pixels_must_be_transparent(self):
        for sp in (0, 2):
            pl = atlas.pack([item("o", "object", 0)], 3)
            img = atlas.assemble(pl, {"o#0": ground(8, 3, PA)}, 8, 3, sp)
            meta = atlas.build_metadata(pl, 8, 3, sp, PAL)
            self.assertIsNone(atlas.verify_atlas(img, meta, PAL))
            self.rejects(atlas.verify_atlas, [(px(img, 0, 8 + sp - 1 if sp else 8, *PA[0], 255), meta, PAL)])  # first gap/empty column
            self.rejects(atlas.verify_atlas, [(px(img, 0, 8 + sp, *PA[0], 255), meta, PAL)])  # inside the empty neighbour cell
            self.assertIsNone(atlas.verify_atlas(img.copy(), meta, PAL))

    def test_does_not_mutate_inputs(self):
        _, _, img, meta = one(8)
        snap_img, snap_meta, snap_pal = img.copy(), copy.deepcopy(meta), PAL.copy()
        atlas.verify_atlas(img, meta, PAL)
        self.assertTrue(np.array_equal(img, snap_img) and np.array_equal(PAL, snap_pal))
        self.assertEqual(meta, snap_meta)
        img.flags.writeable = False
        self.assertIsNone(atlas.verify_atlas(img, meta, PAL))

    def test_every_tile_size_round_trip_with_all_palette_colours(self):
        for n in SIZES:
            pl, imgs, img, meta = one(n)
            self.assertEqual((meta["tile_size"], meta["width"], meta["height"]), (n, 2 * n, n))
            self.assertIsNone(atlas.verify_atlas(img, json.loads(json.dumps(meta)), PAL))
            self.assertEqual(sorted(map(tuple, np.unique(img[img[..., 3] == 255][:, :3], axis=0))), sorted(set(map(tuple, imgs["g#0"][..., :3].reshape(-1, 3)))))

