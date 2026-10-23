"""Tests for tilemap_core.transitions (Wang tiles) and tilemap_core.atlas (unittest, numpy only)."""

import copy
import json
import unicodedata
import unittest

import numpy as np

from tests import _support
from tilemap_core import atlas, pixelize, transitions

SIZES = pixelize.TILE_SIZES
PA = [(10, 20, 30), (200, 100, 50), (90, 180, 70)]  # terrain A colours
PB = [(250, 240, 10), (5, 5, 250), (160, 30, 200)]  # terrain B colours, disjoint from A
PBA = np.array(PB, dtype=np.uint8)
PAL = np.array(PA + PB, dtype=np.uint8)
FREQS = ((1, 2), (2, -1), (3, 1))  # documented integer-frequency cosines; amplitude 0.18, threshold 0.5
VIEW = ("top-down", "claude-sonnet-5", 7, "forest")


def ground(n, seed, pal=PA):
    return _support.opaque_rgba_tile(n, seed, pal)


def is_b(img):
    return (img[..., None, :3] == PBA).all(-1).any(-1)


def reference_is_b(verts, n, seed):
    """A/B decision of a whole map from ONE continuous field over the global pixel grid."""
    ph = np.random.default_rng(seed).uniform(0.0, 2.0 * np.pi, 3)
    y, x = np.mgrid[0 : (verts.shape[0] - 1) * n, 0 : (verts.shape[1] - 1) * n]
    gu, gv, i, j = (x + 0.5) / n, (y + 0.5) / n, y // n, x // n
    u, v = gu - j, gv - i
    f = (verts[i, j] * (1 - u) * (1 - v) + verts[i, j + 1] * u * (1 - v)
         + verts[i + 1, j] * (1 - u) * v + verts[i + 1, j + 1] * u * v)  # fmt: skip
    noise = sum(np.cos(2.0 * np.pi * (fx * gu + fy * gv) + p) for (fx, fy), p in zip(FREQS, ph)) / 3
    return f + 0.18 * noise >= 0.5


class Base(unittest.TestCase):
    def rejects(self, fn, cases):
        for args in cases:
            with self.subTest(args=repr(args)[:60]), self.assertRaises(ValueError):
                fn(*args)


class WangTests(Base):
    def test_masks_pure_ends_palette_exact(self):
        for n in SIZES:
            a, b = ground(n, 1), ground(n, 2, PB)
            tiles = transitions.wang_tiles(a, b, n, 9)
            self.assertEqual(len(tiles), 16)
            for m, (c, img) in enumerate(tiles):
                self.assertEqual(c, (m & 1, m >> 1 & 1, m >> 2 & 1, m >> 3 & 1))  # (ne, se, sw, nw)
                self.assertEqual((img.shape, img.dtype), ((n, n, 4), np.uint8))
                self.assertTrue((img[..., 3] == 255).all())
                self.assertTrue(((img == a).all(-1) | (img == b).all(-1)).all(), (n, m))
            self.assertTrue(np.array_equal(tiles[0][1], a) and np.array_equal(tiles[15][1], b), n)

    def test_b_fraction_grows_with_b_corners(self):
        for n in SIZES:
            frac = {}
            for c, img in transitions.wang_tiles(ground(n, 1), ground(n, 2, PB), n, 3):
                frac.setdefault(sum(c), []).append(is_b(img).mean())
            mean = {k: float(np.mean(v)) for k, v in frac.items()}
            self.assertEqual((mean[0], mean[4]), (0.0, 1.0))
            self.assertTrue(0.0 < mean[1] < mean[2] < mean[3] < 1.0, (n, mean))

    def test_assembled_map_follows_one_continuous_field(self):
        verts = np.random.default_rng(7).integers(0, 2, size=(5, 5))

        def corners(i, j):  # (ne, se, sw, nw) of cell (i, j)
            return tuple(int(v) for v in (verts[i, j + 1], verts[i + 1, j + 1], verts[i + 1, j], verts[i, j]))

        for n in SIZES:
            for seed in (3, 2**64 - 1):
                tiles = dict(transitions.wang_tiles(ground(n, 1), ground(n, 2, PB), n, seed))
                rows = [np.concatenate([tiles[corners(i, j)] for j in range(4)], axis=1) for i in range(4)]
                got = is_b(np.concatenate(rows, axis=0))
                self.assertTrue(0.0 < got.mean() < 1.0)
                self.assertTrue(np.array_equal(got, reference_is_b(verts, n, seed)), (n, seed))

    def test_determinism_seed_aliasing_no_mutation(self):
        a, b = ground(16, 1), ground(16, 2, PB)
        a0, b0, (t1, t2, t3) = a.copy(), b.copy(), [transitions.wang_tiles(a, b, 16, s) for s in (4, 4, 5)]
        self.assertTrue(all(np.array_equal(x[1], y[1]) for x, y in zip(t1, t2)))
        self.assertTrue(any(not np.array_equal(x[1], y[1]) for x, y in zip(t1, t3)))
        self.assertTrue(np.array_equal(a, a0) and np.array_equal(b, b0))
        for _, img in t1:
            self.assertFalse(np.shares_memory(img, a) or np.shares_memory(img, b))
        self.assertEqual(len(transitions.wang_tiles(a, b, np.int64(16), np.uint64(7))), 16)

    def test_rejects_bad_arguments(self):
        a, b, wt = ground(16, 1), ground(16, 2, PB), transitions.wang_tiles
        translucent = px(a, 3, 3, *a[3, 3, :3], 254)
        self.rejects(wt, [(a, b, n, 0) for n in (16.0, True, np.float64(16), 17, 8)])
        self.rejects(wt, [(a, b, 16, s) for s in (-1, 2**64, 1.0, True)])
        self.rejects(wt, [(a.astype(np.int16), b, 16, 0), (a, b[:, :, :3], 16, 0), (a, b[:8], 16, 0),
                          (translucent, b, 16, 0), (a, translucent, 16, 0), (a.tolist(), b, 16, 0)])  # fmt: skip


def obj(n, cw, ch, seed):
    img = ground(max(cw, ch) * n, seed)[: ch * n, : cw * n].copy()
    img[: n // 2, : n // 2], img[-1, -1] = 0, 0
    return img


def item(i, cat, v, w=1, h=1, **kw):
    d = {"key": f"{i}#{v}", "id": i, "name": i.title(), "category": cat, "variant": v, "cells_w": w, "cells_h": h}
    return {**d, **kw}


def fixture(n, spacing, columns=7):
    """Ground x3, 2x2 tree, 2x3 house, overlay + 16 Wang tiles g0 -> g1; returns packed atlas data."""
    g0, g1 = ground(n, 1), ground(n, 4, PB)
    specs = [("g0", "ground", 0, 1, 1), ("g0", "ground", 1, 1, 1), ("tree", "object", 0, 2, 2),
             ("house", "object", 0, 2, 3), ("bush", "overlay", 0, 1, 1), ("g1", "ground", 0, 1, 1)]  # fmt: skip
    items = [item(*sp) for sp in specs]
    imgs = {it["key"]: obj(n, it["cells_w"], it["cells_h"], i) for i, it in enumerate(items)}
    imgs.update({"g0#0": g0, "g0#1": ground(n, 2), "g1#0": g1})
    for c, img in transitions.wang_tiles(g0, g1, n, 11):
        m = c[0] | c[1] << 1 | c[2] << 2 | c[3] << 3
        items.append(item("g0_g1", "ground", m, wang={"a": "g0", "b": "g1", "corners": list(c)}))
        imgs[f"g0_g1#{m}"] = img
    pl = atlas.pack(items, columns)
    meta = atlas.build_metadata(pl, n, columns, spacing, PAL, *VIEW)
    return items, pl, imgs, atlas.assemble(pl, imgs, n, columns, spacing), meta


def px(r, y, x, *v):
    r = r.copy()
    r[y, x] = v
    return r


def tw(d, i, **kw):
    d = copy.deepcopy(d)
    return d["tiles"][i].update(kw) or d


def at(d, i, col, row=0):  # move the 1x1 tile i to (col, row)
    st = d["tile_size"] + d["spacing"]
    return tw(d, i, x=col * st, y=row * st, cells=[{"col": col, "row": row, "x": col * st, "y": row * st}])


class PackTests(Base):
    def test_first_fit_row_major_with_backfill(self):
        ks = [("A", 2, 2), ("B", 1, 1), ("C", 1, 1), ("D", 1, 1), ("E", 3, 1), ("F", 1, 1)]
        items = [{"key": k, "cells_w": w, "cells_h": h, "id": k.lower()} for k, w, h in ks]
        before = copy.deepcopy(items)
        out = atlas.pack(items, 3)
        self.assertEqual(items, before)
        self.assertEqual([(p["key"], p["col"], p["row"], p["id"]) for p in out],
                         [("A", 0, 0, "a"), ("B", 2, 0, "b"), ("C", 2, 1, "c"), ("D", 0, 2, "d"), ("E", 0, 3, "e"),
                          ("F", 1, 2, "f")])  # fmt: skip
        self.assertEqual(atlas.pack(items, 3), out)  # deterministic
        w = {"corners": [0, 0, 0, 1]}  # descriptors are deep-copied
        atlas.pack([{"key": "k", "cells_w": 1, "cells_h": 1, "wang": w}], 2)[0]["wang"]["corners"][0] = 1
        self.assertEqual(w["corners"], [0, 0, 0, 1])

    def test_wide_tile_and_limits(self):
        with self.assertRaisesRegex(ValueError, "huge.*at least 4"):
            atlas.pack([{"key": "huge", "cells_w": 4, "cells_h": 1}], 3)
        one = [{"key": "k", "cells_w": 1, "cells_h": 1}]
        self.assertEqual(atlas.pack(one, np.int64(64))[0]["col"], 0)
        many = lambda n: [{"key": f"k{i}", "cells_w": 1, "cells_h": 1} for i in range(n)]  # noqa: E731
        self.rejects(atlas.pack, [(one, c) for c in (0, 65, 3.0, True, "3", np.float64(3))])
        bad = [([], 3), (many(513), 64), (one + one, 3)]
        bad += [([{**one[0], **kw}], 64) for kw in ({"cells_w": 9}, {"cells_h": 0}, {"key": ""}, {"cells_w": 1.0})]
        self.rejects(atlas.pack, bad)
        self.assertEqual(len(atlas.pack(many(512), 64)), 512)


class AssembleTests(Base):
    def test_size_gaps_and_exact_copies(self):
        for n in SIZES:
            for sp in (0, 2):
                pl, imgs, img, meta = fixture(n, sp)[1:]
                rows, st, snap = meta["rows"], n + sp, copy.deepcopy((pl, imgs))
                self.assertTrue(np.array_equal(img, atlas.assemble(pl, imgs, n, 7, sp)))  # deterministic
                self.assertTrue(pl == snap[0] and all(np.array_equal(imgs[k], snap[1][k]) for k in imgs))
                self.assertFalse(any(np.shares_memory(img, v) for v in imgs.values()))
                self.assertEqual(img.shape, (rows * n + (rows - 1) * sp, 7 * n + 6 * sp, 4))
                self.assertEqual((meta["height"], meta["width"]), img.shape[:2])
                own = np.zeros(img.shape[:2], bool)
                for p in pl:
                    for b in range(p["cells_h"]):
                        for a in range(p["cells_w"]):
                            y, x = (p["row"] + b) * st, (p["col"] + a) * st
                            own[y : y + n, x : x + n] = True
                            want = imgs[p["key"]][b * n : (b + 1) * n, a * n : (a + 1) * n]
                            self.assertTrue(np.array_equal(img[y : y + n, x : x + n], want))
                self.assertTrue((~own).any() and not img[~own].any())  # empty cells/gaps exactly (0,0,0,0)

    def test_rejects_bad_arguments(self):
        pl, imgs = fixture(8, 0)[1:3]
        g0 = imgs["g0#0"]
        pics = [{k: v for k, v in imgs.items() if k != "g1#0"}, {**imgs, "g0#0": g0[:4]},
                {**imgs, "g0#0": g0.astype(np.int16)}, {**imgs, "zzz": g0}, {}]  # fmt: skip
        self.rejects(atlas.assemble, [(pl, p, 8, 7, 0) for p in pics])
        geo = [(9, 7, 0), (8.0, 7, 0), (True, 7, 0), (8, 7.0, 0), (8, 7, 17), (8, 7, -1), (8, 7, 0.0),
               (8, 7, np.float64(0)), (8, 65, 0), (8, 6, 0)]  # fmt: skip
        self.rejects(atlas.assemble, [(pl, imgs, *g) for g in geo])
        self.assertEqual(atlas.assemble(pl, imgs, np.int64(8), np.int64(7), np.int64(16)).shape[1], 7 * 8 + 6 * 16)
        two = {k: imgs[k] for k in ("g0#0", "g0#1")}
        self.rejects(atlas.assemble, [([pl[0], {**pl[1], "col": 0, "row": 0}], two, 8, 7, 0)])  # overlap
        far = lambda row: [{"key": "k", "cells_w": 1, "cells_h": 1, "col": 0, "row": row}]  # noqa: E731
        with self.assertRaisesRegex(ValueError, "too large"):  # 4096 x 16448 px > 2**26
            atlas.assemble(far(256), {}, 64, 64, 0)
        self.assertEqual(atlas.assemble(far(255), {"k": np.zeros((64, 64, 4), np.uint8)}, 64, 64, 0).size, 2**28)


class MetadataTests(Base):
    def test_records_json_deep_copy_and_wrap(self):
        for n in SIZES:
            for sp in (0, 2):
                items, pl, _, img, meta = fixture(n, sp)
                atlas.verify_atlas(img, meta, PAL, 6)
                self.assertEqual((len(meta["tiles"]), meta["format"]), (len(items), "ccn-tilemap/1"))
                self.assertEqual(json.loads(json.dumps(meta)), meta)
                tree, wang = meta["tiles"][2], meta["tiles"][-1]
                self.assertEqual((tree["footprint"], len(tree["cells"]), tree["x"]), ([2, 2], 4, 2 * (n + sp)))
                self.assertEqual((tree["w"], tree["h"]), (n * (2 if sp == 0 else 1),) * 2)
                self.assertEqual([t["tileable"] for t in meta["tiles"][:6]], [True, True, False, False, False, True])
                self.assertEqual((wang["id"], wang["variant"], wang["tileable"]), ("g0_g1", 15, False))
                self.assertEqual(wang["wang"], {"a": "g0", "b": "g1", "corners": [1] * 4})
                for t in meta["tiles"]:
                    if t["wang"]:
                        self.assertEqual(t["variant"], sum(c << i for i, c in enumerate(t["wang"]["corners"])))
                    if t["category"] == "ground":  # invariant 6.4: opaque, 2x2 tiling periodic with period n
                        cell = img[t["y"] : t["y"] + n, t["x"] : t["x"] + n]
                        tiled = np.tile(cell, (2, 2, 1))
                        self.assertTrue((cell[..., 3] == 255).all())
                        self.assertTrue(np.array_equal(np.roll(tiled, n, (0, 1)), tiled))
        snap, again = copy.deepcopy(meta), atlas.build_metadata(pl, n, 7, sp, PAL, *VIEW)
        again["tiles"][0]["cells"].clear()
        again["palette"].append("x")
        pl[-1]["wang"]["corners"][0] ^= 1
        self.assertEqual(atlas.build_metadata(pl, n, 7, sp, PAL, *VIEW)["tiles"][0], snap["tiles"][0])
        self.assertEqual(meta, snap)

    def test_hostile_text_is_cleaned(self):
        evil = "a\x00b\u202e\u2066c/../{x}|<y>[z]`\\" + "q" * 100000
        m = atlas.build_metadata(atlas.pack([item("g0", "ground", 0, name=evil)], 2), 8, 2, 0, PAL, evil, evil, 0, evil)
        for key, cap in (("view", 40), ("model", 100), ("theme", 80)):
            self.assertTrue(m[key].startswith("ab") and len(m[key]) == cap, key)
            banned = set("{}|<>[]`\\")
            self.assertFalse(any(c in banned or unicodedata.category(c)[0] == "C" for c in m[key]), key)
        junk = atlas.pack([item("g0", "ground", 0, name="{}|")], 2)  # cleans to empty: falls back to id
        fallback = atlas.build_metadata(junk, 8, 2, 0, PAL)["tiles"][0]["name"]
        self.assertEqual((len(m["tiles"][0]["name"]), fallback), (40, "g0"))

    def test_uniqueness_and_validation(self):
        a, b = item("g0", "ground", 0), item("g0", "ground", 0, key="g0_g1#0")  # Wang id colliding with a plan id
        with self.assertRaisesRegex(ValueError, "unique"):
            atlas.build_metadata(atlas.pack([a, b], 3), 8, 3, 0, PAL)
        self.assertEqual(len(atlas.build_metadata(atlas.pack([a, {**b, "variant": 1}], 3), 8, 3, 0, PAL)["tiles"]), 2)
        w, base = {"a": "g0", "b": "g1", "corners": [0, 0, 0, 1]}, atlas.pack([a], 2)
        tile = [("id", v) for v in ("Grass", "a b", "../x", "", "_x", "x" * 81, None, 5)] + [("category", "wall")]
        tile += [("variant", v) for v in (-1, 256, 1.0, True)]
        tile += [("wang", {**w, **kw}) for kw in ({"corners": [0, 1, 2, 3]}, {"corners": [0, 1, 0]}, {"a": "A b"},
                                                  {"corners": [True] * 4}, {"b": None})]  # fmt: skip
        tile += [("wang", {"a": "g0", "b": "g1"}), ("wang", "x")]  # missing corners, wrong type
        self.rejects(atlas.build_metadata, [([{**base[0], k: v}], 8, 2, 0, PAL) for k, v in tile])
        geo = [(17, 2, 0), (8.0, 2, 0), (True, 2, 0), (8, 0, 0), (8, 65, 0), (8, 2.0, 0), (8, 2, 17), (8, 2, -1)]
        self.rejects(atlas.build_metadata, [(base, *g, PAL) for g in geo])
        pals = [np.zeros((257, 3), np.uint8), PAL[:0], PAL.astype(np.int16), PAL[:, :2]]
        self.rejects(atlas.build_metadata, [(base, 8, 2, 0, p) for p in pals])
        self.rejects(atlas.build_metadata, [(base, 8, 2, 0, PAL, "", "", s) for s in (-1, 2**64, 1.0, True)])
        placed = [[{**base[0], "col": 2}], [base[0], {**base[0], "key": "o"}]]  # out of grid, overlap
        self.rejects(atlas.build_metadata, [(pl, 8, 2, 0, PAL) for pl in placed])
        i64, p256 = np.int64, np.zeros((256, 3), np.uint8)
        ok = atlas.build_metadata(base, i64(8), i64(64), i64(16), p256, seed=2**64 - 1)
        self.assertEqual((ok["seed"], ok["palette"][255]), (2**64 - 1, "#000000"))


class VerifyTests(Base):
    def test_accepts_valid_and_rejects_each_corruption(self):
        img, meta = fixture(8, 2)[3:]
        own = {(c["x"], c["y"]) for t in meta["tiles"] for c in t["cells"]}
        ex, ey = next((c * 10, r * 10) for r in range(meta["rows"]) for c in range(7) if (c * 10, r * 10) not in own)
        W, ok = img.shape[1], (10, 20, 30, 255)
        two = [{"col": c, "row": 0, "x": c * 10, "y": 0} for c in (0, 1)]  # tile 0 spills into tile 1's cell
        negatives = [
            (img[:-1], meta), (img[:, : W - 1], meta), (img[..., :3], meta), (img.astype(np.float32), meta),
            (px(img, 0, 20, 10, 20, 30, 128), meta), (px(img, 0, 0, 1, 2, 3, 255), meta),
            (px(img, 0, 0, 0, 0, 0, 0), meta), (px(img, 0, 8, *ok), meta), (px(img, ey, ex, *ok), meta),
            (img, {**meta, "tiles": meta["tiles"][:-1]}), (img, {**meta, "tiles": []}), (img, {**meta, "width": W + 1}),
            (img, {**meta, "rows": 3}), (img, {**meta, "spacing": 0}), (img, {**meta, "columns": 6}),
            (img, {**meta, "format": "x"}), (img, {**meta, "palette": meta["palette"][::-1]}),
            (img, tw(meta, 1, id="g0", variant=0)), (img, tw(meta, 1, index=9)), (img, tw(meta, 0, category="wall")),
            (img, tw(meta, 2, footprint=[1, 1])), (img, tw(meta, 0, footprint=[2, 1], cells=two)),
            (img, at(meta, 0, 7)), (img, tw(meta, 0, x=1)),
        ]  # fmt: skip
        self.assertIsNone(atlas.verify_atlas(img, json.loads(json.dumps(meta)), PAL.copy(), 256))
        self.rejects(lambda r, m: atlas.verify_atlas(r, m, PAL, 6), negatives)
        self.rejects(lambda p, s: atlas.verify_atlas(img, meta, p, s),
                     [(np.zeros((257, 3), np.uint8), 256), (PAL, 5), (PAL[::-1], 6)])  # fmt: skip
