"""Tests for tilemap_core.scene_render (chunk D4): corner majority rule, crc32 variant hash, wang bit order, exact size."""

import copy
import unittest
import zlib

import numpy as np

from tilemap_core import atlas, scene_layout, snap, transitions
from tilemap_core.scene_render import PIXELS_MAX, UPSCALE_MAX, label_grid, pick_ground, render_scene

N = 8
RED, RED2, BLUE, GREEN, YELLOW = (200, 10, 10), (210, 12, 12), (10, 20, 220), (10, 200, 30), (250, 250, 0)
GRASS_RED = {200, 210}  # red channel of grass#0 / grass#1


def solid(rgb, n=N):
    a = np.zeros((n, n, 4), np.uint8)
    a[..., :3], a[..., 3] = rgb, 255
    return a


def build(pairs):
    """Catalogue + cells: grass (2 variants), dirt, stone, tree (1x2), leaf overlay, wang set per pair."""
    tiles = [{"id": i, "category": c} for i, c in (("grass", "ground"), ("dirt", "ground"), ("stone", "ground"), ("tree", "object"), ("leaf", "overlay"))]
    cells = {"grass#0": solid(RED), "grass#1": solid(RED2), "dirt#0": solid(BLUE), "stone#0": solid(GREEN)}
    items = [{"key": k, "cells_w": 1, "cells_h": 1, "id": k[:-2], "category": "ground", "variant": int(k[-1])} for k in cells]
    wang_items, wang_cells = snap.wang_items({"tiles": tiles, "transitions": [list(p) for p in pairs]}, cells, 3, N)
    tree, leaf = np.zeros((2 * N, N, 4), np.uint8), np.zeros((N, N, 4), np.uint8)
    tree[:, 2:6], leaf[:4, :4] = (*YELLOW, 255), 255
    cells.update({"tree#0": tree, "leaf#0": leaf}, **wang_cells)
    items += [{"key": "tree#0", "cells_w": 1, "cells_h": 2, "id": "tree", "category": "object", "variant": 0},
              {"key": "leaf#0", "cells_w": 1, "cells_h": 1, "id": "leaf", "category": "overlay", "variant": 0}] + wang_items
    return atlas.build_metadata(atlas.pack(items, 8), N, 8, 0, np.zeros((1, 3), np.uint8)), cells


def scene(terrain, w=4, h=4, **kw):
    return {"format": scene_layout.FORMAT, "width": w, "height": h, "seed": kw.pop("seed", 7), "terrain": terrain, **kw}


def flat(gid="grass", w=4, h=4):
    return [[gid] * (w + 1) for _ in range(h + 1)]


def cell(img, col, row, u=1):
    s = N * u
    return img[row * s : (row + 1) * s, col * s : (col + 1) * s]


def vhash(seed, col, row):
    return zlib.crc32(f"{seed}:{col}:{row}".encode())


class Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta, cls.cells = build([("grass", "dirt")])


class PickGroundTests(Fixture):
    def test_none_when_any_corner_is_none(self):
        for i in range(4):
            self.assertIsNone(pick_ground([None if j == i else "grass" for j in range(4)], self.meta, 0, 0, 0))

    def test_uniform_variant_follows_crc32_of_seed_col_row(self):
        for seed, col, row in [(0, 0, 0), (7, 3, 5), (1, 19, 2), (123456, 2, 31), (2**40, 5, 5)]:
            self.assertEqual(pick_ground(("grass",) * 4, self.meta, col, row, seed), ("grass", vhash(seed, col, row) % 2, None))
        self.assertEqual(pick_ground(("dirt",) * 4, self.meta, 4, 4, 9), ("dirt", 0, None))  # single variant

    def test_variant_depends_on_position_and_seed_and_is_stable(self):
        g4 = ("grass",) * 4
        by_col, by_seed = ({pick_ground(g4, self.meta, c, 0, 1)[1] for c in range(20)}, {pick_ground(g4, self.meta, 0, 0, s)[1] for s in range(20)})
        self.assertEqual((by_col, by_seed), ({0, 1}, {0, 1}))
        self.assertEqual(pick_ground(g4, self.meta, 5, 5, 1), pick_ground(g4, self.meta, 5, 5, 1))

    def test_wang_mask_bit_order_ne_se_sw_nw_second_id_is_one(self):
        for mask in range(1, 15):  # 0 and 15 are uniform -> plain tile
            corners = tuple("dirt" if (mask >> bit) & 1 else "grass" for bit in range(4))  # ne, se, sw, nw
            self.assertEqual(pick_ground(corners, self.meta, 0, 0, 0), ("grass_dirt", mask, None), corners)

    def test_wang_set_in_either_orientation(self):
        corners = ("dirt", "dirt", "grass", "grass")  # ne, se = dirt
        self.assertEqual(pick_ground(corners, build([("grass", "dirt")])[0], 1, 0, 0), ("grass_dirt", 3, None))  # dirt is b
        self.assertEqual(pick_ground(corners, build([("dirt", "grass")])[0], 1, 0, 0), ("dirt_grass", 12, None))  # grass is b

    def test_three_ids_pick_majority_with_warning(self):
        gid, variant, warn = pick_ground(("grass", "dirt", "stone", "stone"), self.meta, 2, 3, 7)
        self.assertEqual((gid, variant), ("stone", 0))
        self.assertEqual(warn, "cell (2, 3): corners ['grass', 'dirt', 'stone', 'stone']: three or more ground ids; drawn as majority ground 'stone'")

    def test_pair_without_set_tie_picks_smallest_id_with_warning(self):
        gid, variant, warn = pick_ground(("stone", "dirt", "dirt", "stone"), self.meta, 1, 0, 7)
        self.assertEqual((gid, variant), ("dirt", 0))
        self.assertEqual(warn, "cell (1, 0): corners ['stone', 'dirt', 'dirt', 'stone']: no transition set for the pair; drawn as majority ground 'dirt'")

    def test_strict_majority_wins_and_uses_the_cell_hash(self):
        self.assertEqual(pick_ground(("stone", "stone", "stone", "dirt"), self.meta, 0, 0, 0)[0], "stone")
        self.assertEqual(pick_ground(("dirt", "stone", "dirt", "dirt"), self.meta, 0, 0, 0)[0], "dirt")
        gid, variant, _ = pick_ground(("grass", "grass", "dirt", "stone"), self.meta, 6, 2, 11)
        self.assertEqual((gid, variant), ("grass", vhash(11, 6, 2) % 2))

    def test_bad_arguments_raise_value_error(self):
        g4 = ("grass",) * 4
        for bad in (dict(corners=g4[:3]), dict(corners=("grass", 1, "grass", "grass")), dict(corners="grass"), dict(col=-1), dict(row=True), dict(seed=-1), dict(catalogue={"tiles": []}), dict(catalogue=None), dict(corners=("marble",) * 4)):
            args = dict(corners=g4, catalogue=self.meta, col=0, row=0, seed=0)
            args.update(bad)
            with self.assertRaises(ValueError, msg=str(bad)):
                pick_ground(**args)

    def test_wrong_wang_corner_bits_and_two_wang_sets_are_rejected(self):
        bad = copy.deepcopy(self.meta)
        next(t for t in bad["tiles"] if t["wang"] and t["variant"] == 1)["wang"]["corners"] = [0, 0, 0, 1]
        with self.assertRaises(ValueError):
            pick_ground(("grass",) * 4, bad, 0, 0, 0)
        bad = copy.deepcopy(self.meta)
        twin = copy.deepcopy(next(t for t in bad["tiles"] if t["wang"] and t["variant"] == 5))
        twin["id"] = "grass_dirt_second"
        bad["tiles"].append(twin)
        with self.assertRaises(ValueError) as cm:
            pick_ground(("grass",) * 4, bad, 0, 0, 0)
        self.assertIn("two wang sets", str(cm.exception))
        with self.assertRaises(ValueError):
            render_scene(scene(flat()), self.cells, bad, N)


class RenderSceneTests(Fixture):
    def test_wang_cells_match_transitions_wang_tiles_for_both_orientations(self):
        for a, b, base in (("grass", "dirt", RED), ("dirt", "grass", BLUE)):
            meta, cells = build([(a, b)])
            ref = transitions.wang_tiles(cells[f"{a}#0"], cells[f"{b}#0"], N, 3)
            for mask in range(1, 15):
                terr = flat(a)
                ne, se, sw, nw = ((mask >> bit) & 1 for bit in range(4))
                terr[0][2], terr[1][2], terr[1][1], terr[0][1] = (b if v else a for v in (ne, se, sw, nw))
                img, warns = render_scene(scene(terr), cells, meta, N)
                self.assertEqual(warns, [])
                self.assertTrue(np.array_equal(cell(img, 1, 0), ref[mask][1]), (a, b, mask))
            self.assertEqual(tuple(ref[0][1][0, 0, :3]), base)

    def test_uniform_corner_masks_0_and_15_render_plain_ground(self):
        meta, cells = build([("grass", "dirt")])
        terr = flat("grass")
        for r, c in ((0, 2), (0, 3), (1, 2), (1, 3)):
            terr[r][c] = "dirt"  # cell (2, 0) all dirt (mask 15)
        img, warns = render_scene(scene(terr), cells, meta, N)
        self.assertEqual(warns, [])
        self.assertTrue(np.array_equal(cell(img, 2, 0), cells["dirt#0"]))
        self.assertIn(int(cell(img, 0, 3)[0, 0, 0]), GRASS_RED)

    def test_uniform_cells_use_hashed_variant_deterministically(self):
        sc = scene(flat(w=16, h=16), 16, 16, seed=1)
        a, wa = render_scene(sc, self.cells, self.meta, N)
        self.assertEqual(wa, [])
        for row, col in np.ndindex(16, 16):
            self.assertTrue(np.array_equal(cell(a, col, row), self.cells[f"grass#{vhash(1, col, row) % 2}"]), (col, row))
        b, wb = render_scene(copy.deepcopy(sc), copy.deepcopy(self.cells), copy.deepcopy(self.meta), N)
        self.assertTrue(np.array_equal(a, b))
        self.assertFalse(np.array_equal(a, render_scene(dict(sc, seed=2), self.cells, self.meta, N)[0]))

    def test_majority_cells_draw_majority_ground_and_warn_in_row_major_order(self):
        terr = flat("stone")
        terr[1][2] = "dirt"  # the four cells around vertex (2, 1) have 3 stone corners + 1 dirt, no stone/dirt set
        img, warns = render_scene(scene(terr), self.cells, self.meta, N)
        self.assertEqual([w.split(":")[0] for w in warns], ["cell (1, 0)", "cell (2, 0)", "cell (1, 1)", "cell (2, 1)"])
        self.assertTrue(all("no transition set for the pair" in w and "'stone'" in w for w in warns))
        for col, row in ((1, 0), (2, 0), (1, 1), (2, 1)):
            self.assertTrue((cell(img, col, row) == (*GREEN, 255)).all())

    def test_three_id_cell_warns_and_draws_majority(self):
        terr = flat("stone")
        terr[0][3], terr[1][3], terr[1][2], terr[0][2] = "dirt", "grass", "stone", "stone"  # cell (2, 0): ne, se, sw, nw
        img, warns = render_scene(scene(terr), self.cells, self.meta, N)
        three = [w for w in warns if w.startswith("cell (2, 0)")]
        self.assertEqual(len(three), 1)
        self.assertIn("three or more ground ids", three[0])
        self.assertTrue((cell(img, 2, 0) == (*GREEN, 255)).all())

    def test_output_shape_dtype_and_nearest_upscale(self):
        base, _ = render_scene(scene(flat(w=5), 5, 4), self.cells, self.meta, N)
        self.assertEqual((base.shape, base.dtype), ((4 * N, 5 * N, 4), np.uint8))
        for u in (2, 3):
            out, _ = render_scene(scene(flat(w=5), 5, 4), self.cells, self.meta, N, u)
            self.assertEqual(out.shape, (4 * N * u, 5 * N * u, 4))
            self.assertTrue(np.array_equal(out, np.kron(base, np.ones((u, u, 1), np.uint8))))
        c16 = {k: np.kron(v, np.ones((2, 2, 1), np.uint8)) for k, v in self.cells.items()}
        self.assertEqual(render_scene(scene(flat(w=4, h=5), 4, 5), c16, self.meta, 16, 2)[0].shape, (5 * 32, 4 * 32, 4))

    def test_background_fills_cells_touching_a_none_corner(self):
        terr = flat()
        terr[0][1] = None  # touches cells (0,0) and (1,0)
        img, _ = render_scene(scene(terr, background="#0A0B0C"), self.cells, self.meta, N)
        for col in (0, 1):
            self.assertTrue((cell(img, col, 0) == (10, 11, 12, 255)).all())
        self.assertIn(int(cell(img, 2, 0)[0, 0, 0]), GRASS_RED)
        self.assertEqual(set(np.unique(img[..., 3])), {255})

    def test_objects_and_overlays_composite_over_ground_by_alpha(self):
        sc = scene(flat(), objects=[{"id": "tree", "variant": None, "col": 3, "row": 1}], overlays=[{"id": "leaf", "variant": 0, "col": 3, "row": 2}])
        img, _ = render_scene(sc, self.cells, self.meta, N)
        ground = render_scene(scene(flat()), self.cells, self.meta, N)[0]
        top, low = cell(img, 3, 1), cell(img, 3, 2)
        self.assertTrue((top[:, 2:6] == (*YELLOW, 255)).all())
        self.assertTrue(np.array_equal(top[:, :2], cell(ground, 3, 1)[:, :2]))  # alpha 0 leaves ground
        self.assertTrue(np.array_equal(top[:, 6:], cell(ground, 3, 1)[:, 6:]))
        self.assertTrue((low[:4, :4] == 255).all())  # overlay over the tree's lower cell
        self.assertTrue((low[4:, 2:6] == (*YELLOW, 255)).all() and (low[:4, 4:6] == (*YELLOW, 255)).all())
        self.assertTrue(np.array_equal(low[4:, :2], cell(ground, 3, 2)[4:, :2]))
        sc["objects"][0]["variant"] = 0
        self.assertTrue(np.array_equal(render_scene(sc, self.cells, self.meta, N)[0], img))

    def test_inputs_are_not_mutated(self):
        sc = scene(flat(), objects=[{"id": "tree", "variant": None, "col": 0, "row": 0}])
        snaps = (copy.deepcopy(sc), copy.deepcopy(self.meta), {k: v.copy() for k, v in self.cells.items()})
        render_scene(sc, self.cells, self.meta, N, 2)
        self.assertEqual((sc, self.meta), snaps[:2])
        self.assertTrue(all(np.array_equal(self.cells[k], snaps[2][k]) for k in self.cells))

    def test_size_and_argument_errors(self):
        sc = scene(flat())
        for bad in (dict(n=12), dict(n=True), dict(n=0), dict(upscale=0), dict(upscale=UPSCALE_MAX + 1), dict(upscale=True), dict(upscale=1.5), dict(meta=None), dict(meta={"tiles": []}), dict(cells=[])):
            args = dict(scene=sc, cells=self.cells, meta=self.meta, n=N, upscale=1)
            args.update(bad)
            with self.assertRaises(ValueError, msg=str(bad)):
                render_scene(**args)
        big = scene(flat(w=32, h=32), 32, 32)
        self.assertGreater(32 * 32 * (32 * 16) ** 2, PIXELS_MAX)
        with self.assertRaises(ValueError) as cm:
            render_scene(big, self.cells, self.meta, 32, UPSCALE_MAX)
        self.assertIn("too large", str(cm.exception))

    def test_invalid_scene_content_raises(self):
        sc, c = scene(flat()), self.cells
        half = solid(GREEN)
        half[0, 0, 3] = 128  # alpha must be only 0 or 255
        tree = lambda **o: dict(sc, objects=[dict({"id": "tree", "variant": None, "col": 0, "row": 0}, **o)])  # noqa: E731
        cases = [(dict(sc, format="x"), c), (scene(flat(), background="808080"), c), (scene(flat(), background="#80808z"), c),
                 (scene(flat("marble")), c), (scene(flat()[:4]), c), (scene(flat(w=5)), c),
                 (sc, {k: v for k, v in c.items() if k != "grass#0"}), (sc, dict(c, **{"grass#0": solid(RED, 16)})), (sc, dict(c, **{"grass#0": half})),
                 (tree(variant=1), dict(c, **{"tree#1": c["tree#0"]})), (tree(row=3), c), (tree(id="leaf"), c), (tree(id="ghost"), c),
                 (dict(sc, overlays=[{"id": "tree", "col": 0, "row": 0}]), c), (dict(sc, objects="tree"), c)]
        for i, (s, cells) in enumerate(cases):
            with self.assertRaises(ValueError, msg=f"case {i}"):
                render_scene(s, cells, self.meta, N)
        no_stone = dict(self.meta, tiles=[t for t in self.meta["tiles"] if t["id"] != "stone"])
        with self.assertRaises(ValueError):  # terrain id without a plain tile
            render_scene(scene(flat("stone")), c, no_stone, N)


class LabelGridTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(5)
        self.imgs = [rng.integers(0, 256, (24, 20, 3), dtype=np.uint8) for _ in range(7)]

    def test_size_dtype_background_and_cell_positions(self):
        for cols, pad, bg, rgb in ((3, 4, "black", 0), (2, 0, "white", 255), (4, 8, "gray", 128), (1, 1, "gray", 128)):
            g = label_grid(self.imgs, cols, pad, bg, False)
            rows = -(-7 // cols)
            self.assertEqual((g.shape, g.dtype), ((rows * 24 + (rows + 1) * pad, cols * 20 + (cols + 1) * pad, 3), np.uint8))
            mask = np.ones(g.shape[:2], bool)
            for i, im in enumerate(self.imgs):
                x, y = pad + (i % cols) * (20 + pad), pad + (i // cols) * (24 + pad)
                self.assertTrue(np.array_equal(g[y : y + 24, x : x + 20], im), (cols, pad, i))
                mask[y : y + 24, x : x + 20] = False
            self.assertTrue((g[mask] == rgb).all(), bg)  # padding and empty slots show the background

    def test_alpha_channel_is_dropped_and_batch_array_accepted(self):
        rgba = [np.dstack([im, np.full(im.shape[:2], 7, np.uint8)]) for im in self.imgs]
        ref = label_grid(self.imgs, 3, 4, "black", True)
        for variant in (rgba, np.stack(self.imgs)):
            self.assertTrue(np.array_equal(label_grid(variant, 3, 4, "black", True), ref))

    def test_show_index_draws_numbers_in_top_left_corner_only(self):
        plain, labelled = label_grid(self.imgs, 3, 4, "black", False), label_grid(self.imgs, 3, 4, "black", True)
        for i, im in enumerate(self.imgs):
            x, y = 4 + (i % 3) * 24, 4 + (i // 3) * 28
            sub, ref = labelled[y : y + 24, x : x + 20], plain[y : y + 24, x : x + 20]
            self.assertFalse(np.array_equal(sub, ref), i)
            self.assertTrue((sub[:2, :2] == 0).all())  # black box under the number
            self.assertTrue(np.array_equal(sub[16:], ref[16:]))  # nothing below the label
            self.assertTrue(np.array_equal(sub[:, 16:], ref[:, 16:]))
        two = label_grid([self.imgs[0]] * 2, 1, 0, "black", True)
        self.assertFalse(np.array_equal(two[:24], two[24:]))  # "1" vs "2"

    def test_boundary_arguments_are_accepted(self):
        self.assertEqual(label_grid(self.imgs[:1], 16, 0, "black", False).shape, (24, 16 * 20, 3))
        self.assertEqual(label_grid(self.imgs[:1], 1, 256, "white", False).shape, (24 + 512, 20 + 512, 3))
    def test_bad_arguments_raise_value_error(self):
        im = self.imgs
        for bad in (dict(columns=0), dict(columns=17), dict(columns=True), dict(padding=-1), dict(padding=257), dict(background="red"), dict(show_index=1), dict(images=[]),
                    dict(images=[im[0], im[1][:5]]), dict(images=[im[0].astype(np.float32)]), dict(images=[im[0][..., 0]]), dict(images=[im[0][..., :2]]), dict(images="x")):
            args = dict(images=im, columns=2, padding=1, background="black", show_index=True)
            args.update(bad)
            with self.assertRaises(ValueError, msg=str(bad)):
                label_grid(**args)


if __name__ == "__main__":
    unittest.main()
