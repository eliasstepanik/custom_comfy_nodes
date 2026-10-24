"""Tests for tilemap_core.snap: palette modes, fixed palette, base pass-through, Wang items, preview, validation."""

import copy
import os
import subprocess
import sys
import unittest

import numpy as np

from tilemap_core import snap, transitions as wang

N = 16
JOBS = [
    {"tile_id": "g0", "category": "ground", "variant": 0, "footprint": [1, 1]},
    {"tile_id": "g1", "category": "ground", "variant": 0, "footprint": [1, 1]},
    {"tile_id": "o0", "category": "object", "variant": 0, "footprint": [2, 1]},
    {"tile_id": "v0", "category": "overlay", "variant": 0, "footprint": [1, 1]},
]
KEYS = ["g0#0", "g1#0", "o0#0", "v0#0"]
PLAN = {"tiles": [{"id": j["tile_id"]} for j in JOBS], "transitions": [["g0", "g1"]]}
COLOURS = ((200, 190, 150), (120, 160, 60), (70, 90, 40), (210, 80, 90))
CREAM = (240, 230, 200)
OTHER_PAL = np.array([[0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 255, 0]], np.uint8)


def make_images():
    rng = np.random.default_rng(7)
    out = []
    for i, job in enumerate(JOBS):
        h = w = 96 + 16 * i
        img = np.clip(np.array(COLOURS[i], np.float32) + rng.normal(0, 25, (h, w, 3)), 0, 255)
        if job["category"] != "ground":  # blob on a flat cream background that the key removes
            yy, xx = np.mgrid[:h, :w]
            bg = np.full((h, w, 3), CREAM, np.float32)
            blob = (yy - h / 2) ** 2 + (xx - w / 2) ** 2 < (h / 3) ** 2
            bg[blob] = img[blob]
            img = bg
        out.append(img.astype(np.uint8))
    return out


IMAGES = make_images()
CELLS, PAL = snap.snap_tiles(IMAGES, JOBS, N)


def same(a, b):
    return a.keys() == b.keys() and all(
        a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and a[k].tobytes() == b[k].tobytes() for k in a
    )


def colours_used(cells):
    return {tuple(px[:3]) for c in cells.values() for px in c.reshape(-1, 4) if px[3]}


class SnapTilesTest(unittest.TestCase):
    def test_cells_follow_job_order_footprint_and_dtype(self):
        self.assertEqual(list(CELLS), KEYS)
        for job, key in zip(JOBS, KEYS):
            cw, ch = job["footprint"]
            self.assertEqual((CELLS[key].shape, CELLS[key].dtype), ((ch * N, cw * N, 4), np.uint8))
        self.assertEqual((PAL.dtype, PAL.shape[1]), (np.uint8, 3))

    def test_ground_is_opaque_and_object_is_keyed_to_binary_alpha(self):
        self.assertEqual(set(np.unique(CELLS["g0#0"][..., 3])), {255})
        self.assertEqual(set(np.unique(CELLS["o0#0"][..., 3])), {0, 255})

    def test_defaults_equal_explicit_and_run_is_deterministic(self):
        explicit, pal = snap.snap_tiles(IMAGES, JOBS, N, dict(snap.SNAP_DEFAULTS))
        self.assertTrue(same(CELLS, explicit))
        self.assertEqual(PAL.tobytes(), pal.tobytes())
        self.assertTrue(same(CELLS, snap.snap_tiles(IMAGES, JOBS, N)[0]))

    def test_keep_without_palette_fits_one_shared_palette_of_at_most_palette_size(self):
        self.assertLessEqual(len(PAL), snap.SNAP_DEFAULTS["palette_size"])
        self.assertTrue(colours_used(CELLS) <= {tuple(c) for c in PAL})
        cells, pal = snap.snap_tiles(IMAGES, JOBS, N, {"palette_size": 4})
        self.assertLessEqual(len(pal), 4)
        self.assertTrue(colours_used(cells) <= {tuple(c) for c in pal})

    def test_seam_blend_changes_ground_only(self):
        cells, _ = snap.snap_tiles(IMAGES, JOBS, N, {"seam_blend": 0.0}, palette=PAL)
        self.assertNotEqual(cells["g0#0"].tobytes(), CELLS["g0#0"].tobytes())
        self.assertEqual(cells["o0#0"].tobytes(), CELLS["o0#0"].tobytes())

    def test_key_tolerance_changes_object_alpha_only(self):
        cells, _ = snap.snap_tiles(IMAGES, JOBS, N, {"key_tolerance": 255}, palette=PAL)
        self.assertNotEqual(cells["o0#0"][..., 3].tobytes(), CELLS["o0#0"][..., 3].tobytes())
        self.assertEqual(cells["g0#0"].tobytes(), CELLS["g0#0"].tobytes())

    def test_fixed_palette_returned_as_given_and_used_as_is(self):
        cells, pal = snap.snap_tiles(IMAGES, JOBS, N, palette=OTHER_PAL)
        self.assertEqual(pal.tobytes(), OTHER_PAL.tobytes())
        used = colours_used(cells)
        self.assertTrue(used and used <= {tuple(c) for c in OTHER_PAL})
        same_cells, same_pal = snap.snap_tiles(IMAGES, JOBS, N, palette=PAL)
        self.assertTrue(same(same_cells, CELLS))
        self.assertEqual(same_pal.tobytes(), PAL.tobytes())

    def test_refit_mode_fits_new_palette_while_keep_with_palette_does_not(self):
        refit, refit_pal = snap.snap_tiles(IMAGES, JOBS, N, {"palette_mode": "refit"})
        self.assertTrue(same(refit, CELLS))
        self.assertEqual(refit_pal.tobytes(), PAL.tobytes())
        keep, keep_pal = snap.snap_tiles(IMAGES, JOBS, N, {"palette_mode": "keep"}, palette=OTHER_PAL)
        self.assertEqual(keep_pal.tobytes(), OTHER_PAL.tobytes())
        self.assertFalse(same(keep, refit))


class BaseCellsTest(unittest.TestCase):
    def setUp(self):
        self.base = {k: CELLS[k].copy() for k in ("g0#0", "o0#0")}
        self.base["o0#0"][0, 0] = (1, 2, 3, 255)  # not a palette colour: must survive untouched
        self.garbage = [np.random.default_rng(i).integers(0, 255, im.shape, dtype=np.uint8) for i, im in enumerate(IMAGES)]

    def test_base_cells_pass_through_byte_for_byte_as_copies(self):
        cells, pal = snap.snap_tiles(self.garbage, JOBS, N, palette=PAL, base_cells=self.base)
        for key, want in self.base.items():
            self.assertEqual((cells[key].dtype, cells[key].shape), (want.dtype, want.shape))
            self.assertEqual(cells[key].tobytes(), want.tobytes())
            self.assertFalse(np.shares_memory(cells[key], want))
        self.assertEqual((pal.tobytes(), list(cells)), (PAL.tobytes(), KEYS))

    def test_non_base_cells_match_run_without_base(self):
        cells, _ = snap.snap_tiles(self.garbage, JOBS, N, palette=PAL, base_cells=self.base)
        want, _ = snap.snap_tiles(self.garbage, JOBS, N, palette=PAL)
        for key in ("g1#0", "v0#0"):
            self.assertEqual(cells[key].tobytes(), want[key].tobytes())
        self.assertNotEqual(want["o0#0"].tobytes(), self.base["o0#0"].tobytes())  # re-quantising would change it

    def test_base_images_are_ignored_even_when_none(self):
        images = [None if k in self.base else im for k, im in zip(KEYS, self.garbage)]
        cells, _ = snap.snap_tiles(images, JOBS, N, palette=PAL, base_cells=self.base)
        self.assertTrue(same(cells, snap.snap_tiles(self.garbage, JOBS, N, palette=PAL, base_cells=self.base)[0]))

    def test_all_base_is_pure_pass_through(self):
        cells, pal = snap.snap_tiles([None] * len(JOBS), JOBS, N, palette=PAL, base_cells=dict(CELLS))
        self.assertTrue(same(cells, CELLS))
        self.assertEqual(pal.tobytes(), PAL.tobytes())


class WangTest(unittest.TestCase):
    SEED = 1234

    def test_sixteen_items_with_keys_ids_and_corners(self):
        items, images = snap.wang_items(PLAN, CELLS, self.SEED, N)
        self.assertEqual([i["key"] for i in items], [f"g0_g1#{m}" for m in range(16)])
        self.assertEqual(list(images), [i["key"] for i in items])
        for mask, item in enumerate(items):
            self.assertEqual((item["id"], item["variant"], item["category"]), ("g0_g1", mask, "ground"))
            self.assertEqual((item["cells_w"], item["cells_h"], item["name"]), (1, 1, f"g0 to g1 {mask}"))
            self.assertEqual(item["wang"], {"a": "g0", "b": "g1", "corners": [(mask >> b) & 1 for b in range(4)]})
            self.assertEqual(images[item["key"]].shape, (N, N, 4))
        self.assertEqual(images["g0_g1#0"].tobytes(), CELLS["g0#0"].tobytes())
        self.assertEqual(images["g0_g1#15"].tobytes(), CELLS["g1#0"].tobytes())

    def test_seed_changes_mixed_tiles_and_is_deterministic(self):
        _, a = snap.wang_items(PLAN, CELLS, self.SEED, N)
        self.assertFalse(same(a, snap.wang_items(PLAN, CELLS, self.SEED + 1, N)[1]))
        self.assertTrue(same(a, snap.wang_items(PLAN, CELLS, self.SEED, N)[1]))

    def test_pair_k_is_seeded_with_seed_plus_k_mod_2_64(self):
        plan = {"tiles": PLAN["tiles"], "transitions": [["g0", "g1"], ["g1", "g0"]]}
        for seed in (self.SEED, 2**64 - 1):
            _, images = snap.wang_items(plan, CELLS, seed, N)
            tiles = wang.wang_tiles(CELLS["g1#0"], CELLS["g0#0"], N, (seed + 1) % 2**64)
            for mask, (_, rgba) in enumerate(tiles):
                self.assertEqual(images[f"g1_g0#{mask}"].tobytes(), rgba.tobytes())

    def test_wang_id_returns_pair_id_and_registers_it(self):
        taken = set()
        self.assertEqual(snap.wang_id("a", "b", taken), "a_b")
        self.assertEqual(taken, {"a_b"})
        with self.assertRaisesRegex(ValueError, "collides"):
            snap.wang_id("a", "b", taken)

    def test_rejections(self):
        clash = {"tiles": PLAN["tiles"] + [{"id": "g0_g1"}], "transitions": PLAN["transitions"]}
        twice = {"tiles": PLAN["tiles"], "transitions": [["g0", "g1"], ["g0", "g1"]]}
        cases = [("collides", clash, CELLS, 1), ("collides", twice, CELLS, 1),
                 ("no cell image", PLAN, {}, 1), ("no cell image", PLAN, {"g0#0": CELLS["g0#0"]}, 1)]
        cases += [("seed", PLAN, CELLS, s) for s in (-1, 2**64, True, 1.0)]
        cases += [("plan must be", p, CELLS, 1) for p in ({}, {"tiles": [], "transitions": "x"}, [])]
        for frag, plan, cells, seed in cases:
            with self.subTest(frag=frag, seed=seed), self.assertRaisesRegex(ValueError, frag):
                snap.wang_items(plan, cells, seed, N)


class PreviewCellsTest(unittest.TestCase):
    def test_row_major_rgb_cells_in_placement_order(self):
        sheet = np.zeros((8, 8, 4), np.uint8)  # 2x2 footprint of 4 px cells, cell (r, c) holds value 1 + 10*(2r+c)
        for r in range(2):
            for c in range(2):
                sheet[r * 4 : r * 4 + 4, c * 4 : c * 4 + 4] = 1 + 10 * (2 * r + c)
        placements = [{"key": "b#0", "cells_w": 1, "cells_h": 1}, {"key": "a#0", "cells_w": 2, "cells_h": 2}]
        out = snap.preview_cells(placements, {"a#0": sheet, "b#0": np.full((4, 4, 4), 100, np.uint8)}, 4)
        self.assertEqual([int(c[0, 0, 0]) for c in out], [100, 1, 11, 21, 31])
        self.assertTrue(all(c.shape == (4, 4, 3) and len(np.unique(c)) == 1 for c in out))

    def test_empty_placements_and_missing_key(self):
        self.assertEqual(snap.preview_cells([], {}, N), [])
        with self.assertRaisesRegex(ValueError, "no cell image"):
            snap.preview_cells([{"key": "zz#0", "cells_w": 1, "cells_h": 1}], CELLS, N)


class CheckSnapParamsTest(unittest.TestCase):
    def test_none_and_empty_give_the_documented_defaults_as_a_copy(self):
        want = {"palette_size": 24, "seam_blend": 0.25, "key_tolerance": 40, "transitions": True,
                "palette_mode": "keep", "extra_colours": 0}
        for given in (None, {}):
            got = snap.check_snap_params(given)
            self.assertEqual(got, want)
            got["palette_size"] = 1
            self.assertEqual(snap.SNAP_DEFAULTS, want)

    def test_partial_mapping_merges_over_defaults(self):
        got = snap.check_snap_params({"palette_size": 8, "palette_mode": "refit"})
        self.assertEqual(got, {**snap.SNAP_DEFAULTS, "palette_size": 8, "palette_mode": "refit"})

    def test_boundary_values_are_accepted_unchanged(self):
        for key, vals in (("palette_size", (1, 256)), ("seam_blend", (0, 0.0, 0.5)), ("key_tolerance", (0, 255)),
                          ("extra_colours", (0, 16)), ("transitions", (True, False))):
            for v in vals:
                with self.subTest(key=key, value=v):
                    self.assertEqual(snap.check_snap_params({key: v})[key], v)

    def test_every_out_of_range_or_wrong_type_value_raises_no_clamp(self):
        bad = {
            "palette_size": (0, 257, -5, 8.0, True, "8", None),
            "seam_blend": (-0.01, 0.51, 5, True, float("nan"), float("inf"), "0.2", None),
            "key_tolerance": (-1, 256, True, 10.0, "40", None),
            "extra_colours": (-1, 17, False, 1.0, None),
            "transitions": (1, 0, "yes", None),
            "palette_mode": ("x", "KEEP", "", None, 1),
        }
        for key, values in bad.items():
            for v in values:
                with self.subTest(key=key, value=v), self.assertRaisesRegex(ValueError, key):
                    snap.check_snap_params({key: v})

    def test_unknown_keys_and_non_mapping_raise(self):
        with self.assertRaisesRegex(ValueError, "unknown keys"):
            snap.check_snap_params({"bogus": 1, "palette_size": 8})
        for given in ([1], "keep", 3):
            with self.subTest(given=given), self.assertRaisesRegex(ValueError, "mapping"):
                snap.check_snap_params(given)


class SnapTilesValidationTest(unittest.TestCase):
    def rejects(self, frag, *args, **kw):
        with self.assertRaisesRegex(ValueError, frag):
            snap.snap_tiles(*args, **kw)

    def test_rejections(self):
        im, jb = IMAGES, JOBS
        self.rejects("images for", im[:-1], jb, N)
        self.rejects("at least one", [], [], N)
        self.rejects("duplicate", im[:2], [jb[0], jb[0]], N)
        self.rejects("job 0", im[:1], [{"tile_id": "x"}], N)
        self.rejects("job 0", im[:1], [{**jb[0], "category": 3}], N)
        self.rejects("n must be", im, jb, 17)
        self.rejects("seam_blend", im, jb, N, {"seam_blend": 0.6})
        self.rejects("uint8", [x.astype(np.float32) for x in im], jb, N)
        self.rejects("image 0", [None] * len(jb), jb, N)
        self.rejects("image 1", [im[0], im[1][..., :2]] + im[2:], jb, N)
        self.rejects("too large", [np.zeros((2049, 2049, 3), np.uint8)] + im[1:], jb, N)

    def test_palette_rejections(self):
        for pal in (PAL.astype(np.int32), PAL[:, :2], PAL[:0], np.zeros((257, 3), np.uint8), PAL.tolist()):
            with self.subTest(shape=getattr(pal, "shape", None)):
                self.rejects("palette", IMAGES, JOBS, N, palette=pal)
        self.rejects("refit", IMAGES, JOBS, N, {"palette_mode": "refit"}, palette=PAL)

    def test_base_cells_rejections(self):
        g0 = CELLS["g0#0"]
        self.rejects("fixed palette", IMAGES, JOBS, N, base_cells={"g0#0": g0})
        self.rejects("not in jobs", IMAGES, JOBS, N, palette=PAL, base_cells={"zz#0": g0})
        for wrong in ({"o0#0": g0}, {"g0#0": g0[..., :3]}, {"g0#0": g0.astype(np.int16)}, {"g0#0": g0.tolist()}):
            self.rejects("shape", IMAGES, JOBS, N, palette=PAL, base_cells=wrong)

    def test_inputs_are_not_modified(self):
        jobs, images = copy.deepcopy(JOBS), [x.copy() for x in IMAGES]
        snap.snap_tiles(images, jobs, N, palette=PAL)
        self.assertTrue(jobs == JOBS and all(a.tobytes() == b.tobytes() for a, b in zip(images, IMAGES)))


class PurityTest(unittest.TestCase):
    def test_importing_snap_pulls_in_no_torch_or_comfy(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        code = (
            "import sys; sys.path.insert(0, sys.argv[1]); import tilemap_core.snap\n"
            "bad = [k for k in sys.modules if k.split('.')[0] in ('torch', 'comfy', 'nodes', 'folder_paths', 'server')]\n"
            "print(bad); sys.exit(1 if bad else 0)\n"
        )
        res = subprocess.run([sys.executable, "-B", "-c", code, root], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)


if __name__ == "__main__":
    unittest.main()
