"""Tests for ``nodes/tilemap_assemble.py`` (D2), part 4: the optional ``base_atlas`` / ``base_alpha`` inputs and
``job_json.base`` (tileset reuse: improve / extend / restyle lineage, kept pixels, repacking, base mismatches).
A base is produced by the node itself (run -> save -> ``tileset_io.load_tileset`` -> ``base_block``), exactly the
loop a user performs. Helpers and the module loader come from ``tests/test_node_assemble.py`` (imported by name).
Stdlib unittest, no network.

Residual gaps: palette-mismatch between ``base.palette`` and the atlas pixels and ``tileset_io`` parsing of saved
files are covered by ``tests/test_tileset_io.py``; here the base palette is always the one the node wrote.
"""

import copy
import json
import unittest

import numpy as np
import torch

from tests.test_node_assemble import (AsmCase, make_images, make_job_json, make_plan, run,  # noqa: F401
                                      setUpModule, tearDownModule)
from tests.test_tileset_io import png_of
from tilemap_core import tileset_io

PATH = "tilemap/ts"


def make_base(transitions=False):
    """Run a plain assemble and return ``(output, base_block, job_json_without_base)``."""
    data = make_job_json()
    out = run(data, transitions=transitions)
    rgba = (out[0][0].numpy() * 255 + 0.5).astype(np.uint8)
    ts = tileset_io.load_tileset(png_of(rgba), out[3])
    return out, tileset_io.base_block(ts, PATH), data


def with_base(data, block, ids=None, **over):
    """``data`` + ``block``; ``ids`` keeps only the jobs of those tile ids."""
    d = copy.deepcopy(data)
    d["base"] = copy.deepcopy(block)
    if ids is not None:
        d["jobs"] = [j for j in d["jobs"] if j["tile_id"] in ids]
    d.update(over)
    return d


def meta_of(out):
    return json.loads(out[3])


def sources(meta):
    return {t["key"]: t["source"] for t in meta["tiles"]}


class BaseCase(AsmCase):
    @classmethod
    def setUpClass(cls):
        cls.base_out, cls.block, cls.data = make_base()
        cls.base_meta = meta_of(cls.base_out)

    def go(self, d, n, start=40, **kw):
        return run(d, images=make_images(n, start=start), base_atlas=self.base_out[0], base_alpha=self.base_out[1],
                   transitions=False, **kw)


class BaseBlockTests(BaseCase):
    def test_block_matches_the_saved_atlas(self):
        self.assertEqual(self.block["path"], PATH)
        self.assertEqual(self.block["keys"], ["g0#0", "g1#0", "o0#0", "v0#0"])
        self.assertEqual(self.block["palette"], self.base_meta["palette"])
        self.assertEqual(self.block["snap"]["palette_mode"], "keep")


class ImproveTests(BaseCase):
    def setUp(self):
        self.out = self.go(with_base(self.data, self.block, {"g0"}), 1)
        self.meta = meta_of(self.out)

    def test_only_the_regenerated_tile_changes(self):
        self.assertEqual(sources(self.meta), {"g0#0": "regenerated", "g1#0": "kept", "o0#0": "kept", "v0#0": "kept"})
        old, new = self.base_out[0][0], self.out[0][0]
        self.assertTrue(torch.equal(new[:, 8:], old[:, 8:]))
        self.assertFalse(torch.equal(new[:, :8], old[:, :8]))

    def test_lineage_is_improve(self):
        self.assertEqual(self.meta["lineage"], {"parent": PATH, "mode": "improve", "generation": 1})

    def test_palette_is_the_base_palette(self):
        self.assertEqual(self.meta["palette"], self.base_meta["palette"])

    def test_provenance_of_kept_and_regenerated_tiles(self):
        by_id = {t["id"]: t for t in self.meta["tiles"]}
        self.assertEqual((by_id["g0"]["seed"], by_id["g0"]["job_index"]), (7, 0))
        self.assertEqual(by_id["g0"]["prompt"], self.base_meta["tiles"][0]["prompt"])
        self.assertEqual(by_id["g1"]["seed"], 8)
        self.assertEqual(by_id["o0"]["source"], "kept")

    def test_no_images_at_all_reproduces_the_base_exactly(self):
        out = run(with_base(self.data, self.block, set()), images=[], base_atlas=self.base_out[0],
                  base_alpha=self.base_out[1], transitions=False)
        self.assertTrue(torch.equal(out[0], self.base_out[0]))
        self.assertTrue(torch.equal(out[1], self.base_out[1]))
        self.assertEqual(meta_of(out)["lineage"]["mode"], "improve")

    def test_wang_tiles_of_an_untouched_set_are_kept_and_touched_ones_regenerated(self):
        base_out = run(transitions=True)
        rgba = (base_out[0][0].numpy() * 255 + 0.5).astype(np.uint8)
        block = tileset_io.base_block(tileset_io.load_tileset(png_of(rgba), base_out[3]), PATH)
        data = make_job_json()
        touched = run(with_base(data, block, {"g0"}), images=make_images(1), base_atlas=base_out[0],
                      base_alpha=base_out[1], transitions=True)
        got = sources(meta_of(touched))
        self.assertEqual({v for k, v in got.items() if k.startswith("g0_g1")}, {"regenerated"})
        self.assertEqual(got["o0#0"], "kept")
        untouched = run(with_base(data, block, {"o0"}), images=make_images(1), base_atlas=base_out[0],
                        base_alpha=base_out[1], transitions=True)
        got = sources(meta_of(untouched))
        self.assertEqual({v for k, v in got.items() if k.startswith("g0_g1")}, {"kept"})
        self.assertEqual(got["o0#0"], "regenerated")
        self.assertEqual(untouched[0].shape, base_out[0].shape)
        self.assertTrue(torch.equal(untouched[0][0][8:], base_out[0][0][8:]))  # everything below row 0 is untouched


    def test_base_palette_may_exceed_the_snap_palette_size(self):
        d = with_base(self.data, self.block, {"g0"})
        d["base"]["snap"]["palette_size"] = 4  # the kept base palette (8 colours) is still verified against 256
        meta = meta_of(self.go(d, 1))
        self.assertEqual(len(meta["palette"]), len(self.base_meta["palette"]))
        self.assertEqual(meta["snap"]["palette_size"], 4)


class ExtendTests(BaseCase):
    def setUp(self):
        plan = make_plan(3, 1, 1)
        self.data2 = make_job_json(plan)
        self.out = self.go(with_base(self.data2, self.block, {"g2"}), 1)
        self.meta = meta_of(self.out)

    def test_new_tile_is_appended_and_old_ones_are_kept(self):
        self.assertEqual(list(sources(self.meta).items()),
                         [("g0#0", "kept"), ("g1#0", "kept"), ("o0#0", "kept"), ("v0#0", "kept"), ("g2#0", "generated")])
        self.assertTrue(torch.equal(self.out[0][0][:, :32], self.base_out[0][0][:, :32]))

    def test_lineage_is_extend(self):
        self.assertEqual(self.meta["lineage"], {"parent": PATH, "mode": "extend", "generation": 1})

    def test_new_tile_gets_its_own_seed_and_job_index(self):
        t = self.meta["tiles"][-1]
        self.assertEqual((t["id"], t["seed"], t["job_index"], t["attempt"]), ("g2", 9, 2, 0))

    def test_several_new_tiles_are_appended_sorted_by_category_id_variant(self):
        plan = make_plan(4, 1, 1)
        tiles = plan["tiles"]
        a, b = [i for i, t in enumerate(tiles) if t["id"] in ("g2", "g3")]
        tiles[a], tiles[b] = tiles[b], tiles[a]  # plan order g3 before g2
        data = make_job_json(plan)
        self.assertEqual([j["tile_id"] for j in data["jobs"]][2:4], ["g3", "g2"])
        meta = meta_of(self.go(with_base(data, self.block, {"g2", "g3"}), 2))
        self.assertEqual([t["key"] for t in meta["tiles"]], ["g0#0", "g1#0", "o0#0", "v0#0", "g2#0", "g3#0"])

    def test_changing_columns_marks_the_set_repacked(self):
        out = run(with_base(self.data2, self.block, {"g2"}), images=make_images(1, start=40), base_atlas=self.base_out[0],
                  base_alpha=self.base_out[1], transitions=False, columns=3)
        meta = meta_of(out)
        self.assertIs(meta["lineage"]["repacked"], True)
        self.assertEqual(meta["columns"], 3)
        self.assertNotIn("repacked", self.meta["lineage"])  # same columns as the base: not repacked


class RestyleTests(BaseCase):
    def setUp(self):
        d = with_base(self.data, self.block, None, snap={"palette_mode": "refit"})
        self.out = self.go(d, 4, start=60)
        self.meta = meta_of(self.out)

    def test_lineage_is_restyle_with_palette_refit(self):
        self.assertEqual(self.meta["lineage"], {"parent": PATH, "mode": "restyle", "generation": 1, "palette_refit": True})

    def test_every_tile_is_regenerated_with_a_new_palette(self):
        self.assertEqual(set(sources(self.meta).values()), {"regenerated"})
        self.assertNotEqual(self.meta["palette"], self.base_meta["palette"])
        self.assertEqual(self.meta["snap"]["palette_mode"], "refit")

    def test_refit_needs_an_image_for_every_tile(self):
        d = with_base(self.data, self.block, {"g0", "g1"}, snap={"palette_mode": "refit"})
        self.rejects(d, images=make_images(2), base_atlas=self.base_out[0], base_alpha=self.base_out[1],
                     transitions=False, contains="palette_mode 'refit' needs an image for every tile of the plan")


class BaseErrorTests(BaseCase):
    def d(self, **over):
        return with_base(self.data, self.block, {"g0"}, **over)

    def bad(self, contains, atlas=None, alpha=None, data=None, **widgets):
        return self.rejects(self.d() if data is None else data, images=make_images(1),
                            base_atlas=self.base_out[0] if atlas is None else atlas,
                            base_alpha=self.base_out[1] if alpha is None else alpha,
                            transitions=False, contains=contains, **widgets)

    def test_alpha_shape_must_match_the_atlas(self):
        self.bad("base_alpha must have the same width and height as base_atlas", alpha=self.base_out[1][:, :-1])

    def test_width_must_fit_tile_size_and_spacing(self):
        self.bad("base_atlas width does not fit tile_size and spacing", spacing=1)
        self.bad("base_atlas width does not fit tile_size and spacing", atlas=self.base_out[0][:, :, :-3],
                 alpha=self.base_out[1][:, :, :-3])

    def test_pixels_that_are_not_the_declared_layout_are_rejected(self):
        shifted = torch.roll(self.base_out[0], 8, dims=2)
        alpha = torch.roll(self.base_out[1], 8, dims=2)
        self.bad("base_atlas does not match the job_json base keys (layout or stray pixels)", atlas=shifted, alpha=alpha)

    def test_stray_pixel_in_an_unused_cell_is_rejected(self):
        atlas = self.base_out[0].clone()
        alpha = self.base_out[1].clone()
        atlas[0, 0, 40:, :3] = 0.25  # row 0, cells 5-7 are not part of the 4 base keys
        alpha[0, 0, 40:] = 1.0
        self.bad("base_atlas does not match the job_json base keys (layout or stray pixels)", atlas=atlas, alpha=alpha)

    def test_transparent_ground_pixel_is_rejected(self):
        alpha = self.base_out[1].clone()
        alpha[0, 3, 12] = 0.0  # inside g1#0 (ground tiles must be opaque)
        self.bad("base_atlas does not match the job_json base keys (layout or stray pixels)", alpha=alpha)

    def test_base_keys_must_exist_in_the_plan(self):
        d = with_base(self.data, self.block, {"g0"})
        d["base"]["keys"] = self.block["keys"] + ["zz#0"]
        self.bad("job_json base keys are not in its plan: ['zz#0']", data=d)

    def test_atlas_inputs_come_with_a_base_or_not_at_all(self):
        msg = "connect base_atlas and base_alpha together, and only when job_json has a base"
        self.rejects(self.d(), images=make_images(1), base_atlas=self.base_out[0], transitions=False, contains=msg)
        self.rejects(self.d(), images=make_images(1), transitions=False, contains=msg)
        self.rejects(self.data, images=make_images(4), base_atlas=self.base_out[0], base_alpha=self.base_out[1],
                     transitions=False, contains=msg)

    def test_tiles_missing_from_both_base_keys_and_jobs_are_rejected(self):
        d = self.d()
        d["base"]["keys"] = ["g0#0", "g1#0"]  # o0/v0 are neither kept nor given a job
        self.bad("job_json jobs do not match its plan", data=d)


if __name__ == "__main__":
    unittest.main()
