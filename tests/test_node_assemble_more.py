"""Tests for ``nodes/tilemap_assemble.py`` (D2), part 3: happy path outputs, ``ccn-tilemap/2`` metadata, tile
provenance, geometry, snap precedence and determinism, all without a base atlas. Surface/widgets/JSON text are
``tests/test_node_assemble.py``; field and image validation are ``tests/test_node_assemble_fields.py``; the base
atlas (improve/extend/restyle) is ``tests/test_node_assemble_base.py``. Helpers and the module loader are
imported by name from part 1. Stdlib unittest, no network.

Residual gaps: pixel content of snapped/blended tiles is covered by ``tests/test_snap*.py``; here only structure
(shapes, counts, keys, sources, positions, palette bound) and byte-for-byte repeatability are asserted.
"""

import copy
import json
import unittest

import numpy as np
import torch

from tests import test_node_assemble as base
from tests.test_node_assemble import (AsmCase, make_images, make_job_json, make_plan, run,  # noqa: F401
                                      setUpModule, tearDownModule)
from tests.test_tileset_io import png_of
from tilemap_core import atlas, atlas_meta, tileset_io

TOP_KEYS = {"cell_px", "columns", "format", "height", "model", "palette", "plan", "rows", "seed", "snap", "spacing",
            "style_prefix", "theme", "tile_size", "tiles", "view", "width"}
PLAN_KEYS = ["g0#0", "g1#0", "o0#0", "v0#0"]


def rgba_of(atlas_t):
    return (atlas_t[0].numpy() * 255 + 0.5).astype(np.uint8)


def meta_of(out):
    return json.loads(out[3])


class HappyPathTests(AsmCase):
    @classmethod
    def setUpClass(cls):
        cls.out = run()
        cls.meta = meta_of(cls.out)

    def test_output_types_and_shapes(self):
        atlas_t, alpha, tiles, text = self.out
        self.assertEqual(tuple(atlas_t.shape), (1, 24, 64, 4))
        self.assertEqual(tuple(alpha.shape), (1, 24, 64))
        self.assertEqual(tuple(tiles.shape), (20, 8, 8, 3))  # 4 plan tiles + 16 Wang tiles for one transition
        self.assertIsInstance(text, str)
        for t in (atlas_t, alpha, tiles):
            self.assertEqual(t.dtype, torch.float32)
            self.assertTrue(0.0 <= float(t.min()) and float(t.max()) <= 1.0)

    def test_alpha_is_the_atlas_alpha_channel_with_one_meaning_opaque(self):
        atlas_t, alpha, _, _ = self.out
        self.assertTrue(torch.equal(alpha, atlas_t[..., 3]))
        self.assertEqual(set(np.unique(alpha.numpy()).tolist()) - {0.0, 1.0}, set())
        self.assertEqual(float(alpha[0, 0:8, 0:8].min()), 1.0)  # ground tile g0 is fully opaque

    def test_atlas_pixels_are_exactly_8bit(self):
        a = self.out[0].numpy()
        self.assertTrue(np.array_equal(a, (np.rint(a * 255) / 255).astype(np.float32)))

    def test_metadata_schema(self):
        meta = self.meta
        self.assertEqual(set(meta), TOP_KEYS)
        self.assertEqual(meta["format"], "ccn-tilemap/2")
        self.assertEqual(meta["format"], atlas_meta.FORMAT)
        self.assertEqual((meta["width"], meta["height"]), (64, 24))
        self.assertEqual((meta["columns"], meta["rows"], meta["spacing"], meta["tile_size"]), (8, 3, 0, 8))
        self.assertEqual((meta["view"], meta["model"], meta["seed"], meta["theme"]), ("top-down", "m", 7, "test forest"))

    def test_metadata_text_is_compact_sorted_json(self):
        text = self.out[3]
        self.assertEqual(text, json.dumps(json.loads(text), sort_keys=True, separators=(",", ":")))

    def test_extra_keys_round_trip_the_input(self):
        data = make_job_json()
        self.assertEqual(self.meta["style_prefix"], "sty")
        self.assertEqual(self.meta["cell_px"], 256)
        self.assertEqual(self.meta["plan"], data["plan"])
        self.assertNotIn("review", self.meta)
        self.assertNotIn("lineage", self.meta)

    def test_snap_records_the_full_resolved_parameters(self):
        self.assertEqual(self.meta["snap"], {"palette_size": 8, "seam_blend": 0.25, "key_tolerance": 40,
                                             "transitions": True, "palette_mode": "keep", "extra_colours": 0})

    def test_palette_is_hex_and_within_the_requested_size(self):
        pal = self.meta["palette"]
        self.assertTrue(1 <= len(pal) <= 8)
        self.assertTrue(all(len(h) == 7 and h[0] == "#" and h == h.lower() for h in pal))
        self.assertEqual(len(set(pal)), len(pal))

    def test_every_atlas_colour_is_in_the_palette(self):
        pal = {tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for h in self.meta["palette"]}
        rgba = rgba_of(self.out[0])
        used = {tuple(p[:3]) for p in rgba.reshape(-1, 4) if p[3] == 255}
        self.assertTrue(used <= pal)

    def test_verify_atlas_accepts_the_output_and_rejects_a_changed_pixel(self):
        rgba = rgba_of(self.out[0])
        pal = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in self.meta["palette"]], np.uint8)
        atlas.verify_atlas(rgba, self.meta, pal, 8)
        bad = rgba.copy()
        bad[0, 0, :3] = (1, 2, 3) if tuple(bad[0, 0, :3]) != (1, 2, 3) else (4, 5, 6)
        with self.assertRaises(ValueError):
            atlas.verify_atlas(bad, self.meta, pal, 8)

    def test_saved_atlas_loads_as_a_reproducible_tileset(self):
        out = run(transitions=False)
        ts = tileset_io.load_tileset(png_of(rgba_of(out[0])), out[3])
        self.assertTrue(ts.palette_exact)
        self.assertTrue(ts.reproducible)
        self.assertEqual(list(ts.cells), PLAN_KEYS)

    def test_tile_positions_match_the_atlas_grid(self):
        for t in self.meta["tiles"]:
            self.assertEqual((t["x"], t["y"]), (t["cells"][0]["col"] * 8, t["cells"][0]["row"] * 8))
            self.assertEqual((t["w"], t["h"]), (8 * t["footprint"][0], 8 * t["footprint"][1]))
        self.assertEqual([t["index"] for t in self.meta["tiles"]], list(range(20)))
        self.assertEqual(len({t["key"] for t in self.meta["tiles"]}), 20)

    def test_object_footprint_is_one_cell_for_a_1x1_plan(self):
        by_id = {t["id"]: t for t in self.meta["tiles"]}
        self.assertEqual([by_id[i]["category"] for i in ("g0", "g1", "o0", "v0")], ["ground", "ground", "object", "overlay"])
        self.assertIs(by_id["g0"]["tileable"], True)


class ProvenanceTests(AsmCase):
    @classmethod
    def setUpClass(cls):
        cls.meta = meta_of(run())
        cls.tiles = cls.meta["tiles"]

    def test_generated_tiles_carry_prompt_seed_attempt_job_index(self):
        for i, t in enumerate(self.tiles[:4]):
            self.assertEqual((t["source"], t["attempt"], t["job_index"], t["seed"]), ("generated", 0, i, 7 + i))
            self.assertTrue(t["prompt"].startswith("sty, top-down view"))
            self.assertIsNone(t["wang"])

    def test_wang_tiles_have_no_prompt_and_the_transition_seed(self):
        wang = self.tiles[4:]
        self.assertEqual(len(wang), 16)
        for t in wang:
            self.assertIsNone(t["prompt"])
            self.assertIsNone(t["job_index"])
            self.assertEqual((t["seed"], t["source"], t["category"]), (7, "generated", "ground"))
            self.assertEqual((t["wang"]["a"], t["wang"]["b"]), ("g0", "g1"))
        self.assertEqual(sorted(tuple(t["wang"]["corners"]) for t in wang), sorted(
            (a, b, c, d) for a in (0, 1) for b in (0, 1) for c in (0, 1) for d in (0, 1)))

    def test_wang_seed_follows_the_job_seed_and_transition_index(self):
        data = make_job_json(seed=100)
        meta = meta_of(run(data))
        self.assertEqual({t["seed"] for t in meta["tiles"][4:]}, {100})
        self.assertEqual([t["seed"] for t in meta["tiles"][:4]], [100, 101, 102, 103])

    def test_wang_seed_adds_the_transition_index(self):
        plan = make_plan(3, 1, 1)
        plan["transitions"] = [["g0", "g1"], ["g1", "g2"]]
        meta = meta_of(run(make_job_json(plan, seed=100), images=make_images(5)))
        seeds = {(t["wang"]["a"], t["wang"]["b"]): t["seed"] for t in meta["tiles"] if t["wang"]}
        self.assertEqual(seeds, {("g0", "g1"): 100, ("g1", "g2"): 101})

    def test_wang_seed_wraps_modulo_2_pow_64(self):
        plan = make_plan(3, 1, 1)
        plan["transitions"] = [["g0", "g1"], ["g1", "g2"]]
        meta = meta_of(run(make_job_json(plan, seed=2**64 - 1), images=make_images(5)))
        seeds = {(t["wang"]["a"], t["wang"]["b"]): t["seed"] for t in meta["tiles"] if t["wang"]}
        self.assertEqual(seeds, {("g0", "g1"): 2**64 - 1, ("g1", "g2"): 0})
        self.assertTrue(all(0 <= t["seed"] < 2**64 for t in meta["tiles"]))

    def test_transitions_off_drops_every_wang_tile(self):
        out = run(transitions=False)
        meta = meta_of(out)
        self.assertEqual([t["key"] for t in meta["tiles"]], PLAN_KEYS)
        self.assertEqual(tuple(out[0].shape), (1, 8, 64, 4))
        self.assertEqual(tuple(out[2].shape), (4, 8, 8, 3))
        self.assertIs(meta["snap"]["transitions"], False)

    def test_plan_without_transitions_has_no_wang_even_when_enabled(self):
        meta = meta_of(run(make_job_json(make_plan(1, 1, 0)), images=make_images(2)))
        self.assertEqual([t["id"] for t in meta["tiles"]], ["g0", "o0"])

    def test_variants_expand_to_one_tile_each(self):
        data = make_job_json(make_plan(1, 1, 0, 3))
        meta = meta_of(run(data, images=make_images(len(data["jobs"])), transitions=False))
        self.assertEqual(len(meta["tiles"]), len(data["jobs"]))
        self.assertEqual(sorted({t["variant"] for t in meta["tiles"]}), [0, 1, 2])


class GeometryTests(AsmCase):
    def test_columns_and_spacing_decide_the_canvas(self):
        for cols, spacing in ((4, 0), (4, 2), (3, 16), (64, 0), (1, 1)):
            with self.subTest(cols=cols, spacing=spacing):
                out = run(transitions=False, columns=cols, spacing=spacing)
                meta = meta_of(out)
                rows = -(-4 // cols) if cols < 4 else 1
                step = 8 + spacing
                self.assertEqual((meta["columns"], meta["spacing"]), (cols, spacing))
                self.assertEqual((meta["width"], meta["height"]), (cols * 8 + (cols - 1) * spacing, rows * 8 + (rows - 1) * spacing))
                self.assertEqual(tuple(out[0].shape), (1, meta["height"], meta["width"], 4))
                self.assertEqual(tuple(out[1].shape), (1, meta["height"], meta["width"]))
                t = meta["tiles"][min(1, len(meta["tiles"]) - 1)]
                self.assertEqual(t["x"] % step, 0)

    def test_spacing_gap_is_fully_transparent(self):
        out = run(transitions=False, columns=4, spacing=2)
        rgba = rgba_of(out[0])
        self.assertEqual(int(rgba[:, 8:10, 3].max()), 0)
        self.assertEqual(int(rgba[:8, 0:8, 3].min()), 255)

    def test_tile_preview_batch_is_independent_of_spacing(self):
        a, b = run(columns=4, spacing=0)[2], run(columns=4, spacing=3)[2]
        self.assertTrue(torch.equal(a, b))


class SnapPrecedenceTests(AsmCase):
    def test_widgets_are_the_defaults(self):
        meta = meta_of(run(palette_size=6, seam_blend=0.5, key_tolerance=9, transitions=False))
        self.assertEqual(meta["snap"], {"palette_size": 6, "seam_blend": 0.5, "key_tolerance": 9, "transitions": False,
                                        "palette_mode": "keep", "extra_colours": 0})

    def test_job_json_snap_overrides_widgets(self):
        data = make_job_json(over={"snap": {"palette_size": 5, "transitions": False, "extra_colours": 2}})
        out = run(data, images=make_images(4), palette_size=12, transitions=True)
        meta = meta_of(out)
        self.assertEqual((meta["snap"]["palette_size"], meta["snap"]["transitions"], meta["snap"]["extra_colours"]), (5, False, 2))
        self.assertLessEqual(len(meta["palette"]), 5)
        self.assertEqual(len(meta["tiles"]), 4)

    def test_snap_defaults_fill_what_neither_side_sets(self):
        meta = meta_of(run(make_job_json(over={"snap": {"seam_blend": 0.1}}), images=make_images(4)))
        self.assertEqual(meta["snap"]["seam_blend"], 0.1)
        self.assertEqual(meta["snap"]["key_tolerance"], 40)

    def test_invalid_snap_values_are_rejected(self):
        for snap in ({"palette_size": 0}, {"palette_size": 257}, {"seam_blend": 0.51}, {"key_tolerance": 256},
                     {"extra_colours": 17}, {"transitions": 1}, {"palette_mode": "nope"}, {"bogus": 1}):
            with self.subTest(snap=snap), self.assertRaises(ValueError):
                run(make_job_json(over={"snap": snap}), images=make_images(4))

    def test_refit_without_a_base_is_a_plain_run_with_no_lineage(self):
        meta = meta_of(run(make_job_json(over={"snap": {"palette_mode": "refit"}}), images=make_images(4)))
        self.assertEqual(meta["snap"]["palette_mode"], "refit")
        self.assertNotIn("lineage", meta)


class ReviewAndDeterminismTests(AsmCase):
    def test_review_block_is_passed_through_unchanged(self):
        review = {"rounds": 2, "issues": [{"tile": "g0", "kind": "seam"}], "note": "ok"}
        meta = meta_of(run(make_job_json(over={"review": review}), images=make_images(4)))
        self.assertEqual(meta["review"], review)

    def test_two_runs_are_byte_identical(self):
        a, b = run(), run()
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(a[:3], b[:3])))
        self.assertEqual(a[3], b[3])

    def test_no_base_identical_with_or_without_the_base_key(self):
        plain = run(make_job_json())
        explicit = run(make_job_json(over={"base": None}))
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(plain[:3], explicit[:3])))
        self.assertEqual(plain[3], explicit[3])

    def test_inputs_are_not_mutated(self):
        data = make_job_json()
        before = copy.deepcopy(data)
        imgs = make_images(4)
        snapshot = [i.clone() for i in imgs]
        run(data, images=imgs)
        self.assertEqual(data, before)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(imgs, snapshot)))

    def test_different_images_change_pixels_but_not_structure(self):
        a = run(images=make_images(4))
        b = run(images=make_images(4, start=50))
        self.assertFalse(torch.equal(a[0], b[0]))
        self.assertEqual([t["key"] for t in meta_of(a)["tiles"]], [t["key"] for t in meta_of(b)["tiles"]])

    def test_seed_changes_metadata_seeds_only(self):
        a, b = meta_of(run(make_job_json(seed=7))), meta_of(run(make_job_json(seed=900)))
        self.assertEqual((a["seed"], b["seed"]), (7, 900))
        self.assertEqual([t["seed"] for t in b["tiles"][:4]], [900, 901, 902, 903])

    def test_tiles_batch_matches_the_atlas_cells(self):
        out = run(transitions=False)
        rgb = out[2].numpy()
        rgba = out[0][0].numpy()
        self.assertTrue(np.array_equal(rgb[0], rgba[:8, :8, :3]))
        self.assertTrue(np.array_equal(rgb[1], rgba[:8, 8:16, :3]))


if __name__ == "__main__":
    unittest.main()
