"""Tests for ``nodes/tilemap_assemble.py`` (D2), part 2: ``job_json`` field validation, the ``base`` block shape and
image-input validation. Part 1 (surface, widgets, JSON text) is ``tests/test_node_assemble.py``, whose helpers
(``make_job_json``, ``run``, ``AsmCase`` ...) and module loader are imported by name. Stdlib unittest, no network.

Residual gaps: see ``tests/test_node_assemble.py``. A 2048x2048 image is only used once (size-check boundary); every
other image is 256x256 to keep the run fast.
"""

import copy
import json
import unittest

import numpy as np
import torch

from tests import test_node_assemble as base
from tests.test_node_assemble import (FIX, MISSING, AsmCase, make_images, make_job_json, make_plan, run,
                                      setUpModule, tearDownModule)  # noqa: F401 (module fixtures re-exported)


class JobJsonFieldTests(AsmCase):
    def test_tile_size(self):
        for bad in (7, 0, 128, "8", 8.0, True, None, MISSING):
            with self.subTest(bad=bad if bad is not MISSING else "missing"):
                self.bad_json(f"job_json tile_size must be one of [8, 16, 24, 32, 48, 64]; {FIX}", tile_size=bad)

    def test_every_tile_size_listed_is_accepted(self):
        for size in (8, 16):
            out = run(make_job_json(size=size), images=make_images(4))
            self.assertEqual(out[0].shape[1:3], (size * 3, size * 8))

    def test_view(self):
        for bad in ("sideways", "", None, 3, MISSING):
            with self.subTest(bad=bad if bad is not MISSING else "missing"):
                self.bad_json("job_json view must be one of", view=bad)

    def test_plan(self):
        bad_plan = make_plan()
        bad_plan["tiles"][0]["id"] = "bad id!"
        for bad in (bad_plan, None, [], "x", MISSING):
            with self.subTest(bad=type(bad).__name__):
                self.bad_json(f"job_json plan failed validation; {FIX}", plan=bad)

    def test_style_cell_model_snap_types(self):
        for field, bads in (("style_prefix", (3, None, ["a"])), ("cell_px", (256.0, "256", True, None)),
                            ("model", (3, None)), ("snap", ([], "x", None))):
            for bad in bads:
                with self.subTest(field=field, bad=bad):
                    self.bad_json(f"job_json style_prefix, cell_px, model or snap is invalid; {FIX}", **{field: bad})

    def test_optional_fields_default(self):
        data = make_job_json(over={"style_prefix": MISSING, "cell_px": MISSING, "model": MISSING, "snap": MISSING})
        meta = json.loads(run(data)[3])
        self.assertEqual((meta["style_prefix"], meta["cell_px"], meta["model"]), ("", 512, ""))

    def test_review_must_be_object_or_null(self):
        for bad in ([], "x", 3):
            with self.subTest(bad=bad):
                self.bad_json(f"job_json review must be an object; {FIX}", review=bad)

    def test_jobs_must_match_plan(self):
        d = make_job_json()
        reordered = list(reversed(d["jobs"]))
        footprint = copy.deepcopy(d["jobs"])
        footprint[2]["footprint"] = [2, 2]
        category = copy.deepcopy(d["jobs"])
        category[0]["category"] = "object"
        variant = copy.deepcopy(d["jobs"])
        variant[1]["variant"] = 1
        extra = d["jobs"] + [copy.deepcopy(d["jobs"][0])]
        cases = {"missing job": d["jobs"][:-1], "reordered": reordered, "footprint": footprint, "category": category,
                 "variant": variant, "duplicate": extra, "empty": [], "not a list": {"a": 1}, "strings": ["g0#0"] * 4,
                 "none": None, "bool footprint": [{**j, "footprint": [True, True]} if j["tile_id"] == "g0" else j for j in d["jobs"]]}
        for label, jobs in cases.items():
            with self.subTest(label):
                self.bad_json(f"job_json jobs do not match its plan; {FIX}", jobs=jobs)

    def test_only_the_signature_decides_identity(self):
        """width/height/etc. of a row are ignored (the plan's own jobs drive the pixels); prompt/seed/attempt are provenance."""
        d = make_job_json()
        tampered = copy.deepcopy(d["jobs"])
        for j in tampered:
            j["width"] = 99999
            j["prompt"] = "custom prompt"
            j["attempt"] = 2
        tampered[0]["seed"] = 12345
        tiles = json.loads(run(make_job_json(over={"jobs": tampered}))[3])["tiles"]
        self.assertEqual([(t["prompt"], t["attempt"]) for t in tiles[:4]], [("custom prompt", 2)] * 4)
        self.assertEqual(tiles[0]["seed"], 12345)
        self.assertEqual(tiles[1]["seed"], 8)  # untouched rows keep the expand seed

    def test_seed(self):
        for bad in (-1, 2**64, 7.0, True, "7"):
            with self.subTest(bad=bad):
                self.bad_json("job_json job seed must be an integer from 0 to 18446744073709551615", seed=bad)

    def test_seed_derived_from_first_job_when_absent(self):
        d = make_job_json(over={"seed": MISSING})
        self.assertEqual(json.loads(run(d)[3])["seed"], 7)  # jobs[0].seed - jobs[0].index
        d = make_job_json(make_plan(1, 1, 0), over={"seed": MISSING})
        self.assertEqual(json.loads(run(d)[3])["seed"], 7)

    def test_seed_limits_accepted(self):
        self.assertEqual(json.loads(run(make_job_json(seed=0))[3])["seed"], 0)
        top = 2**64 - 8
        self.assertEqual(json.loads(run(make_job_json(seed=top))[3])["seed"], top)


class BaseBlockShapeTests(AsmCase):
    def good(self):
        return {"path": "ts/a", "keys": ["g0#0"], "palette": ["#0a141e"], "snap": {}}

    def test_malformed_base_values(self):
        g = self.good()
        cases = {"not dict": [], "empty dict": {}, "extra key": {**g, "x": 1}, "missing key": {k: v for k, v in g.items() if k != "snap"},
                 "path empty": {**g, "path": ""}, "path long": {**g, "path": "p" * 257}, "path int": {**g, "path": 3},
                 "keys empty": {**g, "keys": []}, "keys many": {**g, "keys": [f"k{i}#0" for i in range(513)]},
                 "keys dup": {**g, "keys": ["g0#0", "g0#0"]}, "keys non-str": {**g, "keys": [1]}, "keys tuple-like": {**g, "keys": "g0#0"},
                 "palette empty": {**g, "palette": []}, "palette big": {**g, "palette": ["#000000"] * 257},
                 "palette upper": {**g, "palette": ["#0A141E"]}, "palette short": {**g, "palette": ["#fff"]},
                 "palette trailing": {**g, "palette": ["#0a141e "]}, "palette int": {**g, "palette": [5]},
                 "snap list": {**g, "snap": []}}
        for label, base in cases.items():
            with self.subTest(label):
                self.bad_json(f"job_json base is malformed; {FIX}", base=base)

    def test_base_without_atlas_inputs(self):
        self.rejects(make_job_json(over={"base": self.good()}), contains="connect base_atlas and base_alpha together, and only when job_json has a base")

    def test_atlas_inputs_without_base(self):
        t = torch.zeros(1, 8, 8, 3)
        self.rejects(base_atlas=t, base_alpha=torch.ones(1, 8, 8), contains="only when job_json has a base")
        self.rejects(base_atlas=t, contains="connect base_atlas and base_alpha together")
        self.rejects(base_alpha=torch.ones(1, 8, 8), contains="connect base_atlas and base_alpha together")


class ImageInputTests(AsmCase):
    def test_count_mismatch(self):
        self.rejects(images=make_images(3), contains="got 3 images for 4 jobs; connect one image per job in job order")
        self.rejects(images=make_images(5), contains="got 5 images for 4 jobs")
        self.rejects(images=[], contains="got 0 images for 4 jobs")

    def test_not_a_tensor_or_wrong_rank(self):
        imgs = make_images(4)
        for label, bad in (("ndarray", np.zeros((1, 256, 256, 3), np.float32)), ("rank3", torch.zeros(256, 256, 3)),
                           ("rank5", torch.zeros(1, 1, 8, 8, 3)), ("none", None)):
            with self.subTest(label):
                self.rejects(images=imgs[:3] + [bad], contains="image 3 must be an IMAGE tensor [B,H,W,C]")

    def test_oversize_rejected_before_conversion(self):
        imgs = make_images(4)
        imgs[1] = torch.zeros(1, 2049, 2048, 3)
        self.rejects(images=imgs, contains="image 1 is too large (2048x2049 px)")

    def test_max_pixel_count_boundary_is_accepted_by_size_check(self):
        imgs = make_images(4)
        imgs[0] = torch.full((1, 2048, 2048, 3), 0.5)
        self.assertEqual(len(run(images=imgs, transitions=False)), 4)

    def test_channel_errors_name_the_image(self):
        imgs = make_images(4)
        imgs[2] = torch.zeros(1, 256, 256, 2)
        with self.assertRaisesRegex(ValueError, r"^image 2: "):
            run(images=imgs)

    def test_integer_tensor_rejected(self):
        imgs = make_images(4)
        imgs[0] = torch.zeros(1, 256, 256, 3, dtype=torch.uint8)
        with self.assertRaisesRegex(ValueError, r"^image 0: "):
            run(images=imgs)

    def test_only_batch_element_zero_is_used(self):
        imgs = make_images(4)
        wide = [torch.cat([i, torch.rand_like(i)], 0) for i in imgs]
        a, b = run(images=imgs), run(images=wide)
        self.assertEqual(a[3], b[3])
        self.assertTrue(torch.equal(a[0], b[0]))

    def test_four_channel_images_use_rgb(self):
        imgs = make_images(4)
        rgba = [torch.cat([i, torch.zeros(1, 256, 256, 1)], 3) for i in imgs]
        self.assertTrue(torch.equal(run(images=imgs)[0], run(images=rgba)[0]))

    def test_images_not_mutated(self):
        imgs = make_images(4)
        before = [i.clone() for i in imgs]
        run(images=imgs)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(imgs, before)))

    def test_single_tensor_instead_of_list(self):
        d = make_job_json(plan=make_plan(1, 0, 0))
        out = base.A.CCN_TilemapAssemble().assemble(make_images(1)[0], [json.dumps(d)], [8], [8], [0], [0.25], [40], [True])
        self.assertEqual(len(out), 4)


if __name__ == "__main__":
    unittest.main()
