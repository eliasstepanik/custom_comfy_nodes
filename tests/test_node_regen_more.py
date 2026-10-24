"""Tests for ``nodes/tilemap_regen.py`` (D12), part 2: ``ComfyAdapter.to_rgb`` on real tensors and the
``regen_plan.plan_regen`` -> ``render_jobs`` hand-off. Part 1 (params, validation, call sequence) is
``tests/test_node_regen.py``, whose ``FakeAdapter`` / ``job`` / ``render`` helpers and module loader are reused
(functions and classes imported by name, so no TestCase is collected twice). Stdlib unittest, no network.

Residual gaps: the ComfyUI calls themselves (``nodes``, ``node_helpers``, ``comfy.model_management``) are replaced
by recording fakes in ``sys.modules``, so only the argument wiring of each ``ComfyAdapter`` method is asserted, not
ComfyUI behaviour (LIVE_PROBE P3). See also ``tests/test_node_regen.py``; additionally the real VAE output is not clipped here (the
bridge returns it as is, Assemble quantises), so only value pass-through is asserted. Finding for the owner of
``ComfyAdapter.to_rgb``: its docstring promises a copy, but on a CPU float32 tensor the result shares memory with
the tensor (``np.ascontiguousarray`` of a contiguous view copies nothing); harmless today (decoded images are
single-use), so the sharing is deliberately not asserted either way.
"""

import sys
import types
import unittest
from unittest import mock

import numpy as np
import torch

from tests import _support
from tests import test_node_regen as base
from tilemap_core import expand, plan_schema, regen_plan


def setUpModule():
    base.setUpModule()


def tearDownModule():
    base.tearDownModule()


class ToRgbTests(unittest.TestCase):
    def adapter(self):
        return base.R.ComfyAdapter()

    def test_returns_contiguous_float32_copy_of_batch_element_0(self):
        t = torch.rand(1, 16, 32, 3)
        out = self.adapter().to_rgb(t)
        self.assertEqual(out.shape, (16, 32, 3))
        self.assertEqual(out.dtype, np.float32)
        self.assertTrue(out.flags["C_CONTIGUOUS"])
        self.assertTrue(np.array_equal(out, t[0].numpy()))
        # note: ``out`` is a view of the tensor's memory (docstring says copy) - see Residual gaps, not asserted

    def test_values_not_clipped_or_quantised(self):
        t = torch.tensor([[[[-0.25, 0.5, 1.5]]]])
        out = self.adapter().to_rgb(t)
        self.assertEqual(out.tolist(), [[[-0.25, 0.5, 1.5]]])

    def test_half_precision_becomes_float32(self):
        out = self.adapter().to_rgb(torch.zeros(1, 2, 2, 3, dtype=torch.float16))
        self.assertEqual(out.dtype, np.float32)

    def test_bad_shapes_rejected(self):
        for shape in ((2, 4, 4, 3), (1, 4, 4, 4), (1, 4, 4, 1), (4, 4, 3), (1, 4, 4)):
            with self.subTest(shape=shape), self.assertRaisesRegex(
                    ValueError, r"decoded image must have shape \[1,H,W,3\], got " + str(shape).replace("(", r"\(").replace(")", r"\)")):
                self.adapter().to_rgb(torch.zeros(*shape))

    def test_non_tensor_and_integer_tensor_rejected(self):
        with self.assertRaisesRegex(ValueError, "decoded image must be a torch.Tensor, got ndarray"):
            self.adapter().to_rgb(np.zeros((1, 2, 2, 3), np.float32))
        with self.assertRaisesRegex(ValueError, "decoded image must be a floating-point tensor"):
            self.adapter().to_rgb(torch.zeros(1, 2, 2, 3, dtype=torch.uint8))

    def test_to_rgb_output_passes_render_jobs_size_check(self):
        class Tensors(base.FakeAdapter):
            def decode(self, vae, latent):
                return torch.full((1, latent["h"], latent["w"], 3), 0.5)

            def to_rgb(self, image):
                return base.R.ComfyAdapter().to_rgb(image)
        out, _ = base.render([base.job(w=48, h=32)], ad=Tensors())
        self.assertEqual(out[0].shape, (32, 48, 3))
        self.assertTrue(bool((out[0] == 0.5).all()))


def mk(tid, var=0, cat="ground", seed=100, w=256, h=256):
    return {"tile_id": tid, "variant": var, "category": cat, "seed": seed, "footprint": [1, 1],
            "prompt": f"pixel art {tid}", "width": w, "height": h}


class AdapterWiringTests(unittest.TestCase):
    """``ComfyAdapter`` against fake ComfyUI modules: every call receives exactly the arguments the shipped graph uses."""

    def setUp(self):
        self.calls = []
        calls = self.calls

        class Clip:
            def encode(self, clip, text):
                calls.append(("encode", clip, text))
                return (("cond", text),)

        class Vae:
            def decode(self, vae, latent):
                calls.append(("decode", vae, latent))
                return ("image",)

        fake_nodes = types.ModuleType("nodes")
        fake_nodes.CLIPTextEncode = Clip
        fake_nodes.VAEDecode = Vae
        fake_nodes.common_ksampler = lambda *a, **k: calls.append(("ksampler", a, k)) or ({"samples": "out"},)
        helpers = types.ModuleType("node_helpers")
        helpers.conditioning_set_values = lambda cond, values: calls.append(("set", cond, values)) or ("set", cond)
        mm = types.ModuleType("comfy.model_management")
        mm.throw_exception_if_processing_interrupted = lambda: calls.append(("interrupt",))
        mm.intermediate_device = lambda: "cpu"
        mm.intermediate_dtype = lambda: torch.float16
        comfy = types.ModuleType("comfy")
        comfy.model_management = mm
        patcher = mock.patch.dict(sys.modules, {"nodes": fake_nodes, "node_helpers": helpers, "comfy": comfy,
                                                "comfy.model_management": mm})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.ad = base.R.ComfyAdapter()

    def test_check_interrupt_delegates(self):
        self.ad.check_interrupt()
        self.assertEqual(self.calls, [("interrupt",)])

    def test_encode_and_decode_return_the_first_tuple_element(self):
        self.assertEqual(self.ad.encode("C", "hi"), ("cond", "hi"))
        self.assertEqual(self.ad.decode("V", {"samples": 1}), "image")
        self.assertEqual(self.calls[1], ("decode", "V", {"samples": 1}))

    def test_set_guidance_sets_only_the_guidance_value(self):
        self.ad.set_guidance("cond", 2.5)
        self.assertEqual(self.calls, [("set", "cond", {"guidance": 2.5})])

    def test_empty_latent_is_a_batch_of_one_at_one_eighth_resolution(self):
        lat = self.ad.empty_latent(64, 32)
        self.assertEqual(tuple(lat["samples"].shape), (1, 16, 4, 8))
        self.assertEqual(lat["samples"].dtype, torch.float16)
        self.assertEqual(float(lat["samples"].abs().max()), 0.0)
        self.assertEqual(lat["downscale_ratio_spacial"], 8)

    def test_sample_passes_seed_and_params_in_ksampler_order_with_full_denoise(self):
        p = base.R.RegenParams(steps=7, cfg=2.0, sampler="dpmpp_2m", scheduler="karras")
        out = self.ad.sample("M", {"seed": 123456789012}, p, "pos", "neg", {"samples": "lat"})
        self.assertEqual(out, {"samples": "out"})
        self.assertEqual(self.calls, [("ksampler", ("M", 123456789012, 7, 2.0, "dpmpp_2m", "karras", "pos", "neg",
                                                    {"samples": "lat"}), {"denoise": 1.0})])


class RegenPlanHandoffTests(unittest.TestCase):
    def test_regen_jobs_render_with_new_seeds(self):
        jobs = {"g0#0": mk("g0", seed=100), "g1#0": mk("g1", seed=200)}
        issues = [{"tile": "g0#0", "kind": "seam", "severity": "high", "fix_hint": "no seam"}]
        new, skipped = regen_plan.plan_regen(issues, jobs, {"g0#0": 0, "g1#0": 0}, 4, 3, 1)
        self.assertEqual(skipped, [])
        self.assertEqual(len(new), 1)
        self.assertEqual(new[0]["seed"], (100 + regen_plan.SEED_STEP) % 2**64)
        out, ad = base.render(new)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].shape, (256, 256, 3))
        sampled = [c for c in ad.calls if c[0] == "sample"]
        self.assertEqual(sampled[0][2], new[0]["seed"])  # regenerated seed reaches the sampler
        self.assertNotEqual(sampled[0][2], 100)

    def test_regen_jobs_keep_extra_keys_and_are_not_mutated(self):
        jobs = {"g0#0": mk("g0")}
        new, _ = regen_plan.plan_regen(
            [{"tile": "g0#0", "kind": "fringe", "severity": "medium", "fix_hint": ""}], jobs, {}, 4, 3, 2)
        snapshot = [dict(j) for j in new]
        base.render(new)
        self.assertEqual(new, snapshot)
        self.assertEqual(new[0]["round"], 2)

    def test_expand_jobs_are_valid_render_input(self):
        plan = plan_schema.validate_plan(_support.valid_plan(), 64)
        jobs = expand.expand_jobs(plan, 8, "top-down", "sty", 7, 256)
        out, ad = base.render(jobs)
        self.assertEqual(len(out), len(jobs))
        self.assertTrue(all(o.shape == (256, 256, 3) for o in out))
        self.assertEqual([c[2] for c in ad.calls if c[0] == "sample"], [j["seed"] for j in jobs])

    def test_job_without_size_is_rejected_not_defaulted(self):
        j = mk("g0")
        del j["width"]
        with self.assertRaisesRegex(ValueError, r"jobs\[0\]\.width must be an integer"):
            base.render([j])


if __name__ == "__main__":
    unittest.main()
