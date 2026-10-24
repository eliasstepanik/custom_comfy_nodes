"""Tests for ``nodes/tilemap_regen.py`` (D12): ``RegenParams``, the validators, ``render_jobs`` with a fake adapter,
``ComfyAdapter.to_rgb`` and the regen_plan -> render_jobs hand-off. Stdlib unittest, no network, no ComfyUI.

The module is imported under a synthetic parent package (``_ccn_t14_regen_pkg``) because ``ComfyAdapter.to_rgb``
uses a relative import (``._tensor_io``); ``tearDownModule`` removes every ``sys.modules`` entry it added. The
module defines no node class: it is the sampler bridge, so there is no INPUT_TYPES/FUNCTION surface, only the
"exposes no node, imports neither torch nor comfy" contract asserted below.

Residual gaps: the other ``ComfyAdapter`` methods (encode, set_guidance, empty_latent, sample, decode,
check_interrupt) import ComfyUI at call time and cannot run in the container; they are exercised only through the
fake adapter, so the exact ComfyUI node chain is a live-probe (LIVE_PROBE P3) concern. Review-loop use of
``render_jobs`` is covered by ``tests/test_review_loop.py`` with a fake ``render`` callable.
"""

import importlib
import sys
import types
import unittest
from pathlib import Path

import numpy as np
import torch

from tests import _support
from tilemap_core import expand, plan_schema, regen_plan

PKG = "_ccn_t14_regen_pkg"
ROOT = Path(__file__).resolve().parents[1]
R = None


def load_regen():
    pkg = types.ModuleType(PKG)
    pkg.__path__ = [str(ROOT)]
    sys.modules[PKG] = pkg
    return importlib.import_module(f"{PKG}.nodes.tilemap_regen")


def unload_regen():
    for name in [n for n in sys.modules if n == PKG or n.startswith(PKG + ".")]:
        del sys.modules[name]


def setUpModule():
    global R
    R = load_regen()


def tearDownModule():
    unload_regen()


class FakeAdapter:
    """Records every call; images are filled with the seed so ordering/mapping is observable."""

    def __init__(self, fail_on_interrupt=None, bad_image=None):
        self.calls = []
        self.fail_on_interrupt = fail_on_interrupt
        self.bad_image = bad_image
        self.checks = 0

    def check_interrupt(self):
        self.checks += 1
        self.calls.append(("interrupt",))
        if self.fail_on_interrupt == self.checks:
            raise KeyboardInterrupt("user interrupt")

    def encode(self, clip, text):
        self.calls.append(("encode", clip, text))
        return ("cond", text)

    def set_guidance(self, cond, guidance):
        self.calls.append(("guidance", cond, guidance))
        return ("guided", cond[1], guidance)

    def empty_latent(self, width, height):
        self.calls.append(("latent", width, height))
        return {"w": width, "h": height}

    def sample(self, model, job, params, positive, negative, latent):
        self.calls.append(("sample", model, job["seed"], params, positive, negative, latent))
        return {"seed": job["seed"], "w": latent["w"], "h": latent["h"]}

    def decode(self, vae, latent):
        self.calls.append(("decode", vae, latent["seed"]))
        return latent

    def to_rgb(self, image):
        self.calls.append(("to_rgb", image["seed"]))
        if self.bad_image is not None:
            return self.bad_image
        return np.full((image["h"], image["w"], 3), (image["seed"] % 255) / 255.0, dtype=np.float32)


class Progress:
    def __init__(self):
        self.updates = []

    def update(self, n):
        self.updates.append(n)


def job(seed=5, w=32, h=16, prompt="a tile", **extra):
    return {"prompt": prompt, "seed": seed, "width": w, "height": h, **extra}


def render(jobs, params=None, neg="blurry", ad=None, progress=None):
    ad = FakeAdapter() if ad is None else ad
    out = R.render_jobs("M", "C", "V", jobs, R.RegenParams() if params is None else params, neg,
                        progress=progress, adapter=ad)
    return out, ad


class ModuleSurfaceTests(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(R.SEED_LIMIT, 2**64)
        self.assertEqual(R.STEPS_MAX, 10_000)
        self.assertEqual(R.CFG_MAX, 100.0)
        self.assertEqual(R.SIDE_RANGE, (16, 4096))
        self.assertEqual(R._PARAM_KEYS, ("steps", "cfg", "sampler", "scheduler", "guidance"))

    def test_no_node_class_registered(self):
        # auto-discovery reads NODE_CLASS_MAPPINGS via getattr(..., {}); the bridge must add nothing
        self.assertFalse(getattr(R, "NODE_CLASS_MAPPINGS", {}))

    def test_import_pulls_neither_torch_nor_comfy_at_module_level(self):
        src = (ROOT / "nodes" / "tilemap_regen.py").read_text(encoding="utf-8")
        head = src.split("class ComfyAdapter")[0]
        for banned in ("import torch", "import comfy", "import nodes", "import node_helpers"):
            self.assertNotIn(banned, head)
        self.assertNotIn("comfy", sys.modules)
        self.assertNotIn("node_helpers", sys.modules)


class RegenParamsTests(unittest.TestCase):
    def test_defaults_are_the_shipped_graph(self):
        p = R.RegenParams()
        self.assertEqual((p.steps, p.cfg, p.sampler, p.scheduler, p.guidance), (20, 1.0, "euler", "simple", 3.5))

    def test_frozen(self):
        with self.assertRaises(Exception):
            R.RegenParams().steps = 3

    def test_boundaries_accepted(self):
        R.RegenParams(steps=1, cfg=0, guidance=0)
        R.RegenParams(steps=10_000, cfg=100, guidance=100.0)
        R.RegenParams(cfg=0.0, guidance=100)

    def test_steps_rejected(self):
        for bad in (0, -1, 10_001, 2.0, True, "20", None):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, r"steps must be an integer from 1 to 10000"):
                R.RegenParams(steps=bad)

    def test_cfg_and_guidance_rejected(self):
        for field in ("cfg", "guidance"):
            for bad in (-0.01, 100.01, float("nan"), float("inf"), float("-inf"), True, "1", None):
                with self.subTest(field=field, bad=bad), self.assertRaisesRegex(
                        ValueError, rf"{field} must be a number from 0\.0 to 100\.0"):
                    R.RegenParams(**{field: bad})

    def test_sampler_scheduler_rejected(self):
        for field in ("sampler", "scheduler"):
            for bad in ("", "   ", None, 3, b"euler"):
                with self.subTest(field=field, bad=bad), self.assertRaisesRegex(
                        ValueError, rf"{field} must be a non-empty string"):
                    R.RegenParams(**{field: bad})

    def test_error_message_names_value(self):
        with self.assertRaises(ValueError) as cm:
            R.RegenParams(steps=0)
        self.assertIn("got 0", str(cm.exception))

    def test_from_mapping_defaults_and_override(self):
        self.assertEqual(R.RegenParams.from_mapping({}), R.RegenParams())
        p = R.RegenParams.from_mapping({"steps": 4, "sampler": "dpmpp_2m"})
        self.assertEqual((p.steps, p.sampler, p.cfg, p.guidance), (4, "dpmpp_2m", 1.0, 3.5))

    def test_from_mapping_unknown_keys_sorted_and_listed(self):
        with self.assertRaises(ValueError) as cm:
            R.RegenParams.from_mapping({"zeta": 1, "alpha": 2, "steps": 3})
        msg = str(cm.exception)
        self.assertIn("unknown params ['alpha', 'zeta']", msg)
        self.assertIn("allowed: ['steps', 'cfg', 'sampler', 'scheduler', 'guidance']", msg)

    def test_from_mapping_validates_values(self):
        with self.assertRaisesRegex(ValueError, "steps must be an integer"):
            R.RegenParams.from_mapping({"steps": 0})


class RenderJobsValidationTests(unittest.TestCase):
    def assertRejected(self, jobs, pattern, **kw):
        ad = FakeAdapter()
        with self.assertRaisesRegex(ValueError, pattern):
            R.render_jobs("M", "C", "V", jobs, kw.pop("params", R.RegenParams()), kw.pop("neg", ""),
                          adapter=ad, **kw)
        self.assertEqual(ad.calls, [], "nothing may be sampled when validation fails")

    def test_jobs_container(self):
        for bad in (None, "abc", {"a": 1}, 5, (j for j in [job()])):
            with self.subTest(bad=type(bad).__name__):
                self.assertRejected(bad, r"jobs must be a list of job mappings, got \w+")

    def test_params_type(self):
        self.assertRejected([job()], r"params must be RegenParams or a mapping, got int", params=3)
        self.assertRejected([job()], r"params must be RegenParams or a mapping, got NoneType", params=None)

    def test_params_mapping_validated(self):
        self.assertRejected([job()], "unknown params", params={"bogus": 1})
        self.assertRejected([job()], "steps must be", params={"steps": 0})

    def test_negative_text_type(self):
        for bad in (None, 3, b"x", ["x"]):
            with self.subTest(bad=bad):
                self.assertRejected([job()], r"negative_text must be a string, got \w+", neg=bad)

    def test_str_subclass_negative_rejected(self):
        class S(str):
            pass
        self.assertRejected([job()], "negative_text must be a string", neg=S("x"))

    def test_progress_checked(self):
        self.assertRejected([job()], r"progress must be None or an object with an update\(n\) method", progress=object())
        self.assertRejected([job()], "progress must be None", progress=types.SimpleNamespace(update=3))

    def test_job_must_be_mapping(self):
        self.assertRejected([job(), "x"], r"jobs\[1\] must be a job mapping, got str")

    def test_prompt(self):
        for bad in ("", "  \n", None, 4):
            with self.subTest(bad=bad):
                self.assertRejected([job(prompt=bad)], r"jobs\[0\]\.prompt must be a non-empty string")
        self.assertRejected([{"seed": 1, "width": 16, "height": 16}], r"jobs\[0\]\.prompt must be")

    def test_seed(self):
        for bad in (-1, 2**64, 1.0, True, "1", None):
            with self.subTest(bad=bad):
                self.assertRejected([job(seed=bad)], r"jobs\[0\]\.seed must be an integer from 0 to 18446744073709551615")
        self.assertRejected([{"prompt": "p", "width": 16, "height": 16}], r"jobs\[0\]\.seed")

    def test_sides_range_and_multiple(self):
        for side in ("width", "height"):
            for bad in (0, 15, 4097, -16, 16.0, True, None, "16"):
                with self.subTest(side=side, bad=bad):
                    self.assertRejected([job(**{{"width": "w", "height": "h"}[side]: bad})],
                                        rf"jobs\[0\]\.{side} must be an integer from 16 to 4096")
            for bad in (17, 20, 4095):
                with self.subTest(side=side, bad=bad):
                    self.assertRejected([job(**{{"width": "w", "height": "h"}[side]: bad})],
                                        rf"jobs\[0\]\.{side} must be a multiple of 16, got {bad}")

    def test_bad_job_late_in_list_blocks_every_sample(self):
        self.assertRejected([job(seed=1), job(seed=2), job(seed=-5)], r"jobs\[2\]\.seed")

    def test_accepts_boundaries(self):
        out, _ = render([job(seed=0, w=16, h=16)])
        self.assertEqual(out[0].shape, (16, 16, 3))
        out, _ = render([job(seed=2**64 - 1, w=4096, h=16)])
        self.assertEqual(out[0].shape, (16, 4096, 3))


if __name__ == "__main__":
    unittest.main()
