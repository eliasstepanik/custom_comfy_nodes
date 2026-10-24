"""Tests for ``nodes/_loop_state.py`` (D13a, part 1): validators, ``quantize``, ``reject_base_inputs``,
``parse_options``. Part 2 (``parse_job_json``, ``LlmBudget``, state/log builders) is ``tests/test_loop_state_more.py``.
Stdlib unittest, no network, no ComfyUI.

``load()`` / ``unload()`` / ``job_text`` / ``opts_kw`` are shared with the other D13 test files (functions only,
so no TestCase is collected twice). The nodes are imported under a synthetic parent package whose ``tilemap_core``
aliases the real modules; ``tearDownModule`` removes every ``sys.modules`` entry it added.

Residual gaps: ``sanitize.clean_text`` / ``RegenParams`` internals are covered by their own tests, here only the
error type and the limits that ``parse_options`` forwards are asserted; ``sampler`` / ``scheduler`` membership in
ComfyUI's lists is not checked (no ComfyUI runtime).
"""

import importlib
import json
import pkgutil
import sys
import types
import unittest
import warnings
from pathlib import Path

import numpy as np

import tilemap_core
from tests import _support
from tilemap_core import expand, llm_client, plan_schema, regen_plan, review

PKG = "_ccn_d13_pkg"
ROOT = Path(__file__).resolve().parents[1]
LS = None


def load():
    """Import ``nodes._loop_state`` and ``nodes.tilemap_review_loop`` under a synthetic parent package."""
    pkg = types.ModuleType(PKG)
    pkg.__path__ = [str(ROOT)]
    sys.modules[PKG] = pkg
    core = types.ModuleType(f"{PKG}.tilemap_core")
    core.__path__ = []
    sys.modules[core.__name__] = core
    for info in pkgutil.iter_modules(tilemap_core.__path__):
        mod = importlib.import_module(f"tilemap_core.{info.name}")
        sys.modules[f"{core.__name__}.{info.name}"] = mod
        setattr(core, info.name, mod)
    return (importlib.import_module(f"{PKG}.nodes._loop_state"),
            importlib.import_module(f"{PKG}.nodes.tilemap_review_loop"))


def unload():
    for name in [n for n in sys.modules if n == PKG or n.startswith(PKG + ".")]:
        del sys.modules[name]


def setUpModule():
    global LS
    LS = load()[0]


def tearDownModule():
    unload()


PLAN = plan_schema.validate_plan(_support.valid_plan(), 64)
JOBS = expand.expand_jobs(PLAN, 16, "top-down", "", 7, cell_px=256)
BASE = {"tile_size": 16, "view": "top-down", "plan": PLAN, "jobs": JOBS}
OPTS = dict(review_mode="heuristic", review_model="auto", max_rounds=3, max_regen_per_round=8, max_attempts=3,
            pass_score=80, layout_source="auto", on_review_error="keep", transitions=True, negative="",
            palette_size=8, seam_blend=0.25, key_tolerance=40, steps=20, cfg=1.0, sampler="euler",
            scheduler="simple", guidance=3.5)


MISSING = object()  # sentinel: drop this key from the job_json


def job_text(**over):
    data = dict(BASE, **over)
    return json.dumps({k: v for k, v in data.items() if v is not MISSING})


def opts_kw(**over):
    return {k: [v] for k, v in dict(OPTS, **over).items()}


class ValidatorTests(unittest.TestCase):
    def test_one(self):
        self.assertEqual(LS.one([5], "n"), 5)
        self.assertEqual(LS.one((5,), "n"), 5)
        self.assertEqual(LS.one(5, "n"), 5)
        self.assertIsNone(LS.one(None, "n"))
        for bad, got in (([], 0), ([1, 2], 2), ((1, 2, 3), 3)):
            with self.assertRaisesRegex(ValueError, f"^n must be a single value, got {got}$"):
                LS.one(bad, "n")

    def test_int_in_bounds_and_types(self):
        self.assertEqual((LS.int_in(2, "n", 2, 5), LS.int_in(5, "n", 2, 5)), (2, 5))
        for bad in (1, 6, True, 3.0, "3", None, [3]):
            with self.assertRaisesRegex(ValueError, "^n must be an integer from 2 to 5$"):
                LS.int_in(bad, "n", 2, 5)
        self.assertEqual(LS.int_in(0, "n", 0, 0), 0)

    def test_float_in_bounds_and_types(self):
        out = LS.float_in(1, "x", 0.0, 0.5 + 0.5)
        self.assertEqual(out, 1.0)
        self.assertIs(type(out), float)
        self.assertEqual((LS.float_in(0.0, "x", 0.0, 0.5), LS.float_in(0.5, "x", 0.0, 0.5)), (0.0, 0.5))
        for bad in (-0.01, 0.51, True, False, "0.2", None, float("nan"), float("inf"), -float("inf")):
            with self.assertRaisesRegex(ValueError, "^x must be a number from 0.0 to 0.5$"):
                LS.float_in(bad, "x", 0.0, 0.5)

    def test_choice(self):
        self.assertEqual(LS.choice("a", "c", ("a", "b")), "a")
        for bad in ("A", "", "c", 1, None, ["a"], type("S", (str,), {})("a")):
            with self.assertRaisesRegex(ValueError, r"^c must be one of \['a', 'b'\]$"):
                LS.choice(bad, "c", ("a", "b"))

    def test_job_key(self):
        self.assertEqual(LS.job_key({"tile_id": "g0", "variant": 2, "x": 1}), "g0#2")

    def test_constants(self):
        self.assertEqual(LS.LOG_FORMAT, "ccn-review-log/1")
        self.assertEqual(LS.REVIEW_MODES, ("vision+heuristic", "heuristic", "vision"))
        self.assertEqual(LS.LAYOUT_SOURCES, ("auto", "llm"))
        self.assertEqual(LS.ON_ERROR, ("keep", "strict"))
        self.assertEqual((LS.JOB_JSON_MAX, LS.LLM_TIMEOUT_S, LS.LAYOUT_TOKENS), (400_000, 120, 8000))


class QuantizeRejectTests(unittest.TestCase):
    def test_quantize_values(self):
        src = np.array([[[0.0, 1.0, 0.5], [float("nan"), 2.0, -1.0], [0.002, 0.998, 0.25]]], np.float32)
        before = src.copy()
        out = LS.quantize(src)
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out.tolist(), [[[0, 255, 128], [0, 255, 0], [1, 254, 64]]])
        np.testing.assert_array_equal(src, before)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # NaN must be replaced before the uint8 cast, not just land on 0
            self.assertEqual(LS.quantize(np.array([[[float("nan"), 0.5, 0.5]]], np.float32)).tolist(), [[[0, 128, 128]]])

    def test_quantize_contiguous_from_strided_input(self):
        src = np.random.default_rng(0).random((8, 8, 3)).astype(np.float32)[::2, ::2]
        self.assertFalse(src.flags["C_CONTIGUOUS"])
        self.assertTrue(LS.quantize(src).flags["C_CONTIGUOUS"])

    def test_reject_base_inputs(self):
        LS.reject_base_inputs({})
        LS.reject_base_inputs({"base_atlas": None, "base_alpha": None})
        for kw in ({"base_atlas": [1]}, {"base_alpha": [1]}, {"base_atlas": 0, "base_alpha": 0}):
            with self.assertRaisesRegex(ValueError, "^base_atlas / base_alpha \\(tileset reuse\\) are not supported by this node yet$"):
                LS.reject_base_inputs(kw)


class ParseOptionsTests(unittest.TestCase):
    def test_defaults_roundtrip(self):
        kw = opts_kw()
        before = json.dumps(kw, sort_keys=True)
        o = LS.parse_options(kw)
        self.assertEqual(json.dumps(kw, sort_keys=True), before)
        self.assertEqual((o.review_mode, o.review_model, o.max_rounds, o.max_regen, o.max_attempts, o.pass_score),
                         ("heuristic", "auto", 3, 8, 3, 80))
        self.assertEqual((o.layout_source, o.on_review_error, o.transitions, o.negative), ("auto", "keep", True, ""))
        self.assertEqual(o.snap, {"palette_size": 8, "seam_blend": 0.25, "key_tolerance": 40, "transitions": True})
        self.assertEqual((o.params.steps, o.params.cfg, o.params.sampler, o.params.scheduler, o.params.guidance),
                         (20, 1.0, "euler", "simple", 3.5))

    def test_limits_accepted_at_edges_rejected_beyond(self):
        edges = {"max_rounds": (0, regen_plan.MAX_ROUND), "max_regen_per_round": (1, 32), "max_attempts": (1, 5),
                 "pass_score": (review.SCORE_MIN, review.SCORE_MAX), "palette_size": (4, 64),
                 "key_tolerance": (0, 255), "seam_blend": (0.0, 0.5), "steps": (1, 10000), "cfg": (0, 100),
                 "guidance": (0, 100)}
        self.assertEqual(regen_plan.MAX_ROUND, 6)
        for name, (lo, hi) in edges.items():
            for ok in (lo, hi):
                LS.parse_options(opts_kw(**{name: ok}))
            step = 0.01 if isinstance(lo, float) or name in ("cfg", "guidance") else 1
            for bad in (lo - step, hi + step):
                with self.assertRaises(ValueError, msg=f"{name}={bad}"):
                    LS.parse_options(opts_kw(**{name: bad}))

    def test_attribute_mapping_at_limits(self):
        o = LS.parse_options(opts_kw(max_rounds=0, max_regen_per_round=32, max_attempts=5, pass_score=0))
        self.assertEqual((o.max_rounds, o.max_regen, o.max_attempts, o.pass_score), (0, 32, 5, 0))

    def test_type_errors_are_value_errors(self):
        for name, bad in (("max_rounds", 3.0), ("max_rounds", True), ("max_rounds", "3"), ("palette_size", 8.0),
                          ("seam_blend", True), ("seam_blend", "0.2"), ("steps", 20.0), ("sampler", ""),
                          ("scheduler", "  "), ("sampler", 3), ("review_mode", "vision-only"), ("review_model", "gpt"),
                          ("layout_source", "file"), ("on_review_error", "raise"), ("review_mode", None)):
            with self.assertRaises(ValueError, msg=f"{name}={bad!r}"):
                LS.parse_options(opts_kw(**{name: bad}))

    def test_transitions_and_negative_messages(self):
        for bad in (1, 0, "true", None):
            with self.assertRaisesRegex(ValueError, "^transitions must be a boolean$"):
                LS.parse_options(opts_kw(transitions=bad))
        for bad in (1, None, b"x"):
            with self.assertRaisesRegex(ValueError, "^negative must be text$"):
                LS.parse_options(opts_kw(negative=bad))
        self.assertFalse(LS.parse_options(opts_kw(transitions=False)).snap["transitions"])

    def test_every_review_model_mode_accepted(self):
        for model in llm_client.MODELS:
            self.assertEqual(LS.parse_options(opts_kw(review_model=model)).review_model, model)
        for mode in LS.REVIEW_MODES:
            self.assertEqual(LS.parse_options(opts_kw(review_mode=mode)).review_mode, mode)

    def test_list_wrapping_errors(self):
        kw = opts_kw()
        kw["max_rounds"] = [1, 2]
        with self.assertRaisesRegex(ValueError, "^max_rounds must be a single value, got 2$"):
            LS.parse_options(kw)
        kw = opts_kw()
        del kw["steps"]
        with self.assertRaises(KeyError):
            LS.parse_options(kw)
        scalar = {k: v[0] for k, v in opts_kw().items()}  # scalars work too
        self.assertEqual(LS.parse_options(scalar).max_rounds, 3)


if __name__ == "__main__":
    unittest.main()
