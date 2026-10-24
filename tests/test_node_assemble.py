"""Tests for ``nodes/tilemap_assemble.py`` (D2), part 1: node surface, registration, widget bounds, ``job_json``
validation, image validation. Part 2 (happy paths, ``ccn-tilemap/2`` metadata, base atlas, determinism) is
``tests/test_node_assemble_more.py``, which reuses the helpers below by import. Stdlib unittest, no network.

The node is imported under a synthetic parent package (``_ccn_t14_asm_pkg``) whose ``tilemap_core`` aliases the real
modules (so exception classes are shared); ``tearDownModule`` removes every ``sys.modules`` entry it added. Tile
size 8 and ``cell_px`` 256 keep each ``assemble`` call around 0.1 s.

Residual gaps: pixel-level palette quantisation, seamless blending and Wang tile content are covered by
``tests/test_snap*.py`` / ``test_pixelize*.py``; here only their wiring (keys, counts, sources, shapes) is asserted.
``RecursionError`` while parsing a pathological ``job_json`` is asserted on a 100000-deep array only.
"""

import copy
import importlib
import json
import pkgutil
import sys
import types
import unittest
from pathlib import Path

import numpy as np
import torch

import tilemap_core
from tests import _support
from tilemap_core import expand, plan_schema

PKG = "_ccn_t14_asm_pkg"
ROOT = Path(__file__).resolve().parents[1]
A = None
MISSING = object()  # sentinel: drop this key from the job_json
FIX = "reconnect job_json from CCN_TilemapExpand"


def load_asm():
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
    return importlib.import_module(f"{PKG}.nodes.tilemap_assemble")


def unload_asm():
    for name in [n for n in sys.modules if n == PKG or n.startswith(PKG + ".")]:
        del sys.modules[name]


def setUpModule():
    global A
    A = load_asm()


def tearDownModule():
    unload_asm()


def make_plan(*args):
    return plan_schema.validate_plan(_support.valid_plan(*args), 64)


def make_job_json(plan=None, size=8, seed=7, over=None):
    """Valid ``job_json`` dict (CCN_TilemapExpand shape) built from ``plan``/``size``/``seed``; ``over`` then
    overrides keys (``tile_size``, ``plan``, ``seed``, ...) after the jobs were expanded, and the ``MISSING``
    sentinel drops one."""
    plan = make_plan() if plan is None else plan
    jobs = expand.expand_jobs(plan, size, "top-down", "sty", seed, 256)
    data = {"tile_size": size, "view": "top-down", "plan": plan, "jobs": jobs, "seed": seed,
            "style_prefix": "sty", "cell_px": 256, "snap": {}, "model": "m"}
    for key, value in (over or {}).items():
        if value is MISSING:
            data.pop(key, None)
        else:
            data[key] = value
    return data


def make_images(n, side=256, start=0):
    return [torch.from_numpy(_support.noise_rgb(side, side, start + i).astype(np.float32) / 255)[None] for i in range(n)]


WIDGETS = dict(palette_size=8, columns=8, spacing=0, seam_blend=0.25, key_tolerance=40, transitions=True)


def run(job_json=None, images=None, base_atlas=None, base_alpha=None, raw_job_json=None, **widgets):
    """Call the node the way ComfyUI does with INPUT_IS_LIST: every input wrapped in a one-element list."""
    data = make_job_json() if job_json is None else job_json
    text = raw_job_json if raw_job_json is not None else json.dumps(data)
    images = make_images(len(data["jobs"])) if images is None else images
    w = {**WIDGETS, **widgets}
    extra = {}
    if base_atlas is not None:
        extra["base_atlas"] = [base_atlas]
    if base_alpha is not None:
        extra["base_alpha"] = [base_alpha]
    return A.CCN_TilemapAssemble().assemble(
        images, [text], [w["palette_size"]], [w["columns"]], [w["spacing"]], [w["seam_blend"]],
        [w["key_tolerance"]], [w["transitions"]], **extra)


class AsmCase(unittest.TestCase):
    def rejects(self, *args, contains, **kw):
        with self.assertRaises(ValueError) as cm:
            run(*args, **kw)
        self.assertIs(type(cm.exception), ValueError)
        self.assertIn(contains, str(cm.exception))
        return str(cm.exception)

    def bad_json(self, contains, **changes):
        return self.rejects(make_job_json(over=changes), images=make_images(4), contains=contains)


class SurfaceTests(unittest.TestCase):
    def test_contract(self):
        cls = A.CCN_TilemapAssemble
        self.assertEqual(cls.RETURN_TYPES, ("IMAGE", "MASK", "IMAGE", "STRING"))
        self.assertEqual(cls.RETURN_NAMES, ("atlas", "alpha", "tiles", "metadata"))
        self.assertEqual(cls.FUNCTION, "assemble")
        self.assertEqual(cls.CATEGORY, "custom_comfy_nodes/tilemap")
        self.assertTrue(cls.INPUT_IS_LIST)
        self.assertTrue(callable(getattr(cls(), cls.FUNCTION)))

    def test_input_types(self):
        t = A.CCN_TilemapAssemble.INPUT_TYPES()
        self.assertEqual(list(t), ["required", "optional"])
        req = t["required"]
        self.assertEqual(list(req), ["images", "job_json", "palette_size", "columns", "spacing", "seam_blend",
                                     "key_tolerance", "transitions"])
        self.assertEqual(req["images"], ("IMAGE",))
        self.assertEqual(req["job_json"], ("STRING", {"forceInput": True}))
        self.assertEqual(req["palette_size"], ("INT", {"default": 24, "min": 4, "max": 64, "step": 1}))
        self.assertEqual(req["columns"], ("INT", {"default": 8, "min": 1, "max": 64, "step": 1}))
        self.assertEqual(req["spacing"], ("INT", {"default": 0, "min": 0, "max": 16, "step": 1}))
        self.assertEqual(req["seam_blend"], ("FLOAT", {"default": 0.25, "min": 0.0, "max": 0.5, "step": 0.05}))
        self.assertEqual(req["key_tolerance"], ("INT", {"default": 40, "min": 0, "max": 255, "step": 1}))
        self.assertEqual(req["transitions"], ("BOOLEAN", {"default": True}))
        self.assertEqual(t["optional"], {"base_atlas": ("IMAGE",), "base_alpha": ("MASK",)})

    def test_widget_defaults_are_inside_their_own_bounds(self):
        for name, spec in A.CCN_TilemapAssemble.INPUT_TYPES()["required"].items():
            kind, cfg = spec[0], (spec[1] if len(spec) > 1 else {})
            if kind in ("INT", "FLOAT"):
                self.assertTrue(cfg["min"] <= cfg["default"] <= cfg["max"], name)

    def test_registration(self):
        self.assertEqual(A.NODE_CLASS_MAPPINGS, {"CCN_TilemapAssemble": A.CCN_TilemapAssemble})
        self.assertEqual(A.NODE_DISPLAY_NAME_MAPPINGS, {"CCN_TilemapAssemble": "Tilemap Assemble (pixel-perfect atlas)"})

    def test_module_is_picked_up_by_pack_discovery(self):
        names = [i.name for i in pkgutil.iter_modules([str(ROOT / "nodes")])]
        self.assertIn("tilemap_assemble", names)  # file name does not start with "_"

    def test_constants(self):
        self.assertEqual((A._JOB_JSON_MAX, A._SEED_LIMIT, A._IMAGE_PIXELS_MAX), (400_000, 2**64, 2048 * 2048))


class WidgetTests(AsmCase):
    def test_scalar_widgets_accepted_without_list(self):
        images = make_images(4)
        out = A.CCN_TilemapAssemble().assemble(images, json.dumps(make_job_json()), 8, 8, 0, 0.25, 40, True)
        self.assertEqual(len(out), 4)

    def test_empty_list_inputs_are_missing(self):
        n = A.CCN_TilemapAssemble()
        args = dict(images=make_images(4), job_json=[json.dumps(make_job_json())], palette_size=[8], columns=[8],
                    spacing=[0], seam_blend=[0.25], key_tolerance=[40], transitions=[True])
        for name in ("palette_size", "columns", "spacing", "seam_blend", "key_tolerance", "transitions", "job_json"):
            with self.subTest(name), self.assertRaisesRegex(ValueError, f"^{name} is missing$"):
                n.assemble(**{**args, name: []})

    def test_bounds(self):
        cases = {"palette_size": (3, 65, 8.0, True, "8", None), "columns": (0, 65, 1.5, False),
                 "spacing": (-1, 17, 0.0), "key_tolerance": (-1, 256, 40.0, True)}
        lo = {"palette_size": 4, "columns": 1, "spacing": 0, "key_tolerance": 0}
        hi = {"palette_size": 64, "columns": 64, "spacing": 16, "key_tolerance": 255}
        for name, bads in cases.items():
            for bad in bads:
                with self.subTest(name=name, bad=bad), self.assertRaisesRegex(
                        ValueError, f"^{name} must be an integer from {lo[name]} to {hi[name]}$"):
                    run(**{name: bad})

    def test_seam_blend_bounds(self):
        for bad in (-0.01, 0.51, float("nan"), float("inf"), True, "0.2", None):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, r"^seam_blend must be a number from 0\.0 to 0\.5$"):
                run(seam_blend=bad)

    def test_boundary_values_accepted(self):
        run(palette_size=4, columns=1, spacing=16, seam_blend=0.0, key_tolerance=0)
        run(palette_size=64, columns=64, spacing=0, seam_blend=0.5, key_tolerance=255)
        run(seam_blend=0)  # an int is a valid real number

    def test_transitions_must_be_bool(self):
        for bad in (1, 0, "true", None):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "^transitions must be a boolean$"):
                run(transitions=bad)

    def test_widgets_validated_before_job_json(self):
        with self.assertRaisesRegex(ValueError, "^palette_size must be"):
            run(raw_job_json="not json", palette_size=3)
        with self.assertRaisesRegex(ValueError, "^transitions must be a boolean$"):
            run(raw_job_json="not json", transitions=1)


class JobJsonTextTests(AsmCase):
    def test_not_text(self):
        with self.assertRaisesRegex(ValueError, f"job_json must be text, got int; {FIX}"):
            A.CCN_TilemapAssemble().assemble(make_images(4), [5], [8], [8], [0], [0.25], [40], [True])

    def test_too_long(self):
        self.rejects(raw_job_json="[" + " " * 400_000 + "]", images=[], contains="job_json is too long (400002 chars, limit 400000)")

    def test_one_char_over_the_limit_is_too_long(self):
        self.rejects(raw_job_json="[" + " " * 399_999 + "]", images=[],
                     contains="job_json is too long (400001 chars, limit 400000)")

    def test_length_limit_is_inclusive(self):
        at_limit = "[" + " " * 399_998 + "]"
        self.assertEqual(len(at_limit), 400_000)
        self.rejects(raw_job_json=at_limit, images=[], contains=f"job_json must be a JSON object; {FIX}")

    def test_invalid_json_variants(self):
        good = json.dumps(make_job_json())
        dup = good[:-1] + ', "seed": 8}'
        bad = {"garbage": "{not json", "empty": "", "nan": '{"seed": NaN}', "inf": '{"seed": Infinity}',
               "duplicate key": dup, "deep": "[" * 100_000 + "]" * 100_000, "huge int": '{"seed": ' + "9" * 5000 + "}"}
        for label, text in bad.items():
            with self.subTest(label):
                self.rejects(raw_job_json=text, images=[], contains=f"job_json is not valid JSON; {FIX}")

    def test_not_an_object(self):
        for text in ("[]", "5", '"x"', "null"):
            with self.subTest(text):
                self.rejects(raw_job_json=text, images=[], contains=f"job_json must be a JSON object; {FIX}")

    def test_error_never_echoes_the_input(self):
        secret = "SECRET-TOKEN-123"
        msg = self.rejects(raw_job_json='{"tile_size": "%s"' % secret, images=[], contains="not valid JSON")
        self.assertNotIn(secret, msg)
        msg = self.bad_json("tile_size must be one of", tile_size=secret)
        self.assertNotIn(secret, msg)
        msg = self.bad_json("job_json base is malformed", base={"path": secret})
        self.assertNotIn(secret, msg)


if __name__ == "__main__":
    unittest.main()
