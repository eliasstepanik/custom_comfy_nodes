"""Tests for ``nodes/tilemap_load.py`` (D7, ``CCN_LoadTilemap``) and, through it, ``nodes/_tensor_io.py``
(``np_to_image`` / ``alpha_to_mask``). Stdlib unittest; real torch tensors; files live in temp dirs; ComfyUI's
``folder_paths`` is replaced by a stub in ``sys.modules`` only inside ``mock.patch.dict`` (restored afterwards).

``load_pkg()`` / ``unload_pkg()`` import a node module under a synthetic parent package aliasing the real
``tilemap_core``; the vision and scene test files import them (functions only: no TestCase is collected twice).

Residual gaps: ``tileset_io`` internals (symlinks, size caps, PNG/metadata verification) live in
``tests/test_tileset_io*.py``, here only the node-level ValueError contract is asserted; ComfyUI's real
``folder_paths`` and any GPU/ComfyUI runtime are not run.
"""

import hashlib
import importlib
import json
import math
import os
import pkgutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import tilemap_core
from tests.test_tileset_io import KEYS, PAL, edited, make_fixture, png_of

try:
    import torch
except ImportError:  # the container has torch; a bare dev box may not
    torch = None

ROOT = Path(__file__).resolve().parents[1]
PKG = "_ccn_t13_load_pkg"
L = None


def load_pkg(pkg_name, module):
    """Import ``nodes.<module>`` under a synthetic parent package named ``pkg_name`` (needs no ComfyUI)."""
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(ROOT)]
    sys.modules[pkg_name] = pkg
    core = types.ModuleType(f"{pkg_name}.tilemap_core")
    core.__path__ = []
    sys.modules[core.__name__] = core
    for info in pkgutil.iter_modules(tilemap_core.__path__):
        mod = importlib.import_module(f"tilemap_core.{info.name}")
        sys.modules[f"{core.__name__}.{info.name}"] = mod
        setattr(core, info.name, mod)
    return importlib.import_module(f"{pkg_name}.nodes.{module}")


def unload_pkg(pkg_name):
    for name in [n for n in sys.modules if n == pkg_name or n.startswith(pkg_name + ".")]:
        del sys.modules[name]


def setUpModule():
    global L
    if torch is None:
        raise unittest.SkipTest("torch not installed")
    L = load_pkg(PKG, "tilemap_load")


def tearDownModule():
    unload_pkg(PKG)


def folder_stub(out_dir, in_dir):
    return types.SimpleNamespace(get_output_directory=lambda: out_dir, get_input_directory=lambda: in_dir)


class LoadCase(unittest.TestCase):
    def setUp(self):
        tmp = self.enterContext(tempfile.TemporaryDirectory())
        self.out, self.inp = (os.path.join(tmp, d) for d in ("output", "input"))
        os.makedirs(self.out), os.makedirs(self.inp)
        self.enterContext(mock.patch.dict(sys.modules, {"folder_paths": folder_stub(self.out, self.inp)}))
        self.fx = make_fixture()
        self.node = L.CCN_LoadTilemap()

    def put(self, base, rel, png=None, text=None):
        for ext, data in ((".png", png or self.fx.png), (".json", text or self.fx.text)):
            path = Path(base, *(rel + ext).split("/"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))

    def load(self, path="tilemap/ts", source="output", strict=True):
        return self.node.load(path, source, strict)


class TestContract(unittest.TestCase):
    def test_node_surface_exact(self):
        cls = L.CCN_LoadTilemap
        self.assertEqual(cls.INPUT_TYPES(), {"required": {
            "path": ("STRING", {"default": "tilemap/tilemap_00001_"}),
            "source": (["output", "input"], {"default": "output"}),
            "strict": ("BOOLEAN", {"default": True})}})
        self.assertEqual(cls.RETURN_TYPES, ("IMAGE", "MASK", "IMAGE", "STRING", "STRING", "STRING"))
        self.assertEqual(cls.RETURN_NAMES, ("atlas", "alpha", "tiles", "metadata", "job_json", "summary"))
        self.assertEqual(cls.OUTPUT_IS_LIST, (False, False, True, False, False, False))
        self.assertEqual((cls.FUNCTION, cls.CATEGORY, L.ROOT_DIR), ("load", "custom_comfy_nodes/tilemap", "tilemap"))
        self.assertTrue(callable(getattr(cls, cls.FUNCTION)))
        self.assertEqual(L.NODE_CLASS_MAPPINGS, {"CCN_LoadTilemap": cls})
        self.assertEqual(L.NODE_DISPLAY_NAME_MAPPINGS, {"CCN_LoadTilemap": "Load Tilemap (PNG + JSON)"})
        self.assertEqual(len(cls.RETURN_TYPES), len(cls.OUTPUT_IS_LIST))

    def test_json_helper_is_compact_sorted_and_refuses_nan(self):
        self.assertEqual(L._json({"b": [1, 2], "a": "x"}), '{"a":"x","b":[1,2]}')
        self.assertRaises(ValueError, L._json, {"a": float("nan")})


class TestRoot(LoadCase):
    def test_sources_map_to_trusted_dirs(self):
        self.assertEqual(L._root("output"), (self.out, "tilemap"))
        self.assertEqual(L._root("input"), (self.inp, ""))

    def test_unknown_source_rejected_before_folder_paths(self):
        with mock.patch.dict(sys.modules, {"folder_paths": None}):
            for bad in ("Output", "tmp", "", None, 1, ["output"]):
                with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, r"source must be one of \['output', 'input'\]"):
                    L._root(bad)

    def test_missing_folder_paths_is_value_error(self):
        with mock.patch.dict(sys.modules, {"folder_paths": None}):  # None entry makes `import folder_paths` fail
            for source in ("output", "input"):
                with self.subTest(source=source), self.assertRaises(ValueError) as cm:
                    L._root(source)
                self.assertIs(type(cm.exception), ValueError)
                self.assertEqual(str(cm.exception), "ComfyUI folder_paths is not available")
                self.assertIsNone(cm.exception.__cause__)


class TestLoadHappyPath(LoadCase):
    def test_all_six_outputs_match_the_fixture(self):
        self.put(self.out, "tilemap/ts")
        out = self.load()
        atlas, alpha, tiles, meta, job, summary = out  # exactly six outputs
        fx = self.fx
        self.assertEqual((atlas.dtype, tuple(atlas.shape), alpha.dtype, tuple(alpha.shape)),
                         (torch.float32, (1, 16, 32, 4), torch.float32, (1, 16, 32)))
        self.assertTrue(np.array_equal(atlas[0].numpy(), fx.rgba.astype(np.float32) / np.float32(255.0)))
        self.assertTrue(np.array_equal(alpha[0].numpy(), fx.rgba[..., 3].astype(np.float32) / np.float32(255.0)))
        self.assertEqual((type(tiles), [tuple(t.shape) for t in tiles]), (list, [(1, 8, 8, 4)] * 2 + [(1, 16, 16, 4), (1, 8, 8, 4)]))
        self.assertTrue(all(t.dtype == torch.float32 and 0.0 <= float(t.min()) <= float(t.max()) <= 1.0 for t in tiles))
        self.assertEqual(meta, json.dumps(fx.meta, sort_keys=True, separators=(",", ":")))
        self.assertEqual(summary, "ccn-tilemap/2 tilemap/ts: 32x16 px, tile_size 8, 4 tiles, palette 3 colours\nreproducible: true")

    def test_job_json_carries_jobs_and_base_block(self):
        self.put(self.out, "tilemap/ts")
        job = json.loads(self.load()[4])
        self.assertEqual(job["base"]["path"], "tilemap/ts")
        self.assertEqual(job["base"]["keys"], KEYS)
        self.assertEqual(job["base"]["palette"], self.fx.meta["palette"])
        self.assertEqual((job["tile_size"], job["view"], job["seed"], job["snap"]["palette_mode"]), (8, "top-down", 100, "keep"))
        self.assertEqual(job["base"]["snap"], job["snap"])

    def test_input_source_has_no_forced_subfolder(self):
        self.put(self.inp, "sub/ts")
        out = self.load("sub/ts", "input")
        self.assertIn("ccn-tilemap/2 sub/ts:", out[5])
        self.assertEqual(json.loads(out[4])["base"]["path"], "sub/ts")
        with self.assertRaises(ValueError):  # the same stem is not under output/tilemap/
            self.load("sub/ts", "output")

    def test_output_source_forces_tilemap_subfolder(self):
        self.put(self.out, "ts")  # directly in output/, not output/tilemap/
        with self.assertRaises(ValueError):
            self.load("ts", "output")
        self.put(self.out, "tilemap/ts")
        self.assertIn(" tilemap/ts:", self.load("ts", "output")[5])  # folder is added, not required

    def test_deterministic_and_inputs_not_mutated(self):
        self.put(self.out, "tilemap/ts")
        files = [Path(self.out, "tilemap", "ts" + e) for e in (".png", ".json")]
        before = [f.read_bytes() for f in files]
        first, second = self.load(), self.load()
        self.assertEqual(first[3:], second[3:])
        self.assertTrue(all(torch.equal(first[i], second[i]) for i in (0, 1)))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first[2], second[2])))
        first[0].zero_()  # tensors are independent copies, not views of shared state
        self.assertFalse(torch.equal(first[0], self.load()[0]))
        self.assertEqual([f.read_bytes() for f in files], before)

    def test_tile_sizes_8_to_64_alpha_binary_palette_uniform(self):
        for n in (8, 16, 24, 32, 48, 64):
            with self.subTest(n=n):
                fx = make_fixture(n=n)
                self.put(self.out, "tilemap/ts", fx.png, fx.text)
                atlas, alpha, tiles, meta, _, summary = self.load()
                self.assertEqual(tuple(atlas.shape), (1, fx.meta["height"], fx.meta["width"], 4))
                self.assertEqual(tuple(alpha.shape), (1, fx.meta["height"], fx.meta["width"]))
                self.assertEqual([tuple(t.shape[1:]) for t in tiles], [(n, n, 4)] * 2 + [(2 * n, 2 * n, 4)] + [(n, n, 4)])
                self.assertEqual(set(torch.unique(alpha).tolist()), {0.0, 1.0})  # binary alpha
                px = np.rint(atlas[0].numpy() * 255).astype(np.uint8)
                colours = {tuple(c) for c in px[px[..., 3] == 255][:, :3].tolist()}
                self.assertTrue(colours and colours <= set(PAL), colours)  # uniform palette
                self.assertIn(f"tile_size {n}, 4 tiles", summary)
                self.assertEqual(json.loads(meta)["tile_size"], n)


class TestLoadStrictAndFormats(LoadCase):
    def setUp(self):
        super().setUp()
        bad = self.fx.rgba.copy()
        bad[0, 0, :3] = (1, 1, 1)  # one opaque ground pixel off palette
        self.bad = png_of(bad)

    def test_off_palette_strict_true_raises_value_error(self):
        self.put(self.out, "tilemap/ts", self.bad)
        with self.assertRaises(ValueError) as cm:
            self.load()
        self.assertIs(type(cm.exception), ValueError)
        self.assertIn("not in the palette", str(cm.exception))

    def test_off_palette_strict_false_forces_refit_and_warns(self):
        self.put(self.out, "tilemap/ts", self.bad)
        _, _, _, _, job_text, summary = self.load(strict=False)
        job = json.loads(job_text)
        self.assertEqual((job["snap"]["palette_mode"], job["base"]["snap"]["palette_mode"]), ("refit", "refit"))
        self.assertEqual(job["base"]["snap"]["palette_size"], 3)  # only palette_mode is rewritten
        lines = summary.split("\n")
        self.assertEqual(lines[2], "warning: 1 opaque pixels use colours outside the palette (palette_mode forced to refit)")
        self.assertEqual(len(lines), 3)

    def test_strict_false_forces_refit_even_on_clean_file(self):
        self.put(self.out, "tilemap/ts")
        job = json.loads(self.load(strict=False)[4])
        self.assertEqual((job["snap"]["palette_mode"], job["base"]["snap"]["palette_mode"]), ("refit", "refit"))
        self.assertNotIn("warning", self.load(strict=False)[5])
        self.assertEqual(json.loads(self.load(strict=True)[4])["snap"]["palette_mode"], "keep")

    def test_format_one_and_plan_less_are_not_reproducible(self):
        for name, fn in (("fmt1", lambda m: m.update(format="ccn-tilemap/1")), ("noplan", lambda m: m.pop("plan"))):
            with self.subTest(name):
                self.put(self.out, "tilemap/ts", text=edited(self.fx, fn))
                out = self.load()
                self.assertEqual(out[4], "")
                lines = out[5].split("\n")
                self.assertEqual(lines[1:], ["reproducible: false",
                                             "no plan/prompts stored: improve, extend and restyle need a ccn-tilemap/2 file with a plan"])
                self.assertEqual(len(out[2]), 4)  # tiles still sliced

    def test_strict_must_be_a_real_bool_checked_first(self):
        self.put(self.out, "tilemap/ts")
        for bad in (1, 0, "yes", None, "True", 1.0):
            with self.subTest(bad=bad), self.assertRaises(ValueError) as cm:
                self.load(strict=bad)
            self.assertIs(type(cm.exception), ValueError)
            self.assertEqual(str(cm.exception), "strict must be a boolean")
        with mock.patch.dict(sys.modules, {"folder_paths": None}), self.assertRaisesRegex(ValueError, "strict must be a boolean"):
            self.load(strict="no")  # before folder_paths is touched


class TestLoadErrors(LoadCase):
    def test_every_failure_is_exactly_value_error_without_leaking_root(self):
        self.put(self.out, "tilemap/ts")
        self.put(self.out, "tilemap/badpng", png=b"not a png")
        self.put(self.out, "tilemap/badjson", text="{nope")
        cases = [("tilemap/missing", "output"), ("../ts", "output"), ("/etc/passwd", "output"), ("a/../ts", "output"),
                 ("badpng", "output"), ("badjson", "output"), ("", "output"), ("ts", "bogus")]
        for path, source in cases:
            with self.subTest(path=path, source=source), self.assertRaises(ValueError) as cm:
                self.load(path, source)
            self.assertIs(type(cm.exception), ValueError)
            self.assertNotIn(self.out, str(cm.exception))
        with mock.patch.dict(sys.modules, {"folder_paths": None}), self.assertRaisesRegex(ValueError, "folder_paths is not available"):
            self.load()

    def test_missing_sidecar_names_the_fix(self):
        self.put(self.out, "tilemap/ts")
        Path(self.out, "tilemap", "ts.json").unlink()
        self.assertRaisesRegex(ValueError, "metadata sidecar", self.load)

class TestIsChanged(LoadCase):
    def test_hash_matches_file_content_and_tracks_edits(self):
        self.put(self.out, "tilemap/ts")
        want = hashlib.sha256(self.fx.png + b"\0" + self.fx.text.encode("utf-8")).hexdigest()
        got = L.CCN_LoadTilemap.IS_CHANGED("tilemap/ts", "output", True)
        self.assertEqual(got, want)
        self.assertEqual(L.CCN_LoadTilemap.IS_CHANGED("tilemap/ts", "output", False), want)  # strict is not part of the hash
        self.put(self.out, "tilemap/ts", text=edited(self.fx, lambda m: m.update(seed=7)))
        self.assertNotEqual(L.CCN_LoadTilemap.IS_CHANGED("tilemap/ts", "output", True), want)

    def test_unreadable_pair_is_nan_so_it_reruns(self):
        self.put(self.out, "tilemap/ts")
        for path, source in (("tilemap/none", "output"), ("../ts", "output"), ("ts", "bogus"), ("ts", None)):
            with self.subTest(path=path, source=source):
                got = L.CCN_LoadTilemap.IS_CHANGED(path, source, True)
                self.assertIsInstance(got, float)
                self.assertTrue(math.isnan(got))
        with mock.patch.dict(sys.modules, {"folder_paths": None}):
            self.assertTrue(math.isnan(L.CCN_LoadTilemap.IS_CHANGED("ts", "output", True)))


if __name__ == "__main__":
    unittest.main()
