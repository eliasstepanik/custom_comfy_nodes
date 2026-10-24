"""Tests for ``nodes/tilemap_scene.py`` (D18a, ``CCN_TilemapScene``). Stdlib unittest; real torch tensors; the LLM
call (``llm_client.chat``) is replaced by a fake, nothing touches the network or ComfyUI.

The module is imported through ``load_pkg`` of ``tests/test_node_load.py`` under its own synthetic package name.

Residual gaps: scene geometry, wang corner rules and layout validation are covered by ``test_scene_layout.py`` and
``test_scene_render.py``; here only the node contract (widgets, tensor adaption, source priority, log text) is
asserted. The catalogue of a tileset with wang tiles is tested through ``_catalogue`` with hand-made metadata, not
through a rendered wang atlas. Real ``llm_client.chat`` transport (HTTP, retries) is covered by the llm_client tests.
"""

import copy
import importlib
import json
import unittest
from unittest import mock

import numpy as np

from tests.test_node_load import load_pkg, torch, unload_pkg
from tests.test_tileset_io import PAL, edited, make_fixture
from tilemap_core import llm_client, scene_layout, tileset_io

PKG = "_ccn_t13_scene_pkg"
S = None


def setUpModule():
    global S
    if torch is None:
        raise unittest.SkipTest("torch not installed")
    S = load_pkg(PKG, "tilemap_scene")


def tearDownModule():
    unload_pkg(PKG)


def tensor(rgba):
    return torch.from_numpy(np.ascontiguousarray(rgba).astype(np.float32) / np.float32(255.0))[None]


class SceneCase(unittest.TestCase):
    def setUp(self):
        self.fx = make_fixture()
        self.ts = tileset_io.load_tileset(self.fx.png, self.fx.text, strict=True)
        self.atlas, self.node = tensor(self.ts.rgba), S.CCN_TilemapScene()

    def run_node(self, tiles=None, metadata=None, seed=3, source="auto", upscale=2, scene_json=None):
        return self.node.render([self.atlas] if tiles is None else tiles, [self.fx.text if metadata is None else metadata],
                                [seed], [source], [upscale], None if scene_json is None else [scene_json])

    def fails(self, message, **kw):
        with self.assertRaises(ValueError) as cm:
            self.run_node(**kw)
        self.assertIs(type(cm.exception), ValueError)
        self.assertEqual(str(cm.exception), message)

    def cell_tensors(self):
        return [tensor(c) for c in self.ts.cells.values()]


class TestContract(unittest.TestCase):
    def test_node_surface_exact(self):
        cls = S.CCN_TilemapScene
        self.assertEqual(cls.INPUT_TYPES(), {
            "required": {"tiles": ("IMAGE",), "metadata": ("STRING", {"forceInput": True}),
                         "seed": ("INT", {"default": 0, "min": 0, "max": 2**64 - 1, "step": 1}),
                         "layout_source": (["auto", "llm"], {"default": "auto"}),
                         "upscale": ("INT", {"default": 4, "min": 1, "max": 16, "step": 1})},
            "optional": {"scene_json": ("STRING", {"multiline": True, "default": ""})}})
        self.assertEqual((cls.RETURN_TYPES, cls.RETURN_NAMES), (("IMAGE", "STRING", "STRING"), ("scene", "scene_json", "layout_log")))
        self.assertEqual((cls.INPUT_IS_LIST, cls.FUNCTION, cls.CATEGORY), (True, "render", "custom_comfy_nodes/tilemap"))
        self.assertEqual(S.NODE_CLASS_MAPPINGS, {"CCN_TilemapScene": cls})
        self.assertEqual(S.NODE_DISPLAY_NAME_MAPPINGS, {"CCN_TilemapScene": "Tilemap Scene Preview"})
        self.assertEqual((S.LAYOUT_SOURCES, S.SEED_MAX, S.SCENE_JSON_MAX, S.LOG_LINES_MAX), (("auto", "llm"), 2**64 - 1, 100_000, 20))
        self.assertEqual((S.LLM_MODEL, S.LLM_MAX_TOKENS, S.LLM_TIMEOUT_S, S.UPSCALE_DEFAULT), ("claude-sonnet-5", 8000, 120, 4))
        self.assertIn(S.LLM_MODEL, llm_client.MODELS)

    def test_one_unwraps_single_element_lists_only(self):
        self.assertEqual((S._one([5], "x"), S._one((6,), "x"), S._one(7, "x"), S._one("s", "x")), (5, 6, 7, "s"))
        for bad in ([], [1, 2], (1, 2, 3)):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, rf"^x must be a single value, got {len(bad)}$"):
                S._one(bad, "x")

    def test_int_in_is_strict(self):
        self.assertEqual((S._int_in(0, "n", 0, 5), S._int_in(5, "n", 0, 5)), (0, 5))
        for bad in (-1, 6, True, False, 1.0, "1", None):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, r"^n must be an integer from 0 to 5$"):
                S._int_in(bad, "n", 0, 5)

    def test_catalogue_counts_variants_and_dedupes_wang_pairs(self):
        meta = {"tiles": [
            {"id": "g", "category": "ground", "footprint": [1, 1]}, {"id": "g", "category": "ground", "footprint": [1, 1]},
            {"id": "d", "category": "ground", "footprint": [1, 1]},
            {"id": "w0", "wang": {"a": "g", "b": "d", "mask": 1}}, {"id": "w1", "wang": {"a": "g", "b": "d", "mask": 2}},
            {"id": "w2", "wang": {"a": "d", "b": "g", "mask": 3}}]}
        self.assertEqual(S._catalogue(meta), {"tiles": [
            {"id": "g", "category": "ground", "variants": 2, "footprint": [1, 1]},
            {"id": "d", "category": "ground", "variants": 1, "footprint": [1, 1]}], "transitions": [["g", "d"], ["d", "g"]]})
        self.assertEqual(S._catalogue({}), {"tiles": [], "transitions": []})
        for bad in ([5], [{"id": 3}], [{"category": "x"}], ["g"]):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, r"^metadata tiles\[0\] must be an object with a string id$"):
                S._catalogue({"tiles": bad})

    def test_tensor_io_conversions(self):
        tio = importlib.import_module(f"{PKG}.nodes._tensor_io")
        t = torch.tensor([[[[0.5, 1.5, -0.2, float("nan")]], [[float("inf"), -float("inf"), 0.0, 1.0]]]] * 2)  # [2,2,1,4]
        self.assertEqual([a.tolist() for a in tio.image_list_to_np(t, channels=4)], [[[[128, 255, 0, 0]], [[255, 0, 0, 255]]]] * 2)
        self.assertEqual(len(tio.image_list_to_np(t, channels=3, first_only=True)), 1)
        self.assertEqual(len(tio.image_list_to_np([t, t], first_only=True)), 2)
        with self.assertRaisesRegex(ValueError, r"^array\[1\] shape \(1, 1, 3\) differs from array\[0\] shape \(2, 1, 3\)$"):
            tio.np_to_image([np.zeros((2, 1, 3), np.uint8), np.zeros((1, 1, 3), np.uint8)])

    def test_log_lines_unplaced_problem_and_warning_cap(self):
        self.assertEqual(S._log("auto", "", [], []), "layout_source: auto")
        self.assertEqual(S._log("auto_fallback", "bad", ["a", "b"], ["w"]),
                         "layout_source: auto_fallback\nlayout problem: bad\nunplaced: a, b\nwarning: w")
        twenty = [f"w{i}" for i in range(20)]
        self.assertEqual(S._log("auto", "", [], twenty).count("\n"), 20)
        self.assertNotIn("more warnings", S._log("auto", "", [], twenty))
        lines = S._log("auto", "", [], twenty + ["w20", "w21"]).split("\n")
        self.assertEqual((len(lines), lines[-2], lines[-1]), (22, "warning: w19", "... 2 more warnings"))


class TestRender(SceneCase):
    def test_atlas_happy_path_tensor_and_text_contract(self):
        before = self.atlas.clone()
        image, text, log = self.run_node()
        scene = json.loads(text)
        self.assertEqual((image.dtype, tuple(image.shape)), (torch.float32, (1, scene["height"] * 8 * 2, scene["width"] * 8 * 2, 3)))
        self.assertTrue(bool(((image >= 0) & (image <= 1)).all()))
        self.assertEqual((scene["format"], scene["seed"], log), (scene_layout.FORMAT, 3, "layout_source: auto"))
        self.assertEqual(text, json.dumps(scene, sort_keys=True, separators=(",", ":")))
        self.assertTrue(torch.equal(self.atlas, before))  # input not mutated
        self.assertEqual(self.run_node()[1:], (text, log))
        self.assertTrue(torch.equal(self.run_node()[0], image))  # deterministic

    def test_tile_list_input_matches_atlas_input(self):
        atlas_out, list_out = self.run_node(), self.run_node(tiles=self.cell_tensors())
        self.assertTrue(torch.equal(atlas_out[0], list_out[0]))
        self.assertEqual(atlas_out[1:], list_out[1:])

    def test_tile_sizes_8_to_64_palette_uniform_and_binary_alpha_source(self):
        for n in (8, 16, 24, 32, 48, 64):
            with self.subTest(n=n):
                fx = make_fixture(n=n)
                node_in = tensor(tileset_io.load_tileset(fx.png, fx.text, strict=True).rgba)
                image, text, _ = S.CCN_TilemapScene().render([node_in], [fx.text], [1], ["auto"], [1])
                scene = json.loads(text)
                self.assertEqual(tuple(image.shape), (1, scene["height"] * n, scene["width"] * n, 3))
                pix = np.rint(image[0].numpy() * 255).astype(np.uint8).reshape(-1, 3)
                self.assertLessEqual({tuple(p) for p in pix.tolist()}, set(PAL) | {(128, 128, 128)})

    def test_upscale_bounds_and_nearest_scaling(self):
        one, sixteen = self.run_node(upscale=1)[0], self.run_node(upscale=16)[0]
        self.assertEqual(tuple(sixteen.shape), (1, one.shape[1] * 16, one.shape[2] * 16, 3))
        self.assertTrue(torch.equal(sixteen, one.repeat_interleave(16, 1).repeat_interleave(16, 2)))

    def test_seed_bounds_and_effect(self):
        for seed in (0, S.SEED_MAX):
            with self.subTest(seed=seed):
                self.assertEqual(json.loads(self.run_node(seed=seed)[1])["seed"], seed)
        self.assertNotEqual(self.run_node(seed=1)[1], self.run_node(seed=2)[1])

    def test_given_scene_json_wins_and_is_normalised(self):
        auto = json.loads(self.run_node()[1])
        auto["seed"] = 77
        _, text, log = self.run_node(source="llm", scene_json="  " + json.dumps(auto) + "\n")
        self.assertEqual((json.loads(text)["seed"], log), (77, "layout_source: given"))
        loose = {"format": scene_layout.FORMAT, "width": 4, "height": 4, "terrain": [["g0"] * 5] * 5, "background": "#ABCDEF"}
        self.assertEqual(json.loads(self.run_node(scene_json=json.dumps(loose))[1]),  # validated copy: defaults added, colour lowered
                         dict(loose, seed=0, objects=[], overlays=[], background="#abcdef"))
        self.assertEqual(self.run_node(scene_json=" \n ")[2], "layout_source: auto")
        self.assertEqual(self.run_node(scene_json=" " * S.SCENE_JSON_MAX)[2], "layout_source: auto")

    def test_given_scene_json_errors_are_value_errors(self):
        for bad in ("{", "not json", "[" * 49_000 + "]" * 49_000):
            with self.subTest(bad=bad[:12]):
                self.fails("scene_json is not valid JSON", scene_json=bad)
        for obj in ([], {"format": "wrong"}, {"format": scene_layout.FORMAT}):
            with self.subTest(obj=obj), self.assertRaises(ValueError) as cm:
                self.run_node(scene_json=json.dumps(obj))
            self.assertIs(type(cm.exception), ValueError)

    def test_off_palette_atlas_renders_with_warning_in_log(self):
        rgba = self.ts.rgba.copy()
        rgba[0, 0, :3] = (1, 2, 3)
        rgba[0, 0, 3] = 255
        log = self.run_node(tiles=[tensor(rgba)])[2]
        self.assertIn("warning: 1 opaque pixels use colours outside the palette", log)


class TestLlmLayout(SceneCase):
    def run_llm(self, **chat_kw):
        with mock.patch.object(S.llm_client, "chat", **chat_kw) as chat:
            return self.run_node(source="llm"), chat

    def test_valid_reply_is_used_and_request_shape(self):
        reply = json.dumps(dict(json.loads(self.run_node()[1]), seed=5))
        (_, text, log), chat = self.run_llm(return_value=reply)
        catalogue = S._catalogue(self.ts.meta)
        chat.assert_called_once_with(scene_layout.build_scene_messages(catalogue, "top-down"), "claude-sonnet-5", max_tokens=8000, timeout=120)
        self.assertEqual((json.loads(text)["seed"], log), (5, "layout_source: llm"))

    def test_proxy_error_falls_back_to_auto_and_is_logged(self):
        auto = self.run_node()
        (image, text, log), _ = self.run_llm(side_effect=llm_client.LLMProxyError("boom \x00 key=" + "x" * 400))
        self.assertTrue(torch.equal(image, auto[0]))
        self.assertEqual(text, auto[1])
        head, problem = log.split("\n")
        self.assertEqual(head, "layout_source: auto_fallback")
        self.assertTrue(problem.startswith("layout problem: boom") and "\x00" not in problem and len(problem) <= len("layout problem: ") + 300)

    def test_unusable_reply_falls_back_with_problem(self):
        for reply in ("no json here", '{"format": "bad"}'):
            with self.subTest(reply=reply):
                (_, text, log), _ = self.run_llm(return_value=reply)
                self.assertEqual(text, self.run_node()[1])
                self.assertRegex(log, r"^layout_source: auto_fallback\nlayout problem: .+")

    def test_other_exceptions_propagate_and_given_json_skips_the_call(self):
        with mock.patch.object(S.llm_client, "chat", side_effect=KeyError("k")), self.assertRaises(KeyError):
            self.run_node(source="llm")
        with mock.patch.object(S.llm_client, "chat") as chat:
            self.run_node(source="llm", scene_json=self.run_node()[1])
            self.run_node(source="auto")
        chat.assert_not_called()


class TestErrors(SceneCase):
    def test_widget_validation_messages(self):
        seed_msg, up_msg = "seed must be an integer from 0 to 18446744073709551615", "upscale must be an integer from 1 to 16"
        for bad in (-1, 2**64, True, 1.0, "1", None):
            with self.subTest(seed=bad):
                self.fails(seed_msg, seed=bad)
        for bad in (0, 17, True, 2.0, "2"):
            with self.subTest(upscale=bad):
                self.fails(up_msg, upscale=bad)
        for bad in ("bogus", "", "AUTO", None, 1):
            with self.subTest(source=bad):
                self.fails("layout_source must be one of ['auto', 'llm']", source=bad)
        for bad in (5, b"{}", ["x"]):
            with self.subTest(scene_json=bad):
                self.fails("scene_json must be text of at most 100000 characters", scene_json=bad)
        self.fails("scene_json must be text of at most 100000 characters", scene_json="x" * 100_001)
        self.fails("metadata must be text", metadata=5)

    def test_list_inputs_must_hold_one_value_and_widgets_are_checked_first(self):
        with self.assertRaisesRegex(ValueError, r"^seed must be a single value, got 2$"):
            self.node.render([self.atlas], [self.fx.text], [1, 2], ["auto"], [2])
        with self.assertRaisesRegex(ValueError, r"^layout_source must be a single value, got 0$"):
            self.node.render([self.atlas], [self.fx.text], [1], [], [2])
        self.fails("seed must be an integer from 0 to 18446744073709551615", tiles="not a tensor", seed=-1)

    def test_bad_tiles_and_metadata_are_value_errors(self):
        meta_no_tiles = edited(self.fx, lambda m: m.pop("tiles"))
        meta_bad_fp = edited(self.fx, lambda m: m["tiles"][0].update(footprint=[0, 1]))
        meta_bad_view = edited(self.fx, lambda m: m.update(view="sideways"))
        small = tensor(np.zeros((8, 8, 4), np.uint8))
        cases = {"not a tensor": dict(tiles=["x"]), "rgb tiles": dict(tiles=[torch.zeros(1, 16, 32, 3)]),
                 "bad json": dict(metadata="{"), "no tiles in list form": dict(tiles=self.cell_tensors(), metadata=meta_no_tiles),
                 "bad footprint": dict(tiles=self.cell_tensors(), metadata=meta_bad_fp),
                 "bad view": dict(tiles=self.cell_tensors(), metadata=meta_bad_view),
                 "one wrong-size tile": dict(tiles=[small]), "too few tiles": dict(tiles=self.cell_tensors()[:3]),
                 "swapped tiles": dict(tiles=self.cell_tensors()[::-1])}
        for name, kw in cases.items():
            with self.subTest(name), self.assertRaises(ValueError) as cm:
                self.run_node(**kw)
            self.assertIs(type(cm.exception), ValueError, name)

    def test_tile_list_count_and_shape_messages(self):
        count = "got 3 tile images for 4 metadata tiles; connect the atlas image, or one image per tile in metadata order"
        self.fails(count, tiles=self.cell_tensors()[:3])
        swapped = self.cell_tensors()
        swapped[1], swapped[2] = swapped[2], swapped[1]
        with self.assertRaisesRegex(ValueError, r"^tile image 1 for 'g1#0' must be 8x8 px RGBA \(duplicate keys are refused\)$"):
            self.run_node(tiles=swapped)
        dup = copy.deepcopy(self.fx.meta)
        dup["tiles"][1].update(id="g0", variant=0, key="g0#0")
        with self.assertRaisesRegex(ValueError, r"^tile image 1 for 'g0#0' must be 8x8 px RGBA \(duplicate keys are refused\)$"):
            S._list_cells([t[0].numpy() for t in map(lambda c: c * 255, self.cell_tensors())], dup)
        meta = copy.deepcopy(self.fx.meta)
        meta["tiles"][1].update(footprint=[1])
        with self.assertRaisesRegex(ValueError, r"^metadata tiles\[1\] needs an integer footprint \[w, h\] and the metadata a tile_size$"):
            self.run_node(tiles=self.cell_tensors(), metadata=json.dumps(meta))

    def test_bad_view_message_lists_known_views(self):
        meta = edited(self.fx, lambda m: m.update(view="sideways"))
        with self.assertRaisesRegex(ValueError, r"^metadata view must be one of \['.*top-down.*'\]$"):
            self.run_node(tiles=self.cell_tensors(), metadata=meta)


if __name__ == "__main__":
    unittest.main()
