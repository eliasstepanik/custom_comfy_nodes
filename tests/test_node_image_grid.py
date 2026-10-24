"""Tests for ``nodes/image_grid.py`` (D18b, ``CCN_ImageGridLabeled`` / class ``ImageGridLabeled``). Stdlib unittest; real
torch tensors; no ComfyUI. The module is imported through ``load_pkg`` of ``tests/test_node_load.py`` (synthetic parent
package), and the root ``__init__.py`` is executed once to prove the pack registers the node.

Covers: INPUT_TYPES / RETURN_TYPES / FUNCTION / CATEGORY, registration and display name, output tensor type / shape / dtype /
range, layout arithmetic (columns, padding, rows, background colours, cell positions), RGBA -> RGB, clamping of out-of-range
floats, rounding to 8 bit, index labels, widget boundary values and error cases, the delegation to
``scene_render.label_grid`` and input immutability.

Residual gaps: pixel shapes of the digits depend on PIL's default font (only "labelled differs from plain, only in the
top-left corner of each cell" is asserted); ``label_grid`` itself is covered by ``tests/test_scene_render.py``; a 1-channel,
2-channel or non-4D IMAGE tensor is outside the ComfyUI IMAGE contract and only its current error type is pinned.
"""

import importlib.util
import contextlib
import io
import sys
import unittest
from unittest import mock

import numpy as np

from tests.test_node_load import ROOT, load_pkg, torch, unload_pkg

PKG = "_ccn_t15_grid_pkg"
PACK = "_ccn_t15_grid_pack"
G = None


def setUpModule():
    global G
    if torch is None:
        raise unittest.SkipTest("torch not installed")
    G = load_pkg(PKG, "image_grid")


def tearDownModule():
    unload_pkg(PKG)
    unload_pkg(PACK)


def batch(n, h=6, w=5, c=3, seed=0):
    """Random float32 IMAGE batch ``[n, h, w, c]`` of values k/255 (exact 8-bit levels)."""
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.integers(0, 256, (n, h, w, c)).astype(np.float32) / 255.0)


def u8(t):
    return (t[..., :3] * 255).round().byte().numpy()


def grid_of(images, columns=2, padding=3, background="black", show_index=False):
    (out,) = G.ImageGridLabeled().make_grid(images, columns, padding, background, show_index)
    return out


class ContractTests(unittest.TestCase):
    def test_class_attributes(self):
        cls = G.ImageGridLabeled
        self.assertEqual((cls.RETURN_TYPES, cls.RETURN_NAMES), (("IMAGE",), ("grid",)))
        self.assertEqual((cls.FUNCTION, cls.CATEGORY), ("make_grid", "custom_comfy_nodes/image"))
        self.assertTrue(callable(getattr(cls(), cls.FUNCTION)))

    def test_input_types(self):
        spec = G.ImageGridLabeled.INPUT_TYPES()
        self.assertEqual(set(spec), {"required"})
        self.assertEqual(spec["required"], {
            "images": ("IMAGE",),
            "columns": ("INT", {"default": 2, "min": 1, "max": 16, "step": 1}),
            "padding": ("INT", {"default": 8, "min": 0, "max": 256, "step": 1}),
            "background": (["black", "white", "gray"], {"default": "black"}),
            "show_index": ("BOOLEAN", {"default": True}),
        })
        self.assertEqual(list(spec["required"]), ["images", "columns", "padding", "background", "show_index"])

    def test_defaults_work_with_the_node(self):
        req = G.ImageGridLabeled.INPUT_TYPES()["required"]
        args = [req[k][1]["default"] for k in ("columns", "padding", "background", "show_index")]
        self.assertEqual(tuple(grid_of(batch(3), *args).shape), (1, 2 * 6 + 3 * 8, 2 * 5 + 3 * 8, 3))

    def test_module_mappings(self):
        self.assertEqual(G.NODE_CLASS_MAPPINGS, {"CCN_ImageGridLabeled": G.ImageGridLabeled})
        self.assertEqual(G.NODE_DISPLAY_NAME_MAPPINGS, {"CCN_ImageGridLabeled": "Image Grid (Labeled)"})

    def test_pack_registers_the_node(self):
        load_pkg(PACK, "image_grid")
        spec = importlib.util.spec_from_file_location(PACK, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
        pack = importlib.util.module_from_spec(spec)
        sys.modules[PACK] = pack
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            spec.loader.exec_module(pack)
        self.assertNotIn("image_grid", out.getvalue())
        cls = pack.NODE_CLASS_MAPPINGS["CCN_ImageGridLabeled"]
        self.assertEqual((cls.__name__, cls.__module__), ("ImageGridLabeled", f"{PACK}.nodes.image_grid"))
        self.assertEqual(pack.NODE_DISPLAY_NAME_MAPPINGS["CCN_ImageGridLabeled"], "Image Grid (Labeled)")


class OutputTests(unittest.TestCase):
    def test_returns_one_tuple_with_float32_batch_of_one(self):
        res = G.ImageGridLabeled().make_grid(batch(4), 2, 3, "black", False)
        self.assertIs(type(res), tuple)
        self.assertEqual(len(res), 1)
        out = res[0]
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual((out.dtype, out.ndim, out.shape[0], out.shape[3]), (torch.float32, 4, 1, 3))
        self.assertTrue(bool(((out >= 0) & (out <= 1)).all()))

    def test_size_formula(self):
        for n, cols, pad in ((1, 1, 0), (1, 16, 0), (4, 2, 3), (5, 2, 3), (5, 3, 0), (7, 16, 1), (7, 1, 256), (16, 4, 8), (17, 4, 8)):
            with self.subTest(n=n, cols=cols, pad=pad):
                rows = -(-n // cols)
                out = grid_of(batch(n, 6, 5), cols, pad)
                self.assertEqual(tuple(out.shape), (1, rows * 6 + (rows + 1) * pad, cols * 5 + (cols + 1) * pad, 3))

    def test_cells_are_pasted_at_the_documented_positions(self):
        imgs, pad, cols = batch(5), 3, 2
        out = u8(grid_of(imgs, cols, pad)[0])
        src = u8(imgs)
        for i in range(5):
            x, y = pad + (i % cols) * (5 + pad), pad + (i // cols) * (6 + pad)
            self.assertTrue(np.array_equal(out[y:y + 6, x:x + 5], src[i]), i)

    def test_background_colours_fill_padding_and_empty_slots(self):
        for bg, rgb in (("black", 0), ("white", 255), ("gray", 128)):
            imgs = batch(3)  # 2 columns: slot 4 is empty
            out = u8(grid_of(imgs, 2, 4, bg)[0])
            mask = np.ones(out.shape[:2], bool)
            for i in range(3):
                x, y = 4 + (i % 2) * 9, 4 + (i // 2) * 10
                mask[y:y + 6, x:x + 5] = False
            self.assertTrue((out[mask] == rgb).all(), bg)
            self.assertTrue(mask.any())

    def test_roundtrip_is_exact_for_8_bit_levels(self):
        imgs = batch(2, 4, 4)
        out = grid_of(imgs, 2, 0)[0]
        self.assertTrue(torch.allclose(out[:, 0:4], imgs[0], atol=1e-7))
        self.assertTrue(torch.allclose(out[:, 4:8], imgs[1], atol=1e-7))

    def test_single_image_no_padding_is_the_image(self):
        img = batch(1)
        self.assertTrue(torch.equal(grid_of(img, 1, 0)[0], img[0]))

    def test_rgba_input_is_reduced_to_rgb(self):
        rgb = batch(2, seed=3)
        rgba = torch.cat([rgb, torch.full((2, 6, 5, 1), 0.2)], dim=-1)
        self.assertTrue(torch.equal(grid_of(rgba, 2, 1), grid_of(rgb, 2, 1)))
        self.assertEqual(grid_of(rgba, 2, 1).shape[-1], 3)

    def test_values_are_clamped_and_rounded_to_8_bit(self):
        img = torch.tensor([[[[-0.5, 1.5, 0.5], [2.0, -3.0, 0.0]]]])  # 1x1x2x3
        out = grid_of(img, 1, 0)[0, 0]
        np.testing.assert_allclose(out.numpy(), [[0.0, 1.0, 128 / 255], [1.0, 0.0, 0.0]], atol=1e-7)
        near = torch.full((1, 1, 1, 3), 100.4 / 255)
        self.assertEqual(round(float(grid_of(near, 1, 0)[0, 0, 0, 0]) * 255), 100)
        near = torch.full((1, 1, 1, 3), 100.6 / 255)
        self.assertEqual(round(float(grid_of(near, 1, 0)[0, 0, 0, 0]) * 255), 101)

    def test_input_tensor_is_not_modified(self):
        imgs = batch(3)
        copy = imgs.clone()
        grid_of(imgs, 2, 2, "white", True)
        self.assertTrue(torch.equal(imgs, copy))

    def test_deterministic(self):
        imgs = batch(6)
        self.assertTrue(torch.equal(grid_of(imgs, 3, 2, "gray", True), grid_of(imgs, 3, 2, "gray", True)))

    def test_one_pixel_cells(self):
        out = grid_of(batch(4, 1, 1), 4, 0)
        self.assertEqual(tuple(out.shape), (1, 1, 4, 3))

    def test_non_square_and_larger_images(self):
        out = grid_of(batch(3, 40, 7), 3, 2)
        self.assertEqual(tuple(out.shape), (1, 44, 3 * 7 + 4 * 2, 3))


class LabelTests(unittest.TestCase):
    def test_show_index_changes_only_the_top_left_of_each_cell(self):
        imgs = torch.full((3, 24, 20, 3), 0.5)
        plain, labelled = u8(grid_of(imgs, 2, 4)[0]), u8(grid_of(imgs, 2, 4, show_index=True)[0])
        self.assertFalse(np.array_equal(plain, labelled))
        for i in range(3):
            x, y = 4 + (i % 2) * 24, 4 + (i // 2) * 28
            self.assertFalse(np.array_equal(plain[y:y + 24, x:x + 20], labelled[y:y + 24, x:x + 20]), i)
            self.assertTrue(np.array_equal(plain[y + 16:y + 24, x:x + 20], labelled[y + 16:y + 24, x:x + 20]), i)
        diff = (plain != labelled).any(axis=2)
        mask = np.zeros(diff.shape, bool)
        for i in range(3):
            x, y = 4 + (i % 2) * 24, 4 + (i // 2) * 28
            mask[y:y + 16, x:x + 16] = True
        self.assertFalse(diff[~mask].any())  # padding and the rest of the cells untouched

    def test_labels_differ_per_index(self):
        imgs = torch.full((2, 24, 20, 3), 0.5)
        out = u8(grid_of(imgs, 1, 0, show_index=True)[0])
        self.assertFalse(np.array_equal(out[:24], out[24:]))  # "1" vs "2"

    def test_label_is_white_on_black_box(self):
        out = u8(grid_of(torch.full((1, 24, 20, 3), 0.5), 1, 0, show_index=True)[0])
        self.assertTrue((out[0:2, 0:2] == 0).all())
        glyph = out[:12, :12]
        self.assertTrue((glyph > 0).any() and (glyph[..., 0] > 128).any())  # light (anti-aliased) glyph pixels in the black box
        self.assertTrue((out[:12, 12:16] == 128).all())  # the 50 % grey cell is visible right of the box

    def test_show_index_false_is_pure_layout(self):
        imgs = batch(2, 24, 20)
        out = u8(grid_of(imgs, 2, 0)[0])
        self.assertTrue(np.array_equal(out[:, :20], u8(imgs)[0]))


class ErrorTests(unittest.TestCase):
    def fails(self, regex, images=None, columns=2, padding=3, background="black", show_index=False, exc=ValueError):
        with self.assertRaisesRegex(exc, regex):
            G.ImageGridLabeled().make_grid(batch(3) if images is None else images, columns, padding, background, show_index)

    def test_widget_boundaries_accepted(self):
        self.assertEqual(grid_of(batch(1), 16, 0).shape[2], 16 * 5)
        self.assertEqual(grid_of(batch(1), 1, 256).shape[1:3], (6 + 512, 5 + 512))
        for bg in ("black", "white", "gray"):
            grid_of(batch(1), 1, 0, bg)

    def test_columns_out_of_range(self):
        for bad in (0, -1, 17, 100, True, 2.0, "2", None):
            with self.subTest(columns=bad):
                self.fails(r"columns must be", columns=bad)
        self.fails(r"columns must be in 1\.\.16, got 17", columns=17)

    def test_padding_out_of_range(self):
        for bad in (-1, 257, True, 1.5, "3", None):
            with self.subTest(padding=bad):
                self.fails(r"padding must be", padding=bad)
        self.fails(r"padding must be in 0\.\.256, got 257", padding=257)

    def test_background_and_show_index(self):
        for bad in ("red", "", "BLACK", None, 0):
            with self.subTest(background=bad):
                self.fails("background must be one of", background=bad)
        for bad in (1, 0, None, "True", np.True_):
            with self.subTest(show_index=bad):
                self.fails("show_index a bool", show_index=bad)

    def test_empty_batch(self):
        self.fails("images must be a non-empty", images=torch.zeros((0, 6, 5, 3)))

    def test_wrong_channel_count(self):
        self.fails(r"images\[0\] must be a uint8 array of shape", images=torch.zeros((2, 6, 5, 2)))

    def test_wrong_rank_is_a_value_error_not_a_crash(self):
        self.fails(r"must be a uint8 array|images must be", images=torch.zeros((6, 5, 3)), exc=(ValueError, IndexError))

    def test_error_raised_before_output(self):
        with mock.patch.object(G, "label_grid", wraps=G.label_grid) as spy:
            self.fails("columns must be", columns=0)
        self.assertEqual(spy.call_count, 1)  # the node delegates all argument checks to label_grid


class DelegationTests(unittest.TestCase):
    def test_label_grid_receives_uint8_rgb_batch_and_the_widgets(self):
        imgs = batch(3, seed=9)
        with mock.patch.object(G, "label_grid", return_value=np.zeros((7, 8, 3), np.uint8)) as spy:
            out = G.ImageGridLabeled().make_grid(imgs, 3, 5, "gray", True)[0]
        (arr, cols, pad, bg, show), kw = spy.call_args
        self.assertEqual((cols, pad, bg, show, kw), (3, 5, "gray", True, {}))
        self.assertEqual((arr.dtype, arr.shape), (np.uint8, (3, 6, 5, 3)))
        self.assertTrue(np.array_equal(arr, u8(imgs)))
        self.assertEqual(tuple(out.shape), (1, 7, 8, 3))
        self.assertTrue(bool((out == 0).all()))

    def test_rgba_is_cut_to_three_channels_before_label_grid(self):
        rgba = batch(2, c=4)
        with mock.patch.object(G, "label_grid", return_value=np.zeros((2, 2, 3), np.uint8)) as spy:
            G.ImageGridLabeled().make_grid(rgba, 1, 0, "black", False)
        self.assertEqual(spy.call_args.args[0].shape, (2, 6, 5, 3))

    def test_output_is_scaled_by_255(self):
        with mock.patch.object(G, "label_grid", return_value=np.full((1, 1, 3), 51, np.uint8)):
            out = G.ImageGridLabeled().make_grid(batch(1), 1, 0, "black", False)[0]
        np.testing.assert_allclose(out.flatten().numpy(), [51 / 255.0] * 3, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
