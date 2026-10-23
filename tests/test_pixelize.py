"""Tests for tilemap_core.pixelize: exact sizes, palette-exactness, binary alpha, seamless, keying."""

import unittest

import numpy as np

from tests import _support as S
from tilemap_core import pixelize as P

SHAPES = ((1, 1), (2, 1), (1, 2), (2, 2), (4, 2))  # (cells_w, cells_h)
SOURCES = {
    "noise": S.noise_rgb(37, 53, 1), "wide": S.noise_rgb(20, 90, 2), "tall": S.noise_rgb(90, 20, 3),
    "flat": np.full((8, 8, 3), (10, 200, 30), np.uint8), "circle": S.circle_on_color(64),
    "gradient": S.gradient_rgb(50, 70), "one_px": S.noise_rgb(1, 1, 4), "tiny": S.noise_rgb(2, 3, 5),
}
MAGENTA, WHITE, GREEN, FG = (255, 0, 255), (255, 255, 255), (0, 255, 0), (200, 60, 60)


def grid_of(work):
    return np.ones((work.shape[0] // 8, work.shape[1] // 8), bool)


def pipeline(src, cw, ch, n, k=8, alpha=None):
    work = P.to_work(src, cw, ch, n)
    a = grid_of(work) if alpha is None else alpha
    pal = P.build_palette([work], [a], k)
    return work, a, pal, P.block_reduce(P.index_image(work, pal), a, pal)


def idx_2x2(*counts):
    """16x16 index image whose four 8x8 blocks hold ``counts[b][i]`` copies of index ``i`` (shuffled)."""
    rng = np.random.default_rng(0)
    blocks = [rng.permutation(np.repeat(np.arange(len(c)), c)) for c in counts]
    return np.array(blocks, np.int16).reshape(2, 2, 8, 8).transpose(0, 2, 1, 3).reshape(16, 16)


def aa_circle(size, bg, fg=FG):
    """Anti-aliased disc: 4x supersampled box average (radius 0.35*size)."""
    big = S.circle_on_color(size * 4, bg, fg).astype(np.float64)
    return np.rint(big.reshape(size, 4, size, 4, 3).mean(axis=(1, 3))).astype(np.uint8)


def outlined_rock(width, size=256, r=50, inner=235):
    """Light rock (grey ``inner``) with a dark outline ``width`` px thick on white."""
    yy, xx = np.mgrid[0:size, 0:size]
    d = np.hypot(yy - (size - 1) / 2, xx - (size - 1) / 2)
    img = np.full((size, size, 3), 255, np.uint8)
    img[d <= r] = 30
    img[d <= r - width] = inner
    return img


def centre_dist():
    """Distance (work px) of every block centre of the 32x32 grid from the 256 px work centre."""
    return np.hypot(*(np.mgrid[0:32, 0:32] * 8 + 3.5 - 127.5))


def near_frac(work, bg, tol=40):
    d = np.abs(work.astype(int) - np.array(bg)).max(axis=2).reshape(32, 8, 32, 8)
    return (d < tol).mean(axis=(1, 3))


class SizesTest(unittest.TestCase):
    def test_to_work_exact_shape_all_sizes_shapes_sources(self):
        self.assertEqual((P.WORK_SCALE, P.TILE_SIZES), (8, (8, 16, 24, 32, 48, 64)))
        for n in P.TILE_SIZES:
            for cw, ch in SHAPES:
                for name, src in SOURCES.items():
                    with self.subTest(n=n, cells=(cw, ch), src=name):
                        out = P.to_work(src, cw, ch, n)
                        self.assertEqual((out.shape, out.dtype), ((ch * n * 8, cw * n * 8, 3), np.uint8))

    def test_to_work_centre_crop(self):
        def bands(h, w, lo, hi, axis):
            img = np.zeros((h, w, 3), np.uint8)
            img[...] = (255, 0, 0)
            sl = (slice(None), slice(lo, hi)) if axis else (slice(lo, hi),)
            img[sl] = (0, 255, 0)
            return img

        cases = [  # (source, cells_w, cells_h, expected-green band)
            (bands(40, 80, 20, 60, 1), 1, 1, "wide square"), (bands(80, 40, 20, 60, 0), 1, 1, "tall square"),
            (bands(40, 81, 20, 60, 1), 1, 1, "odd width"), (bands(64, 64, 16, 48, 0), 2, 1, "2x1 from square"),
            (bands(64, 64, 16, 48, 1), 1, 2, "1x2 from square"),
        ]
        for src, cw, ch, label in cases:
            with self.subTest(label):
                out = P.to_work(src, cw, ch, 8)
                self.assertTrue((out == (0, 255, 0)).all(), label)


class PaletteTest(unittest.TestCase):
    def test_flat_colour_gives_single_colour(self):
        work = np.full((16, 16, 3), (12, 34, 56), np.uint8)
        self.assertEqual(P.build_palette([work], [grid_of(work)], 16).tolist(), [[12, 34, 56]])

    def test_excludes_transparent_pixels(self):
        work = np.zeros((16, 32, 3), np.uint8)
        work[:, :16] = (1, 2, 3)
        work[:8, 16:24] = (200, 0, 0)
        work[8:, 16:24] = (0, 200, 0)
        work[:, 24:] = (7, 7, 250)  # transparent colour, must not appear
        alpha = np.ones((2, 4), bool)
        alpha[:, 3] = False
        pal = P.build_palette([work], [alpha], 16)
        self.assertEqual({tuple(c) for c in pal}, {(1, 2, 3), (200, 0, 0), (0, 200, 0)})

    def test_size_cap_sorted_unique_dtype(self):
        works = [S.noise_rgb(64, 64, 1), S.noise_rgb(32, 96, 2)]
        alphas = [grid_of(works[0]), grid_of(works[1])]
        for k in (1, 2, 4, 16, 64, 256):
            with self.subTest(k=k):
                pal = P.build_palette(works, alphas, k)
                self.assertEqual((pal.dtype, pal.ndim, pal.shape[1]), (np.uint8, 2, 3))
                self.assertTrue(1 <= len(pal) <= k)
                self.assertTrue(np.array_equal(pal, np.unique(pal, axis=0)))

    def test_index_image_matches_brute_force_with_duplicates(self):
        rng = np.random.default_rng(7)
        pal = rng.integers(0, 256, (60, 3), dtype=np.uint8)
        pal = np.concatenate([pal, pal[:20], pal[:5]])  # duplicate colours: lowest index must win
        for work in (S.noise_rgb(40, 48, 8), rng.integers(100, 140, (260, 128, 3), dtype=np.uint8)):
            idx = P.index_image(work, pal)
            self.assertEqual((idx.shape, idx.dtype), (work.shape[:2], np.int16))
            p64 = pal.astype(np.int64)
            for y in range(work.shape[0]):
                d = ((work[y].astype(np.int64)[:, None, :] - p64[None]) ** 2).sum(axis=2)
                self.assertTrue(np.array_equal(idx[y], d.argmin(axis=1)), f"row {y}")

    def test_index_image_tie_lowest_index(self):
        work = np.zeros((1, 3, 3), np.uint8)
        work[0, :, 0] = (5, 15, 25)
        up = np.array([[0, 0, 0], [10, 0, 0], [20, 0, 0], [30, 0, 0]], np.uint8)
        self.assertEqual(P.index_image(work, up).tolist(), [[0, 1, 2]])
        self.assertEqual(P.index_image(work, up[::-1].copy()).tolist(), [[2, 1, 0]])

    def test_block_reduce_mode_and_tie_lowest_index(self):
        pal = np.array([[10, 0, 0], [0, 20, 0], [0, 0, 30]], np.uint8)
        out = P.block_reduce(idx_2x2([10, 34, 20], [32, 0, 32], [0, 32, 32], [21, 21, 22]), np.ones((2, 2), bool), pal)
        want = [[pal[1], pal[0]], [pal[1], pal[2]]]
        for r in range(2):
            for c in range(2):
                self.assertEqual(out[r, c].tolist(), want[r][c].tolist() + [255], (r, c))

    def test_pipeline_exact_shape_and_palette_exact_all_sizes(self):
        rng = np.random.default_rng(9)
        for n in P.TILE_SIZES:
            for cw, ch in SHAPES:
                alpha = rng.random((ch * n, cw * n)) < 0.7
                alpha[0, 0] = True
                for k in (3, 24):
                    with self.subTest(n=n, cells=(cw, ch), k=k):
                        _, _, pal, out = pipeline(SOURCES["gradient"], cw, ch, n, k, alpha)
                        self.assertEqual((out.shape, out.dtype), ((ch * n, cw * n, 4), np.uint8))
                        self.assertLessEqual(len(pal), k)
                        self.assertTrue(set(np.unique(out[..., 3])) <= {0, 255})
                        self.assertTrue(set(map(tuple, out[alpha][:, :3])) <= set(map(tuple, pal)))
                        self.assertTrue((out[alpha][:, 3] == 255).all() and not out[~alpha].any())


class SeamlessTest(unittest.TestCase):
    def test_blend_zero_is_independent_copy(self):
        w = S.noise_rgb(16, 24, 1)
        out = P.make_seamless(w, 0)
        self.assertTrue(np.array_equal(out, w) and out.dtype == np.uint8)
        self.assertIsNot(out, w)
        self.assertFalse(np.shares_memory(out, w))

    def test_seam_difference_shrinks_and_interior_kept(self):
        w = S.gradient_rgb(64, 64)
        before = (np.abs(w[:, 0].astype(int) - w[:, -1]).mean(), np.abs(w[0].astype(int) - w[-1]).mean())
        for blend in (0.1, 0.25, 0.5):
            with self.subTest(blend=blend):
                out = P.make_seamless(w, blend)
                self.assertEqual((out.shape, out.dtype), (w.shape, np.uint8))
                after = (np.abs(out[:, 0].astype(int) - out[:, -1]).mean(), np.abs(out[0].astype(int) - out[-1]).mean())
                self.assertLess(after[0] * 10, before[0])
                self.assertLess(after[1] * 10, before[1])
        centre = P.make_seamless(w, 0.25)[28:36, 28:36]
        self.assertTrue(np.array_equal(centre, w[28:36, 28:36]))

    def test_bad_blend_raises(self):
        w = S.noise_rgb(8, 8, 1)
        for bad in (True, False, float("nan"), float("inf"), -0.1, 0.51, 1, "0.2", None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                P.make_seamless(w, bad)
        for good in (0, 0.5, np.float32(0.25)):
            self.assertEqual(P.make_seamless(w, good).shape, w.shape)


class KeyAlphaTest(unittest.TestCase):
    def test_aa_circle_no_fringe_on_any_key_colour(self):
        cd = centre_dist()
        for bg in (MAGENTA, WHITE, GREEN):
            with self.subTest(bg=bg):
                work = aa_circle(256, bg)
                a = P.key_alpha(work, (32, 32))
                self.assertFalse(a[0].any() or a[-1].any() or a[:, 0].any() or a[:, -1].any())
                self.assertFalse((near_frac(work, bg)[a] >= 0.25).any(), "bg-coloured fringe block kept")
                self.assertTrue(a[cd <= 80].all(), "solid disc interior must stay opaque")
                self.assertFalse(a[cd > 104].any())

    def test_outlined_light_rock_survives(self):
        cd = centre_dist()
        true = cd <= 45  # blocks wholly inside the rock
        for width in (2, 4, 8, 16, 24):  # widths 2/4/8 were erased entirely by the old 25%-flood rule
            with self.subTest(outline_px=width):
                a = P.key_alpha(outlined_rock(width), (32, 32))
                self.assertGreaterEqual(int((a & true).sum()), int(0.8 * true.sum()))
                self.assertFalse((a & (cd > 60)).any())

    def test_ring_is_peeled_once_and_cannot_chain(self):
        half = np.full((8, 8, 3), 255, np.uint8)
        half[:, :4] = 0  # mean distance 127 > tol (never flooded), 50% near bg (>= 25%)
        work = np.full((128, 128, 3), 255, np.uint8)
        work[:80, 24:104] = np.tile(half, (10, 10, 1))
        want = np.zeros((16, 16), bool)
        want[1:9, 4:12] = True  # top row (border), left/right/bottom ring peeled; interior kept
        got = P.key_alpha(work, (16, 16))
        self.assertTrue(np.array_equal(got, want), got.astype(int))

    def test_flat_and_tolerance_extremes(self):
        self.assertFalse(P.key_alpha(np.full((64, 64, 3), (3, 4, 5), np.uint8), (8, 8)).any())
        self.assertTrue(P.key_alpha(outlined_rock(8), (32, 32), tol=0).all())

    def test_deterministic_no_mutation_and_int_variants(self):
        work = outlined_rock(8)
        keep, a, b = work.copy(), P.key_alpha(work, (32, 32)), P.key_alpha(work, [32, 32])
        c = P.key_alpha(work, (np.int64(32), np.int64(32)))
        self.assertTrue(np.array_equal(work, keep) and np.array_equal(a, b) and np.array_equal(a, c))
        self.assertEqual((a.dtype, a.shape), (np.bool_, (32, 32)))


class ValidationTest(unittest.TestCase):
    BAD_INTS = (16.0, np.float64(16), True, "16", None)

    def test_strict_ints(self):
        src = S.noise_rgb(16, 16, 1)
        work, a = S.noise_rgb(16, 16, 2), np.ones((2, 2), bool)
        calls = {
            "n": lambda v: P.to_work(src, 1, 1, v), "cells_w": lambda v: P.to_work(src, v, 1, 8),
            "cells_h": lambda v: P.to_work(src, 1, v, 8), "k": lambda v: P.build_palette([work], [a], v),
            "grid_rows": lambda v: P.key_alpha(work, (v, 2)), "grid_cols": lambda v: P.key_alpha(work, (2, v)),
        }
        good = {"n": 8, "cells_w": 1, "cells_h": 1, "k": 4, "grid_rows": 2, "grid_cols": 2}
        for name, call in calls.items():
            for bad in self.BAD_INTS:
                with self.subTest(name=name, bad=repr(bad)), self.assertRaises(ValueError):
                    call(bad)
            with self.subTest(name=name, good="np.int64"):
                call(np.int64(good[name]))

    def test_bad_inputs_raise_value_error(self):
        u8, f32, z = np.zeros((16, 16, 3), np.uint8), np.zeros((16, 16, 3), np.float32), np.zeros
        ok2, idx = np.ones((2, 2), bool), np.zeros((16, 16), np.int16)
        pal, a3, a8 = np.array([[0, 0, 0], [255, 255, 255]], np.uint8), np.ones((3, 3), bool), ok2.astype(np.uint8)
        tw, ms, ka, bp, ii, br = P.to_work, P.make_seamless, P.key_alpha, P.build_palette, P.index_image, P.block_reduce
        nan = float("nan")
        cases = [
            (tw, z((8, 8), np.uint8), 1, 1, 8), (tw, z((8, 8, 4), np.uint8), 1, 1, 8), (tw, f32, 1, 1, 8),
            (tw, z((0, 5, 3), np.uint8), 1, 1, 8), (tw, u8.tolist(), 1, 1, 8), (tw, u8, 1, 1, 7),
            (tw, u8, 1, 1, 0), (tw, u8, 1, 1, 128), (tw, u8, 0, 1, 8), (tw, u8, 1, 17, 8), (tw, u8, 1, -1, 8),
            (ms, z((8, 8), np.uint8)), (ms, f32), (ms, u8[:0]),
            (ka, f32, (2, 2)), (ka, u8[:15], (1, 2)), (ka, u8, (2, 3)), (ka, u8, 2), (ka, u8, (2, 2, 2)),
            (ka, u8, (0, 2)), (ka, u8, (2, 2), -1), (ka, u8, (2, 2), 256), (ka, u8, (2, 2), nan),
            (ka, u8, (2, 2), True),
            (bp, [u8], [ok2], 0), (bp, [u8], [ok2], 257), (bp, [u8, u8], [ok2], 4), (bp, [], [], 4),
            (bp, u8, [ok2], 4), (bp, [u8], [a3], 4), (bp, [u8], [a8], 4),
            (bp, [u8], [np.zeros((2, 2), bool)], 4), (bp, [u8[:15]], [ok2], 4),
            (ii, u8, pal[0]), (ii, u8, pal.astype(int)), (ii, u8, pal[:0]), (ii, u8, z((257, 3), np.uint8)),
            (ii, f32, pal), (ii, u8, z((2, 4), np.uint8)),
            (br, idx.astype(float), ok2, pal), (br, idx[0], ok2, pal), (br, idx[:12], ok2, pal),
            (br, idx[:0], ok2[:0], pal), (br, idx, a3, pal), (br, idx, a8, pal), (br, idx + 2, ok2, pal),
            (br, idx - 1, ok2, pal), (br, idx, ok2, pal[:0]),
        ]
        self.assertGreaterEqual(len(cases), 38)
        for i, (fn, *args) in enumerate(cases):
            with self.subTest(case=i, fn=fn.__name__), self.assertRaises(ValueError):
                fn(*args)

    def test_all_functions_deterministic_and_do_not_mutate_inputs(self):
        work, alpha = S.noise_rgb(64, 64, 4), np.random.default_rng(1).random((8, 8)) < 0.6
        pal = P.build_palette([work], [alpha], 8)
        idx = P.index_image(work, pal)
        runs = {
            "to_work": lambda: P.to_work(work, 2, 1, 8), "seamless": lambda: P.make_seamless(work, 0.2),
            "key": lambda: P.key_alpha(work, (8, 8)), "palette": lambda: P.build_palette([work], [alpha], 8),
            "index": lambda: P.index_image(work, pal), "reduce": lambda: P.block_reduce(idx, alpha, pal),
        }
        args = [work, alpha, pal, idx]
        before = [x.copy() for x in args]
        for name, run in runs.items():
            with self.subTest(name):
                first = run()
                self.assertTrue(all(np.array_equal(x, y) for x, y in zip(args, before)), "input mutated")
                self.assertTrue(np.array_equal(first, run()))


if __name__ == "__main__":
    unittest.main()
