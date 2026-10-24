"""Tests for tilemap_core.palette_extend.extend_palette (D17b).

Covers: residual-only fit, the exact distance-24 boundary (Euclidean RGB, not
Chebyshev), nearest-not-farthest base colour, frequency weighting, alpha mask
(orientation + transparency), candidate filter (mean landing within/at 24 of a
base colour), k_extra 0..16 and base+k <= 256 limits, tile sides 8..64, base
preserved in place and never aliased, inputs never mutated, determinism,
idempotence, chunked distance helper (``_nearest_sq``) and the module-surface
facts (real home is ``palette_extend``, not ``pixelize``). Every ValueError path
lives in ``tests/test_palette_extend_more.py`` (shares the helpers defined here).

Residual untested gaps: >262144 residual pixels (median-cut stride subsample
lives in ``pixelize._median_cut``), exact Pillow median-cut box means (tests
only use cases whose result does not depend on Pillow's rounding), TypeError
from ``len()`` on a generator passed as works/alphas, and the second dedupe
condition "candidate repeats an earlier candidate" (unreachable: ``_median_cut``
already returns unique sorted rows).
"""

import unittest
from unittest import mock

import numpy as np

from tilemap_core import palette_extend as PE
from tilemap_core import pixelize
from tests import _support

BLACK, WHITE = (0, 0, 0), (255, 255, 255)
RED, GREEN, BLUE, YELLOW = (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)


def pal(*cols):
    return np.array(cols, np.uint8).reshape(-1, 3)


def flat(colour, cells_h=1, cells_w=1):
    """(work, alpha): one flat opaque colour, ``cells_h x cells_w`` 8-px cells."""
    work = np.empty((cells_h * 8, cells_w * 8, 3), np.uint8)
    work[:] = colour
    return work, np.ones((cells_h, cells_w), bool)


def strip(*runs):
    """(work, alpha) of 1-cell-high strip; each run is (colour, n_cells)."""
    cols = [c for c, n in runs for _ in range(n)]
    work = np.concatenate([flat(c)[0] for c in cols], axis=1)
    return work, np.ones((1, len(cols)), bool)


def ext(base, tiles, k):
    return PE.extend_palette(base, [t[0] for t in tiles], [t[1] for t in tiles], k)


def d2(colours, base):
    """Brute-force squared RGB distance to the nearest base colour (int64)."""
    c = np.asarray(colours, np.int64).reshape(-1, 3)
    b = np.asarray(base, np.int64).reshape(-1, 3)
    return ((c[:, None, :] - b[None, :, :]) ** 2).sum(axis=2).min(axis=1)


class ModuleSurface(unittest.TestCase):
    def test_constants_and_all(self):
        self.assertEqual(PE.__all__, ["extend_palette"])
        self.assertEqual((PE._EXTEND_MIN_DIST, PE._K_EXTRA_MAX, PE._PALETTE_MAX), (24, 16, 256))

    def test_home_is_palette_extend_not_pixelize(self):
        self.assertFalse(hasattr(pixelize, "extend_palette"))
        self.assertEqual(PE.extend_palette.__module__, "tilemap_core.palette_extend")
        with open(pixelize.__file__, encoding="utf-8") as f:
            self.assertNotIn("palette_extend", f.read())


class Behaviour(unittest.TestCase):
    def test_residual_colour_appended_after_base(self):
        out = ext(pal(BLACK, WHITE), [flat(RED)], 4)
        self.assertEqual(out.tolist(), [list(BLACK), list(WHITE), list(RED)])
        self.assertEqual((out.dtype, out.shape), (np.uint8, (3, 3)))

    def test_base_in_place_unsorted_and_duplicates(self):
        base = pal(WHITE, BLACK, BLACK)
        out = ext(base, [strip((RED, 1), (GREEN, 1), (BLUE, 1))], 3)
        self.assertEqual(out[:3].tolist(), base.tolist())
        self.assertEqual(out[3:].tolist(), [list(BLUE), list(GREEN), list(RED)])  # sorted rows

    def test_appended_rows_sorted_and_unique(self):
        out = ext(pal(BLACK), [strip((YELLOW, 2), (RED, 3), (BLUE, 1), (GREEN, 2))], 4)
        new = out[1:].tolist()
        self.assertEqual(new, sorted(new))
        self.assertEqual(len(new), 4)
        self.assertEqual(len({tuple(r) for r in new}), 4)

    def test_k_extra_caps_appended_count(self):
        tile = strip((RED, 1), (GREEN, 1), (BLUE, 1), (YELLOW, 1), ((0, 255, 255), 1))
        for k in (1, 2, 3):
            self.assertEqual(ext(pal(BLACK), [tile], k).shape[0], 1 + k)
        self.assertEqual(ext(pal(BLACK), [tile], 5).shape[0], 6)

    def test_k_extra_zero_returns_copy_even_with_residual(self):
        base = pal(BLACK)
        out = ext(base, [flat(RED)], 0)
        self.assertEqual(out.tolist(), base.tolist())
        self.assertFalse(np.shares_memory(out, base))

    def test_covered_tile_is_noop_copy(self):
        base = pal(BLACK, WHITE)
        out = ext(base, [strip(((5, 5, 5), 1), ((250, 250, 250), 1))], 8)
        self.assertEqual(out.tolist(), base.tolist())
        self.assertFalse(np.shares_memory(out, base))

    def test_distance_boundary_is_euclidean_and_strict(self):
        base = pal(BLACK)
        for colour, kept in (((24, 0, 0), False), ((25, 0, 0), True), ((23, 0, 0), False),
                             ((0, 0, 24), False), ((0, 25, 0), True),
                             ((16, 16, 8), False),   # 256+256+64 = 576 = 24**2 exactly
                             ((16, 16, 9), True),    # 593 > 576 (Chebyshev would call it covered)
                             ((15, 16, 8), False), ((255, 255, 255), True)):
            out = ext(base, [flat(colour)], 4)
            self.assertEqual(out.shape[0], 2 if kept else 1, colour)
            if kept:
                self.assertEqual(out[1].tolist(), list(colour))

    def test_nearest_base_colour_decides_not_farthest(self):
        base = pal(BLACK, WHITE)
        self.assertEqual(ext(base, [flat((245, 245, 245))], 4).shape[0], 2)  # ~17 from WHITE, ~424 from BLACK
        self.assertEqual(ext(base, [flat((10, 10, 10))], 4).shape[0], 2)
        self.assertEqual(ext(base, [flat((128, 128, 128))], 4).shape[0], 3)

    def test_no_wrap_with_uint8_subtraction(self):
        self.assertEqual(ext(pal(RED), [flat(BLACK)], 2)[1].tolist(), list(BLACK))  # |0-255| must not wrap to 1

    def test_covered_pixels_do_not_influence_fit(self):
        near = (10, 0, 0)  # covered by BLACK
        mixed = strip((RED, 1), (near, 6))
        self.assertEqual(ext(pal(BLACK), [mixed], 1).tolist(), [list(BLACK), list(RED)])

    def test_pixel_exactly_24_away_is_covered_and_excluded_from_fit(self):
        out = ext(pal(BLACK), [strip(((24, 0, 0), 1), ((200, 0, 0), 1))], 1)
        self.assertEqual(out.tolist(), [list(BLACK), [200, 0, 0]])  # a >= test would drag the fit to ~112

    def test_fit_is_frequency_weighted(self):
        out = ext(pal(BLACK), [strip(((200, 0, 0), 3), ((100, 0, 0), 1))], 1)
        self.assertEqual(out.shape[0], 2)
        self.assertTrue(abs(int(out[1, 0]) - 175) <= 2, out[1])  # weighted mean 175, unweighted 150
        self.assertEqual(out[1, 1:].tolist(), [0, 0])

    def test_transparent_cells_ignored_with_correct_orientation(self):
        work = np.concatenate([flat(RED)[0], flat(GREEN)[0]], axis=1)
        self.assertEqual(ext(pal(BLACK), [(work, np.array([[True, False]]))], 2)[1:].tolist(), [list(RED)])
        self.assertEqual(ext(pal(BLACK), [(work, np.array([[False, True]]))], 2)[1:].tolist(), [list(GREEN)])
        tall = np.concatenate([flat(RED)[0], flat(GREEN)[0]], axis=0)
        self.assertEqual(ext(pal(BLACK), [(tall, np.array([[False], [True]]))], 2)[1:].tolist(), [list(GREEN)])
        self.assertEqual(ext(pal(BLACK), [(tall, np.array([[True], [False]]))], 2)[1:].tolist(), [list(RED)])

    def test_all_transparent_returns_base_copy(self):
        work, _ = flat(RED, 2, 2)
        base = pal(BLACK)
        out = ext(base, [(work, np.zeros((2, 2), bool))], 3)
        self.assertEqual(out.tolist(), base.tolist())
        self.assertFalse(np.shares_memory(out, base))

    def test_multiple_tiles_of_different_sizes_and_tuple_inputs(self):
        a, b = flat(RED, 1, 1), flat(BLUE, 3, 2)
        out = PE.extend_palette(pal(BLACK), (a[0], b[0]), (a[1], b[1]), 2)
        self.assertEqual(out[1:].tolist(), [list(BLUE), list(RED)])

    def test_each_tile_side_8_to_64(self):
        for side in (8, 16, 24, 32, 48, 64):
            cells = side // 8
            out = ext(pal(BLACK), [flat((30, 200, 90), cells, cells)], 2)
            self.assertEqual(out.tolist(), [list(BLACK), [30, 200, 90]], side)

    def test_candidate_mean_at_exact_24_is_dropped_beyond_is_kept(self):
        base = pal((128, 128, 128))
        at24 = strip(((128, 103, 128), 1), ((128, 201, 128), 1))  # both >24 away; mean (128,152,128) is exactly 24
        self.assertEqual(ext(base, [at24], 1).tolist(), base.tolist())
        at25 = strip(((128, 103, 128), 1), ((128, 203, 128), 1))  # mean (128,153,128), 25 away
        self.assertEqual(ext(base, [at25], 1).tolist(), [[128, 128, 128], [128, 153, 128]])

    def test_candidate_equal_to_base_colour_is_dropped(self):
        base = pal((128, 128, 128))
        both = strip(((158, 128, 128), 1), ((98, 128, 128), 1))  # mean == base colour
        self.assertEqual(ext(base, [both], 1).tolist(), base.tolist())

    def test_noise_invariants_against_brute_force(self):
        base = pal(BLACK, WHITE, (128, 40, 200))
        work, alpha = _support.noise_rgb(64, 64, 3), np.random.default_rng(4).random((8, 8)) > 0.3
        out = ext(base, [(work, alpha)], 8)
        new = out[3:]
        self.assertEqual(out[:3].tolist(), base.tolist())
        self.assertTrue(0 < new.shape[0] <= 8 and out.dtype == np.uint8)
        self.assertTrue((d2(new, base) > 576).all())
        self.assertEqual(new.tolist(), sorted(new.tolist()))
        self.assertEqual(len({tuple(r) for r in new.tolist()}), new.shape[0])

    def test_deterministic_and_inputs_not_mutated(self):
        base = pal(BLACK, WHITE)
        work, alpha = _support.noise_rgb(32, 24, 9), np.random.default_rng(1).random((4, 3)) > 0.4
        snap = (base.copy(), work.copy(), alpha.copy())
        r1, r2 = ext(base, [(work, alpha)], 6), ext(base, [(work, alpha)], 6)
        self.assertTrue(np.array_equal(r1, r2))
        for now, before in zip((base, work, alpha), snap):
            self.assertTrue(np.array_equal(now, before))
        r1[:] = 7
        self.assertTrue(np.array_equal(base, snap[0]))

    def test_idempotent_for_flat_colours_and_successive_extends_stable(self):
        tile = strip((RED, 2), (GREEN, 1), (BLUE, 1))
        once = ext(pal(BLACK), [tile], 4)
        self.assertTrue(np.array_equal(ext(once, [tile], 4), once))
        step2 = ext(once, [flat(YELLOW)], 2)
        self.assertEqual(step2[: once.shape[0]].tolist(), once.tolist())
        self.assertEqual(step2.shape[0], once.shape[0] + 1)

    def test_big_base_reaches_256_limit(self):
        base = np.stack([np.arange(240, dtype=np.uint8), np.zeros(240, np.uint8), np.zeros(240, np.uint8)], axis=1)
        tile = _support.noise_rgb(64, 64, 2)
        out = ext(base, [(tile, np.ones((8, 8), bool))], 16)
        self.assertTrue(240 < out.shape[0] <= 256)
        self.assertEqual(out[:240].tolist(), base.tolist())

    def test_chunked_distance_matches_brute_force(self):
        colours = _support.noise_rgb(1, 500, 5).reshape(-1, 3)
        base = _support.noise_rgb(1, 7, 6).reshape(-1, 3)
        for chunk in (1, 6, 7, 8, 100, 3500, 4_000_000):
            with mock.patch.object(PE, "_DIST_CHUNK", chunk):
                got = PE._nearest_sq(colours, base)
            self.assertEqual(got.dtype, np.int32)
            self.assertEqual(got.tolist(), d2(colours, base).tolist(), chunk)
        self.assertEqual(PE._nearest_sq(pal(BLACK), pal(WHITE)).tolist(), [3 * 255 * 255])
        self.assertEqual(PE._nearest_sq(pal(RED), pal(RED, BLACK)).tolist(), [0])

    def test_extend_with_forced_chunking_same_result(self):
        base = pal(BLACK, WHITE)
        tile = (_support.noise_rgb(48, 48, 8), np.ones((6, 6), bool))
        expected = ext(base, [tile], 5)
        with mock.patch.object(PE, "_DIST_CHUNK", 3):
            self.assertTrue(np.array_equal(ext(base, [tile], 5), expected))


if __name__ == "__main__":
    unittest.main()
