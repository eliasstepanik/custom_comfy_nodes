"""ValueError paths of tilemap_core.palette_extend.extend_palette (D17b).

Split from ``tests/test_palette_extend.py`` (300-line rule); helpers are imported
from it. Covers base palette shape/dtype/size, k_extra type and 0..16 range,
base+k_extra <= 256 (checked before the tile checks), works/alphas container
and length, empty tile list, work dtype/shape/block alignment, alpha dtype/shape
(per tile, error message names the tile index) and untouched arguments on error.

Residual untested gaps: see the module docstring of ``tests/test_palette_extend.py``.
"""

import unittest

import numpy as np

from tilemap_core import palette_extend as PE
from tests.test_palette_extend import BLACK, RED, WHITE, flat, pal


class Validation(unittest.TestCase):
    def setUp(self):
        self.base = pal(BLACK)
        self.t = flat(RED)

    def call(self, base=None, works=None, alphas=None, k=2):
        return PE.extend_palette(self.base if base is None else base, [self.t[0]] if works is None else works,
                                 [self.t[1]] if alphas is None else alphas, k)

    def bad(self, regex=None, **kw):
        with self.assertRaisesRegex(ValueError, regex or ""):
            self.call(**kw)

    def test_baseline_ok(self):
        self.assertEqual(self.call().shape, (2, 3))

    def test_bad_base(self):
        for b in ([[0, 0, 0]], pal(BLACK).astype(np.int32), np.zeros(3, np.uint8), np.zeros((2, 4), np.uint8),
                  np.zeros((0, 3), np.uint8), np.zeros((257, 3), np.uint8), "abc", np.zeros((1, 3, 1), np.uint8)):
            self.bad(base=b)

    def test_base_size_limits(self):
        self.assertEqual(self.call(base=np.zeros((256, 3), np.uint8), k=0).shape, (256, 3))
        self.call(base=np.zeros((1, 3), np.uint8), k=16)
        self.bad("palette must hold", base=np.zeros((257, 3), np.uint8))

    def test_k_extra_range_and_type(self):
        self.assertEqual(self.call(k=16).shape[0], 2)
        self.call(k=np.int64(3))
        for k in (-1, 17, 1000, True, False, 1.0, 2.5, "2", None, np.float32(1)):
            self.bad(k=k)
        self.bad("k_extra must be in 0..16", k=17)
        self.bad("k_extra must be in 0..16", k=-1)

    def test_total_palette_limit(self):
        self.assertEqual(self.call(base=np.zeros((240, 3), np.uint8), k=16).shape[0], 241)  # 240+16 <= 256
        self.bad("exceeds 256", base=np.zeros((241, 3), np.uint8), k=16)
        self.bad("exceeds 256", base=np.zeros((256, 3), np.uint8), k=1)
        self.bad("exceeds 256", base=np.zeros((250, 3), np.uint8), k=16, works=[], alphas=[])  # checked before tiles

    def test_works_alphas_container_and_length(self):
        w, a = self.t
        self.bad("equal length", works=w, alphas=[a])
        self.bad("equal length", works=[w], alphas=a)
        self.bad("equal length", works=w, alphas=a)
        self.bad("equal length", works=[w, w], alphas=[a])
        self.bad("equal length", works=[w], alphas=[a, a])
        self.bad("at least one tile", works=[], alphas=[])

    def test_bad_work(self):
        w, a = self.t
        for bw in (None, [[0]], w.astype(np.float32), w[..., 0], np.zeros((8, 8, 4), np.uint8),
                   np.zeros((0, 8, 3), np.uint8), np.zeros((8, 0, 3), np.uint8),
                   np.zeros((9, 8, 3), np.uint8), np.zeros((8, 7, 3), np.uint8), np.zeros((16, 12, 3), np.uint8)):
            self.bad(works=[bw], alphas=[a])

    def test_bad_alpha_and_message_names_index(self):
        w, a = self.t
        for ba in (None, a.tolist(), a.astype(np.uint8), a.astype(np.int64), np.ones((2, 1), bool),
                   np.ones((1, 2), bool), np.ones((8, 8), bool), np.ones((1,), bool), np.ones((1, 1, 1), bool)):
            self.bad("alphas\\[0\\]", works=[w], alphas=[ba])
        self.bad("alphas\\[1\\]", works=[w, w], alphas=[a, np.ones((2, 2), bool)])
        self.bad("works\\[1\\]", works=[w, None], alphas=[a, a])
        self.bad("works\\[2\\]", works=[w, w, np.zeros((9, 8, 3), np.uint8)], alphas=[a, a, a])

    def test_alpha_shape_follows_each_work(self):
        w2, a2 = flat(RED, 2, 3)
        self.assertEqual(self.call(works=[w2], alphas=[a2]).shape[0], 2)
        self.bad(works=[w2], alphas=[a2.T])

    def test_bad_input_leaves_arguments_untouched(self):
        base = pal(BLACK, WHITE)
        w, a = flat(RED)
        snap = (base.copy(), w.copy(), a.copy())
        with self.assertRaises(ValueError):
            PE.extend_palette(base, [w], [a], 17)
        for now, before in zip((base, w, a), snap):
            self.assertTrue(np.array_equal(now, before))


if __name__ == "__main__":
    unittest.main()
