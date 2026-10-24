"""Tests for tilemap_core.heuristics: each issue kind (positive/negative/boundary), fields, validation, purity."""

import unittest

import numpy as np

from tilemap_core import heuristics as H

S = 8  # pixelize.WORK_SCALE: work pixels per alpha cell
CREAM = np.array([235, 225, 200], np.uint8)


def job(tid, cat, var=0):
    return {"tile_id": tid, "category": cat, "variant": var, "footprint": [1, 1]}


def run(work, cat, alpha=None, tid="t", var=0):
    key = f"{tid}#{var}"
    return H.heuristic_issues({key: work}, {} if alpha is None else {key: alpha}, [job(tid, cat, var)])


def kinds(work, cat, alpha=None):
    return sorted(i["kind"] for i in run(work, cat, alpha))


def noise(h, w, lo, hi, seed=1):
    return np.random.default_rng(seed).integers(lo, hi, (h, w, 3)).astype(np.uint8)


def flat(h, w, colour):
    return np.broadcast_to(np.asarray(colour, np.uint8), (h, w, 3)).copy()


def blob_on_cream(h=128, w=160, r=30):
    img = flat(h, w, CREAM)
    yy, xx = np.mgrid[:h, :w]
    img[(yy - h // 2) ** 2 + (xx - w // 2) ** 2 < r * r] = (40, 90, 40)
    return img


def cream_share(img):
    return float((np.abs(img.astype(int) - CREAM.astype(int)).max(-1) <= 24).mean())


def ramped(amp, axis, h=128, w=160):
    """Exactly periodic smooth texture (wrap-around blur) plus a linear ramp of ``amp`` along ``axis`` (a seam)."""
    b = np.random.default_rng(3).integers(60, 180, (h, w, 3)).astype(np.float64)
    for _ in range(6):
        b = (b + np.roll(b, 1, 0) + np.roll(b, -1, 0) + np.roll(b, 1, 1) + np.roll(b, -1, 1)) / 5
    b += np.linspace(0, amp, (h, w)[axis]).reshape((1, -1, 1) if axis else (-1, 1, 1))
    return np.clip(b, 0, 255).astype(np.uint8)


def ratio(work):
    w = work.astype(np.int16)
    wrap = np.concatenate([np.abs(w[:, 0] - w[:, -1]).ravel(), np.abs(w[0] - w[-1]).ravel()]).mean()
    dx, dy = np.abs(w[:, 1:] - w[:, :-1]), np.abs(w[1:] - w[:-1])
    return wrap / ((dx.sum() + dy.sum()) / (dx.size + dy.size))


def obj_work(alpha, base=100, over=None):
    g = np.full(alpha.shape, base, np.uint8)
    for rc, v in (over or {}).items():
        g[rc] = v
    return np.repeat(np.repeat(g, S, 0), S, 1)[..., None].repeat(3, 2)


def square(n, rows=12, cols=16, r0=2, c0=3):
    a = np.zeros((rows, cols), bool)
    a[r0:r0 + n, c0:c0 + n] = True
    return a


def edge_cells(a):
    pad = np.pad(a, 1, constant_values=True)
    touch = ~pad[:-2, 1:-1] | ~pad[2:, 1:-1] | ~pad[1:-1, :-2] | ~pad[1:-1, 2:]
    return list(map(tuple, np.argwhere(a & touch).tolist()))


def cov_alpha(n_opaque, rows=10, cols=10):
    a = np.zeros((rows, cols), bool)
    a.flat[:n_opaque] = True
    return a


def gk(work):
    return kinds(work, "ground")


def ring_mask(h, w):
    m = np.ones((h, w), bool)
    m[1:-1, 1:-1] = False
    return m


class ContractTests(unittest.TestCase):
    def test_constants_pin_architecture_values(self):
        self.assertEqual(
            (H.GROUND_COLOUR_COVER, H.GROUND_BORDER_COVER, H.TILEABLE_RATIO, H.CUTOUT_MAX_COVERAGE,
             H.WRONG_CONTENT_MIN_COVERAGE, H.FRINGE_EDGE_FRACTION, H.FRINGE_PERCENTILE),
            (0.35, 0.60, 2.0, 0.90, 0.03, 0.15, 90))
        self.assertEqual((H.LIGHT_LUMA, H.COLOUR_BIN_SHIFT, H.COLOUR_TOLERANCE), (160.0, 5, 24))
        self.assertEqual(S, H.WORK_SCALE)
        self.assertEqual(sorted(H.FIX_HINTS), sorted(("ground_not_texture", "not_tileable", "not_cut_out", "wrong_content", "fringe")))
        self.assertTrue(all(h.strip() and len(h) <= 200 for h in H.FIX_HINTS.values()))

    def test_issue_fields_severity_and_key_format(self):
        cases = [
            (blob_on_cream(), "ground", None, "ground_not_texture", "high"),
            (ramped(200, 1), "ground", None, "not_tileable", "medium"),
            (obj_work(cov_alpha(100)), "overlay", cov_alpha(100), "not_cut_out", "medium"),
            (obj_work(cov_alpha(2)), "object", cov_alpha(2), "wrong_content", "medium"),
        ]
        for work, cat, alpha, kind, sev in cases:
            with self.subTest(kind=kind):
                (iss,) = [i for i in run(work, cat, alpha, tid="a_b", var=3) if i["kind"] == kind]
                self.assertEqual(set(iss), {"tile", "kind", "severity", "fix_hint", "score"})
                self.assertEqual((iss["tile"], iss["severity"], iss["fix_hint"]), ("a_b#3", sev, H.FIX_HINTS[kind]))
                self.assertIs(type(iss["score"]), float)
        (iss,) = run(obj_work(cov_alpha(100)), "object", cov_alpha(100), tid="x#y", var=12)
        self.assertEqual(iss["tile"], "x#y#12")  # '#' inside the id is kept verbatim


class GroundNotTextureTests(unittest.TestCase):
    def test_blob_on_cream_flagged_with_measured_cover_score_others_not(self):
        (iss,) = [i for i in run(blob_on_cream(), "ground") if i["kind"] == "ground_not_texture"]
        self.assertAlmostEqual(iss["score"], cream_share(blob_on_cream()), places=9)
        self.assertEqual(gk(noise(128, 160, 20, 200)), [])
        self.assertEqual(gk(flat(128, 160, (30, 60, 30))), [])
        self.assertIn("ground_not_texture", gk(flat(64, 64, (161,) * 3)))
        self.assertNotIn("ground_not_texture", gk(flat(64, 64, (159,) * 3)))
        self.assertNotIn("ground_not_texture", gk(flat(64, 64, (0, 255, 0))))  # luma ~150

    def test_colour_tolerance_boundary(self):
        for dev, want in ((24, 1.0), (25, 0.7)):  # 30 % of the columns deviate in red by ``dev``
            img = flat(80, 80, CREAM)
            img[:, :24, 0] -= dev
            (iss,) = [i for i in run(img, "ground") if i["kind"] == "ground_not_texture"]
            self.assertAlmostEqual(iss["score"], want, places=9, msg=f"dev={dev}")

    def test_cover_boundary_exactly_035(self):
        ring = ring_mask(80, 80)  # 316 px, always cream; the rest is dark noise except the first extras
        extra_idx = np.flatnonzero(~ring.ravel())
        for n_cream, flagged in ((2240, True), (2239, False)):  # of 6400 px: 0.35 exactly / one pixel under
            img = noise(80, 80, 0, 100, seed=9)
            img[ring] = CREAM
            img.reshape(-1, 3)[extra_idx[: n_cream - 316]] = CREAM
            self.assertEqual(cream_share(img), n_cream / 6400)
            self.assertEqual("ground_not_texture" in gk(img), flagged, n_cream)

    def test_border_boundary_exactly_060(self):
        ring_idx = np.flatnonzero(ring_mask(96, 96).ravel())  # 380 px; interior all cream, cover far above 0.35
        for n_cream, flagged in ((228, True), (227, False)):  # 228 / 380 = 0.60 exactly
            img = flat(96, 96, CREAM)
            img.reshape(-1, 3)[ring_idx[: 380 - n_cream]] = (40, 90, 40)
            self.assertEqual("ground_not_texture" in gk(img), flagged, n_cream)

    def test_border_ring_checks_all_four_sides(self):
        dark = (40, 90, 40)
        sides = {"top": (0, slice(None)), "bottom": (-1, slice(None)), "left": (slice(None), 0), "right": (slice(None), -1)}
        for name, idx in sides.items():
            img = flat(128, 160, CREAM)
            img[idx] = dark  # one dark side leaves >= 60 % of the ring cream -> still flagged
            self.assertIn("ground_not_texture", gk(img), name)
        for pair in (("top", "bottom"), ("left", "right")):
            img = flat(128, 160, CREAM)
            for name in pair:
                img[sides[name]] = dark  # 252 / 572 or 320 / 572 dark -> < 60 % cream
            self.assertNotIn("ground_not_texture", gk(img), pair)


class NotTileableTests(unittest.TestCase):
    def test_ratio_either_side_of_two(self):
        self.assertEqual(gk(ramped(0, 1)), [])  # exactly periodic texture
        self.assertEqual(gk(flat(128, 160, (50, 120, 60))), [])  # interior difference 0 -> skipped
        under = next(a for a in range(2, 200) if 1.5 <= ratio(ramped(a, 1)) <= 1.8)
        over = next(a for a in range(2, 200) if 2.3 <= ratio(ramped(a, 1)) <= 3.2)
        self.assertEqual(gk(ramped(under, 1)), [])
        for axis in (0, 1):  # a seam along either axis counts
            img = ramped(over, axis)
            (iss,) = run(img, "ground")
            self.assertEqual(iss["kind"], "not_tileable", axis)
            self.assertAlmostEqual(iss["score"], ratio(img), places=9, msg=axis)

    def test_wrap_uses_first_and_last_line(self):
        img = flat(64, 64, (100, 100, 100))
        img[:, 1:-1] += noise(64, 62, 0, 2)  # tiny interior noise
        img[:, -1] = 200  # only the last column differs from the first
        self.assertEqual(gk(img), ["not_tileable"])
        self.assertEqual(gk(img.transpose(1, 0, 2).copy()), ["not_tileable"])


class CoverageTests(unittest.TestCase):
    def test_cut_out_and_wrong_content_boundaries(self):
        for n, want in ((100, ["not_cut_out"]), (91, ["not_cut_out"]), (90, []), (50, []), (3, []),
                        (2, ["wrong_content"]), (0, ["wrong_content"])):
            self.assertEqual(kinds(obj_work(cov_alpha(n)), "object", cov_alpha(n)), want, n)  # 90 % / 3 % are clean
        for n, kind in ((95, "not_cut_out"), (1, "wrong_content")):
            a = cov_alpha(n)
            (iss,) = run(obj_work(a), "overlay", a)
            self.assertEqual((iss["kind"], iss["score"]), (kind, n / 100))


class FringeTests(unittest.TestCase):
    def setUp(self):
        self.sq = square(6)
        self.ec = edge_cells(self.sq)
        self.assertEqual(len(self.ec), 20)

    def fringe(self, base, over, alpha=None):
        a = self.sq if alpha is None else alpha
        return kinds(obj_work(a, base, over), "object", a)

    def test_fraction_boundary_15_percent(self):
        self.assertEqual(self.fringe(100, {c: 250 for c in self.ec[:2]}), [])
        (iss,) = run(obj_work(self.sq, 100, {c: 250 for c in self.ec[:3]}), "object", self.sq)
        self.assertEqual((iss["kind"], iss["score"], iss["severity"], iss["fix_hint"]),
                         ("fringe", 0.15, "medium", H.FIX_HINTS["fringe"]))

    def test_dark_outline_and_sub_percentile_edges_not_flagged(self):
        self.assertEqual(self.fringe(200, {c: 60 for c in self.ec}), [])
        mid = {c: 130 for c in self.ec[:3]}
        mid.update({(r, c): 200 for r in (4, 5, 6, 7) for c in (5, 6)})  # 8 bright interior cells set the p90
        self.assertEqual(self.fringe(100, mid), [])

    def test_all_four_neighbour_directions_detected(self):
        big = square(8)
        sides = {"top": [(2, c) for c in range(5, 10)], "bottom": [(9, c) for c in range(5, 10)],
                 "left": [(r, 3) for r in range(4, 9)], "right": [(r, 10) for r in range(4, 9)]}
        for name, cells in sides.items():
            self.assertEqual(self.fringe(100, {c: 250 for c in cells}, big), ["fringe"], name)
        # the grid border is not transparency
        half = np.tile(np.arange(16) < 8, (12, 1))
        self.assertEqual(kinds(obj_work(half, 100, {(0, c): 250 for c in range(7)}), "object", half), [])


class ValidationTests(unittest.TestCase):
    def test_bad_inputs_raise_value_error(self):
        w, ones, g = noise(64, 64, 0, 255), np.ones((8, 8), bool), job("t", "ground")
        t, j = {"t#0": w}, [g]
        bad = {
            "float work": ({"t#0": w.astype(np.float32)}, {}, j), "2 channels": ({"t#0": w[..., :2]}, {}, j),
            "2d work": ({"t#0": w[..., 0]}, {}, j), "height not multiple": ({"t#0": w[:60]}, {}, j),
            "width not multiple": ({"t#0": w[:, :60]}, {}, j), "8 high": ({"t#0": w[:8]}, {}, j), "8 wide": ({"t#0": w[:, :8]}, {}, j),
            "no image": ({}, {}, j), "extra image": ({"t#0": w, "x#0": w}, {}, j), "extra alpha": (t, {"x#0": ones}, j),
            "alpha shape": (t, {"t#0": np.ones((7, 8), bool)}, j), "alpha transposed": ({"t#0": w[:, :32]}, {"t#0": ones}, j),
            "alpha dtype": (t, {"t#0": ones.astype(np.uint8)}, j), "bad category": (t, {}, [job("t", "wall")]),
            "list category": (t, {}, [job("t", ["ground"])]), "duplicate key": (t, {}, [g, g]),
            "job missing keys": (t, {}, [{"tile_id": "t"}]), "jobs str": (t, {}, "t"), "jobs mapping": (t, {}, {"t": 1}),
            "jobs None": (t, {}, None), "job not mapping": (t, {}, ["t"]), "work not mapping": ([w], {}, j),
            "alphas not mapping": (t, [], j), "negative variant": ({"t#-1": w}, {}, [job("t", "ground", -1)]),
            "bool variant": ({"t#True": w}, {}, [job("t", "ground", True)]), "int tile_id": (t, {}, [job(5, "ground")]),
            "object no alpha": (t, {}, [job("t", "object")]), "overlay alpha None": (t, {"t#0": None}, [job("t", "overlay")]),
        }
        for name, (cells, alphas, jobs) in bad.items():
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    H.heuristic_issues(cells, alphas, jobs)

    def test_size_limits_are_exact_not_clamped(self):
        run(noise(16, 16, 0, 255), "ground")  # 2 * WORK_SCALE per side is the minimum, accepted
        run(np.zeros((2048, 2048, 3), np.uint8), "ground")  # 2048 * 2048 px is the maximum, accepted
        with self.assertRaises(ValueError):
            run(np.zeros((2048, 2056, 3), np.uint8), "ground")  # one more work cell column: rejected

    def test_alpha_optional_only_for_ground_and_empty_or_tuple_jobs(self):
        w = noise(64, 64, 20, 200)
        for alphas in ({}, {"t#0": None}, {"t#0": np.ones((8, 8), bool)}):
            self.assertEqual(H.heuristic_issues({"t#0": w}, alphas, [job("t", "ground")]), [])
        self.assertEqual(H.heuristic_issues({}, {}, []), [])
        self.assertEqual(H.heuristic_issues({"t#0": blob_on_cream()}, {}, (job("t", "ground"),))[0]["kind"], "ground_not_texture")


class PurityTests(unittest.TestCase):
    def test_deterministic_job_order_and_inputs_untouched(self):
        jobs = [job("grass", "ground", 0), job("tree", "object", 0), job("tree", "object", 1), job("grass", "ground", 1)]
        works = {"grass#0": blob_on_cream(), "tree#0": obj_work(cov_alpha(100)), "tree#1": obj_work(square(6)),
                 "grass#1": ramped(200, 1)}
        alphas = {"tree#0": cov_alpha(100), "tree#1": square(6)}
        snap = {k: v.copy() for k, v in {**works, **{"a" + k: v for k, v in alphas.items()}}.items()}
        for arr in list(works.values()) + list(alphas.values()):
            arr.setflags(write=False)  # a write attempt would raise
        first = H.heuristic_issues(works, alphas, jobs)
        self.assertEqual(first, H.heuristic_issues(works, alphas, jobs))
        self.assertTrue(all(np.array_equal(works[k], snap[k]) for k in works))
        self.assertTrue(all(np.array_equal(alphas[k], snap["a" + k]) for k in alphas))
        self.assertEqual([(i["tile"], i["kind"]) for i in first],
                         [("grass#0", "ground_not_texture"), ("tree#0", "not_cut_out"), ("grass#1", "not_tileable")])

    def test_issue_order_within_a_tile_follows_detector_order(self):
        a = cov_alpha(5, 12, 16)  # one row of 5 opaque cells: coverage 2.6 % < 3 %, all of them edge cells
        work = obj_work(a, 100, {(0, 0): 250})  # 1 of 5 above the p90 (190) -> 20 % >= 15 %
        self.assertEqual([i["kind"] for i in run(work, "object", a)], ["wrong_content", "fringe"])
        both = flat(128, 160, CREAM)
        both[:, -1] = (40, 90, 40)  # cream stays dominant (>= 60 % of the ring) but the wrap edge jumps
        self.assertEqual([i["kind"] for i in run(both, "ground")], ["ground_not_texture", "not_tileable"])


if __name__ == "__main__":
    unittest.main()
