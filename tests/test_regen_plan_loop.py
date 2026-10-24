"""Tests for tilemap_core.regen_plan (part 2): plan_regen, record_attempts, best_version, snap_for_issues.

Covers ordering, ``max_regen`` truncation, attempt counting and exhaustion, dedupe, hint selection, seed shift,
determinism, input purity, keep-best bookkeeping and the atlas-wide snap overrides. Constants, ``key_weights``
and input validation are in ``tests/test_regen_plan.py``, whose helpers are imported below. Residual gaps are
listed in that file's docstring.
"""

import copy
import unittest

from tests.test_regen_plan import TRANS, iss, jobs, keys, mk, plan
from tilemap_core import regen_plan as R
from tilemap_core.defects import DEFECT_FIXES, DEFECT_KINDS, amendment


class PlanRegen(unittest.TestCase):
    def test_empty_issues_and_types(self):
        out = plan([])
        self.assertEqual(out, ([], []))
        self.assertIsInstance(out, tuple)

    def test_job_fields(self):
        (job,), skipped = plan([iss("grass#1", "seam", "high", "  flat please ")], att={"grass#1": 1}, rnd=4)
        base = jobs()["grass#1"]
        self.assertEqual(skipped, [])
        self.assertEqual(job["prompt"], base["prompt"] + ", " + amendment(["seam"], "flat please"))
        self.assertEqual((job["attempt"], job["round"], job["defects"]), (2, 4, ["seam"]))
        self.assertEqual(job["seed"], 200 + 2 * 10007)
        for k in ("tile_id", "variant", "category", "footprint"):
            self.assertEqual(job[k], base[k])

    def test_seed_per_attempt_and_wrap(self):
        for used in range(5):
            job = plan([iss("dirt")], att={"dirt#0": used}, attempts=5)[0][0]
            self.assertEqual((job["attempt"], job["seed"]), (used + 1, 100 + (used + 1) * 10007))
        self.assertEqual(plan([iss("a")], js={"a#0": mk("a", seed=2**64 - 1)})[0][0]["seed"], 10006)

    def test_defects_canonical_order_and_transition_expansion(self):
        job = plan([iss("dirt", k) for k in reversed(DEFECT_KINDS)])[0][0]
        self.assertEqual(job["defects"], [k for k in DEFECT_KINDS if k != "transition"])  # transition -> 2 other kinds
        for k in job["defects"]:
            self.assertIn(DEFECT_FIXES[k]["prompt"], job["prompt"])
        self.assertEqual(plan([iss("dirt", "transition")])[0][0]["defects"], ["not_tileable", "ground_not_texture"])
        self.assertEqual(plan([iss("tree", "transition")])[0][0]["defects"], ["other"])

    def test_order_weight_desc_then_key(self):
        issues = [iss("tree", sev="low"), iss("leaf"), iss("dirt"), iss("grass#0", sev="high")]
        self.assertEqual(keys(plan(issues)[0]), ["grass#0", "dirt#0", "leaf#0", "tree#0"])

    def test_max_regen_truncates_without_skipping(self):
        issues = [iss("tree", sev="low"), iss("leaf", sev="high"), iss("dirt")]
        new, skipped = plan(issues, regen=2)
        self.assertEqual((keys(new), skipped), (["leaf#0", "dirt#0"], []))  # the cut key stays eligible next round
        self.assertEqual(keys(plan(issues, regen=1)[0]), ["leaf#0"])
        self.assertEqual(len(plan(issues, regen=32)[0]), 3)
        js = {f"t{n:02d}#0": mk(f"t{n:02d}") for n in range(40)}
        new, _ = plan([iss(f"t{n:02d}") for n in range(40)], regen=32, js=js)
        self.assertEqual(keys(new), [f"t{n:02d}#0" for n in range(32)])

    def test_exhaustion_boundary(self):
        for m in (1, 3, 5):
            self.assertEqual(plan([iss("dirt")], att={"dirt#0": m}, attempts=m), ([], ["dirt#0"]))
            new, skipped = plan([iss("dirt")], att={"dirt#0": m - 1}, attempts=m)
            self.assertEqual((len(new), skipped, new[0]["attempt"]), (1, [], m))
        self.assertEqual(plan([iss("dirt")], att={"dirt#0": 9}, attempts=2)[1], ["dirt#0"])  # over the cap too

    def test_missing_attempts_zero_and_unrelated_ignored(self):
        new, skipped = plan([iss("dirt")], att={"tree#0": 5, "ghost#9": 5})
        self.assertEqual((new[0]["attempt"], skipped), (1, []))

    def test_skipped_ordered_and_use_no_slots(self):
        issues = [iss("tree", sev="high"), iss("leaf", sev="high"), iss("dirt", sev="low")]
        new, skipped = plan(issues, att={"tree#0": 3, "leaf#0": 3}, regen=1)
        self.assertEqual((skipped, keys(new)), (["leaf#0", "tree#0"], ["dirt#0"]))

    def test_dedupe_each_key_once(self):
        new, _ = plan([iss("dirt"), iss("dirt#0", "fringe"), iss("dirt", "seam", "high"), iss("grass"), iss("grass#0")])
        self.assertEqual(sorted(keys(new)), ["dirt#0", "grass#0", "grass#1"])
        self.assertEqual(next(j for j in new if j["tile_id"] == "dirt")["defects"], ["seam", "fringe"])

    def test_hint_selection(self):
        job = plan([iss("dirt", "seam", "low", "LOWHINT"), iss("dirt", "fringe", "high", "HIGHHINT")])[0][0]
        self.assertTrue(job["prompt"].endswith("HIGHHINT") and "LOWHINT" not in job["prompt"])
        a = plan([iss("dirt", "fringe", "high", "AAA"), iss("dirt", "seam", "high", "ZZZ")])[0][0]
        self.assertTrue(a["prompt"].endswith("ZZZ"))  # earlier canonical kind (seam) wins on equal weight
        b = plan([iss("dirt", "seam", "high", "bbb"), iss("dirt", "seam", "high", "aaa")])[0][0]
        self.assertTrue(b["prompt"].endswith("aaa"))  # same kind: smaller text wins
        c = plan([iss("dirt", "seam", "high", "   "), iss("dirt", "fringe", "low", "KEEP")])[0][0]
        self.assertTrue(c["prompt"].endswith("KEEP"))  # blank hint ignored
        self.assertEqual(plan([iss("dirt", "seam", "high", "  ")])[0][0]["prompt"], "dirt tile 0, " + amendment(["seam"]))

    def test_wang_issue_plans_both_parents(self):
        new, _ = plan([iss("grass_dirt/5", "transition", "high")], trans=TRANS)
        self.assertEqual(keys(new), ["dirt#0", "grass#0"])

    def test_deterministic_not_aliased_inputs_not_mutated(self):
        issues, js, att = [iss("grass", "transition", "high", "h"), iss("tree")], jobs(), {"grass#0": 1}
        before = copy.deepcopy((issues, js, att))
        first = plan(issues, att=att, js=js)
        self.assertEqual(plan(issues, att=att, js=js), first)
        self.assertEqual(plan(list(reversed(issues)), att=att, js=js), first)
        first[0][0]["footprint"].append(9)
        first[0][0]["prompt"] = "changed"
        self.assertEqual((issues, js, att), before)

    def test_loop_exhausts_after_max_attempts_rounds(self):
        att, seeds = {}, []
        for rnd in range(1, 4):
            new, skipped = plan([iss("dirt")], att=att, attempts=3, rnd=rnd)
            self.assertEqual((len(new), skipped), (1, []))
            seeds.append(new[0]["seed"])
            att = R.record_attempts(att, new)
        self.assertEqual(plan([iss("dirt")], att=att, attempts=3, rnd=4), ([], ["dirt#0"]))
        self.assertEqual(len(set(seeds)), 3)


class RecordAttempts(unittest.TestCase):
    def test_counts_and_copies(self):
        att = {"dirt#0": 1, "tree#0": 2}
        new, _ = plan([iss("dirt"), iss("leaf")], att=att)
        out = R.record_attempts(att, new)
        self.assertEqual(out, {"dirt#0": 2, "tree#0": 2, "leaf#0": 1})
        self.assertEqual(att, {"dirt#0": 1, "tree#0": 2})
        same = R.record_attempts(att, [])
        self.assertEqual(same, att)
        self.assertIsNot(same, att)

    def test_stale_or_double_counted_plan_rejected(self):
        new, _ = plan([iss("dirt")])
        with self.assertRaisesRegex(ValueError, "expected 2"):
            R.record_attempts({"dirt#0": 1}, new)
        with self.assertRaises(ValueError):
            R.record_attempts({}, new + new)  # same key listed twice

    def test_bad_inputs(self):
        ok = {"tile_id": "a", "variant": 0, "attempt": 1}
        for bad_job in (5, {"variant": 0, "attempt": 1}, {**ok, "variant": "0"}, {**ok, "tile_id": 3},
                        {**ok, "attempt": 2}, {**ok, "attempt": True}, {**ok, "attempt": 1.0}, {**ok, "attempt": None}):
            with self.assertRaises(ValueError, msg=repr(bad_job)):
                R.record_attempts({}, [bad_job])
        for att in ({"a#0": -1}, None):
            with self.assertRaises(ValueError):
                R.record_attempts(att, [])
        self.assertEqual(R.record_attempts({}, [ok]), {"a#0": 1})


class BestVersion(unittest.TestCase):
    def test_first_candidate_and_lower_weight_win(self):
        self.assertEqual(R.best_version(None, ("img", 3)), (("img", 3), True))
        self.assertEqual(R.best_version(("old", 4), ("new", 3)), (("new", 3), True))
        self.assertEqual(R.best_version(("old", 1), ("new", 0)), (("new", 0), True))

    def test_tie_and_higher_keep_best(self):
        best = ("old", 2)
        for cand in (("new", 2), ("new", 5)):
            got, replaced = R.best_version(best, cand)
            self.assertIs(got, best)
            self.assertFalse(replaced)

    def test_images_are_opaque(self):
        other = [1]
        self.assertIs(R.best_version((object(), 5), (other, 1))[0][0], other)

    def test_weight_bounds_and_shape(self):
        self.assertEqual(R.best_version(None, ("i", 0))[0][1], 0)
        self.assertEqual(R.best_version(None, ("i", 2**31 - 1))[0][1], 2**31 - 1)
        for w in (-1, 2**31, True, 1.5, "1", None):
            with self.assertRaises(ValueError, msg=repr(w)):
                R.best_version(None, ("i", w))
            with self.assertRaises(ValueError, msg=repr(w)):
                R.best_version(("i", w), ("j", 1))
        for bad in (["i", 1], ("i",), ("i", 1, 2), "ab", None, 5):
            with self.assertRaises(ValueError, msg=repr(bad)):
                R.best_version(None, bad)
        with self.assertRaises(ValueError):
            R.best_version(["i", 1], ("j", 0))


class SnapForIssues(unittest.TestCase):
    def test_no_issues_returns_validated_defaults(self):
        out = R.snap_for_issues([])
        self.assertEqual(out["key_tolerance"], 40)
        self.assertEqual(R.snap_for_issues([], {"palette_size": 10})["palette_size"], 10)
        self.assertEqual(R.snap_for_issues([], None), out)

    def test_fringe_raises_never_lowers(self):
        fr = [iss("a", "fringe")]
        self.assertEqual(R.snap_for_issues(fr, {"key_tolerance": 10})["key_tolerance"], 60)
        self.assertEqual(R.snap_for_issues(fr)["key_tolerance"], 60)
        self.assertEqual(R.snap_for_issues(fr, {"key_tolerance": 100})["key_tolerance"], 100)
        self.assertEqual(R.snap_for_issues(fr, {"key_tolerance": 60})["key_tolerance"], 60)

    def test_other_kinds_change_nothing(self):
        others = [iss("a", k) for k in DEFECT_KINDS if k != "fringe"]
        self.assertEqual(R.snap_for_issues(others, {"key_tolerance": 5}), R.snap_for_issues([], {"key_tolerance": 5}))

    def test_other_params_preserved_inputs_untouched_refs_unresolved(self):
        params = {"palette_size": 12, "seam_blend": 0.5, "transitions": False, "key_tolerance": 0}
        before = copy.deepcopy(params)
        out = R.snap_for_issues([iss("ghost/99", "fringe"), iss("b", "fringe")], params)
        self.assertEqual(params, before)
        self.assertEqual((out["palette_size"], out["seam_blend"], out["transitions"], out["key_tolerance"]),
                         (12, 0.5, False, 60))
        self.assertIsNot(out, params)

    def test_bad_inputs(self):
        for issues, params in (([{"tile": "a", "kind": "nope", "severity": "low"}], None), ("x", None),
                               ([], {"bogus": 1}), ([iss("a", "fringe")], {"key_tolerance": 256})):
            with self.assertRaises(ValueError, msg=repr((issues, params))):
                R.snap_for_issues(issues, params)


if __name__ == "__main__":
    unittest.main()
