"""Tests for tilemap_core.regen_plan (part 1): constants, key_weights, input validation; owns the shared helpers.

Covers limits at/just beyond (max_regen 1/32 ok, 0/33 rejected; max_attempts 1/5 ok, 0/6 rejected; round_no 1/6),
weights, dedupe, Wang resolution and exact ``ValueError`` paths. ``plan_regen`` behaviour, ``record_attempts``,
``best_version`` and ``snap_for_issues`` live in ``tests/test_regen_plan_loop.py``, which imports the helpers
``mk``, ``jobs``, ``iss``, ``plan``, ``keys`` from here.

Residual gaps (not tested, both files):
* the non-numeric branch of ``snap_for_issues`` (``... else value``): no defect kind overrides a non-number today;
* ``amend_job`` prompt-length trimming and ``sanitize.clean_text`` internals (owned by defects/sanitize tests);
* ``transitions`` given as a one-shot iterator (only lists/tuples are used);
* kind-rank tie-break when a ``transition`` issue's hint competes with another hint of equal weight;
* the order ``_resolve`` returns bare-id variants in (equivalent: callers re-sort by weight and key).
"""

import copy
import unittest

from tilemap_core import regen_plan as R

TRANS = [["grass", "dirt"]]
GOOD = {"tile": "dirt", "kind": "seam", "severity": "medium", "fix_hint": ""}


def mk(tid, var=0, cat="ground", seed=100):
    return {"tile_id": tid, "variant": var, "category": cat, "seed": seed, "footprint": [1, 1], "prompt": f"{tid} tile {var}"}


def jobs():
    js = [mk("grass", 0), mk("grass", 1, seed=200), mk("dirt"), mk("tree", cat="object"), mk("leaf", cat="overlay")]
    return {f"{j['tile_id']}#{j['variant']}": j for j in js}


def iss(tile, kind="seam", sev="medium", hint=""):
    return {"tile": tile, "kind": kind, "severity": sev, "fix_hint": hint}


def plan(issues, att=None, regen=8, attempts=3, rnd=1, js=None, trans=()):
    return R.plan_regen(issues, js or jobs(), {} if att is None else att, regen, attempts, rnd, transitions=trans)


def keys(new):
    return [f"{j['tile_id']}#{j['variant']}" for j in new]


class Constants(unittest.TestCase):
    def test_values(self):
        self.assertEqual(R.SEVERITY_WEIGHT, {"low": 1, "medium": 2, "high": 4})
        self.assertEqual((R.SEED_STEP, R.SEED_LIMIT, R.MAX_ROUND), (10007, 2**64, 6))
        self.assertEqual((R.MAX_REGEN_RANGE, R.MAX_ATTEMPTS_RANGE), ((1, 32), (1, 5)))
        self.assertTrue(all(hasattr(R, n) for n in R.__all__))


class KeyWeights(unittest.TestCase):
    def test_empty_issues(self):
        self.assertEqual(R.key_weights([], jobs()), {})

    def test_severity_weights_and_order(self):
        got = R.key_weights([iss("tree"), iss("dirt#0"), iss("leaf", sev="high"), iss("grass#0", sev="low")], jobs())
        self.assertEqual(list(got.items()), [("leaf#0", 4), ("dirt#0", 2), ("tree#0", 2), ("grass#0", 1)])

    def test_distinct_kinds_sum_duplicates_count_once_at_highest(self):
        self.assertEqual(R.key_weights([iss("dirt", "seam"), iss("dirt", "fringe", "low")], jobs()), {"dirt#0": 3})
        got = R.key_weights([iss("dirt", sev="low"), iss("dirt#0", sev="high"), iss("dirt", sev="medium")], jobs())
        self.assertEqual(got, {"dirt#0": 4})

    def test_bare_id_blames_every_variant_and_strips(self):
        self.assertEqual(list(R.key_weights([iss(" grass ", sev="high")], jobs()).items()), [("grass#0", 4), ("grass#1", 4)])

    def test_transition(self):
        self.assertEqual(R.key_weights([iss("dirt", "transition", "high")], jobs()), {"dirt#0": 8})
        self.assertEqual(R.key_weights([iss("tree", "transition", "high")], jobs()), {"tree#0": 4})
        got = R.key_weights([iss("dirt", "ground_not_texture", "low"), iss("dirt", "transition", "high")], jobs())
        self.assertEqual(got, {"dirt#0": 8})

    def test_wang_references_blame_both_parents(self):
        for ref in ("grass_dirt/3", "grass_dirt#3", "grass_dirt", " grass_dirt/15 ", "grass_dirt/0"):
            self.assertEqual(R.key_weights([iss(ref, sev="low")], jobs(), TRANS), {"dirt#0": 1, "grass#0": 1}, ref)
        self.assertEqual(R.key_weights([iss("dirt_dirt/1")], jobs(), [["dirt", "dirt"]]), {"dirt#0": 2})

    def test_unresolvable_references(self):
        for ref in ("nope", "grass#7", "grass_dirt/3", "ghost#1", "grass_dirt#3"):
            with self.assertRaisesRegex(ValueError, "no transitions given", msg=ref):
                R.key_weights([iss(ref)], jobs())
        for ref in ("grass_dirt/16", "grass_dirt/-1", "grass_dirt/x", "grass_dirt/", "grass_dirt/\u0663", "ghost/1",
                    "grass#7", "dirt_grass/1"):
            with self.assertRaises(ValueError, msg=ref):
                R.key_weights([iss(ref)], jobs(), TRANS)
        js = jobs()
        del js["dirt#0"]
        with self.assertRaisesRegex(ValueError, "not in jobs_by_key"):
            R.key_weights([iss("grass_dirt/1")], js, TRANS)

    def test_bad_transitions(self):
        for bad in ("ab", None, 5, [["a"]], [["a", "b", "c"]], [["a", 1]], [["", "b"]], ["ab"], [("a", "b"), ("a", "b")]):
            with self.assertRaises(ValueError, msg=repr(bad)):
                R.key_weights([], jobs(), bad)
        R.key_weights([], jobs(), (("a", "b"), ("b", "a")))  # reversed pair is a different set

    def test_inputs_not_mutated(self):
        issues, js, tr = [iss("grass", "transition", "high", "h")], jobs(), [["grass", "dirt"]]
        before = copy.deepcopy((issues, js, tr))
        R.key_weights(issues, js, tr)
        self.assertEqual((issues, js, tr), before)


class Validation(unittest.TestCase):
    def bad(self, **kw):
        a = dict(issues=[], js=jobs(), att={}, regen=8, attempts=3, rnd=1)
        a.update(kw)
        with self.assertRaises(ValueError, msg=repr(kw)):
            R.plan_regen(a["issues"], a["js"], a["att"], a["regen"], a["attempts"], a["rnd"])

    def test_limits_boundaries_accepted(self):
        for kw in ({"regen": 1}, {"regen": 32}, {"attempts": 1}, {"attempts": 5}, {"rnd": 1}, {"rnd": 6}):
            self.assertEqual(plan([], **kw), ([], []))

    def test_limits_beyond_rejected(self):
        for v in (0, 33, -1, True, 8.0, "8", None):
            self.bad(regen=v)
        for v in (0, 6, -1, True, 3.0, None):
            self.bad(attempts=v)
        for v in (0, 7, True, 1.0, None):
            self.bad(rnd=v)

    def test_bad_jobs(self):
        for js in ({}, [], {"a#0": "x"}, {**jobs(), "dirt#0": mk("dirt", 1)}):
            self.bad(js=js)
        for patch in ({"tile_id": 5}, {"variant": -1}, {"variant": True}, {"variant": "0"}, {"category": "wall"},
                      {"prompt": None}, {"seed": -1}, {"seed": 2**64}, {"seed": True}, {"seed": 1.0}, {"tile_id": "x"}):
            self.bad(js={**jobs(), "dirt#0": {**mk("dirt"), **patch}})
        self.assertEqual(len(plan([iss("a")], js={"a#0": mk("a", seed=2**64 - 1)})[0]), 1)

    def test_bad_attempts(self):
        for att in (None, [], {"dirt#0": -1}, {"dirt#0": True}, {"dirt#0": 1.5}, {5: 1}, {"dirt#0": 2**31}):
            self.bad(att=att)
        self.assertEqual(plan([], att={"dirt#0": 2**31 - 1}), ([], []))

    def test_bad_issues(self):
        for bad in (None, "x", {"tile": "dirt"}, [5], [{**GOOD, "tile": ""}], [{**GOOD, "tile": "  "}],
                    [{**GOOD, "tile": 3}], [{**GOOD, "kind": "nope"}], [{**GOOD, "kind": None}],
                    [{**GOOD, "severity": "urgent"}], [{**GOOD, "fix_hint": 4}], [{"tile": "dirt"}]):
            self.bad(issues=bad)
        self.assertEqual(len(plan((GOOD,))[0]), 1)  # tuple accepted
        self.assertEqual(len(plan([{k: v for k, v in GOOD.items() if k != "fix_hint"}])[0]), 1)  # hint optional

    def test_hint_length(self):
        self.bad(issues=[iss("dirt", hint="x" * 201)])
        self.assertEqual(len(plan([iss("dirt", hint="x" * 200)])[0]), 1)
