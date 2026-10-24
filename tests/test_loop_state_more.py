"""Tests for ``nodes/_loop_state.py`` (D13a, part 2): ``parse_job_json``, ``LlmBudget``, ``is_passed``,
``round_entry``, ``build_log``, ``build_job_json``. Helpers come from ``tests/test_loop_state.py`` (functions only,
so no TestCase is collected twice). Stdlib unittest, no network, no ComfyUI.

Residual gaps: the ``RecursionError`` branch of ``parse_job_json`` depends on the interpreter's parser limit (the
test asserts the ValueError but cannot prove which exception path ran); ``sanitize.clean_text`` internals
(control-character stripping) are covered by its own tests, here only the length bound is asserted.
"""

import json
import unittest

from tests.test_loop_state import BASE, JOBS, MISSING, PLAN, job_text, load, unload
from tilemap_core import expand, llm_client, review

LS = None


def setUpModule():
    global LS
    LS = load()[0]


def tearDownModule():
    unload()


class ParseJobJsonTests(unittest.TestCase):
    def fails(self, text, pattern):
        with self.assertRaisesRegex(ValueError, pattern):
            LS.parse_job_json(text)

    def test_valid_returns_all_parts(self):
        data, size, view, plan, jobs, seed = LS.parse_job_json(job_text())
        self.assertEqual((size, view, seed), (16, "top-down", 7))
        self.assertEqual(plan, PLAN)
        self.assertEqual(jobs, JOBS)
        self.assertEqual(sorted(data), ["jobs", "plan", "tile_size", "view"])
        LS.parse_job_json(job_text(base=None))

    def test_text_type_and_size_boundary(self):
        msg = r"^job_json must be text of at most 400000 characters; reconnect job_json from CCN_TilemapExpand$"
        for bad in (None, b"{}", 5, ["{}"]):
            self.fails(bad, msg)
        self.fails(" " * 400_001, msg)
        self.fails(" " * 400_000, r"^job_json is not valid JSON; reconnect")  # exactly at the limit passes the size check

    def test_invalid_json_and_shape(self):
        self.fails("{", r"^job_json is not valid JSON; reconnect job_json from CCN_TilemapExpand$")
        self.fails("", "not valid JSON")
        self.fails("[" * 200_000, "not valid JSON")  # deep nesting: RecursionError or ValueError, never leaks
        for text in ("[]", '"x"', "7", "null"):
            self.fails(text, r"^job_json must be a JSON object; reconnect")

    def test_base_must_be_null(self):
        for base in ({}, 0, False, "", []):
            self.fails(job_text(base=base), r"^job_json\.base \(tileset reuse\) is not supported by this node yet$")

    def test_tile_size_and_view(self):
        for size in expand.TILE_SIZES:
            LS.parse_job_json(job_text(tile_size=size))
        self.assertEqual(tuple(expand.TILE_SIZES), (8, 16, 24, 32, 48, 64))
        for size in (0, 7, 9, 15, 17, 65, "16", 16.0, True, None, [16]):
            self.fails(job_text(tile_size=size), r"^job_json needs a valid tile_size and view; reconnect")
        for view in ("top-down", "three-quarter", "side-view"):
            LS.parse_job_json(job_text(view=view))
        for view in ("Top-Down", "iso", "", None, 1, ["top-down"]):
            self.fails(job_text(view=view), r"^job_json needs a valid tile_size and view; reconnect")
        self.fails(job_text(tile_size=MISSING), "valid tile_size")
        self.fails(job_text(view=MISSING), "valid tile_size")

    def test_plan_validation(self):
        self.fails(job_text(plan=MISSING), r"^job_json plan failed validation; reconnect")
        self.fails(job_text(plan={"theme": "x", "tiles": [], "transitions": []}), "plan failed validation")
        self.fails(job_text(plan="plan"), "plan failed validation")

    def test_jobs_must_match_plan(self):
        pat = r"^job_json jobs do not match its plan; reconnect job_json from CCN_TilemapExpand$"
        self.fails(job_text(jobs=MISSING), pat)
        self.fails(job_text(jobs=None), pat)
        self.fails(job_text(jobs={}), pat)
        self.fails(job_text(jobs=[]), pat)
        self.fails(job_text(jobs=JOBS[:-1]), pat)
        self.fails(job_text(jobs=JOBS + [JOBS[0]]), pat)
        self.fails(job_text(jobs=list(reversed(JOBS))), pat)
        self.fails(job_text(jobs=JOBS[:1] + ["x"] + JOBS[2:]), pat)
        for key, value in (("tile_id", "zz"), ("category", "object"), ("variant", 1), ("footprint", [2, 2])):
            bad = [dict(JOBS[0], **{key: value})] + JOBS[1:]
            self.fails(job_text(jobs=bad), pat)
        extra = [dict(j, prompt="other prompt", width=512, seed=j["seed"]) for j in JOBS]
        LS.parse_job_json(job_text(jobs=extra))  # only tile_id/category/variant/footprint are compared

    def test_seed_range_taken_from_first_job(self):
        limit = 2**64
        for seed in (0, 1, limit - 1):
            self.assertEqual(LS.parse_job_json(job_text(jobs=[dict(JOBS[0], seed=seed)] + JOBS[1:]))[5], seed)
        msg = r"^job_json job seed must be an integer from 0 to 18446744073709551615; reconnect"
        for seed in (-1, limit, limit + 1, True, 7.0, "7", None):
            self.fails(job_text(jobs=[dict(JOBS[0], seed=seed)] + JOBS[1:]), msg)
        self.fails(job_text(jobs=[{k: v for k, v in JOBS[0].items() if k != "seed"}] + JOBS[1:]), msg)
        bad_later = JOBS[:1] + [dict(JOBS[1], seed=-5)] + JOBS[2:]
        self.assertEqual(LS.parse_job_json(job_text(jobs=bad_later))[5], 7)  # only jobs[0] decides


class BudgetStateTests(unittest.TestCase):
    def test_budget_counts_and_refuses_without_calling(self):
        seen = []

        def chat(messages, model, **kw):
            seen.append((messages, model, kw))
            return f"r{len(seen)}"

        b = LS.LlmBudget(chat, "auto", 2)
        self.assertEqual((b.chat, b.model, b.budget, b.calls), (chat, "auto", 2, 0))
        msgs = [{"role": "user", "content": "hi"}]
        self.assertEqual(b.ask(msgs), "r1")
        self.assertEqual(b.ask(msgs, 99), "r2")
        self.assertEqual(seen[0], (msgs, "auto", {"max_tokens": review.MAX_TOKENS, "timeout": 120}))
        self.assertEqual(seen[1][2], {"max_tokens": 99, "timeout": 120})
        self.assertEqual(review.MAX_TOKENS, 1500)
        with self.assertRaisesRegex(llm_client.LLMProxyError, "^LLM call budget used up$"):
            b.ask(msgs)
        self.assertEqual((b.calls, len(seen)), (2, 2))

    def test_zero_budget_never_calls_and_failures_count(self):
        def boom(*a, **k):
            raise AssertionError("must not be called")

        with self.assertRaises(llm_client.LLMProxyError):
            LS.LlmBudget(boom, "m", 0).ask([])

        def fail(*a, **k):
            raise llm_client.LLMProxyError("down")

        b = LS.LlmBudget(fail, "m", 1)
        with self.assertRaisesRegex(llm_client.LLMProxyError, "down"):
            b.ask([])
        self.assertEqual(b.calls, 1)  # a failing call still spends budget
        with self.assertRaisesRegex(llm_client.LLMProxyError, "budget used up"):
            b.ask([])

    def test_is_passed(self):
        high, low = [{"severity": "high"}], [{"severity": "low"}, {"severity": "medium"}]
        ok = {"pass": True, "score": 80}
        self.assertTrue(LS.is_passed(None, [], 80))
        self.assertFalse(LS.is_passed(None, low, 80))
        self.assertTrue(LS.is_passed(ok, [], 80))
        self.assertTrue(LS.is_passed(ok, low, 80))
        self.assertFalse(LS.is_passed(ok, high, 80))
        self.assertFalse(LS.is_passed(ok, low + high, 80))
        self.assertFalse(LS.is_passed(dict(ok, score=79), [], 80))
        self.assertTrue(LS.is_passed(dict(ok, score=79), [], 79))
        self.assertFalse(LS.is_passed(dict(ok, **{"pass": False}), [], 0))
        self.assertTrue(LS.is_passed({"pass": True, "score": 0}, [], 0))

    def test_round_entry(self):
        info, rev = {"review_mode": "heuristic", "fallback": "x"}, ["b#0", "a#0"]
        issue = {"tile": "a#0", "kind": "seam", "severity": "low", "source": "vision", "fix_hint": "fix", "cell": [0, 0]}
        e = LS.round_entry(2, info, [issue], None, rev)
        self.assertEqual(e, {"review_mode": "heuristic", "fallback": "x", "round": 2, "reverted": ["a#0", "b#0"],
                             "regen": [], "skipped": [],
                             "issues": [{"tile": "a#0", "kind": "seam", "severity": "low", "source": "vision"}]})
        self.assertEqual((info, rev), ({"review_mode": "heuristic", "fallback": "x"}, ["b#0", "a#0"]))
        v = {"pass": False, "score": 41, "summary": "meh", "issues": [issue], "problems": []}
        e = LS.round_entry(1, {}, [], v, [])
        self.assertEqual((e["score"], e["passed"], e["summary"], e["issues"]), (41, False, "meh", []))
        self.assertNotIn("score", LS.round_entry(1, {}, [], {}, []))
        self.assertIsNot(LS.round_entry(1, {}, [], None, [])["regen"], LS.round_entry(1, {}, [], None, [])["regen"])

    def test_build_log(self):
        rounds = [{"round": 1}]
        raw = LS.build_log("passed", 3, "llm", "", ["b#0", "a#0"], rounds)
        self.assertEqual(json.loads(raw), {"format": "ccn-review-log/1", "stop": "passed", "llm_calls": 3,
                                           "layout_source": "llm", "layout_problem": "", "unreviewed": ["a#0", "b#0"],
                                           "rounds": rounds})
        self.assertEqual(list(json.loads(raw)), sorted(json.loads(raw)))
        long = LS.build_log("s", 0, "auto", "p" * 500, [], [])
        self.assertLessEqual(len(json.loads(long)["layout_problem"]), 200)
        self.assertGreater(len(json.loads(long)["layout_problem"]), 150)
        self.assertIn("\u00e4\u00f6", LS.build_log("s", 0, "auto", "\u00e4\u00f6", [], []))  # ensure_ascii=False

    def test_build_job_json(self):
        data, snap_p, attempts = dict(BASE, snap={"old": 1}), {"palette_size": 8, "transitions": True}, {"g0#0": 1}
        snapshot = json.dumps(data, sort_keys=True)
        text = " original  text "
        self.assertIs(LS.build_job_json(text, data, snap_p, 5, attempts, "passed", 0), text)
        raw = LS.build_job_json(text, data, snap_p, 2, attempts, "max_rounds", 3)
        out = json.loads(raw)
        self.assertEqual(out["snap"], snap_p)
        self.assertEqual(out["review"], {"rounds": 2, "attempts": {"g0#0": 1}, "stop": "max_rounds"})
        self.assertEqual({k: v for k, v in out.items() if k not in ("snap", "review")}, BASE)
        self.assertEqual(json.dumps(data, sort_keys=True), snapshot)
        self.assertIn('"review":{"attempts":{"g0#0":1},"rounds":2,"stop":"max_rounds"}', raw)  # compact + sorted
        self.assertLess(raw.index('"jobs"'), raw.index('"plan"'))  # sort_keys
        self.assertIn("\u00e4", LS.build_job_json(text, dict(data, theme="\u00e4"), snap_p, 1, {}, "x", 1))


if __name__ == "__main__":
    unittest.main()
