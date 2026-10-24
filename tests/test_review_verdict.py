"""Tests for tilemap_core.review verdict handling: strict parse_verdict and merge_issues (shared helpers: test_review)."""

import copy
import json
import unittest

from tilemap_core import review as R
from tests.test_review import ISSUE, KNOWN, ROWS, ReviewTestCase, verdict, with_issue_field


class TestParseVerdict(ReviewTestCase):
    def test_valid_verdict(self):
        v = R.parse_verdict(verdict(), KNOWN)
        self.assertEqual((v["pass"], v["score"], v["problems"]), (False, 62, []))
        self.assertEqual(v["summary"], "ground tiles are cut-out blobs")
        self.assertEqual(v["issues"], [{"tile": "grass#0", "cell": [3, 2], "kind": "ground_not_texture",
                                        "severity": "high", "fix_hint": "fill the whole frame",
                                        "source": "vision"}])

    def test_fenced_and_prose_wrapped_json(self):
        self.assertEqual(R.parse_verdict("Here:\n```json\n" + verdict() + "\n```", KNOWN)["score"], 62)
        self.assertEqual(R.parse_verdict("verdict -> " + verdict() + " done", KNOWN)["score"], 62)

    def test_known_tiles_may_be_any_iterable(self):
        self.assertEqual(len(R.parse_verdict(verdict(), iter(KNOWN))["issues"]), 1)
        self.assertEqual(len(R.parse_verdict(verdict(), sorted(KNOWN))["issues"]), 1)
        self.assertEqual(len(R.parse_verdict(verdict(), R.tile_refs(ROWS))["issues"]), 1)

    def test_rejects_unknown_top_level_key(self):
        self.assertRejects(R.parse_verdict, verdict(extra=1), KNOWN, contains="extra")

    def test_rejects_unknown_issue_key(self):
        self.assertRejects(R.parse_verdict, with_issue_field(why="x"), KNOWN, contains="why")

    def test_rejects_missing_required_keys(self):
        self.assertRejects(R.parse_verdict, '{"pass": false, "score": 3}', KNOWN, contains="issues")
        self.assertRejects(R.parse_verdict, '{"score": 3, "issues": []}', KNOWN, contains="pass")
        self.assertRejects(R.parse_verdict, '{"pass": false, "issues": []}', KNOWN, contains="score")
        d = json.loads(verdict())
        del d["issues"][0]["severity"]
        self.assertRejects(R.parse_verdict, json.dumps(d), KNOWN, contains="severity")

    def test_optional_fields_default(self):
        v = R.parse_verdict('{"pass": false, "score": 3, "issues": []}', KNOWN)
        self.assertEqual((v["summary"], v["issues"]), ("", []))
        d = json.loads(verdict())
        del d["issues"][0]["cell"], d["issues"][0]["fix_hint"]
        issue = R.parse_verdict(json.dumps(d), KNOWN)["issues"][0]
        self.assertEqual((issue["cell"], issue["fix_hint"]), (None, ""))

    def test_unknown_tile_dropped_with_problem_note(self):
        d = json.loads(verdict())
        d["issues"].append({"tile": "ghost#9", "kind": "seam", "severity": "low"})
        v = R.parse_verdict(json.dumps(d), KNOWN)
        self.assertEqual([i["tile"] for i in v["issues"]], ["grass#0"])
        self.assertEqual(len(v["problems"]), 1)
        self.assertIn("issues[1]", v["problems"][0])
        self.assertIn("ghost", v["problems"][0])

    def test_tile_reference_is_stripped_and_matched_exactly(self):
        v = R.parse_verdict(with_issue_field(tile="  grass  "), KNOWN)
        self.assertEqual(v["issues"][0]["tile"], "grass")
        v = R.parse_verdict(with_issue_field(tile="grass#1"), KNOWN)  # variant not in the catalogue
        self.assertEqual((v["issues"], len(v["problems"])), ([], 1))

    def test_bad_tile_field_is_error_not_drop(self):
        for bad in ("", "   ", "x" * 81, 5, None):
            self.assertRejects(R.parse_verdict, with_issue_field(tile=bad), KNOWN)

    def test_score_never_clamped_and_must_be_int(self):
        for bad in (101, -1, 1000, 62.5, True, False, "62", None):
            self.assertRejects(R.parse_verdict, verdict(score=bad), KNOWN)
        self.assertEqual(R.parse_verdict(verdict(score=0), KNOWN)["score"], 0)
        self.assertEqual(R.parse_verdict(verdict(score=100, **{"pass": True, "issues": []}), KNOWN)["score"], 100)

    def test_pass_must_be_bool(self):
        for bad in (1, 0, "true", None):
            self.assertRejects(R.parse_verdict, verdict(**{"pass": bad}), KNOWN)

    def test_pass_true_needs_pass_score(self):
        ok = {"pass": True, "issues": []}
        self.assertTrue(R.parse_verdict(verdict(score=80, **ok), KNOWN)["pass"])
        self.assertRejects(R.parse_verdict, verdict(score=79, **ok), KNOWN, contains="80")
        self.assertTrue(R.parse_verdict(verdict(score=50, **ok), KNOWN, pass_score=50)["pass"])
        self.assertRejects(R.parse_verdict, verdict(score=90, **ok), KNOWN, pass_score=91)

    def test_pass_true_with_high_issue_rejected_but_medium_ok(self):
        self.assertRejects(R.parse_verdict, verdict(**{"pass": True, "score": 95}), KNOWN, contains="high")
        medium = [dict(ISSUE, severity="medium")]
        self.assertTrue(R.parse_verdict(verdict(**{"pass": True, "score": 95, "issues": medium}), KNOWN)["pass"])

    def test_high_issue_on_dropped_tile_does_not_block_pass(self):
        ghost = [dict(ISSUE, tile="ghost")]
        v = R.parse_verdict(verdict(**{"pass": True, "score": 95, "issues": ghost}), KNOWN)
        self.assertEqual((v["pass"], v["issues"], len(v["problems"])), (True, [], 1))

    def test_pass_false_needs_reason(self):
        self.assertRejects(R.parse_verdict, verdict(score=90, issues=[]), KNOWN, contains="false needs")
        self.assertFalse(R.parse_verdict(verdict(score=79, issues=[]), KNOWN)["pass"])
        self.assertFalse(R.parse_verdict(verdict(score=90), KNOWN)["pass"])  # high score but an issue
        self.assertRejects(R.parse_verdict, verdict(score=80, issues=[]), KNOWN)  # exactly pass_score, no issue
        self.assertFalse(R.parse_verdict(verdict(score=80, issues=[]), KNOWN, pass_score=81)["pass"])

    def test_pass_false_reason_judged_after_unknown_tile_drop(self):
        d = json.loads(verdict(score=90))
        d["issues"] = [{"tile": "ghost", "kind": "seam", "severity": "low"}]
        self.assertRejects(R.parse_verdict, json.dumps(d), KNOWN)

    def test_pass_score_validated(self):
        for bad in (-1, 101, 80.0, True, "80", None):
            self.assertRejects(R.parse_verdict, verdict(), KNOWN, pass_score=bad)

    def test_kind_severity_cell_and_hint_validation(self):
        for field, bad in (("kind", "weird"), ("kind", 3), ("severity", "urgent"), ("severity", "HIGH"),
                           ("cell", [1]), ("cell", [1, -1]), ("cell", [1, 256]), ("cell", [1.5, 1]),
                           ("cell", [True, 1]), ("cell", "1,2"), ("fix_hint", 5), ("fix_hint", "x" * 201)):
            self.assertRejects(R.parse_verdict, with_issue_field(**{field: bad}), KNOWN)

    def test_hint_boundary_and_cleaning(self):
        self.assertEqual(len(R.parse_verdict(with_issue_field(fix_hint="x" * 200), KNOWN)["issues"][0]["fix_hint"]),
                         200)
        cleaned = R.parse_verdict(with_issue_field(fix_hint="  a<b>{c}\x00 d\n e  "), KNOWN)["issues"][0]["fix_hint"]
        self.assertEqual(cleaned, "abc d e")

    def test_cell_bounds_accepted(self):
        for cell in ([0, 0], [255, 255]):
            self.assertEqual(R.parse_verdict(with_issue_field(cell=cell), KNOWN)["issues"][0]["cell"], cell)

    def test_all_contract_kinds_and_severities_accepted(self):
        from tilemap_core.defects import DEFECT_KINDS
        for kind in DEFECT_KINDS:
            for sev in R.SEVERITIES:
                issue = R.parse_verdict(with_issue_field(kind=kind, severity=sev), KNOWN)["issues"][0]
                self.assertEqual((issue["kind"], issue["severity"], issue["source"]), (kind, sev, "vision"))

    def test_max_issues_boundary(self):
        one = {"tile": "grass", "kind": "seam", "severity": "low"}
        ok = R.parse_verdict(verdict(issues=[one] * R.MAX_ISSUES), KNOWN)
        self.assertEqual(len(ok["issues"]), R.MAX_ISSUES)
        self.assertEqual(R.MAX_ISSUES, 32)
        self.assertRejects(R.parse_verdict, verdict(issues=[one] * (R.MAX_ISSUES + 1)), KNOWN, contains="32")

    def test_unknown_tile_issues_count_toward_max(self):
        ghost = {"tile": "ghost", "kind": "seam", "severity": "low"}
        self.assertRejects(R.parse_verdict, verdict(issues=[ghost] * 33), KNOWN)

    def test_summary_cut_and_cleaned(self):
        v = R.parse_verdict(verdict(summary="word " * 200), KNOWN)
        self.assertLessEqual(len(v["summary"]), R.SUMMARY_MAX)
        self.assertEqual(len(v["summary"]), 299)  # 300 cut, trailing space stripped
        self.assertEqual(R.parse_verdict(verdict(summary="a\x00<b>\nc"), KNOWN)["summary"], "ab c")
        self.assertRejects(R.parse_verdict, verdict(summary=5), KNOWN)

    def test_garbage_replies_rejected(self):
        for txt in ("", "no json", "[1]", None, 5,
                    '{"pass": true, "pass": false, "score": 1, "issues": []}',  # duplicate key
                    '{"pass": false, "score": NaN, "issues": []}',
                    '{"pass": false, "score": 1, "issues": "x"}',
                    '{"pass": false, "score": 1, "issues": [5]}',
                    '{"pass": false, "score": 1, "issues": {}}'):
            self.assertRejects(R.parse_verdict, txt, KNOWN)

    def test_known_tiles_type_checked(self):
        for bad in ("grass#0", b"grass#0", None, 5, [1, 2]):
            self.assertRejects(R.parse_verdict, verdict(), bad, contains="known_tiles")

    def test_error_text_hygiene_on_hostile_reply(self):
        hostile = '{"pass": false, "score": 1, "issues": [], "k\\u0000\\n\\u202e' + "z" * 500 + '": 1}'
        text = self.assertRejects(R.parse_verdict, hostile, KNOWN)
        self.assertTrue(all(c.isprintable() for c in text), repr(text))

    def test_fail_helper_sanitises_single_line_and_caps_length(self):
        err = R._fail("a\nb\x00c", "x  y\t\u202e" + "z" * 400)
        text = str(err)
        self.assertIsInstance(err, ValueError)
        self.assertTrue(text.startswith("a b c: x y"), repr(text[:20]))
        self.assertEqual(len(text), 300)
        self.assertTrue(text.isprintable())
        self.assertEqual(str(R._fail("p", "r")), "p: r")


class TestMergeIssues(ReviewTestCase):
    V = [{"tile": "grass#0", "kind": "seam", "severity": "low"},
         {"tile": "tree#1", "kind": "fringe", "severity": "medium", "fix_hint": "v-hint"}]
    H = [{"tile": "grass#0", "kind": "seam", "severity": "high", "fix_hint": "hh", "score": 3.0},
         {"tile": "tree#1", "kind": "fringe", "severity": "low", "fix_hint": "x"},
         {"tile": "tree#1", "kind": "not_cut_out", "severity": "medium", "fix_hint": "y"}]

    def setUp(self):
        self.out = {(i["tile"], i["kind"]): i for i in R.merge_issues(self.V, self.H)}

    def test_dedupe_by_tile_and_kind(self):
        self.assertEqual(len(self.out), 3)
        self.assertEqual(len(R.merge_issues(self.V, self.H)), 3)
        diff_kind = R.merge_issues([{"tile": "a", "kind": "seam", "severity": "low"}],
                                   [{"tile": "a", "kind": "fringe", "severity": "low"}])
        self.assertEqual(len(diff_kind), 2)
        diff_tile = R.merge_issues([{"tile": "a", "kind": "seam", "severity": "low"}],
                                   [{"tile": "b", "kind": "seam", "severity": "low"}])
        self.assertEqual(len(diff_tile), 2)

    def test_higher_severity_wins_and_source_both(self):
        win = self.out[("grass#0", "seam")]  # heuristic high beats vision low
        self.assertEqual((win["severity"], win["source"], win["fix_hint"], win["score"]), ("high", "both", "hh", 3.0))
        keep = self.out[("tree#1", "fringe")]  # vision medium beats heuristic low
        self.assertEqual((keep["severity"], keep["source"], keep["fix_hint"]), ("medium", "both", "v-hint"))

    def test_tie_keeps_vision_entry(self):
        tie = R.merge_issues([{"tile": "a", "kind": "seam", "severity": "high", "fix_hint": "V"}],
                             [{"tile": "a", "kind": "seam", "severity": "high", "fix_hint": "H"}])
        self.assertEqual((tie[0]["fix_hint"], tie[0]["source"]), ("V", "both"))

    def test_single_source_labels_and_vision_first_order(self):
        self.assertEqual(self.out[("tree#1", "not_cut_out")]["source"], "heuristic")
        only_v = R.merge_issues([{"tile": "a", "kind": "seam", "severity": "low", "source": "bogus"}], [])
        self.assertEqual(only_v[0]["source"], "vision")  # source is always set by the merge
        order = R.merge_issues(self.V, self.H)
        self.assertEqual([(i["tile"], i["kind"]) for i in order[:2]], [("grass#0", "seam"), ("tree#1", "fringe")])

    def test_inputs_not_mutated_and_new_dicts(self):
        v, h = copy.deepcopy(self.V), copy.deepcopy(self.H)
        out = R.merge_issues(v, h)
        self.assertEqual((v, h), (self.V, self.H))
        self.assertTrue(all(o is not i for o in out for i in v + h))

    def test_empty_and_tuple_inputs(self):
        self.assertEqual(R.merge_issues([], []), [])
        self.assertEqual(len(R.merge_issues(tuple(self.V), tuple(self.H))), 3)

    def test_malformed_issues_rejected(self):
        good = {"tile": "a", "kind": "seam", "severity": "low"}
        for bad in ("x", None, [5], [{"tile": "a", "kind": "zzz", "severity": "low"}],
                    [{"tile": "a", "kind": "seam", "severity": "urgent"}],
                    [{"tile": 3, "kind": "seam", "severity": "low"}], [{"kind": "seam", "severity": "low"}]):
            self.assertRejects(R.merge_issues, bad, [])
            self.assertRejects(R.merge_issues, [good], bad)


if __name__ == "__main__":
    unittest.main()
