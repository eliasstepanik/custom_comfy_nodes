"""Tests for tilemap_core.defects: kind table, kinds/hint validation, phrase order, caps, snap merge, amend_job.

Covers every name of ``defects.__all__`` (normal cases, exact ValueError paths, hint-length and PROMPT_MAX boundaries,
determinism, input immutability). The module has no pixel / alpha / palette logic: tile-size cases do not apply.

Residual gaps (documented, not tested):
* PROMPT_MAX is reachable only through ``mock.patch`` (longest real tail ~1.3k chars); patched cases pin the cut logic.
  ``amend_job`` with ``room < 0`` is unreachable with real constants (tail is cut to PROMPT_MAX); only ``room == 0``.
* Only ``fringe`` has a snap override, so the larger-value-wins merge runs on a patched table.
* ``base[:room].strip()`` vs ``base[:room]`` in ``amend_job`` is equivalent (cleaned base never starts with a space).
"""

import copy
import types
import unittest
from unittest import mock

from tilemap_core import defects as D
from tilemap_core import sanitize, snap

KINDS = (
    "style_mismatch", "seam", "fringe", "wrong_content", "not_cut_out", "not_tileable",
    "wrong_scale", "transition", "ground_not_texture", "other",
)
# (prompt keyword, negative keyword) per kind: pins each phrase to its own kind.
KEYWORDS = {
    "style_mismatch": ("pixel art style", "photo"), "seam": ("without any visible seam", "visible edge"),
    "fringe": ("no bright halo", "glow"), "wrong_content": ("clearly recognisable", "tiny subject"),
    "not_cut_out": ("plain flat white background", "cropped object"), "not_tileable": ("all four edges", "vignette"),
    "wrong_scale": ("correct scale", "mismatched scale"), "transition": ("no isolated blobs", "cut-out shape"),
    "ground_not_texture": ("repeating ground pattern", "flat cream background"), "other": ("game-ready", "watermark"),
}


def fixes(kind, key):
    return D.DEFECT_FIXES[kind][key]


class TestTable(unittest.TestCase):
    def test_constants_and_kinds_exact_canonical(self):
        self.assertEqual((D.PROMPT_MAX, D.HINT_MAX), (2000, 200))
        self.assertEqual(D.DEFECT_KINDS, KINDS)

    def test_all_exports_exist(self):
        names = "DEFECT_KINDS DEFECT_FIXES PROMPT_MAX HINT_MAX amendment negative_text snap_overrides amend_job".split()
        self.assertEqual(sorted(D.__all__), sorted(names))
        self.assertTrue(all(hasattr(D, n) for n in names))

    def test_fixes_exact_columns_and_phrases_clean_non_empty_unique(self):
        self.assertEqual(set(D.DEFECT_FIXES), set(KINDS))
        for kind in KINDS:
            self.assertEqual(sorted(D.DEFECT_FIXES[kind]), ["negative", "prompt", "snap"], kind)
        for col in ("prompt", "negative"):
            texts = [fixes(k, col) for k in KINDS]
            self.assertEqual(len(set(texts)), len(KINDS), col)
            for text in texts:
                self.assertTrue(text and sanitize.clean_text(text, D.PROMPT_MAX) == text, text)

    def test_phrases_belong_to_their_kind(self):
        for kind, (pos, neg) in KEYWORDS.items():
            self.assertIn(pos, fixes(kind, "prompt"), kind)
            self.assertIn(neg, fixes(kind, "negative"), kind)

    def test_snap_column_only_fringe_with_valid_snap_keys(self):
        for kind in KINDS:
            expect = {"key_tolerance": 60} if kind == "fringe" else {}
            self.assertEqual(fixes(kind, "snap"), expect, kind)
            self.assertTrue(set(fixes(kind, "snap")) <= set(snap.SNAP_DEFAULTS), kind)
        self.assertGreater(fixes("fringe", "snap")["key_tolerance"], snap.SNAP_DEFAULTS["key_tolerance"])


class TestKindsValidation(unittest.TestCase):
    """``_check_kinds`` through all three kind-taking functions: same errors everywhere."""

    FUNCS = (D.amendment, D.negative_text, D.snap_overrides, lambda k: D.amend_job({"prompt": "p"}, k))

    def assertBadKinds(self, kinds, contains):
        for fn in self.FUNCS:
            with self.assertRaises(ValueError) as cm:
                fn(kinds)
            self.assertIn(contains, str(cm.exception))

    def test_empty_and_non_collection_inputs_rejected(self):
        for empty in ([], (), set(), frozenset(), iter([]), (k for k in ())):
            self.assertBadKinds(empty, "must not be empty")
        for bad in ("seam", "", b"seam", {"seam": 1}, 5, None, 1.5, True):
            self.assertBadKinds(bad, "must be a list, tuple or set")
        with self.assertRaises(ValueError) as cm:  # kinds are checked before the hint
            D.amendment([], None)
        self.assertIn("must not be empty", str(cm.exception))

    def test_unknown_or_non_str_kind_rejected(self):
        class Sub(str):
            pass

        for kinds, shown in ((["bogus"], "'bogus'"), (["seam", "bogus"], "'bogus'"), (["Seam"], "'Seam'"),
                             ([" seam"], "' seam'"), ([None], "None"), ([1], "1"), ([["seam"]], "['seam']"),
                             ([Sub("seam")], "'seam'")):
            self.assertBadKinds(kinds, "unknown defect kind " + shown)

    def test_error_lists_allowed_kinds(self):
        self.assertBadKinds(["bogus"], "allowed: " + str(list(KINDS)))

    def test_accepts_any_non_string_iterable_and_does_not_mutate(self):
        want = D.amendment(["seam", "fringe"])
        for kinds in (("fringe", "seam"), {"seam", "fringe"}, frozenset(["seam", "fringe"]),
                      (k for k in ("fringe", "seam")), iter(["seam", "fringe"]), {"seam": 1, "fringe": 2}.keys()):
            self.assertEqual(D.amendment(kinds), want)
        kinds = ["other", "seam", "seam"]
        D.amendment(kinds), D.negative_text(kinds), D.snap_overrides(kinds)
        self.assertEqual(kinds, ["other", "seam", "seam"])


class TestAmendment(unittest.TestCase):
    def test_single_kind_is_its_phrase_for_every_kind(self):
        for kind in KINDS:
            self.assertEqual(D.amendment([kind]), fixes(kind, "prompt"), kind)

    def test_canonical_order_dedup_and_separator(self):
        want = ", ".join([fixes("seam", "prompt"), fixes("wrong_content", "prompt"), fixes("other", "prompt")])
        self.assertEqual(D.amendment(["other", "seam", "wrong_content"]), want)
        self.assertEqual(D.amendment(["wrong_content", "other", "seam", "seam", "other"]), want)
        every = ", ".join(fixes(k, "prompt") for k in KINDS)
        self.assertEqual((D.amendment(reversed(KINDS)), D.amendment(KINDS)), (every, every))
        self.assertLessEqual(len(every), D.PROMPT_MAX)

    def test_hint_appended_last_after_comma_space(self):
        self.assertEqual(D.amendment(["seam"], "fill the frame"), fixes("seam", "prompt") + ", fill the frame")
        self.assertEqual(D.amendment(["seam"], hint="x"), fixes("seam", "prompt") + ", x")

    def test_hint_is_sanitised(self):
        self.assertEqual(D.amendment(["other"], "  more\t\n  {green} | [grass]\x00 <b>\u202e "),
                         fixes("other", "prompt") + ", more green grass b")

    def test_blank_hint_adds_nothing(self):
        base = fixes("other", "prompt")
        for hint in ("", "   ", "\n\t", "{}|<>", "\x00\u200b"):
            self.assertEqual(D.amendment(["other"], hint), base, repr(hint))

    def test_hint_length_boundary_is_checked_after_cleaning(self):
        base = fixes("other", "prompt")
        self.assertEqual(D.amendment(["other"], "h" * 200), base + ", " + "h" * 200)
        with self.assertRaises(ValueError) as cm:
            D.amendment(["other"], "h" * 201)
        self.assertIn("too long (201 chars, limit 200)", str(cm.exception))
        # raw length 230 but 200 after cleaning (30 banned characters): accepted
        self.assertEqual(D.amendment(["other"], "h" * 200 + "{" * 15 + "}" * 15), base + ", " + "h" * 200)
        # whitespace runs collapse: raw 400 chars, cleaned "a b" -> accepted
        self.assertEqual(D.amendment(["other"], "a" + " " * 398 + "b"), base + ", a b")
        # non-ASCII counts by character, not byte
        self.assertEqual(D.amendment(["other"], "\u00e4" * 200), base + ", " + "\u00e4" * 200)
        with self.assertRaises(ValueError):
            D.amendment(["other"], "\u00e4" * 201)

    def test_non_string_hint_rejected(self):
        for bad in (None, 5, b"hint", ["x"], 1.5):
            with self.assertRaises(ValueError) as cm:
                D.amendment(["seam"], bad)
            self.assertIn("hint must be a string", str(cm.exception))

    def test_cut_at_prompt_max(self):
        with mock.patch.object(D, "PROMPT_MAX", 60):
            out = D.amendment(["style_mismatch", "seam"])
        full = ", ".join([fixes("style_mismatch", "prompt"), fixes("seam", "prompt")])
        self.assertEqual(out, full[:60].rstrip())
        self.assertEqual(len(out), 60)
        with mock.patch.object(D, "PROMPT_MAX", len(full)):
            self.assertEqual(D.amendment(["style_mismatch", "seam"]), full)
        with mock.patch.object(D, "PROMPT_MAX", len(full) - 1):
            self.assertEqual(D.amendment(["style_mismatch", "seam"]), full[:-1])


class TestNegativeText(unittest.TestCase):
    def test_single_kind_is_its_negative_for_every_kind(self):
        for kind in KINDS:
            self.assertEqual(D.negative_text([kind]), fixes(kind, "negative"), kind)

    def test_canonical_order_dedup_and_separator(self):
        want = ", ".join([fixes("seam", "negative"), fixes("fringe", "negative"), fixes("other", "negative")])
        self.assertEqual(D.negative_text(["other", "fringe", "seam", "fringe"]), want)
        every = D.negative_text(reversed(KINDS))
        self.assertEqual(every, ", ".join(fixes(k, "negative") for k in KINDS))
        self.assertLessEqual(len(every), D.PROMPT_MAX)

    def test_cut_at_prompt_max(self):
        full = ", ".join([fixes("style_mismatch", "negative"), fixes("seam", "negative")])
        with mock.patch.object(D, "PROMPT_MAX", 40):
            self.assertEqual(D.negative_text(["seam", "style_mismatch"]), full[:40].rstrip())
        with mock.patch.object(D, "PROMPT_MAX", len(full)):
            self.assertEqual(D.negative_text(["seam", "style_mismatch"]), full)


class TestSnapOverrides(unittest.TestCase):
    def test_only_fringe_overrides(self):
        for kind in KINDS:
            self.assertEqual(D.snap_overrides([kind]), {"key_tolerance": 60} if kind == "fringe" else {}, kind)
        self.assertEqual(D.snap_overrides(KINDS), {"key_tolerance": 60})
        self.assertEqual(D.snap_overrides(["seam", "other"]), {})

    def test_returns_fresh_dict_not_the_table(self):
        for kind, want in (("fringe", {"key_tolerance": 60}), ("seam", {})):
            out = D.snap_overrides([kind])
            out["key_tolerance"] = 1
            out["extra"] = 2
            self.assertEqual(fixes(kind, "snap"), want)
            self.assertEqual(D.snap_overrides([kind]), want)

    def test_shared_key_larger_value_wins_whichever_comes_first(self):
        # canonical order is seam, then fringe (60): the patched seam value is below, equal to and above it.
        for seam_val, want in ((80, 80), (30, 60), (60, 60)):
            patch = {"seam": {**D.DEFECT_FIXES["seam"], "snap": {"key_tolerance": seam_val}}}
            with mock.patch.dict(D.DEFECT_FIXES, patch):
                self.assertEqual(D.snap_overrides(["fringe", "seam"]), {"key_tolerance": want}, seam_val)

    def test_distinct_keys_are_merged(self):
        table = {"seam": {**D.DEFECT_FIXES["seam"], "snap": {"seam_blend": 0.5, "key_tolerance": 10}}}
        with mock.patch.dict(D.DEFECT_FIXES, table):
            self.assertEqual(D.snap_overrides(["seam", "fringe"]), {"seam_blend": 0.5, "key_tolerance": 60})


class TestAmendJob(unittest.TestCase):
    JOB = {"prompt": "a mossy cobblestone ground", "seed": 7, "width": 512, "snap": {"palette": [1, 2]}, "tags": ["a"]}

    def test_prompt_is_base_comma_tail_and_other_keys_unchanged(self):
        out = D.amend_job(self.JOB, ["not_tileable", "seam"], "no vignette")
        tail = D.amendment(["seam", "not_tileable"], "no vignette")
        self.assertEqual(out["prompt"], "a mossy cobblestone ground, " + tail)
        self.assertEqual({**out, "prompt": 0}, {**self.JOB, "prompt": 0})
        self.assertIs(type(out), dict)

    def test_input_not_mutated_and_deep_copied(self):
        job = copy.deepcopy(self.JOB)
        out = D.amend_job(job, ["fringe"])
        self.assertEqual(job, self.JOB)
        self.assertIsNot(out, job)
        self.assertIsNot(out["snap"], job["snap"])
        self.assertIsNot(out["snap"]["palette"], job["snap"]["palette"])
        out["snap"]["palette"].append(3)
        out["tags"].append("b")
        self.assertEqual(job, self.JOB)

    def test_mapping_job_accepted_returns_plain_dict(self):
        out = D.amend_job(types.MappingProxyType({"prompt": "x", "seed": 1}), ["other"])
        self.assertIs(type(out), dict)
        self.assertEqual(out, {"prompt": "x, " + fixes("other", "prompt"), "seed": 1})

    def test_base_prompt_is_sanitised(self):
        out = D.amend_job({"prompt": "  grass \n {lush}\t|field\x00 "}, ["other"])
        self.assertEqual(out["prompt"], "grass lush field, " + fixes("other", "prompt"))

    def test_empty_or_blank_base_prompt_gives_tail_only(self):
        tail = fixes("other", "prompt")
        for base in ("", "   ", "\n\t", "{}|", "\x00"):
            self.assertEqual(D.amend_job({"prompt": base}, ["other"])["prompt"], tail, repr(base))

    def test_repeated_amendment_stacks_and_is_deterministic(self):
        once = D.amend_job(self.JOB, ["seam"])
        self.assertEqual(once, D.amend_job(self.JOB, ["seam"]))
        twice = D.amend_job(once, ["seam"])
        self.assertEqual(twice["prompt"], once["prompt"] + ", " + fixes("seam", "prompt"))

    def test_long_base_is_shortened_tail_kept(self):
        tail = D.amendment(["seam"], "keep me")
        room = D.PROMPT_MAX - len(tail) - 2
        self.assertGreater(room, 100)
        for n, kept in ((room - 1, room - 1), (room, room), (room + 1, room), (room + 500, room), (5000, room)):
            out = D.amend_job({"prompt": "a" * n}, ["seam"], "keep me")["prompt"]
            self.assertEqual(out, "a" * kept + ", " + tail, n)
            self.assertLessEqual(len(out), D.PROMPT_MAX)
        self.assertEqual(len(D.amend_job({"prompt": "a" * room}, ["seam"], "keep me")["prompt"]), D.PROMPT_MAX)

    def test_cut_prompt_is_rstripped(self):
        tail = D.amendment(["seam"])
        room = D.PROMPT_MAX - len(tail) - 2
        base = "a" * (room - 1) + " " + "b" * 50
        self.assertEqual(D.amend_job({"prompt": base}, ["seam"])["prompt"], "a" * (room - 1) + ", " + tail)

    def test_base_cut_to_nothing_when_tail_fills_prompt_max(self):
        tail = D.amendment(["seam"])
        with mock.patch.object(D, "PROMPT_MAX", len(tail) + 2):
            self.assertEqual(D.amend_job({"prompt": "base"}, ["seam"])["prompt"], tail)
        with mock.patch.object(D, "PROMPT_MAX", len(tail) + 3):
            self.assertEqual(D.amend_job({"prompt": "base"}, ["seam"])["prompt"], "b, " + tail)

    def test_bad_job_rejected(self):
        for bad in (None, "prompt", ["prompt"], 5, {}, {"seed": 1}, {"prompt": None}, {"prompt": 5}, {"prompt": b"x"},
                    {"prompt": ["x"]}):
            with self.assertRaises(ValueError) as cm:
                D.amend_job(bad, ["seam"])
            self.assertIn("job must be a mapping with a string 'prompt'", str(cm.exception))

    def test_bad_kinds_and_hint_rejected(self):
        for kinds, hint in (([], ""), (["bogus"], ""), ("seam", ""), (["seam"], None), (["seam"], "h" * 201)):
            with self.assertRaises(ValueError):
                D.amend_job(self.JOB, kinds, hint)
        self.assertEqual(self.JOB["prompt"], "a mossy cobblestone ground")


if __name__ == "__main__":
    unittest.main()
