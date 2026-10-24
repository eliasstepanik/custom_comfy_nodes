"""Tests for ``plan_schema.validate_extension`` (tileset reuse, plan row D15b; unittest, stdlib only).

Covers: the ``(merged_plan, new_keys)`` result and its key order (category order, plan order, variants
counted from the old total), every merge rule (same theme, known id may only raise ``variants``, no
redefinition of category/name/footprint, new transitions only between ground ids, no repeated pair),
the empty-extension error, ``max_tiles`` / ``MAX_JOBS`` / 4-pair limits on the merged plan, the exact
exception types (``ValueError`` for ``max_tiles``, ``PlanError`` for everything else), input-not-mutated
and determinism. ``validate_plan`` / ``extract_json`` themselves are covered by ``tests/test_plan.py``.

Residual gaps (not covered): the ``RecursionError`` and non-string-keys paths of the underlying
``_tile`` helper are only reached through ``validate_plan`` tests; the exact wording of messages is pinned
only where this file greps for it with ``assertRaisesRegex``.
"""

import copy
import unittest

from tilemap_core.plan_schema import MAX_JOBS, PlanError, validate_extension, validate_plan


def g(i, **kw):
    return {"id": f"g{i}", "name": f"Ground {i}", "category": "ground",
            "description": f"visual description of ground {i}", **kw}


def obj(i, **kw):
    return {"id": f"o{i}", "name": f"Object {i}", "category": "object",
            "description": f"visual description of object {i}", **kw}


def ovl(i, **kw):
    return {"id": f"v{i}", "name": f"Overlay {i}", "category": "overlay",
            "description": f"visual description of overlay {i}", **kw}


def existing():
    """Valid plan: g0 (1 variant), g1 (2 variants), o0 (1 variant, 2x3), transition g0-g1; 4 variants."""
    return validate_plan({"theme": "Forest", "tiles": [g(0), g(1, variants=2), obj(0, footprint=[2, 3])],
                          "transitions": [["g0", "g1"]]}, 8)


def add(*tiles, transitions=None, theme="Forest"):
    out = {"theme": theme, "tiles": list(tiles)}
    if transitions is not None:
        out["transitions"] = transitions
    return out


class Base(unittest.TestCase):
    def ok(self, addition, max_tiles=24, base=None):
        return validate_extension(existing() if base is None else base, addition, max_tiles)

    def bad(self, addition, regex, max_tiles=24, base=None):
        with self.assertRaisesRegex(PlanError, regex):
            validate_extension(existing() if base is None else base, addition, max_tiles)


class ResultTests(Base):
    def test_new_tile_appended_and_keys(self):
        merged, keys = self.ok(add(g(2, variants=2)))
        self.assertIsInstance(merged, dict)
        self.assertEqual([t["id"] for t in merged["tiles"]], ["g0", "g1", "o0", "g2"])
        self.assertEqual(merged["tiles"][3]["variants"], 2)
        self.assertEqual(keys, ["g2#0", "g2#1"])
        self.assertEqual(merged["theme"], "Forest")
        self.assertEqual(merged["transitions"], [["g0", "g1"]])

    def test_result_is_a_tuple_of_plan_and_list(self):
        res = self.ok(add(g(2)))
        self.assertIs(type(res), tuple)
        self.assertEqual(len(res), 2)
        self.assertIs(type(res[1]), list)

    def test_raising_variants_yields_only_added_keys(self):
        merged, keys = self.ok(add(g(1, variants=4)))
        self.assertEqual(keys, ["g1#2", "g1#3"])
        self.assertEqual(merged["tiles"][1]["variants"], 4)
        merged, keys = self.ok(add(g(0, variants=2)))
        self.assertEqual(keys, ["g0#1"])

    def test_keys_ordered_by_category_then_plan_order(self):
        merged, keys = self.ok(add(ovl(0, variants=2), obj(1), g(2), obj(0, variants=2), g(1, variants=3)))
        self.assertEqual(keys, ["g1#2", "g2#0", "o0#1", "o1#0", "v0#0", "v0#1"])
        self.assertEqual([t["id"] for t in merged["tiles"]], ["g0", "g1", "o0", "v0", "o1", "g2"])

    def test_keys_count_equals_variant_growth(self):
        base = existing()
        merged, keys = self.ok(add(g(2, variants=3), g(1, variants=4), ovl(0)))
        before = sum(t["variants"] for t in base["tiles"])
        after = sum(t["variants"] for t in merged["tiles"])
        self.assertEqual((before, after, len(keys), len(set(keys))), (4, 4 + 3 + 2 + 1, 6, 6))

    def test_unchanged_tiles_keep_all_fields(self):
        base = existing()
        merged, _ = self.ok(add(g(2)))
        self.assertEqual(merged["tiles"][:3], base["tiles"])

    def test_raised_tile_keeps_old_description(self):
        edited = g(1, variants=3)
        edited["description"] = "completely different description text"
        merged, _ = self.ok(add(edited))
        self.assertEqual(merged["tiles"][1]["description"], existing()["tiles"][1]["description"])

    def test_merged_plan_is_normalised_and_valid(self):
        merged, _ = self.ok(add(g(2, evil="x"), obj(2, footprint=[2, 2]), transitions=[["g1", "g2"]]))
        self.assertEqual(validate_plan(merged, len(merged["tiles"])), merged)
        self.assertNotIn("evil", merged["tiles"][3])

    def test_new_ids_are_slugged_and_defaults_applied(self):
        merged, keys = self.ok(add({**g(2), "id": "Dark Soil!"}))
        self.assertEqual((merged["tiles"][3]["id"], merged["tiles"][3]["variants"]), ("dark_soil", 1))
        self.assertEqual(merged["tiles"][3]["footprint"], [1, 1])
        self.assertEqual(keys, ["dark_soil#0"])

    def test_existing_id_matched_after_slugging_and_name_cleaning(self):
        merged, keys = self.ok(add({**g(1, variants=3), "id": "G1", "name": "  Ground\n1 "}))
        self.assertEqual(keys, ["g1#2"])
        self.assertEqual(len(merged["tiles"]), 3)

    def test_footprint_omitted_or_equal_is_not_a_redefinition(self):
        self.ok(add(obj(0, variants=2)))
        self.ok(add(obj(0, variants=2, footprint=[2, 3])))
        self.ok(add(obj(0, variants=2, footprint=(2, 3))))

    def test_theme_compared_after_cleaning(self):
        merged, _ = self.ok(add(g(2), theme="  Forest\n"))
        self.assertEqual(merged["theme"], "Forest")

    def test_none_empty_and_missing_tiles_with_new_transition(self):
        base = validate_plan({"theme": "Forest", "tiles": [g(0), g(1), g(2)], "transitions": []}, 8)
        for tiles in (None, []):
            with self.subTest(tiles=tiles):
                merged, keys = validate_extension(base, {"theme": "Forest", "tiles": tiles,
                                                         "transitions": [["g0", "g1"]]}, 8)
                self.assertEqual((keys, merged["transitions"]), ([], [["g0", "g1"]]))
        merged, keys = validate_extension(base, {"theme": "Forest", "transitions": [["g1", "g2"]]}, 8)
        self.assertEqual((keys, merged["transitions"]), ([], [["g1", "g2"]]))

    def test_merged_never_aliases_inputs(self):
        base, addition = existing(), add(g(2), transitions=[["g1", "g2"]])
        before_b, before_a = copy.deepcopy(base), copy.deepcopy(addition)
        merged, keys = validate_extension(base, addition, 24)
        self.assertEqual((base, addition), (before_b, before_a))
        merged["tiles"][0]["variants"] = 4
        merged["tiles"][1]["footprint"].append(9)
        merged["transitions"].append(["x", "y"])
        keys.append("zzz")
        self.assertEqual((base, addition), (before_b, before_a))

    def test_deterministic(self):
        a = validate_extension(existing(), add(g(2, variants=2), g(1, variants=3), transitions=[["g1", "g2"]]), 24)
        b = validate_extension(existing(), add(g(2, variants=2), g(1, variants=3), transitions=[["g1", "g2"]]), 24)
        self.assertEqual(a, b)


class TransitionTests(Base):
    def test_new_pairs_appended_after_existing(self):
        merged, _ = self.ok(add(g(2), transitions=[["g2", "g0"], ["g1", "g2"]]))
        self.assertEqual(merged["transitions"], [["g0", "g1"], ["g2", "g0"], ["g1", "g2"]])

    def test_pair_between_existing_grounds_only(self):
        base = validate_plan({"theme": "Forest", "tiles": [g(0), g(1), g(2)], "transitions": [["g0", "g1"]]}, 8)
        merged, keys = validate_extension(base, add(transitions=[["g1", "g2"]]), 8)
        self.assertEqual((keys, merged["transitions"]), ([], [["g0", "g1"], ["g1", "g2"]]))

    def test_repeated_pair_rejected_in_either_order(self):
        self.bad(add(g(2), transitions=[["g0", "g1"]]), r"transitions\[0\]: pair 'g0', 'g1' already exists")
        self.bad(add(g(2), transitions=[["g1", "g0"]]), r"transitions\[0\]: pair 'g1', 'g0' already exists")
        self.bad(add(g(2), transitions=[["g1", "g2"], ["g1", "g0"]]), r"transitions\[1\]: pair .* already exists")

    def test_duplicate_inside_addition_rejected(self):
        self.bad(add(g(2), transitions=[["g1", "g2"], ["g2", "g1"]]), r"transitions\[1\]: duplicate pair")

    def test_unknown_non_ground_and_same_id_rejected(self):
        self.bad(add(g(2), transitions=[["g2", "nope"]]), r"transitions\[0\]: unknown tile id 'nope'")
        self.bad(add(g(2), transitions=[["g2", "o0"]]), r"transitions\[0\]: tile 'o0' is not a ground tile")
        self.bad(add(ovl(0), transitions=[["g0", "v0"]]), r"tile 'v0' is not a ground tile")
        self.bad(add(g(2), transitions=[["g2", "g2"]]), r"needs two different ids")

    def test_new_tile_must_be_in_same_addition(self):
        self.bad(add(transitions=[["g0", "g2"]]), r"unknown tile id 'g2'")

    def test_bad_transition_shapes(self):
        self.bad(add(g(2), transitions="g0"), r"transitions: must be a list")
        self.bad(add(g(2), transitions=[["g0"]]), r"transitions\[0\]: must be a pair")
        self.bad(add(g(2), transitions=[["g0", 5]]), r"transitions\[0\]\[1\]: must be a string identifier")

    def test_more_than_four_new_pairs_rejected(self):
        grounds = [g(i) for i in range(2, 7)]
        pairs = [[f"g{i}", f"g{i + 1}"] for i in range(2, 6)] + [["g0", "g2"]]
        self.bad(add(*grounds, transitions=pairs), r"transitions: at most 4 pairs, got 5")

    def test_merged_pair_limit_is_four(self):
        grounds = [g(i) for i in range(2, 6)]
        merged, _ = self.ok(add(*grounds, transitions=[["g1", "g2"], ["g2", "g3"], ["g3", "g4"]]))
        self.assertEqual(len(merged["transitions"]), 4)
        self.bad(add(*grounds, transitions=[["g1", "g2"], ["g2", "g3"], ["g3", "g4"], ["g4", "g5"]]),
                 r"transitions: at most 4 pairs, got 5")


class RuleTests(Base):
    def test_theme_rules(self):
        self.bad(add(g(2), theme="Desert"), r"theme: must equal the existing theme 'Forest'")
        self.bad(add(g(2), theme="forest"), r"theme: must equal the existing theme")
        self.bad({"tiles": [g(2)]}, r"theme: must equal the existing theme 'Forest'")
        self.bad(add(g(2), theme=5), r"theme: must be a string")
        self.bad(add(g(2), theme="{}"), r"theme: needs at least 1 usable characters")

    def test_redefinition_rejected(self):
        self.bad(add({**g(0), "category": "overlay", "variants": 2}), r"tiles\[0\]: cannot redefine category, name or footprint of 'g0'")
        self.bad(add({**g(0), "name": "Other", "variants": 2}), r"tiles\[0\]: cannot redefine .* 'g0'")
        self.bad(add(obj(0, variants=2, footprint=[1, 1])), r"cannot redefine .* 'o0'")
        self.bad(add(obj(0, variants=2, footprint=[3, 2])), r"cannot redefine .* 'o0'")

    def test_variants_must_grow(self):
        self.bad(add(g(1, variants=2)), r"tiles\[0\]\.variants: 'g1' has 2; give a larger total")
        self.bad(add(g(1, variants=1)), r"tiles\[0\]\.variants: 'g1' has 2")
        self.bad(add(g(0)), r"tiles\[0\]\.variants: 'g0' has 1")
        self.bad(add(g(1, variants=5)), r"tiles\[0\]\.variants: must be from 1 to 4")

    def test_duplicate_id_in_addition(self):
        self.bad(add(g(2), g(2)), r"tiles\[1\]\.id: duplicate id 'g2'; ids must be unique")
        self.bad(add(g(1, variants=3), g(1, variants=4)), r"tiles\[1\]\.id: duplicate id 'g1'")

    def test_invalid_new_tile_names_the_path(self):
        self.bad(add(g(2), {**g(3), "category": "water"}), r"tiles\[1\]\.category: must be one of ground, object, overlay")
        self.bad(add({"id": "x", "name": "X", "category": "ground"}), r"tiles\[0\]\.description: missing")
        self.bad(add(g(2, footprint=[2, 2])), r"tiles\[0\]\.footprint: only category 'object'")
        self.bad(add("g2"), r"tiles\[0\]: must be an object")

    def test_empty_extension_rejected(self):
        for addition in (add(), {"theme": "Forest"}, {"theme": "Forest", "tiles": None, "transitions": []},
                         add(transitions=[])):
            with self.subTest(addition=addition):
                self.bad(addition, r"extension adds nothing: give new tiles, more variants or new transitions")

    def test_tiles_must_be_list(self):
        self.bad({"theme": "Forest", "tiles": "g2"}, r"tiles: must be a list of tile objects, got 'g2'")
        self.bad({"theme": "Forest", "tiles": {"id": "g2"}}, r"tiles: must be a list of tile objects, got a dict")

    def test_addition_must_be_object(self):
        for bad in (None, [], "x", 5, [add(g(2))]):
            with self.subTest(bad=repr(bad)):
                self.bad(bad, r"extension must be a JSON object")


class LimitTests(Base):
    def test_max_tiles_bound(self):
        merged, _ = self.ok(add(g(2), g(3)), max_tiles=5)
        self.assertEqual(len(merged["tiles"]), 5)
        self.bad(add(g(2), g(3)), r"tiles: need 1 to 4 tiles, got 5", max_tiles=4)

    def test_raising_variants_does_not_count_as_new_tile(self):
        merged, _ = self.ok(add(g(1, variants=4)), max_tiles=3)
        self.assertEqual(len(merged["tiles"]), 3)

    def test_max_jobs_bound(self):
        self.assertEqual(MAX_JOBS, 64)
        full = [g(i, variants=4) for i in range(2, 17)]  # 15 new tiles x 4 = 60 on top of the existing 4
        merged, keys = self.ok(add(*full[:15]), max_tiles=24)
        self.assertEqual(sum(t["variants"] for t in merged["tiles"]), 64)
        self.assertEqual(len(keys), 60)
        self.bad(add(*full, g(17, variants=4)), r"tiles: sum of variants is 68, limit 64", max_tiles=24)
        self.bad(add(*full[:14], g(1, variants=4), g(0, variants=4)), r"sum of variants is 65, limit 64")
        merged, keys = self.ok(add(*full[:14], g(1, variants=4)), max_tiles=24)
        self.assertEqual((sum(t["variants"] for t in merged["tiles"]), len(keys)), (62, 58))

    def test_max_tiles_argument_validation(self):
        for bad in (0, -1, True, 2.0, "8", None):
            with self.subTest(bad=repr(bad)):
                with self.assertRaisesRegex(ValueError, "max_tiles must be an integer >= 1") as cm:
                    validate_extension(existing(), add(g(2)), bad)
                self.assertIs(type(cm.exception), ValueError)

    def test_max_tiles_checked_before_plans(self):
        broken = existing()
        broken["tiles"][0]["category"] = "water"
        for existing_plan, addition in ((broken, add(g(2))), (existing(), "not a dict"), (existing(), add(g(2), theme="X"))):
            for bad in (True, 0):
                with self.subTest(bad=bad, addition=str(addition)[:20]):
                    with self.assertRaises(ValueError) as cm:
                        validate_extension(existing_plan, addition, bad)
                    self.assertIs(type(cm.exception), ValueError)

    def test_invalid_existing_plan(self):
        broken = existing()
        broken["tiles"][0]["category"] = "water"
        self.bad(add(g(2)), r"tiles\[0\]\.category: must be one of", base=broken)
        self.bad(add(g(2)), r"plan must be a JSON object", base="plan")
        self.bad(add(g(2)), r"theme: missing", base={"tiles": [g(0)]})


if __name__ == "__main__":
    unittest.main()
