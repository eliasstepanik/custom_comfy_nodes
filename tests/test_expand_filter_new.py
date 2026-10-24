"""Tests for ``expand.filter_new`` (tileset reuse, plan row D16a; unittest, stdlib only).

Covers: job selection by ``"<tile_id>#<variant>"`` key, re-indexing from ``len(jobs) - len(kept)``,
the seed formula ``(base_seed + index) % 2**64`` surviving the move (including wrap-around at 2**64),
job order kept, fresh deep copies (input not mutated, no aliasing, extra fields kept), all collection
types for ``new_keys``, duplicate keys, empty selection, every ``ValueError`` path (container types,
non-string keys, unknown keys, malformed jobs incl. bool/float ints) and the integration with
``plan_schema.validate_extension`` + ``expand_jobs`` (kept jobs are the tail of the merged expansion).

Residual gaps (not covered): ``expand_jobs`` itself is covered by ``tests/test_plan.py``; a ``new_keys``
generator is accepted (iterated once) but only the list/tuple/set/frozenset/dict forms are pinned.
"""

import copy
import unittest

from tilemap_core import expand
from tilemap_core.expand import SEED_LIMIT, filter_new
from tilemap_core.plan_schema import validate_extension, validate_plan


def job(index, tile_id="a", variant=0, seed=None, **kw):
    return {"index": index, "tile_id": tile_id, "variant": variant,
            "seed": index if seed is None else seed, "prompt": f"p{index}", "footprint": [1, 1], **kw}


def jobs_for(tiles, base_seed=100):
    """Jobs like ``expand_jobs`` makes them: sequential index, seed = base_seed + index."""
    out = []
    for tile_id, count in tiles:
        for v in range(count):
            out.append(job(len(out), tile_id, v, base_seed + len(out)))
    return out


class SelectionTests(unittest.TestCase):
    def test_tail_extension_reindexed_from_base_count(self):
        jobs = jobs_for([("a", 2), ("b", 1), ("c", 2)])
        kept = filter_new(jobs, ["c#0", "c#1"])
        self.assertEqual([(j["tile_id"], j["variant"], j["index"]) for j in kept], [("c", 0, 3), ("c", 1, 4)])

    def test_seed_formula_preserved(self):
        jobs = jobs_for([("a", 2), ("b", 3)], base_seed=1000)
        kept = filter_new(jobs, ["b#1", "b#2"])
        self.assertEqual([j["index"] for j in kept], [3, 4])
        self.assertEqual([j["seed"] for j in kept], [1003, 1004])
        for j in kept:
            self.assertEqual(j["seed"], 1000 + j["index"])

    def test_seed_moves_by_index_offset(self):
        jobs = [job(0, "a", 0, 7), job(1, "b", 0, 9), job(2, "b", 1, 11)]
        kept = filter_new(jobs, ["b#0"])
        self.assertEqual((kept[0]["index"], kept[0]["seed"]), (2, 10))

    def test_seed_wraps_modulo_2_64(self):
        top = SEED_LIMIT - 1
        jobs = [job(0, "a", 0, top), job(1, "b", 0, 0), job(2, "b", 1, 1)]
        kept = filter_new(jobs, ["b#0", "b#1"])
        self.assertEqual([(j["index"], j["seed"]) for j in kept], [(1, 0), (2, 1)])
        self.assertEqual(SEED_LIMIT, 2**64)
        wrap = [job(0, "a", 0, top), job(1, "b", 0, 0), job(2, "c", 0, 1)]
        self.assertEqual(filter_new(wrap, ["a#0"])[0]["seed"], 2 - 1)  # (top - 0 + 2) % 2**64 == 1
        down = [job(0, "a", 0, 5), job(1, "a", 1, 6), job(2, "b", 0, 7)]
        self.assertEqual(filter_new(down, ["a#0"])[0]["seed"], 5 + 2)

    def test_job_order_kept_not_key_order(self):
        jobs = jobs_for([("a", 1), ("b", 1), ("c", 1), ("d", 1)])
        kept = filter_new(jobs, ["d#0", "b#0", "c#0"])
        self.assertEqual([(j["tile_id"], j["index"]) for j in kept], [("b", 1), ("c", 2), ("d", 3)])

    def test_job_order_kept_when_ids_not_alphabetical(self):
        jobs = jobs_for([("zed", 1), ("alpha", 1), ("mid", 1), ("beta", 1)])
        kept = filter_new(jobs, ["alpha#0", "zed#0", "beta#0"])
        self.assertEqual([(j["tile_id"], j["index"]) for j in kept], [("zed", 1), ("alpha", 2), ("beta", 3)])

    def test_non_contiguous_selection_gets_contiguous_new_indices(self):
        jobs = jobs_for([("a", 1), ("b", 1), ("c", 1), ("d", 1), ("e", 1)])
        kept = filter_new(jobs, ["b#0", "d#0"])
        self.assertEqual([j["index"] for j in kept], [3, 4])
        self.assertEqual([j["tile_id"] for j in kept], ["b", "d"])

    def test_variant_selected_individually(self):
        jobs = jobs_for([("a", 3)])
        kept = filter_new(jobs, ["a#1"])
        self.assertEqual([(j["variant"], j["index"]) for j in kept], [(1, 2)])
        self.assertEqual(kept[0]["seed"], 100 + 2)

    def test_tile_id_with_hash_like_text_is_matched_exactly(self):
        jobs = [job(0, "a", 1), job(1, "a#1", 0), job(2, "b", 0)]
        # job 0 has key "a#1", job 1 has key "a#1#0": only the exact key matches
        self.assertEqual([j["variant"] for j in filter_new(jobs, ["a#1"])], [1])
        both = filter_new(jobs, ["a#1", "a#1#0"])
        self.assertEqual([j["tile_id"] for j in both], ["a", "a#1"])
        self.assertEqual([j["index"] for j in both], [1, 2])

    def test_all_jobs_kept_leaves_indices_and_seeds_unchanged(self):
        jobs = jobs_for([("a", 2), ("b", 1)])
        kept = filter_new(jobs, ["a#0", "a#1", "b#0"])
        self.assertEqual(kept, jobs)

    def test_empty_selection_returns_empty_list(self):
        for keys in ([], (), set(), {}):
            with self.subTest(keys=keys):
                out = filter_new(jobs_for([("a", 2)]), keys)
                self.assertEqual(out, [])
                self.assertIs(type(out), list)

    def test_empty_jobs_and_keys(self):
        self.assertEqual(filter_new([], []), [])

    def test_duplicate_keys_select_once(self):
        jobs = jobs_for([("a", 1), ("b", 1)])
        kept = filter_new(jobs, ["b#0", "b#0", "b#0"])
        self.assertEqual([(j["tile_id"], j["index"]) for j in kept], [("b", 1)])

    def test_new_keys_container_types(self):
        jobs = jobs_for([("a", 1), ("b", 2)])
        expected = [("b", 0, 1), ("b", 1, 2)]
        for keys in (["b#0", "b#1"], ("b#0", "b#1"), {"b#0", "b#1"}, frozenset({"b#0", "b#1"}),
                     {"b#0": 1, "b#1": 2}, iter(["b#0", "b#1"])):
            with self.subTest(kind=type(keys).__name__):
                got = filter_new(jobs, keys)
                self.assertEqual([(j["tile_id"], j["variant"], j["index"]) for j in got], expected)


class PurityTests(unittest.TestCase):
    def test_input_not_mutated(self):
        jobs = jobs_for([("a", 2), ("b", 2)])
        before = copy.deepcopy(jobs)
        keys = ["b#0", "b#1"]
        filter_new(jobs, keys)
        self.assertEqual(jobs, before)
        self.assertEqual(keys, ["b#0", "b#1"])

    def test_returned_jobs_are_fresh_deep_copies(self):
        jobs = [job(0, "a"), job(1, "b", footprint=[2, 3])]
        kept = filter_new(jobs, ["b#0"])
        self.assertIsNot(kept[0], jobs[1])
        self.assertIsNot(kept[0]["footprint"], jobs[1]["footprint"])
        kept[0]["footprint"].append(9)
        kept[0]["prompt"] = "changed"
        self.assertEqual(jobs[1]["footprint"], [2, 3])
        self.assertEqual(jobs[1]["prompt"], "p1")

    def test_all_kept_still_copies(self):
        jobs = [job(0, "a")]
        kept = filter_new(jobs, ["a#0"])
        self.assertIsNot(kept[0], jobs[0])
        self.assertIsNot(kept, jobs)

    def test_other_fields_carried_over_unchanged(self):
        extra = {"name": "Tree", "category": "object", "tileable": False, "width": 512, "height": 768,
                 "extra": {"k": [1, 2]}}
        jobs = [job(0, "a"), job(1, "t", 0, **extra)]
        kept = filter_new(jobs, ["t#0"])[0]
        for k, v in extra.items():
            self.assertEqual(kept[k], v, k)
        self.assertEqual(kept["prompt"], "p1")
        self.assertEqual(set(kept), set(jobs[1]))

    def test_deterministic(self):
        jobs = jobs_for([("a", 2), ("b", 2)])
        self.assertEqual(filter_new(jobs, {"b#1", "b#0"}), filter_new(jobs, ["b#0", "b#1"]))


class ErrorTests(unittest.TestCase):
    def raises(self, jobs, keys, regex):
        with self.assertRaisesRegex(ValueError, regex) as cm:
            filter_new(jobs, keys)
        return cm.exception

    def test_jobs_must_be_list(self):
        for bad in (None, (), "abc", {}, 5, iter([])):
            with self.subTest(bad=type(bad).__name__):
                self.raises(bad, [], "jobs must be a list, got ")
        self.raises((job(0),), ["a#0"], "jobs must be a list, got tuple")

    def test_new_keys_must_be_collection(self):
        for bad in ("a#0", b"a#0", "", None, 5, 1.5, True):
            with self.subTest(bad=repr(bad)):
                self.raises([job(0)], bad, "new_keys must be a collection of 'id#variant' strings, got ")

    def test_new_keys_must_hold_strings(self):
        for bad in ([1], ["a#0", 2], [None], [("a", 0)], [b"a#0"], {"a#0", 3}):
            with self.subTest(bad=repr(bad)):
                self.raises([job(0)], bad, "new_keys must contain only strings")

    def test_unknown_key_rejected_and_listed_sorted(self):
        exc = self.raises(jobs_for([("a", 1)]), ["z#0", "a#0", "a#5", "b#0"], r"new_keys without a job: \['a#5', 'b#0', 'z#0'\]")
        self.assertIs(type(exc), ValueError)

    def test_key_of_wrong_shape_has_no_job(self):
        for bad in ("a", "a#", "a#00", "a# 0", "A#0", "a#0 ", "#0", "a#-1", "a#0.0"):
            with self.subTest(bad=bad):
                self.raises(jobs_for([("a", 1)]), [bad], "new_keys without a job")

    def test_job_must_be_dict(self):
        for bad in (None, 5, "a#0", ["a"], (1, 2)):
            with self.subTest(bad=repr(bad)):
                self.raises([job(0), bad], ["a#0"], "every job must be a dict, got ")

    def test_job_fields_validated_even_when_not_selected(self):
        base = job(0)
        broken = [dict(base, index="0"), dict(base, index=True), dict(base, index=1.0), dict(base, index=None),
                  dict(base, seed="1"), dict(base, seed=False), dict(base, seed=2.5),
                  dict(base, variant="0"), dict(base, variant=True), dict(base, variant=None),
                  dict(base, tile_id=5), dict(base, tile_id=None), dict(base, tile_id=b"a")]
        for bad in broken:
            with self.subTest(bad={k: bad[k] for k in ("index", "seed", "variant", "tile_id")}):
                self.raises([job(1, "b"), bad], ["b#0"], "job lacks an integer index/seed/variant or a string tile_id")
        for field in ("index", "seed", "variant", "tile_id"):
            partial = job(0)
            del partial[field]
            with self.subTest(missing=field):
                self.raises([partial], [], "job lacks an integer index/seed/variant or a string tile_id")

    def test_failed_call_does_not_mutate(self):
        jobs = jobs_for([("a", 1)])
        before = copy.deepcopy(jobs)
        with self.assertRaises(ValueError):
            filter_new(jobs, ["nope#0"])
        self.assertEqual(jobs, before)


class ExtensionIntegrationTests(unittest.TestCase):
    def plan(self):
        tile = lambda i, cat="ground", **kw: {"id": f"t{i}", "name": f"Tile {i}", "category": cat,
                                              "description": f"visual description {i} of a tile", **kw}
        return validate_plan({"theme": "Forest", "tiles": [tile(0, variants=2), tile(1), tile(2, "object")],
                              "transitions": [["t0", "t1"]]}, 8)

    def test_kept_jobs_are_new_tail_with_consistent_seeds(self):
        base = self.plan()
        addition = {"theme": "Forest", "tiles": [
            {"id": "t0", "name": "Tile 0", "category": "ground", "description": "x" * 12, "variants": 3},
            {"id": "t3", "name": "Tile 3", "category": "ground", "description": "visual description 3 of a tile"},
            {"id": "t4", "name": "Tile 4", "category": "overlay", "description": "visual description 4 of a tile",
             "variants": 2}]}
        merged, keys = validate_extension(base, addition, 8)
        stored = expand.expand_jobs(base, 16, "top-down", "S", 42)
        full = expand.expand_jobs(merged, 16, "top-down", "S", 42)
        kept = filter_new(full, keys)
        self.assertEqual(len(kept), len(keys))
        self.assertEqual([f"{j['tile_id']}#{j['variant']}" for j in kept], keys)
        self.assertEqual([j["index"] for j in kept], list(range(len(stored), len(stored) + len(keys))))
        for j in kept:
            self.assertEqual(j["seed"], (42 + j["index"]) % SEED_LIMIT)
        self.assertEqual({(j["tile_id"], j["variant"]) for j in kept} & {(j["tile_id"], j["variant"]) for j in stored},
                         set())

    def test_prompts_and_sizes_survive(self):
        base = self.plan()
        merged, keys = validate_extension(base, {"theme": "Forest", "tiles": [
            {"id": "t9", "name": "Big", "category": "object", "description": "a big tree with leaves",
             "footprint": [2, 2]}]}, 8)
        full = expand.expand_jobs(merged, 16, "top-down", "S", 0)
        original = [j for j in full if j["tile_id"] == "t9"][0]
        kept = filter_new(full, keys)[0]
        for field in ("prompt", "width", "height", "footprint", "category", "name", "tileable", "variant"):
            self.assertEqual(kept[field], original[field], field)
        self.assertEqual(kept["index"], 4)  # 4 stored base jobs; the new object is last in expansion order


if __name__ == "__main__":
    unittest.main()
