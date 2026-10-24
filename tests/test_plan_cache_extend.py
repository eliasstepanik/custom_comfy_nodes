"""Tests for ``plan_cache.plan_hash`` and ``cache_key(existing_hash=...)`` (tileset reuse, plan row D16b).

Covers: ``plan_hash`` digest format, key-order independence, sensitivity to every nested value, exact
digest of a fixed plan (canonical JSON ``sort_keys``, compact separators, ASCII escapes), the error
paths (``ValueError`` for non-dict, non-JSON values, circular and too-deep plans), input-not-mutated;
``cache_key`` back-compat (empty ``existing_hash`` gives the same pinned digest as the four-argument
call, which is the digest produced before the parameter existed), the extra key component, the exact
canonical string that is hashed (rebuilt independently with ``hashlib``), and every ``existing_hash``
validation branch (type, length, case, non-hex characters, surrounding whitespace).

Residual gaps (not covered): ``plan_cache.load``/``store`` are covered by ``tests/test_plan.py`` and do
not take ``existing_hash``; hash collisions are not tested; non-ASCII ``model``/``prompt`` are only
covered by one escaped-ASCII example.
"""

import copy
import hashlib
import json
import unittest

from tilemap_core import plan_cache
from tilemap_core.plan_cache import cache_key, plan_hash
from tilemap_core.plan_prompt import SCHEMA_VERSION

HEX64 = "a" * 64
OTHER64 = "b" * 64
PLAN = {"theme": "Forest", "tiles": [{"id": "grass", "name": "Grass", "category": "ground",
                                       "description": "green grass field texture", "variants": 1,
                                       "footprint": [1, 1]}], "transitions": []}
# Digests computed once from the production code (canonical JSON, sha256) and pinned here.
PLAN_HASH_EMPTY_DICT = "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
PLAN_HASH_A1 = "015abd7f5cc57a2dd94b7590f04ad8084273905ee33ec5cebeae62276a97f862"
KEY_NO_EXISTING = "50046734b6a643434b046a4204acd351584e62685701e24be988858223df2f86"
KEY_WITH_EXISTING = "809a9d9ee9987c52b5eacce821c3d6ce71ae9e7d5f5eda4149626d11e744c0eb"


def sha(canon):
    return hashlib.sha256(canon.encode("ascii")).hexdigest()


class PlanHashTests(unittest.TestCase):
    def test_pinned_digests(self):
        self.assertEqual(plan_hash({}), PLAN_HASH_EMPTY_DICT)
        self.assertEqual(plan_hash({"a": 1}), PLAN_HASH_A1)

    def test_is_sha256_of_canonical_json(self):
        self.assertEqual(plan_hash(PLAN), sha(json.dumps(PLAN, sort_keys=True, separators=(",", ":"))))
        self.assertEqual(plan_hash({"b": [1, 2], "a": {"y": None, "x": True}}),
                         sha('{"a":{"x":true,"y":null},"b":[1,2]}'))

    def test_format_64_lowercase_hex(self):
        h = plan_hash(PLAN)
        self.assertEqual((len(h), h == h.lower(), set(h) <= set("0123456789abcdef")), (64, True, True))
        self.assertIsInstance(h, str)

    def test_key_order_independent_at_every_level(self):
        swapped = {"transitions": [], "tiles": [{"footprint": [1, 1], "variants": 1,
                                                  "description": "green grass field texture",
                                                  "category": "ground", "name": "Grass", "id": "grass"}],
                   "theme": "Forest"}
        self.assertEqual(plan_hash(PLAN), plan_hash(swapped))

    def test_list_order_matters(self):
        self.assertNotEqual(plan_hash({"a": [1, 2]}), plan_hash({"a": [2, 1]}))

    def test_every_value_matters(self):
        base = plan_hash(PLAN)
        edits = [("theme", "Forests"), ("tiles", 0, "id", "grass2"), ("tiles", 0, "name", "Grass2"),
                 ("tiles", 0, "category", "object"), ("tiles", 0, "description", "green grass field textur"),
                 ("tiles", 0, "variants", 2), ("tiles", 0, "footprint", 1, 2), ("transitions", [["a", "b"]])]
        seen = {base}
        for edit in edits:
            p = copy.deepcopy(PLAN)
            node = p
            for step in edit[:-2]:
                node = node[step]
            node[edit[-2]] = edit[-1]
            with self.subTest(edit=edit[:-1]):
                seen.add(plan_hash(p))
        self.assertEqual(len(seen), len(edits) + 1)

    def test_types_distinguished(self):
        self.assertEqual(len({plan_hash({"a": 1}), plan_hash({"a": "1"}), plan_hash({"a": 1.0}),
                              plan_hash({"a": True}), plan_hash({"a": None})}), 5)

    def test_non_ascii_is_escaped_and_stable(self):
        self.assertEqual(plan_hash({"t": "caf\u00e9"}), sha('{"t":"caf\\u00e9"}'))
        self.assertNotEqual(plan_hash({"t": "caf\u00e9"}), plan_hash({"t": "cafe"}))

    def test_input_not_mutated_and_deterministic(self):
        p = copy.deepcopy(PLAN)
        self.assertEqual(plan_hash(p), plan_hash(p))
        self.assertEqual(p, PLAN)

    def test_non_dict_rejected(self):
        for bad in (None, [], [PLAN], "plan", 5, b"{}", ("a",), json.dumps(PLAN)):
            with self.subTest(bad=repr(bad)[:30]):
                with self.assertRaisesRegex(ValueError, "plan must be a dict, got ") as cm:
                    plan_hash(bad)
                self.assertIs(type(cm.exception), ValueError)

    def test_not_json_serialisable_rejected(self):
        circular = {}
        circular["self"] = circular
        deep = cur = {}
        for _ in range(100000):
            cur["n"] = {}
            cur = cur["n"]
        cases = (("object", {"a": object()}), ("set", {"a": {1, 2}}), ("tuple key", {(1, 2): 3}),
                 ("circular", circular), ("too deep", deep))
        for label, bad in cases:
            with self.subTest(case=label):
                with self.assertRaisesRegex(ValueError, "plan is not JSON-serialisable"):
                    plan_hash(bad)

    def test_nan_infinity_accepted_by_json_dumps_do_not_crash(self):
        # json.dumps writes NaN/Infinity by default; plan_hash stays total and deterministic on them.
        self.assertEqual(plan_hash({"a": float("inf")}), plan_hash({"a": float("inf")}))
        self.assertNotEqual(plan_hash({"a": float("inf")}), plan_hash({"a": float("-inf")}))


class CacheKeyTests(unittest.TestCase):
    def test_pinned_digests(self):
        self.assertEqual(cache_key("m", "p", "top-down", 8), KEY_NO_EXISTING)
        self.assertEqual(cache_key("m", "p", "top-down", 8, "a" * 64), KEY_WITH_EXISTING)

    def test_schema_version_pinned(self):
        self.assertEqual(SCHEMA_VERSION, 1)

    def test_empty_existing_hash_is_old_key(self):
        old = sha(json.dumps({"max_tiles": 8, "model": "m", "prompt": "p", "schema_version": SCHEMA_VERSION,
                              "view": "top-down"}, sort_keys=True, separators=(",", ":")))
        self.assertEqual(cache_key("m", "p", "top-down", 8), old)
        self.assertEqual(cache_key("m", "p", "top-down", 8, ""), old)
        self.assertEqual(cache_key("m", "p", "top-down", 8, existing_hash=""), old)

    def test_existing_hash_is_a_further_component(self):
        canon = json.dumps({"existing_hash": HEX64, "max_tiles": 8, "model": "m", "prompt": "p",
                            "schema_version": SCHEMA_VERSION, "view": "top-down"},
                           sort_keys=True, separators=(",", ":"))
        self.assertEqual(cache_key("m", "p", "top-down", 8, HEX64), sha(canon))
        self.assertIn('"existing_hash":"' + HEX64 + '"', canon)

    def test_distinct_existing_hashes_distinct_keys(self):
        keys = {cache_key("m", "p", "top-down", 8, h) for h in ("", HEX64, OTHER64, "0" * 64, "f" * 64)}
        self.assertEqual(len(keys), 5)

    def test_existing_hash_does_not_collide_with_plain_fields(self):
        self.assertNotEqual(cache_key("m", "p", "top-down", 8, HEX64), cache_key("m", "p", "top-down", 8))
        self.assertNotEqual(cache_key("m", "p", "top-down", 8, HEX64), cache_key("m", "p", "top-down", 9, HEX64))
        self.assertNotEqual(cache_key("m", "p", "top-down", 8, HEX64), cache_key("m", "q", "top-down", 8, HEX64))
        self.assertNotEqual(cache_key("m", "p", "top-down", 8, HEX64), cache_key("n", "p", "top-down", 8, HEX64))
        self.assertNotEqual(cache_key("m", "p", "top-down", 8, HEX64), cache_key("m", "p", "side-view", 8, HEX64))

    def test_works_with_real_plan_hash(self):
        key = cache_key("m", "p", "top-down", 8, plan_hash(PLAN))
        self.assertRegex(key, r"^[0-9a-f]{64}$")
        edited = copy.deepcopy(PLAN)
        edited["theme"] = "Other"
        self.assertNotEqual(key, cache_key("m", "p", "top-down", 8, plan_hash(edited)))
        self.assertEqual(key, cache_key("m", "p", "top-down", 8, plan_hash(copy.deepcopy(PLAN))))

    def test_key_usable_as_cache_file_name(self):
        key = cache_key("m", "p", "top-down", 8, HEX64)
        self.assertEqual(plan_cache._path("/x", key), "/x/" + key + ".json")

    def test_non_ascii_text_escaped(self):
        self.assertEqual(cache_key("m", "caf\u00e9", "top-down", 8),
                         sha(json.dumps({"max_tiles": 8, "model": "m", "prompt": "caf\u00e9",
                                         "schema_version": SCHEMA_VERSION, "view": "top-down"},
                                        sort_keys=True, separators=(",", ":"))))

    def test_existing_hash_must_be_text(self):
        for bad in (None, 5, b"a" * 64, ["a" * 64], 0, False):
            with self.subTest(bad=repr(bad)[:20]):
                with self.assertRaisesRegex(ValueError, "existing_hash must be text, got ") as cm:
                    cache_key("m", "p", "top-down", 8, bad)
                self.assertIs(type(cm.exception), ValueError)

    def test_existing_hash_must_be_64_lowercase_hex(self):
        msg = "existing_hash must be empty or a 64-character lowercase hex digest"
        bad_values = ["a" * 63, "a" * 65, "A" * 64, "g" * 64, "a" * 63 + "G", " " + "a" * 63, "a" * 64 + "\n",
                      "a" * 32, "0x" + "a" * 62, "a" * 63 + "\u00e9", " ", "a b" * 21 + "a"]
        for bad in bad_values:
            with self.subTest(bad=repr(bad)[:24]):
                with self.assertRaisesRegex(ValueError, msg) as cm:
                    cache_key("m", "p", "top-down", 8, bad)
                self.assertIs(type(cm.exception), ValueError)

    def test_boundary_hashes_accepted(self):
        for ok in ("0" * 64, "f" * 64, "0123456789abcdef" * 4):
            with self.subTest(ok=ok[:8]):
                self.assertRegex(cache_key("m", "p", "top-down", 8, ok), r"^[0-9a-f]{64}$")

    def test_existing_validation_does_not_mask_other_errors(self):
        for args in (("m", "p", "top-down", True), ("m", "p", "top-down", 2.0), (None, "p", "v", 1),
                     ("m", 5, "v", 1), ("m", "p", b"v", 1)):
            with self.subTest(args=repr(args)):
                with self.assertRaises(ValueError):
                    cache_key(*args, HEX64)
                with self.assertRaises(ValueError):
                    cache_key(*args, "bad")

    def test_deterministic(self):
        self.assertEqual(cache_key("m", "p", "top-down", 8, HEX64), cache_key("m", "p", "top-down", 8, HEX64))


if __name__ == "__main__":
    unittest.main()
