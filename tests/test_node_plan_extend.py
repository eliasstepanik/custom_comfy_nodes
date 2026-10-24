"""Tests for ``CCN_TilemapPlan`` with ``plan_mode`` extend (tileset reuse, D16c). Stdlib unittest; no network: the LLM is a
scripted fake of ``llm_client.chat``; the cache is a temp dir.

Covers: the extend request (``plan_prompt.build_extend_messages``), the ``existing_plan`` forms (``CCN_LoadTilemap``
job_json with ``base`` / bare plan / earlier ``plan_json`` output), ``meta.extend`` (``new_keys`` + ``base``), the cache
key including ``plan_hash(existing)``, cache hits and the two "useless cache file" misses, the repair path through
``plan_schema.validate_extension``, and every ``ValueError`` of ``_existing`` / ``_check_base``.

Helpers (``PlanCase``, ``reply``, ``MODEL``) come from ``tests/test_node_plan.py``.

Residual gaps: merge rules of ``validate_extension`` live in ``tests/test_plan_extension.py`` (only one rule is exercised
here, through the repair retry); the base palette/snap values are opaque to this node and are only passed through.
"""

import copy
import json
import unittest

from tests import _support
from tests.test_node_plan import MODEL, PlanCase, setUpModule, tearDownModule  # noqa: F401  (module fixtures)
from tilemap_core import plan_cache, plan_prompt, plan_schema

BASE_PLAN = plan_schema.validate_plan(_support.valid_plan(2, 1, 0), 10)  # g0, g1, o0 ; transition g0-g1 ; 3 variants
ALL = ["g0#0", "g1#0", "o0#0"]
BASE = {"path": "tilemap/ts_00001_", "keys": ALL, "palette": ["#000000", "#ffffff"], "snap": {"palette_mode": "refit"}}


def addition(*extra_ids, theme="test forest", transitions=None):
    tiles = [{"id": i, "name": f"Name {i}", "category": "ground" if i.startswith("g") else "overlay",
              "description": f"visual description of tile {i}"} for i in extra_ids]
    out = {"theme": theme, "tiles": tiles}
    if transitions is not None:
        out["transitions"] = transitions
    return json.dumps(out)


DEFAULT = object()  # "use the standard job_json" (None is a value under test)


def job_json(base=BASE, plan=None, **top):
    data = {"tile_size": 16, "view": "top-down", "plan": BASE_PLAN if plan is None else plan, "jobs": [], **top}
    if base is not None:
        data["base"] = base
    return json.dumps(data)


class ExtendCase(PlanCase):
    def extend(self, existing=DEFAULT, **kw):
        kw.setdefault("prompt", "add flowers")
        kw.setdefault("max_tiles", 10)
        return self.plan(plan_mode="extend", existing_plan=job_json() if existing is DEFAULT else existing, **kw)

    def ekey(self, plan=BASE_PLAN, prompt="add flowers", view="top-down", max_tiles=10):
        return self.key(prompt, view, max_tiles, ehash=plan_cache.plan_hash(plan))

    def meta(self, result):
        return json.loads(result[0])["meta"]


class ExtendHappyTests(ExtendCase):
    def test_job_json_with_base(self):
        chat = self.fake(addition("v0", "g2", transitions=[["g1", "g2"]]))
        result = self.extend()
        data = json.loads(result[0])
        self.assertEqual({t["id"] for t in data["tiles"]}, {"g0", "g1", "g2", "o0", "v0"})
        self.assertEqual(data["transitions"], [["g0", "g1"], ["g1", "g2"]])
        self.assertEqual(data["meta"], {"tile_size": 16, "view": "top-down", "model": MODEL,
                                        "extend": {"new_keys": ["g2#0", "v0#0"], "base": BASE}})
        self.assertEqual(result[2], MODEL)
        self.assertTrue(result[1].endswith("\nextension: 2 new Flux jobs"), result[1])
        self.assertIn("plan from LLM", result[1])
        self.assertEqual(chat.call_count, 1)
        self.assertEqual(chat.call_args.args[0], plan_prompt.build_extend_messages("add flowers", "top-down", BASE_PLAN))
        self.assertEqual(chat.call_args.kwargs, {"max_tokens": 8000, "timeout": 180})

    def test_bare_plan_has_no_base_in_meta(self):
        self.fake(addition("v0"), addition("v0"))
        for text in (json.dumps(BASE_PLAN), job_json(base=None)):
            meta = self.meta(self.extend(existing=text, prompt=text[:8]))
            self.assertEqual(meta["extend"], {"new_keys": ["v0#0"]})

    def test_plan_json_of_an_earlier_run_is_a_valid_existing_plan(self):
        self.fake(json.dumps(_support.valid_plan(2, 1, 0)), addition("v0"))
        first = self.plan()
        result = self.extend(existing=first[0])
        self.assertEqual(self.meta(result)["extend"], {"new_keys": ["v0#0"]})
        self.assertEqual(len(json.loads(result[0])["tiles"]), 4)  # g0, g1, o0 of the first run + v0

    def test_extension_raising_variants_only(self):
        grown = {"theme": "test forest", "tiles": [{"id": "g1", "name": "ground 1", "category": "ground",
                                                    "description": BASE_PLAN["tiles"][1]["description"], "variants": 3}]}
        self.fake(json.dumps(grown))
        result = self.extend()
        self.assertEqual(self.meta(result)["extend"]["new_keys"], ["g1#1", "g1#2"])
        self.assertIn("g1 x3", result[1])
        self.assertIn("extension: 2 new Flux jobs", result[1])

    def test_new_keys_follow_category_order_not_reply_order(self):
        self.fake(addition("v0", "g2", "v1", "g3"))
        keys = self.meta(self.extend())["extend"]["new_keys"]
        self.assertEqual(keys, ["g2#0", "g3#0", "v0#0", "v1#0"])

    def test_tile_size_and_view_are_taken_from_the_node(self):
        self.fake(addition("v0"))
        meta = self.meta(self.extend(existing=job_json(tile_size=32, view="side-view"), tile_size=32, view="side-view"))
        self.assertEqual((meta["tile_size"], meta["view"]), (32, "side-view"))

    def test_input_not_mutated_by_the_node(self):
        text, before = job_json(), copy.deepcopy(BASE_PLAN)
        self.fake(addition("v0"))
        self.extend(existing=text)
        self.assertEqual(BASE_PLAN, before)
        self.assertEqual(json.loads(text)["base"], BASE)


class ExtendCacheTests(ExtendCase):
    def test_cache_file_name_includes_existing_hash(self):
        self.fake(addition("v0"))
        self.extend()
        self.assertEqual(self.files(), [self.ekey() + ".json"])
        self.assertNotEqual(self.ekey(), self.key("add flowers", "top-down", 10))
        merged = plan_cache.load(self.cache, self.ekey(), 10)
        self.assertEqual([t["id"] for t in merged["tiles"]], ["g0", "g1", "o0", "v0"])

    def test_cache_hit_recomputes_new_keys_and_skips_llm(self):
        chat = self.fake(addition("v0", "g2"))
        first = self.extend()
        second = self.extend()
        self.assertEqual(chat.call_count, 1)
        self.assertEqual(self.meta(second), self.meta(first))
        self.assertEqual(self.meta(second)["extend"]["new_keys"], ["g2#0", "v0#0"])
        self.assertEqual(second[0], first[0])
        self.assertTrue(second[1].splitlines()[1].endswith("plan from cache"))
        self.assertTrue(second[1].endswith("extension: 2 new Flux jobs"))

    def test_different_existing_plan_is_a_miss(self):
        chat = self.fake(addition("v0"), addition("v0"))
        self.extend()
        other = copy.deepcopy(BASE_PLAN)
        other["tiles"][2]["variants"] = 2
        base = {**BASE, "keys": ALL + ["o0#1"]}
        self.extend(existing=job_json(base=base, plan=other))
        self.assertEqual(chat.call_count, 2)
        self.assertEqual(len(self.files()), 2)

    def test_same_plan_in_other_key_order_is_a_hit(self):
        chat = self.fake(addition("v0"))
        self.extend()
        reordered = json.dumps({"plan": dict(reversed(list(BASE_PLAN.items()))), "view": "top-down", "tile_size": 16, "base": BASE})
        self.extend(existing=reordered)
        self.assertEqual(chat.call_count, 1)

    def test_base_does_not_change_the_key(self):
        chat = self.fake(addition("v0"))
        self.extend()
        self.extend(existing=job_json(base=None))
        self.assertEqual(chat.call_count, 1)

    def test_new_and_extend_with_same_prompt_do_not_collide(self):
        chat = self.fake(json.dumps(_support.valid_plan(2, 1, 0)), addition("v0"))
        self.plan(prompt="add flowers")
        self.extend()
        self.assertEqual(chat.call_count, 2)
        self.assertEqual(len(self.files()), 2)

    def test_refresh_plan_asks_again_in_extend_mode(self):
        chat = self.fake(addition("v0"), addition("v0", "v1"))
        self.extend()
        result = self.extend(refresh_plan=True)
        self.assertEqual(chat.call_count, 2)
        self.assertEqual(self.meta(result)["extend"]["new_keys"], ["v0#0", "v1#0"])
        self.assertEqual(self.meta(self.extend())["extend"]["new_keys"], ["v0#0", "v1#0"])  # refreshed file is served next

    def test_cache_file_that_adds_nothing_is_a_miss(self):
        plan_cache.store(self.cache, self.ekey(), BASE_PLAN, 10)
        chat = self.fake(addition("v0"))
        result = self.extend()
        self.assertEqual(chat.call_count, 1)
        self.assertEqual(self.meta(result)["extend"]["new_keys"], ["v0#0"])

    def test_cache_file_that_drops_base_tiles_is_a_miss(self):
        stray = plan_schema.validate_plan(_support.valid_plan(2, 0, 1), 10)  # no o0, but a new v0
        plan_cache.store(self.cache, self.ekey(), stray, 10)
        chat = self.fake(addition("v0"))
        result = self.extend()
        self.assertEqual(chat.call_count, 1)
        self.assertIn("o0", [t["id"] for t in json.loads(result[0])["tiles"]])
        self.assertIn("plan from LLM", result[1])

    def test_cache_file_valid_for_a_different_max_tiles_is_not_used(self):
        self.fake(addition("v0"), addition("v0"))
        self.extend(max_tiles=10)
        self.extend(max_tiles=9)
        self.assertEqual(len(self.files()), 2)


class ExtendRepairTests(ExtendCase):
    def test_empty_extension_is_repaired_once(self):
        chat = self.fake(json.dumps({"theme": "test forest", "tiles": []}), addition("v0"))
        result = self.extend()
        self.assertEqual(chat.call_count, 2)
        self.assertIn("extension adds nothing", chat.call_args_list[1].args[0][-1]["content"])
        self.assertEqual(self.meta(result)["extend"]["new_keys"], ["v0#0"])

    def test_wrong_theme_then_second_failure_raises_plan_error(self):
        chat = self.fake(addition("v0", theme="Desert"), addition("v0", theme="Snow"))
        with self.assertRaises(plan_schema.PlanError) as cm:
            self.extend()
        msg = str(cm.exception)
        self.assertEqual(chat.call_count, 2)
        self.assertIn("First error: theme: must equal the existing theme 'test forest'", msg)
        self.assertIn("Second error: theme: must equal the existing theme 'test forest'", msg)
        self.assertEqual(self.files(), [])

    def test_max_tiles_overflow_of_merged_plan_is_repaired(self):
        chat = self.fake(addition("g2", "v0"), addition("v0"))
        result = self.extend(max_tiles=4)  # base has 3 tiles; +2 is 5 > 4
        self.assertEqual(chat.call_count, 2)
        self.assertEqual(self.meta(result)["extend"]["new_keys"], ["v0#0"])

    def test_unknown_id_redefinition_is_a_schema_error_not_a_crash(self):
        redefine = json.dumps({"theme": "test forest", "tiles": [{"id": "g0", "name": "Other", "category": "ground",
                                                                  "description": "visual description of tile", "variants": 2}]})
        self.fake(redefine, addition("v0"))
        self.assertEqual(self.meta(self.extend())["extend"]["new_keys"], ["v0#0"])


class ExtendValidationTests(ExtendCase):
    def bad(self, regex, existing=DEFAULT, exc=ValueError, **kw):
        chat = self.fake()
        with self.assertRaisesRegex(exc, regex):
            self.extend(existing=existing, **kw)
        self.assertEqual((chat.call_count, self.files()), (0, []))

    def test_missing_or_blank_existing_plan(self):
        for blank in ("  \n", None, 5):
            with self.subTest(existing=blank):
                self.bad(r"^plan_mode extend needs existing_plan: connect CCN_LoadTilemap job_json$", existing=blank)
        chat = self.fake()
        with self.assertRaisesRegex(ValueError, "needs existing_plan"):
            self.plan(plan_mode="extend", max_tiles=10)  # default existing_plan=""
        self.assertEqual(chat.call_count, 0)

    def test_unparseable_or_invalid_existing_plan(self):
        for text in ("not json", "[]", "{}", json.dumps({"theme": "x", "tiles": []}), json.dumps({"plan": 3, "tiles": 1})):
            with self.subTest(text=text):
                self.bad(r".", existing=text, exc=plan_schema.PlanError)

    def test_invalid_plan_inside_job_json(self):
        self.bad(r".", existing=job_json(plan={"theme": "x", "tiles": []}), exc=plan_schema.PlanError)

    def test_tile_size_and_view_must_match_the_node(self):
        self.bad(r"^existing_plan tile_size is 32, the node is set to 16$", existing=job_json(tile_size=32))
        self.bad(r"^existing_plan view is 'side-view', the node is set to 'top-down'$", existing=job_json(view="side-view"))
        self.bad(r"^existing_plan tile_size is 16, the node is set to 8$", tile_size=8)

    def test_base_block_shape(self):
        bad_bases = [{}, {"path": "p", "keys": [], "palette": []}, {**BASE, "extra": 1}, {**BASE, "keys": "g0#0"}, "x", [], 7,
                     {k: v for k, v in BASE.items() if k != "snap"}]
        for base in bad_bases:
            with self.subTest(base=base):
                self.bad(r"^base must be an object with exactly the keys \['keys', 'palette', 'path', 'snap'\] and a keys list$",
                         existing=job_json(base=base))

    def test_max_tiles_below_existing_tile_count(self):
        self.bad(r"^max_tiles 2 is below the 3 existing tiles$", max_tiles=2)

    def test_empty_prompt_in_extend_mode(self):
        self.bad("prompt is empty after removing unsupported characters", prompt="  ")

    def test_all_jobs_used_leaves_no_room(self):
        full = plan_schema.validate_plan({"theme": "t", "tiles": [
            {"id": f"g{i}", "name": f"G {i}", "category": "ground", "description": "visual description g", "variants": 4}
            for i in range(16)]}, 16)
        self.assertEqual(sum(t["variants"] for t in full["tiles"]), plan_schema.MAX_JOBS)
        self.bad(r"already uses all 64 variants", existing=json.dumps(full), max_tiles=20)


if __name__ == "__main__":
    unittest.main()
