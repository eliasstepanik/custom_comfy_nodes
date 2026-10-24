"""Tests for the extension path of ``CCN_TilemapExpand`` (``plan_json.meta.extend``, D16c): only the new jobs are
kept, re-indexed after the base jobs (``expand.filter_new``), ``job_json`` carries ``plan`` (merged), ``jobs`` (new only)
and ``base``; ``_extension_jobs`` rejects every inconsistent ``meta.extend`` / ``base.keys`` combination.
Stdlib unittest; ``plan_json`` strings are hand-built. Helpers come from ``tests/test_node_plan_expand.py``.

Residual gaps: the ``CCN_TilemapPlan`` -> ``CCN_TilemapExpand`` round trip with a fake LLM is done once here
(``RoundTripTests``); assemble-side handling of ``base`` (``tilemap_assemble``) is covered by the assemble tests.
"""

import copy
import io
import json
import contextlib
import tempfile
import types
import unittest
from unittest import mock

from tests import _support, test_node_plan
from tests.test_node_plan import setUpModule, tearDownModule  # noqa: F401  (module fixtures)
from tilemap_core import expand, llm_client, plan_schema

MERGED = plan_schema.validate_plan(_support.valid_plan(2, 1, 1, variants=2), 10)  # 8 jobs
KEYS = [f"{i}#{v}" for i in ("g0", "g1", "o0", "v0") for v in (0, 1)]
BASE = {"path": "tilemap/ts_00001_", "keys": KEYS[:6], "palette": ["#000000"], "snap": {"palette_mode": "refit"}}
NEW = KEYS[6:]  # v0#0, v0#1


def ext_json(new=NEW, base=BASE, plan=MERGED, **extra):
    ext = {"new_keys": list(new), **({} if base is None else {"base": base})}
    return json.dumps({**plan, "meta": {"tile_size": 16, "view": "top-down", "extend": ext, **extra}})


def run(text=None, seed=100, **kw):
    return test_node_plan.P.CCN_TilemapExpand().expand(ext_json() if text is None else text, "S", seed, 512, **kw)


class ExtendExpandTests(unittest.TestCase):
    def test_only_new_jobs_reindexed_after_the_base(self):
        prompts, widths, heights, seeds, text = run()
        job = json.loads(text)
        self.assertEqual([(j["tile_id"], j["variant"], j["index"]) for j in job["jobs"]], [("v0", 0, 6), ("v0", 1, 7)])
        self.assertEqual(seeds, [106, 107])  # base_seed + final index
        self.assertEqual((len(prompts), widths, heights), (2, [512, 512], [512, 512]))
        self.assertEqual(job["plan"], MERGED)  # the plan stays the merged one
        self.assertEqual(job["base"], BASE)
        self.assertEqual(set(job), {"tile_size", "view", "plan", "jobs", "base"})
        full = expand.expand_jobs(MERGED, 16, "top-down", "S", 100, 512)
        self.assertEqual(job["jobs"], expand.filter_new(full, NEW))

    def test_without_base_block(self):
        job = json.loads(run(ext_json(base=None))[4])
        self.assertNotIn("base", job)
        self.assertEqual([j["index"] for j in job["jobs"]], [6, 7])

    def test_without_base_any_new_keys_subset_is_allowed(self):
        job = json.loads(run(ext_json(new=["g1#1"], base=None))[4])
        self.assertEqual([(j["tile_id"], j["variant"], j["index"], j["seed"]) for j in job["jobs"]], [("g1", 1, 7, 107)])

    def test_new_keys_in_the_middle_of_the_job_order(self):
        base = {**BASE, "keys": [k for k in KEYS if k not in ("g1#1", "v0#0")]}
        job = json.loads(run(ext_json(new=["g1#1", "v0#0"], base=base))[4])
        self.assertEqual([(j["tile_id"], j["variant"], j["index"]) for j in job["jobs"]], [("g1", 1, 6), ("v0", 0, 7)])

    def test_empty_new_keys_gives_no_jobs(self):
        base = {**BASE, "keys": KEYS}
        prompts, widths, heights, seeds, text = run(ext_json(new=[], base=base))
        self.assertEqual((prompts, widths, heights, seeds), ([], [], [], []))
        self.assertEqual(json.loads(text)["jobs"], [])

    def test_model_input_with_extension(self):
        self.assertEqual(json.loads(run(model="claude-opus-5")[4])["model"], "claude-opus-5")

    def test_merged_plan_is_not_mutated(self):
        before = copy.deepcopy(MERGED)
        run()
        self.assertEqual(MERGED, before)


class ExtensionErrorTests(unittest.TestCase):
    def bad(self, regex, text):
        with self.assertRaisesRegex(ValueError, regex):
            run(text)

    def test_extend_block_shape(self):
        shape = r"^plan_json meta.extend must be an object with new_keys \(list\) and optional base$"
        for ext in (None, [], "x", 5, {}, {"base": BASE}, {"new_keys": "v0#0"}, {"new_keys": NEW, "extra": 1}, {"new_keys": None},
                    {"new_keys": NEW, "baze": BASE}):
            with self.subTest(extend=ext):
                text = json.dumps({**MERGED, "meta": {"tile_size": 16, "view": "top-down", "extend": ext}})
                self.bad(shape, text)

    def test_base_block_shape(self):
        shape = r"^base must be an object with exactly the keys \['keys', 'palette', 'path', 'snap'\] and a keys list$"
        for base in ({}, "x", [], None, {**BASE, "extra": 1}, {**BASE, "keys": "x"}, {"path": "p", "keys": [], "palette": []}):
            with self.subTest(base=base):
                self.bad(shape, ext_json(base=base) if base is not None else json.dumps(
                    {**MERGED, "meta": {"tile_size": 16, "view": "top-down", "extend": {"new_keys": NEW, "base": None}}}))

    def test_base_keys_must_partition_the_plan_keys(self):
        msg = r"^base.keys must hold every plan tile key that is not in new_keys, and none of new_keys$"
        self.bad(msg, ext_json(base={**BASE, "keys": KEYS[:5]}))  # o0#1 missing
        self.bad(msg, ext_json(base={**BASE, "keys": KEYS[:7]}))  # holds a new key
        self.bad(msg, ext_json(base={**BASE, "keys": KEYS}))  # holds all
        self.bad(msg, ext_json(base={**BASE, "keys": []}))
        self.assertEqual(json.loads(run(ext_json(base={**BASE, "keys": KEYS[:6] + ["old#0"]}))[4])["base"]["keys"][-1], "old#0")

    def test_new_key_without_a_job(self):
        with self.assertRaisesRegex(ValueError, r"new_keys without a job: \['zz#0'\]"):
            run(ext_json(new=NEW + ["zz#0"], base=None))

    def test_new_keys_must_be_strings(self):
        with self.assertRaisesRegex(ValueError, "new_keys must contain only strings"):
            run(ext_json(new=[1], base=None))

    def test_error_is_raised_before_any_output(self):
        with self.assertRaises(ValueError):
            run(ext_json(base={**BASE, "keys": []}))


class RoundTripTests(unittest.TestCase):
    """Plan (extend, fake LLM) -> Expand: new jobs continue after the base indices."""

    def test_plan_extend_then_expand(self):
        base_plan = plan_schema.validate_plan(_support.valid_plan(2, 1, 0), 10)
        base = {"path": "tilemap/a_", "keys": ["g0#0", "g1#0", "o0#0"], "palette": ["#112233"], "snap": {}}
        existing = json.dumps({"tile_size": 16, "view": "top-down", "plan": base_plan, "jobs": [], "base": base})
        add = {"theme": "test forest", "tiles": [{"id": "v0", "name": "Flowers", "category": "overlay",
                                                    "description": "small red flowers in the grass", "variants": 2}]}
        with tempfile.TemporaryDirectory() as user, \
                mock.patch.object(test_node_plan.P, "folder_paths", types.SimpleNamespace(get_user_directory=lambda: user)), \
                mock.patch.object(llm_client, "chat", return_value=json.dumps(add)), contextlib.redirect_stdout(io.StringIO()):
            text, summary, used = test_node_plan.P.CCN_TilemapPlan().plan(
                "flowers", "top-down", 16, 10, "claude-sonnet-5", plan_mode="extend", existing_plan=existing)
            prompts, w, h, seeds, job_json = test_node_plan.P.CCN_TilemapExpand().expand(text, "S", 50, 512, model=used)
        job = json.loads(job_json)
        self.assertEqual([(j["tile_id"], j["variant"], j["index"]) for j in job["jobs"]], [("v0", 0, 3), ("v0", 1, 4)])
        self.assertEqual((seeds, len(prompts)), ([53, 54], 2))
        self.assertEqual(job["base"], base)
        self.assertEqual((job["model"], len(job["plan"]["tiles"])), ("claude-sonnet-5", 4))
        self.assertIn("extension: 2 new Flux jobs", summary)
        self.assertNotIn("extend", job["plan"])


if __name__ == "__main__":
    unittest.main()
