"""Tests for ``CCN_TilemapExpand`` of ``nodes/tilemap_plan.py`` (D16c). Stdlib unittest; no LLM, no network, no disk.

Covers: widget/return contract, list outputs and their order, ``job_json`` content (``tile_size``/``view``/``plan``/``jobs``,
optional ``model``, optional ``base``), seeds / cell sizes, style prefix handling, every ``ValueError`` of the meta/model
checks, and the extension path (``meta.extend`` -> ``expand.filter_new`` re-indexing after the base jobs, ``base.keys``
consistency checks of ``_extension_jobs``). ``plan_json`` strings are built by hand, not by ``CCN_TilemapPlan``.

Helpers (``setUpModule``, ``P``) come from ``tests/test_node_plan.py``.

Residual gaps: ``expand_jobs`` / ``filter_new`` arithmetic (clamping, flooring, seed wrap) is covered by
``tests/test_expand_filter_new.py``; only a few values are re-checked here to prove the node passes its widgets through
in the right order. Planner -> expand round trips of the real LLM path are not run.
"""

import copy
import json
import unittest

from tests import _support, test_node_plan
from tests.test_node_plan import setUpModule, tearDownModule  # noqa: F401  (module fixtures)
from tilemap_core import expand, llm_client, plan_schema

PLAN = plan_schema.validate_plan(_support.valid_plan(2, 1, 1, variants=2), 10)  # 8 jobs, ids g0 g1 o0 v0


def plan_json(plan=PLAN, **meta):
    meta = {"tile_size": 16, "view": "top-down", **meta}
    return json.dumps({**plan, "meta": meta})


def run(text=None, style="GRPZA", seed=100, cell_px=512, **kw):
    return test_node_plan.P.CCN_TilemapExpand().expand(plan_json() if text is None else text, style, seed, cell_px, **kw)


class ContractTests(unittest.TestCase):
    def test_class_attributes(self):
        cls = test_node_plan.P.CCN_TilemapExpand
        self.assertEqual(cls.RETURN_TYPES, ("STRING", "INT", "INT", "INT", "STRING"))
        self.assertEqual(cls.RETURN_NAMES, ("prompt", "width", "height", "seed", "job_json"))
        self.assertEqual(cls.OUTPUT_IS_LIST, (True, True, True, True, False))
        self.assertEqual((cls.FUNCTION, cls.CATEGORY), ("expand", "custom_comfy_nodes/tilemap"))
        self.assertEqual(len(cls.RETURN_TYPES), len(cls.RETURN_NAMES))
        self.assertEqual(len(cls.RETURN_TYPES), len(cls.OUTPUT_IS_LIST))

    def test_input_types(self):
        spec = test_node_plan.P.CCN_TilemapExpand.INPUT_TYPES()
        self.assertEqual(list(spec["required"]), ["plan_json", "style_prefix", "seed", "cell_px"])
        self.assertEqual(spec["required"]["plan_json"], ("STRING", {"forceInput": True}))
        self.assertEqual(spec["required"]["style_prefix"], ("STRING", {"default": "GRPZA, umempart, pixel art"}))
        self.assertEqual(spec["required"]["seed"], ("INT", {"default": 0, "min": 0, "max": 2**64 - 1}))
        self.assertEqual(spec["required"]["cell_px"], ("INT", {"default": 512, "min": 256, "max": 1024, "step": 64}))
        self.assertEqual(spec["optional"], {"model": ("STRING", {"forceInput": True, "default": ""})})

    def test_widget_bounds_match_expand_limits(self):
        req = test_node_plan.P.CCN_TilemapExpand.INPUT_TYPES()["required"]
        self.assertEqual((req["cell_px"][1]["min"], req["cell_px"][1]["max"]), expand._CELL_PX_RANGE)
        self.assertEqual(req["seed"][1]["max"], expand.SEED_LIMIT - 1)
        # the widget extremes are accepted by the node
        run(seed=0, cell_px=256)
        run(seed=2**64 - 1, cell_px=1024)


class ExpandTests(unittest.TestCase):
    def test_lists_and_job_json(self):
        prompts, widths, heights, seeds, text = run()
        job = json.loads(text)
        self.assertEqual(set(job), {"tile_size", "view", "plan", "jobs"})
        self.assertEqual((job["tile_size"], job["view"]), (16, "top-down"))
        self.assertEqual(job["plan"], PLAN)  # meta dropped, plan normalised
        self.assertEqual(len(job["jobs"]), 8)
        self.assertEqual(job["jobs"], expand.expand_jobs(PLAN, 16, "top-down", "GRPZA", 100, 512))
        self.assertEqual(prompts, [j["prompt"] for j in job["jobs"]])
        self.assertEqual((widths, heights), ([512] * 8, [512] * 8))
        self.assertEqual(seeds, list(range(100, 108)))
        for out in (prompts, widths, heights, seeds):
            self.assertIs(type(out), list)
        self.assertIs(type(text), str)
        self.assertEqual(text, json.dumps(job, separators=(",", ":")))

    def test_job_order_is_category_then_plan_then_variant(self):
        job = json.loads(run()[4])
        self.assertEqual([f"{j['tile_id']}#{j['variant']}" for j in job["jobs"]],
                         ["g0#0", "g0#1", "g1#0", "g1#1", "o0#0", "o0#1", "v0#0", "v0#1"])
        self.assertEqual([j["index"] for j in job["jobs"]], list(range(8)))

    def test_widgets_reach_expand_jobs_in_the_right_places(self):
        prompts, widths, heights, seeds, text = run(style="my style", seed=7, cell_px=256)
        self.assertTrue(all(p.startswith("my style, top-down view") for p in prompts))
        self.assertEqual((widths[0], heights[0], seeds[:2]), (256, 256, [7, 8]))
        big = {"theme": "t", "tiles": [{"id": "o0", "name": "Tree", "category": "object", "description": "a big tree for the map",
                                          "footprint": [2, 3]}]}
        _, w, h, _, _ = run(plan_json(plan_schema.validate_plan(big, 5), tile_size=24, view="side-view"), cell_px=512)
        self.assertEqual((w, h), ([682 // 16 * 16], [1024]))

    def test_seed_wraps_at_2_pow_64(self):
        seeds = run(seed=2**64 - 1)[3]
        self.assertEqual(seeds[:3], [2**64 - 1, 0, 1])

    def test_style_prefix_is_cleaned_and_may_be_empty(self):
        prompts = run(style="")[0]
        self.assertTrue(prompts[0].startswith("top-down view"))
        prompts = run(style=" a <b> \n c ")[0]
        self.assertTrue(prompts[0].startswith("a b c, top-down"), prompts[0][:20])

    def test_tile_size_and_view_come_from_meta(self):
        for size, view in ((8, "side-view"), (64, "three-quarter")):
            job = json.loads(run(plan_json(tile_size=size, view=view))[4])
            self.assertEqual((job["tile_size"], job["view"]), (size, view))
            self.assertIn(plan_prompt_phrase(view), job["jobs"][0]["prompt"])

    def test_tile_size_text_in_meta_is_accepted(self):
        self.assertEqual(json.loads(run(plan_json(tile_size="32"))[4])["tile_size"], 32)

    def test_extra_meta_and_plan_keys_are_dropped(self):
        job = json.loads(run(plan_json(model="whatever", note="x"))[4])
        self.assertNotIn("meta", job["plan"])
        self.assertNotIn("model", job)  # the model input, not meta, sets job["model"]

    def test_model_input(self):
        self.assertNotIn("model", json.loads(run(model="")[4]))
        self.assertNotIn("model", json.loads(run(model="  \n ")[4]))
        self.assertEqual(json.loads(run(model="  claude-opus-5 ")[4])["model"], "claude-opus-5")
        self.assertEqual(json.loads(run(model="custom/m-1.0:x")[4])["model"], "custom/m-1.0:x")

    def test_input_plan_text_is_not_required_to_be_canonical(self):
        text = "Here is the plan:\n```json\n" + plan_json() + "\n```"
        self.assertEqual(run(text)[4], run()[4])


def plan_prompt_phrase(view):
    from tilemap_core import plan_prompt
    return plan_prompt.VIEWS[view]


class ErrorTests(unittest.TestCase):
    def bad(self, regex, text=None, exc=ValueError, **kw):
        with self.assertRaisesRegex(exc, regex):
            run(text, **kw)

    def test_not_a_plan_json(self):
        for text in ("", "not json", "[1]", "{"):
            with self.subTest(text=text):
                self.bad(r".", text, exc=plan_schema.PlanError)
        self.bad("reply must be text", 5, exc=plan_schema.PlanError)

    def test_missing_or_bad_meta(self):
        self.bad(r"^plan_json has no 'meta' block: connect the output of CCN_TilemapPlan$", json.dumps(PLAN))
        self.bad("no 'meta' block", json.dumps({**PLAN, "meta": []}))
        self.bad("no 'meta' block", json.dumps({**PLAN, "meta": "x"}))

    def test_meta_tile_size(self):
        for bad in (None, 0, 7, True, 16.5, "x", [16], "-16"):
            with self.subTest(tile_size=bad):
                text = json.dumps({**PLAN, "meta": {"tile_size": bad, "view": "top-down"}})
                self.bad(r"^tile_size must be one of \[8, 16, 24, 32, 48, 64\], got ", text)
        self.bad("tile_size must be", json.dumps({**PLAN, "meta": {"view": "top-down"}}))

    def test_meta_view(self):
        for bad in (None, "", "iso", 3, ["top-down"]):
            with self.subTest(view=bad):
                text = json.dumps({**PLAN, "meta": {"tile_size": 16, "view": bad}})
                self.bad(r"^plan_json meta.view must be one of \['top-down', 'three-quarter', 'side-view'\]$", text)
        self.bad("meta.view must be", json.dumps({**PLAN, "meta": {"tile_size": 16}}))

    def test_meta_is_checked_before_the_plan(self):
        self.bad("tile_size must be", json.dumps({"theme": "x", "tiles": [], "meta": {"tile_size": 1, "view": "top-down"}}))

    def test_invalid_plan_body(self):
        broken = copy.deepcopy(PLAN)
        broken["tiles"][0]["category"] = "sky"
        self.bad("category", plan_json(broken), exc=plan_schema.PlanError)
        self.bad(".", plan_json({"theme": "t", "tiles": []}), exc=plan_schema.PlanError)

    def test_plan_over_max_jobs(self):
        many = {"theme": "t", "tiles": [{"id": f"g{i}", "name": f"G {i}", "category": "ground", "description": "visual desc here",
                                          "variants": 4} for i in range(17)]}
        self.bad(r".", json.dumps({**many, "meta": {"tile_size": 16, "view": "top-down"}}), exc=plan_schema.PlanError)

    def test_model_must_be_text_and_safe(self):
        for bad in (None, 5, ["m"], b"m"):
            with self.subTest(model=bad):
                self.bad(r"^model must be text, got ", model=bad)
        for bad in ("bad name", "a" * 101, "x;y"):
            with self.subTest(model=bad):
                self.bad("invalid model name", model=bad, exc=llm_client.LLMProxyError)

    def test_widget_values(self):
        self.bad("base_seed must be an integer", seed=-1)
        self.bad("base_seed must be an integer", seed=2**64)
        self.bad("base_seed must be an integer", seed=True)
        self.bad("cell_px must be an integer", cell_px=255)
        self.bad("cell_px must be an integer", cell_px=1025)
        self.bad("cell_px must be an integer", cell_px="512")


if __name__ == "__main__":
    unittest.main()
