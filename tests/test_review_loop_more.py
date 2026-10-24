"""Tests for ``nodes/tilemap_review_loop.py`` (D13b, part 2): vision review, repair, fallback, strict mode, LLM
budget (``2 * max_rounds``), LLM scene layout, private helpers (``_fit``, ``_grid``, ``_combo``) and the node class
``CCN_TilemapReviewLoop``. Helpers (fake chat / render, image builders) come from ``tests/test_review_loop.py``.
Fake callables only: no network, GPU or ComfyUI. ``comfy`` is faked through ``mock.patch.dict(sys.modules)``
(restored automatically); the torch round trip of ``run()`` is skipped when torch is missing.

Residual gaps: real ``llm_client.chat`` (default ``chat`` argument) is covered by ``llm_client`` tests; the
``interrupt`` path is only checked with a plain ``RuntimeError`` (anything that is not ``LLMProxyError`` /
``ReviewError`` propagates); image ids longer than 32 characters (excluded from the vision catalogue) cannot occur
with plan-validated ids and are not tested; ``run(images=None)`` would call the real ComfyUI ``render_jobs``.
"""

import importlib.util
import json
import sys
import unittest
from unittest import mock

import numpy as np

from tests.test_loop_state import PLAN, job_text, load, opts_kw, unload
from tests import test_review_loop as base
from tests.test_review_loop import KEYS, Chat, Render, bad, good, log_of, never, run
from tilemap_core import llm_client, review, scene_layout

LS = RL = None
CAT = {"tiles": [{k: t[k] for k in ("id", "category", "variants", "footprint") if k in t} for t in PLAN["tiles"]],
       "transitions": PLAN["transitions"]}
SCENE = json.dumps(scene_layout.auto_layout(CAT, "top-down", 7)[0])


def setUpModule():
    global LS, RL
    LS, RL = load()
    base.LS, base.RL = LS, RL  # the shared ``run`` helper reads the modules from its own module globals


def tearDownModule():
    base.LS = base.RL = None
    unload()


def verdict(ok=True, score=90, issues=()):
    return json.dumps({"pass": ok, "score": score, "summary": "s", "issues": list(issues)})


SEAM = {"tile": "g0", "kind": "seam", "severity": "high"}


class VisionReviewTests(unittest.TestCase):
    def test_vision_pass_sends_two_images_and_stops(self):
        chat = Chat(verdict())
        res = run(good(), chat=chat, max_rounds=3, review_mode="vision", review_model="claude-haiku-4-5-20251001")
        log = log_of(res)
        self.assertEqual((log["stop"], log["llm_calls"], len(log["rounds"])), ("passed", 1, 1))
        entry = log["rounds"][0]
        self.assertEqual((entry["review_mode"], entry["score"], entry["passed"], entry["summary"]),
                         ("vision", 90, True, "s"))
        messages, model, kw = chat.calls[0]
        self.assertEqual((model, kw), ("claude-haiku-4-5-20251001", {"max_tokens": 1500, "timeout": 120}))
        parts = [p for p in messages[1]["content"] if p["type"] == "image_url"]
        self.assertEqual(len(parts), 2)
        self.assertTrue(all(p["image_url"]["url"].startswith("data:image/") for p in parts))

    def test_vision_failure_triggers_regeneration(self):
        chat = Chat(verdict(False, 50, [SEAM]))
        res = run(good(), chat=chat, render=Render(dict(zip(KEYS, good()))), max_rounds=1, review_mode="vision+heuristic")
        entry = log_of(res)["rounds"][0]
        self.assertEqual((entry["review_mode"], entry["passed"], entry["regen"]), ("vision+heuristic", False, ["g0#0"]))
        self.assertEqual(entry["issues"], [{"tile": "g0", "kind": "seam", "severity": "high", "source": "vision"}])

    def test_high_issue_blocks_pass_even_with_high_score(self):
        chat = Chat(verdict(False, 95, [SEAM]))
        res = run(good(), chat=chat, render=Render(dict(zip(KEYS, good()))), max_rounds=1, review_mode="vision")
        self.assertEqual(log_of(res)["stop"], "max_rounds")

    def test_pass_score_boundary(self):
        for score, stop in ((80, "passed"), (79, "max_rounds")):
            chat = Chat(verdict(score >= 80, score, [] if score >= 80 else [dict(SEAM, severity="low")]))
            res = run(good(), chat=chat, render=Render(dict(zip(KEYS, good()))), max_rounds=1, review_mode="vision")
            self.assertEqual(log_of(res)["stop"], stop, score)

    def test_one_repair_call_for_invalid_reply(self):
        chat = Chat("not json", verdict())
        res = run(good(), chat=chat, max_rounds=1, review_mode="vision")
        log = log_of(res)
        self.assertEqual((log["stop"], log["llm_calls"], len(chat.calls)), ("passed", 2, 2))
        repair = json.dumps(chat.calls[1][0])
        self.assertIn("[image omitted]", repair)
        self.assertNotIn("data:image", repair)

    def test_failed_repair_falls_back_to_heuristics_and_continues(self):
        chat = Chat("nope", "still nope")
        res = run(good(), chat=chat, max_rounds=3, review_mode="vision")
        log = log_of(res)
        entry = log["rounds"][0]
        self.assertEqual((log["stop"], log["llm_calls"], entry["review_mode"]), ("passed", 2, "heuristic"))
        self.assertTrue(0 < len(entry["fallback"]) <= 200)
        self.assertNotIn("score", entry)  # heuristic-only verdict

    def test_llm_error_text_is_bounded_and_sanitised(self):
        err = llm_client.LLMProxyError("boom {secret} <b>" + "x" * 500)
        chat = Chat(err, err)
        res = run(bad(), chat=chat, render=Render(dict(zip(KEYS, bad()))), max_rounds=2, review_mode="vision")
        log = log_of(res)
        fb = log["rounds"][0]["fallback"]
        self.assertEqual((len(fb), fb[:12]), (200, "boom secret "))
        self.assertFalse(set("{}<>") & set(fb))
        self.assertEqual((log["stop"], len(log["rounds"][0]["regen"]), len(log["rounds"])), ("max_rounds", 2, 2))

    def test_strict_raises_with_reason(self):
        for chat in (Chat(llm_client.LLMProxyError("down")), Chat("bad", "bad")):
            with self.assertRaisesRegex(ValueError, r"^vision review failed \(on_review_error=strict\): .+"):
                run(good(), chat=chat, max_rounds=1, review_mode="vision", on_review_error="strict")
        with self.assertRaisesRegex(ValueError, r"\): down$"):
            run(good(), chat=Chat(llm_client.LLMProxyError("down")), max_rounds=1, review_mode="vision",
                on_review_error="strict")

    def test_unexpected_errors_propagate(self):
        with self.assertRaises(RuntimeError):
            run(good(), chat=Chat(RuntimeError("bug")), max_rounds=1, review_mode="vision")
    @staticmethod
    def catalogue_text(**opt):
        chat = Chat(verdict())
        run(good(), chat=chat, max_rounds=1, review_mode="vision", **opt)
        return chat.calls[0][0][1]["content"][0]["text"]

    def test_catalogue_text_lists_tileable_flags_and_wang_rows(self):
        text = self.catalogue_text()
        self.assertIn("1 | g0#0 | ground | 1x1 | tileable\n", text)
        self.assertIn("3 | o0#0 | object | 1x1 | not tileable\n", text)
        self.assertIn("5 | g0_g1#0 | ground | 1x1 | tileable\n", text)
        self.assertNotIn("g0_g1", self.catalogue_text(transitions=False))

    def test_wang_issue_blames_both_parent_ground_tiles(self):
        wang = dict(SEAM, tile="g0_g1#5", kind="transition")
        res = run(good(), chat=Chat(verdict(False, 50, [wang])), render=Render(dict(zip(KEYS, good()))), max_rounds=1,
                  review_mode="vision")
        entry = log_of(res)["rounds"][0]
        self.assertEqual((entry["issues"][0]["tile"], entry["regen"]), ("g0_g1#5", ["g0#0", "g1#0"]))
        # transitions off: the catalogue has no Wang rows, so the reference is unknown and the issue is dropped
        res = run(good(), chat=Chat(verdict(False, 50, [wang])), render=never, max_rounds=1, review_mode="vision",
                  transitions=False)
        entry = log_of(res)["rounds"][0]
        self.assertEqual((entry["issues"], entry["regen"], log_of(res)["stop"]), ([], [], "no_jobs"))

    def test_heuristic_issues_are_merged_into_a_vision_verdict(self):
        res = run(bad(), chat=Chat(verdict()), max_rounds=1, review_mode="vision+heuristic")
        entry = log_of(res)["rounds"][0]
        self.assertEqual((log_of(res)["stop"], entry["passed"]), ("passed", True))
        self.assertEqual([(i["tile"], i["kind"], i["source"]) for i in entry["issues"]],
                         [("o0#0", "fringe", "heuristic"), ("v0#0", "fringe", "heuristic")])

    def test_passing_verdict_is_withheld_while_reverted(self):
        seam = dict(SEAM, tile="o0", severity="medium")
        chat = Chat(verdict(False, 50, [seam]), verdict(True, 90, [seam]), "x", "x")
        res = run(good(), chat=chat, render=Render(dict(zip(KEYS, good()))), max_rounds=2, review_mode="vision")
        log = log_of(res)
        self.assertEqual([(r["reverted"], r["passed"]) for r in log["rounds"]], [([], False), (["o0#0"], True)])
        self.assertEqual(log["stop"], "max_rounds")  # round 2 passed but was reverted, so it is not final

    def test_best_version_tracks_the_lowest_weight_seen(self):
        def reply(severity):
            return verdict(False, 50, [dict(SEAM, tile="o0", severity=severity)])

        chat = Chat(reply("high"), reply("low"), reply("medium"), "x", "x", "x")  # weights 4, 1, 2
        res = run(good(), chat=chat, render=Render(dict(zip(KEYS, good()))), max_rounds=3, review_mode="vision")
        rounds = log_of(res)["rounds"]
        self.assertEqual([r["reverted"] for r in rounds], [[], [], ["o0#0"]])  # 2 is not below the best (1)


class BudgetAndLayoutTests(unittest.TestCase):
    def test_budget_is_two_calls_per_round(self):
        for rounds in (1, 2, 3):
            chat = Chat(*["junk"] * 20)
            res = run(bad(), chat=chat, render=Render(dict(zip(KEYS, bad()))), max_rounds=rounds, max_attempts=5,
                      review_mode="vision")
            self.assertEqual((len(chat.calls), log_of(res)["llm_calls"]), (2 * rounds, 2 * rounds), rounds)

    def test_layout_counts_against_budget_and_repair_is_refused(self):
        chat = Chat(SCENE, "junk", "junk", "junk")
        res = run(bad(), chat=chat, render=Render(dict(zip(KEYS, bad()))), max_rounds=1, review_mode="vision",
                  layout_source="llm")
        log = log_of(res)
        self.assertEqual((len(chat.calls), log["llm_calls"], log["layout_source"]), (2, 2, "llm"))
        self.assertEqual(log["rounds"][0]["fallback"], "LLM call budget used up")
        self.assertEqual(chat.calls[0][2], {"max_tokens": 8000, "timeout": 120})

    def test_llm_layout_variants(self):
        log = log_of(run(good(), chat=Chat(SCENE), max_rounds=1, layout_source="llm"))
        self.assertEqual((log["layout_source"], log["layout_problem"], log["llm_calls"]), ("llm", "", 1))
        log = log_of(run(good(), chat=Chat("no scene"), max_rounds=1, layout_source="llm"))
        self.assertEqual((log["layout_source"], log["llm_calls"]), ("auto_fallback", 1))
        self.assertIn("no JSON object found", log["layout_problem"])
        log = log_of(run(good(), chat=Chat(llm_client.LLMProxyError("proxy {down}")), max_rounds=1, layout_source="llm"))
        self.assertEqual((log["layout_source"], log["layout_problem"]), ("auto_fallback", "proxy down"))

    def test_auto_layout_makes_no_llm_call(self):
        log = log_of(run(good(), max_rounds=1, layout_source="auto"))
        self.assertEqual((log["layout_source"], log["llm_calls"]), ("auto", 0))

    def test_llm_scene_changes_rendered_scene(self):
        a = run(good(), max_rounds=1)[2]
        scene = json.loads(SCENE)
        scene["terrain"] = [["g1"] * len(row) for row in scene["terrain"]]
        b = run(good(), chat=Chat(json.dumps(scene)), max_rounds=1, layout_source="llm")[2]
        self.assertEqual(a.shape, b.shape)
        self.assertFalse(np.array_equal(a, b))


class HelperTests(unittest.TestCase):
    def test_fit_downscales_only_above_limit(self):
        limit = review.IMAGE_MAX_SIDE
        small = np.zeros((limit, 10, 3), np.uint8)
        self.assertIs(RL._fit(small), small)
        self.assertEqual(RL._fit(np.zeros((limit + 1, 10, 3), np.uint8)).shape[0], limit)
        self.assertEqual(RL._fit(np.zeros((10, 2 * limit, 3), np.uint8)).shape, (5, limit, 3))
        self.assertEqual(RL._fit(np.zeros((2, 4 * limit, 3), np.uint8)).shape, (1, limit, 3))

    def test_grid_labels_each_tile(self):
        rows = [{"id": "a", "variant": 0}, {"id": "b", "variant": 0}]
        cells = {"a#0": np.full((16, 16, 4), 255, np.uint8), "b#0": np.zeros((16, 32, 4), np.uint8)}
        grid = RL._grid(rows, cells, 16)
        self.assertEqual((grid.ndim, grid.shape[2], grid.dtype), (3, 3, np.uint8))
        self.assertGreater(grid.shape[1], 2 * 32)

    def test_combo_uses_comfy_lists_when_present(self):
        fake = mock.Mock()
        fake.KSampler.SAMPLERS = ["a", "b"]
        with mock.patch.dict(sys.modules, {"comfy": mock.Mock(samplers=fake), "comfy.samplers": fake}):
            self.assertEqual(RL._combo("SAMPLERS", "euler"), ["a", "b"])
        with mock.patch.dict(sys.modules, {"comfy": None, "comfy.samplers": None}):
            self.assertEqual(RL._combo("SAMPLERS", "euler"), ["euler"])


class OutputTests(unittest.TestCase):
    def test_scene_is_rgb_of_the_rendered_atlas(self):
        flat = np.full((64, 64, 3), (200, 30, 90), np.uint8)
        scene = run([flat] + good()[1:], max_rounds=0)[2]
        self.assertEqual(scene[0, 0].tolist(), [200, 30, 90])  # top-left cell is g0

    def test_fringe_regeneration_raises_key_tolerance_in_job_json(self):
        snap = json.loads(run(bad(), render=Render(dict(zip(KEYS, bad()))), max_rounds=2)[1])["snap"]
        self.assertEqual(snap["key_tolerance"], 60)
        self.assertEqual(json.loads(run(good(), max_rounds=2)[1])["snap"]["key_tolerance"], 40)


class NodeClassTests(unittest.TestCase):
    def test_declaration(self):
        cls = RL.CCN_TilemapReviewLoop
        self.assertTrue(cls.INPUT_IS_LIST)
        self.assertEqual((cls.RETURN_TYPES, cls.RETURN_NAMES, cls.OUTPUT_IS_LIST, cls.FUNCTION, cls.CATEGORY),
                         (("IMAGE", "STRING", "IMAGE", "STRING"), ("images", "job_json", "scene", "review_log"),
                          (True, False, False, False), "run", "custom_comfy_nodes/tilemap"))
        self.assertEqual(RL.NODE_CLASS_MAPPINGS, {"CCN_TilemapReviewLoop": cls})
        self.assertEqual(RL.NODE_DISPLAY_NAME_MAPPINGS,
                         {"CCN_TilemapReviewLoop": "Tilemap Review Loop (review and regenerate)"})

    def test_input_types_match_parse_options_limits(self):
        spec = RL.CCN_TilemapReviewLoop.INPUT_TYPES()
        req = spec["required"]
        self.assertEqual(set(req), set(opts_kw()) | {"model", "clip", "vae", "job_json"} - {"max_regen"} | {"max_regen_per_round"})
        self.assertEqual(set(spec["optional"]), {"images", "base_atlas", "base_alpha"})
        limits = {"max_rounds": (0, 6, 3), "max_regen_per_round": (1, 32, 8), "max_attempts": (1, 5, 3),
                  "pass_score": (0, 100, review.PASS_SCORE_DEFAULT), "palette_size": (4, 64, 24),
                  "key_tolerance": (0, 255, 40), "steps": (1, 10000, 20), "seam_blend": (0.0, 0.5, 0.25),
                  "cfg": (0.0, 100.0, 1.0), "guidance": (0.0, 100.0, 3.5)}
        for name, (lo, hi, default) in limits.items():
            cfg = req[name][1]
            self.assertEqual((cfg["min"], cfg["max"], cfg["default"]), (lo, hi, default), name)
            LS.parse_options(opts_kw(**{name: cfg["default"]}))
        self.assertEqual(req["review_mode"][0], list(LS.REVIEW_MODES))
        self.assertEqual(req["review_model"][0], list(llm_client.MODELS))
        self.assertEqual((req["transitions"], req["job_json"][1]), (("BOOLEAN", {"default": True}), {"forceInput": True}))

    def test_run_rejects_base_inputs_before_loading_torch(self):
        node = RL.CCN_TilemapReviewLoop()
        kw = {k: [v] for k, v in opts_kw().items()}
        with self.assertRaisesRegex(ValueError, "tileset reuse"):
            node.run(model=["M"], clip=["C"], vae=["V"], job_json=[job_text()], base_atlas=["x"], **opts_kw())
        with self.assertRaisesRegex(ValueError, "tileset reuse"):
            node.run(model=["M"], clip=["C"], vae=["V"], job_json=[job_text()], base_alpha=["x"], **kw)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch not installed")
    def test_run_end_to_end_with_tensors(self):
        import torch

        node = RL.CCN_TilemapReviewLoop()
        tensors = [torch.from_numpy(i.astype(np.float32) / 255)[None] for i in good()]
        out, text, scene, log = node.run(model=["M"], clip=["C"], vae=["V"], job_json=[job_text()], images=tensors,
                                         **opts_kw(max_rounds=1))
        self.assertEqual((len(out), tuple(out[0].shape), tuple(scene.shape)), (4, (1, 64, 64, 3), (1, 176, 128, 3)))
        self.assertEqual(json.loads(log)["stop"], "passed")
        self.assertEqual(json.loads(text)["review"]["stop"], "passed")
        with self.assertRaisesRegex(ValueError, "must be a single value, got 2"):
            node.run(model=["M", "M"], clip=["C"], vae=["V"], job_json=[job_text()], images=tensors, **opts_kw())


if __name__ == "__main__":
    unittest.main()
