"""Tests for ``nodes/tilemap_review_loop.py`` (D13b, part 1): ``review_loop`` with heuristic review: round limits,
stop reasons, keep-best, regeneration bookkeeping, snap overrides, determinism. Vision/LLM paths and the node class
are in ``tests/test_review_loop_more.py`` (which imports the helpers below; functions/constants only, so no TestCase
is collected twice). Fake chat and render callables: no network, GPU or ComfyUI.

Residual gaps: ``_grid`` / ``_scene`` pixel output is only checked by shape (rendering is covered by
``scene_render`` / ``snap`` tests); real ``render_jobs`` (ComfyUI sampler) is never run; exact regeneration seeds
and prompt amendments belong to ``regen_plan`` tests; ``wang`` transition tiles are only exercised through the
default ``transitions=True`` path (no per-tile assertions on the generated transition items).
"""

import json
import unittest

import numpy as np

from tests import _support
from tests.test_loop_state import JOBS, job_text, load, opts_kw, unload
from tilemap_core.llm_client import LLMProxyError

LS = RL = None
KEYS = ["g0#0", "g1#0", "o0#0", "v0#0"]


def setUpModule():
    global LS, RL
    LS, RL = load()


def tearDownModule():
    unload()


def periodic(seed, n=64):
    """Tileable ground texture that raises no heuristic issue."""
    y, x = np.mgrid[0:n, 0:n]
    noise = np.random.default_rng(seed).integers(0, 3, (n, n))
    out = np.zeros((n, n, 3))
    for c in range(3):
        out[..., c] = 128 + 60 * np.sin(2 * np.pi * (x * (1 + c) + y * (seed + 1)) / n + c) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def circle(bg):
    return _support.circle_on_color(64, bg=bg, fg=(120, 120, 120), radius_frac=0.3)


def good():
    """Four clean tiles in job order (no heuristic issue)."""
    return [periodic(1), periodic(2), circle((255, 0, 255)), circle((255, 0, 255))]


def bad():
    """Ground fine, object + overlay get a ``fringe`` issue (white key colour)."""
    return [periodic(1), periodic(2), circle((255, 255, 255)), circle((255, 255, 255))]


def floats(images):
    return [i.astype(np.float32) / 255 for i in images]


class Chat:
    """Scripted fake LLM: pops replies, raises exceptions, records ``(messages, model, kwargs)``."""

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def __call__(self, messages, model, **kw):
        self.calls.append((messages, model, kw))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class Render:
    """Fake generator: ``source`` maps ``tile#variant`` -> uint8 image; records the jobs it was asked for."""

    def __init__(self, source):
        self.source, self.calls = source, []

    def __call__(self, model, clip, vae, jobs, params, negative):
        self.calls.append((model, clip, vae, jobs, params, negative))
        return floats([self.source[f"{j['tile_id']}#{j['variant']}"] for j in jobs])


def never(*args, **kwargs):
    raise AssertionError("must not be called")


def run(images, chat=never, render=never, text=None, **opt):
    o = LS.parse_options(opts_kw(**opt))
    return RL.review_loop("M", "C", "V", text or job_text(), images, o, chat=chat, render=render)


def log_of(result):
    return json.loads(result[3])


class RoundLimitTests(unittest.TestCase):
    def test_zero_rounds_returns_input_unchanged(self):
        imgs, text = good(), job_text()
        out, new_text, scene, log = run(imgs, text=text, max_rounds=0)
        self.assertIs(new_text, text)
        self.assertTrue(all(a is b for a, b in zip(out, imgs)))
        self.assertEqual((scene.shape, scene.dtype, scene.flags["C_CONTIGUOUS"]), ((176, 128, 3), np.uint8, True))
        self.assertEqual(json.loads(log), {"format": "ccn-review-log/1", "stop": "no_rounds", "llm_calls": 0,
                                           "layout_source": "auto", "layout_problem": "", "unreviewed": [],
                                           "rounds": []})

    def test_zero_rounds_llm_layout_has_no_budget(self):
        res = run(good(), max_rounds=0, layout_source="llm", review_mode="vision")  # chat=never: never called
        log = log_of(res)
        self.assertEqual((log["layout_source"], log["layout_problem"], log["llm_calls"]),
                         ("auto_fallback", "LLM call budget used up", 0))

    def test_layout_error_text_is_bounded_and_sanitised(self):
        err = LLMProxyError("lay {out} <b>" + "y" * 500)
        problem = log_of(run(good(), chat=Chat(err), max_rounds=1, layout_source="llm"))["layout_problem"]
        self.assertEqual((len(problem), problem[:11]), (200, "lay out b" + "yy"[:2]))
        self.assertFalse(set("{}<>") & set(problem))

    def test_images_none_renders_first_pass_once(self):
        render = Render(dict(zip(KEYS, good())))
        out, text, scene, log = run(None, render=render, max_rounds=0, negative="blurry")
        self.assertEqual(len(render.calls), 1)
        model, clip, vae, jobs, params, negative = render.calls[0]
        self.assertEqual(((model, clip, vae), jobs, negative), (("M", "C", "V"), JOBS, "blurry"))
        self.assertEqual((params.steps, params.sampler), (20, "euler"))
        for a, b in zip(out, good()):
            self.assertEqual(a.dtype, np.uint8)
            np.testing.assert_array_equal(a, b)

    def test_image_count_must_match_jobs(self):
        for n in (0, 3, 5):
            with self.assertRaisesRegex(ValueError, f"^got {n} images for 4 jobs; connect one image per job in job order$"):
                run((good() * 2)[:n])

    def test_max_rounds_is_a_hard_cap(self):
        for rounds in (1, 2, 3, 6):
            render = Render(dict(zip(KEYS, bad())))
            res = run(bad(), render=render, max_rounds=rounds, max_attempts=5)
            self.assertEqual(len(log_of(res)["rounds"]), rounds)
            self.assertEqual(len(render.calls), min(rounds, 5))  # round 6 finds all 5 attempts used
            self.assertEqual(json.loads(res[1])["review"]["rounds"], rounds)


class StopReasonTests(unittest.TestCase):
    def test_passes_at_first_clean_round(self):
        imgs = good()
        res = run(imgs, max_rounds=3)
        log = log_of(res)
        self.assertEqual((log["stop"], len(log["rounds"]), log["unreviewed"], log["llm_calls"]), ("passed", 1, [], 0))
        self.assertEqual((log["rounds"][0]["issues"], log["rounds"][0]["regen"], log["rounds"][0]["reverted"]), ([], [], []))
        self.assertTrue(all(a is b for a, b in zip(res[0], imgs)))
        self.assertEqual(json.loads(res[1])["review"], {"rounds": 1, "attempts": {}, "stop": "passed"})

    def test_max_rounds_stop_reports_unreviewed(self):
        render = Render(dict(zip(KEYS, bad())))
        log = log_of(run(bad(), render=render, max_rounds=2))
        self.assertEqual(log["stop"], "max_rounds")
        self.assertEqual(log["unreviewed"], ["o0#0", "v0#0"])
        self.assertEqual([r["regen"] for r in log["rounds"]], [["o0#0", "v0#0"]] * 2)
        self.assertEqual([i["kind"] for i in log["rounds"][0]["issues"]], ["fringe", "fringe"])
        self.assertTrue(all(i["source"] == "heuristic" and i["severity"] == "medium" for i in log["rounds"][0]["issues"]))
        self.assertEqual({j["tile_id"] for call in render.calls for j in call[3]}, {"o0", "v0"})

    def test_no_jobs_when_attempts_exhausted(self):
        render = Render(dict(zip(KEYS, bad())))
        res = run(bad(), render=render, max_rounds=5, max_attempts=1)
        log = log_of(res)
        self.assertEqual((log["stop"], len(log["rounds"]), len(render.calls)), ("no_jobs", 2, 1))
        self.assertEqual((log["unreviewed"], log["rounds"][1]["regen"]), ([], []))
        self.assertTrue(log["rounds"][1]["skipped"])
        self.assertEqual(json.loads(res[1])["review"], {"rounds": 2, "attempts": {"o0#0": 1, "v0#0": 1}, "stop": "no_jobs"})

    def test_attempt_limit_boundaries(self):
        for attempts in (1, 2, 5):
            render = Render(dict(zip(KEYS, bad())))
            log = log_of(run(bad(), render=render, max_rounds=6, max_attempts=attempts))
            self.assertEqual((len(render.calls), log["stop"]), (attempts, "no_jobs"), attempts)

    def test_max_regen_per_round_limits_batch(self):
        render = Render(dict(zip(KEYS, bad())))
        log = log_of(run(bad(), render=render, max_rounds=1, max_regen_per_round=1))
        self.assertEqual([len(c[3]) for c in render.calls], [1])
        self.assertEqual(len(log["rounds"][0]["regen"]), 1)
        render = Render(dict(zip(KEYS, bad())))
        run(bad(), render=render, max_rounds=1, max_regen_per_round=2)
        self.assertEqual(len(render.calls[0][3]), 2)

    def test_regenerated_jobs_get_new_seeds_each_round(self):
        render = Render(dict(zip(KEYS, bad())))
        run(bad(), render=render, max_rounds=2)
        first, second = ({j["tile_id"]: j["seed"] for j in c[3]} for c in render.calls)
        base = {j["tile_id"]: j["seed"] for j in JOBS}
        for tid in ("o0", "v0"):
            self.assertEqual(len({base[tid], first[tid], second[tid]}), 3)


class KeepBestTests(unittest.TestCase):
    def test_equal_weight_regeneration_is_reverted(self):
        imgs = bad()
        render = Render(dict(zip(KEYS, bad())))
        rounds = log_of(run(imgs, render=render, max_rounds=3))["rounds"]
        self.assertEqual([r["reverted"] for r in rounds], [[], ["o0#0", "v0#0"], ["o0#0", "v0#0"]])
        # a tile regenerated in the last round is not reviewed again, so it is returned as regenerated
        res = run(imgs, render=Render(dict(zip(KEYS, bad()))), max_rounds=3, max_attempts=1)
        self.assertEqual(log_of(res)["stop"], "no_jobs")
        self.assertTrue(all(a is b for a, b in zip(res[0], imgs)))  # reverted to the very first versions
        last = run(imgs, render=Render(dict(zip(KEYS, bad()))), max_rounds=1)[0]
        self.assertEqual([a is b for a, b in zip(last, imgs)], [True, True, False, False])

    def test_strictly_better_regeneration_is_kept_and_passes(self):
        imgs = bad()
        render = Render(dict(zip(KEYS, good())))
        res = run(imgs, render=render, max_rounds=3)
        log = log_of(res)
        self.assertEqual((log["stop"], len(log["rounds"]), log["rounds"][1]["reverted"], log["unreviewed"]),
                         ("passed", 2, [], []))
        self.assertIs(res[0][0], imgs[0])
        self.assertIs(res[0][1], imgs[1])
        for got, want in zip(res[0][2:], good()[2:]):
            np.testing.assert_array_equal(got, want)
        self.assertEqual(len(render.calls), 1)

    def test_worse_replacement_is_reverted_not_kept(self):
        imgs = good()
        imgs[2] = circle((255, 255, 255))  # fringe on o0 only
        source = dict(zip(KEYS, good()))
        source["o0#0"] = _support.noise_rgb(64, 64, 3)  # replacement is worse (not_cut_out + fringe)
        res = run(imgs, render=Render(source), max_rounds=3, max_attempts=1)
        log = log_of(res)
        self.assertEqual((log["rounds"][0]["regen"], log["rounds"][1]["reverted"]), (["o0#0"], ["o0#0"]))
        self.assertEqual((log["stop"], log["unreviewed"]), ("no_jobs", []))
        self.assertIs(res[0][2], imgs[2])


class SnapAndDeterminismTests(unittest.TestCase):
    def test_job_json_snap_overrides_widgets(self):
        res = run(good(), text=job_text(snap={"palette_size": 16, "key_tolerance": 10}), max_rounds=1)
        snap = json.loads(res[1])["snap"]
        self.assertEqual((snap["palette_size"], snap["key_tolerance"], snap["seam_blend"]), (16, 10, 0.25))

    def test_widget_snap_used_without_override(self):
        snap = json.loads(run(good(), max_rounds=1, palette_size=5, seam_blend=0.5)[1])["snap"]
        self.assertEqual((snap["palette_size"], snap["seam_blend"], snap["transitions"]), (5, 0.5, True))
        snap = json.loads(run(good(), max_rounds=1, transitions=False)[1])["snap"]
        self.assertFalse(snap["transitions"])

    def test_bad_snap_is_rejected(self):
        for value in ([1], "x", 5):
            with self.assertRaisesRegex(ValueError, "^job_json snap must be an object$"):
                run(good(), text=job_text(snap=value), max_rounds=1)
        for value in ({}, [], 0, "", None, False):  # falsy means no override
            run(good(), text=job_text(snap=value), max_rounds=1)
        with self.assertRaises(ValueError):
            run(good(), text=job_text(snap={"bogus": 1}), max_rounds=1)
        with self.assertRaises(ValueError):
            run(good(), text=job_text(snap={"palette_size": 257}), max_rounds=1)

    def test_invalid_job_json_is_rejected_before_any_work(self):
        with self.assertRaisesRegex(ValueError, "not valid JSON"):
            run(good(), text="{", max_rounds=1)

    def test_deterministic_and_inputs_untouched(self):
        outs = []
        for _ in range(2):
            imgs = good()[:2] + bad()[2:]
            snapshot = [i.copy() for i in imgs]
            res = run(imgs, render=Render(dict(zip(KEYS, bad()))), max_rounds=3)
            for a, b in zip(imgs, snapshot):
                np.testing.assert_array_equal(a, b)
            outs.append(res)
        (i1, j1, s1, l1), (i2, j2, s2, l2) = outs
        self.assertEqual((j1, l1), (j2, l2))
        np.testing.assert_array_equal(s1, s2)
        for a, b in zip(i1, i2):
            np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
