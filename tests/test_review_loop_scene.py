"""Tests for the gate-1 fixes in ``nodes/tilemap_review_loop.py``: (1) the returned ``scene`` always shows the
returned images (re-rendered when keep-best reverts or a last-round regeneration changed tiles after the round-start
render, not rendered twice otherwise); (2) ``progress`` is threaded from ``run()`` through ``review_loop`` to
``render_jobs`` (``comfy.utils.ProgressBar`` built lazily, any failure yields ``None``).

Helpers (fake chat / render, image builders, ``run``) come from ``tests/test_review_loop.py``. Fake callables only;
``comfy`` is faked through ``mock.patch.dict(sys.modules)`` (restored automatically). The ``run()`` tests need torch
and are skipped without it.

Residual gaps: ``comfy.utils.ProgressBar`` is only a fake here (the real bar needs a ComfyUI server); the real
``render_jobs`` is not driven (its ``progress`` contract is covered by ``tests/test_node_regen.py``).
"""

import importlib.util
import json
import sys
import types
import unittest
from unittest import mock

import numpy as np

from tests import _support
from tests import test_review_loop as base
from tests.test_loop_state import opts_kw, job_text, load, unload
from tests.test_review_loop import KEYS, Render, bad, floats, good, log_of, periodic, run
from tilemap_core import scene_layout, snap

LS = RL = None
HAS_TORCH = importlib.util.find_spec("torch") is not None


def setUpModule():
    global LS, RL
    LS, RL = load()
    base.LS, base.RL = LS, RL  # the shared ``run`` helper reads the modules from its own module globals


def tearDownModule():
    base.LS = base.RL = None
    unload()


def alt(fg):
    """Like ``bad()`` (white key colour, ``fringe`` on o0/v0) but with a differently coloured object."""
    disc = _support.circle_on_color(64, bg=(255, 255, 255), fg=fg, radius_frac=0.3)
    return [periodic(1), periodic(2), disc, disc]


def fresh(res, scene_fn, **opt):
    """Independent render of the returned atlas: same scene function / layout / snap as the node, unpatched."""
    o = LS.parse_options(opts_kw(**opt))
    data, size, view, plan, jobs, seed = LS.parse_job_json(res[1])
    given = data.get("snap") or {}
    cat = {"tiles": [{k: t[k] for k in ("id", "category", "variants", "footprint") if k in t} for t in plan["tiles"]],
           "transitions": plan["transitions"]}
    layout = scene_layout.auto_layout(cat, view, seed)[0]
    cur = dict(zip(KEYS, res[0]))
    return scene_fn(cur, jobs, plan, size, seed, snap.check_snap_params({**o.snap, **given}), True, layout)[2][..., :3]


class SeqRender:
    """Fake generator whose n-th call returns the n-th ``source`` (to make successive regenerations differ)."""

    def __init__(self, *sources):
        self.sources, self.calls = sources, []

    def __call__(self, model, clip, vae, jobs, params, negative):
        src = self.sources[len(self.calls)]
        self.calls.append(jobs)
        return floats([src[f"{j['tile_id']}#{j['variant']}"] for j in jobs])


class SceneTests(unittest.TestCase):
    def spy(self):
        """Wrap ``RL._scene``; return ``(original, list of rgb arrays rendered so far)``."""
        orig, seen = RL._scene, []

        def wrapper(*args):
            out = orig(*args)
            seen.append(out[2][..., :3].copy())
            return out

        patcher = mock.patch.object(RL, "_scene", wrapper)
        patcher.start()
        self.addCleanup(patcher.stop)
        return orig, seen

    def test_last_round_regeneration_is_shown(self):
        orig, seen = self.spy()
        imgs = bad()
        res = run(imgs, render=Render(dict(zip(KEYS, alt((200, 60, 60))))), max_rounds=1)
        self.assertEqual(log_of(res)["stop"], "max_rounds")
        self.assertEqual([a is b for a, b in zip(res[0], imgs)], [True, True, False, False])
        self.assertEqual(len(seen), 2)  # round-start render + one re-render after the regeneration
        self.assertFalse(np.array_equal(res[2], seen[0]))  # the preview is not the stale round-start one
        np.testing.assert_array_equal(res[2], seen[1])
        np.testing.assert_array_equal(res[2], fresh(res, orig, max_rounds=1))

    def test_revert_then_regeneration_is_shown(self):
        orig, seen = self.spy()
        render = SeqRender(dict(zip(KEYS, alt((200, 60, 60)))), dict(zip(KEYS, alt((60, 60, 200)))))
        res = run(bad(), render=render, max_rounds=2)
        log = log_of(res)
        self.assertEqual((log["stop"], log["rounds"][1]["reverted"], log["rounds"][1]["regen"]),
                         ("max_rounds", ["o0#0", "v0#0"], ["o0#0", "v0#0"]))
        self.assertEqual(len(seen), 3)
        self.assertFalse(np.array_equal(res[2], seen[1]))  # round 2 showed the first regeneration
        np.testing.assert_array_equal(res[2], fresh(res, orig, max_rounds=2))

    def test_no_jobs_after_revert_shows_the_reverted_tiles(self):
        orig, seen = self.spy()
        imgs = bad()
        res = run(imgs, render=Render(dict(zip(KEYS, alt((200, 60, 60))))), max_rounds=3, max_attempts=1)
        self.assertEqual(log_of(res)["stop"], "no_jobs")
        self.assertTrue(all(a is b for a, b in zip(res[0], imgs)))  # kept = the very first versions
        self.assertEqual(len(seen), 3)  # two round starts + one re-render of the reverted atlas
        self.assertFalse(np.array_equal(seen[1], seen[0]))  # round 2 started on the regenerated tiles
        np.testing.assert_array_equal(res[2], seen[0])  # ...but the returned scene shows the originals again
        np.testing.assert_array_equal(res[2], fresh(res, orig, max_rounds=3, max_attempts=1))

    def test_nothing_changed_renders_the_scene_once(self):
        orig, seen = self.spy()
        res = run(good(), max_rounds=3)
        self.assertEqual(log_of(res)["stop"], "passed")
        self.assertEqual(len(seen), 1)
        np.testing.assert_array_equal(res[2], seen[0])
        np.testing.assert_array_equal(res[2], fresh(res, orig, max_rounds=3))

    def test_passed_after_a_kept_regeneration_renders_once_per_round(self):
        orig, seen = self.spy()
        res = run(bad(), render=Render(dict(zip(KEYS, good()))), max_rounds=3)
        self.assertEqual(log_of(res)["stop"], "passed")
        self.assertEqual(len(seen), 2)  # round 1 + round 2 (which passes): no extra re-render
        np.testing.assert_array_equal(res[2], seen[1])
        np.testing.assert_array_equal(res[2], fresh(res, orig, max_rounds=3))

    def test_zero_rounds_identity_and_single_render(self):
        orig, seen = self.spy()
        imgs, text = good(), job_text()
        out, new_text, scene, log = run(imgs, text=text, max_rounds=0)
        self.assertIs(new_text, text)
        self.assertTrue(all(a is b for a, b in zip(out, imgs)))
        self.assertEqual(len(seen), 1)
        np.testing.assert_array_equal(scene, seen[0])
        np.testing.assert_array_equal(scene, fresh((out, text, scene, log), orig, max_rounds=0))


class Bar:
    """Fake ``comfy.utils.ProgressBar``: remembers every instance's total."""

    made = []

    def __init__(self, total):
        self.total, self.updates = total, []
        Bar.made.append(self)

    def update(self, n):
        self.updates.append(n)


class ProgressRender(Render):
    """Like ``Render`` but with the ``progress`` parameter of ``render_jobs``: one ``update(1)`` per job."""

    def __init__(self, source):
        super().__init__(source)
        self.progress = []

    def __call__(self, model, clip, vae, jobs, params, negative, progress=None):
        self.progress.append(progress)
        out = super().__call__(model, clip, vae, jobs, params, negative)
        for _ in jobs:
            if progress is not None:
                progress.update(1)
        return out


def fake_comfy(bar=Bar):
    utils = types.ModuleType("comfy.utils")
    utils.ProgressBar = bar
    return mock.patch.dict(sys.modules, {"comfy": mock.Mock(utils=utils), "comfy.utils": utils})


def loop(images, render, progress, **opt):
    o = LS.parse_options(opts_kw(**opt))
    return RL.review_loop("M", "C", "V", job_text(), images, o, chat=base.never, render=render, progress=progress)


class ProgressTests(unittest.TestCase):
    def setUp(self):
        Bar.made = []
        self.addCleanup(setattr, Bar, "made", [])

    def test_default_passes_no_progress_keyword(self):
        seen = []

        def render(model, clip, vae, jobs, params, negative, **kw):
            seen.append(kw)
            return floats([bad()[i] for i in (2, 3)])

        o = LS.parse_options(opts_kw(max_rounds=1))
        RL.review_loop("M", "C", "V", job_text(), bad(), o, chat=base.never, render=render)
        self.assertEqual(seen, [{}])

    def test_progress_object_reaches_render_for_regeneration(self):
        render, progress = ProgressRender(dict(zip(KEYS, alt((200, 60, 60))))), Bar(2)
        loop(bad(), render, progress, max_rounds=1)
        self.assertEqual(render.progress, [progress])
        self.assertEqual(progress.updates, [1, 1])  # o0 and v0 regenerated

    def test_progress_reaches_first_pass_render_too(self):
        render, progress = ProgressRender(dict(zip(KEYS, good()))), Bar(4)
        res = loop(None, render, progress, max_rounds=1)
        self.assertEqual(log_of(res)["stop"], "passed")
        self.assertEqual((render.progress, progress.updates), ([progress], [1] * 4))

    def test_progress_total_counts_first_pass_and_regeneration_cap(self):
        text = job_text()  # four jobs
        for opt, first, want in ((dict(max_rounds=3), True, 4 + 3 * 4), (dict(max_rounds=3), False, 12),
                                 (dict(max_rounds=2, max_regen_per_round=1), False, 2),
                                 (dict(max_rounds=2, max_regen_per_round=1), True, 6),
                                 (dict(max_rounds=0), True, 4)):
            Bar.made.clear()
            with fake_comfy():
                got = RL._progress(LS.parse_options(opts_kw(**opt)), text, first)
            self.assertIs(got, Bar.made[0])
            self.assertEqual(got.total, want, (opt, first))

    def test_no_planned_units_builds_no_bar(self):
        with fake_comfy():
            got = RL._progress(LS.parse_options(opts_kw(max_rounds=0)), job_text(), False)
        self.assertEqual((got, Bar.made), (None, []))

    def test_progress_is_none_without_comfy_or_on_any_failure(self):
        o = LS.parse_options(opts_kw(max_rounds=1))

        class Boom:
            def __init__(self, total):
                raise RuntimeError("no server")

        with mock.patch.dict(sys.modules, {"comfy": None, "comfy.utils": None}):
            self.assertIsNone(RL._progress(o, job_text(), True))
        with fake_comfy(Boom):
            self.assertIsNone(RL._progress(o, job_text(), True))
        with fake_comfy():
            self.assertIsNone(RL._progress(o, "{not json", True))
            self.assertEqual(Bar.made, [])

    @unittest.skipUnless(HAS_TORCH, "torch is needed for the tensor round trip of run()")
    def test_run_builds_bar_and_passes_it_to_review_loop(self):
        import torch

        tensors = [torch.from_numpy(i.astype(np.float32) / 255)[None] for i in good()]
        node = RL.CCN_TilemapReviewLoop()
        with fake_comfy(), mock.patch.object(RL, "review_loop", wraps=RL.review_loop) as spy:
            node.run(model=["M"], clip=["C"], vae=["V"], job_json=[job_text()], images=tensors, **opts_kw(max_rounds=2))
        self.assertEqual([b.total for b in Bar.made], [2 * 4])  # images given: no first pass, 2 rounds x 4 tiles
        self.assertIs(spy.call_args.kwargs["progress"], Bar.made[0])

    @unittest.skipUnless(HAS_TORCH, "torch is needed for the tensor round trip of run()")
    def test_run_works_without_comfy(self):
        import torch

        tensors = [torch.from_numpy(i.astype(np.float32) / 255)[None] for i in good()]
        node = RL.CCN_TilemapReviewLoop()
        with mock.patch.dict(sys.modules, {"comfy": None, "comfy.utils": None}), \
                mock.patch.object(RL, "review_loop", wraps=RL.review_loop) as spy:
            out, text, scene, log = node.run(model=["M"], clip=["C"], vae=["V"], job_json=[job_text()], images=tensors,
                                             **opts_kw(max_rounds=1))
        self.assertIsNone(spy.call_args.kwargs["progress"])
        self.assertEqual((len(out), json.loads(log)["stop"]), (4, "passed"))


if __name__ == "__main__":
    unittest.main()
