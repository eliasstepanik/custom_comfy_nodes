"""Tests for ``nodes/tilemap_regen.py`` (D12), part 3: ``render_jobs`` behaviour with a fake adapter (call
sequence, ordering, interrupt, progress, determinism, decoded-image checks). Validation of arguments and
``RegenParams`` is ``tests/test_node_regen.py``; ``to_rgb`` and the regen_plan hand-off are
``tests/test_node_regen_more.py``. The loader and the ``FakeAdapter`` / ``Progress`` / ``job`` / ``render``
helpers are reused from part 1 (imported by name, so no TestCase is collected twice).

Residual gaps: see ``tests/test_node_regen.py``.
"""

import types
import unittest

import numpy as np

from tests import test_node_regen as base
from tests.test_node_regen import FakeAdapter, Progress, job, render


def setUpModule():
    base.setUpModule()


def tearDownModule():
    base.tearDownModule()


class RenderJobsBehaviourTests(unittest.TestCase):
    def test_empty_jobs_never_touches_adapter(self):
        for empty in ([], ()):
            out, ad = render(empty)
            self.assertEqual(out, [])
            self.assertEqual(ad.calls, [])

    def test_empty_jobs_with_default_adapter_does_not_import_comfy(self):
        self.assertEqual(base.R.render_jobs(None, None, None, [], base.R.RegenParams(), ""), [])

    def test_empty_jobs_still_validates_arguments(self):
        with self.assertRaises(ValueError):
            base.R.render_jobs(None, None, None, [], {"bogus": 1}, "")

    def test_call_sequence_for_one_job(self):
        p = base.R.RegenParams(steps=7, cfg=2.0, guidance=1.5)
        out, ad = render([job(seed=9, w=32, h=48, prompt="hello")], params=p, neg="neg text")
        self.assertEqual([c[0] for c in ad.calls],
                         ["encode", "interrupt", "encode", "guidance", "latent", "sample", "decode", "to_rgb"])
        self.assertEqual(ad.calls[0], ("encode", "C", "neg text"))
        self.assertEqual(ad.calls[2], ("encode", "C", "hello"))
        self.assertEqual(ad.calls[3], ("guidance", ("cond", "hello"), 1.5))
        self.assertEqual(ad.calls[4], ("latent", 32, 48))
        _, model, seed, params, positive, negative, latent = ad.calls[5]
        self.assertEqual((model, seed, params), ("M", 9, p))
        self.assertEqual(positive, ("guided", "hello", 1.5))
        self.assertEqual(negative, ("cond", "neg text"))  # negative: encoded once, never guided
        self.assertEqual(ad.calls[6], ("decode", "V", 9))
        self.assertEqual(out[0].shape, (48, 32, 3))
        self.assertEqual(out[0].dtype, np.float32)

    def test_negative_encoded_once_and_never_guided(self):
        _, ad = render([job(seed=1), job(seed=2), job(seed=3)], neg="N")
        encodes = [c for c in ad.calls if c[0] == "encode"]
        self.assertEqual([c[2] for c in encodes], ["N", "a tile", "a tile", "a tile"])
        self.assertTrue(all(c[1] != ("cond", "N") for c in ad.calls if c[0] == "guidance"))

    def test_order_alignment_and_per_job_size(self):
        jobs = [job(seed=10, w=16, h=32), job(seed=20, w=64, h=16), job(seed=30, w=48, h=48)]
        out, ad = render(jobs)
        self.assertEqual([o.shape for o in out], [(32, 16, 3), (16, 64, 3), (48, 48, 3)])
        self.assertEqual([float(o.flat[0]) for o in out], [float(np.float32(s / 255.0)) for s in (10, 20, 30)])
        self.assertEqual([c[2] for c in ad.calls if c[0] == "sample"], [10, 20, 30])

    def test_interrupt_checked_before_each_job(self):
        _, ad = render([job(seed=1), job(seed=2)])
        kinds = [c[0] for c in ad.calls]
        self.assertEqual(kinds.count("interrupt"), 2)
        second = kinds.index("interrupt", kinds.index("interrupt") + 1)
        self.assertEqual(kinds[second - 1], "to_rgb")  # job 1 finished before the second check

    def test_interrupt_aborts_without_partial_result(self):
        ad = FakeAdapter(fail_on_interrupt=2)
        prog = Progress()
        with self.assertRaises(KeyboardInterrupt):
            render([job(seed=1), job(seed=2), job(seed=3)], ad=ad, progress=prog)
        self.assertEqual(len([c for c in ad.calls if c[0] == "sample"]), 1)
        self.assertEqual(prog.updates, [1])

    def test_adapter_exception_propagates_unchanged(self):
        class Boom(FakeAdapter):
            def sample(self, *a):
                raise RuntimeError("cuda oom")
        with self.assertRaisesRegex(RuntimeError, "cuda oom"):
            render([job()], ad=Boom())

    def test_progress_updates_once_per_job(self):
        prog = Progress()
        render([job(seed=1), job(seed=2), job(seed=3)], progress=prog)
        self.assertEqual(prog.updates, [1, 1, 1])
        render([job()], progress=None)  # no progress object is fine

    def test_jobs_not_mutated_and_extra_keys_ignored(self):
        jobs = [job(seed=4, key="g0#0", category="ground", attempt=2)]
        snapshot = [dict(j) for j in jobs]
        render(tuple(jobs))
        self.assertEqual(jobs, snapshot)

    def test_params_mapping_equals_dataclass(self):
        a, ad_a = render([job()], params={"steps": 5, "guidance": 2.0})
        b, ad_b = render([job()], params=base.R.RegenParams(steps=5, guidance=2.0))
        self.assertTrue(np.array_equal(a[0], b[0]))
        self.assertEqual([c[3] for c in ad_a.calls if c[0] == "sample"], [c[3] for c in ad_b.calls if c[0] == "sample"])

    def test_deterministic_for_same_inputs(self):
        jobs = [job(seed=11), job(seed=12)]
        a, _ = render(jobs)
        b, _ = render(jobs)
        self.assertTrue(all(np.array_equal(x, y) for x, y in zip(a, b)))

    def test_seed_passed_verbatim_to_sampler(self):
        _, ad = render([job(seed=2**64 - 1)])
        self.assertEqual([c[2] for c in ad.calls if c[0] == "sample"], [2**64 - 1])

    def test_bad_decoded_image_rejected(self):
        bads = {
            "wrong size": np.zeros((16, 16, 3), np.float32),
            "swapped hw": np.zeros((32, 16, 3), np.float32),
            "float64": np.zeros((16, 32, 3), np.float64),
            "rgba": np.zeros((16, 32, 4), np.float32),
            "not array": [[0.0]],
        }
        for label, bad in bads.items():
            with self.subTest(label):
                with self.assertRaisesRegex(ValueError, r"jobs\[0\] decoded image has .*expected float32 shape \(16, 32, 3\)"):
                    render([job(w=32, h=16)], ad=FakeAdapter(bad_image=bad))

    def test_bad_image_reports_job_index_and_type_name(self):
        class Second(FakeAdapter):
            def to_rgb(self, image):
                return super().to_rgb(image) if image["seed"] == 1 else "oops"
        with self.assertRaisesRegex(ValueError, r"jobs\[1\] decoded image has str,"):
            render([job(seed=1), job(seed=2)], ad=Second())


if __name__ == "__main__":
    unittest.main()
