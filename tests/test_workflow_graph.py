"""Lint every shipped workflow: ``workflows/*.api.json`` and each LiteGraph-format JSON in ``example_workflows/``.

Files in ``example_workflows/`` without ``nodes`` and ``links`` keys are not UI workflows and are skipped. The
linter itself (spec table, API / UI / twin / review-wiring checks) is in ``tests/_workflow_lint.py``.
"""

import glob
import json
import os
import unittest

from tests._workflow_lint import ROOT, check_api, check_pair, check_review, check_ui


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


API_FILES = sorted(glob.glob(os.path.join(ROOT, "workflows", "*.api.json")))
UI_FILES = []
for _p in sorted(glob.glob(os.path.join(ROOT, "example_workflows", "*.json"))):
    _d = _load(_p)
    if isinstance(_d, dict) and "nodes" in _d and "links" in _d:
        UI_FILES.append(_p)


class WorkflowGraphTests(unittest.TestCase):
    def test_discovery(self):
        self.assertTrue(API_FILES, "no workflows/*.api.json found")
        self.assertTrue(UI_FILES, "no UI-format workflow in example_workflows/ found")

    def test_api_graphs(self):
        for p in API_FILES:
            with self.subTest(file=os.path.basename(p)):
                self.assertEqual(check_api(_load(p)), [])

    def test_ui_graphs(self):
        for p in UI_FILES:
            with self.subTest(file=os.path.basename(p)):
                self.assertEqual(check_ui(_load(p)), [])

    def test_api_ui_twins_match(self):
        twins = 0
        for p in API_FILES:
            ui = os.path.join(ROOT, "example_workflows", os.path.basename(p).replace(".api.json", ".json"))
            if ui in UI_FILES:
                twins += 1
                with self.subTest(file=os.path.basename(p)):
                    self.assertEqual(check_pair(_load(p), _load(ui)), [])
        self.assertGreater(twins, 0, "no API/UI twin found")

    def test_review_loop_wiring(self):
        checked = 0
        for p in API_FILES:
            g = _load(p)
            if any(n["class_type"] == "CCN_TilemapReviewLoop" for n in g.values()):
                checked += 1
                with self.subTest(file=os.path.basename(p)):
                    self.assertEqual(check_review(g), [])
        self.assertGreater(checked, 0, "no workflow with a CCN_TilemapReviewLoop found")


if __name__ == "__main__":
    unittest.main()
