"""Tests for tilemap_core.scene_layout, spec ccn-scene/1 (unittest, stdlib + numpy only)."""

import copy
import json
import sys
import unittest

import numpy as np

from tilemap_core import llm_client, scene_layout as S, transitions

CAT = {"tiles": [
    {"id": "grass", "category": "ground"}, {"id": "dirt", "category": "ground"}, {"id": "sand", "category": "ground"},
    {"id": "tree", "category": "object", "variants": 3, "footprint": [2, 3]},
    {"id": "rock", "category": "object", "variants": 2},
    {"id": "flowers", "category": "overlay", "variants": 2}],
    "transitions": [["grass", "dirt"], ["dirt", "sand"]]}
PAIRS = (("grass", "dirt"), ("dirt", "sand"))
VIEWS = ("top-down", "three-quarter", "side-view")
SEEDS = (0, 1, 7, 512)
WANT = {("tree", 0), ("tree", 1), ("tree", 2), ("rock", 0), ("rock", 1), ("flowers", 0), ("flowers", 1)}
FOOT = {"tree": (2, 3), "rock": (1, 1), "flowers": (1, 1)}
MINIMAL = {"format": "ccn-scene/1", "width": 4, "height": 4, "terrain": [["grass"] * 5 for _ in range(5)]}


def mut(scene, fn):
    s = copy.deepcopy(scene)
    fn(s)
    return s


def pair_masks(scene):
    """{(a, b): set of corner masks} over cells whose corners only use that pair."""
    out = {}
    for r in range(scene["height"]):
        for c in range(scene["width"]):
            cs = S.cell_corners(scene["terrain"], c, r)
            if None in cs:
                continue
            for a, b in PAIRS:
                if set(cs) <= {a, b}:
                    out.setdefault((a, b), set()).add(S.corner_mask(*[int(x == b) for x in cs]))
    return out


class BitOrder(unittest.TestCase):
    def test_corner_mask_matches_atlas_wang_tiles(self):
        a = np.zeros((16, 16, 4), np.uint8)
        a[..., 3] = 255
        b = a.copy()
        b[..., 0] = 255
        for mask, (corners, _) in enumerate(transitions.wang_tiles(a, b, 16, 1)):
            self.assertEqual(S.corner_mask(*corners), mask)

    def test_corner_mask_weights_are_ne1_se2_sw4_nw8(self):
        self.assertEqual([S.corner_mask(*c) for c in ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))], [1, 2, 4, 8])

    def test_cell_corners_order_ne_se_sw_nw(self):
        grid = [["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"]]
        self.assertEqual(S.cell_corners(grid, 0, 0), ("b", "e", "d", "a"))
        self.assertEqual(S.cell_corners(grid, 1, 1), ("f", "i", "h", "e"))


class ValidateScene(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.good, _ = S.auto_layout(CAT, "top-down", 1)

    def test_minimal_scene_gets_defaults_and_is_a_copy(self):
        out = S.validate_scene(MINIMAL, CAT)
        self.assertEqual((out["background"], out["seed"], out["objects"], out["overlays"]), ("#808080", 0, [], []))
        self.assertIsNot(out["terrain"], MINIMAL["terrain"])
        self.assertIsNot(out["terrain"][0], MINIMAL["terrain"][0])

    def test_null_sky_and_uppercase_background_accepted_and_lowercased(self):
        sky = dict(MINIMAL, terrain=[[None] * 5] + [["dirt"] * 5 for _ in range(4)], background="#ABCDEF")
        out = S.validate_scene(sky, CAT)
        self.assertEqual(out["background"], "#abcdef")
        self.assertIsNone(out["terrain"][0][0])

    def test_overlay_may_overlap_object_and_variant_null_kept(self):
        s = mut(self.good, lambda s: s["overlays"].append(dict(s["objects"][0], id="flowers", variant=None)))
        self.assertIsNone(S.validate_scene(s, CAT)["overlays"][-1]["variant"])

    def test_dimension_bounds_are_4_to_32_with_exact_terrain(self):
        grid = lambda n: dict(MINIMAL, width=n, height=n, terrain=[["grass"] * (n + 1) for _ in range(n + 1)])
        for n in (4, 32):
            self.assertEqual(S.validate_scene(grid(n), CAT)["width"], n)
        for n in (3, 33):
            with self.assertRaises(ValueError) as cm:
                S.validate_scene(grid(n), CAT)
            self.assertTrue(str(cm.exception).startswith("width: "), str(cm.exception))

    def test_auto_layout_output_round_trips(self):
        self.assertEqual(S.validate_scene(self.good, CAT), self.good)

    def test_plan_shaped_catalogue_with_extra_plan_keys_accepted(self):
        cat = {"theme": "t", "tiles": [dict(t, name=t["id"], description="d") for t in CAT["tiles"]], "transitions": []}
        self.assertEqual(S.validate_scene(self.good, cat)["width"], self.good["width"])

    def test_rejects_name_the_field_path(self):
        def set_obj(i, d):
            return lambda s: s["objects"].__setitem__(i, d)
        rejects = {
            "scene": [[], "x", None],
            "unknown key": [mut(self.good, lambda s: s.update(x=1))],
            "format": [mut(self.good, lambda s: s.update(format="ccn-scene/2")), mut(self.good, lambda s: s.pop("format"))],
            "terrain": [mut(self.good, lambda s: s.pop("terrain")), mut(self.good, lambda s: s["terrain"].pop())],
            "width": [mut(self.good, lambda s: s.update(width=w)) for w in (3, 33, True, 8.0)],
            "height": [mut(self.good, lambda s: s.update(height=8.0))],
            "terrain[2]": [mut(self.good, lambda s: s["terrain"][2].pop())],
            "terrain[0][0]": [mut(self.good, lambda s: s["terrain"][0].__setitem__(0, v)) for v in ("lava", "tree", 1)],
            "background": [mut(self.good, lambda s: s.update(background=b)) for b in ("red", "#fff", 5)],
            "seed": [mut(self.good, lambda s: s.update(seed=v)) for v in (True, -1, 2**64)],
            "objects[0].id": [mut(self.good, set_obj(0, {"id": i, "col": 0, "row": 0})) for i in ("x", "flowers", 3)],
            "objects[0].variant": [mut(self.good, set_obj(0, {"id": "rock", "variant": v, "col": 0, "row": 0})) for v in (2, -1, True)],
            "objects[0].col": [mut(self.good, lambda s: s["objects"][0].pop("col")), mut(self.good, lambda s: s["objects"][0].update(col=True)),
                               mut(self.good, lambda s: s.update(objects=[{"id": "tree", "variant": 0, "col": s["width"] - 1, "row": 0}]))],
            "objects[0].row": [mut(self.good, lambda s: s.update(objects=[{"id": "tree", "variant": 0, "col": 0, "row": s["height"] - 2}]))],
            "objects[0]": [mut(self.good, lambda s: s["objects"][0].update(z=1))],
            f"objects[{len(self.good['objects'])}]: object overlaps": [mut(self.good, lambda s: s["objects"].append(dict(s["objects"][0])))],
            "overlays[0].id": [mut(self.good, lambda s: s["overlays"].__setitem__(0, {"id": "tree", "col": 0, "row": 0}))],
            "objects": [mut(self.good, lambda s: s.update(objects="x"))],
        }
        for path, cases in rejects.items():
            for i, bad in enumerate(cases):
                with self.subTest(path=path, case=i):
                    with self.assertRaises(ValueError) as cm:
                        S.validate_scene(bad, CAT)
                    self.assertIn(path, str(cm.exception))

    def test_bad_catalogues_raise_value_error(self):
        g = {"id": "g", "category": "ground"}
        for bad in ({}, {"tiles": []}, {"tiles": [{"id": "t", "category": "object"}]}, {"tiles": [g, dict(g)]},
                    {"tiles": [g], "transitions": [["g", "g"]]}, {"tiles": [g], "transitions": [["g", "zz"]]},
                    {"tiles": [g, {"id": "h", "category": "ground"}], "transitions": [["g", "h"]] * 5},
                    {"tiles": [{"id": "Bad Id", "category": "ground"}]}, {"tiles": [dict(g, category="mystery")]},
                    {"tiles": [g, {"id": "o", "category": "object", "footprint": [5, 1]}]}):
            with self.subTest(cat=bad), self.assertRaises(ValueError):
                S.validate_scene(self.good, bad)


class AutoLayout(unittest.TestCase):
    def test_all_16_masks_for_each_pair_all_views_and_seeds(self):
        for v in VIEWS:
            for seed in SEEDS:
                mk = pair_masks(S.auto_layout(CAT, v, seed)[0])
                for pair in PAIRS:
                    with self.subTest(view=v, seed=seed, pair=pair):
                        self.assertEqual(mk.get(pair), set(range(16)))

    def test_every_id_variant_placed_once_valid_and_on_uniform_ground(self):
        for v in VIEWS:
            for seed in SEEDS:
                sc, unplaced = S.auto_layout(CAT, v, seed)
                items = sc["objects"] + sc["overlays"]
                with self.subTest(view=v, seed=seed):
                    self.assertEqual(unplaced, [])
                    self.assertEqual({(o["id"], o["variant"]) for o in items}, WANT)
                    self.assertEqual(len(items), len(WANT))
                    self.assertEqual(S.validate_scene(sc, CAT), sc)
                for o in items:
                    fw, fh = FOOT[o["id"]]
                    # side view: item stands on the ground row directly below its footprint
                    cells = ([(x, o["row"] + fh) for x in range(o["col"], o["col"] + fw)] if v == "side-view" else
                             [(x, y) for x in range(o["col"], o["col"] + fw) for y in range(o["row"], o["row"] + fh)])
                    for x, y in cells:
                        cs = S.cell_corners(sc["terrain"], x, y)
                        self.assertTrue(len(set(cs)) == 1 and cs[0] is not None, (v, seed, o, x, y))

    def test_items_placed_before_terrain_painting_pair_ground_still_present(self):
        sc, _ = S.auto_layout(CAT, "top-down", 5)
        rows = sc["terrain"][: sc["height"] - 4]  # above the sampler band
        flat = [t for r in rows for t in r]
        self.assertIn("dirt", flat)
        self.assertIn("grass", flat)

    def test_sampler_patch_is_bottom_right_for_every_pair(self):
        for v in VIEWS:
            sc, _ = S.auto_layout(CAT, v, 3)
            t, sx, y = sc["terrain"], sc["width"] - 4 - 6, sc["height"] - 4
            self.assertEqual((t[y][sx], t[y][sx + 3], t[y][sx + 6], t[y][sx + 9]), ("grass", "dirt", "dirt", "sand"))

    def test_grid_size_rules(self):
        one = {"tiles": [{"id": "g", "category": "ground"}]}
        many = {"tiles": [{"id": "g", "category": "ground"}, {"id": "o", "category": "object", "variants": 60, "footprint": [4, 2]}]}
        pair = {"tiles": [{"id": "a", "category": "ground"}, {"id": "b", "category": "ground"}], "transitions": [["a", "b"]]}
        sizes = lambda cat, v: tuple(S.auto_layout(cat, v, 1)[0][k] for k in ("width", "height"))
        self.assertEqual(sizes(one, "top-down"), (8, 6))
        self.assertEqual(sizes(one, "side-view"), (8, 14))
        self.assertEqual(sizes(pair, "top-down"), (8, 11))
        self.assertEqual(sizes(CAT, "top-down"), (10, 12))
        self.assertEqual(sizes(CAT, "side-view"), (10, 14))
        self.assertEqual(sizes(many, "top-down"), (24, 16))

    def test_deterministic_per_seed_and_seed_changes_layout(self):
        self.assertEqual(S.auto_layout(CAT, "top-down", 5), S.auto_layout(copy.deepcopy(CAT), "top-down", 5))
        lay = lambda v, k: json.dumps([S.auto_layout(CAT, v, k)[0][f] for f in ("terrain", "objects", "overlays")])
        self.assertNotEqual(lay("top-down", 5), lay("top-down", 6))
        self.assertGreater(len({lay("side-view", k) for k in range(8)}), 1)
        self.assertEqual(S.auto_layout(CAT, "top-down", 5)[0]["seed"], 5)

    def test_side_view_has_sky_above_ground(self):
        sc, _ = S.auto_layout(CAT, "side-view", 3)
        self.assertIsNone(sc["terrain"][0][0])
        self.assertIsNotNone(sc["terrain"][-1][0])
        self.assertTrue(all(v is None for v in sc["terrain"][round(0.6 * sc["height"]) - 1]))

    def test_unplaced_reported_when_items_do_not_fit(self):
        cat = {"tiles": [{"id": "g", "category": "ground"}, {"id": "o", "category": "object", "variants": 60, "footprint": [4, 2]}]}
        sc, unplaced = S.auto_layout(cat, "top-down", 1)
        self.assertGreater(len(unplaced), 0)
        self.assertEqual(len(sc["objects"]) + len(unplaced), 60)
        self.assertTrue(all(u.startswith("o#") for u in unplaced))
        self.assertEqual(S.validate_scene(sc, cat), sc)

    def test_single_ground_no_pairs_places_everything(self):
        cat = {"tiles": [{"id": "g", "category": "ground"}, {"id": "r", "category": "object"}]}
        sc, unplaced = S.auto_layout(cat, "three-quarter", 0)
        self.assertEqual((unplaced, len(sc["objects"])), ([], 1))
        self.assertEqual({v for row in sc["terrain"] for v in row}, {"g"})

    def test_bad_view_seed_or_catalogue_raise(self):
        for args in ((CAT, "iso", 1), (CAT, "top-down", -1), (CAT, "top-down", True), ({}, "top-down", 1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                S.auto_layout(*args)


class Messages(unittest.TestCase):
    def test_roles_content_and_llm_client_accepts_them(self):
        msgs = S.build_scene_messages(CAT, "top-down")
        llm_client._check_messages(msgs)
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])
        user = msgs[1]["content"]
        for part in ("top-down", "- tree: object, variants 0..2, footprint 2x3", "- grass: ground", "grass+dirt, dirt+sand"):
            self.assertIn(part, user)
        self.assertIn("ccn-scene/1", msgs[0]["content"])

    def test_deterministic_and_no_transitions_says_none(self):
        self.assertEqual(S.build_scene_messages(CAT, "side-view"), S.build_scene_messages(CAT, "side-view"))
        self.assertIn("Transitions: none", S.build_scene_messages(MINIMAL_CAT(), "top-down")[1]["content"])

    def test_unknown_view_raises(self):
        with self.assertRaises(ValueError):
            S.build_scene_messages(CAT, "iso")


def MINIMAL_CAT():
    return {"tiles": [{"id": "g", "category": "ground"}]}


class ParseScene(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.good, _ = S.auto_layout(CAT, "top-down", 1)

    def test_valid_fenced_reply_is_llm_source(self):
        sc, src, problem = S.parse_scene("Here:\n```json\n" + json.dumps(self.good) + "\n```", CAT, "top-down", 1)
        self.assertEqual((sc, src, problem), (self.good, "llm", ""))

    def test_missing_seed_defaults_to_argument(self):
        sc, src, _ = S.parse_scene(json.dumps({k: v for k, v in self.good.items() if k != "seed"}), CAT, "top-down", 9)
        self.assertEqual((src, sc["seed"]), ("llm", 9))

    def test_invalid_replies_fall_back_to_auto_layout_with_problem(self):
        bad_replies = ["", "no json", "{", json.dumps(dict(self.good, width=99)), '{"a":1,"a":2}', "[1]",
                       json.dumps(dict(self.good, terrain=[[1]])), "x" * 70000, None, 5]
        for bad in bad_replies:
            with self.subTest(reply=repr(bad)[:20]):
                sc, src, problem = S.parse_scene(bad, CAT, "top-down", 4)
                self.assertEqual(src, "auto_fallback")
                self.assertTrue(problem)
                self.assertEqual(sc, S.auto_layout(CAT, "top-down", 4)[0])

    def test_fallback_follows_view_and_seed(self):
        sc, src, _ = S.parse_scene("nope", CAT, "side-view", 7)
        self.assertEqual((src, sc), ("auto_fallback", S.auto_layout(CAT, "side-view", 7)[0]))

    def test_problem_text_is_cleaned_and_bounded(self):
        _, _, problem = S.parse_scene('{"format":"\\u0000\\u202e<b>"}', CAT)
        self.assertTrue(all(ch.isprintable() for ch in problem))
        self.assertNotIn("<", problem)
        _, _, long_problem = S.parse_scene(json.dumps(dict(self.good, **{"k" * 400: 1})), CAT)
        self.assertLessEqual(len(long_problem), 300)

    def test_bad_caller_arguments_still_raise(self):
        for args in (("{}", {}), ("{}", CAT, "iso"), ("{}", CAT, "top-down", -1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                S.parse_scene(*args)

    def test_module_is_pure_no_torch_or_comfy(self):
        self.assertNotIn("torch", sys.modules)
        self.assertFalse([m for m in sys.modules if m == "comfy" or m.startswith("comfy.")])


if __name__ == "__main__":
    unittest.main()
