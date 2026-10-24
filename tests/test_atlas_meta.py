"""Tests for tilemap_core.atlas_meta (``ccn-tilemap/2`` helpers) and its wiring into atlas.build_metadata.

Shared helpers come from tests.test_atlas_transitions (T0); atlas.py behaviour proper is in test_atlas.py.

Residual gaps (not tested): exact wording of error messages beyond the key phrases asserted; ``_show`` of
non-str non-hashable exotic objects; recursion limits (depth is capped at 8, far below the interpreter limit);
``check_extra`` on dict subclasses with custom ``__iter__`` (copied via ``.items()`` only).
"""

import collections
import copy
import enum
import json
import unittest
from types import MappingProxyType

import numpy as np

from tests.test_atlas_transitions import PAL, Base, item
from tilemap_core import atlas, atlas_meta as M

NAN, INF = float("nan"), float("inf")
V1 = ("format", "tile_size", "columns", "rows", "spacing", "width", "height", "view", "model", "seed", "theme", "palette", "tiles")
FULL = {"style_prefix": "pixel art", "cell_px": 512, "plan": {"theme": "forest", "tiles": [{"id": "g", "variants": 2}], "ok": True, "none": None},
        "snap": {"palette_size": 24}, "sampler": {"steps": 20, "cfg": 7.5}, "lineage": {"parent": None}, "review": {"score": 62, "issues": []}}  # fmt: skip


def chain(levels, leaf=None):
    d = {} if leaf is None else leaf
    for _ in range(levels):
        d = {"k": d}
    return d


class FormatTests(Base):
    def test_constants(self):
        self.assertEqual((M.FORMAT, M.FORMATS), ("ccn-tilemap/2", ("ccn-tilemap/1", "ccn-tilemap/2")))
        self.assertEqual(M.EXTRA_KEYS, ("style_prefix", "cell_px", "plan", "snap", "sampler", "lineage", "review"))
        self.assertIs(atlas.FORMAT, M.FORMAT)
        self.assertEqual(sorted(M.__all__), sorted(["FORMAT", "FORMATS", "EXTRA_KEYS", "read_format", "check_extra", "tile_provenance"]))
        self.assertFalse(set(M.EXTRA_KEYS) & set(V1))

    def test_read_format_accepts_v1_and_v2(self):
        for fmt in M.FORMATS:
            self.assertEqual(M.read_format({"format": fmt}), fmt)
        self.assertEqual(M.read_format(MappingProxyType({"format": "ccn-tilemap/2", "x": 1})), "ccn-tilemap/2")

    def test_read_format_rejects(self):
        bad = [{}, {"format": None}, {"format": 2}, {"format": ["ccn-tilemap/2"]}, {"format": "ccn-tilemap/3"}, {"format": "ccn-tilemap/0"},
               {"format": "CCN-TILEMAP/2"}, {"format": "ccn-tilemap/2 "}, {"format": ""}, None, [], "ccn-tilemap/2", 5, [("format", "ccn-tilemap/2")]]  # fmt: skip
        self.rejects(M.read_format, [(m,) for m in bad])

    def test_read_format_message_is_bounded_and_escaped(self):
        with self.assertRaises(ValueError) as cm:
            M.read_format({"format": "\x00\n\u202e" + "a" * 1000})
        msg = str(cm.exception)
        self.assertLess(len(msg), 250)
        self.assertFalse(any(c in msg for c in "\x00\n\u202e"))
        with self.assertRaisesRegex(ValueError, r"<int>"):
            M.read_format({"format": 12345678})


class ExtraTests(Base):
    def test_none_and_empty(self):
        self.assertEqual(M.check_extra(None), {})
        empty = {}
        self.assertEqual(M.check_extra(empty), {})
        self.assertIsNot(M.check_extra(empty), empty)
        self.rejects(M.check_extra, [(v,) for v in ([], "x", 0, False, [("plan", {})], MappingProxyType({}), (("plan", {}),))])

    def test_full_extra_deep_copy_json_safe_no_mutation(self):
        src = copy.deepcopy(FULL)
        out = M.check_extra(src)
        self.assertEqual(out, FULL)
        self.assertEqual(list(out), list(FULL))  # order kept
        self.assertEqual(src, FULL)
        self.assertEqual(json.loads(json.dumps(out)), out)
        self.assertIsNot(out["plan"], src["plan"])
        self.assertIsNot(out["plan"]["tiles"][0], src["plan"]["tiles"][0])
        out["plan"]["tiles"].append(1)
        out["review"]["issues"].append(2)
        self.assertEqual(src, FULL)
        for k in M.EXTRA_KEYS:  # every single key alone is accepted
            self.assertEqual(M.check_extra({k: FULL[k]}), {k: FULL[k]})

    def test_rejects_v1_unknown_and_non_string_keys(self):
        for k in V1:
            with self.subTest(k=k), self.assertRaisesRegex(ValueError, "collides"):
                M.check_extra({k: 1})
        for k in ("foo", "Plan", "plan ", "", "style-prefix"):
            with self.subTest(k=k), self.assertRaisesRegex(ValueError, "not allowed"):
                M.check_extra({k: {}})
        for bad in ({1: {}}, {b"plan": {}}, {None: 1}, {"plan": {}, 2: 3}):
            with self.subTest(bad=repr(bad)), self.assertRaisesRegex(ValueError, "keys must be strings"):
                M.check_extra(bad)
        with self.assertRaises(ValueError) as cm:
            M.check_extra({"x" * 5000: 1})
        self.assertLess(len(str(cm.exception)), 400)

    def test_style_prefix_must_be_string(self):
        for ok in ("", "GRPZA, umempart, pixel art", "x" * 10000):
            self.assertEqual(M.check_extra({"style_prefix": ok}), {"style_prefix": ok})
        self.rejects(M.check_extra, [({"style_prefix": v},) for v in (1, None, ["a"], b"x", {}, True, 1.5)])

    def test_cell_px_bounds_and_type(self):
        for ok in (1, 2, 512, 4095, 4096):
            self.assertEqual(M.check_extra({"cell_px": ok}), {"cell_px": ok})
        bad = (0, -1, 4097, 2**20, True, False, 512.0, 1.0, "512", None, [512], np.int64(512), NAN)
        self.rejects(M.check_extra, [({"cell_px": v},) for v in bad])

    def test_dict_blocks_must_be_objects(self):
        for k in ("plan", "snap", "sampler", "lineage", "review"):
            self.assertEqual(M.check_extra({k: {}}), {k: {}})
            self.assertEqual(M.check_extra({k: {"a": [1, {"b": None}]}}), {k: {"a": [1, {"b": None}]}})
            self.rejects(M.check_extra, [({k: v},) for v in ([], [{}], "x", 1, None, True, 1.5)])

    def test_scalar_exactness(self):
        class S(str):
            pass

        class E(enum.IntEnum):
            A = 1

        ok = {"a": None, "b": True, "c": 1.5, "d": "s", "e": [1, [2.0, "x"]], "f": {"g": {}}, "h": -0.0, "i": 1e308, "j": [], "k": False}
        self.assertEqual(M.check_extra({"plan": ok}), {"plan": ok})
        for v in (NAN, INF, -INF, (1, 2), {1, 2}, b"x", np.int64(1), np.float64(1.5), object(), S("x"), E.A, 1j, bytearray(b"x"), range(2)):
            with self.subTest(v=repr(v)[:30]):
                self.rejects(M.check_extra, [({"plan": {"a": v}},), ({"plan": {"a": [v]}},)])
        self.rejects(M.check_extra, [({"plan": {1: "x"}},), ({"plan": {"a": {None: 1}}},), ({"plan": {b"a": 1}},)])
        out = M.check_extra({"plan": collections.OrderedDict(a=[1])})
        self.assertIs(type(out["plan"]), dict)

    def test_int_magnitude_cap(self):
        for v in (0, 2**64, -(2**64), 2**63):
            self.assertEqual(M.check_extra({"plan": {"n": v}}), {"plan": {"n": v}})
        for v in (2**64 + 1, -(2**64) - 1, 10**30):
            with self.subTest(v=v), self.assertRaisesRegex(ValueError, "out of range"):
                M.check_extra({"plan": {"n": v}})

    def test_depth_cap(self):
        M.check_extra({"plan": chain(8)})  # deepest dict at depth 8
        M.check_extra({"plan": {"x": chain(7, [])}})  # chain of lists, deepest list at depth 8
        with self.assertRaisesRegex(ValueError, "deeper than 8"):
            M.check_extra({"plan": chain(9)})
        lists = []
        for _ in range(8):
            lists = [lists]
        self.rejects(M.check_extra, [({"plan": {"x": lists}},), ({"plan": {"x": chain(8, 1)}},)])
        self.assertEqual(M.check_extra({"plan": {"x": chain(7, 1)}}), {"plan": {"x": chain(7, 1)}})

    def test_error_paths_name_the_field_and_cap_key_length(self):
        with self.assertRaisesRegex(ValueError, r"extra plan\.a\[1\]\.b "):
            M.check_extra({"plan": {"a": [0, {"b": NAN}]}})
        with self.assertRaises(ValueError) as cm:
            M.check_extra({"plan": {"k" * 100: NAN}})
        self.assertIn("k" * 40, str(cm.exception))
        self.assertNotIn("k" * 41, str(cm.exception))
        bad = {"plan": {"ok": [1]}, "snap": {"x": NAN}}
        snap = copy.deepcopy(bad)
        self.assertRaises(ValueError, M.check_extra, bad)
        self.assertEqual(bad["plan"], snap["plan"])
        self.assertEqual(list(bad), ["plan", "snap"])


class ProvenanceTests(Base):
    def test_empty_and_uncarried_yield_nothing(self):
        for p in ({}, {"id": "g"}, {"key": "k"}, {"key": "k", "id": "g", "cells_w": 1}, MappingProxyType({})):
            self.assertEqual(M.tile_provenance(p, "k"), {})

    def test_full_and_single_fields(self):
        full = {"prompt": "grass", "seed": 5, "attempt": 2, "source": "kept", "job_index": 3, "key": "k", "junk": 1}
        self.assertEqual(M.tile_provenance(full, "k"), {"prompt": "grass", "seed": 5, "attempt": 2, "source": "kept", "job_index": 3, "key": "k"})
        for field, value in (("prompt", None), ("prompt", ""), ("seed", 0), ("attempt", 0), ("source", "generated"), ("job_index", None)):
            self.assertEqual(M.tile_provenance({field: value}, "k"), {field: value, "key": "k"})

    def test_prompt(self):
        for v in ("x", "", "p" * 4000):
            self.assertEqual(M.tile_provenance({"prompt": v}, "k")["prompt"], v)
        self.rejects(M.tile_provenance, [({"prompt": v}, "k") for v in ("p" * 4001, 5, b"x", ["a"], {}, False)])

    def test_seed(self):
        for v in (0, 1, 2**64 - 1, np.int64(7), np.uint64(2**64 - 1)):
            got = M.tile_provenance({"seed": v}, "k")["seed"]
            self.assertEqual((got, type(got)), (int(v), int))
        self.rejects(M.tile_provenance, [({"seed": v}, "k") for v in (-1, 2**64, True, False, 1.0, "1", None, [1], np.float64(1))])

    def test_attempt_and_job_index_bounds(self):
        for field in ("attempt", "job_index"):
            for v in (0, 1, 2**31 - 1):
                self.assertEqual(M.tile_provenance({field: v}, "k")[field], v)
            self.rejects(M.tile_provenance, [({field: v}, "k") for v in (-1, 2**31, True, 1.5, "1", [0])])
        self.assertEqual(M.tile_provenance({"attempt": np.int32(4)}, "k")["attempt"], 4)
        self.rejects(M.tile_provenance, [({"attempt": None}, "k")])  # only job_index may be None

    def test_source_enum(self):
        for v in ("generated", "kept", "regenerated"):
            self.assertEqual(M.tile_provenance({"source": v}, "k")["source"], v)
        self.rejects(M.tile_provenance, [({"source": v}, "k") for v in ("Generated", "", "other", "kept ", None, 1, ["kept"], b"kept")])

    def test_key_rules(self):
        for key in ("k", "k" * 200):
            self.assertEqual(M.tile_provenance({"seed": 1}, key)["key"], key)
        self.rejects(M.tile_provenance, [({"seed": 1}, k) for k in ("", "k" * 201, None, 5, b"k", ["k"])])
        self.rejects(M.tile_provenance, [({}, "")])  # key invalid even when nothing is carried
        self.rejects(M.tile_provenance, [({"key": "other", "seed": 1}, "k"), ({"key": 5}, "k"), ({"key": None}, "k"), ({"key": "K"}, "k")])
        self.assertEqual(M.tile_provenance({"key": "k", "seed": 1}, "k"), {"seed": 1, "key": "k"})

    def test_non_mapping_item_and_no_mutation(self):
        self.rejects(M.tile_provenance, [(v, "k") for v in (None, [], "seed", 5, [("seed", 1)], ("seed",))])
        p = {"prompt": "a", "seed": 1, "key": "k"}
        snap = copy.deepcopy(p)
        out = M.tile_provenance(p, "k")
        out["seed"] = 99
        self.assertEqual(p, snap)
        self.assertEqual(M.tile_provenance(p, "k"), M.tile_provenance(p, "k"))

    def test_messages_cap_key_and_name_field(self):
        long = "k" * 200
        with self.assertRaises(ValueError) as cm:
            M.tile_provenance({"seed": -1}, long)
        self.assertIn("seed", str(cm.exception))
        self.assertIn("k" * 60, str(cm.exception))
        self.assertNotIn("k" * 61, str(cm.exception))
        with self.assertRaisesRegex(ValueError, "differs from placement key"):
            M.tile_provenance({"key": "a"}, "b")
        with self.assertRaisesRegex(ValueError, "must be one of"):
            M.tile_provenance({"source": "x"}, "k")


class WiringTests(Base):
    def test_build_metadata_merges_extra_after_v1_keys_and_validates(self):
        pl = atlas.pack([item("g0", "ground", 0, prompt="grass", seed=3, attempt=1, source="generated", job_index=0)], 2)
        meta = atlas.build_metadata(pl, 8, 2, 0, PAL, "top-down", "m", 1, "t", FULL)
        self.assertEqual(list(meta)[:13], list(V1))
        self.assertEqual(list(meta)[13:], list(FULL))
        self.assertEqual({k: meta[k] for k in FULL}, FULL)
        self.assertEqual(meta["format"], "ccn-tilemap/2")
        t = meta["tiles"][0]
        self.assertEqual((t["prompt"], t["seed"], t["attempt"], t["source"], t["job_index"], t["key"]), ("grass", 3, 1, "generated", 0, "g0#0"))
        self.assertEqual(json.loads(json.dumps(meta)), meta)
        self.assertEqual(set(atlas.build_metadata(pl, 8, 2, 0, PAL)), set(V1))  # no extra -> only /1 keys
        self.assertEqual(set(atlas.build_metadata(atlas.pack([item("g0", "ground", 0)], 2), 8, 2, 0, PAL)["tiles"][0]),
                         {"index", "id", "name", "category", "variant", "tileable", "footprint", "x", "y", "w", "h", "cells", "wang"})  # fmt: skip

    def test_build_metadata_rejects_bad_extra_and_provenance(self):
        base = atlas.pack([item("g0", "ground", 0)], 2)
        self.rejects(atlas.build_metadata, [(base, 8, 2, 0, PAL, "", "", 0, "", e) for e in ({"seed": 1}, {"foo": 1}, {"cell_px": 0}, [], {"plan": []})])
        for kw in ({"seed": -1}, {"source": "x"}, {"attempt": True}, {"prompt": 5}, {"job_index": -1}):
            self.rejects(atlas.build_metadata, [([{**base[0], **kw}], 8, 2, 0, PAL)])
        extra = copy.deepcopy(FULL)
        meta = atlas.build_metadata(base, 8, 2, 0, PAL, extra=extra)
        meta["plan"]["tiles"].clear()
        self.assertEqual(extra, FULL)
