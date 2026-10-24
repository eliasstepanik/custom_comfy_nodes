"""Tests for ``nodes/tilemap_plan.py`` ``CCN_TilemapPlan`` in ``plan_mode`` new (D16c). Stdlib unittest; no network:
``llm_client.chat`` is replaced by a scripted fake (or run for real through ``tests/_support`` fake opener when the
test is about key hygiene); the plan cache lives in a temp dir via a ``folder_paths`` stub.

The node module is imported through ``load_pkg`` of ``tests/test_node_load.py`` under its own synthetic package name;
``load_pack`` here additionally executes the root ``__init__.py`` under a second name to prove the pack registers the nodes.
Sibling files: ``test_node_plan_more.py`` (repair, validation errors, error hygiene), ``test_node_plan_extend.py`` (extend
mode), ``test_node_plan_expand.py`` (``CCN_TilemapExpand``); they import ``PlanCase`` / ``reply`` / ``setUpModule`` from here.

Residual gaps: ``plan_schema`` / ``plan_cache`` / ``plan_prompt`` internals have their own test files (here only the node
wiring and its exact error contract are asserted); the real HTTP transport of ``llm_client.chat`` is covered by
``tests/test_llm_proxy_error.py`` (one end-to-end run through a fake opener is done here for key hygiene); ComfyUI's real
``folder_paths`` is not available, only its ``get_user_directory`` is stubbed.
"""

import contextlib
import importlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

from tests import _support
from tests.test_node_load import ROOT, load_pkg, unload_pkg
from tilemap_core import llm_client, plan_cache, plan_prompt, plan_schema

PKG = "_ccn_t15_plan_pkg"
PACK = "_ccn_t15_pack"
P = None
MODEL = "claude-sonnet-5"


def setUpModule():
    global P
    P = load_pkg(PKG, "tilemap_plan")


def tearDownModule():
    unload_pkg(PKG)
    unload_pkg(PACK)


def load_pack(name=PACK):
    """Execute the root ``__init__.py`` as package ``name`` (auto-discovery of all node modules); return it + its stdout."""
    load_pkg(name, "tilemap_plan")  # aliases tilemap_core under the new package name
    spec = importlib.util.spec_from_file_location(name, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        spec.loader.exec_module(mod)
    return mod, out.getvalue()


def reply(plan=None, **kw):
    """LLM reply text: JSON of ``plan`` (default ``_support.valid_plan(**kw)``)."""
    return json.dumps(_support.valid_plan(**kw) if plan is None else plan)


class PlanCase(unittest.TestCase):
    """Fresh temp user dir (``folder_paths`` stub) and a scripted ``llm_client.chat`` per test."""

    def setUp(self):
        self.user = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(mock.patch.object(P, "folder_paths", types.SimpleNamespace(get_user_directory=lambda: self.user)))
        self.cache = os.path.join(self.user, "ccn_tilemap_cache")
        self.node = P.CCN_TilemapPlan()

    def fake(self, *replies):
        """Replace ``llm_client.chat``; each call returns the next reply (an Exception instance is raised)."""
        chat = self.enterContext(mock.patch.object(llm_client, "chat", side_effect=list(replies)))
        return chat

    def plan(self, **kw):
        args = dict(prompt="forest", view="top-down", tile_size=16, max_tiles=10, model=MODEL)
        args.update(kw)
        with contextlib.redirect_stdout(io.StringIO()):
            return self.node.plan(**args)

    def files(self):
        return sorted(os.listdir(self.cache)) if os.path.isdir(self.cache) else []

    def key(self, prompt="forest", view="top-down", max_tiles=10, model=MODEL, ehash=""):
        return plan_cache.cache_key(model, prompt, view, max_tiles, ehash)


class ContractTests(unittest.TestCase):
    def test_class_attributes(self):
        cls = P.CCN_TilemapPlan
        self.assertEqual(cls.RETURN_TYPES, ("STRING", "STRING", "STRING"))
        self.assertEqual(cls.RETURN_NAMES, ("plan_json", "summary", "model_used"))
        self.assertEqual((cls.FUNCTION, cls.CATEGORY), ("plan", "custom_comfy_nodes/tilemap"))
        self.assertTrue(callable(getattr(cls(), cls.FUNCTION)))

    def test_input_types(self):
        spec = P.CCN_TilemapPlan.INPUT_TYPES()
        self.assertEqual(list(spec["required"]), ["prompt", "view", "tile_size", "max_tiles", "model", "model_override", "refresh_plan"])
        req, opt = spec["required"], spec["optional"]
        self.assertEqual(req["prompt"], ("STRING", {"multiline": True, "default": ""}))
        self.assertEqual(req["view"], (["top-down", "three-quarter", "side-view"], {"default": "top-down"}))
        self.assertEqual(req["tile_size"], ([8, 16, 24, 32, 48, 64], {"default": 16}))
        self.assertEqual(req["max_tiles"], ("INT", {"default": 24, "min": 1, "max": 48, "step": 1}))
        self.assertEqual(req["model"], (list(llm_client.MODELS), {"default": MODEL}))
        self.assertEqual(req["model_override"], ("STRING", {"default": ""}))
        self.assertEqual(req["refresh_plan"], ("BOOLEAN", {"default": False}))
        self.assertEqual(list(opt), ["plan_mode", "existing_plan"])
        self.assertEqual(opt["plan_mode"], (["new", "extend"], {"default": "new"}))
        self.assertEqual(opt["existing_plan"], ("STRING", {"forceInput": True, "default": ""}))
        self.assertIn(req["model"][1]["default"], req["model"][0])
        self.assertIn(req["view"][1]["default"], req["view"][0])

    def test_is_changed_forces_rerun_only_on_refresh(self):
        cls = P.CCN_TilemapPlan
        self.assertEqual(cls.IS_CHANGED(), "")
        self.assertEqual(cls.IS_CHANGED(refresh_plan=False, prompt="x", plan_mode="new"), "")
        for value in (cls.IS_CHANGED(refresh_plan=True), cls.IS_CHANGED(True, prompt="x")):
            self.assertNotEqual(value, value)  # NaN never equals the previous value

    def test_module_mappings(self):
        self.assertEqual(set(P.NODE_CLASS_MAPPINGS), {"CCN_TilemapPlan", "CCN_TilemapExpand"})
        self.assertIs(P.NODE_CLASS_MAPPINGS["CCN_TilemapPlan"], P.CCN_TilemapPlan)
        self.assertIs(P.NODE_CLASS_MAPPINGS["CCN_TilemapExpand"], P.CCN_TilemapExpand)
        self.assertEqual(P.NODE_DISPLAY_NAME_MAPPINGS, {"CCN_TilemapPlan": "Tilemap Plan (LLM)", "CCN_TilemapExpand": "Tilemap Expand (Flux jobs)"})

    def test_pack_registers_plan_and_expand(self):
        pack, out = load_pack()
        self.assertNotIn("tilemap_plan", out)  # no "failed to load nodes.tilemap_plan" / duplicate message
        for name, label in (("CCN_TilemapPlan", "Tilemap Plan (LLM)"), ("CCN_TilemapExpand", "Tilemap Expand (Flux jobs)")):
            cls = pack.NODE_CLASS_MAPPINGS[name]
            self.assertEqual((cls.__name__, cls.__module__), (name, f"{PACK}.nodes.tilemap_plan"))
            self.assertEqual(pack.NODE_DISPLAY_NAME_MAPPINGS[name], label)

    def test_constants(self):
        self.assertEqual((P.CACHE_DIRNAME, P.MAX_TILES_RANGE, P.LLM_MAX_TOKENS, P.LLM_TIMEOUT_S), ("ccn_tilemap_cache", (1, 48), 8000, 180))


class CacheDirTests(unittest.TestCase):
    def test_user_directory_is_used(self):
        with mock.patch.object(P, "folder_paths", types.SimpleNamespace(get_user_directory=lambda: "/u/dir")):
            self.assertEqual(P._cache_dir(), os.path.join("/u/dir", "ccn_tilemap_cache"))

    def test_temp_dir_without_comfyui_or_without_getter(self):
        for stub in (None, types.SimpleNamespace()):
            with mock.patch.object(P, "folder_paths", stub), mock.patch.object(P.tempfile, "gettempdir", return_value="/tmp/x"):
                self.assertEqual(P._cache_dir(), os.path.join("/tmp/x", "ccn_tilemap_cache"))


class NewModeTests(PlanCase):
    def test_happy_path(self):
        chat = self.fake(reply())
        text, summary, used = self.plan()
        data = json.loads(text)
        self.assertEqual((used, data["meta"]), (MODEL, {"tile_size": 16, "view": "top-down", "model": MODEL}))
        self.assertEqual(data["theme"], "test forest")
        self.assertEqual([t["id"] for t in data["tiles"]], ["g0", "g1", "o0", "v0"])
        self.assertEqual(data["transitions"], [["g0", "g1"]])
        self.assertNotIn("extend", data["meta"])
        expect = {k: v for k, v in data.items() if k != "meta"}
        self.assertEqual(expect, plan_schema.validate_plan(_support.valid_plan(), 10))
        self.assertEqual(text, json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
        self.assertEqual(chat.call_count, 1)

    def test_summary_text(self):
        plan = _support.valid_plan(2, 1, 1, variants=2)
        self.fake(reply(plan))
        _, summary, _ = self.plan(tile_size=24, view="side-view")
        self.assertEqual(summary, "Theme: test forest\n4 tiles, 8 Flux jobs | 24px | side-view | claude-sonnet-5 | plan from LLM\n"
                                  "ground: g0 x2, g1 x2\nobject: o0 x2\noverlay: v0 x2\ntransitions: g0<->g1")
        self.fake(reply(_support.valid_plan(1, 0, 0)))
        _, summary, _ = self.plan(prompt="only ground")
        self.assertEqual(summary, "Theme: test forest\n1 tiles, 1 Flux jobs | 16px | top-down | claude-sonnet-5 | plan from LLM\nground: g0")

    def test_llm_call_arguments_and_messages(self):
        chat = self.fake(reply())
        self.plan(prompt="a  forest", max_tiles=7, view="three-quarter")
        (messages, model), kw = chat.call_args
        self.assertEqual(messages, plan_prompt.build_messages("a  forest", "three-quarter", 7))
        self.assertEqual((model, kw), (MODEL, {"max_tokens": 8000, "timeout": 180}))  # no temperature

    def test_cache_file_written_with_key_and_validated_plan(self):
        self.fake(reply())
        self.plan()
        self.assertEqual(self.files(), [self.key() + ".json"])
        with open(os.path.join(self.cache, self.key() + ".json"), encoding="utf-8") as fh:
            self.assertEqual(json.load(fh), plan_schema.validate_plan(_support.valid_plan(), 10))

    def test_cache_hit_skips_llm_and_reports_cache(self):
        chat = self.fake(reply())
        first = self.plan()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            text, summary, used = self.node.plan("forest", "top-down", 16, 10, MODEL)
        self.assertEqual(chat.call_count, 1)
        self.assertIn("plan cache hit", out.getvalue())
        self.assertEqual((text, used), (first[0], first[2]))
        self.assertTrue(summary.splitlines()[1].endswith("plan from cache"))
        self.assertEqual(summary.replace("plan from cache", "plan from LLM"), first[1])

    def test_cache_miss_for_each_key_component(self):
        chat = self.fake(*[reply()] * 5)
        self.plan()
        for kw in ({"prompt": "desert"}, {"view": "side-view"}, {"max_tiles": 11}, {"model": "claude-opus-5"}):
            before = chat.call_count
            self.plan(**kw)
            self.assertEqual(chat.call_count, before + 1, kw)
        self.assertEqual(len(self.files()), 5)

    def test_tile_size_is_not_part_of_the_key(self):
        chat = self.fake(reply())
        a, b = self.plan(tile_size=16), self.plan(tile_size=32)
        self.assertEqual(chat.call_count, 1)
        self.assertEqual(json.loads(a[0])["meta"]["tile_size"], 16)
        self.assertEqual(json.loads(b[0])["meta"]["tile_size"], 32)
        self.assertIn("| 32px |", b[1])

    def test_refresh_plan_ignores_cache_and_rewrites_it(self):
        chat = self.fake(reply(), reply(_support.valid_plan(3, 1, 1)))
        self.plan()
        text, summary, _ = self.plan(refresh_plan=True)
        self.assertEqual(chat.call_count, 2)
        self.assertEqual(len(json.loads(text)["tiles"]), 5)
        self.assertIn("plan from LLM", summary)
        self.assertEqual(len(plan_cache.load(self.cache, self.key(), 10)["tiles"]), 5)

    def test_corrupt_or_invalid_cache_file_is_a_miss(self):
        os.makedirs(self.cache)
        path = os.path.join(self.cache, self.key() + ".json")
        for junk in ("not json", json.dumps({"theme": "x", "tiles": []})):
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(junk)
            chat = self.fake(reply())
            _, summary, _ = self.plan()
            self.assertEqual(chat.call_count, 1)
            self.assertIn("plan from LLM", summary)
            self.assertEqual(len(plan_cache.load(self.cache, self.key(), 10)["tiles"]), 4)

    def test_cached_plan_is_revalidated_with_the_users_max_tiles(self):
        plan_cache.store(self.cache, self.key(max_tiles=3), _support.valid_plan(), 10)  # 4 tiles stored under the max_tiles=3 key
        chat = self.fake(reply(_support.valid_plan(1, 1, 1)))
        _, summary, _ = self.plan(max_tiles=3)
        self.assertEqual(chat.call_count, 1)  # stored file has 4 tiles > 3: miss
        self.assertIn("plan from LLM", summary)

    def test_unwritable_cache_is_not_fatal(self):
        with open(os.path.join(self.user, "ccn_tilemap_cache"), "w") as fh:  # a file where the dir should be
            fh.write("x")
        self.fake(reply())
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            text, _, _ = self.node.plan("forest", "top-down", 16, 10, MODEL)
        self.assertEqual(len(json.loads(text)["tiles"]), 4)
        self.assertIn("plan cache not writable, continuing without cache", out.getvalue())
        with mock.patch.object(plan_cache, "store", side_effect=PermissionError("denied")):
            self.fake(reply())
            self.assertEqual(len(json.loads(self.plan(prompt="other")[0])["tiles"]), 4)

    def test_non_oserror_from_store_propagates(self):
        self.fake(reply())
        with mock.patch.object(plan_cache, "store", side_effect=ValueError("too big")), self.assertRaisesRegex(ValueError, "too big"):
            self.plan()

    def test_reply_wrapped_in_prose_or_fences_needs_no_repair(self):
        chat = self.fake("Here you go:\n```json\n" + reply() + "\n```\nEnjoy!")
        self.plan()
        self.assertEqual(chat.call_count, 1)

    def test_model_override(self):
        chat = self.fake(reply(), reply())
        self.assertEqual(self.plan(model_override="  my-model_1.0  ")[2], "my-model_1.0")
        self.assertEqual(chat.call_args.args[1], "my-model_1.0")
        self.assertEqual(json.loads(self.plan(model_override="my-model_1.0")[0])["meta"]["model"], "my-model_1.0")  # cache hit too
        self.assertEqual(json.loads(self.plan(model_override="   ", prompt="b")[0])["meta"]["model"], MODEL)
        self.assertIn(self.key(model="my-model_1.0") + ".json", self.files())

    def test_bad_model_override_fails_before_the_llm(self):
        chat = self.fake()
        for bad in ("bad name", "a" * 101, "x;y", "é"):
            with self.subTest(bad=bad), self.assertRaisesRegex(llm_client.LLMProxyError, "invalid model name"):
                self.plan(model_override=bad)
        self.assertEqual((chat.call_count, self.files()), (0, []))
