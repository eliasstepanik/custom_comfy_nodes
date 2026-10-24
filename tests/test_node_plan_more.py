"""More tests for ``CCN_TilemapPlan`` in ``plan_mode`` new (D16c): the one-shot repair retry, the exact final error text,
input validation order (all argument errors fire before any LLM call or cache write) and error-text hygiene (the API
key never reaches an exception, the summary, the plan text, or the cache files). Stdlib unittest; no network.

Helpers (``PlanCase``, ``reply``, ``MODEL``, ``setUpModule``/``tearDownModule``) come from ``tests/test_node_plan.py``.

Residual gaps: the retry/backoff of ``llm_client.chat`` itself (HTTP 429/5xx) is covered by the llm_client tests; here the
real ``chat`` runs once through a fake opener only to prove that key redaction survives the node. Messages of
``plan_schema`` errors are matched by their stable prefixes only.
"""

import contextlib
import inspect
import io
import json
import os
import unittest
from unittest import mock

from tests import _support
from tests.test_node_plan import MODEL, PlanCase, reply, setUpModule, tearDownModule  # noqa: F401  (module fixtures)
from tests import test_node_plan
from tilemap_core import llm_client, plan_cache, plan_prompt, plan_schema

KEY = _support.CANARY_KEY
BAD = "this is not a plan"


class RepairTests(PlanCase):
    def test_repair_after_bad_json_is_one_extra_call_with_error_text(self):
        chat = self.fake(BAD, reply())
        text, summary, _ = self.plan()
        self.assertEqual(chat.call_count, 2)
        first, second = chat.call_args_list
        base = plan_prompt.build_messages("forest", "top-down", 10)
        self.assertEqual(first.args[0], base)
        err = str(self.error_of(BAD, 10))
        self.assertEqual(second.args[0], plan_prompt.build_repair_messages(base, BAD, err))
        self.assertEqual(second.args[0][-2:], [{"role": "assistant", "content": BAD}, second.args[0][-1]])
        self.assertEqual((second.args[1], second.kwargs), (MODEL, {"max_tokens": 8000, "timeout": 180}))
        self.assertEqual(len(json.loads(text)["tiles"]), 4)
        self.assertIn("plan from LLM", summary)
        self.assertEqual(len(self.files()), 1)

    @staticmethod
    def error_of(text, max_tiles):
        try:
            plan_schema.parse_and_validate(text, max_tiles)
        except plan_schema.PlanError as exc:
            return exc
        raise AssertionError("text unexpectedly valid")

    def test_schema_violation_triggers_repair_with_users_max_tiles(self):
        too_many = reply(_support.valid_plan(2, 2, 2))  # 6 tiles > max_tiles 5
        chat = self.fake(too_many, reply(_support.valid_plan(2, 1, 1)))
        text, _, _ = self.plan(max_tiles=5)
        self.assertEqual(chat.call_count, 2)
        self.assertIn("tile", chat.call_args_list[1].args[0][-1]["content"])
        self.assertEqual(len(json.loads(text)["tiles"]), 4)
        self.assertEqual(self.files(), [self.key(max_tiles=5) + ".json"])

    def test_two_bad_replies_raise_plan_error_with_both_errors_and_excerpt(self):
        second = "\x00junk\n\n\treply\x1b[31m  " + "word \n" * 100
        chat = self.fake(BAD, second)
        with self.assertRaises(plan_schema.PlanError) as cm:
            self.plan()
        msg = str(cm.exception)
        self.assertEqual(chat.call_count, 2)  # exactly one repair, never more
        self.assertTrue(msg.startswith("the LLM did not return a valid tile plan after one repair attempt. First error: "))
        self.assertIn(str(self.error_of(BAD, 10)), msg)
        self.assertIn(" Second error: ", msg)
        head, _, excerpt = msg.partition("Reply start: ")
        self.assertTrue(excerpt.startswith("junk reply [31m word word"), excerpt[:30])
        self.assertTrue(excerpt.isprintable() and 190 <= len(excerpt) <= 200, len(excerpt))
        self.assertIsNone(cm.exception.__cause__)
        self.assertEqual(self.files(), [])  # nothing cached on failure

    def test_each_error_part_is_cut_to_300_chars(self):
        err = plan_schema.PlanError("E" * 1000)
        with mock.patch.object(plan_schema, "parse_and_validate", side_effect=err):
            self.fake(BAD, BAD)
            with self.assertRaises(plan_schema.PlanError) as cm:
                self.plan()
        msg = str(cm.exception)
        self.assertIn("First error: " + "E" * 300 + " Second error: " + "E" * 300 + " Reply start: ", msg)
        self.assertNotIn("E" * 301, msg)

    def test_llm_proxy_error_propagates_at_once_without_repair(self):
        chat = self.fake(llm_client.LLMProxyError("LLM proxy upstream error: boom"))
        with self.assertRaisesRegex(llm_client.LLMProxyError, "^LLM proxy upstream error: boom$"):
            self.plan()
        self.assertEqual((chat.call_count, self.files()), (1, []))

    def test_proxy_error_on_the_repair_call_propagates(self):
        chat = self.fake(BAD, llm_client.LLMProxyError("second call down"))
        with self.assertRaisesRegex(llm_client.LLMProxyError, "second call down"):
            self.plan()
        self.assertEqual(chat.call_count, 2)

    def test_repair_reply_is_stored_not_the_first(self):
        self.fake(BAD, reply(_support.valid_plan(3, 0, 0)))
        self.plan()
        self.assertEqual(len(plan_cache.load(self.cache, self.key(), 10)["tiles"]), 3)


class ValidationTests(PlanCase):
    def bad(self, regex, exc=ValueError, **kw):
        chat = self.fake()
        with self.assertRaisesRegex(exc, regex):
            self.plan(**kw)
        self.assertEqual((chat.call_count, self.files()), (0, []), kw)

    def test_tile_size(self):
        for bad in (0, 7, 17, 128, -16, True, 16.0, None, "", "abc", "16.0", "-16", "１６", [16], "1 6"):
            with self.subTest(tile_size=bad):
                self.bad(r"^tile_size must be one of \[8, 16, 24, 32, 48, 64\], got ", tile_size=bad)

    def test_tile_size_accepts_every_option_as_int_or_text(self):
        for size in (8, 16, 24, 32, 48, 64):
            for value in (size, str(size), f" {size} "):
                self.fake(reply())
                with self.subTest(value=value):
                    self.assertEqual(json.loads(self.plan(tile_size=value, prompt=f"p{size}{value}")[0])["meta"]["tile_size"], size)

    def test_view(self):
        for bad in ("", "isometric", "Top-Down", None, 3):
            with self.subTest(view=bad):
                self.bad(r"^view must be one of \['top-down', 'three-quarter', 'side-view'\], got ", view=bad)

    def test_max_tiles(self):
        for bad in (0, -1, 49, 1000, True, False, 5.0, "5", None):
            with self.subTest(max_tiles=bad):
                self.bad(r"^max_tiles must be an integer from 1 to 48, got ", max_tiles=bad)

    def test_max_tiles_bounds_are_accepted(self):
        self.fake(reply(_support.valid_plan(1, 0, 0)), reply())
        self.assertEqual(len(json.loads(self.plan(max_tiles=1)[0])["tiles"]), 1)
        self.assertEqual(len(json.loads(self.plan(max_tiles=48)[0])["tiles"]), 4)

    def test_plan_mode(self):
        for bad in ("", "EXTEND", "reuse", None):
            with self.subTest(plan_mode=bad):
                self.bad(r"^plan_mode must be one of \['new', 'extend'\], got ", plan_mode=bad)

    def test_prompt(self):
        for bad in ("", "   ", "\n\t", "<>"):
            with self.subTest(prompt=bad):
                self.bad(r"prompt is empty after removing unsupported characters", prompt=bad)
        self.bad(r"prompt must be a string, got int", prompt=5)

    def test_model_must_be_a_safe_id(self):
        self.bad("invalid model name", exc=llm_client.LLMProxyError, model="bad model")
        self.bad("invalid model name", exc=llm_client.LLMProxyError, model="")

    def test_existing_plan_without_extend_is_refused(self):
        self.bad(r"^existing_plan is only used with plan_mode extend$", existing_plan=json.dumps(_support.valid_plan()))
        self.fake(reply())
        self.plan(existing_plan="  \n ")  # blank text is the unconnected default

    def test_validation_order_tile_size_before_view_before_max_tiles(self):
        self.bad("tile_size", tile_size=1, view="x", max_tiles=0)
        self.bad("view must", view="x", max_tiles=0)
        self.bad("max_tiles must", max_tiles=0, model="bad model")
        self.bad("invalid model name", exc=llm_client.LLMProxyError, model="bad model", plan_mode="zzz")


class HygieneTests(PlanCase):
    def proxied(self, *contents):
        opener = _support.make_opener(*[(200, _support.chat_json(c)) for c in contents])
        self.enterContext(mock.patch.object(llm_client, "_default_opener", opener))
        self.enterContext(mock.patch.dict(os.environ, {llm_client.KEY_ENV: KEY}))
        return opener

    def tree_text(self):
        parts = []
        for name in self.files():
            with open(os.path.join(self.cache, name), encoding="utf-8") as fh:
                parts.append(fh.read())
        return "\n".join(parts)

    def test_success_through_real_chat_never_exposes_key(self):
        opener = self.proxied(reply())
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            result = self.node.plan("forest", "top-down", 16, 10, MODEL)
        self.assertEqual(len(opener.requests), 1)
        self.assertEqual(opener.requests[0]["headers"]["Authorization"], f"Bearer {KEY}")
        for text in (*result, out.getvalue(), self.tree_text()):
            self.assertNotIn(KEY, text)
        self.assertNotIn("temperature", opener.requests[0]["body"])

    def test_upstream_error_echoing_the_key_is_redacted(self):
        self.proxied(json.dumps({"error": {"message": f"bad credentials {KEY} rejected", "type": "proxy_error"}}))
        with self.assertRaises(llm_client.LLMProxyError) as cm:
            self.plan()
        self.assertIn("bad credentials", str(cm.exception))
        self.assertFalse(any(KEY[i:i + 8] in str(cm.exception) for i in range(len(KEY) - 7)))
        self.assertEqual(self.files(), [])

    def test_missing_key_message_is_fixed_text(self):
        self.enterContext(mock.patch.dict(os.environ, {llm_client.KEY_ENV: ""}))
        self.enterContext(mock.patch.object(llm_client, "KEY_FILE", os.path.join(self.user, "nokey.json")))
        with self.assertRaisesRegex(llm_client.LLMProxyError, "API key not configured"):
            self.plan()

    def test_plan_error_text_never_contains_key_material_from_inputs(self):
        # the prompt (user text) may legitimately be echoed by the LLM; the API key is not an input of the node
        self.proxied(BAD, BAD)
        with self.assertRaises(plan_schema.PlanError) as cm:
            self.plan()
        self.assertNotIn(KEY, str(cm.exception))
        self.assertNotIn("Authorization", str(cm.exception))
        self.assertNotIn("Bearer", str(cm.exception))

    def test_node_signature_has_no_key_parameter(self):
        sig = inspect.signature(test_node_plan.P.CCN_TilemapPlan.plan)
        self.assertFalse([n for n in sig.parameters if "key" in n.lower() or "token" in n.lower() or "secret" in n.lower()])
        spec = test_node_plan.P.CCN_TilemapPlan.INPUT_TYPES()
        self.assertFalse([n for g in spec.values() for n in g if "key" in n.lower() or "token" in n.lower()])


if __name__ == "__main__":
    unittest.main()
