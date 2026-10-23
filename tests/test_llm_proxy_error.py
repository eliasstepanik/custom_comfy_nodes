"""Tests for the claude-sonnet-5 'temperature' regression: proxy_error in HTTP 200, optional temperature,
plan-node error reporting and the CCN_LLMChat temperature widget (unittest, stdlib only, fake opener, no network)."""

import contextlib
import importlib
import json
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from tests import _support
from tilemap_core import expand, llm_client, plan_cache, plan_prompt, plan_schema, sanitize
from tilemap_core.llm_client import LLMProxyError

KEY = _support.CANARY_KEY
MSGS = [{"role": "user", "content": "hi"}]
MODEL = "claude-sonnet-5"
DEPRECATED = "Anthropic API error 400: temperature is deprecated for this model."
PKG = "_ccn_f4_pkg"  # synthetic parent package so the relative imports of nodes/* resolve without ComfyUI
_ALIASED = (expand, llm_client, plan_cache, plan_prompt, plan_schema, sanitize)
NODES = {}


def setUpModule():
    pkg = types.ModuleType(PKG)
    pkg.__path__ = [str(Path(__file__).resolve().parents[1])]
    sys.modules[PKG] = pkg
    core = types.ModuleType(f"{PKG}.tilemap_core")  # same module objects as the top-level tilemap_core
    core.__path__ = []
    sys.modules[core.__name__] = core
    for mod in _ALIASED:
        name = mod.__name__.split(".")[-1]
        sys.modules[f"{core.__name__}.{name}"] = mod
        setattr(core, name, mod)
    for name in ("llm_proxy", "tilemap_plan"):
        NODES[name] = importlib.import_module(f"{PKG}.nodes.{name}")


def tearDownModule():
    for name in [n for n in sys.modules if n == PKG or n.startswith(PKG + ".")]:
        del sys.modules[name]
    NODES.clear()


def proxy_error_content(message=DEPRECATED, kind="proxy_error"):
    return json.dumps({"error": {"message": message, "type": kind}})


def opener_for(*contents):
    return _support.make_opener(*[(200, _support.chat_json(c)) for c in contents])


def raw_opener(payload):
    return _support.make_opener((200, json.dumps(payload).encode("utf-8")))


def leaks(text, key=KEY, width=8):
    return any(key[i:i + width] in text for i in range(len(key) - width + 1))


def chat(opener, **kw):
    kw.setdefault("sleep", mock.Mock())
    return llm_client.chat(MSGS, MODEL, api_key=KEY, opener=opener, **kw)


@contextlib.contextmanager
def proxied(opener):
    """Route the real llm_client.chat of the nodes through ``opener`` with a fake key."""
    with mock.patch.object(llm_client, "_default_opener", opener):
        with mock.patch.dict(os.environ, {llm_client.KEY_ENV: KEY}):
            yield


class ProxyErrorContentTests(unittest.TestCase):
    def test_proxy_error_content_raises_once_without_key(self):
        opener = opener_for(proxy_error_content(f"{DEPRECATED} echo={KEY}"))
        sleep = mock.Mock()
        with self.assertRaises(LLMProxyError) as cm:
            chat(opener, sleep=sleep)
        self.assertIn("temperature is deprecated", str(cm.exception))
        self.assertNotIn(KEY, str(cm.exception))
        self.assertFalse(leaks(str(cm.exception)))
        self.assertEqual(len(opener.requests), 1)
        sleep.assert_not_called()

    def test_top_level_error_payloads_raise(self):
        for payload in ({"error": {"message": "boom upstream", "type": "api_error"}}, {"error": "boom upstream"}):
            opener = raw_opener(payload)
            with self.subTest(payload=payload), self.assertRaisesRegex(LLMProxyError, "boom upstream"):
                chat(opener)
            self.assertEqual(len(opener.requests), 1)

    def test_normal_and_foreign_error_content_returned_unchanged(self):
        other = proxy_error_content("model said this", kind="other")
        for content in (json.dumps(_support.valid_plan()), other, '{"not": "an error"}', "{ plain text, no JSON"):
            with self.subTest(content=content[:40]):
                self.assertEqual(chat(opener_for(content)), content)

    def test_redaction_when_key_straddles_truncation_boundary(self):
        cut = llm_client.ERROR_BODY_CHARS
        for offset in range(1, len(KEY)):  # key starts `offset` chars before the cut
            msg = "x" * (cut - offset) + KEY + " tail " * 20
            for payload in (
                {"choices": [{"message": {"content": proxy_error_content(msg)}}]},
                {"error": {"message": msg}},
            ):
                with self.subTest(offset=offset), self.assertRaises(LLMProxyError) as cm:
                    chat(raw_opener(payload))
                self.assertFalse(leaks(str(cm.exception)), str(cm.exception)[-60:])
                # redact-then-cut leaves no key stub at all; cut-then-redact keeps a stub below the fragment minimum
                self.assertNotIn(KEY[:3], str(cm.exception))


class TemperatureBodyTests(unittest.TestCase):
    def body(self, **kw):
        opener = opener_for("ok")
        chat(opener, **kw)
        return opener.requests[0]["body"]

    def test_none_and_default_omit_temperature(self):
        self.assertNotIn("temperature", self.body())
        self.assertNotIn("temperature", self.body(temperature=None))

    def test_zero_and_value_are_sent(self):
        zero = self.body(temperature=0.0)
        self.assertIn("temperature", zero)
        self.assertEqual(zero["temperature"], 0.0)
        self.assertEqual(self.body(temperature=0.25)["temperature"], 0.25)


class PlanNodeTests(unittest.TestCase):
    MESSAGES = plan_prompt.build_messages("forest", "top-down", 5)

    def ask(self, opener):
        plan_mod = NODES["tilemap_plan"]
        with proxied(opener), mock.patch.object(llm_client, "chat", wraps=llm_client.chat) as spy:
            try:
                return spy, plan_mod._ask_plan(self.MESSAGES, MODEL, 5), None
            except Exception as exc:  # noqa: BLE001 - the tests assert the type
                return spy, None, exc

    def test_proxy_error_propagates_without_repair_or_temperature(self):
        opener = opener_for(proxy_error_content())
        spy, plan, exc = self.ask(opener)
        self.assertIsNone(plan)
        self.assertIsInstance(exc, LLMProxyError)
        self.assertIn("temperature is deprecated", str(exc))
        self.assertNotIn("theme: missing", str(exc))
        self.assertEqual((spy.call_count, len(opener.requests)), (1, 1))
        self.assertNotIn("temperature", spy.call_args.kwargs)
        self.assertNotIn("temperature", opener.requests[0]["body"])

    def test_both_calls_omit_temperature(self):
        opener = opener_for("not a plan", json.dumps(_support.valid_plan()))
        spy, plan, exc = self.ask(opener)
        self.assertIsNone(exc)
        self.assertEqual(len(plan["tiles"]), 4)
        self.assertEqual((spy.call_count, len(opener.requests)), (2, 2))
        for call, req in zip(spy.call_args_list, opener.requests):
            self.assertNotIn("temperature", call.kwargs)
            self.assertNotIn("temperature", req["body"])

    def test_final_error_has_flat_reply_excerpt_of_at_most_200_chars(self):
        bad = "\x00junk\n\n\treply\x1b[31m\r\n  " + "word \n" * 100
        spy, plan, exc = self.ask(opener_for("first bad reply", bad))
        self.assertIsInstance(exc, plan_schema.PlanError)
        self.assertEqual(spy.call_count, 2)
        head, sep, excerpt = str(exc).partition("Reply start: ")
        self.assertEqual(sep, "Reply start: ")
        self.assertTrue(excerpt.startswith("junk reply [31m word word"), excerpt[:40])
        self.assertTrue(190 <= len(excerpt) <= 200, len(excerpt))  # cut at 200, a trailing space is stripped
        self.assertTrue(excerpt.isprintable())
        self.assertNotIn("  ", excerpt)
        self.assertNotIn("first bad reply", str(exc))

    def test_short_reply_excerpt_is_not_padded(self):
        _, _, exc = self.ask(opener_for("one", "two\nlines"))
        self.assertTrue(str(exc).endswith("Reply start: two lines"), str(exc)[-40:])


class LLMChatNodeTests(unittest.TestCase):
    def run_node(self, temperature, content="hello"):
        opener = opener_for(content)
        node = NODES["llm_proxy"].NODE_CLASS_MAPPINGS["CCN_LLMChat"]()
        with proxied(opener):
            out = node.chat("", "say hi", MODEL, "", 64, temperature, 30)
        return out, opener.requests[0]["body"]

    def test_widget_spec(self):
        spec = NODES["llm_proxy"].NODE_CLASS_MAPPINGS["CCN_LLMChat"].INPUT_TYPES()["required"]["temperature"]
        self.assertEqual(spec[0], "FLOAT")
        self.assertEqual((spec[1]["default"], spec[1]["min"], spec[1]["max"]), (-1.0, -1.0, 2.0))

    def test_negative_means_not_sent(self):
        for value in (-1.0, -0.05):
            with self.subTest(value=value):
                out, body = self.run_node(value)
                self.assertEqual(out, ("hello",))
                self.assertNotIn("temperature", body)

    def test_zero_and_positive_are_sent(self):
        for value in (0.0, 0.7, 2.0):
            with self.subTest(value=value):
                _, body = self.run_node(value)
                self.assertIn("temperature", body)
                self.assertEqual(body["temperature"], value)

    def test_proxy_error_surfaces_as_runtime_error(self):
        with self.assertRaisesRegex(RuntimeError, "temperature is deprecated") as cm:
            self.run_node(0.7, proxy_error_content())
        self.assertNotIn(KEY, str(cm.exception))


if __name__ == "__main__":
    unittest.main()
