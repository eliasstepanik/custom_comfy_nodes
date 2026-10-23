"""Tests for tilemap_core.sanitize and tilemap_core.llm_client (fake opener, no network except loopback)."""

import http.server
import json
import logging
import os
import threading
import traceback
import unicodedata
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from tests import _support
from tilemap_core import llm_client, sanitize
from tilemap_core.llm_client import LLMProxyError

KEY = _support.CANARY_KEY
MSGS = [{"role": "user", "content": "hi"}]
URL = llm_client.BASE_URL + llm_client.CHAT_PATH
HOSTILE = ["../../etc/passwd", "..\\..\\win", "/abs/olute", "a\x00b", "\u202eevil", "a\u200bb", "x" * 10000,
           "a/" * 5000, "C:\\Windows", "~/x", "ÄÖÜ ß é", "tilemap/../../x", ".hidden", "\n\t", "日本語"]


class SanitizeTests(unittest.TestCase):
    def test_slug_hostile_and_bounds(self):
        cases = {"../../etc/passwd": "etc_passwd", "a\x00b": "a_b", "\u202eabc": "abc", "a\u200bb": "ab",
                 "Café Ünï": "cafe_uni", "  Hello, World!  ": "hello_world"}
        for raw, want in cases.items():
            self.assertEqual(sanitize.slug(raw), want, repr(raw))
        for raw in HOSTILE:
            try:
                out = sanitize.slug(raw)
            except ValueError:
                continue
            self.assertRegex(out, r"^[a-z0-9]([a-z0-9_]*[a-z0-9])?$", repr(raw))
            self.assertLessEqual(len(out), 32)
        self.assertEqual(sanitize.slug("a" * 10000, max_len=5), "aaaaa")
        self.assertEqual(sanitize.slug("abc_def", max_len=4), "abc")  # cut then re-stripped
        for bad in ("", "   ", "___", "日本語", "\u202e\u200b", "../..", None, 123, b"abc"):
            with self.assertRaises(ValueError, msg=repr(bad)):
                sanitize.slug(bad)
        with self.assertRaises(ValueError):
            sanitize.slug("abc", max_len=0)

    def test_clean_text_strips_dangerous_chars(self):
        self.assertEqual(sanitize.clean_text("a{b}|c<d>[e]\\f`g", 100), "abcdefg")
        self.assertEqual(sanitize.clean_text("a\x00b\x07c\u202ed\u200be\ud800", 100), "abcde")
        self.assertEqual(sanitize.clean_text(" a\t\tb\n\nc \u00a0 d ", 100), "a b c d")
        self.assertEqual(sanitize.clean_text("\uff1cscript\uff1e", 100), "script")  # NFKC folds, then stripped
        for raw in HOSTILE + ["<img src=x onerror=1>{{x}}|`rm`"]:
            out = sanitize.clean_text(raw, 500)
            self.assertFalse(set(out) & set("{}|<>[]\\`"), repr(raw))
            self.assertTrue(all(unicodedata.category(c)[0] != "C" for c in out), repr(raw))
        self.assertEqual(sanitize.clean_text("x" * 10000, 80) + sanitize.clean_text("abcdef", 3), "x" * 80 + "abc")
        self.assertEqual(sanitize.clean_text("ab cd", 3), "ab")  # trailing space not kept
        self.assertEqual(sanitize.clean_text("{{{{" * 100 + "ab", 2), "ab")  # strip first, cut second
        for args in (("abc", -1), (None, 10)):
            with self.assertRaises(ValueError):
                sanitize.clean_text(*args)

    def test_prefix_known_mappings(self):
        cases = {"../../etc/passwd": "tilemap/etc/passwd", "/abs/path": "tilemap/abs/path", "": "tilemap/tilemap",
                 "tilemap": "tilemap/tilemap", "a/b/c/d/e": "tilemap/a/b", "tilemap/a/b/c": "tilemap/a/b",
                 "..\\..\\x": "tilemap/x", "a\x00b": "tilemap/a_b", "C:\\Windows": "tilemap/C_/Windows",
                 "a b": "tilemap/a_b", "/./../": "tilemap/tilemap", "x" * 200: "tilemap/" + "x" * 64}
        for raw, want in cases.items():
            self.assertEqual(sanitize.sanitize_prefix(raw), want, repr(raw))
        with self.assertRaises(ValueError):
            sanitize.sanitize_prefix(None)
        for raw in HOSTILE + ["y" * 200 + "/" + "z" * 200, "tilemap\\..\\..\\q"]:
            out = sanitize.sanitize_prefix(raw)
            segs = out.split("/")
            self.assertEqual(segs[0], "tilemap", repr(raw))
            self.assertFalse(os.path.isabs(out) or "\\" in out, repr(raw))
            self.assertNotIn("..", out)
            self.assertTrue(2 <= len(segs) <= 3, out)
            for seg in segs:
                self.assertRegex(seg, r"^[A-Za-z0-9_-]{1,64}$", repr(raw))


def call(opener, sleeps=None, **kw):
    kw.setdefault("api_key", KEY)
    return llm_client.chat(MSGS, "claude-sonnet-5", opener=opener,
                           sleep=sleeps.append if sleeps is not None else (lambda s: None), **kw)


def errs(code, body=b"", n=1, headers=None):
    return [_support.http_error(code, body, headers) for _ in range(n)]


def fragments(text):
    """Every 4-char window of the canary key that occurs in ``text``."""
    return [KEY[i:i + 4] for i in range(len(KEY) - 3) if KEY[i:i + 4] in text]


class ChatTests(unittest.TestCase):
    def test_request_shape_and_parse(self):
        op = _support.make_opener((200, _support.chat_json("hello")), (200, _support.chat_json("{}")))
        self.assertEqual(call(op, timeout=45, max_tokens=77, temperature=0.25), "hello")
        req = op.requests[0]
        self.assertEqual((req["url"], req["method"], req["timeout"]), (URL, "POST", 45))
        self.assertEqual((req["headers"]["Authorization"], req["headers"]["Content-type"]),
                         (f"Bearer {KEY}", "application/json"))
        self.assertIs(req["body"]["stream"], False)
        self.assertEqual({k: req["body"][k] for k in ("model", "messages", "max_tokens", "temperature")},
                         {"model": "claude-sonnet-5", "messages": MSGS, "max_tokens": 77, "temperature": 0.25})
        self.assertNotIn("response_format", req["body"])
        call(op, response_format={"type": "json_object"})
        self.assertEqual(op.requests[1]["body"]["response_format"], {"type": "json_object"})

    def test_bad_responses_raise_llm_proxy_error(self):
        big = llm_client.MAX_RESPONSE_BYTES
        cases = {"json": b"{x", "choices": b"{}", "empty": _support.chat_json(" "), "utf8": b"\xff\xfe", "list": b"[]",
                 "exceeds": b"x" * (big + 1)}
        for name, body in cases.items():
            with self.assertRaisesRegex(LLMProxyError, "exceeds" if name == "exceeds" else "^(?!.*exceeds)", msg=name):
                call(_support.make_opener((200, body)))
        with self.assertRaisesRegex(LLMProxyError, "invalid JSON"):  # exactly at the cap: read fully, bad JSON
            call(_support.make_opener((200, b"x" * big)))

    def test_429_retry_after_then_503_then_success(self):
        op = _support.make_opener(*errs(429, headers={"Retry-After": "7"}), (503, b""), (200, _support.chat_json("ok")))
        sleeps = []
        self.assertEqual(call(op, sleeps), "ok")
        self.assertEqual((sleeps, len(op.requests)), ([7.0, 2.0], 3))

    def test_retry_after_parsing_and_cap(self):
        for value, want in {"7": 7.0, "999": 30.0, "-5": 0.0, "abc": 1.0, "nan": 1.0}.items():
            sleeps, ok = [], (200, _support.chat_json("ok"))
            call(_support.make_opener(*errs(429, headers={"Retry-After": value}), ok), sleeps)
            self.assertEqual(sleeps, [want], value)

    def test_500_exhaustion_backoff(self):
        op, sleeps = _support.make_opener(*errs(500, b"boom", 4)), []
        with self.assertRaises(LLMProxyError) as cm:
            call(op, sleeps)
        self.assertEqual((sleeps, len(op.requests)), ([1.0, 2.0, 4.0], 4))
        self.assertIn("after 4 attempts: HTTP 500: boom", str(cm.exception))

    def test_urlerror_and_timeout_retried(self):
        sleeps, ok = [], (200, _support.chat_json("ok"))
        op = _support.make_opener(urllib.error.URLError("down"), TimeoutError("slow"), ok)
        self.assertEqual(call(op, sleeps), "ok")
        self.assertEqual(sleeps, [1.0, 2.0])
        with self.assertRaisesRegex(LLMProxyError, "timeout"):
            call(_support.make_opener(TimeoutError("x"), TimeoutError("x")), retries=1)

    def test_401_403_400_302_not_retried(self):
        for code in (400, 401, 403, 302):
            op, sleeps = _support.make_opener(*errs(code, b"no", headers={"Location": "https://evil.example/"})), []
            with self.assertRaisesRegex(LLMProxyError, f"HTTP {code}"):
                call(op, sleeps)
            self.assertEqual((len(op.requests), sleeps), (1, []), code)


class KeyTests(unittest.TestCase):
    @staticmethod
    def _file(d, content):
        path = Path(d) / "k.json"
        path.write_bytes(content if isinstance(content, bytes) else json.dumps(content).encode())
        return str(path)

    def test_env_beats_file_blank_env_falls_back(self):
        with _support.temp_dir() as d:
            f = self._file(d, {"api_key": "file-key"})
            self.assertEqual(llm_client.load_api_key({llm_client.KEY_ENV: "  env-key  "}, f), "env-key")
            for blank in ("", "   ", None):
                self.assertEqual(llm_client.load_api_key({llm_client.KEY_ENV: blank}, f), "file-key")
            self.assertEqual(llm_client.load_api_key({}, self._file(d, {"api_key": " padded "})), "padded")

    def test_missing_or_broken_key_error_text(self):
        with _support.temp_dir() as d:
            for content in (b"{bad", {"api_key": ""}, {"api_key": 5}, {"other": 1}, [KEY], {"api_key": "a b"},
                            b"\xff", b" " * (llm_client.MAX_KEY_FILE_BYTES + 1)):
                with self.assertRaisesRegex(LLMProxyError, "LLM_PROXY_API_KEY.*/cache/llm_proxy.json"):
                    llm_client.load_api_key({}, self._file(d, content))
            with self.assertRaises(LLMProxyError):
                llm_client.load_api_key({}, str(Path(d) / "absent.json"))
        for bad in ("two words", "tab\tkey", "ünicode"):  # invalid charset: rejected, never echoed
            with self.assertRaises(LLMProxyError) as cm:
                llm_client.load_api_key({llm_client.KEY_ENV: bad}, "/x")
            self.assertNotIn(bad, str(cm.exception))
        op = _support.make_opener((200, _support.chat_json("ok")))  # chat() without api_key reads os.environ
        with mock.patch.dict(os.environ, {llm_client.KEY_ENV: "env-secret"}):
            llm_client.chat(MSGS, "auto", opener=op)
        self.assertEqual(op.requests[0]["headers"]["Authorization"], "Bearer env-secret")


class SsrfRedirectTests(unittest.TestCase):
    def test_bad_urls_refused_before_any_request(self):
        bad = ["http://llm.sailehd.systems/v1", "https://evil.example/v1", "https://llm.sailehd.systems.evil.example",
               "https://user:pw@llm.sailehd.systems/v1", "https://llm.sailehd.systems@evil.example/v1",
               "https://evil.example#@llm.sailehd.systems/v1", "ftp://llm.sailehd.systems/v1"]
        for base in bad:
            op = _support.make_opener((200, _support.chat_json("pwned")))
            with mock.patch.object(llm_client, "BASE_URL", base), self.assertRaises(LLMProxyError, msg=base):
                call(op)
            self.assertEqual(op.requests, [], base)
        self.assertEqual(call(_support.make_opener((200, _support.chat_json("ok")))), "ok")

    def test_default_opener_does_not_follow_302(self):
        hits = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                hits.append((self.path, self.headers.get("Authorization")))
                self.send_response(302 if self.path == "/start" else 200)
                self.send_header("Location", "/target")
                self.end_headers()

            def log_message(self, *args):
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{server.server_port}/start",
                                         headers={"Authorization": "Bearer " + KEY})
            with self.assertRaises(urllib.error.HTTPError) as cm:
                llm_client._default_opener(req, timeout=5)
            cm.exception.close()
        finally:
            server.shutdown()
            server.server_close()
        self.assertEqual(cm.exception.code, 302)
        self.assertEqual([p for p, _ in hits], ["/start"])  # redirect target never requested


class SecretLeakTests(unittest.TestCase):
    def assert_clean(self, *script, retries=0):
        """Run chat expecting LLMProxyError; no 4-char key fragment in str/repr/traceback/log output."""
        cap, root, lines = logging.Handler(0), logging.getLogger(), []
        cap.emit = lambda record: lines.append(record.getMessage())
        old_level = root.level
        root.addHandler(cap)
        root.setLevel(0)
        try:
            with self.assertRaises(LLMProxyError) as cm:
                call(_support.make_opener(*script), retries=retries)
        finally:
            root.removeHandler(cap)
            root.setLevel(old_level)
        exc = cm.exception
        tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
        hay = "\n".join([str(exc), repr(exc), *tb, *lines])
        self.assertEqual(fragments(hay), [], hay[:400])
        return str(exc)

    def test_key_echoed_in_error_bodies(self):
        for code in (401, 403, 429, 500):
            msg = self.assert_clean(*errs(code, f"invalid key {KEY} rejected".encode(), 2), retries=1)
            self.assertIn(f"HTTP {code}: invalid key *** rejected", msg)

    def test_key_in_json_message_and_network_reasons(self):
        body = json.dumps({"error": {"message": f"bad token {KEY}", "type": "auth"}}).encode()
        self.assertIn("***", self.assert_clean(*errs(401, body)))
        self.assert_clean(urllib.error.URLError(f"reason {KEY}"))
        self.assert_clean(TimeoutError(KEY), ConnectionError(KEY), retries=1)
        self.assert_clean((200, f"not json {KEY}".encode()))
        self.assert_clean((418, KEY.encode()))

    def test_key_straddling_300_char_cut(self):
        for off in [0, 1, *range(llm_client.ERROR_BODY_CHARS - 60, llm_client.ERROR_BODY_CHARS + 2)]:
            body = b"p" * off + KEY.encode() + b"p" * 50
            self.assert_clean(*errs(401, body))
            self.assert_clean(*errs(500, body, 2), retries=1)

    def test_key_straddling_2048_read_edge(self):
        with mock.patch.object(llm_client, "ERROR_BODY_CHARS", 10 * llm_client.ERROR_READ_BYTES):
            for off in range(llm_client.ERROR_READ_BYTES - len(KEY) - 1, llm_client.ERROR_READ_BYTES + 2):
                self.assert_clean(*errs(401, b"p" * off + KEY.encode() + b"p" * 20))

    def test_redact_before_truncate_order_matters(self):
        body = "p" * (llm_client.ERROR_BODY_CHARS - 10) + KEY
        naive = body[:llm_client.ERROR_BODY_CHARS].replace(KEY, "***")  # truncate-then-redact (the old bug)
        self.assertTrue(fragments(naive), "input must expose the truncate-first hazard")
        msg = self.assert_clean(*errs(401, body.encode()))
        self.assertIn("p" * 100, msg)  # body text still shown, only the key is hidden
        for k in (1, 2, 3):  # 1-3 key chars at the cut: only redact-before-truncate hides them
            body = b"p" * (llm_client.ERROR_BODY_CHARS - k) + KEY.encode()
            self.assertFalse(self.assert_clean(*errs(401, body)).endswith(KEY[:k]))
        self.assertEqual(llm_client._redact("xx" + KEY[:12], KEY), "xx***")
        self.assertEqual(llm_client._redact(KEY[-12:] + "xx", KEY), "***xx")
        self.assertEqual(llm_client._redact(f"a{KEY}b{KEY}", KEY), "a***b***")

    def test_error_snippet_capped_and_unreadable_body(self):
        msg = self.assert_clean(*errs(400, b"z" * 5000))
        self.assertIn("z" * llm_client.ERROR_BODY_CHARS, msg)
        self.assertNotIn("z" * (llm_client.ERROR_BODY_CHARS + 1), msg)
        err = errs(401)[0]
        err.read = mock.Mock(side_effect=OSError("closed"))
        with self.assertRaisesRegex(LLMProxyError, "^LLM proxy HTTP 401$"):
            call(_support.make_opener(err))


if __name__ == "__main__":
    unittest.main()
