"""Tests for ``nodes/llm_vision.py`` (D14, ``CCN_LLMVision``) and, through it, ``nodes/_tensor_io.image_list_to_np``.

Stdlib unittest, real torch IMAGE tensors, no network: the node calls ``llm_client.chat`` (module attribute), which the
tests wrap so the REAL ``chat`` runs with a scripted ``tests/_support`` opener, a canary key and a no-op ``sleep``. That
checks the request the proxy would receive (``image_url`` PNG data-URI parts, model, token/temperature/timeout fields)
and the error hygiene (no key, no image data, no exception text in messages). Node import: ``load_pkg`` from
``tests/test_node_load.py`` (functions only); ``tearDownModule`` removes the synthetic package from ``sys.modules``.

Residual gaps: live provider behaviour is not exercised; ``encode_image_part`` size-cap internals live in
``tests/test_review.py`` (only the ReviewError passthrough is asserted here); the retry/redaction matrix of
``llm_client.chat`` lives in its own tests (a few samples here).
"""

import base64
import io
import sys
import unittest
import urllib.error
from unittest import mock

import numpy as np
from PIL import Image

from tests import _support
from tests.test_node_load import load_pkg, torch, unload_pkg
from tilemap_core import llm_client, review

PKG = "_ccn_t13_vision_pkg"
KEY = _support.CANARY_KEY
V = None
REAL_CHAT = llm_client.chat


def setUpModule():
    global V
    if torch is None:
        raise unittest.SkipTest("torch not installed")
    V = load_pkg(PKG, "llm_vision")


def tearDownModule():
    unload_pkg(PKG)


def img(h=8, w=8, c=3, b=1, seed=0):
    arr = np.random.default_rng(seed).integers(0, 256, (b, h, w, c), dtype=np.uint8)
    return torch.from_numpy(arr.astype(np.float32) / np.float32(255.0))


def kw(**over):
    base = dict(image=img(), prompt="what is this?", model="claude-sonnet-5", max_tokens=512, max_side=1024)
    base.update(over)
    return base


class VisionCase(unittest.TestCase):
    def setUp(self):
        self.node = V.LLMVision()
        self.opener = _support.make_opener(*[(200, _support.chat_json("a cat"))] * 4)
        self.calls = []

        def chat(messages, model, **kwargs):
            self.calls.append(kwargs)
            return REAL_CHAT(messages, model, api_key=KEY, opener=self.opener, sleep=lambda _s: None, **kwargs)

        self.enterContext(mock.patch.object(llm_client, "chat", side_effect=chat))

    def run_node(self, **over):
        return self.node.describe(**kw(**over))

    def body(self, i=0):
        return self.opener.requests[i]["body"]

    def fails_any(self, **over):
        with self.assertRaises(RuntimeError) as cm:
            self.run_node(**over)
        exc = cm.exception
        self.assertIs(type(exc), RuntimeError)
        self.assertTrue(exc.__suppress_context__ and exc.__cause__ is None)  # raised `from None`
        return exc

    def fails(self, message, **over):
        exc = self.fails_any(**over)
        self.assertEqual(str(exc), message)
        return exc


class TestContract(unittest.TestCase):
    def test_input_types_exact(self):
        spec = V.LLMVision.INPUT_TYPES()
        self.assertEqual(list(spec), ["required", "optional"])
        req, opt = spec["required"], spec["optional"]
        self.assertEqual(list(req), ["image", "prompt", "model", "max_tokens", "max_side"])
        self.assertEqual(list(opt), ["system", "temperature", "timeout_s"])
        self.assertEqual(req["image"], ("IMAGE", {"tooltip": "Batch of 1..2 images; more are rejected, not dropped."}))
        self.assertEqual(req["prompt"], text_spec := ("STRING", {"multiline": True, "default": ""}))
        self.assertEqual(req["model"], (["claude-sonnet-5", "claude-haiku-4-5-20251001", "claude-opus-5", "auto"], {"default": "claude-sonnet-5"}))
        self.assertEqual(req["max_tokens"], ("INT", {"default": 512, "min": 1, "max": 32000, "step": 1}))
        self.assertEqual((req["max_side"][0], {k: v for k, v in req["max_side"][1].items() if k != "tooltip"}),
                         ("INT", {"default": 1024, "min": 16, "max": 1024, "step": 1}))
        self.assertEqual(opt["system"], text_spec)
        self.assertEqual((opt["temperature"][0], {k: v for k, v in opt["temperature"][1].items() if k != "tooltip"}),
                         ("FLOAT", {"default": -1.0, "min": -1.0, "max": 2.0, "step": 0.05}))
        self.assertEqual(opt["timeout_s"], ("INT", {"default": 120, "min": 5, "max": 600, "step": 1}))

    def test_surface_and_registration(self):
        cls = V.LLMVision
        self.assertEqual((cls.RETURN_TYPES, cls.RETURN_NAMES, cls.FUNCTION, cls.CATEGORY), (("STRING",), ("text",), "describe", "custom_comfy_nodes/tilemap"))
        self.assertEqual(V.NODE_CLASS_MAPPINGS, {"CCN_LLMVision": cls})
        self.assertEqual(V.NODE_DISPLAY_NAME_MAPPINGS, {"CCN_LLMVision": "LLM Vision (llm-proxy)"})
        self.assertFalse(hasattr(cls, "IS_CHANGED"))  # identical inputs must reuse the cached paid reply

    def test_widget_defaults_pass_validation(self):
        self.assertEqual((V._temperature(-1.0), V.TOKENS_DEFAULT, V.SIDE_DEFAULT), (None, 512, 1024))


class TestRequest(VisionCase):
    def test_happy_path_request_shape(self):
        out = self.run_node()
        self.assertEqual((out, self.calls), (("a cat",), [{"max_tokens": 512, "temperature": None, "timeout": 120}]))
        req = self.opener.requests[0]
        self.assertEqual((req["url"], req["method"], req["timeout"]), (llm_client.BASE_URL + llm_client.CHAT_PATH, "POST", 120))
        self.assertEqual(req["headers"]["Authorization"], f"Bearer {KEY}")
        body = self.body()
        self.assertEqual((body["model"], body["max_tokens"], body["stream"]), ("claude-sonnet-5", 512, False))
        self.assertNotIn("temperature", body)
        self.assertEqual(len(body["messages"]), 1)  # blank system: no system message
        msg = body["messages"][0]
        self.assertEqual(msg["role"], "user")
        self.assertEqual([p["type"] for p in msg["content"]], ["text", "image_url"])
        self.assertEqual(msg["content"][0]["text"], "what is this?")
        url = msg["content"][1]["image_url"]["url"]
        head, _, data = url.partition(",")
        png = Image.open(io.BytesIO(base64.b64decode(data)))
        self.assertEqual((head, png.format, png.size), ("data:image/png;base64", "PNG", (768, 768)))  # nearest-neighbour x96
        self.assertNotIn(KEY, str(body))

    def test_two_images_and_system_message(self):
        self.run_node(image=img(b=2), system="be brief")
        msgs = self.body()["messages"]
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])
        self.assertEqual(msgs[0]["content"], "be brief")
        self.assertEqual([p["type"] for p in msgs[1]["content"]], ["text", "image_url", "image_url"])

    def test_blank_system_is_dropped_and_text_is_cleaned(self):
        self.run_node(system=" \n\t ", prompt="a\r\nb\x00c")
        msgs = self.body()["messages"]
        self.assertEqual((len(msgs), msgs[0]["content"][0]["text"]), (1, "a\nbc"))

    def test_temperature_semantics(self):
        for temp, sent in ((-1.0, None), (-0.05, None), (0, 0.0), (0.0, 0.0), (0.7, 0.7), (2.0, 2.0), (1, 1.0)):
            with self.subTest(temp=temp):
                self.calls.clear()
                self.opener._script.append((200, _support.chat_json("t")))
                self.run_node(temperature=temp)
                self.assertEqual(self.calls[0]["temperature"], sent)
                body = self.opener.requests[-1]["body"]
                self.assertEqual(("temperature" in body, body.get("temperature")), (sent is not None, sent))

    def test_boundary_values_are_accepted_and_forwarded(self):
        for tokens, side, timeout in ((1, 16, 5), (32000, 1024, 600), (512.0, 100.0, 60.0)):
            with self.subTest(tokens=tokens, side=side, timeout=timeout):
                self.calls.clear()
                self.opener._script.append((200, _support.chat_json("ok")))
                self.run_node(max_tokens=tokens, max_side=side, timeout_s=timeout)
                self.assertEqual(self.calls[0], {"max_tokens": int(tokens), "temperature": None, "timeout": int(timeout)})
                self.assertEqual((type(self.calls[0]["max_tokens"]), type(self.calls[0]["timeout"]), self.opener.requests[-1]["timeout"]), (int, int, int(timeout)))

    def test_images_are_downscaled_to_max_side_before_encoding(self):
        seen = []
        real = V.encode_image_part
        with mock.patch.object(V, "encode_image_part", side_effect=lambda a: seen.append(a) or real(a)):
            self.run_node(image=img(h=64, w=32), max_side=16)
            self.run_node(image=img(h=64, w=32), max_side=64)
            self.run_node(image=img(h=40, w=100, c=4), max_side=50)
        self.assertEqual([(a.shape, a.dtype) for a in seen], [((16, 8, 3), np.uint8), ((64, 32, 3), np.uint8), ((20, 50, 3), np.uint8)])

    def test_fit_helper(self):
        a = np.zeros((10, 20, 3), np.uint8)
        self.assertIs(V._fit(a, 20), a)
        self.assertIs(V._fit(a, 1024), a)
        self.assertEqual(V._fit(a, 10).shape, (5, 10, 3))
        for shape, side, want in (((2, 1000), 16, (1, 16)), ((1000, 2), 16, (16, 1)), ((30, 100), 50, (15, 50))):
            self.assertEqual(V._fit(np.zeros(shape + (3,), np.uint8), side).shape, want + (3,))  # never 0 px

    def test_rgba_alpha_dropped_and_input_tensor_unchanged(self):
        rgba = img(c=4, seed=3)
        before = rgba.clone()
        seen = []
        real = V.encode_image_part
        with mock.patch.object(V, "encode_image_part", side_effect=lambda a: seen.append(a) or real(a)):
            self.run_node(image=rgba)
        self.assertTrue(torch.equal(rgba, before))
        self.assertEqual(seen[0].shape, (8, 8, 3))
        self.assertTrue(np.array_equal(seen[0], np.rint(before[0, :, :, :3].numpy() * 255).astype(np.uint8)))

    def test_deterministic_requests(self):
        for seed in (5, 5, 6):
            self.run_node(image=img(seed=seed), system="s")
        self.assertEqual((self.body(0) == self.body(1), self.body(0) == self.body(2)), (True, False))


class TestInputErrors(VisionCase):
    def test_prompt_and_system_errors(self):
        for bad in ("", "  \n\t", "\x00\x01"):
            self.fails("CCN_LLMVision: prompt is empty", prompt=bad)
        self.fails("CCN_LLMVision: prompt must be text, got int", prompt=5)
        self.fails("CCN_LLMVision: system must be text, got NoneType", system=None)
        self.fails("CCN_LLMVision: prompt is too long (60001 chars, max 60000)", prompt="x" * 60001)
        self.assertIn("system is too long", str(self.fails_any(system="y" * 60001)))
        self.run_node(prompt="x" * 60000)  # exactly the cap passes
        self.assertEqual(len(self.opener.requests), 1)

    def test_numeric_widget_errors(self):
        cases = [("max_side", 15, "max_side must be 16..1024, got 15"), ("max_side", 1025, "max_side must be 16..1024, got 1025"),
                 ("max_tokens", 0, "max_tokens must be 1..32000, got 0"), ("max_tokens", 32001, "max_tokens must be 1..32000, got 32001"),
                 ("timeout_s", 4, "timeout_s must be 5..600, got 4"), ("timeout_s", 601, "timeout_s must be 5..600, got 601")]
        for name, value, msg in cases:
            with self.subTest(name=name, value=value):
                self.fails(f"CCN_LLMVision: {msg}", **{name: value})
        for name in ("max_side", "max_tokens", "timeout_s"):
            for bad in (True, False, 16.5, float("nan"), float("inf"), "20", None):
                self.fails(f"CCN_LLMVision: {name} must be a whole number", **{name: bad})

    def test_temperature_errors(self):
        for bad in (2.01, 3):
            self.fails(f"CCN_LLMVision: temperature must be -1.0..2.0, got {bad}", temperature=bad)
        self.fails("CCN_LLMVision: temperature must be -1.0..2.0, got -1.01", temperature=-1.01)
        for bad in (True, float("nan"), float("inf"), "hot", None):
            self.fails("CCN_LLMVision: temperature must be a number", temperature=bad)

    def test_image_errors(self):
        self.fails("CCN_LLMVision: image batch has 3 images, at most 2 are allowed", image=img(b=3))
        self.fails("CCN_LLMVision: image[0] must be a torch.Tensor, got ndarray", image=np.zeros((1, 8, 8, 3), np.float32))
        self.assertIn("floating-point", str(self.fails_any(image=torch.zeros((1, 8, 8, 3), dtype=torch.uint8))))
        self.assertIn("shape [B,H,W,C]", str(self.fails_any(image=torch.zeros((8, 8, 3)))))
        self.assertIn("has 1 channels, expected 3", str(self.fails_any(image=img(c=1))))
        self.assertIn("is empty", str(self.fails_any(image=torch.zeros((0, 8, 8, 3)))))
        self.assertEqual(self.opener.requests, [])

    def test_validation_precedes_network_and_key_lookup(self):
        with mock.patch.object(llm_client, "load_api_key", side_effect=AssertionError("key read")):
            self.fails("CCN_LLMVision: prompt is empty", prompt="")
        self.assertEqual(self.opener.requests, [])


class TestErrorHygiene(VisionCase):
    def test_invalid_model_is_wrapped_before_any_request(self):
        for bad in ("bad model!", "", None, "x" * 101):
            with self.subTest(model=bad):
                exc = self.fails_any(model=bad)
                self.assertEqual(str(exc), "CCN_LLMVision: invalid model name (allowed: letters, digits, . _ : / -, max 100 chars)")
        self.assertEqual(self.opener.requests, [])
        self.assertEqual((self.run_node(model="auto"), self.body()["model"]), (("a cat",), "auto"))

    def test_http_error_hides_key_and_image_data(self):
        body = f"bad key {KEY}".encode()
        self.opener._script[:] = [urllib.error.HTTPError("http://x", 401, "no", {}, io.BytesIO(body))]
        exc = self.fails_any()
        text = str(exc)
        self.assertTrue(text.startswith("CCN_LLMVision: LLM proxy HTTP 401"))
        self.assertNotIn(KEY, text)
        sent = self.body()["messages"][0]["content"][1]["image_url"]["url"]
        self.assertNotIn(sent[:64], text + repr(exc))  # the request body (image data) never reaches the message
        self.assertEqual(text, "CCN_LLMVision: LLM proxy HTTP 401: bad key ***")

    def test_proxy_error_in_http_200_is_reported(self):
        payload = b'{"error": {"message": "temperature is deprecated for this model.", "type": "proxy_error"}}'
        self.opener._script[:] = [(200, payload)]
        exc = self.fails_any(temperature=0.5)
        self.assertIn("temperature is deprecated", str(exc))
        self.assertEqual(self.body()["temperature"], 0.5)

    def test_network_failure_retries_then_reports_without_internals(self):
        self.opener._script[:] = [urllib.error.URLError("dns")] * 4
        exc = self.fails_any()
        self.assertEqual((str(exc), len(self.opener.requests)),
                         ("CCN_LLMVision: LLM proxy request failed after 4 attempts: network error (URLError)", 4))

    def test_review_error_passthrough(self):
        with mock.patch.object(V, "encode_image_part", side_effect=review.ReviewError("image: too big")):
            exc = self.fails_any()
        self.assertEqual(str(exc), "CCN_LLMVision: image: too big")

    def test_unexpected_exception_shows_type_only(self):
        for err in (KeyError(f"secret-{KEY}"), MemoryError(), ZeroDivisionError("division by zero")):
            with self.subTest(err=type(err).__name__), mock.patch.object(llm_client, "chat", side_effect=err):
                exc = self.fails_any()
                self.assertEqual(str(exc), f"CCN_LLMVision: unexpected {type(err).__name__} during vision request")
                self.assertNotIn("secret", str(exc))
                self.assertTrue(exc.__suppress_context__ and exc.__cause__ is None)

    def test_non_exception_base_errors_are_not_swallowed(self):
        with mock.patch.object(llm_client, "chat", side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
            self.run_node()


if __name__ == "__main__":
    unittest.main()
