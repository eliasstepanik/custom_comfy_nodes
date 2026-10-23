"""Shared helpers for the tilemap_core unittest suite.

Stdlib + numpy + PIL only. No torch, no network, no import-time IO, no writes
outside ``tempfile`` directories. Tests import it as ``from tests import _support``
(run with ``python -B -m unittest discover -s tests -t .`` from the repo root).
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import tempfile
import urllib.error
import urllib.request
from typing import Any, Iterator, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

# Obviously fake; matches llm_client's key rule (printable ASCII, no spaces, <= 512).
CANARY_KEY = "sk-test-CANARY-0123456789abcdefghijklmnopqrstuvwxyz-NOT-A-REAL-KEY"


class FakeResponse:
    """Minimal urllib-style response: ``status``, ``headers``, ``read(n)``, context manager."""

    def __init__(self, body: bytes = b"", status: int = 200, headers: Optional[Mapping[str, str]] = None) -> None:
        self.status = status
        self.headers = dict(headers or {})
        self.closed = False
        self._buf = io.BytesIO(body)

    def read(self, n: int = -1) -> bytes:
        """Return up to ``n`` bytes (all when ``n`` < 0), like ``http.client.HTTPResponse``."""
        return self._buf.read(n)

    def close(self) -> None:
        """Mark the response closed."""
        self.closed = True

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def http_error(code: int, body: bytes = b"", headers: Optional[Mapping[str, str]] = None) -> urllib.error.HTTPError:
    """Build an ``HTTPError`` whose ``read()`` yields ``body`` and whose ``headers`` is a dict."""
    return urllib.error.HTTPError("https://fake.invalid/", code, f"fake {code}", dict(headers or {}), io.BytesIO(body))


ScriptItem = Union[FakeResponse, BaseException, Tuple[int, bytes]]


class FakeOpener:
    """Scripted ``opener(req, timeout=...)`` for ``tilemap_core.llm_client.chat``.

    Each call consumes the next script item: a ``FakeResponse`` is returned, an
    exception instance is raised, a ``(status, body)`` tuple becomes a
    ``FakeResponse``. Calling past the end of the script raises ``AssertionError``.
    ``requests`` records one dict per call: url, method, headers, body (parsed
    JSON, or raw bytes if not JSON), timeout.
    """

    def __init__(self, script: Sequence[ScriptItem]) -> None:
        self._script = list(script)
        self.requests: list = []

    def __call__(self, req: urllib.request.Request, timeout: float = 0) -> FakeResponse:
        raw = req.data or b""
        try:
            body: Any = json.loads(raw.decode("utf-8"))
        except ValueError:
            body = raw
        self.requests.append(
            {
                "url": req.full_url,
                "method": req.get_method(),
                "headers": dict(req.header_items()),
                "body": body,
                "timeout": timeout,
            }
        )
        if len(self.requests) > len(self._script):
            raise AssertionError(f"opener called {len(self.requests)} times, script has {len(self._script)} items")
        item = self._script[len(self.requests) - 1]
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, tuple):
            return FakeResponse(item[1], item[0])
        return item


def make_opener(*script: ScriptItem) -> FakeOpener:
    """Return a ``FakeOpener`` replaying ``script`` in order (see ``FakeOpener``)."""
    return FakeOpener(script)


def chat_json(content: str) -> bytes:
    """OpenAI-style chat completion body whose ``choices[0].message.content`` is ``content``."""
    payload = {"choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}]}
    return json.dumps(payload).encode("utf-8")


def valid_plan(n_ground: int = 2, n_object: int = 1, n_overlay: int = 1, variants: int = 1) -> dict:
    """Plan dict accepted by ``plan_schema.validate_plan``; ground ``g0``/``g1`` get one transition pair."""
    tiles: list = []
    for cat, count, prefix in (("ground", n_ground, "g"), ("object", n_object, "o"), ("overlay", n_overlay, "v")):
        for i in range(count):
            tiles.append(
                {
                    "id": f"{prefix}{i}",
                    "name": f"{cat} {i}",
                    "category": cat,
                    "description": f"A pixel art {cat} tile number {i} for the test map.",
                    "variants": variants,
                    "footprint": [1, 1],
                }
            )
    return {
        "theme": "test forest",
        "tiles": tiles,
        "transitions": [["g0", "g1"]] if n_ground >= 2 else [],
    }


def noise_rgb(h: int, w: int, seed: int = 0) -> np.ndarray:
    """Deterministic uniform-noise ``uint8`` RGB array of shape ``(h, w, 3)``."""
    return np.random.default_rng(seed).integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def circle_on_color(
    size: int,
    bg: Tuple[int, int, int] = (255, 0, 255),
    fg: Tuple[int, int, int] = (200, 60, 60),
    radius_frac: float = 0.35,
) -> np.ndarray:
    """``uint8`` RGB ``(size, size, 3)``: a centred ``fg`` disc (radius ``size*radius_frac``) on flat ``bg``."""
    yy, xx = np.mgrid[0:size, 0:size]
    c = (size - 1) / 2.0
    inside = (yy - c) ** 2 + (xx - c) ** 2 <= (size * radius_frac) ** 2
    img = np.empty((size, size, 3), dtype=np.uint8)
    img[:] = bg
    img[inside] = fg
    return img


def gradient_rgb(h: int, w: int) -> np.ndarray:
    """``uint8`` RGB ``(h, w, 3)``: red grows left to right, green top to bottom, blue is constant 128."""
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[..., 0] = np.linspace(0, 255, w).astype(np.uint8)[None, :]
    img[..., 1] = np.linspace(0, 255, h).astype(np.uint8)[:, None]
    img[..., 2] = 128
    return img


def opaque_rgba_tile(n: int, seed: int, palette: Sequence[Tuple[int, int, int]]) -> np.ndarray:
    """``uint8`` RGBA ``(n, n, 4)``, alpha 255, every pixel one of ``palette`` (random choice per pixel)."""
    pal = np.asarray(palette, dtype=np.uint8)
    idx = np.random.default_rng(seed).integers(0, len(pal), size=(n, n))
    out = np.empty((n, n, 4), dtype=np.uint8)
    out[..., :3] = pal[idx]
    out[..., 3] = 255
    return out


@contextlib.contextmanager
def temp_dir() -> Iterator[str]:
    """Yield a fresh temporary directory path, removed on exit."""
    with tempfile.TemporaryDirectory(prefix="ccn-test-") as d:
        yield d


@contextlib.contextmanager
def silence_logging(level: int = logging.CRITICAL) -> Iterator[None]:
    """Disable log records up to ``level`` for the duration (expected-warning noise in tests)."""
    previous = logging.root.manager.disable
    logging.disable(level)
    try:
        yield
    finally:
        logging.disable(previous)


def demo() -> None:
    """Self-check: ``python -B -m tests._support`` from the repo root."""
    from tilemap_core import llm_client
    from tilemap_core.plan_schema import validate_plan

    for args in ((2, 1, 1, 1), (1, 0, 0, 1), (3, 2, 2, 2), (4, 4, 4, 1), (2, 0, 0, 4)):
        plan = valid_plan(*args)
        out = validate_plan(plan, len(plan["tiles"]))
        assert len(out["tiles"]) == sum(args[:3]), args
        assert len(out["transitions"]) == (1 if args[0] >= 2 else 0), args

    opener = make_opener(http_error(429, b"slow down", {"Retry-After": "0"}), (200, chat_json("hello")))
    slept: list = []
    text = llm_client.chat(
        [{"role": "user", "content": "hi"}], "claude-sonnet-5", api_key=CANARY_KEY, opener=opener, sleep=slept.append
    )
    assert text == "hello" and len(opener.requests) == 2 and slept == [0.0]
    req = opener.requests[0]
    assert req["url"] == llm_client.BASE_URL + llm_client.CHAT_PATH and req["method"] == "POST"
    assert req["headers"]["Authorization"] == f"Bearer {CANARY_KEY}"
    assert req["body"]["model"] == "claude-sonnet-5" and req["body"]["stream"] is False
    try:
        opener(urllib.request.Request("https://fake.invalid/", data=b"{}"), timeout=1)
    except AssertionError:
        pass
    else:
        raise AssertionError("exhausted opener did not fail")
    tuple_opener = make_opener((503, b"x"))
    tuple_resp = tuple_opener(urllib.request.Request("https://fake.invalid/", data=b"{}"), timeout=1)
    assert isinstance(tuple_resp, FakeResponse) and tuple_resp.status == 503 and tuple_resp.read() == b"x"
    with FakeResponse(b"abc") as resp:
        assert resp.read(2) == b"ab" and resp.read(5) == b"c"
    assert resp.closed

    n = noise_rgb(5, 7, 1)
    assert n.shape == (5, 7, 3) and n.dtype == np.uint8 and np.array_equal(n, noise_rgb(5, 7, 1))
    assert not np.array_equal(n, noise_rgb(5, 7, 2))
    c = circle_on_color(32)
    assert c.shape == (32, 32, 3) and c.dtype == np.uint8
    assert tuple(c[0, 0]) == (255, 0, 255) and tuple(c[16, 16]) == (200, 60, 60)
    g = gradient_rgb(4, 6)
    assert g.shape == (4, 6, 3) and g.dtype == np.uint8 and g[0, 0, 0] == 0 and g[0, -1, 0] == 255
    pal = [(10, 20, 30), (200, 100, 50)]
    t = opaque_rgba_tile(8, 3, pal)
    assert t.shape == (8, 8, 4) and t.dtype == np.uint8 and (t[..., 3] == 255).all()
    assert {tuple(p) for p in t[..., :3].reshape(-1, 3)} <= set(pal)
    with temp_dir() as d, silence_logging():
        assert d
    print("tests._support self-check OK")


if __name__ == "__main__":
    demo()
