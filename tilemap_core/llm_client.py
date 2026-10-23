"""llm-proxy chat client: API key loading and a retrying ``chat()`` over urllib.

Security model:
- The URL is always ``BASE_URL + CHAT_PATH``; callers cannot supply a URL, and
  the scheme (https) and host (allowlist) are re-checked before every request.
- The default opener does not follow redirects, so the Bearer key never leaves
  the allowlisted host.
- The API key is never logged and never placed in exception text; any echo of
  it in server output is replaced by ``***``.
- Response bodies are capped at ``MAX_RESPONSE_BYTES`` and parsed with
  ``json.loads`` only.

Stdlib only; no import-time IO or network.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Mapping, Optional

BASE_URL = "https://llm.sailehd.systems/v1"
CHAT_PATH = "/chat/completions"
KEY_ENV = "LLM_PROXY_API_KEY"
KEY_FILE = "/cache/llm_proxy.json"
MODELS = ["claude-sonnet-5", "claude-haiku-4-5-20251001", "claude-opus-5", "auto"]
MAX_RESPONSE_BYTES = 1_048_576
RETRY_STATUS = {429, 500, 502, 503, 504}
ALLOWED_HOSTS = frozenset({"llm.sailehd.systems"})
MAX_BACKOFF_S = 30.0
MAX_KEY_FILE_BYTES = 65_536
ERROR_BODY_CHARS = 300
ERROR_READ_BYTES = 2048  # >= 512 (max key length) + margin; _redact covers a key cut by this edge
_MIN_FRAGMENT = 4

_MODEL_RE = re.compile(r"^[A-Za-z0-9._:/-]{1,100}$")
_KEY_RE = re.compile(r"^[\x21-\x7e]{1,512}$")  # printable, no whitespace: no header injection
_ROLES = ("system", "user", "assistant")
_NOT_CONFIGURED = (
    "LLM proxy API key not configured: set LLM_PROXY_API_KEY or create "
    '/cache/llm_proxy.json ({"api_key": ...}, mode 600)'
)


class LLMProxyError(RuntimeError):
    """Any llm-proxy failure. Message is user-readable and never contains the key."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects: the resulting 3xx surfaces as an HTTPError."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _default_opener(req: urllib.request.Request, timeout: float):
    """Open ``req`` with an opener that never follows redirects."""
    return urllib.request.build_opener(_NoRedirect()).open(req, timeout=timeout)


def _valid_key(value: Any) -> bool:
    return isinstance(value, str) and _KEY_RE.match(value) is not None


def load_api_key(environ: Mapping[str, str] = os.environ, key_file: str = KEY_FILE) -> str:
    """Return the API key: env ``LLM_PROXY_API_KEY`` first, else ``key_file`` JSON.

    Raises ``LLMProxyError`` with a fixed message (never containing the key).
    """
    env_val = (environ.get(KEY_ENV) or "").strip()
    if env_val:
        if not _valid_key(env_val):
            raise LLMProxyError(f"{KEY_ENV} has invalid characters (printable ASCII, no spaces expected)")
        return env_val
    try:
        with open(key_file, "rb") as fh:
            raw = fh.read(MAX_KEY_FILE_BYTES + 1)
        if len(raw) > MAX_KEY_FILE_BYTES:
            raise ValueError("key file too large")
        data = json.loads(raw.decode("utf-8"))
        key = data.get("api_key") if isinstance(data, dict) else None
        key = key.strip() if isinstance(key, str) else key
    except (OSError, ValueError):  # JSONDecodeError and UnicodeDecodeError are ValueError
        raise LLMProxyError(_NOT_CONFIGURED) from None
    if not _valid_key(key):
        raise LLMProxyError(_NOT_CONFIGURED)
    return key


def validate_model(name: str) -> str:
    """Return ``name`` if it is a safe model id, else raise ``LLMProxyError``."""
    if not isinstance(name, str) or _MODEL_RE.match(name) is None:
        raise LLMProxyError("invalid model name (allowed: letters, digits, . _ : / -, max 100 chars)")
    return name


def _check_url(url: str) -> str:
    """SSRF guard: https scheme and allowlisted host only."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https" or parts.hostname not in ALLOWED_HOSTS or parts.username or parts.password:
        raise LLMProxyError("refusing request: URL must be https on an allowlisted host")
    return url


def _redact(text: str, key: str) -> str:
    """Replace ``key`` in ``text`` with ``***``, including key fragments cut by a read edge.

    Call this on the full (size-capped) text BEFORE any slicing/truncation, or a
    key straddling the cut leaks a prefix. A key fragment (>= ``_MIN_FRAGMENT``
    chars) at the very end (prefix of the key) or start (suffix) of ``text`` is
    also dropped, because a bounded read can cut the key in half.
    """
    if not key:
        return text
    text = text.replace(key, "***")
    for n in range(min(len(key) - 1, len(text)), _MIN_FRAGMENT - 1, -1):
        if text.endswith(key[:n]):
            text = text[:-n] + "***"
            break
    for n in range(min(len(key) - 1, len(text)), _MIN_FRAGMENT - 1, -1):
        if text.startswith(key[-n:]):
            text = "***" + text[n:]
            break
    return text


def _check_messages(messages: Any) -> None:
    if not isinstance(messages, list) or not messages:
        raise LLMProxyError("messages must be a non-empty list")
    for m in messages:
        if not isinstance(m, dict) or m.get("role") not in _ROLES or not isinstance(m.get("content"), (str, list)):
            raise LLMProxyError("each message needs a role (system/user/assistant) and str or list content")


def _retry_delay(headers: Any, attempt: int) -> float:
    """Seconds to sleep: Retry-After if numeric, else ``2**attempt``; capped at 30."""
    delay = float(2 ** attempt)
    raw = headers.get("Retry-After") if headers is not None else None
    if raw:
        try:
            delay = float(raw)
        except (TypeError, ValueError):
            pass
    if delay != delay:  # NaN
        delay = float(2 ** attempt)
    return max(0.0, min(MAX_BACKOFF_S, delay))


def _upstream_error(err: Any, key: str) -> LLMProxyError:
    """Error for an upstream failure: key redacted FIRST, THEN the message cut to 300 chars."""
    if isinstance(err, dict) and isinstance(err.get("message"), str):
        err = err["message"]
    elif not isinstance(err, str):
        err = json.dumps(err)
    return LLMProxyError("LLM proxy upstream error: " + _redact(err, key)[:ERROR_BODY_CHARS])


def _extract_text(payload: Any, key: str = "") -> str:
    """Return ``choices[0].message.content`` (str, or joined ``text`` parts).

    The proxy can report upstream failures as HTTP 200: a top-level ``error``
    or content ``{"error": {"type": "proxy_error", ...}}``. Both raise
    ``LLMProxyError`` (deterministic, never retried); ``key`` is redacted.
    """
    if isinstance(payload, dict) and isinstance(payload.get("error"), (dict, str)):
        raise _upstream_error(payload["error"], key)
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise LLMProxyError("LLM proxy response has no choices[0].message.content") from None
    if isinstance(content, list):
        content = "".join(
            p["text"] for p in content if isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str)
        )
    if not isinstance(content, str) or not content.strip():
        raise LLMProxyError("LLM proxy returned empty content")
    if content.lstrip().startswith("{"):
        try:
            obj = json.loads(content)
        except (ValueError, RecursionError):  # plain text that merely starts with "{"
            obj = None
        err = obj.get("error") if isinstance(obj, dict) else None
        if isinstance(err, dict) and err.get("type") == "proxy_error":
            raise _upstream_error(err, key)
    return content


def _read_capped(resp: Any, limit: int) -> bytes:
    data = resp.read(limit + 1)
    if len(data) > limit:
        raise LLMProxyError(f"LLM proxy response exceeds {limit} bytes")
    return data


def _close(resp: Any) -> None:
    close = getattr(resp, "close", None)
    if callable(close):
        close()


def chat(
    messages: list,
    model: str,
    max_tokens: int = 1024,
    temperature: Optional[float] = None,
    timeout: float = 120,
    retries: int = 3,
    api_key: Optional[str] = None,
    opener: Optional[Callable[..., Any]] = None,
    sleep: Callable[[float], Any] = time.sleep,
    response_format: Optional[dict] = None,
) -> str:
    """POST a non-streaming chat completion and return the assistant text.

    ``opener(req, timeout=...)`` must return an object with ``read(n)`` (and
    optionally ``status``/``close``); default is a no-redirect urllib opener.
    Retries (``retries`` extra attempts) on 429/5xx, URLError and timeouts with
    exponential backoff via ``sleep``. Other 4xx fail at once.
    ``response_format`` (e.g. ``{"type": "json_object"}``) is sent when given.
    ``temperature`` is sent only when not ``None`` (default: provider default).
    An upstream error the proxy wraps in HTTP 200 raises ``LLMProxyError`` at once.
    """
    validate_model(model)
    _check_messages(messages)
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not 0 < timeout <= 600:
        raise LLMProxyError("timeout must be in (0, 600] seconds")
    if not isinstance(retries, int) or isinstance(retries, bool) or not 0 <= retries <= 10:
        raise LLMProxyError("retries must be an integer 0..10")
    if response_format is not None and not isinstance(response_format, dict):
        raise LLMProxyError("response_format must be a dict")
    key = api_key if api_key is not None else load_api_key()
    if not _valid_key(key):
        raise LLMProxyError("API key has invalid characters")

    body: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if temperature is not None:  # claude-sonnet-5/opus-5 etc. reject it, so it is opt-in
        body["temperature"] = temperature
    if response_format is not None:
        body["response_format"] = response_format
    try:
        data = json.dumps(body).encode("utf-8")
    except (TypeError, ValueError):
        raise LLMProxyError("request is not JSON serialisable") from None

    url = _check_url(BASE_URL + CHAT_PATH)
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "User-Agent": "ccn-tilemap/1",
    }
    open_ = opener if opener is not None else _default_opener
    last = "unknown error"
    for attempt in range(retries + 1):
        retry_after = None
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            resp = open_(req, timeout=timeout)
            try:
                status = getattr(resp, "status", 200)
                raw = _read_capped(resp, MAX_RESPONSE_BYTES)
            finally:
                _close(resp)
            if status in RETRY_STATUS:
                last = f"HTTP {status}"
            elif not isinstance(status, int) or not 200 <= status < 300:
                raise LLMProxyError(_redact(f"LLM proxy HTTP {status}", key)[:ERROR_BODY_CHARS])
            else:
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except ValueError:  # includes UnicodeDecodeError
                    raise LLMProxyError("LLM proxy returned invalid JSON") from None
                return _extract_text(payload, key)
        except urllib.error.HTTPError as exc:
            try:
                # redact the whole read body first, THEN truncate (a key straddling the cut would leak a prefix)
                snippet = _redact(exc.read(ERROR_READ_BYTES).decode("utf-8", "replace"), key)[:ERROR_BODY_CHARS]
            except Exception:  # body unreadable: status alone is enough
                snippet = ""
            msg = _redact(f"HTTP {exc.code}: {snippet}".rstrip(": "), key)
            if exc.code not in RETRY_STATUS:
                raise LLMProxyError(f"LLM proxy {msg}") from None
            last = msg
            retry_after = exc.headers
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = "timeout" if isinstance(exc, TimeoutError) else f"network error ({type(exc).__name__})"
        if attempt < retries:
            sleep(_retry_delay(retry_after, attempt))
    raise LLMProxyError(f"LLM proxy request failed after {retries + 1} attempts: {_redact(last, key)}")
