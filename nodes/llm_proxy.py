"""CCN_LLMChat: generic single-turn chat node backed by the llm-proxy.

Security: the API key is NEVER a widget, input or output. ``llm_client.chat``
resolves it from ``LLM_PROXY_API_KEY`` / ``/cache/llm_proxy.json`` itself and
redacts it from its own errors; this node additionally re-raises every failure
without exception chaining, so no traceback can carry request internals.
The URL is fixed in ``llm_client`` (no URL input => no SSRF). No import-time IO
or network. No ``IS_CHANGED``: ComfyUI's default caching applies, so identical
inputs reuse the previous reply instead of making another paid call; edit any
input to ask again.
"""

from __future__ import annotations

import math
import unicodedata
from typing import Any

from ..tilemap_core import llm_client
from ..tilemap_core.llm_client import LLMProxyError

MAX_TEXT_CHARS = 60_000  # per text input; longer input is rejected, never silently cut
TOKENS_MIN, TOKENS_MAX, TOKENS_DEFAULT = 1, 32_000, 1024
# Negative temperature = "not sent" (provider default). claude-sonnet-5/opus-5 reject the parameter.
TEMP_MIN, TEMP_MAX, TEMP_DEFAULT = -1.0, 2.0, -1.0
TIMEOUT_MIN, TIMEOUT_MAX, TIMEOUT_DEFAULT = 5, 600, 120
JSON_FORMAT = {"type": "json_object"}


class _InputError(ValueError):
    """Rejected widget value. Only this class (plus LLMProxyError) has its text shown to the user."""


def _clean_text(value: Any, name: str) -> str:
    """Return ``value`` as text without control/format characters (newline and tab kept).

    Unlike ``tilemap_core.sanitize.clean_text`` this keeps braces, brackets and
    line breaks, which a free-form chat prompt needs. Raises ``_InputError`` if
    ``value`` is not a string or exceeds ``MAX_TEXT_CHARS`` after cleaning.
    """
    if not isinstance(value, str):
        raise _InputError(f"{name} must be text, got {type(value).__name__}")
    text = unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    if len(text) > MAX_TEXT_CHARS:
        raise _InputError(f"{name} is too long ({len(text)} chars, max {MAX_TEXT_CHARS})")
    return text


def _clamp_int(value: Any, lo: int, hi: int, default: int, name: str) -> int:
    """Coerce to int and clamp to ``[lo, hi]``; NaN/inf fall back to ``default``."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        raise _InputError(f"{name} must be a number") from None
    if not math.isfinite(num):
        return default
    return max(lo, min(hi, int(num)))


def _clamp_float(value: Any, lo: float, hi: float, default: float, name: str) -> float:
    """Coerce to float and clamp to ``[lo, hi]``; NaN/inf fall back to ``default``."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        raise _InputError(f"{name} must be a number") from None
    if not math.isfinite(num):
        return default
    return max(lo, min(hi, num))


class LLMChat:
    """Send one system + user prompt to the llm-proxy and return the reply text."""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "system": ("STRING", {"multiline": True, "default": ""}),
                "user": ("STRING", {"multiline": True, "default": ""}),
                "model": (list(llm_client.MODELS), {"default": llm_client.MODELS[0]}),
                "model_override": ("STRING", {"default": ""}),
                "max_tokens": (
                    "INT",
                    {"default": TOKENS_DEFAULT, "min": TOKENS_MIN, "max": TOKENS_MAX, "step": 1},
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": TEMP_DEFAULT,
                        "min": TEMP_MIN,
                        "max": TEMP_MAX,
                        "step": 0.05,
                        "tooltip": "Negative = not sent (provider default; required for claude-sonnet-5/opus-5).",
                    },
                ),
                "timeout_s": (
                    "INT",
                    {"default": TIMEOUT_DEFAULT, "min": TIMEOUT_MIN, "max": TIMEOUT_MAX, "step": 1},
                ),
            },
            "optional": {
                "json_mode": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "chat"
    CATEGORY = "custom_comfy_nodes/llm"
    DESCRIPTION = (
        "Single-turn chat with the llm-proxy. The API key comes from the server environment only. "
        "temperature: negative = not sent (provider default; required for claude-sonnet-5/opus-5)."
    )

    def chat(
        self,
        system: str,
        user: str,
        model: str,
        model_override: str,
        max_tokens: int,
        temperature: float,
        timeout_s: int,
        json_mode: bool = False,
    ) -> tuple:
        """Run the chat call and return ``(reply_text,)``.

        Raises ``RuntimeError`` with a user-readable message on any failure; the
        message never contains the API key.
        """
        try:
            user_text = _clean_text(user, "user")
            system_text = _clean_text(system, "system")
            if not user_text.strip():
                raise _InputError("user prompt is empty")
            override = model_override.strip() if isinstance(model_override, str) else ""
            chosen = override or model  # llm_client.chat validates the id
            messages = [{"role": "user", "content": user_text}]
            if system_text.strip():
                messages.insert(0, {"role": "system", "content": system_text})
            temp_value = _clamp_float(temperature, TEMP_MIN, TEMP_MAX, TEMP_DEFAULT, "temperature")
            temp = None if temp_value < 0 else temp_value  # 0.0 is a real value, only negatives mean "not sent"
            reply = llm_client.chat(
                messages,
                chosen,
                max_tokens=_clamp_int(max_tokens, TOKENS_MIN, TOKENS_MAX, TOKENS_DEFAULT, "max_tokens"),
                temperature=temp,
                timeout=_clamp_int(timeout_s, TIMEOUT_MIN, TIMEOUT_MAX, TIMEOUT_DEFAULT, "timeout_s"),
                response_format=dict(JSON_FORMAT) if json_mode is True else None,
            )
        except (LLMProxyError, _InputError) as exc:  # messages are key-free by construction
            raise RuntimeError(f"CCN_LLMChat: {exc}") from None
        except Exception as exc:  # unexpected: expose only the type, never exception text
            raise RuntimeError(f"CCN_LLMChat: unexpected {type(exc).__name__} during chat request") from None
        return (reply,)


NODE_CLASS_MAPPINGS = {"CCN_LLMChat": LLMChat}
NODE_DISPLAY_NAME_MAPPINGS = {"CCN_LLMChat": "LLM Chat (llm-proxy)"}
