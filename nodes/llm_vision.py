"""CCN_LLMVision: probe/debug node that sends a prompt plus 1..2 images to a vision-capable llm-proxy model.

Spec: architecture-review-loop.md section 8 (node surface). Same key rules as ``CCN_LLMChat``: the API key is never
a widget, input or output; ``llm_client.chat`` resolves and redacts it; every failure is re-raised without exception
chaining and carries no request body (so no base64 image data) and no key. The URL is fixed in ``llm_client``.
Images go out as OpenAI-style ``image_url`` PNG data-URI parts (live-verified for claude-haiku-4-5 / claude-sonnet-5,
see KB note llm-proxy-vision-image-url-works). Encoding and size caps (<=1024 px, <=1 MiB PNG, nearest-neighbour
upscale) live in ``tilemap_core.review.encode_image_part``; the proxy's HTTP-200 ``proxy_error`` bodies are turned
into ``LLMProxyError`` by ``llm_client``. Invalid widget values are rejected, never clamped. No import-time IO,
no ``IS_CHANGED`` (identical inputs reuse the cached reply instead of making another paid call).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from PIL import Image

from ..tilemap_core import llm_client
from ..tilemap_core.llm_client import LLMProxyError
from ..tilemap_core.review import IMAGE_COUNT, IMAGE_MAX_SIDE, ReviewError, encode_image_part
from ._tensor_io import image_list_to_np
from .llm_proxy import (
    TEMP_DEFAULT,
    TEMP_MAX,
    TEMP_MIN,
    TIMEOUT_DEFAULT,
    TIMEOUT_MAX,
    TIMEOUT_MIN,
    TOKENS_MAX,
    TOKENS_MIN,
    _clean_text,
    _InputError,
)

TOKENS_DEFAULT = 512  # vision replies are short probes; the spec default
SIDE_MIN, SIDE_DEFAULT = 16, IMAGE_MAX_SIDE  # max_side upper bound is the encoder's source-side cap


def _int_in(value: Any, lo: int, hi: int, name: str) -> int:
    """Return ``value`` as an int in ``[lo, hi]``; anything else (bool, fraction, NaN, out of range) is rejected."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _InputError(f"{name} must be a whole number")
    if isinstance(value, float) and (not math.isfinite(value) or value != int(value)):
        raise _InputError(f"{name} must be a whole number")
    if not lo <= value <= hi:
        raise _InputError(f"{name} must be {lo}..{hi}, got {int(value)}")
    return int(value)


def _temperature(value: Any) -> Any:
    """Return ``None`` for a negative value ("not sent"), else the float; out of ``[TEMP_MIN, TEMP_MAX]`` is rejected."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise _InputError("temperature must be a number")
    if not TEMP_MIN <= value <= TEMP_MAX:
        raise _InputError(f"temperature must be {TEMP_MIN}..{TEMP_MAX}, got {value}")
    return None if value < 0 else float(value)  # 0.0 is a real value, only negatives mean "not sent"


def _fit(img: np.ndarray, max_side: int) -> np.ndarray:
    """Downscale (LANCZOS, aspect kept) when the long side exceeds ``max_side``; otherwise return ``img`` as is."""
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side <= max_side:
        return img
    scale = max_side / long_side
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return np.asarray(Image.fromarray(img).resize(size, Image.LANCZOS))


def build_messages(system: str, prompt: str, images: list) -> list:
    """Chat messages: optional system text, then ONE user message holding the text part and the image parts."""
    content = [{"type": "text", "text": prompt}] + [encode_image_part(img) for img in images]
    messages = [{"role": "user", "content": content}]
    if system.strip():
        messages.insert(0, {"role": "system", "content": system})
    return messages


class LLMVision:
    """Send a prompt and 1..2 images to the llm-proxy and return the reply text."""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": f"Batch of 1..{IMAGE_COUNT} images; more are rejected, not dropped."}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "model": (list(llm_client.MODELS), {"default": llm_client.MODELS[0]}),
                "max_tokens": (
                    "INT",
                    {"default": TOKENS_DEFAULT, "min": TOKENS_MIN, "max": TOKENS_MAX, "step": 1},
                ),
                "max_side": (
                    "INT",
                    {
                        "default": SIDE_DEFAULT,
                        "min": SIDE_MIN,
                        "max": IMAGE_MAX_SIDE,
                        "step": 1,
                        "tooltip": "Longer images are downscaled to this long side before sending.",
                    },
                ),
            },
            "optional": {
                "system": ("STRING", {"multiline": True, "default": ""}),
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
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "describe"
    CATEGORY = "custom_comfy_nodes/tilemap"
    DESCRIPTION = (
        "Ask a vision model about 1-2 images via the llm-proxy (probe/debug). The API key comes from the "
        "server environment only. temperature: negative = not sent (required for claude-sonnet-5/opus-5)."
    )

    def describe(
        self,
        image: Any,
        prompt: str,
        model: str,
        max_tokens: int,
        max_side: int,
        system: str = "",
        temperature: float = TEMP_DEFAULT,
        timeout_s: int = TIMEOUT_DEFAULT,
    ) -> tuple:
        """Run the vision request and return ``(reply_text,)``.

        Raises ``RuntimeError`` with a user-readable message on any failure; the message never contains the
        API key or image data.
        """
        try:
            prompt_text = _clean_text(prompt, "prompt")
            system_text = _clean_text(system, "system")
            if not prompt_text.strip():
                raise _InputError("prompt is empty")
            side = _int_in(max_side, SIDE_MIN, IMAGE_MAX_SIDE, "max_side")
            tokens = _int_in(max_tokens, TOKENS_MIN, TOKENS_MAX, "max_tokens")
            timeout = _int_in(timeout_s, TIMEOUT_MIN, TIMEOUT_MAX, "timeout_s")
            temp = _temperature(temperature)
            try:
                arrays = image_list_to_np(image, channels=3)  # RGB uint8; alpha of an RGBA input is dropped
            except ValueError as exc:  # messages name shapes/dtypes only
                raise _InputError(str(exc)) from None
            if len(arrays) > IMAGE_COUNT:
                raise _InputError(f"image batch has {len(arrays)} images, at most {IMAGE_COUNT} are allowed")
            messages = build_messages(system_text, prompt_text, [_fit(a, side) for a in arrays])
            reply = llm_client.chat(messages, model, max_tokens=tokens, temperature=temp, timeout=timeout)
        except (LLMProxyError, _InputError, ReviewError) as exc:  # messages are key- and image-data-free by construction
            raise RuntimeError(f"CCN_LLMVision: {exc}") from None
        except Exception as exc:  # unexpected: expose only the type, never exception text
            raise RuntimeError(f"CCN_LLMVision: unexpected {type(exc).__name__} during vision request") from None
        return (reply,)


NODE_CLASS_MAPPINGS = {"CCN_LLMVision": LLMVision}
NODE_DISPLAY_NAME_MAPPINGS = {"CCN_LLMVision": "LLM Vision (llm-proxy)"}
