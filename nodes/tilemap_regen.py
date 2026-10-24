"""In-node Flux sampling for the tilemap review loop (``render_jobs``).

The review loop (``nodes/tilemap_review_loop.py``) regenerates defective tiles inside the node instead of
queueing a new graph. This module is the sampler bridge: for each job it runs exactly the chain of the
shipped ``tilemap_generator.api.json`` first pass (CLIPTextEncode -> FluxGuidance -> EmptySD3LatentImage ->
KSampler -> VAEDecode, negative = encoded text without guidance), so for the same seed/prompt/size the
pixels equal the graph path (LIVE_PROBE P3).

Everything ComfyUI-facing sits behind :class:`ComfyAdapter`, whose methods import ``nodes``,
``node_helpers``, ``comfy.*`` and ``torch`` lazily; importing this module pulls neither. Tests (and the
loop harness) pass their own adapter to ``render_jobs``.

Jobs are the dicts produced by ``tilemap_core.expand`` / ``tilemap_core.regen_plan`` (``prompt``, ``seed``,
``width``, ``height``). Sampling is one job at a time, in list order; ``check_interrupt`` runs before each
job, so a UI interrupt stops the loop within one Flux job and nothing partial is returned (the exception
propagates). No clamping anywhere: bad input raises ``ValueError``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional

import numpy as np

SEED_LIMIT = 2**64  # same range as tilemap_core.expand / regen_plan seeds
STEPS_MAX = 10_000  # KSampler widget limit
CFG_MAX = 100.0  # KSampler / FluxGuidance widget limit
SIDE_RANGE = (16, 4096)  # job width/height; EmptySD3LatentImage needs a multiple of 16
_LATENT_CHANNELS = 16  # Flux / SD3 latent space
_LATENT_DOWNSCALE = 8
_PARAM_KEYS = ("steps", "cfg", "sampler", "scheduler", "guidance")


def _int_in(value: Any, name: str, lo: int, hi: int) -> int:
    """Strict integer: a real ``int`` (no bool/float) inside ``lo..hi``."""
    if type(value) is not int or not lo <= value <= hi:
        raise ValueError(f"{name} must be an integer from {lo} to {hi}, got {value!r}")
    return value


def _num_in(value: Any, name: str, lo: float, hi: float) -> float:
    """Strict number: int or float (no bool), finite, inside ``lo..hi``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number from {lo} to {hi}, got {value!r}")
    if not lo <= value <= hi:  # NaN fails every comparison, inf fails the bound
        raise ValueError(f"{name} must be a number from {lo} to {hi}, got {value!r}")
    return float(value)


def _name(value: Any, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a non-empty string, got {value!r}")
    return value


@dataclass(frozen=True)
class RegenParams:
    """Sampler settings; defaults are the shipped graph (steps 20, cfg 1, euler/simple, guidance 3.5)."""

    steps: int = 20
    cfg: float = 1.0
    sampler: str = "euler"
    scheduler: str = "simple"
    guidance: float = 3.5

    def __post_init__(self) -> None:
        _int_in(self.steps, "steps", 1, STEPS_MAX)
        _num_in(self.cfg, "cfg", 0.0, CFG_MAX)
        _name(self.sampler, "sampler")
        _name(self.scheduler, "scheduler")
        _num_in(self.guidance, "guidance", 0.0, CFG_MAX)

    @classmethod
    def from_mapping(cls, values: Mapping) -> "RegenParams":
        """Build from a widget dict; unknown keys are rejected, missing keys take the defaults."""
        extra = sorted(str(k) for k in values if k not in _PARAM_KEYS)
        if extra:
            raise ValueError(f"unknown params {extra}; allowed: {list(_PARAM_KEYS)}")
        return cls(**{k: values[k] for k in _PARAM_KEYS if k in values})


class ComfyAdapter:
    """The ComfyUI calls ``render_jobs`` needs; every import happens inside the method that uses it.

    Mirrors the node chain of the shipped graph; replace it with a fake to run offline.
    """

    def check_interrupt(self) -> None:
        """Raise ComfyUI's ``InterruptProcessingException`` if the user pressed Interrupt."""
        import comfy.model_management as mm

        mm.throw_exception_if_processing_interrupted()

    def encode(self, clip: Any, text: str) -> Any:
        """``CLIPTextEncode.encode`` -> conditioning."""
        import nodes

        return nodes.CLIPTextEncode().encode(clip, text)[0]

    def set_guidance(self, cond: Any, guidance: float) -> Any:
        """``FluxGuidance``: conditioning with the guidance value set."""
        import node_helpers

        return node_helpers.conditioning_set_values(cond, {"guidance": guidance})

    def empty_latent(self, width: int, height: int) -> dict:
        """``EmptySD3LatentImage`` (batch 1): same device, dtype and downscale marker as that node."""
        import comfy.model_management as mm
        import torch

        samples = torch.zeros(
            [1, _LATENT_CHANNELS, height // _LATENT_DOWNSCALE, width // _LATENT_DOWNSCALE],
            device=mm.intermediate_device(),
            dtype=mm.intermediate_dtype(),
        )
        return {"samples": samples, "downscale_ratio_spacial": _LATENT_DOWNSCALE}

    def sample(self, model: Any, job: Mapping, params: RegenParams, positive: Any, negative: Any, latent: dict) -> dict:
        """``KSampler`` (denoise 1.0) via ``nodes.common_ksampler`` -> latent dict."""
        import nodes

        return nodes.common_ksampler(
            model, job["seed"], params.steps, params.cfg, params.sampler, params.scheduler,
            positive, negative, latent, denoise=1.0,
        )[0]

    def decode(self, vae: Any, latent: dict) -> Any:
        """``VAEDecode.decode`` -> IMAGE tensor [1, H, W, 3]."""
        import nodes

        return nodes.VAEDecode().decode(vae, latent)[0]

    def to_rgb(self, image: Any) -> np.ndarray:
        """IMAGE tensor [1, H, W, 3] -> float32 ``HxWx3`` copy, values untouched (quantised later by Assemble)."""
        from ._tensor_io import _tensor_to_np

        arr = _tensor_to_np(image, "decoded image")
        if arr.ndim != 4 or arr.shape[0] != 1 or arr.shape[3] != 3:
            raise ValueError(f"decoded image must have shape [1,H,W,3], got {tuple(arr.shape)}")
        return np.ascontiguousarray(arr[0])


def _check_progress(progress: Any) -> None:
    if progress is not None and not callable(getattr(progress, "update", None)):
        raise ValueError("progress must be None or an object with an update(n) method")


def _check_job(job: Any, i: int) -> None:
    if not isinstance(job, Mapping):
        raise ValueError(f"jobs[{i}] must be a job mapping, got {type(job).__name__}")
    _name(job.get("prompt"), f"jobs[{i}].prompt")
    _int_in(job.get("seed"), f"jobs[{i}].seed", 0, SEED_LIMIT - 1)
    for side in ("width", "height"):
        value = _int_in(job.get(side), f"jobs[{i}].{side}", *SIDE_RANGE)
        if value % 16:
            raise ValueError(f"jobs[{i}].{side} must be a multiple of 16, got {value}")


def _check_image(img: Any, job: Mapping, i: int) -> np.ndarray:
    want = (job["height"], job["width"], 3)
    if not isinstance(img, np.ndarray) or img.shape != want or img.dtype != np.float32:
        got = tuple(img.shape) if isinstance(img, np.ndarray) else type(img).__name__
        raise ValueError(f"jobs[{i}] decoded image has {got}, expected float32 shape {want}")
    return img


def render_jobs(
    model: Any,
    clip: Any,
    vae: Any,
    jobs: Any,
    params: Any,
    negative_text: str,
    progress: Optional[Any] = None,
    adapter: Optional[ComfyAdapter] = None,
) -> List[np.ndarray]:
    """Sample every job with Flux in list order and return one RGB image per job.

    Args:
        model, clip, vae: the live MODEL / CLIP / VAE objects (LoRAs already patched in upstream).
        jobs: list/tuple of job dicts; each needs ``prompt`` (non-empty str), ``seed`` (int
            0..2**64-1), ``width`` and ``height`` (int multiples of 16, 16..4096). Other keys ignored.
            Not mutated. An empty list returns ``[]`` without touching ComfyUI.
        params: :class:`RegenParams` or a mapping of its fields (unknown keys rejected).
        negative_text: text for the negative conditioning (str, may be empty); encoded once, without
            guidance, exactly like the shipped graph. Inert at cfg 1 (Flux), kept for cfg > 1.
        progress: ``None`` or an object with ``update(n)`` (``comfy.utils.ProgressBar``); called with
            ``1`` after each finished job.
        adapter: ComfyUI bridge; defaults to :class:`ComfyAdapter` (lazy ComfyUI imports).

    Returns:
        ``list`` aligned with ``jobs``: C-contiguous float32 ``height x width x 3`` arrays, the VAE output
        as is (nominal 0..1, not clipped; ``_tensor_io.image_list_to_np`` / ``np.clip`` quantise later).

    Raises:
        ValueError: on any invalid argument (checked for all jobs before the first sample) or a decoded
            image whose shape/dtype differs from the job size / float32.
        Exception: anything the adapter raises, unchanged; notably ComfyUI's interrupt exception, which
            aborts the whole call (no partial list).
    """
    if not isinstance(jobs, (list, tuple)):
        raise ValueError(f"jobs must be a list of job mappings, got {type(jobs).__name__}")
    if isinstance(params, Mapping):
        params = RegenParams.from_mapping(params)
    elif not isinstance(params, RegenParams):
        raise ValueError(f"params must be RegenParams or a mapping, got {type(params).__name__}")
    if type(negative_text) is not str:
        raise ValueError(f"negative_text must be a string, got {type(negative_text).__name__}")
    _check_progress(progress)
    for i, job in enumerate(jobs):
        _check_job(job, i)
    if not jobs:
        return []
    ad = ComfyAdapter() if adapter is None else adapter
    negative = ad.encode(clip, negative_text)
    out: List[np.ndarray] = []
    for i, job in enumerate(jobs):
        ad.check_interrupt()
        positive = ad.set_guidance(ad.encode(clip, job["prompt"]), params.guidance)
        latent = ad.empty_latent(job["width"], job["height"])
        sampled = ad.sample(model, job, params, positive, negative, latent)
        out.append(_check_image(ad.to_rgb(ad.decode(vae, sampled)), job, i))
        if progress is not None:
            progress.update(1)
    return out
