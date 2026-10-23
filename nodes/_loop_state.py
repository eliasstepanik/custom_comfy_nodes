"""State helpers for ``CCN_TilemapReviewLoop``: widget validators, ``job_json`` parsing, LLM call budget, log builders.

Everything here is pure Python plus numpy: no torch, no ComfyUI, no network. Validation is strict (``ValueError``,
nothing is clamped). The loop itself lives in ``tilemap_review_loop.py``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from types import SimpleNamespace

import numpy as np

from ..tilemap_core import expand as tile_expand
from ..tilemap_core import llm_client, plan_prompt, plan_schema, regen_plan, review, sanitize
from .tilemap_regen import RegenParams

LOG_FORMAT = "ccn-review-log/1"
REVIEW_MODES = ("vision+heuristic", "heuristic", "vision")
LAYOUT_SOURCES = ("auto", "llm")
ON_ERROR = ("keep", "strict")
JOB_JSON_MAX = 400_000
LLM_TIMEOUT_S = 120
LAYOUT_TOKENS = 8000
_KEYS = ("tile_id", "category", "variant", "footprint")
_FIX = "reconnect job_json from CCN_TilemapExpand"
_ENTRY_ISSUE = ("tile", "kind", "severity", "source")


def one(value: object, name: str) -> object:
    """INPUT_IS_LIST delivers every input as a list; a scalar widget is element 0 (any other length is an error)."""
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"{name} must be a single value, got {len(value)}")
        return value[0]
    return value


def int_in(value: object, name: str, lo: int, hi: int) -> int:
    """Exact ``int`` (no bool, no float) within ``lo..hi`` inclusive."""
    if type(value) is not int or not lo <= value <= hi:
        raise ValueError(f"{name} must be an integer from {lo} to {hi}")
    return value


def float_in(value: object, name: str, lo: float, hi: float) -> float:
    """Finite int/float (no bool) within ``lo..hi`` inclusive, returned as ``float``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lo <= value <= hi:
        raise ValueError(f"{name} must be a number from {lo} to {hi}")
    return float(value)


def choice(value: object, name: str, options: tuple) -> str:
    """Exact ``str`` that is one of ``options``."""
    if type(value) is not str or value not in options:
        raise ValueError(f"{name} must be one of {list(options)}")
    return value


def job_key(job: Mapping) -> str:
    """Tile key ``'<tile_id>#<variant>'`` of a job."""
    return f"{job['tile_id']}#{job['variant']}"


def quantize(img: np.ndarray) -> np.ndarray:
    """Float RGB (nominal 0..1) -> contiguous uint8, rounding half up like ``_tensor_io``; NaN becomes 0."""
    return np.ascontiguousarray((np.clip(np.nan_to_num(img), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))


def reject_base_inputs(kw: Mapping) -> None:
    """``base_atlas`` / ``base_alpha`` (tileset reuse) have no consumer in this node: connecting one is an error."""
    if kw.get("base_atlas") is not None or kw.get("base_alpha") is not None:
        raise ValueError("base_atlas / base_alpha (tileset reuse) are not supported by this node yet")


def parse_options(kw: Mapping) -> SimpleNamespace:
    """Validate every widget (each value unwrapped with ``one``) and return a namespace.

    Fields: ``review_mode, review_model, max_rounds, max_regen, max_attempts, pass_score, layout_source,
    on_review_error, transitions, negative, snap`` (``palette_size, seam_blend, key_tolerance, transitions``) and
    ``params`` (``RegenParams``). Ranges: ``max_rounds`` 0..6, ``max_regen_per_round`` 1..32, ``max_attempts`` 1..5.
    """
    def get(name: str) -> object:
        return one(kw[name], name)

    transitions, negative = get("transitions"), get("negative")
    if not isinstance(transitions, bool):
        raise ValueError("transitions must be a boolean")
    if not isinstance(negative, str):
        raise ValueError("negative must be text")
    snap_p = {"palette_size": int_in(get("palette_size"), "palette_size", 4, 64),
              "seam_blend": float_in(get("seam_blend"), "seam_blend", 0.0, 0.5),
              "key_tolerance": int_in(get("key_tolerance"), "key_tolerance", 0, 255), "transitions": transitions}
    return SimpleNamespace(
        review_mode=choice(get("review_mode"), "review_mode", REVIEW_MODES),
        review_model=choice(get("review_model"), "review_model", tuple(llm_client.MODELS)),
        max_rounds=int_in(get("max_rounds"), "max_rounds", 0, regen_plan.MAX_ROUND),
        max_regen=int_in(get("max_regen_per_round"), "max_regen_per_round", *regen_plan.MAX_REGEN_RANGE),
        max_attempts=int_in(get("max_attempts"), "max_attempts", *regen_plan.MAX_ATTEMPTS_RANGE),
        pass_score=int_in(get("pass_score"), "pass_score", review.SCORE_MIN, review.SCORE_MAX),
        layout_source=choice(get("layout_source"), "layout_source", LAYOUT_SOURCES),
        on_review_error=choice(get("on_review_error"), "on_review_error", ON_ERROR),
        transitions=transitions, negative=negative, snap=snap_p,
        params=RegenParams(steps=get("steps"), cfg=get("cfg"), sampler=get("sampler"), scheduler=get("scheduler"),
                           guidance=get("guidance")))


def parse_job_json(text: object) -> tuple:
    """Untrusted ``job_json`` -> ``(data, tile_size, view, plan, jobs, seed)``; jobs must match the plan.

    Rejects a non-null ``base`` (tileset reuse), a bad ``tile_size``/``view``, an invalid plan, jobs that differ from
    ``expand_jobs`` (by tile_id/category/variant/footprint) and a job seed outside ``0..2**64-1``.
    """
    if not isinstance(text, str) or len(text) > JOB_JSON_MAX:
        raise ValueError(f"job_json must be text of at most {JOB_JSON_MAX} characters; {_FIX}")
    try:
        data = json.loads(text)
    except (ValueError, RecursionError):
        raise ValueError(f"job_json is not valid JSON; {_FIX}") from None
    if not isinstance(data, dict):
        raise ValueError(f"job_json must be a JSON object; {_FIX}")
    if data.get("base") is not None:
        raise ValueError("job_json.base (tileset reuse) is not supported by this node yet")
    size, view = data.get("tile_size"), data.get("view")
    if type(size) is not int or size not in tile_expand.TILE_SIZES or not isinstance(view, str) or view not in plan_prompt.VIEWS:
        raise ValueError(f"job_json needs a valid tile_size and view; {_FIX}")
    try:
        plan = plan_schema.validate_plan(data.get("plan"), plan_schema.MAX_JOBS)
    except plan_schema.PlanError:
        raise ValueError(f"job_json plan failed validation; {_FIX}") from None

    def sig(rows: list) -> str:
        return json.dumps([[j.get(k) if isinstance(j, dict) else None for k in _KEYS] for j in rows])

    jobs = data.get("jobs")
    if not isinstance(jobs, list) or sig(jobs) != sig(tile_expand.expand_jobs(plan, size, view, "", 0)):
        raise ValueError(f"job_json jobs do not match its plan; {_FIX}")
    seed = jobs[0].get("seed")
    if type(seed) is not int or not 0 <= seed < regen_plan.SEED_LIMIT:
        raise ValueError(f"job_json job seed must be an integer from 0 to {regen_plan.SEED_LIMIT - 1}; {_FIX}")
    return data, size, view, plan, jobs, seed


class LlmBudget:
    """Counts LLM proxy calls through ``chat`` and refuses past ``budget`` (the loop uses ``2 * max_rounds``)."""

    def __init__(self, chat, model: str, budget: int) -> None:
        self.chat, self.model, self.budget, self.calls = chat, model, budget, 0

    def ask(self, messages: list, max_tokens: int = review.MAX_TOKENS) -> str:
        """One chat call (temperature omitted); raises ``LLMProxyError`` without calling when the budget is used up."""
        if self.calls >= self.budget:
            raise llm_client.LLMProxyError("LLM call budget used up")
        self.calls += 1
        return self.chat(messages, self.model, max_tokens=max_tokens, timeout=LLM_TIMEOUT_S)


def is_passed(verdict: Mapping | None, issues: list, pass_score: int) -> bool:
    """Round passes: with a verdict it must pass, reach ``pass_score`` and no issue may be high; without one, no issues."""
    if verdict is None:
        return not issues
    return bool(verdict["pass"]) and verdict["score"] >= pass_score and not any(i["severity"] == "high" for i in issues)


def round_entry(rnd: int, info: Mapping, issues: list, verdict: Mapping | None, reverted: list) -> dict:
    """One ``rounds`` log entry: counters and verdict text only (``regen`` / ``skipped`` start empty for the loop to fill)."""
    entry = dict(info, round=rnd, reverted=sorted(reverted), regen=[], skipped=[],
                 issues=[{k: i[k] for k in _ENTRY_ISSUE} for i in issues])
    if verdict:
        entry.update(score=verdict["score"], passed=verdict["pass"], summary=verdict["summary"])
    return entry


def build_log(stop: str, llm_calls: int, layout_source: str, layout_problem: str, unreviewed: list, rounds: list) -> str:
    """``review_log`` JSON; never contains images, request bodies or keys."""
    return json.dumps({"format": LOG_FORMAT, "stop": stop, "llm_calls": llm_calls, "layout_source": layout_source,
                       "layout_problem": sanitize.clean_text(layout_problem, 200), "unreviewed": sorted(unreviewed),
                       "rounds": rounds}, sort_keys=True, ensure_ascii=False)


def build_job_json(text: str, data: Mapping, snap_p: Mapping, rounds: int, attempts: Mapping, stop: str,
                   max_rounds: int) -> str:
    """Output ``job_json``: ``text`` unchanged for ``max_rounds == 0`` (pass-through), else ``data`` + snap + review."""
    if max_rounds == 0:
        return text
    return json.dumps(dict(data, snap=dict(snap_p), review={"rounds": rounds, "attempts": dict(attempts), "stop": stop}),
                      sort_keys=True, separators=(",", ":"), ensure_ascii=False)
