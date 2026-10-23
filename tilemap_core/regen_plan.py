"""Regeneration planning for the review loop: merged review issues -> a bounded list of Flux jobs.

Pure stdlib, deterministic, no I/O, no randomness; every bad input raises ``ValueError`` (nothing is clamped).
Implements section 6 of the review-loop architecture:

1. every issue resolves to job keys ``"<tile_id>#<variant>"``: ``"<id>#<v>"`` is that key, a bare ``"<id>"``
   is every variant of the id, a Wang reference (``"<wang_id>/<mask>"``, ``"<wang_id>#<mask>"`` or the bare
   ``"<wang_id>"``; needs ``transitions``) is both parent ground tiles ``#0``;
2. issues are grouped per key; a key's weight is the sum of its distinct kinds' severity weights
   (``SEVERITY_WEIGHT``: low 1, medium 2, high 4; a duplicate ``(key, kind)`` counts once, at its highest
   severity). A ``transition`` issue is blamed on the ground tiles as ``ground_not_texture`` +
   ``not_tileable`` (on a non-ground key: ``other``);
3. keys with ``attempts[key] >= max_attempts`` go to ``skipped`` (the loop keeps their best version);
4. the rest are sorted by weight (descending) then key, and cut to ``max_regen``;
5. each survivor becomes a copy of its **base** job (pass the first-pass job, never an amended one, so
   amendments do not pile up) with the ``defects.amend_job`` prompt tail (kinds + the hint of its most severe
   issue), ``seed = (base seed + attempt_no * SEED_STEP) % 2**64`` and ``attempt`` / ``round`` / ``defects``.

``attempt_no = attempts.get(key, 0) + 1``: attempt 0 is the first pass, so ``max_attempts`` is the number of
regenerations a key may get. ``snap_for_issues`` applies the atlas-wide snap overrides of
``defects.snap_overrides``; ``record_attempts`` and ``best_version`` are the loop's bookkeeping.
"""

from typing import Any, Iterable, Mapping, Optional, Sequence

from . import snap as snap_mod
from .defects import DEFECT_KINDS, amend_job, snap_overrides
from .plan_schema import CATEGORIES
from .review import SEVERITIES

__all__ = [
    "SEVERITY_WEIGHT", "SEED_STEP", "SEED_LIMIT", "MAX_REGEN_RANGE", "MAX_ATTEMPTS_RANGE", "MAX_ROUND",
    "key_weights", "plan_regen", "record_attempts", "best_version", "snap_for_issues",
]

SEVERITY_WEIGHT = {"low": 1, "medium": 2, "high": 4}
SEED_STEP = 10007  # seed shift per attempt (prime, so attempts of neighbouring jobs rarely collide)
SEED_LIMIT = 2**64
MAX_REGEN_RANGE = (1, 32)  # ``max_regen_per_round`` widget range
MAX_ATTEMPTS_RANGE = (1, 5)  # ``max_attempts`` widget range
MAX_ROUND = 6  # ``max_rounds`` upper bound
_WANG_MASKS = 16
_TRANSITION_KINDS = ("ground_not_texture", "not_tileable")
assert tuple(SEVERITY_WEIGHT) == SEVERITIES  # one severity vocabulary with review.py


def _int_in(value: Any, name: str, lo: int, hi: int) -> int:
    if type(value) is not int or not lo <= value <= hi:
        raise ValueError(f"{name} must be an integer from {lo} to {hi}, got {value!r}")
    return value


def _check_jobs(jobs_by_key: Any) -> dict:
    if not isinstance(jobs_by_key, Mapping) or not jobs_by_key:
        raise ValueError("jobs_by_key must be a non-empty mapping of '<tile_id>#<variant>' to job")
    for key, job in jobs_by_key.items():
        if not isinstance(job, Mapping):
            raise ValueError(f"jobs_by_key[{key!r}] must be a job mapping")
        tile_id, variant = job.get("tile_id"), job.get("variant")
        if type(tile_id) is not str or type(variant) is not int or variant < 0:
            raise ValueError(f"jobs_by_key[{key!r}] needs a str tile_id and an int variant >= 0")
        if key != f"{tile_id}#{variant}":
            raise ValueError(f"jobs_by_key key {key!r} does not match its job {tile_id}#{variant}")
        if job.get("category") not in CATEGORIES:
            raise ValueError(f"jobs_by_key[{key!r}].category must be one of {list(CATEGORIES)}")
        if type(job.get("prompt")) is not str:
            raise ValueError(f"jobs_by_key[{key!r}].prompt must be a string")
        _int_in(job.get("seed"), f"jobs_by_key[{key!r}].seed", 0, SEED_LIMIT - 1)
    return dict(jobs_by_key)


def _check_attempts(attempts: Any) -> dict:
    if not isinstance(attempts, Mapping):
        raise ValueError("attempts must be a mapping of job key to attempts used so far")
    for key, count in attempts.items():
        if type(key) is not str:
            raise ValueError(f"attempts key {key!r} must be a string")
        _int_in(count, f"attempts[{key!r}]", 0, 2**31 - 1)
    return dict(attempts)


def _check_issues(issues: Any) -> list:
    if not isinstance(issues, (list, tuple)):
        raise ValueError("issues must be a list of issue dicts")
    for n, item in enumerate(issues):
        if not isinstance(item, Mapping):
            raise ValueError(f"issues[{n}] must be an issue dict")
        tile = item.get("tile")
        if type(tile) is not str or not tile.strip():
            raise ValueError(f"issues[{n}].tile must be a non-empty string")
        if item.get("kind") not in DEFECT_KINDS:
            raise ValueError(f"issues[{n}].kind must be one of {list(DEFECT_KINDS)}, got {item.get('kind')!r}")
        if item.get("severity") not in SEVERITY_WEIGHT:
            raise ValueError(f"issues[{n}].severity must be one of {list(SEVERITIES)}, got {item.get('severity')!r}")
        if type(item.get("fix_hint", "")) is not str:
            raise ValueError(f"issues[{n}].fix_hint must be a string")
    return list(issues)


def _wang_map(transitions: Any) -> dict:
    """``{wang_id: (a, b)}`` from the plan's transition pairs (``snap.wang_items`` names a set ``a_b``)."""
    if isinstance(transitions, (str, bytes)) or not isinstance(transitions, Iterable):
        raise ValueError("transitions must be a list of [a, b] id pairs")
    out: dict = {}
    for pair in transitions:
        if (isinstance(pair, (str, bytes)) or not isinstance(pair, Sequence) or len(pair) != 2
                or not all(type(p) is str and p for p in pair)):
            raise ValueError(f"transition {pair!r} must be a pair of non-empty id strings")
        wid = f"{pair[0]}_{pair[1]}"
        if wid in out:
            raise ValueError(f"transition {wid!r} is listed twice")
        out[wid] = (pair[0], pair[1])
    return out


def _resolve(tile: str, jobs: dict, wang: dict) -> list:
    """Job keys an issue's tile reference stands for; unknown references raise (callers validate refs)."""
    ref = tile.strip()
    if "/" in ref:
        wid, _, mask = ref.partition("/")
        if not (mask.isascii() and mask.isdigit() and int(mask) < _WANG_MASKS):
            raise ValueError(f"tile reference {ref!r}: Wang mask must be 0..{_WANG_MASKS - 1}")
    elif ref in jobs:
        return [ref]
    else:
        wid = ref.partition("#")[0]
        if "#" not in ref:
            same = sorted((j["variant"], k) for k, j in jobs.items() if j["tile_id"] == ref)
            if same:
                return [k for _, k in same]
    if wid not in wang:
        raise ValueError(f"tile reference {ref!r} matches no job" + ("" if wang else " (no transitions given)"))
    parents = [f"{p}#0" for p in dict.fromkeys(wang[wid])]
    missing = [p for p in parents if p not in jobs]
    if missing:
        raise ValueError(f"tile reference {ref!r}: parent jobs {missing} are not in jobs_by_key")
    return parents


def _collect(issues: Any, jobs: dict, transitions: Any) -> dict:
    """``{key: {"kinds": {kind: weight}, "hints": [(-weight, kind rank, hint)]}}`` for every blamed key."""
    wang = _wang_map(transitions)
    found: dict = {}
    for item in _check_issues(issues):
        weight = SEVERITY_WEIGHT[item["severity"]]
        hint = item.get("fix_hint", "").strip()
        for key in _resolve(item["tile"], jobs, wang):
            if item["kind"] != "transition":
                kinds = (item["kind"],)
            else:
                kinds = _TRANSITION_KINDS if jobs[key]["category"] == "ground" else ("other",)
            entry = found.setdefault(key, {"kinds": {}, "hints": []})
            for kind in kinds:
                entry["kinds"][kind] = max(entry["kinds"].get(kind, 0), weight)
            if hint:
                entry["hints"].append((-weight, DEFECT_KINDS.index(kinds[0]), hint))
    return found


def key_weights(issues: Sequence[Mapping], jobs_by_key: Mapping, transitions: Iterable = ()) -> dict:
    """Return ``{job key: weight}`` of the blamed keys (sorted by weight descending, then key).

    Weight = sum over the key's distinct kinds of the highest severity weight (low 1, medium 2, high 4).
    A key without issues is absent, i.e. weight 0. Raises ``ValueError`` on malformed issues or on a tile
    reference that resolves to no job.
    """
    found = _collect(issues, _check_jobs(jobs_by_key), transitions)
    weights = {k: sum(v["kinds"].values()) for k, v in found.items()}
    return {k: weights[k] for k in sorted(weights, key=lambda k: (-weights[k], k))}


def plan_regen(issues: Sequence[Mapping], jobs_by_key: Mapping, attempts: Mapping, max_regen: int,
               max_attempts: int, round_no: int, *, transitions: Iterable = ()) -> tuple:
    """Return ``(new_jobs, skipped)``: at most ``max_regen`` regeneration jobs and the exhausted keys.

    ``issues``: merged review issues (``review.merge_issues`` shape; only ``tile``, ``kind``, ``severity``
    and ``fix_hint`` are read). ``jobs_by_key``: the BASE job per key. ``attempts``: regenerations already
    used per key (missing = 0). ``max_regen`` 1..32, ``max_attempts`` 1..5, ``round_no`` 1..6.
    ``transitions``: the plan's ``[a, b]`` pairs, needed only to resolve Wang references.

    ``new_jobs`` is ordered by weight (descending) then key, holds each key at most once, and each job is
    ``amend_job(base, kinds, hint)`` plus the new ``seed`` and ``attempt`` (= used + 1), ``round`` and
    ``defects`` (kinds, canonical order). ``skipped`` lists keys with ``attempts >= max_attempts`` in the same
    order (the loop keeps their best version). Keys cut by ``max_regen`` are in neither list: they stay
    eligible next round. Inputs are never modified. Raises ``ValueError`` on any bad input.
    """
    _int_in(max_regen, "max_regen", *MAX_REGEN_RANGE)
    _int_in(max_attempts, "max_attempts", *MAX_ATTEMPTS_RANGE)
    _int_in(round_no, "round_no", 1, MAX_ROUND)
    jobs = _check_jobs(jobs_by_key)
    used = _check_attempts(attempts)
    found = _collect(issues, jobs, transitions)
    weight = {k: sum(v["kinds"].values()) for k, v in found.items()}
    order = sorted(weight, key=lambda k: (-weight[k], k))
    skipped = [k for k in order if used.get(k, 0) >= max_attempts]
    new_jobs = []
    for key in [k for k in order if k not in skipped][:max_regen]:
        attempt = used.get(key, 0) + 1
        kinds = [k for k in DEFECT_KINDS if k in found[key]["kinds"]]
        hint = min(found[key]["hints"])[2] if found[key]["hints"] else ""
        job = amend_job(jobs[key], kinds, hint)
        job["seed"] = (jobs[key]["seed"] + attempt * SEED_STEP) % SEED_LIMIT
        job.update(attempt=attempt, round=round_no, defects=kinds)
        new_jobs.append(job)
    return new_jobs, skipped


def record_attempts(attempts: Mapping, new_jobs: Sequence[Mapping]) -> dict:
    """Return a copy of ``attempts`` with every job of ``new_jobs`` counted (``attempts[key] = job['attempt']``).

    Raises ``ValueError`` when a job's ``attempt`` is not ``attempts.get(key, 0) + 1`` (a plan counted twice,
    planned against stale counters, or a key listed twice).
    """
    out = _check_attempts(attempts)
    for job in new_jobs:
        if not isinstance(job, Mapping) or type(job.get("tile_id")) is not str or type(job.get("variant")) is not int:
            raise ValueError("new_jobs must be job mappings with tile_id and variant")
        key = f"{job['tile_id']}#{job['variant']}"
        if job.get("attempt") != out.get(key, 0) + 1 or type(job.get("attempt")) is not int:
            raise ValueError(f"job {key!r} has attempt {job.get('attempt')!r}, expected {out.get(key, 0) + 1}")
        out[key] = job["attempt"]
    return out


def _version(value: Any, name: str) -> tuple:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{name} must be an (image, weight) tuple")
    _int_in(value[1], f"{name} weight", 0, 2**31 - 1)
    return value


def best_version(best: Optional[tuple], candidate: tuple) -> tuple:
    """Keep-best bookkeeping: return ``(winner, replaced)`` for ``(image, weight)`` tuples.

    ``best`` is the version kept so far (``None`` = none yet), ``candidate`` the regenerated tile with the
    weight its next review gave (0 when no issue names it). The candidate wins only when its weight is
    strictly lower; a tie keeps ``best``. Images are opaque and never inspected. A tile that is not reviewed
    again after its last regeneration is the caller's call (it keeps the regenerated version, logged as
    ``unreviewed``).
    """
    cand = _version(candidate, "candidate")
    if best is None:
        return cand, True
    kept = _version(best, "best")
    return (cand, True) if cand[1] < kept[1] else (kept, False)


def snap_for_issues(issues: Sequence[Mapping], snap_params: Optional[Mapping] = None) -> dict:
    """Return the full, validated ``snap`` block with the atlas-wide overrides of the issues' kinds applied.

    Overrides come from ``defects.snap_overrides`` over the distinct kinds of ``issues`` (today only
    ``fringe`` raises ``key_tolerance``). A number is only ever raised, never lowered below the current
    value; other values are replaced. No issues -> the validated ``snap_params`` unchanged. Raises
    ``ValueError`` on malformed issues or ``snap_params`` (see ``snap.check_snap_params``).
    """
    current = snap_mod.check_snap_params(snap_params)
    kinds = {i["kind"] for i in _check_issues(issues)}
    if not kinds:
        return current
    for name, value in snap_overrides(kinds).items():
        old = current[name]
        numeric = all(type(v) in (int, float) for v in (old, value))
        current[name] = max(old, value) if numeric else value
    return snap_mod.check_snap_params(current)
