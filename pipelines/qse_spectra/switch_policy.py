#!/usr/bin/env python3
"""Declared online switch policy for the record-first QSE construction.

The record phase hands off to Davidson residual synthesis under a **declared**
rule, not a hard-coded one. The rule is part of the cell identity
(:class:`MethodPolicy`), so changing when the method escalates declares new
cells instead of silently reinterpreting admitted evidence.

Two triggers exist:

``pool_exhausted_only`` (canonical)
    Hand off only when the record selector's terminal receipt reports
    ``pool_exhausted``. This is the settled canonical policy.

``window_residual_stall`` (ablation only)
    Hand off earlier, when the selector's **online** target-residual norm
    stops improving: ``stall_rounds`` consecutive acquisition rounds each
    failing to improve the running best by a factor of ``improvement_ratio``.

**Boundary, load-bearing:** a trigger predicate may read only the selector's
online round trace. Exact spectra, reference excitations, and window errors
computed against them are post-run qualification data and must never reach a
switch decision. :func:`assert_online_only_receipt` enforces that
structurally by rejecting any receipt key outside the selector's declared
online vocabulary.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

TRIGGER_POOL_EXHAUSTED = "pool_exhausted_only"
TRIGGER_WINDOW_RESIDUAL_STALL = "window_residual_stall"
TRIGGERS = (TRIGGER_POOL_EXHAUSTED, TRIGGER_WINDOW_RESIDUAL_STALL)

_TRIGGER_REQUIRED_PARAMETERS: dict[str, frozenset[str]] = {
    TRIGGER_POOL_EXHAUSTED: frozenset(),
    TRIGGER_WINDOW_RESIDUAL_STALL: frozenset({"stall_rounds", "improvement_ratio"}),
}

# The selector's online receipt vocabulary. Anything outside this set is
# either exact data or an undeclared field; both are refused.
_ONLINE_STOP_KEYS = frozenset(
    {
        "stop_reason",
        "final_max_target_residual_norm",
        "residual_converged_pending",
        "residual_stop_threshold",
        "round_trace",
    }
)
_ONLINE_ROUND_KEYS = frozenset(
    {
        "round",
        "selected_count_before",
        "retained_frame_size",
        "max_target_residual_norm",
        "window_top",
        "window_margin",
        "window_pressure",
        "window_closed",
        "novel_candidate_count",
        "max_candidate_metric_novelty",
        "max_candidate_residual_capture",
        "selected_original_index",
        "selected_score",
        "selected_metric_novelty",
        "selected_residual_capture",
    }
)


class SwitchPolicyError(ValueError):
    """Raised on an invalid switch policy or a non-online trigger input."""


@dataclass(frozen=True)
class MethodPolicy:
    """The online decision rule one cell is run under."""

    switch_trigger: str = TRIGGER_POOL_EXHAUSTED
    parameters: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        trigger = str(self.switch_trigger)
        if trigger not in TRIGGERS:
            raise SwitchPolicyError(
                f"unknown switch trigger {trigger!r}; declared triggers are {list(TRIGGERS)}."
            )
        names = {str(name) for name, _value in self.parameters}
        if len(names) != len(self.parameters):
            raise SwitchPolicyError("switch-policy parameter names must be unique.")
        required = _TRIGGER_REQUIRED_PARAMETERS[trigger]
        if names != required:
            raise SwitchPolicyError(
                f"trigger {trigger!r} requires exactly the parameters "
                f"{sorted(required)}; got {sorted(names)}."
            )
        for name, value in self.parameters:
            number = float(value)
            if not math.isfinite(number) or number <= 0.0:
                raise SwitchPolicyError(
                    f"switch-policy parameter {name!r} must be finite and positive."
                )
        if trigger == TRIGGER_WINDOW_RESIDUAL_STALL:
            ratio = float(dict(self.parameters)["improvement_ratio"])
            if ratio > 1.0:
                raise SwitchPolicyError(
                    "improvement_ratio must lie in (0, 1]: it is the factor by "
                    "which a round must beat the running best to count as progress."
                )

    @staticmethod
    def from_mapping(payload: Mapping[str, Any]) -> "MethodPolicy":
        parameters = payload.get("parameters") or {}
        if not isinstance(parameters, Mapping):
            raise SwitchPolicyError("switch-policy parameters must be a mapping.")
        return MethodPolicy(
            switch_trigger=str(payload.get("switch_trigger", TRIGGER_POOL_EXHAUSTED)),
            parameters=tuple(sorted((str(k), float(v)) for k, v in parameters.items())),
        )

    def payload(self) -> dict[str, Any]:
        return {
            "schema": "static_qse_method_policy_v1",
            "switch_trigger": str(self.switch_trigger),
            "parameters": {name: float(value) for name, value in self.parameters},
        }

    def digest(self) -> str:
        blob = json.dumps(self.payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    @property
    def is_canonical(self) -> bool:
        return self == CANONICAL_METHOD_POLICY


CANONICAL_METHOD_POLICY = MethodPolicy(switch_trigger=TRIGGER_POOL_EXHAUSTED)


def window_residual_stall_policy(
    *, stall_rounds: int = 2, improvement_ratio: float = 0.5
) -> MethodPolicy:
    """Convenience constructor for the ablation trigger."""

    return MethodPolicy(
        switch_trigger=TRIGGER_WINDOW_RESIDUAL_STALL,
        parameters=(
            ("improvement_ratio", float(improvement_ratio)),
            ("stall_rounds", float(stall_rounds)),
        ),
    )


def assert_online_only_receipt(record_stop: Mapping[str, Any]) -> None:
    """Refuse any switch input outside the selector's online vocabulary.

    This is the structural guarantee that exact references cannot reach a
    switch decision: the predicate is handed the selector receipt and
    nothing else, and the receipt's shape is checked here.
    """

    if not isinstance(record_stop, Mapping):
        raise SwitchPolicyError("switch input must be the selector's stop receipt.")
    unexpected = sorted(set(map(str, record_stop)) - _ONLINE_STOP_KEYS)
    if unexpected:
        raise SwitchPolicyError(
            "switch input carries non-online receipt keys "
            f"{unexpected}; exact references must never reach a switch decision."
        )
    for index, row in enumerate(record_stop.get("round_trace") or ()):
        if not isinstance(row, Mapping):
            raise SwitchPolicyError(f"round_trace[{index}] is not a mapping.")
        extra = sorted(set(map(str, row)) - _ONLINE_ROUND_KEYS)
        if extra:
            raise SwitchPolicyError(
                f"round_trace[{index}] carries non-online keys {extra}."
            )


def resolve_switch_point(
    policy: MethodPolicy,
    record_stop: Mapping[str, Any],
    *,
    record_count: int,
) -> dict[str, Any]:
    """Decide whether, and after how many records, the handoff fires.

    Returns a typed decision record. ``handoff_k`` is ``None`` for the
    canonical policy (the maintained terminal support is the seed) and an
    integer record count for an early trigger.
    """

    assert_online_only_receipt(record_stop)
    trigger = str(policy.switch_trigger)
    stop_reason = str(record_stop.get("stop_reason", ""))
    base = {
        "switch_trigger": trigger,
        "policy_digest": policy.digest(),
        "inputs_are_online_only": True,
        "uses_exact_reference_for_decision": False,
        "record_stop_reason": stop_reason,
    }

    if trigger == TRIGGER_POOL_EXHAUSTED:
        fired = stop_reason == "pool_exhausted"
        return {
            **base,
            "fired": fired,
            "handoff_k": None,
            "reason": (
                "record_pool_exhausted" if fired else f"record_stop_{stop_reason or 'missing'}"
            ),
        }

    parameters = dict(policy.parameters)
    stall_rounds = int(parameters["stall_rounds"])
    improvement_ratio = float(parameters["improvement_ratio"])
    rows: Sequence[Mapping[str, Any]] = tuple(record_stop.get("round_trace") or ())

    best = math.inf
    stalls = 0
    for row in rows:
        residual = row.get("max_target_residual_norm")
        if residual is None:
            continue
        value = float(residual)
        if value < best * improvement_ratio:
            best = value
            stalls = 0
            continue
        best = min(best, value)
        stalls += 1
        if stalls < stall_rounds:
            continue
        selected_before = int(row.get("selected_count_before") or 0)
        acquired = selected_before + (
            1 if row.get("selected_original_index") is not None else 0
        )
        handoff_k = max(1, min(int(record_count), acquired))
        return {
            **base,
            "fired": True,
            "handoff_k": handoff_k,
            "reason": "window_residual_stall",
            "evidence": {
                "stall_round": int(row.get("round") or 0),
                "consecutive_stalled_rounds": stalls,
                "running_best_residual": None if math.isinf(best) else float(best),
                "round_residual": value,
                "improvement_ratio": improvement_ratio,
                "stall_rounds": stall_rounds,
            },
        }

    return {
        **base,
        "fired": stop_reason == "pool_exhausted",
        "handoff_k": None,
        "reason": (
            "record_pool_exhausted"
            if stop_reason == "pool_exhausted"
            else "no_stall_detected"
        ),
        "evidence": {
            "consecutive_stalled_rounds": stalls,
            "improvement_ratio": improvement_ratio,
            "stall_rounds": stall_rounds,
        },
    }


__all__ = [
    "CANONICAL_METHOD_POLICY",
    "MethodPolicy",
    "SwitchPolicyError",
    "TRIGGERS",
    "TRIGGER_POOL_EXHAUSTED",
    "TRIGGER_WINDOW_RESIDUAL_STALL",
    "assert_online_only_receipt",
    "resolve_switch_point",
    "window_residual_stall_policy",
]
