"""Shared mixed-horizon reporting contract for Paper-I continuation plots.

The Paper-I resource tuple remains a fixed controller-round-50 observation.
Continuation points are trajectory-only evidence and are therefore represented
additively instead of replacing ``points`` or ``terminal`` in an existing
page adapter.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any, Mapping, Sequence


BASE_HORIZON = 50
CONTINUATION_HORIZON = 70
STRONG_HOLSTEIN_REGIMES = (
    "weak_strong",
    "intermediate_strong",
    "strong_strong_u8",
)
CONTINUATION_ADAPTER_SCHEMA = (
    "paper_i_strong_sector_r50_to_r70_continuation_adapter_v1"
)


class MixedHorizonContinuationError(ValueError):
    """Raised when a continuation cannot extend its source-locked base."""


def horizon_policy() -> dict[str, Any]:
    """Return the shared plot/table interpretation for Pages 9 and 10."""

    return {
        "schema": "paper_i_mixed_horizon_trajectory_policy_v1",
        "base_horizon": BASE_HORIZON,
        "continuation_horizon": CONTINUATION_HORIZON,
        "continuation_scope": "strong_holstein_sector_only",
        "continuation_regimes": list(STRONG_HOLSTEIN_REGIMES),
        "trajectory_field": "trajectory_points",
        "trajectory_terminal_field": "trajectory_terminal",
        "paper_facing_cost_field": "paper_facing_fixed_round_50",
        "paper_facing_cost_round": BASE_HORIZON,
        "paper_facing_cost_policy": (
            "fixed_controller_round_50_v1; continuation points do not replace "
            "the Qiskit tuple or S_alg"
        ),
        "display_label": (
            "strong-sector curves may extend beyond k=50; all Qiskit tuples "
            "and S_alg entries remain fixed at k=50"
        ),
    }


def _point(raw: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise MixedHorizonContinuationError(f"{label} must be an object")
    try:
        raw_k = raw["k"]
        if isinstance(raw_k, bool) or not isinstance(raw_k, int):
            raise TypeError("round must be an integer")
        k = raw_k
        error = float(raw["error"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MixedHorizonContinuationError(
            f"{label} lacks a valid k/error pair"
        ) from exc
    if k < 1:
        raise MixedHorizonContinuationError(f"{label} has an invalid round")
    if not math.isfinite(error) or error < 0.0:
        raise MixedHorizonContinuationError(f"{label} has an invalid error")
    point = {"k": k, "error": error}
    if "energy" in raw:
        energy = float(raw["energy"])
        if not math.isfinite(energy):
            raise MixedHorizonContinuationError(
                f"{label} has a non-finite energy"
            )
        point["energy"] = energy
    return point


def _points(raw: Any, *, label: str) -> list[dict[str, Any]]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise MixedHorizonContinuationError(f"{label} must be a sequence")
    points = [_point(row, label=f"{label}[{index}]") for index, row in enumerate(raw)]
    if not points:
        raise MixedHorizonContinuationError(f"{label} is empty")
    rounds = [row["k"] for row in points]
    if rounds != list(range(rounds[0], rounds[-1] + 1)):
        raise MixedHorizonContinuationError(f"{label} is noncontiguous")
    return points


def merge_trajectory_points(
    base_points: Any,
    continuation_points: Any | None,
    *,
    label: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Merge a base trace with either a suffix or a complete resumed trace.

    Overlapping rounds are accepted only when their errors agree. This lets a
    completed continuation serialize rounds 1--70 while preventing k=50 from
    being appended twice. The returned second value is the strict k>50 suffix.
    """

    base = _points(base_points, label=f"{label} base points")
    if [row["k"] for row in base] != list(range(1, BASE_HORIZON + 1)):
        raise MixedHorizonContinuationError(
            f"{label} base must contain exactly rounds 1--{BASE_HORIZON}"
        )
    if continuation_points is None:
        return base, []

    supplied = _points(
        continuation_points,
        label=f"{label} continuation points",
    )
    base_by_round = {row["k"]: row for row in base}
    suffix: list[dict[str, Any]] = []
    for row in supplied:
        if row["k"] <= BASE_HORIZON:
            expected = base_by_round.get(row["k"])
            if expected is None or not math.isclose(
                row["error"],
                expected["error"],
                rel_tol=1.0e-11,
                abs_tol=1.0e-15,
            ):
                raise MixedHorizonContinuationError(
                    f"{label} continuation disagrees at k={row['k']}"
                )
            continue
        suffix.append(row)

    if suffix:
        expected_rounds = list(
            range(BASE_HORIZON + 1, suffix[-1]["k"] + 1)
        )
        if [row["k"] for row in suffix] != expected_rounds:
            raise MixedHorizonContinuationError(
                f"{label} continuation must begin at k={BASE_HORIZON + 1}"
            )
        if suffix[-1]["k"] > CONTINUATION_HORIZON:
            raise MixedHorizonContinuationError(
                f"{label} continuation exceeds k={CONTINUATION_HORIZON}"
            )
    return [*base, *suffix], suffix


def coalesce_continuation_points(
    base_points: Any,
    continuation_sources: Sequence[Any],
    *,
    label: str,
) -> list[dict[str, Any]]:
    """Combine independently authenticated prefixes without duplicating rounds."""

    base, _ = merge_trajectory_points(base_points, None, label=label)
    by_round = {row["k"]: row for row in base}
    for index, source in enumerate(continuation_sources):
        normalized, _ = merge_trajectory_points(
            base,
            source,
            label=f"{label} source {index}",
        )
        for row in normalized[BASE_HORIZON:]:
            previous = by_round.get(row["k"])
            if previous is not None:
                if not math.isclose(
                    previous["error"],
                    row["error"],
                    rel_tol=1.0e-11,
                    abs_tol=1.0e-15,
                ):
                    raise MixedHorizonContinuationError(
                        f"{label} continuation sources disagree at k={row['k']}"
                    )
                continue
            by_round[row["k"]] = row
    merged = [by_round[k] for k in sorted(by_round)]
    if [row["k"] for row in merged] != list(range(1, merged[-1]["k"] + 1)):
        raise MixedHorizonContinuationError(
            f"{label} continuation sources leave a trajectory gap"
        )
    return merged


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "sha256"}
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def validate_continuation_adapter(
    raw: Any,
    *,
    expected_route_contract_sha256: str,
) -> dict[str, dict[str, Any]]:
    """Validate the additive result adapter consumed after a k=70 fetch."""

    if not isinstance(raw, Mapping):
        raise MixedHorizonContinuationError("continuation adapter must be an object")
    if (
        raw.get("schema") != CONTINUATION_ADAPTER_SCHEMA
        or raw.get("paper_evidence_adopted") is not False
        or raw.get("route_contract_sha256") != expected_route_contract_sha256
        or raw.get("base_horizon") != BASE_HORIZON
        or raw.get("continuation_horizon") != CONTINUATION_HORIZON
        or raw.get("sha256") != _canonical_sha256(raw)
    ):
        raise MixedHorizonContinuationError("continuation adapter identity drifted")
    if not isinstance(raw.get("status"), str) or not raw["status"]:
        raise MixedHorizonContinuationError("continuation adapter status is invalid")
    cells = raw.get("cells")
    if not isinstance(cells, list):
        raise MixedHorizonContinuationError("continuation adapter cells are invalid")
    result: dict[str, dict[str, Any]] = {}
    for index, cell in enumerate(cells):
        if not isinstance(cell, Mapping):
            raise MixedHorizonContinuationError(
                f"continuation adapter cell {index} is invalid"
            )
        regime = str(cell.get("regime_id", ""))
        if regime not in STRONG_HOLSTEIN_REGIMES or regime in result:
            raise MixedHorizonContinuationError(
                f"continuation adapter regime is invalid: {regime!r}"
            )
        status = str(cell.get("status", ""))
        if status not in {
            "queued",
            "running",
            "recoverable_prefix_incomplete",
            "complete",
            "failed_preserving_prefix",
        }:
            raise MixedHorizonContinuationError(
                f"{regime}: unsupported adapter status {status!r}"
            )
        raw_points = cell.get("trajectory_points", cell.get("points"))
        points = _points(raw_points, label=f"{regime} adapter points")
        observed = cell.get("observed_through_round")
        if isinstance(observed, bool) or not isinstance(observed, int):
            raise MixedHorizonContinuationError(
                f"{regime}: adapter endpoint is not an integer"
            )
        if points[-1]["k"] != observed or observed > CONTINUATION_HORIZON:
            raise MixedHorizonContinuationError(
                f"{regime}: adapter endpoint drifted"
            )
        if status == "complete" and observed != CONTINUATION_HORIZON:
            raise MixedHorizonContinuationError(
                f"{regime}: complete adapter must reach k={CONTINUATION_HORIZON}"
            )
        source_bindings = cell.get("source_bindings")
        if not isinstance(source_bindings, Mapping) or not source_bindings:
            raise MixedHorizonContinuationError(
                f"{regime}: adapter source bindings are absent"
            )
        for role, source_binding in source_bindings.items():
            if (
                not isinstance(source_binding, Mapping)
                or not isinstance(source_binding.get("sha256"), str)
                or len(source_binding["sha256"]) != 64
            ):
                raise MixedHorizonContinuationError(
                    f"{regime}: {role} source binding is invalid"
                )
        result[regime] = {
            "points": points,
            "status": status,
            "source": {
                "kind": "authenticated_k70_continuation_adapter",
                "source_bindings": copy.deepcopy(dict(source_bindings)),
                "adapter_sha256": str(raw["sha256"]),
            },
        }
    return result


def digested_continuation_adapter(
    *,
    route_contract_sha256: str,
    cells: Sequence[Mapping[str, Any]],
    status: str,
) -> dict[str, Any]:
    """Build the small additive adapter written by a retrieval workflow."""

    adapter: dict[str, Any] = {
        "schema": CONTINUATION_ADAPTER_SCHEMA,
        "status": status,
        "paper_evidence_adopted": False,
        "route_contract_sha256": route_contract_sha256,
        "base_horizon": BASE_HORIZON,
        "continuation_horizon": CONTINUATION_HORIZON,
        "cells": [copy.deepcopy(dict(cell)) for cell in cells],
    }
    adapter["sha256"] = _canonical_sha256(adapter)
    return adapter


def decorate_route(
    route: Mapping[str, Any],
    *,
    regime_id: str,
    continuation_points: Any | None = None,
    continuation_status: str | None = None,
    continuation_source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Add trajectory-only continuation fields without changing k=50 costs."""

    selected = regime_id in STRONG_HOLSTEIN_REGIMES
    if continuation_points is not None and not selected:
        raise MixedHorizonContinuationError(
            f"{regime_id}: continuation is outside the strong Holstein sector"
        )
    terminal = route.get("terminal")
    if (
        not isinstance(terminal, Mapping)
        or isinstance(terminal.get("k"), bool)
        or terminal.get("k") != BASE_HORIZON
    ):
        raise MixedHorizonContinuationError(
            f"{regime_id}: fixed-round terminal must be k={BASE_HORIZON}"
        )
    for field in ("N2q", "D2q", "Dc", "W1q", "S_alg"):
        value = terminal.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise MixedHorizonContinuationError(
                f"{regime_id}: fixed-round {field} is invalid"
            )

    trajectory, suffix = merge_trajectory_points(
        route.get("points"),
        continuation_points,
        label=regime_id,
    )
    observed = trajectory[-1]["k"]
    if not selected:
        status = "not_selected"
    elif suffix and observed == CONTINUATION_HORIZON:
        status = "complete"
    elif suffix:
        status = continuation_status or "recoverable_prefix_incomplete"
    else:
        status = continuation_status or "pending"
    allowed_statuses = {
        "not_selected",
        "pending",
        "pending_base_horizon",
        "queued",
        "running",
        "recoverable_prefix_incomplete",
        "complete",
        "failed_preserving_prefix",
    }
    if status not in allowed_statuses:
        raise MixedHorizonContinuationError(
            f"{regime_id}: unsupported continuation status {status!r}"
        )
    if status == "complete" and observed != CONTINUATION_HORIZON:
        raise MixedHorizonContinuationError(
            f"{regime_id}: complete continuation must end at "
            f"k={CONTINUATION_HORIZON}"
        )

    decorated = copy.deepcopy(dict(route))
    decorated.update(
        {
            "base_horizon": BASE_HORIZON,
            "continuation_horizon": (
                CONTINUATION_HORIZON if selected else None
            ),
            "trajectory_points": trajectory,
            "trajectory_terminal": copy.deepcopy(trajectory[-1]),
            "paper_facing_fixed_round_50": copy.deepcopy(dict(terminal)),
            "continuation": {
                "status": status,
                "selected": selected,
                "base_round": BASE_HORIZON,
                "target_round": (
                    CONTINUATION_HORIZON if selected else None
                ),
                "observed_through_round": observed,
                "continuation_point_count": len(suffix),
                "source": (
                    copy.deepcopy(dict(continuation_source))
                    if continuation_source is not None
                    else None
                ),
            },
        }
    )
    return decorated


def missing_route_continuation_status(*, regime_id: str) -> dict[str, Any]:
    """Describe continuation state when the base route is still unavailable."""

    selected = regime_id in STRONG_HOLSTEIN_REGIMES
    return {
        "status": "pending_base_horizon" if selected else "not_selected",
        "selected": selected,
        "base_round": BASE_HORIZON,
        "target_round": CONTINUATION_HORIZON if selected else None,
        "observed_through_round": None,
        "continuation_point_count": 0,
        "source": None,
    }
