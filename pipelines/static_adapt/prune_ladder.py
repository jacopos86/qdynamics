#!/usr/bin/env python3
"""Typed prune compensator pools for static ADAPT recoverability ladders.

This helper owns nomination/refit-window structure and recovery telemetry only.
Actual deletion acceptance remains the existing remove-refit energy safety ladder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from pipelines.static_adapt.commutation_metadata import (
    EXACTNESS_EXACT,
    AlgebraicPairMetadata,
)

RECOVERY_UNSET = "unset"
RECOVERY_FLAT_REDUNDANT = "flat_redundant"
RECOVERY_CURVATURE_COMPENSATED = "curvature_compensated"
RECOVERY_FAILED = "failed"

RUNG_FROZEN_DELETE = "frozen_delete"
RUNG_COMM_REFIT = "comm_refit"
RUNG_COMM_CORR_REFIT = "comm_corr_refit"
RUNG_COMM_CORR_NC_REFIT = "comm_corr_nc_refit"
RUNG_TERMINAL_REFIT = "terminal_refit"


@dataclass(frozen=True)
class TypedCompensatorPools:
    removal_index: int
    removal_label: str
    survivor_original_indices: tuple[int, ...]
    comm_indices: tuple[int, ...]
    corr_indices: tuple[int, ...]
    nc_indices: tuple[int, ...]
    age_indices: tuple[int, ...]
    term_indices: tuple[int, ...]
    relation_summary: dict[str, int]


@dataclass(frozen=True)
class PruneRungSpec:
    rung_index: int
    rung_kind: str
    active_logical_indices: tuple[int, ...]
    opened_pool_kinds: tuple[str, ...]
    recovery_class_if_accepted: str | None


@dataclass(frozen=True)
class PruneLadderPlan:
    candidate_index: int
    candidate_label: str
    pools: TypedCompensatorPools
    rungs: tuple[PruneRungSpec, ...]


class PruneLadderError(ValueError):
    """Raised when typed prune ladder inputs are inconsistent."""


def original_to_post_delete_index(original_index: int, removal_index: int) -> int | None:
    """Map original coordinate index to post-delete coordinate index."""

    idx = int(original_index)
    remove = int(removal_index)
    if idx == remove:
        return None
    return int(idx if idx < remove else idx - 1)


def _unique_sorted(values: Sequence[int] | None) -> tuple[int, ...]:
    if values is None:
        return ()
    return tuple(sorted({int(x) for x in values}))


def _lookup_pair(
    pair_metadata: Mapping[tuple[str, str], AlgebraicPairMetadata] | Callable[[str, str], AlgebraicPairMetadata] | None,
    lhs_label: str,
    rhs_label: str,
) -> AlgebraicPairMetadata | None:
    if pair_metadata is None:
        return None
    if callable(pair_metadata):
        return pair_metadata(str(lhs_label), str(rhs_label))
    direct = pair_metadata.get((str(lhs_label), str(rhs_label)))
    if direct is not None:
        return direct
    return pair_metadata.get((str(rhs_label), str(lhs_label)))


def _post_delete_indices(
    original_indices: Sequence[int],
    *,
    removal_index: int,
    num_labels: int,
) -> tuple[int, ...]:
    out: set[int] = set()
    for original in original_indices:
        idx = int(original)
        if idx < 0 or idx >= int(num_labels):
            raise PruneLadderError(f"Coordinate index {idx!r} is outside [0, {num_labels}).")
        mapped = original_to_post_delete_index(idx, int(removal_index))
        if mapped is not None:
            out.add(int(mapped))
    return tuple(sorted(out))


def build_typed_compensator_pools(
    *,
    removal_index: int,
    labels: Sequence[str],
    pair_metadata: Mapping[tuple[str, str], AlgebraicPairMetadata] | Callable[[str, str], AlgebraicPairMetadata] | None = None,
    correlated_indices: Sequence[int] | None = None,
    age_indices: Sequence[int] | None = None,
    terminal_indices: Sequence[int] | None = None,
) -> TypedCompensatorPools:
    """Build exact typed compensator pools in post-delete logical indices.

    ``W_comm`` contains only exact support-overlapping commuting survivors.
    Disjoint commuting survivors are counted as telemetry, not redundancy
    compensators. Approximate/unknown pairs do not enter comm/noncomm pools.
    """

    if not labels:
        raise PruneLadderError("labels must be non-empty.")
    n = len(labels)
    remove = int(removal_index)
    if remove < 0 or remove >= n:
        raise PruneLadderError(f"removal_index={remove!r} is outside [0, {n}).")
    removal_label = str(labels[remove])
    survivors = tuple(idx for idx in range(n) if idx != remove)
    comm_original: list[int] = []
    nc_original: list[int] = []
    summary = {
        "exact_overlap_commuting": 0,
        "exact_overlap_noncommuting": 0,
        "exact_disjoint_commuting": 0,
        "approx_or_unknown": 0,
        "missing_pair_metadata": 0,
    }
    for idx in survivors:
        other_label = str(labels[idx])
        pair = _lookup_pair(pair_metadata, removal_label, other_label)
        if pair is None:
            summary["missing_pair_metadata"] += 1
            summary["approx_or_unknown"] += 1
            continue
        if pair.exactness != EXACTNESS_EXACT or pair.commutes is None:
            summary["approx_or_unknown"] += 1
            continue
        if bool(pair.support_overlap) and bool(pair.commutes):
            summary["exact_overlap_commuting"] += 1
            comm_original.append(int(idx))
        elif bool(pair.support_overlap) and not bool(pair.commutes):
            summary["exact_overlap_noncommuting"] += 1
            nc_original.append(int(idx))
        elif not bool(pair.support_overlap) and bool(pair.commutes):
            summary["exact_disjoint_commuting"] += 1
        else:
            summary["approx_or_unknown"] += 1

    terminal_original = tuple(range(n)) if terminal_indices is None else _unique_sorted(terminal_indices)
    return TypedCompensatorPools(
        removal_index=int(remove),
        removal_label=removal_label,
        survivor_original_indices=tuple(int(x) for x in survivors),
        comm_indices=_post_delete_indices(comm_original, removal_index=remove, num_labels=n),
        corr_indices=_post_delete_indices(_unique_sorted(correlated_indices), removal_index=remove, num_labels=n),
        nc_indices=_post_delete_indices(nc_original, removal_index=remove, num_labels=n),
        age_indices=_post_delete_indices(_unique_sorted(age_indices), removal_index=remove, num_labels=n),
        term_indices=_post_delete_indices(terminal_original, removal_index=remove, num_labels=n),
        relation_summary={str(key): int(val) for key, val in summary.items()},
    )


def _prefix_by_rho(indices: Sequence[int], rho: float) -> tuple[int, ...]:
    vals = tuple(sorted({int(x) for x in indices}))
    if not vals:
        return ()
    r = float(rho)
    if r <= 0.0:
        return ()
    if r >= 1.0:
        return vals
    count = max(1, int(round(len(vals) * r)))
    return vals[:count]


def _union_sorted(*groups: Sequence[int]) -> tuple[int, ...]:
    out: set[int] = set()
    for group in groups:
        out.update(int(x) for x in group)
    return tuple(sorted(out))


def build_nested_prune_rungs(
    pools: TypedCompensatorPools,
    *,
    rho_schedule: Sequence[float] = (0.0, 1.0, 1.0, 1.0, 1.0),
    terminal_full: bool = False,
) -> tuple[PruneRungSpec, ...]:
    """Build monotone typed post-delete refit rungs."""

    rho = list(float(x) for x in rho_schedule)
    while len(rho) < 5:
        rho.append(1.0)
    comm = _prefix_by_rho(pools.comm_indices, rho[1])
    corr = _prefix_by_rho(pools.corr_indices, rho[2])
    nc = _prefix_by_rho(pools.nc_indices, rho[3])
    age = _prefix_by_rho(pools.age_indices, rho[4])
    terminal = tuple(pools.term_indices) if bool(terminal_full) else ()

    r0 = ()
    r1 = _union_sorted(comm)
    r2 = _union_sorted(r1, corr)
    r3 = _union_sorted(r2, nc)
    r4 = _union_sorted(r3, age, terminal)
    terminal_pool_kinds = ("comm", "corr", "nc", "age", "terminal") if bool(terminal_full) else ("comm", "corr", "nc", "age")
    return (
        PruneRungSpec(
            rung_index=0,
            rung_kind=RUNG_FROZEN_DELETE,
            active_logical_indices=r0,
            opened_pool_kinds=(),
            recovery_class_if_accepted=RECOVERY_FLAT_REDUNDANT,
        ),
        PruneRungSpec(
            rung_index=1,
            rung_kind=RUNG_COMM_REFIT,
            active_logical_indices=r1,
            opened_pool_kinds=("comm",),
            recovery_class_if_accepted=RECOVERY_FLAT_REDUNDANT,
        ),
        PruneRungSpec(
            rung_index=2,
            rung_kind=RUNG_COMM_CORR_REFIT,
            active_logical_indices=r2,
            opened_pool_kinds=("comm", "corr"),
            recovery_class_if_accepted=RECOVERY_FLAT_REDUNDANT,
        ),
        PruneRungSpec(
            rung_index=3,
            rung_kind=RUNG_COMM_CORR_NC_REFIT,
            active_logical_indices=r3,
            opened_pool_kinds=("comm", "corr", "nc"),
            recovery_class_if_accepted=RECOVERY_CURVATURE_COMPENSATED,
        ),
        PruneRungSpec(
            rung_index=4,
            rung_kind=RUNG_TERMINAL_REFIT,
            active_logical_indices=r4,
            opened_pool_kinds=terminal_pool_kinds,
            recovery_class_if_accepted=RECOVERY_CURVATURE_COMPENSATED,
        ),
    )


def build_prune_ladder_plan(
    *,
    removal_index: int,
    labels: Sequence[str],
    pair_metadata: Mapping[tuple[str, str], AlgebraicPairMetadata] | Callable[[str, str], AlgebraicPairMetadata] | None = None,
    correlated_indices: Sequence[int] | None = None,
    age_indices: Sequence[int] | None = None,
    terminal_indices: Sequence[int] | None = None,
    rho_schedule: Sequence[float] = (0.0, 1.0, 1.0, 1.0, 1.0),
    terminal_full: bool = False,
) -> PruneLadderPlan:
    pools = build_typed_compensator_pools(
        removal_index=int(removal_index),
        labels=labels,
        pair_metadata=pair_metadata,
        correlated_indices=correlated_indices,
        age_indices=age_indices,
        terminal_indices=terminal_indices,
    )
    return PruneLadderPlan(
        candidate_index=int(removal_index),
        candidate_label=str(labels[int(removal_index)]),
        pools=pools,
        rungs=build_nested_prune_rungs(
            pools,
            rho_schedule=rho_schedule,
            terminal_full=bool(terminal_full),
        ),
    )


def rung_windows_for_recoverability(plan: PruneLadderPlan) -> list[tuple[str, list[int]]]:
    """Adapt typed rungs to the existing recoverability ladder window shape."""

    return [
        (str(rung.rung_kind), [int(idx) for idx in rung.active_logical_indices])
        for rung in plan.rungs
    ]


def classify_recoverability_from_ladder_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Classify accepted recovery from ladder telemetry without changing acceptance."""

    for fallback_idx, row in enumerate(rows):
        if not bool(row.get("accepted", False)):
            continue
        rung_index = int(row.get("rung_index", fallback_idx))
        recovery = RECOVERY_FLAT_REDUNDANT if rung_index <= 2 else RECOVERY_CURVATURE_COMPENSATED
        return {
            "recovery_class": recovery,
            "accepted": True,
            "accepted_rung_index": int(rung_index),
            "accepted_rung_kind": str(row.get("rung_kind", "")),
            "acceptance_source": row.get("acceptance_source"),
        }
    return {
        "recovery_class": RECOVERY_FAILED,
        "accepted": False,
        "accepted_rung_index": None,
        "accepted_rung_kind": None,
        "acceptance_source": None,
    }
