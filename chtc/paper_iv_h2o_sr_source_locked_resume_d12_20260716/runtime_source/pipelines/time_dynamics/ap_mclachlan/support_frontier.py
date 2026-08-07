"""Parent-indexed support-frontier helpers for AP-McLachlan.

This module never selects or commits support patches.  It only narrows, or
intentionally fails open to, the child append atoms submitted to the canonical
child-level support-patch selector.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from pipelines.time_dynamics.ap_mclachlan.support_atoms import SupportAtom


APPEND_MACRO_SCOUT_POLICY_V2 = "parent_frontier_child_append_prefilter_v2"
APPEND_MACRO_SCOUT_SCORE_MODE_OFF = "off"
APPEND_MACRO_SCOUT_SCORE_MODE_PARENT_TANGENT_SCHUR_GAIN = "parent_tangent_schur_gain"
APPEND_MACRO_SCOUT_SCORE_MODE_PARENT_LINEAR_RESIDUAL_V1 = "parent_linear_residual_v1"
APPEND_MACRO_SCOUT_SCORE_MODE_CACHED_CHILD_UCB = "cached_child_ucb"
APPEND_MACRO_SCOUT_SCORE_MODE_FULL_CHILD_BLOCK_DIAGNOSTIC = (
    "full_child_block_diagnostic"
)
APPEND_MACRO_SCOUT_SCORE_MODES = frozenset(
    {
        APPEND_MACRO_SCOUT_SCORE_MODE_OFF,
        APPEND_MACRO_SCOUT_SCORE_MODE_PARENT_TANGENT_SCHUR_GAIN,
        APPEND_MACRO_SCOUT_SCORE_MODE_PARENT_LINEAR_RESIDUAL_V1,
        APPEND_MACRO_SCOUT_SCORE_MODE_CACHED_CHILD_UCB,
        APPEND_MACRO_SCOUT_SCORE_MODE_FULL_CHILD_BLOCK_DIAGNOSTIC,
    }
)
APPEND_MACRO_SCOUT_CHEAP_SCORE_MODES = frozenset(
    {
        APPEND_MACRO_SCOUT_SCORE_MODE_PARENT_TANGENT_SCHUR_GAIN,
        APPEND_MACRO_SCOUT_SCORE_MODE_PARENT_LINEAR_RESIDUAL_V1,
    }
)


class SupportFrontierFailOpen(RuntimeError):
    """Signal that a non-authoritative frontier scout must preserve all children."""

    def __init__(
        self,
        reason: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(str(reason))
        self.reason = str(reason)
        self.metadata = dict(metadata or {})


@dataclass(frozen=True)
class SupportFrontierScore:
    """Non-authoritative parent/frontier score for telemetry and filtering."""

    parent_label: str
    score: float | None = None
    rank_score: float | None = None
    insertion_gain: float | None = None
    accepted_eligible: bool = False
    rejection_reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SupportFrontierParentRecord:
    """Parent-level audit record for a child append frontier."""

    parent_label: str
    ordinal: int
    child_count: int
    child_atom_ids: tuple[str, ...]
    child_atom_labels: tuple[str, ...]
    score: float | None = None
    rank_score: float | None = None
    insertion_gain: float | None = None
    selected_by_scout: bool = False
    included_in_frontier: bool = False
    rejection_reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "parent_label": str(self.parent_label),
            "ordinal": int(self.ordinal),
            "child_count": int(self.child_count),
            "child_atom_ids": [str(v) for v in self.child_atom_ids],
            "child_atom_labels": [str(v) for v in self.child_atom_labels],
            "score": _finite_or_none(self.score),
            "rank_score": _finite_or_none(self.rank_score),
            "insertion_gain": _finite_or_none(self.insertion_gain),
            "selected_by_scout": bool(self.selected_by_scout),
            "included_in_frontier": bool(self.included_in_frontier),
            "rejection_reason": (
                None if self.rejection_reason is None else str(self.rejection_reason)
            ),
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class SupportFrontierResult:
    """Filtered child append atoms plus non-authoritative frontier metadata."""

    child_append_atoms: tuple[SupportAtom, ...]
    metadata: Mapping[str, Any]


ParentFrontierScorer = Callable[[str, tuple[SupportAtom, ...], int], SupportFrontierScore]
DiagnosticParentScorer = ParentFrontierScorer


def build_append_support_frontier(
    *,
    atoms: Sequence[SupportAtom],
    enabled: bool,
    score_mode: str,
    parent_cap: int,
    score_min: float,
    fail_open: bool,
    residual_ratio: float,
    expand_if_residual_high: float,
    exchange_requested: bool,
    exchange_fail_open: bool,
    audit_parent_count: int,
    audit_parent_fraction: float,
    parent_cost_alpha: float,
    cheap_parent_scorer: ParentFrontierScorer | None = None,
    diagnostic_parent_scorer: DiagnosticParentScorer | None = None,
) -> SupportFrontierResult:
    """Return child append atoms after optional parent-frontier scouting.

    The returned atoms are always a subset of the input child atoms, unless the
    configured safety policy fails open and returns the original input.
    """

    atom_tuple = tuple(atoms)
    mode = _normalize_score_mode(score_mode)
    grouped = _group_atoms_by_parent(atom_tuple)
    parent_count = int(len(grouped))
    base_metadata = {
        "macro_scout_policy": APPEND_MACRO_SCOUT_POLICY_V2,
        "macro_scout_enabled": bool(enabled),
        "macro_scout_score_mode": str(mode),
        "macro_scout_parent_cap": int(parent_cap),
        "macro_scout_score_min": float(score_min),
        "macro_scout_fail_open": bool(fail_open),
        "macro_scout_expand_if_residual_high": float(expand_if_residual_high),
        "macro_scout_exchange_fail_open": bool(exchange_fail_open),
        "macro_scout_exchange_requested": bool(exchange_requested),
        "macro_scout_exchange_fail_open_frontier_preserved": False,
        "macro_scout_exchange_filtering_diagnostic_only": bool(
            exchange_requested and not exchange_fail_open
        ),
        "macro_scout_exchange_filtering_certification": (
            "uncertified_noncanonical_diagnostic"
            if bool(exchange_requested) and not bool(exchange_fail_open)
            else (
                "canonical_fail_open"
                if bool(exchange_requested) and bool(exchange_fail_open)
                else "not_exchange_requested"
            )
        ),
        "macro_scout_parent_cost_alpha": float(parent_cost_alpha),
        "macro_scout_parent_count_total": int(parent_count),
        "macro_scout_child_count_before": int(len(atom_tuple)),
        "macro_scout_measurement_saving_score_available": False,
        "macro_scout_diagnostic_full_child_set_scoring": bool(
            mode == APPEND_MACRO_SCOUT_SCORE_MODE_FULL_CHILD_BLOCK_DIAGNOSTIC
        ),
        "macro_scout_parent_audit": [],
    }
    if not atom_tuple:
        return _frontier_result(
            atom_tuple,
            metadata=base_metadata,
            reason="no_append_atoms",
            applied=False,
            fail_open_applied=False,
            parent_count_scored=0,
            parent_count_selected=0,
        )
    if not bool(enabled) or mode == APPEND_MACRO_SCOUT_SCORE_MODE_OFF:
        return _frontier_result(
            atom_tuple,
            metadata=base_metadata,
            reason="macro_scout_disabled",
            applied=False,
            fail_open_applied=False,
            parent_count_scored=0,
            parent_count_selected=0,
        )
    if int(parent_cap) <= 0:
        return _frontier_result(
            atom_tuple,
            metadata=base_metadata,
            reason="parent_cap_zero",
            applied=False,
            fail_open_applied=True,
            parent_count_scored=0,
            parent_count_selected=0,
        )
    if parent_count <= 1:
        return _frontier_result(
            atom_tuple,
            metadata=base_metadata,
            reason="single_parent_no_filter",
            applied=False,
            fail_open_applied=False,
            parent_count_scored=0,
            parent_count_selected=parent_count,
        )
    residual_high = (
        float(expand_if_residual_high) > 0.0
        and float(residual_ratio) >= float(expand_if_residual_high)
    )
    if residual_high:
        return _frontier_result(
            atom_tuple,
            metadata={**base_metadata, "macro_scout_residual_high": True},
            reason="residual_high_fail_open",
            applied=False,
            fail_open_applied=True,
            parent_count_scored=0,
            parent_count_selected=parent_count,
        )
    if bool(exchange_requested) and bool(exchange_fail_open):
        return _frontier_result(
            atom_tuple,
            metadata={
                **base_metadata,
                "macro_scout_exchange_fail_open_applied": True,
                "macro_scout_exchange_fail_open_frontier_preserved": True,
            },
            reason="exchange_fail_open_frontier_preserved",
            applied=False,
            fail_open_applied=True,
            parent_count_scored=0,
            parent_count_selected=parent_count,
        )

    if mode == APPEND_MACRO_SCOUT_SCORE_MODE_CACHED_CHILD_UCB:
        return _measurement_unavailable_result(
            atom_tuple,
            metadata=base_metadata,
            reason=f"{mode}_measurements_unavailable",
            fail_open=bool(fail_open),
            parent_count=parent_count,
        )

    if mode in APPEND_MACRO_SCOUT_CHEAP_SCORE_MODES:
        if cheap_parent_scorer is None:
            return _measurement_unavailable_result(
                atom_tuple,
                metadata=base_metadata,
                reason=f"{mode}_measurements_unavailable",
                fail_open=bool(fail_open),
                parent_count=parent_count,
            )
        scorer = cheap_parent_scorer
        result_reason = f"{mode}_parent_cap_applied"
        measurement_saving_available = True
    elif mode == APPEND_MACRO_SCOUT_SCORE_MODE_FULL_CHILD_BLOCK_DIAGNOSTIC:
        if diagnostic_parent_scorer is None:
            if bool(fail_open):
                return _frontier_result(
                    atom_tuple,
                    metadata=base_metadata,
                    reason="diagnostic_parent_scorer_missing_fail_open",
                    applied=False,
                    fail_open_applied=True,
                    parent_count_scored=0,
                    parent_count_selected=parent_count,
                )
            return _frontier_result(
                (),
                metadata=base_metadata,
                reason="diagnostic_parent_scorer_missing",
                applied=True,
                fail_open_applied=False,
                parent_count_scored=0,
                parent_count_selected=0,
            )
        scorer = diagnostic_parent_scorer
        result_reason = "diagnostic_full_child_block_parent_cap_applied"
        measurement_saving_available = False
    else:
        raise ValueError(f"Unsupported append macro scout score mode {mode!r}.")

    records: list[SupportFrontierParentRecord] = []
    ranked: list[tuple[str, float, int]] = []
    for ordinal, (parent, parent_atoms) in enumerate(grouped.items()):
        try:
            score = scorer(str(parent), tuple(parent_atoms), int(ordinal))
        except SupportFrontierFailOpen as exc:
            metadata = {**base_metadata, **dict(exc.metadata or {})}
            return _measurement_unavailable_result(
                atom_tuple,
                metadata=metadata,
                reason=exc.reason,
                fail_open=bool(fail_open),
                parent_count=parent_count,
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            records.append(
                _parent_record(
                    parent,
                    ordinal,
                    parent_atoms,
                    rejection_reason="diagnostic_parent_scoring_failed",
                    metadata={"error": str(exc)},
                )
            )
            continue
        rank_score = score.rank_score if score.rank_score is not None else score.score
        finite_rank = rank_score is not None and np.isfinite(float(rank_score))
        if not finite_rank:
            records.append(
                _parent_record(
                    parent,
                    ordinal,
                    parent_atoms,
                    score=score.score,
                    rank_score=score.rank_score,
                    insertion_gain=score.insertion_gain,
                    rejection_reason=score.rejection_reason or "nonfinite_parent_score",
                    metadata=score.metadata,
                )
            )
            continue
        if float(rank_score) < float(score_min):
            records.append(
                _parent_record(
                    parent,
                    ordinal,
                    parent_atoms,
                    score=score.score,
                    rank_score=score.rank_score,
                    insertion_gain=score.insertion_gain,
                    rejection_reason="parent_score_below_min",
                    metadata=score.metadata,
                )
            )
            continue
        records.append(
            _parent_record(
                parent,
                ordinal,
                parent_atoms,
                score=score.score,
                rank_score=score.rank_score,
                insertion_gain=score.insertion_gain,
                rejection_reason=score.rejection_reason,
                metadata=score.metadata,
            )
        )
        ranked.append((str(parent), float(rank_score), int(ordinal)))

    if not ranked:
        if bool(fail_open):
            audit = _audit_payload(records, parent_count, audit_parent_count, audit_parent_fraction)
            return _frontier_result(
                atom_tuple,
                metadata={**base_metadata, "macro_scout_parent_audit": audit},
                reason="no_finite_parent_scores_fail_open",
                applied=False,
                fail_open_applied=True,
                parent_count_scored=0,
                parent_count_selected=parent_count,
            )
        return _frontier_result(
            (),
            metadata={
                **base_metadata,
                "macro_scout_parent_audit": _audit_payload(
                    records,
                    parent_count,
                    audit_parent_count,
                    audit_parent_fraction,
                ),
            },
            reason="no_finite_parent_scores",
            applied=True,
            fail_open_applied=False,
            parent_count_scored=0,
            parent_count_selected=0,
        )

    ranked.sort(key=lambda item: (-float(item[1]), int(item[2])))
    keep = {
        parent
        for parent, _rank, _ordinal in ranked[: max(0, int(parent_cap))]
    }
    filtered = tuple(atom for atom in atom_tuple if str(atom.parent_label) in keep)
    if not filtered and bool(fail_open):
        audit = _audit_payload(
            _mark_records(records, keep),
            parent_count,
            audit_parent_count,
            audit_parent_fraction,
        )
        return _frontier_result(
            atom_tuple,
            metadata={**base_metadata, "macro_scout_parent_audit": audit},
            reason="empty_filtered_frontier_fail_open",
            applied=False,
            fail_open_applied=True,
            parent_count_scored=len(ranked),
            parent_count_selected=parent_count,
        )
    selected_records = _mark_records(records, keep)
    return _frontier_result(
        filtered,
        metadata={
            **base_metadata,
            "macro_scout_measurement_saving_score_available": bool(
                measurement_saving_available
            ),
            "macro_scout_parent_audit": _audit_payload(
                selected_records,
                parent_count,
                audit_parent_count,
                audit_parent_fraction,
            ),
        },
        reason=result_reason,
        applied=True,
        fail_open_applied=False,
        parent_count_scored=len(ranked),
        parent_count_selected=len(keep),
    )


def validate_append_macro_scout_score_mode(score_mode: str) -> str:
    mode = _normalize_score_mode(score_mode)
    if mode not in APPEND_MACRO_SCOUT_SCORE_MODES:
        raise ValueError(
            "append_macro_scout_score_mode must be one of "
            f"{sorted(APPEND_MACRO_SCOUT_SCORE_MODES)!r}."
        )
    return mode


def _normalize_score_mode(score_mode: str) -> str:
    mode = str(score_mode or APPEND_MACRO_SCOUT_SCORE_MODE_OFF).strip().lower()
    if mode == "parent_linear_residual":
        return APPEND_MACRO_SCOUT_SCORE_MODE_PARENT_LINEAR_RESIDUAL_V1
    if mode == "diagnostic_full_child_set_v1":
        return APPEND_MACRO_SCOUT_SCORE_MODE_FULL_CHILD_BLOCK_DIAGNOSTIC
    return mode


def _group_atoms_by_parent(
    atoms: Sequence[SupportAtom],
) -> dict[str, tuple[SupportAtom, ...]]:
    grouped: dict[str, list[SupportAtom]] = {}
    for atom in tuple(atoms):
        grouped.setdefault(str(atom.parent_label), []).append(atom)
    return {parent: tuple(parent_atoms) for parent, parent_atoms in grouped.items()}


def _parent_record(
    parent: str,
    ordinal: int,
    atoms: Sequence[SupportAtom],
    *,
    score: float | None = None,
    rank_score: float | None = None,
    insertion_gain: float | None = None,
    selected_by_scout: bool = False,
    included_in_frontier: bool = False,
    rejection_reason: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SupportFrontierParentRecord:
    atom_tuple = tuple(atoms)
    return SupportFrontierParentRecord(
        parent_label=str(parent),
        ordinal=int(ordinal),
        child_count=int(len(atom_tuple)),
        child_atom_ids=tuple(str(atom.atom_id) for atom in atom_tuple),
        child_atom_labels=tuple(str(atom.atom_label) for atom in atom_tuple),
        score=None if score is None else float(score),
        rank_score=None if rank_score is None else float(rank_score),
        insertion_gain=None if insertion_gain is None else float(insertion_gain),
        selected_by_scout=bool(selected_by_scout),
        included_in_frontier=bool(included_in_frontier),
        rejection_reason=rejection_reason,
        metadata=dict(metadata or {}),
    )


def _mark_records(
    records: Sequence[SupportFrontierParentRecord],
    keep: set[str],
) -> tuple[SupportFrontierParentRecord, ...]:
    marked: list[SupportFrontierParentRecord] = []
    for record in tuple(records):
        selected = str(record.parent_label) in keep
        marked.append(
            SupportFrontierParentRecord(
                parent_label=record.parent_label,
                ordinal=record.ordinal,
                child_count=record.child_count,
                child_atom_ids=record.child_atom_ids,
                child_atom_labels=record.child_atom_labels,
                score=record.score,
                rank_score=record.rank_score,
                insertion_gain=record.insertion_gain,
                selected_by_scout=selected,
                included_in_frontier=selected,
                rejection_reason=record.rejection_reason,
                metadata=record.metadata,
            )
        )
    return tuple(marked)


def _audit_payload(
    records: Sequence[SupportFrontierParentRecord],
    parent_count: int,
    audit_parent_count: int,
    audit_parent_fraction: float,
) -> list[dict[str, Any]]:
    limit = max(
        0,
        int(audit_parent_count),
        int(math.ceil(max(0.0, float(audit_parent_fraction)) * int(parent_count))),
    )
    if limit <= 0:
        return []
    sorted_records = sorted(
        tuple(records),
        key=lambda record: (
            not bool(record.included_in_frontier),
            record.rank_score is None,
            -(float(record.rank_score) if record.rank_score is not None else -math.inf),
            int(record.ordinal),
        ),
    )
    return [record.to_json_dict() for record in sorted_records[:limit]]


def _frontier_result(
    atoms: Sequence[SupportAtom],
    *,
    metadata: Mapping[str, Any],
    reason: str,
    applied: bool,
    fail_open_applied: bool,
    parent_count_scored: int,
    parent_count_selected: int,
) -> SupportFrontierResult:
    atom_tuple = tuple(atoms)
    result_metadata = {
        **dict(metadata),
        "macro_scout_applied": bool(applied),
        "macro_scout_reason": str(reason),
        "macro_scout_fail_open_applied": bool(fail_open_applied),
        "macro_scout_exchange_fail_open_applied": bool(
            dict(metadata).get("macro_scout_exchange_fail_open_applied", False)
        ),
        "macro_scout_exchange_fail_open_frontier_preserved": bool(
            dict(metadata).get(
                "macro_scout_exchange_fail_open_frontier_preserved", False
            )
        ),
        "macro_scout_exchange_filtering_diagnostic_only": bool(
            dict(metadata).get("macro_scout_exchange_filtering_diagnostic_only", False)
        ),
        "macro_scout_exchange_filtering_certification": str(
            dict(metadata).get(
                "macro_scout_exchange_filtering_certification",
                "not_exchange_requested",
            )
        ),
        "macro_scout_residual_high": bool(
            dict(metadata).get("macro_scout_residual_high", False)
        ),
        "macro_scout_parent_count_scored": int(parent_count_scored),
        "macro_scout_parent_count_selected": int(parent_count_selected),
        "macro_scout_child_count_after": int(len(atom_tuple)),
    }
    return SupportFrontierResult(
        child_append_atoms=atom_tuple,
        metadata=result_metadata,
    )


def _measurement_unavailable_result(
    atoms: Sequence[SupportAtom],
    *,
    metadata: Mapping[str, Any],
    reason: str,
    fail_open: bool,
    parent_count: int,
) -> SupportFrontierResult:
    if bool(fail_open):
        return _frontier_result(
            atoms,
            metadata=metadata,
            reason=reason,
            applied=False,
            fail_open_applied=True,
            parent_count_scored=0,
            parent_count_selected=int(parent_count),
        )
    return _frontier_result(
        (),
        metadata=metadata,
        reason=reason,
        applied=True,
        fail_open_applied=False,
        parent_count_scored=0,
        parent_count_selected=0,
    )


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


__all__ = [
    "APPEND_MACRO_SCOUT_POLICY_V2",
    "APPEND_MACRO_SCOUT_CHEAP_SCORE_MODES",
    "APPEND_MACRO_SCOUT_SCORE_MODE_CACHED_CHILD_UCB",
    "APPEND_MACRO_SCOUT_SCORE_MODE_FULL_CHILD_BLOCK_DIAGNOSTIC",
    "APPEND_MACRO_SCOUT_SCORE_MODE_OFF",
    "APPEND_MACRO_SCOUT_SCORE_MODE_PARENT_LINEAR_RESIDUAL_V1",
    "APPEND_MACRO_SCOUT_SCORE_MODE_PARENT_TANGENT_SCHUR_GAIN",
    "APPEND_MACRO_SCOUT_SCORE_MODES",
    "SupportFrontierFailOpen",
    "SupportFrontierResult",
    "SupportFrontierScore",
    "build_append_support_frontier",
    "validate_append_macro_scout_score_mode",
]
