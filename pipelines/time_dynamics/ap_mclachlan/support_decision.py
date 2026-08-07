"""Typed support-patch decision records for AP-McLachlan controllers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class RungDiagnostics:
    """Telemetry for one append/prune candidate-batch rung."""

    rung_size: int
    candidate_set_count_before_prefilter: int
    candidate_set_count_scored: int
    prefilter_policy: str
    best_score: float | None = None
    best_atom_ids: tuple[str, ...] = ()
    rejection_reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "rung_size": int(self.rung_size),
            "candidate_set_count_before_prefilter": int(
                self.candidate_set_count_before_prefilter
            ),
            "candidate_set_count_scored": int(self.candidate_set_count_scored),
            "prefilter_policy": str(self.prefilter_policy),
            "best_score": _finite_or_none(self.best_score),
            "best_atom_ids": [str(atom_id) for atom_id in self.best_atom_ids],
            "rejection_reason": (
                None if self.rejection_reason is None else str(self.rejection_reason)
            ),
            "metadata": _json_safe(dict(self.metadata or {})),
        }


@dataclass(frozen=True)
class PatchActionProposal:
    """One stay/append/delete/exchange proposal before controller selection."""

    action_kind: str
    removed_atom_ids: tuple[str, ...] = ()
    inserted_atom_ids: tuple[str, ...] = ()
    support_patch: Any | None = None
    support_patch_score: Any | None = None
    resource_cost: Any | None = None
    after_residual_ratio: float | None = None
    normalized_score: float | None = None
    safety_report: Any | None = None
    rejection_reason: str | None = None
    commit_eligible: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "action_kind": str(self.action_kind),
            "removed_atom_ids": [str(atom_id) for atom_id in self.removed_atom_ids],
            "inserted_atom_ids": [str(atom_id) for atom_id in self.inserted_atom_ids],
            "support_patch": _to_json_obj(self.support_patch),
            "support_patch_score": _to_json_obj(self.support_patch_score),
            "resource_cost": _to_json_obj(self.resource_cost),
            "after_residual_ratio": _finite_or_none(self.after_residual_ratio),
            "normalized_score": _finite_or_none(self.normalized_score),
            "safety_report": _to_json_obj(self.safety_report),
            "rejection_reason": (
                None if self.rejection_reason is None else str(self.rejection_reason)
            ),
            "commit_eligible": bool(self.commit_eligible),
            "metadata": _json_safe(dict(self.metadata or {})),
        }


@dataclass(frozen=True)
class SupportPatchDecisionContext:
    """Frozen inputs needed to score support-patch proposals at one time point."""

    time_index: int
    time: float
    state: Any
    theta_runtime: Any
    hamiltonian: Any
    base_evaluation: Any
    base_step: Any
    inverse_policy: Any
    config: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


def proposal_counts_by_kind(proposals: Sequence[PatchActionProposal]) -> dict[str, int]:
    out = {
        "proposal_count_total": int(len(proposals)),
        "proposal_count_commit_eligible": 0,
        "proposal_count_append": 0,
        "proposal_count_insert": 0,
        "proposal_count_delete": 0,
        "proposal_count_exchange": 0,
        "proposal_count_stay": 0,
        "proposal_count_rejected": 0,
    }
    for proposal in proposals:
        kind = str(proposal.action_kind)
        key = f"proposal_count_{kind}"
        if key in out:
            out[key] += 1
        if kind == "insert":
            out["proposal_count_append"] += 1
        if bool(proposal.commit_eligible):
            out["proposal_count_commit_eligible"] += 1
        if proposal.rejection_reason is not None:
            out["proposal_count_rejected"] += 1
    return out


def _to_json_obj(value: Any) -> Any:
    if value is None:
        return None
    to_json = getattr(value, "to_json_dict", None)
    if callable(to_json):
        return to_json()
    return _json_safe(value)


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    out = float(value)
    return out if np.isfinite(out) else None


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(val) for val in value]
    try:
        out = float(value)
    except (TypeError, ValueError):
        return str(value)
    return out if np.isfinite(out) else None


__all__ = [
    "PatchActionProposal",
    "RungDiagnostics",
    "SupportPatchDecisionContext",
    "proposal_counts_by_kind",
]
