"""Scaffold/runtime contract dataclasses shared across pipeline lanes.

The contracts in this module intentionally avoid importing paper-lane
implementation modules. Richer concrete types are owned by the implementation
layers; these dataclasses define the stable payload shape crossing those layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping


PoolSourceKind = Literal["resolved_pool", "selected_terms_only"]
PoolCompleteness = Literal["complete", "partial", "selected_only"]


@dataclass(frozen=True)
class CandidatePoolSource:
    source_kind: PoolSourceKind
    pool_key: str | None
    completeness: PoolCompleteness
    pool_build_kwargs: Mapping[str, Any] = field(default_factory=dict)
    filter_payload: Mapping[str, Any] = field(default_factory=dict)

    @property
    def candidate_pool_complete(self) -> bool:
        return str(self.completeness) == "complete"


@dataclass(frozen=True)
class ScaffoldRuntimeInput:
    resolved_problem: Any
    psi_ref: Any
    psi_initial: Any
    base_layout: Any
    theta_runtime: Any
    theta_logical: Any | None
    structure_locked: bool
    exact_energy: float | None
    selected_terms: tuple[Any, ...] = field(default_factory=tuple)
    candidate_pool_terms: tuple[Any, ...] = field(default_factory=tuple)
    candidate_pool_source: CandidatePoolSource = field(
        default_factory=lambda: CandidatePoolSource(
            source_kind="selected_terms_only",
            pool_key=None,
            completeness="selected_only",
        )
    )
    provenance: Mapping[str, Any] = field(default_factory=dict)
    extensions: Mapping[str, Any] = field(default_factory=dict)

    @property
    def h_poly(self) -> Any:
        return self.resolved_problem.hamiltonian

    @property
    def controller_profile(self) -> str:
        return str(self.resolved_problem.default_controller_profile)

    @property
    def continuation_mode(self) -> str:
        return str(self.resolved_problem.default_continuation_mode)

    @property
    def can_structural_edit(self) -> bool:
        return (not bool(self.structure_locked)) and self.candidate_pool_source.candidate_pool_complete


@dataclass(frozen=True)
class ReplayScaffoldContext:
    cfg: Any
    h_poly: Any
    psi_ref: Any
    payload_in: dict[str, Any]
    family_info: dict[str, Any]
    family_pool: tuple[Any, ...]
    pool_meta: dict[str, Any]
    replay_terms: tuple[Any, ...]
    base_layout: Any
    adapt_theta_runtime: Any
    adapt_theta_logical: Any
    adapt_depth: int
    handoff_state_kind: str
    provenance_source: str
    family_terms_count: int
    append_family_info: dict[str, Any] | None = None
    append_family_pool: tuple[Any, ...] | None = None
    append_pool_meta: dict[str, Any] | None = None
    append_family_terms_count: int | None = None


__all__ = [
    "CandidatePoolSource",
    "PoolCompleteness",
    "PoolSourceKind",
    "ReplayScaffoldContext",
    "ScaffoldRuntimeInput",
]
