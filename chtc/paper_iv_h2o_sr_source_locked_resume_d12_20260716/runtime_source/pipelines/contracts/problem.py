"""Data-only problem-family contracts shared across pipeline lanes.

This module owns stable shapes for Hamiltonian-family requests, register layout,
sector/reference/target metadata, and resolved runtime context.  It intentionally
contains no static-ADAPT resolver logic, Hamiltonian construction, chemistry
loaders, NumPy state construction, or paper-lane imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Union


def canonical_problem_key(problem_key: str) -> str:
    return str(problem_key).strip().lower()


@dataclass(frozen=True)
class ProblemRequest:
    problem_key: str
    num_sites: int
    t: float
    u: float
    dv: float
    omega0: float
    g_ep: float
    n_ph_max: int
    boson_encoding: str
    ordering: str
    boundary: str
    include_zero_point: bool = True
    molecular_problem_json: str | None = None
    molecular_vibronic_h2_fixture_json: str | None = None
    molecular_vibronic_h2o_fixture_json: str | None = None
    v_nn: float = 0.0
    t_prime: float = 0.0
    n_fermions: int | None = None
    molecular_vibronic_h2o_linear_fd_fixture_json: str | None = None


@dataclass(frozen=True)
class RegisterBlockSpec:
    name: str
    kind: str
    start_qubit: int
    stop_qubit: int

    @property
    def size(self) -> int:
        return max(0, int(self.stop_qubit) - int(self.start_qubit))


@dataclass(frozen=True)
class RegisterLayoutSpec:
    total_qubits: int
    fermion_qubits: int
    boson_qubits: int
    ordering: str
    boson_encoding: str | None
    blocks: tuple[RegisterBlockSpec, ...]
    qubit_order_label: str = "left-to-right = q_(n-1)...q_0"

    def block(self, name: str) -> RegisterBlockSpec | None:
        target = str(name)
        for blk in self.blocks:
            if blk.name == target:
                return blk
        return None


@dataclass(frozen=True)
class FixedCountConstraint:
    quantity: str
    value: int
    scope: str = "full_register"
    kind: str = "fixed_count"


@dataclass(frozen=True)
class ParityConstraint:
    quantity: str
    parity: str
    scope: str = "full_register"
    kind: str = "parity"


@dataclass(frozen=True)
class WeightedChargeConstraint:
    quantity: str
    weights: tuple[tuple[str, int], ...]
    value: int
    scope: str = "full_register"
    kind: str = "weighted_charge"


@dataclass(frozen=True)
class TruncationConstraint:
    quantity: str
    max_local_occupancy: int
    scope: str = "full_register"
    kind: str = "truncation"


SectorConstraint = Union[
    FixedCountConstraint,
    ParityConstraint,
    WeightedChargeConstraint,
    TruncationConstraint,
]


@dataclass(frozen=True)
class SectorSelection:
    label: str
    comparison_space_label: str
    constraints: tuple[SectorConstraint, ...]
    num_particles: tuple[int, int] | None = None


@dataclass(frozen=True)
class ReferenceStateSpec:
    kind: str
    source_label: str
    state_kind: str
    build_state: Callable[[], Any]


@dataclass(frozen=True)
class ExactTargetSpec:
    kind: str
    comparison_space_label: str
    resolve_energy: Callable[..., float]
    exact_state_policy: str
    build_fallback_anchor_state: Callable[[], Any]
    fallback_policy: str


@dataclass(frozen=True)
class HamiltonianFamilyCapabilities:
    """Typed, data-only contract for Hamiltonian-family runtime seams."""

    observable_kind: str = "unsupported"
    primary_density_modes: tuple[str, ...] = ("auto",)
    drive_operator_kind: str | None = None
    supports_measurement_observables: bool = False
    supports_driven_realtime: bool = False
    supports_drive_mode_off: bool = False
    supports_drive_exact_v1: bool = False
    supports_drive_benchmark_exact: bool = False
    supports_strict_qpu_faithful: bool = False
    supports_hamiltonian_flow_projective: bool = False
    report_manifest_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedProblemContext:
    family_key: str
    request: ProblemRequest
    layout: RegisterLayoutSpec
    hamiltonian: Any
    sector: SectorSelection
    reference_state: ReferenceStateSpec
    exact_target: ExactTargetSpec
    default_controller_profile: str
    default_continuation_mode: str
    admissible_pool_keys: tuple[str, ...]
    default_pool_key: str | None
    default_pool_resolution_scope: str
    default_sector_label: str
    default_reference_label: str
    exact_target_label: str
    exact_comparison_space_label: str
    default_num_particles: tuple[int, int]
    capabilities: HamiltonianFamilyCapabilities = field(
        default_factory=HamiltonianFamilyCapabilities
    )
    runtime_data: dict[str, Any] | None = None


@dataclass(frozen=True)
class ProblemFamilySpec:
    family_key: str
    default_controller_profile: str
    default_continuation_mode: str
    admissible_pool_keys: tuple[str, ...]
    default_pool_key: str | None
    default_pool_resolution_scope: str
    supported_boson_encodings: tuple[str, ...]
    default_sector_label: str
    default_reference_label: str
    exact_target_label: str
    exact_comparison_space_label: str
    _layout_builder: Callable[[ProblemRequest], RegisterLayoutSpec]
    capabilities: HamiltonianFamilyCapabilities = field(
        default_factory=HamiltonianFamilyCapabilities
    )
    _context_resolver: Callable[..., ResolvedProblemContext] | None = None


__all__ = [
    "ExactTargetSpec",
    "FixedCountConstraint",
    "HamiltonianFamilyCapabilities",
    "ParityConstraint",
    "ProblemFamilySpec",
    "ProblemRequest",
    "ReferenceStateSpec",
    "RegisterBlockSpec",
    "RegisterLayoutSpec",
    "ResolvedProblemContext",
    "SectorConstraint",
    "SectorSelection",
    "TruncationConstraint",
    "WeightedChargeConstraint",
    "canonical_problem_key",
]
