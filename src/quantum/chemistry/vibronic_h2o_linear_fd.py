from __future__ import annotations

from dataclasses import dataclass, field
from math import comb
import json
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from src.quantum.chemistry.molecular_hamiltonian import (
    build_one_body_jw_polynomial,
    build_two_body_jw_polynomial,
)
from src.quantum.chemistry.molecular_uccsd import build_molecular_uccsd_pool
from src.quantum.chemistry.psi4_adapter import RestrictedClosedShellMolecularProblem
from src.quantum.chemistry.vibronic_h2 import (
    _clean_real_polynomial,
    _lift_fermion_polynomial,
    pauli_polynomial_from_jsonable,
)
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


H2O_LINEAR_FD_FIXTURE_SCHEMA = "molecular_vibronic_h2o_linear_fd_fixture_v1"
H2O_LINEAR_FD_FAMILY_KEY = "molecular_vibronic_h2o_linear_fd"
H2O_UMBRELLA_FAMILY_KEY = "molecular_vibronic_h2o"
H2O_LINEAR_FD_MODEL_ROLE = "production_linear_fd"
H2O_LINEAR_FD_DERIVATIVE_SOURCE = "finite_difference_mass_weighted_normal_modes_center_aligned_v1"
H2O_LINEAR_FD_REQUIRED_MODE_LABELS = ("bend", "symmetric_stretch", "antisymmetric_stretch")
H2O_LINEAR_FD_CUTOFF_ENERGY_TOLERANCE_HARTREE = 1.6e-3
H2O_LINEAR_FD_CUTOFF_BOUNDARY_WEIGHT_TOLERANCE = 1.0e-2
H2O_LINEAR_FD_DERIVATIVE_RESOLVED_POOL_KEY = "full_meta_derivative_resolved_v2"

SpinOrdering = Literal["blocked"]
BosonEncoding = Literal["binary"]
ProductionStatus = Literal[
    "production_validated",
    "production_candidate",
    "diagnostic_failed",
    "prototype_smoke_only",
]
ReferenceMethod = Literal["dense_sector_eigh", "sparse_sector_eigsh", "not_computed"]
ExactStateRepresentation = Literal["sparse_full_register_qn_to_q0", "external_sidecar"]


@dataclass(frozen=True)
class GeometryRecord:
    geometry_id: str
    molecule: str
    symbols: tuple[str, ...]
    coordinates_bohr: np.ndarray
    masses_me: np.ndarray
    charge: int
    multiplicity: int
    method: str
    basis: str
    reference: str
    optimized: bool
    coordinate_units: str = "bohr"
    mass_units: str = "electron_mass"
    symmetry: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NormalModeRecord:
    mode_index: int
    label: str
    frequency_hartree: float
    frequency_cm1: float | None
    mass_weighted_eigenvector: np.ndarray
    q_step_au: float
    q_step_alt_au: float | None
    normalization: str = "sum_Aalpha e_mu_Aalpha e_nu_Aalpha = delta_munu"
    coordinate_convention: str = "mass_weighted_normal_au"
    ladder_convention: str = "Q=(2*omega)^(-1/2)*(a_dag+a)"
    trans_rot_residual: float | None = None


@dataclass(frozen=True)
class DisplacedGeometryRecord:
    displacement_id: str
    purpose: str
    mode_indices: tuple[int, ...]
    signs: tuple[int, ...]
    q_displacements_au: tuple[float, ...]
    geometry_id: str
    snapshot_id: str | None
    coordinates_bohr: np.ndarray


@dataclass(frozen=True)
class ActiveSpaceRecord:
    active_space_kind: str
    frozen_core_indices: tuple[int, ...]
    active_indices_center: tuple[int, ...]
    external_indices: tuple[int, ...]
    n_spatial_orbitals: int
    num_particles: tuple[int, int]
    scalar_energy_hartree: float
    one_body_integrals: np.ndarray
    two_body_integrals: np.ndarray
    orbital_character: Mapping[str, Any] = field(default_factory=dict)
    frozen_core_convention: str = "closed_shell_core_contraction"
    tensor_convention: str = "chemist_eri_pqrs"
    spin_orbital_ordering: SpinOrdering = "blocked"

    @property
    def n_spin_orbitals(self) -> int:
        return 2 * int(self.n_spatial_orbitals)

    @property
    def sector_dimension(self) -> int:
        n_alpha, n_beta = self.num_particles
        return comb(int(self.n_spatial_orbitals), int(n_alpha)) * comb(
            int(self.n_spatial_orbitals), int(n_beta)
        )


@dataclass(frozen=True)
class AlignmentThresholds:
    min_active_singular_value: float = 0.98
    max_active_residual_fro: float = 1.0e-5
    max_active_to_external_leakage_fro: float = 1.0e-2
    max_hermiticity_residual: float = 1.0e-10
    max_eri_symmetry_residual: float = 1.0e-8


@dataclass(frozen=True)
class AlignmentDiagnosticsRecord:
    alignment_id: str
    center_snapshot_id: str
    displaced_snapshot_id: str
    displacement_id: str
    block: str
    singular_values: np.ndarray
    min_singular_value: float
    alignment_residual_fro: float
    active_to_external_leakage_fro: float | None
    external_to_active_leakage_fro: float | None
    hermiticity_residual: float
    eri_symmetry_residual: float
    rotation_orthogonality_residual: float
    thresholds: AlignmentThresholds = field(default_factory=AlignmentThresholds)
    passed: bool = False
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class AlignedActiveTensorRecord:
    aligned_tensor_id: str
    source_snapshot_id: str
    displacement_id: str | None
    scalar_energy_hartree: float
    one_body_integrals: np.ndarray
    two_body_integrals: np.ndarray
    alignment_id: str | None
    tensor_convention: str = "chemist_eri_pqrs"


@dataclass(frozen=True)
class DerivativeNorms:
    scalar_abs: float
    one_body_fro: float
    two_body_fro: float
    pauli_l1: float | None = None
    sector_spectral_norm: float | None = None
    low_energy_norm: float | None = None


@dataclass(frozen=True)
class FirstDerivativeRecord:
    derivative_id: str
    mode_index: int
    mode_label: str
    q_step_au: float
    plus_aligned_tensor_id: str
    minus_aligned_tensor_id: str
    scalar_derivative_hartree_per_q: float
    one_body_derivative: np.ndarray
    two_body_derivative: np.ndarray
    scalar_derivative_included: bool
    scalar_derivative_convention: str
    derivative_source: str = H2O_LINEAR_FD_DERIVATIVE_SOURCE
    pauli_operator: Mapping[str, Any] | None = None
    norms: DerivativeNorms | None = None
    finite_difference_drift: float | None = None
    finite_difference_diagnostics: Mapping[str, Any] = field(default_factory=dict)
    active_equilibrium_force: float | None = None
    passed: bool = False
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class BosonModeRegister:
    mode_index: int
    mode_label: str
    qubit_start: int
    n_qubits: int
    n_ph_max: int
    encoding: BosonEncoding = "binary"

    @property
    def qubits(self) -> tuple[int, ...]:
        return tuple(range(int(self.qubit_start), int(self.qubit_start) + int(self.n_qubits)))

    @property
    def valid_dimension(self) -> int:
        return int(self.n_ph_max) + 1

    @property
    def encoded_dimension(self) -> int:
        return 2 ** int(self.n_qubits)


@dataclass(frozen=True)
class RegisterLayout:
    n_fermion_qubits: int
    fermion_qubits: tuple[int, ...]
    boson_modes: tuple[BosonModeRegister, ...]
    spin_orbital_ordering: SpinOrdering = "blocked"

    @property
    def n_boson_qubits(self) -> int:
        return sum(int(block.n_qubits) for block in self.boson_modes)

    @property
    def n_total_qubits(self) -> int:
        return int(self.n_fermion_qubits) + int(self.n_boson_qubits)

    @property
    def boson_qubits(self) -> tuple[int, ...]:
        qubits: list[int] = []
        for block in self.boson_modes:
            qubits.extend(block.qubits)
        return tuple(qubits)


@dataclass(frozen=True)
class PhysicalSectorRecord:
    n_alpha: int
    n_beta: int
    n_ph_max_by_mode: tuple[int, ...]
    mode_labels: tuple[str, ...]

    @property
    def num_particles(self) -> tuple[int, int]:
        return int(self.n_alpha), int(self.n_beta)


@dataclass(frozen=True)
class EncodedOperatorBundle:
    h_electronic: Mapping[str, Any]
    dH_dQ_by_mode: tuple[Mapping[str, Any], ...]
    h_vibronic: Mapping[str, Any]
    q_by_mode: tuple[Mapping[str, Any], ...] = ()
    p_by_mode: tuple[Mapping[str, Any], ...] = ()
    n_by_mode: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class BoundaryWeightRecord:
    total_boundary_weight: float
    per_mode_boundary_weight: Mapping[str, float]
    state_source: str


@dataclass(frozen=True)
class ExactStateVectorRecord:
    available: bool
    representation: ExactStateRepresentation
    n_qubits: int
    norm: float
    amplitudes_qn_to_q0: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    sidecar_path: str | None = None
    sha256: str | None = None
    reason_unavailable: str | None = None


@dataclass(frozen=True)
class ExactReferenceRecord:
    available: bool
    method: ReferenceMethod
    sector_dimension: int
    full_qubit_dimension: int
    ground_energy_hartree: float | None
    low_energies_hartree: tuple[float, ...] = ()
    boundary_weight: BoundaryWeightRecord | None = None
    ground_state: ExactStateVectorRecord | None = None
    solver_tolerance: float | None = None
    reason_unavailable: str | None = None


@dataclass(frozen=True)
class CutoffDiagnosticsRecord:
    work_cutoffs: tuple[int, ...]
    reference_cutoffs: tuple[int, ...] | None
    work_ground_energy_hartree: float | None
    reference_ground_energy_hartree: float | None
    delta_energy_hartree: float | None
    work_boundary_weight: BoundaryWeightRecord | None
    passed: bool
    policy: str
    energy_tolerance_hartree: float | None = None
    boundary_weight_tolerance: float | None = None
    energy_passed: bool | None = None
    boundary_passed: bool | None = None


@dataclass(frozen=True)
class EvidenceHooksRecord:
    static_ground_state_ready: bool
    exact_reference_ready: bool
    dynamics_hooks_ready: bool = False
    qse_hooks_ready: bool = False
    qse_probe_families: tuple[str, ...] = ()
    qse_generator_families: tuple[str, ...] = ()


@dataclass(frozen=True)
class FixtureManifest:
    schema: str
    schema_version: int
    family_key: str
    molecule_family_key: str
    model_role: str
    production_status: ProductionStatus
    derivative_source: str
    created_utc: str
    generator_version: str
    repository_commit: str | None = None
    provenance_hashes: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ProductionVibronicH2OFixture:
    manifest: FixtureManifest
    geometry: GeometryRecord
    normal_modes: tuple[NormalModeRecord, ...]
    displacements: tuple[DisplacedGeometryRecord, ...]
    active_space: ActiveSpaceRecord
    aligned_tensors: tuple[AlignedActiveTensorRecord, ...]
    alignment_diagnostics: tuple[AlignmentDiagnosticsRecord, ...]
    first_derivatives: tuple[FirstDerivativeRecord, ...]
    layout: RegisterLayout
    physical_sector: PhysicalSectorRecord
    encoded_operators: EncodedOperatorBundle
    reference_state: np.ndarray
    exact_reference: ExactReferenceRecord
    cutoff_diagnostics: CutoffDiagnosticsRecord | None
    evidence_hooks: EvidenceHooksRecord
    pool: tuple[Mapping[str, Any], ...] = ()
    report_summary: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProductionVibronicH2OLinearFDRuntimeModel:
    h_vibronic: Any
    h_electronic: Any
    dH_dQ_by_mode: tuple[Any, ...]
    q_by_mode: tuple[Any, ...]
    p_by_mode: tuple[Any, ...]
    n_by_mode: tuple[Any, ...]
    pool: tuple[AnsatzTerm, ...]
    psi_ref: np.ndarray
    n_spatial_orbitals: int
    num_particles: tuple[int, int]
    n_fermion_qubits: int
    n_boson_qubits: int
    n_total_qubits: int
    mode_labels: tuple[str, ...]
    mode_cutoffs: tuple[int, ...]


@dataclass(frozen=True)
class CachedProductionVibronicH2OLinearFDFixture:
    fixture: ProductionVibronicH2OFixture
    model: ProductionVibronicH2OLinearFDRuntimeModel
    fixture_path: Path
    metadata: Mapping[str, Any]


def _as_mapping(payload: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object.")
    return payload


def _as_tuple_int(values: Any, *, label: str) -> tuple[int, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be a sequence.")
    return tuple(int(v) for v in values)


def _as_tuple_float(values: Any, *, label: str) -> tuple[float, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be a sequence.")
    return tuple(float(v) for v in values)


def _array_payload(array: np.ndarray, *, units: str | None = None, convention: str | None = None) -> dict[str, Any]:
    arr = np.asarray(array)
    flat = arr.reshape(-1)
    if np.iscomplexobj(arr):
        data: list[Any] = [[float(v.real), float(v.imag)] for v in flat]
        dtype = "complex128"
    else:
        data = [float(v) for v in flat]
        dtype = "float64"
    payload: dict[str, Any] = {
        "shape": [int(v) for v in arr.shape],
        "dtype": dtype,
        "data": data,
    }
    if units is not None:
        payload["units"] = str(units)
    if convention is not None:
        payload["convention"] = str(convention)
    return payload


def _array_from_payload(payload: Any, *, label: str) -> np.ndarray:
    obj = _as_mapping(payload, label=label)
    shape = tuple(int(v) for v in obj.get("shape", ()))
    dtype = str(obj.get("dtype", "float64"))
    data = obj.get("data")
    if data is None:
        raise ValueError(f"{label} missing array data.")
    if dtype.startswith("complex"):
        flat = np.asarray([complex(float(v[0]), float(v[1])) for v in data], dtype=complex)
    else:
        flat = np.asarray(data, dtype=float)
    try:
        return flat.reshape(shape)
    except Exception as exc:
        raise ValueError(f"{label} array shape/data mismatch: {exc}") from exc


def _optional_array_from_payload(payload: Any, *, label: str) -> np.ndarray | None:
    if payload is None:
        return None
    return _array_from_payload(payload, label=label)


def _array_json_or_none(array: np.ndarray | None, *, units: str | None = None, convention: str | None = None) -> dict[str, Any] | None:
    if array is None:
        return None
    return _array_payload(array, units=units, convention=convention)


def _hermiticity_residual(matrix: np.ndarray, *, eps: float = 1.0e-14) -> float:
    arr = np.asarray(matrix)
    denom = max(float(np.linalg.norm(arr)), float(eps))
    return float(np.linalg.norm(arr - arr.T.conj()) / denom)


def _eri_symmetry_residual_chemist(tensor: np.ndarray, *, eps: float = 1.0e-14) -> float:
    arr = np.asarray(tensor)
    denom = max(float(np.linalg.norm(arr)), float(eps))
    pair_swap = np.swapaxes(np.swapaxes(arr, 0, 1), 2, 3)
    bra_ket_swap = np.transpose(arr, (2, 3, 0, 1))
    return float(max(np.linalg.norm(arr - pair_swap), np.linalg.norm(arr - bra_ket_swap)) / denom)


def _eri_symmetry_abs_residual_chemist(tensor: np.ndarray) -> float:
    arr = np.asarray(tensor)
    pair_swap = np.swapaxes(np.swapaxes(arr, 0, 1), 2, 3)
    bra_ket_swap = np.transpose(arr, (2, 3, 0, 1))
    return float(max(np.linalg.norm(arr - pair_swap), np.linalg.norm(arr - bra_ket_swap)))


def _passes_eri_symmetry_chemist(
    tensor: np.ndarray,
    *,
    rel_tol: float = 1.0e-8,
    abs_tol: float = 1.0e-12,
) -> bool:
    return (
        _eri_symmetry_residual_chemist(tensor) <= float(rel_tol)
        or _eri_symmetry_abs_residual_chemist(tensor) <= float(abs_tol)
    )


def validate_normal_mode(mode: NormalModeRecord, *, n_atoms: int, atol: float = 1.0e-8) -> None:
    vec = np.asarray(mode.mass_weighted_eigenvector, dtype=float)
    if vec.shape != (int(n_atoms), 3):
        raise ValueError(
            f"mode {mode.mode_index}: expected mass-weighted eigenvector shape {(int(n_atoms), 3)}, got {vec.shape}."
        )
    norm = float(np.sum(vec * vec))
    if abs(norm - 1.0) > float(atol):
        raise ValueError(f"mode {mode.mode_index}: mass-weighted eigenvector norm {norm} != 1.")
    if str(mode.coordinate_convention) != "mass_weighted_normal_au":
        raise ValueError(f"mode {mode.mode_index}: unsupported coordinate convention {mode.coordinate_convention!r}.")
    if float(mode.frequency_hartree) <= 0.0 or not np.isfinite(float(mode.frequency_hartree)):
        raise ValueError(f"mode {mode.mode_index}: frequency_hartree must be positive and finite.")
    if float(mode.q_step_au) <= 0.0 or not np.isfinite(float(mode.q_step_au)):
        raise ValueError(f"mode {mode.mode_index}: q_step_au must be positive and finite.")


def fixed_sector_dimension(
    *,
    n_spatial_orbitals: int,
    num_particles: tuple[int, int],
    mode_cutoffs: Sequence[int],
) -> int:
    n_spatial = int(n_spatial_orbitals)
    n_alpha, n_beta = (int(num_particles[0]), int(num_particles[1]))
    if n_spatial <= 0:
        raise ValueError("n_spatial_orbitals must be positive.")
    if not (0 <= n_alpha <= n_spatial and 0 <= n_beta <= n_spatial):
        raise ValueError("invalid active-space particle counts.")
    dim = comb(n_spatial, n_alpha) * comb(n_spatial, n_beta)
    for cutoff in mode_cutoffs:
        if int(cutoff) < 0:
            raise ValueError("mode cutoffs must be nonnegative.")
        dim *= int(cutoff) + 1
    return int(dim)


def assess_h2o_linear_fd_cutoff_diagnostics(
    *,
    delta_energy_hartree: float | None,
    work_boundary_weight: BoundaryWeightRecord | None,
    energy_tolerance_hartree: float = H2O_LINEAR_FD_CUTOFF_ENERGY_TOLERANCE_HARTREE,
    boundary_weight_tolerance: float = H2O_LINEAR_FD_CUTOFF_BOUNDARY_WEIGHT_TOLERANCE,
) -> dict[str, Any]:
    energy_tolerance = float(energy_tolerance_hartree)
    boundary_tolerance = float(boundary_weight_tolerance)
    if not np.isfinite(energy_tolerance) or energy_tolerance <= 0.0:
        raise ValueError("cutoff energy tolerance must be positive and finite.")
    if not np.isfinite(boundary_tolerance) or not 0.0 <= boundary_tolerance <= 1.0:
        raise ValueError("cutoff boundary-weight tolerance must be finite and in [0, 1].")

    energy_passed = (
        None
        if delta_energy_hartree is None
        else bool(
            np.isfinite(float(delta_energy_hartree))
            and abs(float(delta_energy_hartree)) <= energy_tolerance
        )
    )
    boundary_passed = (
        None
        if work_boundary_weight is None
        else bool(
            np.isfinite(float(work_boundary_weight.total_boundary_weight))
            and float(work_boundary_weight.total_boundary_weight) <= boundary_tolerance
        )
    )
    return {
        "energy_tolerance_hartree": energy_tolerance,
        "boundary_weight_tolerance": boundary_tolerance,
        "energy_passed": energy_passed,
        "boundary_passed": boundary_passed,
        "passed": bool(energy_passed is True and boundary_passed is True),
    }


def validate_production_vibronic_h2o_fixture(
    fixture: ProductionVibronicH2OFixture,
    *,
    require_exact_policy: bool = True,
    require_production_validated: bool = True,
) -> None:
    manifest = fixture.manifest
    if str(manifest.schema) != H2O_LINEAR_FD_FIXTURE_SCHEMA:
        raise ValueError(f"not a production H2O linear-FD fixture: {manifest.schema!r}")
    if str(manifest.family_key) != H2O_LINEAR_FD_FAMILY_KEY:
        raise ValueError(f"unexpected production family_key: {manifest.family_key!r}")
    if str(manifest.model_role) != H2O_LINEAR_FD_MODEL_ROLE:
        raise ValueError(f"unexpected model_role: {manifest.model_role!r}")
    if str(manifest.derivative_source) != H2O_LINEAR_FD_DERIVATIVE_SOURCE:
        raise ValueError(f"unexpected derivative source: {manifest.derivative_source!r}")
    if require_production_validated and str(manifest.production_status) != "production_validated":
        raise ValueError(f"fixture is not production_validated: {manifest.production_status!r}")

    geometry = fixture.geometry
    active = fixture.active_space
    layout = fixture.layout
    modes = fixture.normal_modes

    coordinates = np.asarray(geometry.coordinates_bohr, dtype=float)
    masses = np.asarray(geometry.masses_me, dtype=float)
    if coordinates.shape != (len(geometry.symbols), 3):
        raise ValueError("geometry coordinate shape does not match symbols.")
    if masses.shape != (len(geometry.symbols),):
        raise ValueError("geometry masses shape does not match symbols.")
    if not np.all(np.isfinite(coordinates)) or not np.all(np.isfinite(masses)):
        raise ValueError("geometry coordinates/masses must be finite.")
    if np.any(masses <= 0.0):
        raise ValueError("geometry masses must be positive.")

    if int(layout.n_fermion_qubits) != 2 * int(active.n_spatial_orbitals):
        raise ValueError("fermion qubit count does not match active spatial orbital count.")
    if tuple(layout.fermion_qubits) != tuple(range(int(layout.n_fermion_qubits))):
        raise ValueError("unexpected fermion qubit block.")
    if str(layout.spin_orbital_ordering) != str(active.spin_orbital_ordering):
        raise ValueError("layout and active-space spin ordering disagree.")

    if len(layout.boson_modes) != len(modes):
        raise ValueError("number of boson mode blocks does not match retained modes.")
    if len(fixture.first_derivatives) != len(modes):
        raise ValueError("number of derivative records does not match retained modes.")
    if len(fixture.encoded_operators.dH_dQ_by_mode) != len(modes):
        raise ValueError("number of encoded derivative operators does not match retained modes.")

    expected_cutoffs = tuple(int(block.n_ph_max) for block in layout.boson_modes)
    if tuple(fixture.physical_sector.n_ph_max_by_mode) != expected_cutoffs:
        raise ValueError("physical sector cutoffs do not match boson register metadata.")
    if tuple(fixture.physical_sector.num_particles) != tuple(active.num_particles):
        raise ValueError("physical sector particle counts do not match active space.")
    if tuple(fixture.physical_sector.mode_labels) != tuple(block.mode_label for block in layout.boson_modes):
        raise ValueError("physical sector mode labels do not match boson register metadata.")

    expected_qubit = int(layout.n_fermion_qubits)
    for block in layout.boson_modes:
        if str(block.encoding) != "binary":
            raise ValueError(f"mode {block.mode_index}: unsupported boson encoding {block.encoding!r}.")
        if int(block.qubit_start) != expected_qubit:
            raise ValueError("boson mode blocks must be contiguous after the fermion register.")
        if int(block.n_qubits) <= 0:
            raise ValueError("boson mode blocks must have positive qubit counts.")
        if int(block.n_ph_max) < 0:
            raise ValueError("boson mode cutoffs must be nonnegative.")
        if int(block.valid_dimension) > int(block.encoded_dimension):
            raise ValueError("boson mode cutoff does not fit in its encoded qubits.")
        expected_qubit += int(block.n_qubits)
    if expected_qubit != int(layout.n_total_qubits):
        raise ValueError("register layout total qubit count is inconsistent.")

    if np.asarray(active.one_body_integrals).shape != (
        int(active.n_spatial_orbitals),
        int(active.n_spatial_orbitals),
    ):
        raise ValueError("active one-body tensor has wrong shape.")
    if np.asarray(active.two_body_integrals).shape != (
        int(active.n_spatial_orbitals),
        int(active.n_spatial_orbitals),
        int(active.n_spatial_orbitals),
        int(active.n_spatial_orbitals),
    ):
        raise ValueError("active two-body tensor has wrong shape.")
    if _hermiticity_residual(active.one_body_integrals) > 1.0e-10:
        raise ValueError("active one-body tensor is not Hermitian.")
    if not _passes_eri_symmetry_chemist(active.two_body_integrals):
        raise ValueError("active two-body tensor symmetry failed.")

    for mode, block in zip(modes, layout.boson_modes):
        validate_normal_mode(mode, n_atoms=len(geometry.symbols))
        if int(mode.mode_index) != int(block.mode_index) or str(mode.label) != str(block.mode_label):
            raise ValueError("normal-mode records and boson blocks disagree.")

    for diag in fixture.alignment_diagnostics:
        if not bool(diag.passed):
            raise ValueError(f"alignment diagnostic failed: {diag.alignment_id}")

    mode_ids = {int(mode.mode_index) for mode in modes}
    for deriv in fixture.first_derivatives:
        if int(deriv.mode_index) not in mode_ids:
            raise ValueError(f"derivative record has unknown mode index: {deriv.mode_index}")
        if str(deriv.derivative_source) != H2O_LINEAR_FD_DERIVATIVE_SOURCE:
            raise ValueError(f"mode {deriv.mode_index}: unexpected derivative source.")
        if not bool(deriv.scalar_derivative_included):
            raise ValueError(f"mode {deriv.mode_index}: scalar derivative is not included.")
        if not str(deriv.scalar_derivative_convention).strip():
            raise ValueError(f"mode {deriv.mode_index}: scalar derivative convention missing.")
        if not bool(deriv.passed):
            raise ValueError(f"derivative diagnostic failed: {deriv.derivative_id}")
        if np.asarray(deriv.one_body_derivative).shape != np.asarray(active.one_body_integrals).shape:
            raise ValueError(f"mode {deriv.mode_index}: derivative one-body tensor has wrong shape.")
        if np.asarray(deriv.two_body_derivative).shape != np.asarray(active.two_body_integrals).shape:
            raise ValueError(f"mode {deriv.mode_index}: derivative two-body tensor has wrong shape.")
        if _hermiticity_residual(deriv.one_body_derivative) > 1.0e-10:
            raise ValueError(f"mode {deriv.mode_index}: derivative one-body tensor is not Hermitian.")
        if not _passes_eri_symmetry_chemist(deriv.two_body_derivative):
            raise ValueError(f"mode {deriv.mode_index}: derivative ERI symmetry failed.")

    expected_sector_dim = fixed_sector_dimension(
        n_spatial_orbitals=int(active.n_spatial_orbitals),
        num_particles=tuple(active.num_particles),
        mode_cutoffs=expected_cutoffs,
    )
    if int(fixture.exact_reference.sector_dimension) != int(expected_sector_dim):
        raise ValueError("exact reference sector dimension does not match active sector/cutoffs.")
    if int(fixture.exact_reference.full_qubit_dimension) != 2 ** int(layout.n_total_qubits):
        raise ValueError("exact reference full-qubit dimension does not match layout.")
    if require_exact_policy and not fixture.exact_reference.available and not fixture.exact_reference.reason_unavailable:
        raise ValueError("exact reference unavailable without an explicit fallback policy.")

    reference_state = np.asarray(fixture.reference_state, dtype=complex).reshape(-1)
    if reference_state.shape != (2 ** int(layout.n_total_qubits),):
        raise ValueError("reference state has wrong shape.")
    norm = float(np.vdot(reference_state, reference_state).real)
    if abs(norm - 1.0) > 1.0e-10:
        raise ValueError("reference state is not normalized.")


def _normalize_h2o_mode_label(label: str) -> str:
    text = str(label).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "bend": "bend",
        "bending": "bend",
        "symmetric": "symmetric_stretch",
        "sym": "symmetric_stretch",
        "sym_stretch": "symmetric_stretch",
        "symmetric_stretch": "symmetric_stretch",
        "antisymmetric": "antisymmetric_stretch",
        "asymmetric": "antisymmetric_stretch",
        "asym": "antisymmetric_stretch",
        "asym_stretch": "antisymmetric_stretch",
        "antisymmetric_stretch": "antisymmetric_stretch",
        "asymmetric_stretch": "antisymmetric_stretch",
    }
    return aliases.get(text, text)


def _validate_exact_state_record(record: ExactStateVectorRecord, *, n_qubits: int) -> None:
    if int(record.n_qubits) != int(n_qubits):
        raise ValueError("exact-state qubit count does not match fixture layout.")
    if not np.isfinite(float(record.norm)) or float(record.norm) <= 0.0:
        raise ValueError("exact-state norm must be positive and finite.")
    if bool(record.available) and abs(float(record.norm) - 1.0) > 1.0e-8:
        raise ValueError("available exact-state norm is not normalized.")
    if str(record.representation) == "sparse_full_register_qn_to_q0":
        if bool(record.available) and not record.amplitudes_qn_to_q0:
            raise ValueError("available embedded exact state has no amplitudes.")
        amplitude_norm = 0.0
        for bitstr, coeff in record.amplitudes_qn_to_q0.items():
            if not isinstance(bitstr, str) or len(bitstr) != int(n_qubits) or any(ch not in "01" for ch in bitstr):
                raise ValueError(f"invalid exact-state bitstring: {bitstr!r}")
            if not isinstance(coeff, Mapping):
                raise ValueError(f"exact-state amplitude for {bitstr!r} must be an object.")
            amplitude = complex(float(coeff.get("re", 0.0)), float(coeff.get("im", 0.0)))
            amplitude_norm += float(abs(amplitude) ** 2)
        if bool(record.available):
            if not np.isfinite(float(amplitude_norm)) or float(amplitude_norm) <= 0.0:
                raise ValueError("available embedded exact state has non-positive amplitude norm.")
            if abs(float(amplitude_norm) - float(record.norm)) > 1.0e-8:
                raise ValueError("embedded exact-state amplitudes disagree with declared norm.")
            if abs(float(amplitude_norm) - 1.0) > 1.0e-8:
                raise ValueError("embedded exact-state amplitudes are not normalized.")
    elif str(record.representation) == "external_sidecar":
        if bool(record.available) and not (record.sidecar_path and record.sha256):
            raise ValueError("available exact-state sidecar requires sidecar_path and sha256.")
    else:
        raise ValueError(f"unsupported exact-state representation: {record.representation!r}")
    if not bool(record.available) and not record.reason_unavailable:
        raise ValueError("unavailable exact state requires reason_unavailable.")


def validate_paper_iv_h2o_linear_fd_evidence_fixture(
    fixture: ProductionVibronicH2OFixture,
    *,
    require_exact_state: bool = False,
    require_reference_cutoff: bool = True,
    require_pool: bool = True,
    require_cutoff_converged: bool = False,
) -> None:
    """Validate the fail-closed Paper-IV production evidence contract.

    This is intentionally stricter than the base schema validator.  The base
    validator permits small synthetic or intermediate production-schema fixtures;
    this gate is for the all-three-mode H2O linear finite-difference evidence
    path used by the Paper-IV static SNAKE application.
    """

    validate_production_vibronic_h2o_fixture(
        fixture,
        require_exact_policy=True,
        require_production_validated=True,
    )

    modes = fixture.normal_modes
    layout = fixture.layout
    active = fixture.active_space
    expected_cutoffs = tuple(int(block.n_ph_max) for block in layout.boson_modes)
    mode_labels = tuple(_normalize_h2o_mode_label(mode.label) for mode in modes)
    required_labels = tuple(H2O_LINEAR_FD_REQUIRED_MODE_LABELS)

    if len(modes) != 3:
        raise ValueError("Paper-IV H2O evidence requires exactly three normal modes.")
    if tuple(sorted(mode_labels)) != tuple(sorted(required_labels)):
        raise ValueError(
            "Paper-IV H2O evidence requires bend, symmetric_stretch, and antisymmetric_stretch modes."
        )
    if len(set(int(mode.mode_index) for mode in modes)) != 3:
        raise ValueError("Paper-IV H2O evidence requires unique mode indices.")
    if tuple(_normalize_h2o_mode_label(block.mode_label) for block in layout.boson_modes) != mode_labels:
        raise ValueError("Paper-IV mode labels disagree between normal modes and boson registers.")
    if tuple(fixture.physical_sector.mode_labels) != tuple(block.mode_label for block in layout.boson_modes):
        raise ValueError("Paper-IV physical sector mode labels disagree with register layout.")

    is_default_target = (
        int(active.n_spatial_orbitals) == 6
        and tuple(active.num_particles) == (4, 4)
        and int(layout.n_fermion_qubits) == 12
    )
    variant_declared = bool(
        fixture.report_summary.get("paper_iv_active_space_variant")
        or fixture.report_summary.get("active_space_variant")
        or str(active.active_space_kind).lower().startswith("synthetic")
    )
    if not is_default_target and not variant_declared:
        raise ValueError("non-CAS((8e,6o)) Paper-IV fixture must declare an active-space variant.")

    encoded = fixture.encoded_operators
    for label, rows in (
        ("encoded derivative", encoded.dH_dQ_by_mode),
        ("encoded Q", encoded.q_by_mode),
        ("encoded P", encoded.p_by_mode),
        ("encoded N", encoded.n_by_mode),
    ):
        if len(rows) != 3:
            raise ValueError(f"Paper-IV H2O evidence requires one {label} operator per mode.")

    derivative_mode_ids = [int(deriv.mode_index) for deriv in fixture.first_derivatives]
    if sorted(derivative_mode_ids) != sorted(int(mode.mode_index) for mode in modes):
        raise ValueError("Paper-IV derivative records do not cover exactly the retained modes.")
    for deriv in fixture.first_derivatives:
        if deriv.finite_difference_drift is None or not np.isfinite(float(deriv.finite_difference_drift)):
            raise ValueError(f"mode {deriv.mode_index}: finite-difference drift diagnostic missing.")

    exact = fixture.exact_reference
    if not bool(exact.available):
        raise ValueError("Paper-IV H2O evidence requires an available same-cutoff exact reference.")
    if exact.ground_energy_hartree is None or not np.isfinite(float(exact.ground_energy_hartree)):
        raise ValueError("Paper-IV same-cutoff exact ground energy is missing or non-finite.")
    if require_exact_state:
        if exact.ground_state is None or not bool(exact.ground_state.available):
            raise ValueError("Paper-IV fidelity reporting requires an available same-cutoff exact state.")
    if exact.ground_state is not None:
        _validate_exact_state_record(exact.ground_state, n_qubits=int(layout.n_total_qubits))

    cutoff = fixture.cutoff_diagnostics
    if cutoff is None:
        raise ValueError("Paper-IV H2O evidence requires cutoff diagnostics.")
    if tuple(cutoff.work_cutoffs) != expected_cutoffs:
        raise ValueError("Paper-IV cutoff diagnostics work cutoffs do not match fixture layout.")
    if cutoff.work_ground_energy_hartree is None or not np.isfinite(float(cutoff.work_ground_energy_hartree)):
        raise ValueError("Paper-IV cutoff diagnostics missing finite work ground energy.")
    if require_reference_cutoff:
        if cutoff.reference_cutoffs is None:
            raise ValueError("Paper-IV reference-cutoff reporting requires reference cutoffs.")
        if len(cutoff.reference_cutoffs) != 3:
            raise ValueError("Paper-IV reference cutoffs must be a three-mode vector.")
        if cutoff.reference_ground_energy_hartree is None or not np.isfinite(float(cutoff.reference_ground_energy_hartree)):
            raise ValueError("Paper-IV reference-cutoff ground energy is missing or non-finite.")
        if cutoff.delta_energy_hartree is None or not np.isfinite(float(cutoff.delta_energy_hartree)):
            raise ValueError("Paper-IV cutoff energy drift is missing or non-finite.")

    energy_tolerance = (
        H2O_LINEAR_FD_CUTOFF_ENERGY_TOLERANCE_HARTREE
        if cutoff.energy_tolerance_hartree is None
        else float(cutoff.energy_tolerance_hartree)
    )
    boundary_tolerance = (
        H2O_LINEAR_FD_CUTOFF_BOUNDARY_WEIGHT_TOLERANCE
        if cutoff.boundary_weight_tolerance is None
        else float(cutoff.boundary_weight_tolerance)
    )
    assessment = assess_h2o_linear_fd_cutoff_diagnostics(
        delta_energy_hartree=cutoff.delta_energy_hartree,
        work_boundary_weight=cutoff.work_boundary_weight,
        energy_tolerance_hartree=energy_tolerance,
        boundary_weight_tolerance=boundary_tolerance,
    )
    if bool(cutoff.passed) != bool(assessment["passed"]):
        raise ValueError("Paper-IV cutoff diagnostics pass flag disagrees with its quantitative thresholds.")
    if cutoff.energy_passed is not None and bool(cutoff.energy_passed) != bool(assessment["energy_passed"]):
        raise ValueError("Paper-IV cutoff energy pass flag disagrees with its quantitative threshold.")
    if cutoff.boundary_passed is not None and bool(cutoff.boundary_passed) != bool(assessment["boundary_passed"]):
        raise ValueError("Paper-IV cutoff boundary pass flag disagrees with its quantitative threshold.")
    if require_cutoff_converged and not bool(assessment["passed"]):
        raise ValueError("Paper-IV vibrational cutoff convergence criteria are not satisfied.")

    if require_pool and not fixture.pool:
        raise ValueError("Paper-IV H2O evidence requires a non-empty production pool.")
    for row in fixture.pool:
        label = str(row.get("label", "")).lower()
        generator_family = str(row.get("generator_family", "")).strip().lower()
        execution_mode = str(row.get("execution_mode", "termwise_product")).strip().lower()
        if (
            generator_family == "linear_vibronic_derivative_momentum"
            and execution_mode != "grouped_exact"
        ):
            raise ValueError(
                "Paper-IV derivative-momentum generators require grouped_exact "
                f"execution; got {execution_mode!r} for {label!r}."
            )
        legacy_marker = any(marker in label for marker in ("frontier", "surrogate", "active2", "smoke"))
        if legacy_marker and not bool(row.get("diagnostic_excluded_from_paper_evidence", False)):
            raise ValueError(f"legacy smoke/prototype pool label is not allowed in production evidence: {label!r}")


def _polynomial_from_operator_payload(payload: Mapping[str, Any], *, expected_nq: int, label: str) -> Any:
    try:
        return pauli_polynomial_from_jsonable(payload, expected_nq=int(expected_nq))
    except Exception as exc:
        raise ValueError(f"{label} could not be parsed as a PauliPolynomial: {exc}") from exc


def _canonicalized_symmetric_spectral_factors(
    matrix: np.ndarray,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[tuple[float, np.ndarray], ...]:
    """Return deterministic weighted rank-one factors of a real symmetric matrix."""

    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("spectral factorization requires a square matrix.")
    if not np.all(np.isfinite(array)):
        raise ValueError("spectral factorization matrix contains non-finite entries.")
    if float(absolute_tolerance) < 0.0 or float(relative_tolerance) < 0.0:
        raise ValueError("spectral factor tolerances must be non-negative.")

    symmetry_residual = float(np.linalg.norm(array - array.T))
    symmetry_scale = max(1.0, float(np.linalg.norm(array)))
    if symmetry_residual > 1.0e-8 * symmetry_scale:
        raise ValueError(
            "spectral factorization matrix is not symmetric; "
            f"residual={symmetry_residual:.3e}."
        )
    symmetric = 0.5 * (array + array.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
    threshold = max(
        float(absolute_tolerance),
        float(relative_tolerance) * scale,
    )
    order = sorted(
        range(int(eigenvalues.size)),
        key=lambda idx: (-abs(float(eigenvalues[idx])), -float(eigenvalues[idx]), idx),
    )
    factors: list[tuple[float, np.ndarray]] = []
    for idx in order:
        eigenvalue = float(eigenvalues[idx])
        if abs(eigenvalue) <= threshold:
            continue
        vector = np.asarray(eigenvectors[:, idx], dtype=float).copy()
        pivot = int(np.argmax(np.abs(vector)))
        if float(vector[pivot]) < 0.0:
            vector *= -1.0
        factors.append((eigenvalue, np.outer(vector, vector)))
    return tuple(factors)


def _symmetric_pair_basis(n_spatial_orbitals: int) -> np.ndarray:
    n_spatial = int(n_spatial_orbitals)
    if n_spatial <= 0:
        raise ValueError("n_spatial_orbitals must be positive.")
    rows: list[np.ndarray] = []
    for p in range(n_spatial):
        for q in range(p, n_spatial):
            row = np.zeros((n_spatial, n_spatial), dtype=float)
            if p == q:
                row[p, q] = 1.0
            else:
                row[p, q] = 1.0 / np.sqrt(2.0)
                row[q, p] = 1.0 / np.sqrt(2.0)
            rows.append(row)
    return np.asarray(rows, dtype=float)


def _chemist_eri_spectral_factors(
    tensor: np.ndarray,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[tuple[float, np.ndarray], ...]:
    """Factor a chemists'-notation ERI derivative in symmetric pair space."""

    array = np.asarray(tensor, dtype=float)
    if array.ndim != 4 or len(set(array.shape)) != 1:
        raise ValueError("two-body derivative must have shape (n,n,n,n).")
    if not np.all(np.isfinite(array)):
        raise ValueError("two-body derivative contains non-finite entries.")
    pair_basis = _symmetric_pair_basis(int(array.shape[0]))
    pair_matrix = np.einsum(
        "apq,pqrs,brs->ab",
        pair_basis,
        array,
        pair_basis,
        optimize=True,
    )
    projected = np.einsum(
        "apq,ab,brs->pqrs",
        pair_basis,
        pair_matrix,
        pair_basis,
        optimize=True,
    )
    projection_residual = float(np.linalg.norm(array - projected))
    projection_scale = max(1.0, float(np.linalg.norm(array)))
    if projection_residual > 1.0e-8 * projection_scale:
        raise ValueError(
            "two-body derivative violates the required chemists'-ERI pair "
            f"symmetry; projection residual={projection_residual:.3e}."
        )
    pair_exchange_residual = float(np.linalg.norm(pair_matrix - pair_matrix.T))
    if pair_exchange_residual > 1.0e-8 * projection_scale:
        raise ValueError(
            "two-body derivative violates chemists'-ERI pair-exchange "
            f"symmetry; residual={pair_exchange_residual:.3e}."
        )
    pair_factors = _canonicalized_symmetric_spectral_factors(
        pair_matrix,
        absolute_tolerance=float(absolute_tolerance),
        relative_tolerance=float(relative_tolerance),
    )
    factors: list[tuple[float, np.ndarray]] = []
    for eigenvalue, pair_support in pair_factors:
        # pair_support = u u^T.  Recover u from its deterministic principal
        # column without introducing an arbitrary square-root sign.
        pivot = int(np.argmax(np.diag(pair_support)))
        pivot_norm = float(np.sqrt(max(0.0, pair_support[pivot, pivot])))
        if pivot_norm <= 0.0:
            continue
        pair_vector = np.asarray(pair_support[:, pivot], dtype=float) / pivot_norm
        if float(pair_vector[pivot]) < 0.0:
            pair_vector *= -1.0
        orbital_support = np.einsum(
            "A,Aij->ij",
            pair_vector,
            pair_basis,
            optimize=True,
        )
        factors.append(
            (
                float(eigenvalue),
                np.einsum(
                    "pq,rs->pqrs",
                    orbital_support,
                    orbital_support,
                    optimize=True,
                ),
            )
        )
    return tuple(factors)


def _h2o_linear_fd_derivative_problem(
    fixture: ProductionVibronicH2OFixture,
    *,
    one_body_integrals: np.ndarray,
    two_body_integrals: np.ndarray,
) -> RestrictedClosedShellMolecularProblem:
    active = fixture.active_space
    n_alpha, n_beta = active.num_particles
    return RestrictedClosedShellMolecularProblem(
        geometry_spec=str(fixture.geometry.geometry_id),
        basis=str(fixture.geometry.basis),
        charge=int(fixture.geometry.charge),
        multiplicity=int(fixture.geometry.multiplicity),
        reference=str(fixture.geometry.reference),
        n_spatial_orbitals=int(active.n_spatial_orbitals),
        n_alpha=int(n_alpha),
        n_beta=int(n_beta),
        hf_energy=0.0,
        nuclear_repulsion_energy=0.0,
        one_body_integrals_mo=np.asarray(one_body_integrals, dtype=float),
        two_body_integrals_mo=np.asarray(two_body_integrals, dtype=float),
    )


def build_h2o_linear_fd_derivative_resolved_pool_v2(
    fixture: ProductionVibronicH2OFixture,
    *,
    absolute_factor_tolerance: float = 1.0e-10,
    relative_factor_tolerance: float = 1.0e-10,
) -> tuple[AnsatzTerm, ...]:
    """Build an additive, chemically resolved H2O vibronic generator pool.

    The production ``full_meta`` pool remains the immutable baseline.  This
    opt-in pool retains every baseline generator and adds spectral one-/two-body
    components of D_mu P_mu plus Q_mu-conditioned UCCSD excitations.
    """

    model = build_production_vibronic_h2o_linear_fd_runtime_model(fixture)
    if len(model.mode_labels) != len(fixture.first_derivatives):
        raise ValueError("H2O derivative records do not match the encoded mode count.")
    if len(model.q_by_mode) != len(model.mode_labels) or len(model.p_by_mode) != len(
        model.mode_labels
    ):
        raise ValueError("H2O fixture is missing encoded Q/P operators for a mode.")

    n_spatial = int(model.n_spatial_orbitals)
    zeros_one = np.zeros((n_spatial, n_spatial), dtype=float)
    zeros_two = np.zeros((n_spatial,) * 4, dtype=float)
    derivative_by_label = {
        str(record.mode_label): record for record in fixture.first_derivatives
    }
    if len(derivative_by_label) != len(fixture.first_derivatives):
        raise ValueError("H2O derivative mode labels must be unique.")

    terms: list[AnsatzTerm] = list(model.pool)
    electronic_terms = build_molecular_uccsd_pool(
        n_spatial_orbitals=n_spatial,
        num_particles=tuple(int(value) for value in model.num_particles),
        ordering="blocked",
    )
    for mode_index, mode_label in enumerate(model.mode_labels):
        derivative = derivative_by_label.get(str(mode_label))
        if derivative is None:
            raise ValueError(f"H2O fixture is missing derivative data for mode {mode_label!r}.")
        p_operator = model.p_by_mode[mode_index]
        q_operator = model.q_by_mode[mode_index]

        one_body_factors = _canonicalized_symmetric_spectral_factors(
            np.asarray(derivative.one_body_derivative, dtype=float),
            absolute_tolerance=float(absolute_factor_tolerance),
            relative_tolerance=float(relative_factor_tolerance),
        )
        for rank, (weight, support) in enumerate(one_body_factors):
            factor_problem = _h2o_linear_fd_derivative_problem(
                fixture,
                one_body_integrals=float(weight) * np.asarray(support, dtype=float),
                two_body_integrals=zeros_two,
            )
            factor_polynomial = _lift_fermion_polynomial(
                build_one_body_jw_polynomial(factor_problem),
                boson_qubits=int(model.n_boson_qubits),
            )
            coupled = _clean_real_polynomial(factor_polynomial * p_operator)
            if coupled.return_polynomial():
                terms.append(
                    AnsatzTerm(
                        label=(
                            f"coupled::{mode_label}::"
                            f"dH_dQ_one_body_factor[{rank}]_times_p"
                        ),
                        polynomial=coupled,
                        execution_mode="grouped_exact",
                    )
                )

        two_body_factors = _chemist_eri_spectral_factors(
            np.asarray(derivative.two_body_derivative, dtype=float),
            absolute_tolerance=float(absolute_factor_tolerance),
            relative_tolerance=float(relative_factor_tolerance),
        )
        for rank, (weight, support) in enumerate(two_body_factors):
            factor_problem = _h2o_linear_fd_derivative_problem(
                fixture,
                one_body_integrals=zeros_one,
                two_body_integrals=float(weight) * np.asarray(support, dtype=float),
            )
            factor_polynomial = _lift_fermion_polynomial(
                build_two_body_jw_polynomial(factor_problem),
                boson_qubits=int(model.n_boson_qubits),
            )
            coupled = _clean_real_polynomial(factor_polynomial * p_operator)
            if coupled.return_polynomial():
                terms.append(
                    AnsatzTerm(
                        label=(
                            f"coupled::{mode_label}::"
                            f"dH_dQ_two_body_factor[{rank}]_times_p"
                        ),
                        polynomial=coupled,
                        execution_mode="grouped_exact",
                    )
                )

        for electronic_term in electronic_terms:
            lifted_electronic = _lift_fermion_polynomial(
                electronic_term.polynomial,
                boson_qubits=int(model.n_boson_qubits),
            )
            conditional = _clean_real_polynomial(q_operator * lifted_electronic)
            if conditional.return_polynomial():
                terms.append(
                    AnsatzTerm(
                        label=(
                            f"conditional::{mode_label}::q_times_"
                            f"{electronic_term.label}"
                        ),
                        polynomial=conditional,
                        execution_mode="grouped_exact",
                    )
                )

    labels = [str(term.label) for term in terms]
    if len(labels) != len(set(labels)):
        raise ValueError("Derivative-resolved H2O pool produced duplicate labels.")
    return tuple(terms)


def build_production_vibronic_h2o_linear_fd_runtime_model(
    fixture: ProductionVibronicH2OFixture,
    *,
    require_paper_iv_evidence: bool = False,
    require_exact_state: bool = False,
    require_reference_cutoff: bool = True,
    require_cutoff_converged: bool = False,
) -> ProductionVibronicH2OLinearFDRuntimeModel:
    if require_paper_iv_evidence:
        validate_paper_iv_h2o_linear_fd_evidence_fixture(
            fixture,
            require_exact_state=require_exact_state,
            require_reference_cutoff=require_reference_cutoff,
            require_cutoff_converged=require_cutoff_converged,
        )
    else:
        validate_production_vibronic_h2o_fixture(fixture)

    layout = fixture.layout
    encoded = fixture.encoded_operators
    n_fermion = int(layout.n_fermion_qubits)
    n_total = int(layout.n_total_qubits)
    h_electronic = _polynomial_from_operator_payload(
        encoded.h_electronic,
        expected_nq=n_fermion,
        label="encoded h_electronic",
    )
    dH_dQ_by_mode = tuple(
        _polynomial_from_operator_payload(row, expected_nq=n_fermion, label=f"encoded dH_dQ_by_mode[{idx}]")
        for idx, row in enumerate(encoded.dH_dQ_by_mode)
    )
    h_vibronic = _polynomial_from_operator_payload(
        encoded.h_vibronic,
        expected_nq=n_total,
        label="encoded h_vibronic",
    )
    q_by_mode = tuple(
        _polynomial_from_operator_payload(row, expected_nq=n_total, label=f"encoded q_by_mode[{idx}]")
        for idx, row in enumerate(encoded.q_by_mode)
    )
    p_by_mode = tuple(
        _polynomial_from_operator_payload(row, expected_nq=n_total, label=f"encoded p_by_mode[{idx}]")
        for idx, row in enumerate(encoded.p_by_mode)
    )
    n_by_mode = tuple(
        _polynomial_from_operator_payload(row, expected_nq=n_total, label=f"encoded n_by_mode[{idx}]")
        for idx, row in enumerate(encoded.n_by_mode)
    )
    pool_terms: list[AnsatzTerm] = []
    for idx, row in enumerate(fixture.pool):
        obj = _as_mapping(row, label=f"pool[{idx}]")
        label = str(obj.get("label", ""))
        if not label:
            raise ValueError(f"pool[{idx}] missing label.")
        polynomial_payload = _as_mapping(obj.get("polynomial"), label=f"pool[{idx}].polynomial")
        pool_terms.append(
            AnsatzTerm(
                label=label,
                polynomial=_polynomial_from_operator_payload(
                    polynomial_payload,
                    expected_nq=n_total,
                    label=f"pool[{idx}].polynomial",
                ),
                execution_mode=str(obj.get("execution_mode", "termwise_product")),
            )
        )

    psi_ref = np.asarray(fixture.reference_state, dtype=complex).reshape(-1)
    return ProductionVibronicH2OLinearFDRuntimeModel(
        h_vibronic=h_vibronic,
        h_electronic=h_electronic,
        dH_dQ_by_mode=dH_dQ_by_mode,
        q_by_mode=q_by_mode,
        p_by_mode=p_by_mode,
        n_by_mode=n_by_mode,
        pool=tuple(pool_terms),
        psi_ref=psi_ref,
        n_spatial_orbitals=int(fixture.active_space.n_spatial_orbitals),
        num_particles=tuple(fixture.active_space.num_particles),
        n_fermion_qubits=int(layout.n_fermion_qubits),
        n_boson_qubits=int(layout.n_boson_qubits),
        n_total_qubits=int(layout.n_total_qubits),
        mode_labels=tuple(block.mode_label for block in layout.boson_modes),
        mode_cutoffs=tuple(int(block.n_ph_max) for block in layout.boson_modes),
    )


def load_cached_production_vibronic_h2o_linear_fd_fixture(
    path: str | Path,
    *,
    require_exact_state: bool = False,
    require_reference_cutoff: bool = True,
    require_cutoff_converged: bool = False,
) -> CachedProductionVibronicH2OLinearFDFixture:
    fixture_path = Path(path)
    fixture = load_production_vibronic_h2o_fixture(fixture_path)
    validate_paper_iv_h2o_linear_fd_evidence_fixture(
        fixture,
        require_exact_state=require_exact_state,
        require_reference_cutoff=require_reference_cutoff,
        require_cutoff_converged=require_cutoff_converged,
    )
    model = build_production_vibronic_h2o_linear_fd_runtime_model(
        fixture,
        require_paper_iv_evidence=False,
    )
    return CachedProductionVibronicH2OLinearFDFixture(
        fixture=fixture,
        model=model,
        fixture_path=fixture_path,
        metadata={
            "schema": str(fixture.manifest.schema),
            "family_key": str(fixture.manifest.family_key),
            "model_role": str(fixture.manifest.model_role),
            "production_status": str(fixture.manifest.production_status),
            "derivative_source": str(fixture.manifest.derivative_source),
            "mode_labels": tuple(model.mode_labels),
            "mode_cutoffs": tuple(model.mode_cutoffs),
            "vibrational_cutoff_converged": (
                None
                if fixture.cutoff_diagnostics is None
                else bool(fixture.cutoff_diagnostics.passed)
            ),
            "cutoff_delta_energy_hartree": (
                None
                if fixture.cutoff_diagnostics is None
                else fixture.cutoff_diagnostics.delta_energy_hartree
            ),
            "cutoff_total_boundary_weight": (
                None
                if fixture.cutoff_diagnostics is None
                or fixture.cutoff_diagnostics.work_boundary_weight is None
                else float(
                    fixture.cutoff_diagnostics.work_boundary_weight.total_boundary_weight
                )
            ),
        },
    )


def _decode_mode_occupation(index: int, block: BosonModeRegister) -> int:
    mask = (1 << int(block.n_qubits)) - 1
    return int((int(index) >> int(block.qubit_start)) & mask)


def h2o_linear_fd_boundary_weight_for_state(
    state: np.ndarray,
    *,
    layout: RegisterLayout,
    state_source: str,
    tol: float = 1.0e-14,
    reject_invalid_boson_code: bool = True,
) -> BoundaryWeightRecord:
    """Return total/per-mode boundary weights over valid boson occupations.

    Binary mode registers may contain padded computational states above
    ``n_ph_max``.  Those states are outside the physical oscillator subspace and
    are never counted as boundary occupation.  By default, nonzero amplitude on
    such states is a hard error for paper-facing diagnostics.
    """

    arr = np.asarray(state, dtype=complex).reshape(-1)
    expected_dim = 2 ** int(layout.n_total_qubits)
    if arr.shape != (expected_dim,):
        raise ValueError("state length does not match H2O linear-FD register layout.")
    per_mode = {str(block.mode_label): 0.0 for block in layout.boson_modes}
    total = 0.0
    invalid_weight = 0.0
    for index, amp in enumerate(arr):
        weight = float(abs(complex(amp)) ** 2)
        if weight <= float(tol):
            continue
        hit_boundary = False
        for block in layout.boson_modes:
            occupation = _decode_mode_occupation(index, block)
            if occupation > int(block.n_ph_max):
                invalid_weight += weight
                continue
            if occupation == int(block.n_ph_max):
                per_mode[str(block.mode_label)] += weight
                hit_boundary = True
        if hit_boundary:
            total += weight
    if invalid_weight > float(tol) and reject_invalid_boson_code:
        raise ValueError("state has nonzero amplitude on invalid padded binary boson code states.")
    return BoundaryWeightRecord(
        total_boundary_weight=float(total),
        per_mode_boundary_weight={label: float(value) for label, value in per_mode.items()},
        state_source=str(state_source),
    )


def production_vibronic_h2o_fixture_to_jsonable(fixture: ProductionVibronicH2OFixture) -> dict[str, Any]:
    return {
        "manifest": _manifest_to_json(fixture.manifest),
        "geometry": _geometry_to_json(fixture.geometry),
        "normal_modes": [_normal_mode_to_json(row) for row in fixture.normal_modes],
        "displacements": [_displacement_to_json(row) for row in fixture.displacements],
        "active_space": _active_space_to_json(fixture.active_space),
        "aligned_active_tensors": [_aligned_tensor_to_json(row) for row in fixture.aligned_tensors],
        "alignment_diagnostics": [_alignment_diag_to_json(row) for row in fixture.alignment_diagnostics],
        "first_derivative_records": [_first_derivative_to_json(row) for row in fixture.first_derivatives],
        "register_layout": _layout_to_json(fixture.layout),
        "physical_sector": _physical_sector_to_json(fixture.physical_sector),
        "encoded_operators": _encoded_operators_to_json(fixture.encoded_operators),
        "reference_state": _array_payload(np.asarray(fixture.reference_state, dtype=complex)),
        "exact_reference": _exact_reference_to_json(fixture.exact_reference),
        "cutoff_diagnostics": (
            None if fixture.cutoff_diagnostics is None else _cutoff_diagnostics_to_json(fixture.cutoff_diagnostics)
        ),
        "evidence_hooks": _evidence_hooks_to_json(fixture.evidence_hooks),
        "pool": [dict(row) for row in fixture.pool],
        "report_summary": dict(fixture.report_summary),
        "provenance": dict(fixture.provenance),
    }


def production_vibronic_h2o_fixture_from_jsonable(payload: Mapping[str, Any]) -> ProductionVibronicH2OFixture:
    obj = _as_mapping(payload, label="production fixture")
    schema = obj.get("schema")
    if schema is None and isinstance(obj.get("manifest"), Mapping):
        schema = obj["manifest"].get("schema")
    if schema != H2O_LINEAR_FD_FIXTURE_SCHEMA:
        raise ValueError(f"not a production H2O linear-FD fixture: {schema!r}")
    manifest = _manifest_from_json(obj.get("manifest"))
    fixture = ProductionVibronicH2OFixture(
        manifest=manifest,
        geometry=_geometry_from_json(obj.get("geometry")),
        normal_modes=tuple(_normal_mode_from_json(row) for row in _list_payload(obj.get("normal_modes"), "normal_modes")),
        displacements=tuple(
            _displacement_from_json(row) for row in _list_payload(obj.get("displacements", []), "displacements")
        ),
        active_space=_active_space_from_json(obj.get("active_space")),
        aligned_tensors=tuple(
            _aligned_tensor_from_json(row)
            for row in _list_payload(obj.get("aligned_active_tensors", []), "aligned_active_tensors")
        ),
        alignment_diagnostics=tuple(
            _alignment_diag_from_json(row)
            for row in _list_payload(obj.get("alignment_diagnostics"), "alignment_diagnostics")
        ),
        first_derivatives=tuple(
            _first_derivative_from_json(row)
            for row in _list_payload(obj.get("first_derivative_records"), "first_derivative_records")
        ),
        layout=_layout_from_json(obj.get("register_layout")),
        physical_sector=_physical_sector_from_json(obj.get("physical_sector")),
        encoded_operators=_encoded_operators_from_json(obj.get("encoded_operators")),
        reference_state=_array_from_payload(obj.get("reference_state"), label="reference_state"),
        exact_reference=_exact_reference_from_json(obj.get("exact_reference")),
        cutoff_diagnostics=(
            None if obj.get("cutoff_diagnostics") is None else _cutoff_diagnostics_from_json(obj.get("cutoff_diagnostics"))
        ),
        evidence_hooks=_evidence_hooks_from_json(obj.get("evidence_hooks", {})),
        pool=tuple(dict(row) for row in _list_payload(obj.get("pool", []), "pool")),
        report_summary=dict(_as_mapping(obj.get("report_summary", {}), label="report_summary")),
        provenance=dict(_as_mapping(obj.get("provenance", {}), label="provenance")),
    )
    return fixture


def load_production_vibronic_h2o_fixture(path: str | Path) -> ProductionVibronicH2OFixture:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    fixture = production_vibronic_h2o_fixture_from_jsonable(payload)
    validate_production_vibronic_h2o_fixture(fixture)
    return fixture


def write_production_vibronic_h2o_fixture(path: str | Path, fixture: ProductionVibronicH2OFixture) -> None:
    validate_production_vibronic_h2o_fixture(fixture)
    Path(path).write_text(
        json.dumps(production_vibronic_h2o_fixture_to_jsonable(fixture), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _list_payload(payload: Any, label: str) -> list[Any]:
    if not isinstance(payload, list):
        raise ValueError(f"{label} must be a JSON array.")
    return payload


def _manifest_to_json(row: FixtureManifest) -> dict[str, Any]:
    return {
        "schema": str(row.schema),
        "schema_version": int(row.schema_version),
        "family_key": str(row.family_key),
        "molecule_family_key": str(row.molecule_family_key),
        "model_role": str(row.model_role),
        "production_status": str(row.production_status),
        "derivative_source": str(row.derivative_source),
        "created_utc": str(row.created_utc),
        "generator_version": str(row.generator_version),
        "repository_commit": row.repository_commit,
        "provenance_hashes": dict(row.provenance_hashes),
    }


def _manifest_from_json(payload: Any) -> FixtureManifest:
    obj = _as_mapping(payload, label="manifest")
    return FixtureManifest(
        schema=str(obj["schema"]),
        schema_version=int(obj.get("schema_version", 1)),
        family_key=str(obj["family_key"]),
        molecule_family_key=str(obj.get("molecule_family_key", H2O_UMBRELLA_FAMILY_KEY)),
        model_role=str(obj["model_role"]),
        production_status=str(obj["production_status"]),  # type: ignore[arg-type]
        derivative_source=str(obj["derivative_source"]),
        created_utc=str(obj.get("created_utc", "")),
        generator_version=str(obj.get("generator_version", "")),
        repository_commit=None if obj.get("repository_commit") is None else str(obj.get("repository_commit")),
        provenance_hashes={
            str(k): str(v) for k, v in _as_mapping(obj.get("provenance_hashes", {}), label="provenance_hashes").items()
        },
    )


def _geometry_to_json(row: GeometryRecord) -> dict[str, Any]:
    return {
        "geometry_id": row.geometry_id,
        "molecule": row.molecule,
        "symbols": list(row.symbols),
        "coordinates_bohr": _array_payload(row.coordinates_bohr, units="bohr"),
        "masses_me": _array_payload(row.masses_me, units="electron_mass"),
        "charge": int(row.charge),
        "multiplicity": int(row.multiplicity),
        "method": row.method,
        "basis": row.basis,
        "reference": row.reference,
        "optimized": bool(row.optimized),
        "coordinate_units": row.coordinate_units,
        "mass_units": row.mass_units,
        "symmetry": dict(row.symmetry),
        "provenance": dict(row.provenance),
    }


def _geometry_from_json(payload: Any) -> GeometryRecord:
    obj = _as_mapping(payload, label="geometry")
    return GeometryRecord(
        geometry_id=str(obj["geometry_id"]),
        molecule=str(obj["molecule"]),
        symbols=tuple(str(v) for v in obj["symbols"]),
        coordinates_bohr=_array_from_payload(obj["coordinates_bohr"], label="geometry.coordinates_bohr"),
        masses_me=_array_from_payload(obj["masses_me"], label="geometry.masses_me"),
        charge=int(obj["charge"]),
        multiplicity=int(obj["multiplicity"]),
        method=str(obj["method"]),
        basis=str(obj["basis"]),
        reference=str(obj["reference"]),
        optimized=bool(obj["optimized"]),
        coordinate_units=str(obj.get("coordinate_units", "bohr")),
        mass_units=str(obj.get("mass_units", "electron_mass")),
        symmetry=dict(_as_mapping(obj.get("symmetry", {}), label="geometry.symmetry")),
        provenance=dict(_as_mapping(obj.get("provenance", {}), label="geometry.provenance")),
    )


def _normal_mode_to_json(row: NormalModeRecord) -> dict[str, Any]:
    return {
        "mode_index": int(row.mode_index),
        "label": row.label,
        "frequency_hartree": float(row.frequency_hartree),
        "frequency_cm1": row.frequency_cm1,
        "mass_weighted_eigenvector": _array_payload(row.mass_weighted_eigenvector, convention=row.normalization),
        "q_step_au": float(row.q_step_au),
        "q_step_alt_au": row.q_step_alt_au,
        "normalization": row.normalization,
        "coordinate_convention": row.coordinate_convention,
        "ladder_convention": row.ladder_convention,
        "trans_rot_residual": row.trans_rot_residual,
    }


def _normal_mode_from_json(payload: Any) -> NormalModeRecord:
    obj = _as_mapping(payload, label="normal_mode")
    return NormalModeRecord(
        mode_index=int(obj["mode_index"]),
        label=str(obj["label"]),
        frequency_hartree=float(obj["frequency_hartree"]),
        frequency_cm1=None if obj.get("frequency_cm1") is None else float(obj["frequency_cm1"]),
        mass_weighted_eigenvector=_array_from_payload(
            obj["mass_weighted_eigenvector"], label="normal_mode.mass_weighted_eigenvector"
        ),
        q_step_au=float(obj["q_step_au"]),
        q_step_alt_au=None if obj.get("q_step_alt_au") is None else float(obj["q_step_alt_au"]),
        normalization=str(obj.get("normalization", "sum_Aalpha e_mu_Aalpha e_nu_Aalpha = delta_munu")),
        coordinate_convention=str(obj.get("coordinate_convention", "mass_weighted_normal_au")),
        ladder_convention=str(obj.get("ladder_convention", "Q=(2*omega)^(-1/2)*(a_dag+a)")),
        trans_rot_residual=None if obj.get("trans_rot_residual") is None else float(obj["trans_rot_residual"]),
    )


def _displacement_to_json(row: DisplacedGeometryRecord) -> dict[str, Any]:
    return {
        "displacement_id": row.displacement_id,
        "purpose": row.purpose,
        "mode_indices": list(row.mode_indices),
        "signs": list(row.signs),
        "q_displacements_au": list(row.q_displacements_au),
        "geometry_id": row.geometry_id,
        "snapshot_id": row.snapshot_id,
        "coordinates_bohr": _array_payload(row.coordinates_bohr, units="bohr"),
    }


def _displacement_from_json(payload: Any) -> DisplacedGeometryRecord:
    obj = _as_mapping(payload, label="displacement")
    return DisplacedGeometryRecord(
        displacement_id=str(obj["displacement_id"]),
        purpose=str(obj["purpose"]),
        mode_indices=_as_tuple_int(obj["mode_indices"], label="displacement.mode_indices"),
        signs=_as_tuple_int(obj["signs"], label="displacement.signs"),
        q_displacements_au=_as_tuple_float(obj["q_displacements_au"], label="displacement.q_displacements_au"),
        geometry_id=str(obj["geometry_id"]),
        snapshot_id=None if obj.get("snapshot_id") is None else str(obj["snapshot_id"]),
        coordinates_bohr=_array_from_payload(obj["coordinates_bohr"], label="displacement.coordinates_bohr"),
    )


def _active_space_to_json(row: ActiveSpaceRecord) -> dict[str, Any]:
    return {
        "active_space_kind": row.active_space_kind,
        "frozen_core_indices": list(row.frozen_core_indices),
        "active_indices_center": list(row.active_indices_center),
        "external_indices": list(row.external_indices),
        "n_spatial_orbitals": int(row.n_spatial_orbitals),
        "num_particles": list(row.num_particles),
        "scalar_energy_hartree": float(row.scalar_energy_hartree),
        "one_body_integrals": _array_payload(row.one_body_integrals),
        "two_body_integrals": _array_payload(row.two_body_integrals, convention=row.tensor_convention),
        "orbital_character": dict(row.orbital_character),
        "frozen_core_convention": row.frozen_core_convention,
        "tensor_convention": row.tensor_convention,
        "spin_orbital_ordering": row.spin_orbital_ordering,
    }


def _active_space_from_json(payload: Any) -> ActiveSpaceRecord:
    obj = _as_mapping(payload, label="active_space")
    return ActiveSpaceRecord(
        active_space_kind=str(obj["active_space_kind"]),
        frozen_core_indices=_as_tuple_int(obj.get("frozen_core_indices", ()), label="active_space.frozen_core_indices"),
        active_indices_center=_as_tuple_int(obj["active_indices_center"], label="active_space.active_indices_center"),
        external_indices=_as_tuple_int(obj.get("external_indices", ()), label="active_space.external_indices"),
        n_spatial_orbitals=int(obj["n_spatial_orbitals"]),
        num_particles=tuple(_as_tuple_int(obj["num_particles"], label="active_space.num_particles"))[:2],  # type: ignore[arg-type]
        scalar_energy_hartree=float(obj["scalar_energy_hartree"]),
        one_body_integrals=_array_from_payload(obj["one_body_integrals"], label="active_space.one_body_integrals"),
        two_body_integrals=_array_from_payload(obj["two_body_integrals"], label="active_space.two_body_integrals"),
        orbital_character=dict(_as_mapping(obj.get("orbital_character", {}), label="active_space.orbital_character")),
        frozen_core_convention=str(obj.get("frozen_core_convention", "closed_shell_core_contraction")),
        tensor_convention=str(obj.get("tensor_convention", "chemist_eri_pqrs")),
        spin_orbital_ordering=str(obj.get("spin_orbital_ordering", "blocked")),  # type: ignore[arg-type]
    )


def _aligned_tensor_to_json(row: AlignedActiveTensorRecord) -> dict[str, Any]:
    return {
        "aligned_tensor_id": row.aligned_tensor_id,
        "source_snapshot_id": row.source_snapshot_id,
        "displacement_id": row.displacement_id,
        "scalar_energy_hartree": float(row.scalar_energy_hartree),
        "one_body_integrals": _array_payload(row.one_body_integrals),
        "two_body_integrals": _array_payload(row.two_body_integrals, convention=row.tensor_convention),
        "alignment_id": row.alignment_id,
        "tensor_convention": row.tensor_convention,
    }


def _aligned_tensor_from_json(payload: Any) -> AlignedActiveTensorRecord:
    obj = _as_mapping(payload, label="aligned_active_tensor")
    return AlignedActiveTensorRecord(
        aligned_tensor_id=str(obj["aligned_tensor_id"]),
        source_snapshot_id=str(obj["source_snapshot_id"]),
        displacement_id=None if obj.get("displacement_id") is None else str(obj["displacement_id"]),
        scalar_energy_hartree=float(obj["scalar_energy_hartree"]),
        one_body_integrals=_array_from_payload(obj["one_body_integrals"], label="aligned_active_tensor.one_body_integrals"),
        two_body_integrals=_array_from_payload(obj["two_body_integrals"], label="aligned_active_tensor.two_body_integrals"),
        alignment_id=None if obj.get("alignment_id") is None else str(obj["alignment_id"]),
        tensor_convention=str(obj.get("tensor_convention", "chemist_eri_pqrs")),
    )


def _thresholds_to_json(row: AlignmentThresholds) -> dict[str, Any]:
    return {
        "min_active_singular_value": float(row.min_active_singular_value),
        "max_active_residual_fro": float(row.max_active_residual_fro),
        "max_active_to_external_leakage_fro": float(row.max_active_to_external_leakage_fro),
        "max_hermiticity_residual": float(row.max_hermiticity_residual),
        "max_eri_symmetry_residual": float(row.max_eri_symmetry_residual),
    }


def _thresholds_from_json(payload: Any) -> AlignmentThresholds:
    obj = _as_mapping(payload or {}, label="alignment_thresholds")
    return AlignmentThresholds(
        min_active_singular_value=float(obj.get("min_active_singular_value", 0.98)),
        max_active_residual_fro=float(obj.get("max_active_residual_fro", 1.0e-6)),
        max_active_to_external_leakage_fro=float(obj.get("max_active_to_external_leakage_fro", 1.0e-2)),
        max_hermiticity_residual=float(obj.get("max_hermiticity_residual", 1.0e-10)),
        max_eri_symmetry_residual=float(obj.get("max_eri_symmetry_residual", 1.0e-8)),
    )


def _alignment_diag_to_json(row: AlignmentDiagnosticsRecord) -> dict[str, Any]:
    return {
        "alignment_id": row.alignment_id,
        "center_snapshot_id": row.center_snapshot_id,
        "displaced_snapshot_id": row.displaced_snapshot_id,
        "displacement_id": row.displacement_id,
        "block": row.block,
        "singular_values": _array_payload(row.singular_values),
        "min_singular_value": float(row.min_singular_value),
        "alignment_residual_fro": float(row.alignment_residual_fro),
        "active_to_external_leakage_fro": row.active_to_external_leakage_fro,
        "external_to_active_leakage_fro": row.external_to_active_leakage_fro,
        "hermiticity_residual": float(row.hermiticity_residual),
        "eri_symmetry_residual": float(row.eri_symmetry_residual),
        "rotation_orthogonality_residual": float(row.rotation_orthogonality_residual),
        "thresholds": _thresholds_to_json(row.thresholds),
        "passed": bool(row.passed),
        "warnings": list(row.warnings),
    }


def _alignment_diag_from_json(payload: Any) -> AlignmentDiagnosticsRecord:
    obj = _as_mapping(payload, label="alignment_diagnostic")
    return AlignmentDiagnosticsRecord(
        alignment_id=str(obj["alignment_id"]),
        center_snapshot_id=str(obj["center_snapshot_id"]),
        displaced_snapshot_id=str(obj["displaced_snapshot_id"]),
        displacement_id=str(obj["displacement_id"]),
        block=str(obj.get("block", "active")),
        singular_values=_array_from_payload(obj["singular_values"], label="alignment_diagnostic.singular_values"),
        min_singular_value=float(obj["min_singular_value"]),
        alignment_residual_fro=float(obj["alignment_residual_fro"]),
        active_to_external_leakage_fro=(
            None if obj.get("active_to_external_leakage_fro") is None else float(obj["active_to_external_leakage_fro"])
        ),
        external_to_active_leakage_fro=(
            None if obj.get("external_to_active_leakage_fro") is None else float(obj["external_to_active_leakage_fro"])
        ),
        hermiticity_residual=float(obj["hermiticity_residual"]),
        eri_symmetry_residual=float(obj["eri_symmetry_residual"]),
        rotation_orthogonality_residual=float(obj["rotation_orthogonality_residual"]),
        thresholds=_thresholds_from_json(obj.get("thresholds", {})),
        passed=bool(obj["passed"]),
        warnings=tuple(str(v) for v in obj.get("warnings", ())),
    )


def _norms_to_json(row: DerivativeNorms | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "scalar_abs": float(row.scalar_abs),
        "one_body_fro": float(row.one_body_fro),
        "two_body_fro": float(row.two_body_fro),
        "pauli_l1": row.pauli_l1,
        "sector_spectral_norm": row.sector_spectral_norm,
        "low_energy_norm": row.low_energy_norm,
    }


def _norms_from_json(payload: Any) -> DerivativeNorms | None:
    if payload is None:
        return None
    obj = _as_mapping(payload, label="derivative_norms")
    return DerivativeNorms(
        scalar_abs=float(obj["scalar_abs"]),
        one_body_fro=float(obj["one_body_fro"]),
        two_body_fro=float(obj["two_body_fro"]),
        pauli_l1=None if obj.get("pauli_l1") is None else float(obj["pauli_l1"]),
        sector_spectral_norm=None if obj.get("sector_spectral_norm") is None else float(obj["sector_spectral_norm"]),
        low_energy_norm=None if obj.get("low_energy_norm") is None else float(obj["low_energy_norm"]),
    )


def _first_derivative_to_json(row: FirstDerivativeRecord) -> dict[str, Any]:
    return {
        "derivative_id": row.derivative_id,
        "mode_index": int(row.mode_index),
        "mode_label": row.mode_label,
        "q_step_au": float(row.q_step_au),
        "plus_aligned_tensor_id": row.plus_aligned_tensor_id,
        "minus_aligned_tensor_id": row.minus_aligned_tensor_id,
        "scalar_derivative_hartree_per_q": float(row.scalar_derivative_hartree_per_q),
        "one_body_derivative": _array_payload(row.one_body_derivative),
        "two_body_derivative": _array_payload(row.two_body_derivative),
        "scalar_derivative_included": bool(row.scalar_derivative_included),
        "scalar_derivative_convention": row.scalar_derivative_convention,
        "derivative_source": row.derivative_source,
        "pauli_operator": None if row.pauli_operator is None else dict(row.pauli_operator),
        "norms": _norms_to_json(row.norms),
        "finite_difference_drift": row.finite_difference_drift,
        "finite_difference_diagnostics": dict(row.finite_difference_diagnostics),
        "active_equilibrium_force": row.active_equilibrium_force,
        "passed": bool(row.passed),
        "warnings": list(row.warnings),
    }


def _first_derivative_from_json(payload: Any) -> FirstDerivativeRecord:
    obj = _as_mapping(payload, label="first_derivative_record")
    return FirstDerivativeRecord(
        derivative_id=str(obj["derivative_id"]),
        mode_index=int(obj["mode_index"]),
        mode_label=str(obj["mode_label"]),
        q_step_au=float(obj["q_step_au"]),
        plus_aligned_tensor_id=str(obj["plus_aligned_tensor_id"]),
        minus_aligned_tensor_id=str(obj["minus_aligned_tensor_id"]),
        scalar_derivative_hartree_per_q=float(obj["scalar_derivative_hartree_per_q"]),
        one_body_derivative=_array_from_payload(obj["one_body_derivative"], label="first_derivative.one_body_derivative"),
        two_body_derivative=_array_from_payload(obj["two_body_derivative"], label="first_derivative.two_body_derivative"),
        scalar_derivative_included=bool(obj["scalar_derivative_included"]),
        scalar_derivative_convention=str(obj["scalar_derivative_convention"]),
        derivative_source=str(obj.get("derivative_source", H2O_LINEAR_FD_DERIVATIVE_SOURCE)),
        pauli_operator=None if obj.get("pauli_operator") is None else dict(_as_mapping(obj["pauli_operator"], label="pauli_operator")),
        norms=_norms_from_json(obj.get("norms")),
        finite_difference_drift=(
            None if obj.get("finite_difference_drift") is None else float(obj["finite_difference_drift"])
        ),
        finite_difference_diagnostics=dict(
            _as_mapping(
                obj.get("finite_difference_diagnostics", {}),
                label="first_derivative.finite_difference_diagnostics",
            )
        ),
        active_equilibrium_force=(
            None if obj.get("active_equilibrium_force") is None else float(obj["active_equilibrium_force"])
        ),
        passed=bool(obj["passed"]),
        warnings=tuple(str(v) for v in obj.get("warnings", ())),
    )


def _layout_to_json(row: RegisterLayout) -> dict[str, Any]:
    return {
        "n_fermion_qubits": int(row.n_fermion_qubits),
        "fermion_qubits": list(row.fermion_qubits),
        "boson_modes": [
            {
                "mode_index": int(block.mode_index),
                "mode_label": block.mode_label,
                "qubit_start": int(block.qubit_start),
                "n_qubits": int(block.n_qubits),
                "n_ph_max": int(block.n_ph_max),
                "encoding": block.encoding,
            }
            for block in row.boson_modes
        ],
        "spin_orbital_ordering": row.spin_orbital_ordering,
        "n_total_qubits": int(row.n_total_qubits),
    }


def _layout_from_json(payload: Any) -> RegisterLayout:
    obj = _as_mapping(payload, label="register_layout")
    return RegisterLayout(
        n_fermion_qubits=int(obj["n_fermion_qubits"]),
        fermion_qubits=_as_tuple_int(obj["fermion_qubits"], label="register_layout.fermion_qubits"),
        boson_modes=tuple(
            BosonModeRegister(
                mode_index=int(block["mode_index"]),
                mode_label=str(block["mode_label"]),
                qubit_start=int(block["qubit_start"]),
                n_qubits=int(block["n_qubits"]),
                n_ph_max=int(block["n_ph_max"]),
                encoding=str(block.get("encoding", "binary")),  # type: ignore[arg-type]
            )
            for block in _list_payload(obj["boson_modes"], "register_layout.boson_modes")
        ),
        spin_orbital_ordering=str(obj.get("spin_orbital_ordering", "blocked")),  # type: ignore[arg-type]
    )


def _physical_sector_to_json(row: PhysicalSectorRecord) -> dict[str, Any]:
    return {
        "n_alpha": int(row.n_alpha),
        "n_beta": int(row.n_beta),
        "n_ph_max_by_mode": list(row.n_ph_max_by_mode),
        "mode_labels": list(row.mode_labels),
    }


def _physical_sector_from_json(payload: Any) -> PhysicalSectorRecord:
    obj = _as_mapping(payload, label="physical_sector")
    return PhysicalSectorRecord(
        n_alpha=int(obj["n_alpha"]),
        n_beta=int(obj["n_beta"]),
        n_ph_max_by_mode=_as_tuple_int(obj["n_ph_max_by_mode"], label="physical_sector.n_ph_max_by_mode"),
        mode_labels=tuple(str(v) for v in obj["mode_labels"]),
    )


def _encoded_operators_to_json(row: EncodedOperatorBundle) -> dict[str, Any]:
    return {
        "h_electronic": dict(row.h_electronic),
        "dH_dQ_by_mode": [dict(v) for v in row.dH_dQ_by_mode],
        "h_vibronic": dict(row.h_vibronic),
        "q_by_mode": [dict(v) for v in row.q_by_mode],
        "p_by_mode": [dict(v) for v in row.p_by_mode],
        "n_by_mode": [dict(v) for v in row.n_by_mode],
    }


def _encoded_operators_from_json(payload: Any) -> EncodedOperatorBundle:
    obj = _as_mapping(payload, label="encoded_operators")
    return EncodedOperatorBundle(
        h_electronic=dict(_as_mapping(obj["h_electronic"], label="encoded_operators.h_electronic")),
        dH_dQ_by_mode=tuple(
            dict(_as_mapping(row, label="encoded_operators.dH_dQ_by_mode[]"))
            for row in _list_payload(obj["dH_dQ_by_mode"], "encoded_operators.dH_dQ_by_mode")
        ),
        h_vibronic=dict(_as_mapping(obj["h_vibronic"], label="encoded_operators.h_vibronic")),
        q_by_mode=tuple(
            dict(_as_mapping(row, label="encoded_operators.q_by_mode[]"))
            for row in _list_payload(obj.get("q_by_mode", []), "encoded_operators.q_by_mode")
        ),
        p_by_mode=tuple(
            dict(_as_mapping(row, label="encoded_operators.p_by_mode[]"))
            for row in _list_payload(obj.get("p_by_mode", []), "encoded_operators.p_by_mode")
        ),
        n_by_mode=tuple(
            dict(_as_mapping(row, label="encoded_operators.n_by_mode[]"))
            for row in _list_payload(obj.get("n_by_mode", []), "encoded_operators.n_by_mode")
        ),
    )


def _boundary_weight_to_json(row: BoundaryWeightRecord | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "total_boundary_weight": float(row.total_boundary_weight),
        "per_mode_boundary_weight": {str(k): float(v) for k, v in row.per_mode_boundary_weight.items()},
        "state_source": row.state_source,
    }


def _boundary_weight_from_json(payload: Any) -> BoundaryWeightRecord | None:
    if payload is None:
        return None
    obj = _as_mapping(payload, label="boundary_weight")
    return BoundaryWeightRecord(
        total_boundary_weight=float(obj["total_boundary_weight"]),
        per_mode_boundary_weight={
            str(k): float(v)
            for k, v in _as_mapping(obj.get("per_mode_boundary_weight", {}), label="per_mode_boundary_weight").items()
        },
        state_source=str(obj["state_source"]),
    )


def _exact_state_to_json(row: ExactStateVectorRecord | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "available": bool(row.available),
        "representation": str(row.representation),
        "n_qubits": int(row.n_qubits),
        "norm": float(row.norm),
        "amplitudes_qn_to_q0": {
            str(bitstr): {"re": float(coeff.get("re", 0.0)), "im": float(coeff.get("im", 0.0))}
            for bitstr, coeff in row.amplitudes_qn_to_q0.items()
        },
        "sidecar_path": row.sidecar_path,
        "sha256": row.sha256,
        "reason_unavailable": row.reason_unavailable,
    }


def _exact_state_from_json(payload: Any) -> ExactStateVectorRecord | None:
    if payload is None:
        return None
    obj = _as_mapping(payload, label="exact_state")
    return ExactStateVectorRecord(
        available=bool(obj["available"]),
        representation=str(obj["representation"]),  # type: ignore[arg-type]
        n_qubits=int(obj["n_qubits"]),
        norm=float(obj["norm"]),
        amplitudes_qn_to_q0={
            str(bitstr): {
                "re": float(_as_mapping(coeff, label=f"exact_state.amplitudes[{bitstr!r}]").get("re", 0.0)),
                "im": float(_as_mapping(coeff, label=f"exact_state.amplitudes[{bitstr!r}]").get("im", 0.0)),
            }
            for bitstr, coeff in _as_mapping(
                obj.get("amplitudes_qn_to_q0", {}),
                label="exact_state.amplitudes_qn_to_q0",
            ).items()
        },
        sidecar_path=None if obj.get("sidecar_path") is None else str(obj["sidecar_path"]),
        sha256=None if obj.get("sha256") is None else str(obj["sha256"]),
        reason_unavailable=None if obj.get("reason_unavailable") is None else str(obj["reason_unavailable"]),
    )


def _exact_reference_to_json(row: ExactReferenceRecord) -> dict[str, Any]:
    return {
        "available": bool(row.available),
        "method": row.method,
        "sector_dimension": int(row.sector_dimension),
        "full_qubit_dimension": int(row.full_qubit_dimension),
        "ground_energy_hartree": row.ground_energy_hartree,
        "low_energies_hartree": list(row.low_energies_hartree),
        "boundary_weight": _boundary_weight_to_json(row.boundary_weight),
        "ground_state": _exact_state_to_json(row.ground_state),
        "solver_tolerance": row.solver_tolerance,
        "reason_unavailable": row.reason_unavailable,
    }


def _exact_reference_from_json(payload: Any) -> ExactReferenceRecord:
    obj = _as_mapping(payload, label="exact_reference")
    return ExactReferenceRecord(
        available=bool(obj["available"]),
        method=str(obj["method"]),  # type: ignore[arg-type]
        sector_dimension=int(obj["sector_dimension"]),
        full_qubit_dimension=int(obj["full_qubit_dimension"]),
        ground_energy_hartree=None if obj.get("ground_energy_hartree") is None else float(obj["ground_energy_hartree"]),
        low_energies_hartree=tuple(float(v) for v in obj.get("low_energies_hartree", ())),
        boundary_weight=_boundary_weight_from_json(obj.get("boundary_weight")),
        ground_state=_exact_state_from_json(obj.get("ground_state")),
        solver_tolerance=None if obj.get("solver_tolerance") is None else float(obj["solver_tolerance"]),
        reason_unavailable=None if obj.get("reason_unavailable") is None else str(obj["reason_unavailable"]),
    )


def _cutoff_diagnostics_to_json(row: CutoffDiagnosticsRecord) -> dict[str, Any]:
    return {
        "work_cutoffs": list(row.work_cutoffs),
        "reference_cutoffs": None if row.reference_cutoffs is None else list(row.reference_cutoffs),
        "work_ground_energy_hartree": row.work_ground_energy_hartree,
        "reference_ground_energy_hartree": row.reference_ground_energy_hartree,
        "delta_energy_hartree": row.delta_energy_hartree,
        "work_boundary_weight": _boundary_weight_to_json(row.work_boundary_weight),
        "passed": bool(row.passed),
        "policy": row.policy,
        "energy_tolerance_hartree": row.energy_tolerance_hartree,
        "boundary_weight_tolerance": row.boundary_weight_tolerance,
        "energy_passed": row.energy_passed,
        "boundary_passed": row.boundary_passed,
    }


def _cutoff_diagnostics_from_json(payload: Any) -> CutoffDiagnosticsRecord:
    obj = _as_mapping(payload, label="cutoff_diagnostics")
    boundary_weight = _boundary_weight_from_json(obj.get("work_boundary_weight"))
    energy_tolerance_raw = obj.get("energy_tolerance_hartree")
    energy_tolerance = float(
        H2O_LINEAR_FD_CUTOFF_ENERGY_TOLERANCE_HARTREE
        if energy_tolerance_raw is None
        else energy_tolerance_raw
    )
    boundary_tolerance_raw = obj.get("boundary_weight_tolerance")
    boundary_tolerance = float(
        H2O_LINEAR_FD_CUTOFF_BOUNDARY_WEIGHT_TOLERANCE
        if boundary_tolerance_raw is None
        else boundary_tolerance_raw
    )
    delta_energy = None if obj.get("delta_energy_hartree") is None else float(obj["delta_energy_hartree"])
    assessment = assess_h2o_linear_fd_cutoff_diagnostics(
        delta_energy_hartree=delta_energy,
        work_boundary_weight=boundary_weight,
        energy_tolerance_hartree=energy_tolerance,
        boundary_weight_tolerance=boundary_tolerance,
    )
    return CutoffDiagnosticsRecord(
        work_cutoffs=_as_tuple_int(obj["work_cutoffs"], label="cutoff_diagnostics.work_cutoffs"),
        reference_cutoffs=(
            None
            if obj.get("reference_cutoffs") is None
            else _as_tuple_int(obj["reference_cutoffs"], label="cutoff_diagnostics.reference_cutoffs")
        ),
        work_ground_energy_hartree=(
            None if obj.get("work_ground_energy_hartree") is None else float(obj["work_ground_energy_hartree"])
        ),
        reference_ground_energy_hartree=(
            None
            if obj.get("reference_ground_energy_hartree") is None
            else float(obj["reference_ground_energy_hartree"])
        ),
        delta_energy_hartree=delta_energy,
        work_boundary_weight=boundary_weight,
        passed=bool(assessment["passed"]),
        policy=str(obj["policy"]),
        energy_tolerance_hartree=float(assessment["energy_tolerance_hartree"]),
        boundary_weight_tolerance=float(assessment["boundary_weight_tolerance"]),
        energy_passed=assessment["energy_passed"],
        boundary_passed=assessment["boundary_passed"],
    )


def _evidence_hooks_to_json(row: EvidenceHooksRecord) -> dict[str, Any]:
    return {
        "static_ground_state_ready": bool(row.static_ground_state_ready),
        "exact_reference_ready": bool(row.exact_reference_ready),
        "dynamics_hooks_ready": bool(row.dynamics_hooks_ready),
        "qse_hooks_ready": bool(row.qse_hooks_ready),
        "qse_probe_families": list(row.qse_probe_families),
        "qse_generator_families": list(row.qse_generator_families),
    }


def _evidence_hooks_from_json(payload: Any) -> EvidenceHooksRecord:
    obj = _as_mapping(payload, label="evidence_hooks")
    return EvidenceHooksRecord(
        static_ground_state_ready=bool(obj.get("static_ground_state_ready", False)),
        exact_reference_ready=bool(obj.get("exact_reference_ready", False)),
        dynamics_hooks_ready=bool(obj.get("dynamics_hooks_ready", False)),
        qse_hooks_ready=bool(obj.get("qse_hooks_ready", False)),
        qse_probe_families=tuple(str(v) for v in obj.get("qse_probe_families", ())),
        qse_generator_families=tuple(str(v) for v in obj.get("qse_generator_families", ())),
    )


__all__ = [
    "H2O_LINEAR_FD_DERIVATIVE_SOURCE",
    "H2O_LINEAR_FD_DERIVATIVE_RESOLVED_POOL_KEY",
    "H2O_LINEAR_FD_CUTOFF_BOUNDARY_WEIGHT_TOLERANCE",
    "H2O_LINEAR_FD_CUTOFF_ENERGY_TOLERANCE_HARTREE",
    "H2O_LINEAR_FD_FAMILY_KEY",
    "H2O_LINEAR_FD_FIXTURE_SCHEMA",
    "H2O_LINEAR_FD_MODEL_ROLE",
    "H2O_UMBRELLA_FAMILY_KEY",
    "ActiveSpaceRecord",
    "AlignedActiveTensorRecord",
    "AlignmentDiagnosticsRecord",
    "AlignmentThresholds",
    "BosonModeRegister",
    "BoundaryWeightRecord",
    "CutoffDiagnosticsRecord",
    "DerivativeNorms",
    "DisplacedGeometryRecord",
    "EncodedOperatorBundle",
    "EvidenceHooksRecord",
    "ExactReferenceRecord",
    "ExactStateVectorRecord",
    "FirstDerivativeRecord",
    "FixtureManifest",
    "GeometryRecord",
    "NormalModeRecord",
    "PhysicalSectorRecord",
    "CachedProductionVibronicH2OLinearFDFixture",
    "ProductionVibronicH2OLinearFDRuntimeModel",
    "ProductionVibronicH2OFixture",
    "RegisterLayout",
    "build_h2o_linear_fd_derivative_resolved_pool_v2",
    "build_production_vibronic_h2o_linear_fd_runtime_model",
    "assess_h2o_linear_fd_cutoff_diagnostics",
    "fixed_sector_dimension",
    "h2o_linear_fd_boundary_weight_for_state",
    "load_cached_production_vibronic_h2o_linear_fd_fixture",
    "load_production_vibronic_h2o_fixture",
    "production_vibronic_h2o_fixture_from_jsonable",
    "production_vibronic_h2o_fixture_to_jsonable",
    "validate_normal_mode",
    "validate_paper_iv_h2o_linear_fd_evidence_fixture",
    "validate_production_vibronic_h2o_fixture",
    "write_production_vibronic_h2o_fixture",
]
