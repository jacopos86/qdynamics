"""Offline fixed-structure Hubbard-Holstein VQE conditioning-stress construction.

This module builds *conventional* fixed-architecture Hubbard-Holstein VQE
ansaetze whose McLachlan tangent geometry is deliberately ill-conditioned while
the prepared ground state stays variationally accurate.  It is a construction
backend for diagnostic inputs to AP-McLachlan stress tests; it is not part of
the AP-McLachlan method, controller, or canonical scientific defaults.

Scientific contract
-------------------
For a fixed ordered architecture ``A`` of symmetry-legal ``full_meta``
Pauli/polyterm children,

    |psi_A(theta)> = U_A(theta)|psi_ref>,   U_A(theta) = prod_k exp(-i theta_k c_k P_k)

with one runtime coordinate per Pauli/polynomial term (the repository
``per_pauli_term`` convention).  The architecture is fixed *before* the inner
VQE; the inner VQE only optimizes angles.  Nothing here calls ADAPT selection or
grows an ansatz.

Two qualification gates are hard, untradeable filters:

1. ground-state accuracy   ``dE = E(theta_0) - E_0 <= delta_e_max``;
2. driven-snapshot fit     ``d_FS(U_A(theta_i)|psi_ref>, |psi_exact(t_i)>) <= ray_distance_max``.

Only architectures that clear (1) enter the conditioning frontier, and only
snapshots that clear (2) contribute time-local conditioning evidence.

Conditioning is then read from the horizontally projected tangent Gram matrix

    G_ij = Re <T_i - psi <psi|T_i>, T_j - psi <psi|T_j>>

built by the *same* AP-McLachlan evaluator used in propagation, and reported as
``rank``, ``nullity``, ``s_min_kept``, ``s_max`` and ``kappa_eff``.

Exact-reference isolation
-------------------------
Exact ground states and exact driven trajectories are used offline here, by this
builder only.  They are never written into the runtime artifact beyond digests
and scalar diagnostics, and they never enter AP-McLachlan propagation,
support-patch scoring, repair, append, prune, exchange, or online tuning.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from pipelines.scaffold.hh_continuation_generators import (
    build_generator_metadata,
    build_runtime_split_child_sets,
    build_runtime_split_children,
    rebuild_polynomial_from_serialized_terms,
    serialize_polynomial_terms_exyz,
)
from pipelines.scaffold.hh_fixed_manifold_loader import _make_replay_run_cfg
from pipelines.scaffold.hh_vqe_from_adapt_family import (
    RunConfig as ReplayRunConfig,
    _build_hh_hamiltonian,
    _build_pool_for_family,
)
from pipelines.static_adapt.builders.problem_registry import resolve_problem_context
from pipelines.static_adapt.builders.problem_setup import (
    _exact_gs_energy_for_problem,
    resolve_exact_reference_state_for_problem,
)
from pipelines.contracts.problem import ProblemRequest
from pipelines.time_dynamics.ap_mclachlan.geometry_eval import evaluate_mclachlan_geometry
from pipelines.time_dynamics.ap_mclachlan.hamiltonian import (
    TimeDependentHamiltonian,
    assert_zero_drive_static_parity,
)
from pipelines.time_dynamics.ap_mclachlan.inverse import (
    McLachlanInversePolicy,
    supported_inverse,
)
from pipelines.time_dynamics.ap_mclachlan.reference_energy_generation import (
    ReferenceEnergyGenerationConfig,
    _dense_hamiltonian_provider,
    _driven_reference_states,
    _static_reference_states,
)
from pipelines.time_dynamics.ap_mclachlan.state import (
    AP_PARAMETERIZATION_PER_PAULI_TERM,
    APMcLachlanState,
)
from pipelines.time_dynamics.adapters.drive_terms import resolve_realtime_drive_model
from src.quantum.ansatz_parameterization import (
    AnsatzParameterLayout,
    build_parameter_layout,
    project_runtime_theta_block_mean,
    serialize_layout,
)
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.hubbard_latex_python_pairs import boson_qubits_per_site
from src.quantum.vqe_latex_python_pairs import (
    AnsatzTerm,
    hamiltonian_matrix,
    vqe_minimize,
)


FIXED_VQE_CONDITIONING_SCHEMA_V1 = "fixed_vqe_conditioning_stress_v1"
FIXED_VQE_CONDITIONING_ARTIFACT_SCHEMA_V1 = "fixed_vqe_conditioning_stress_artifact_v1"
FIXED_VQE_STRESS_SUBJECT_KIND_V1 = "fixed_vqe_conditioning_stress_v1"
FIXED_VQE_STRESS_ROUTE_FAMILY = "locked_imported_scaffold_v1"

CHILD_KIND_PAULI = "pauli_child"
CHILD_KIND_POLYTERM = "polyterm_child"
CHILD_KINDS = (CHILD_KIND_PAULI, CHILD_KIND_POLYTERM)

CONSTRUCTION_MODE_CONVENTIONAL = "conventional_fixed_layered_v1"
CONSTRUCTION_MODE_EXACT_NULL_TEST = "exact_null_test_v1"
CONSTRUCTION_MODE_NEAR_NULL_TEST = "near_null_test_v1"
CONSTRUCTION_MODES = (
    CONSTRUCTION_MODE_CONVENTIONAL,
    CONSTRUCTION_MODE_EXACT_NULL_TEST,
    CONSTRUCTION_MODE_NEAR_NULL_TEST,
)

# The Hubbard-Holstein HVA layer parents whose complete ordered child runs
# reproduce the conventional termwise HVA layers.
CONVENTIONAL_HH_LAYER_PARENTS = ("hop_layer", "onsite_layer", "eph_layer")

DEFAULT_DELTA_E_MAX = 1.0e-6
DEFAULT_SNAPSHOT_RAY_DISTANCE_MAX = 1.0e-6
DEFAULT_GRAM_RETAINED_RCOND = 1.0e-10
DEFAULT_SNAPSHOT_TIMES = (0.0, 0.5, 1.0, 1.5, 2.0, 2.75, 3.75, 5.0)

# The AP-McLachlan solve convention applies a ridge before its eigendecomposition.
# A diagnostic that must *see* exact nullity has to read the unridged spectrum,
# so this backend pins ridge_lambda=0 and keeps the retained-mode convention.
GRAM_DIAGNOSTIC_RIDGE_LAMBDA = 0.0


# ---------------------------------------------------------------------------
# Typed configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FixedVQEModelConfig:
    """Hubbard-Holstein model and register settings for the construction."""

    num_sites: int = 2
    t: float = 1.0
    u: float = 1.0
    dv: float = 0.0
    omega0: float = 1.0
    g_ep: float = 0.5
    n_ph_max: int = 1
    boson_encoding: str = "binary"
    ordering: str = "blocked"
    boundary: str = "open"
    sector_n_up: int | None = None
    sector_n_dn: int | None = None
    paop_r: int = 1
    paop_split_paulis: bool = False
    paop_prune_eps: float = 0.0
    paop_normalization: str = "none"

    def __post_init__(self) -> None:
        if int(self.num_sites) <= 0:
            raise ValueError("fixed-VQE conditioning requires num_sites >= 1.")
        if int(self.n_ph_max) < 0:
            raise ValueError("fixed-VQE conditioning requires n_ph_max >= 0.")
        for name in ("t", "u", "dv", "omega0", "g_ep"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"model {name} must be finite.")

    @property
    def num_particles(self) -> tuple[int, int]:
        if self.sector_n_up is None or self.sector_n_dn is None:
            return ((int(self.num_sites) + 1) // 2, int(self.num_sites) // 2)
        return (int(self.sector_n_up), int(self.sector_n_dn))

    @property
    def boson_qubits_per_site(self) -> int:
        return int(boson_qubits_per_site(int(self.n_ph_max), str(self.boson_encoding)))

    @property
    def total_qubits(self) -> int:
        return int(2 * int(self.num_sites) + int(self.num_sites) * self.boson_qubits_per_site)

    def to_settings_payload(self) -> dict[str, Any]:
        """Return the ``settings`` block consumed by the scaffold runtime loader."""

        n_up, n_dn = self.num_particles
        return {
            "problem": "hh",
            "L": int(self.num_sites),
            "t": float(self.t),
            "u": float(self.u),
            "dv": float(self.dv),
            "omega0": float(self.omega0),
            "g_ep": float(self.g_ep),
            "n_ph_max": int(self.n_ph_max),
            "boson_encoding": str(self.boson_encoding),
            "ordering": str(self.ordering),
            "boundary": str(self.boundary),
            "sector_n_up": int(n_up),
            "sector_n_dn": int(n_dn),
            "include_zero_point": True,
            "paop_r": int(self.paop_r),
            "paop_split_paulis": bool(self.paop_split_paulis),
            "paop_prune_eps": float(self.paop_prune_eps),
            "paop_normalization": str(self.paop_normalization),
        }

    def to_json_dict(self) -> dict[str, Any]:
        payload = self.to_settings_payload()
        payload["total_qubits"] = int(self.total_qubits)
        return payload


@dataclass(frozen=True)
class FixedVQEDriveConfig:
    """Canonical weak-weak Gaussian density drive settings for the snapshots."""

    enabled: bool = True
    drive_A: float = 0.1
    drive_omega: float = 1.0
    drive_tbar: float = 2.0
    drive_phi: float = 0.0
    drive_pattern: str = "staggered"
    drive_custom_weights: tuple[float, ...] | None = None
    drive_include_identity: bool = False
    drive_time_sampling: str = "midpoint"
    drive_t0: float = 0.0

    def __post_init__(self) -> None:
        for name in ("drive_A", "drive_omega", "drive_tbar", "drive_phi", "drive_t0"):
            if not np.isfinite(float(getattr(self, name))):
                raise ValueError(f"drive {name} must be finite.")
        if self.drive_custom_weights is not None:
            object.__setattr__(
                self,
                "drive_custom_weights",
                tuple(float(x) for x in self.drive_custom_weights),
            )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "A": float(self.drive_A),
            "omega": float(self.drive_omega),
            "tbar": float(self.drive_tbar),
            "phi": float(self.drive_phi),
            "pattern": str(self.drive_pattern),
            "custom_weights": (
                None
                if self.drive_custom_weights is None
                else [float(x) for x in self.drive_custom_weights]
            ),
            "include_identity": bool(self.drive_include_identity),
            "time_sampling": str(self.drive_time_sampling),
            "t0": float(self.drive_t0),
        }


@dataclass(frozen=True)
class SnapshotScheduleConfig:
    """Explicit exact-snapshot times for the driven diagnostic."""

    times: tuple[float, ...] = DEFAULT_SNAPSHOT_TIMES
    rtol: float = 1.0e-10
    atol: float = 1.0e-12
    max_internal_step: float | None = None
    norm_drift_tolerance: float = 1.0e-8

    def __post_init__(self) -> None:
        times = tuple(float(x) for x in self.times)
        if not times:
            raise ValueError("snapshot schedule requires at least one time.")
        if any(not np.isfinite(x) for x in times):
            raise ValueError("snapshot times must be finite.")
        if len(set(times)) != len(times):
            raise ValueError("snapshot times must not contain duplicates.")
        if any(right < left for left, right in zip(times[:-1], times[1:])):
            raise ValueError("snapshot times must be monotonically increasing.")
        object.__setattr__(self, "times", times)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "times": [float(x) for x in self.times],
            "time_count": int(len(self.times)),
            "rtol": float(self.rtol),
            "atol": float(self.atol),
            "max_internal_step": (
                None if self.max_internal_step is None else float(self.max_internal_step)
            ),
            "norm_drift_tolerance": float(self.norm_drift_tolerance),
        }

    def to_reference_generation_config(self) -> ReferenceEnergyGenerationConfig:
        return ReferenceEnergyGenerationConfig(
            rtol=float(self.rtol),
            atol=float(self.atol),
            max_internal_step=self.max_internal_step,
            norm_drift_tolerance=float(self.norm_drift_tolerance),
        )


@dataclass(frozen=True)
class GroundStateQualificationConfig:
    """Hard ground-state energy gate plus the inner fixed-structure VQE settings."""

    delta_e_max: float = DEFAULT_DELTA_E_MAX
    method: str = "L-BFGS-B"
    maxiter: int = 20000
    restarts: int = 4
    seed: int = 7
    initial_point_stddev: float = 0.4
    # A layered fixed ansatz starts at the identity by convention, so the first
    # restart begins at theta = 0 and the remaining restarts are random.
    zero_first_restart: bool = True
    bounds: tuple[float, float] | None = (-math.pi, math.pi)
    energy_backend: str = "one_apply_compiled"

    def __post_init__(self) -> None:
        if not np.isfinite(float(self.delta_e_max)) or float(self.delta_e_max) < 0.0:
            raise ValueError("delta_e_max must be finite and non-negative.")
        if int(self.restarts) <= 0:
            raise ValueError("ground-state VQE restarts must be positive.")
        if int(self.maxiter) <= 0:
            raise ValueError("ground-state VQE maxiter must be positive.")

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "delta_e_max": float(self.delta_e_max),
            "gate_kind": "hard_untradeable",
            "method": str(self.method),
            "maxiter": int(self.maxiter),
            "restarts": int(self.restarts),
            "seed": int(self.seed),
            "initial_point_stddev": float(self.initial_point_stddev),
            "zero_first_restart": bool(self.zero_first_restart),
            "bounds": None if self.bounds is None else [float(x) for x in self.bounds],
            "energy_backend": str(self.energy_backend),
        }


@dataclass(frozen=True)
class SnapshotFitConfig:
    """Independent per-snapshot Fubini--Study fit settings and eligibility gate."""

    ray_distance_max: float = DEFAULT_SNAPSHOT_RAY_DISTANCE_MAX
    method: str = "L-BFGS-B"
    maxiter: int = 6000
    restarts: int = 3
    seed: int = 11
    initial_point_stddev: float = 0.25
    warm_start_from_neighbor: bool = True
    warm_start_from_ground_state: bool = True

    def __post_init__(self) -> None:
        if not np.isfinite(float(self.ray_distance_max)) or float(self.ray_distance_max) < 0.0:
            raise ValueError("ray_distance_max must be finite and non-negative.")
        if int(self.restarts) <= 0:
            raise ValueError("snapshot-fit restarts must be positive.")

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "ray_distance_max": float(self.ray_distance_max),
            "gate_kind": "hard_untradeable",
            "distance_definition": "fubini_study_ray_distance_orthogonal_projection_v1",
            "method": str(self.method),
            "maxiter": int(self.maxiter),
            "restarts": int(self.restarts),
            "seed": int(self.seed),
            "initial_point_stddev": float(self.initial_point_stddev),
            "warm_start_from_neighbor": bool(self.warm_start_from_neighbor),
            "warm_start_from_ground_state": bool(self.warm_start_from_ground_state),
            "warm_start_scope": "acceleration_only_each_fit_checked_independently",
        }


@dataclass(frozen=True)
class GramSpectrumConfig:
    """Explicit retained-mode convention for the tangent Gram diagnostics."""

    retained_rcond: float = DEFAULT_GRAM_RETAINED_RCOND
    ridge_lambda: float = GRAM_DIAGNOSTIC_RIDGE_LAMBDA
    conditioning_warning_log10_kappa: float = 8.0
    store_full_spectrum: bool = True

    def __post_init__(self) -> None:
        if not np.isfinite(float(self.retained_rcond)) or float(self.retained_rcond) < 0.0:
            raise ValueError("retained_rcond must be finite and non-negative.")
        if not np.isfinite(float(self.ridge_lambda)) or float(self.ridge_lambda) < 0.0:
            raise ValueError("ridge_lambda must be finite and non-negative.")

    @property
    def inverse_policy(self) -> McLachlanInversePolicy:
        return McLachlanInversePolicy(
            pinv_rcond=float(self.retained_rcond),
            ridge_lambda=float(self.ridge_lambda),
            solve_damping=0.0,
        )

    def to_json_dict(self) -> dict[str, Any]:
        policy = self.inverse_policy
        return {
            "retained_rcond": float(self.retained_rcond),
            "ridge_lambda": float(self.ridge_lambda),
            "retained_mode_convention": str(policy.policy_id),
            "retained_threshold_rule": "abs_eig > retained_rcond * max_abs_eig",
            "conditioning_warning_log10_kappa": float(self.conditioning_warning_log10_kappa),
            "store_full_spectrum": bool(self.store_full_spectrum),
        }


@dataclass(frozen=True)
class GeneratorPoolConfig:
    """Which symmetry-legal ``full_meta`` children may enter an architecture."""

    pool_key: str = "full_meta"
    include_pauli_children: bool = True
    include_polyterm_children: bool = True
    polyterm_subset_sizes: tuple[int, ...] = (2,)
    max_atoms_per_parent: int | None = 8
    max_pool_atoms: int | None = 512
    require_hard_symmetry_guard: bool = True
    # Conventional Hubbard-Holstein layer parents are emitted first so that
    # atom-count caps never silently drop the parents a conventional layered
    # architecture is built from.
    priority_parent_labels: tuple[str, ...] = CONVENTIONAL_HH_LAYER_PARENTS

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "priority_parent_labels",
            tuple(str(x) for x in self.priority_parent_labels),
        )
        sizes = tuple(sorted({int(x) for x in self.polyterm_subset_sizes}))
        if any(size < 2 for size in sizes):
            raise ValueError("polyterm subset sizes must all be >= 2.")
        object.__setattr__(self, "polyterm_subset_sizes", sizes)
        if not (self.include_pauli_children or self.include_polyterm_children):
            raise ValueError("generator pool must include Pauli or polyterm children.")
        if not bool(self.require_hard_symmetry_guard):
            raise ValueError(
                "the fixed-sector/binary-padding hard symmetry guard is mandatory for "
                "Hubbard-Holstein full_meta children."
            )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "pool_key": str(self.pool_key),
            "include_pauli_children": bool(self.include_pauli_children),
            "include_polyterm_children": bool(self.include_polyterm_children),
            "polyterm_subset_sizes": [int(x) for x in self.polyterm_subset_sizes],
            "max_atoms_per_parent": (
                None if self.max_atoms_per_parent is None else int(self.max_atoms_per_parent)
            ),
            "max_pool_atoms": (
                None if self.max_pool_atoms is None else int(self.max_pool_atoms)
            ),
            "require_hard_symmetry_guard": True,
            "priority_parent_labels": [str(x) for x in self.priority_parent_labels],
        }


@dataclass(frozen=True)
class ArchitectureSearchConfig:
    """Deterministic, restartable Pareto/evolutionary architecture search."""

    construction_mode: str = CONSTRUCTION_MODE_CONVENTIONAL
    layer_counts: tuple[int, ...] = (1, 2, 3)
    atoms_per_layer: tuple[int, ...] = (4, 6, 8)
    allow_repeated_occurrences: bool = True
    population_size: int = 8
    generations: int = 3
    mutation_count: int = 2
    seed: int = 20260814
    max_architecture_workers: int = 1
    max_snapshot_workers: int = 1
    retain_beyond_pareto: int = 0
    # Parent-complete layers reproduce the conventional termwise HVA/Hamiltonian
    # layer constructions and are far more likely to clear the delta-E gate than
    # unstructured random draws, so they seed the population first.
    seed_parent_complete_layers: bool = True
    seed_parent_labels: tuple[str, ...] = ()
    seed_parent_complete_repeats: tuple[int, ...] = (1, 2)

    def __post_init__(self) -> None:
        if str(self.construction_mode) not in set(CONSTRUCTION_MODES):
            raise ValueError(
                f"construction_mode must be one of {CONSTRUCTION_MODES}; "
                f"got {self.construction_mode!r}."
            )
        layers = tuple(int(x) for x in self.layer_counts)
        widths = tuple(int(x) for x in self.atoms_per_layer)
        if not layers or any(x <= 0 for x in layers):
            raise ValueError("layer_counts must contain positive integers.")
        if not widths or any(x <= 0 for x in widths):
            raise ValueError("atoms_per_layer must contain positive integers.")
        object.__setattr__(self, "layer_counts", layers)
        object.__setattr__(self, "atoms_per_layer", widths)
        if int(self.population_size) <= 0:
            raise ValueError("population_size must be positive.")
        if int(self.generations) <= 0:
            raise ValueError("generations must be positive.")
        for name in ("max_architecture_workers", "max_snapshot_workers"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive.")
        repeats = tuple(sorted({int(x) for x in self.seed_parent_complete_repeats}))
        if any(x <= 0 for x in repeats):
            raise ValueError("seed_parent_complete_repeats must contain positive integers.")
        object.__setattr__(self, "seed_parent_complete_repeats", repeats)
        object.__setattr__(
            self, "seed_parent_labels", tuple(str(x) for x in self.seed_parent_labels)
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "construction_mode": str(self.construction_mode),
            "layer_counts": [int(x) for x in self.layer_counts],
            "atoms_per_layer": [int(x) for x in self.atoms_per_layer],
            "allow_repeated_occurrences": bool(self.allow_repeated_occurrences),
            "population_size": int(self.population_size),
            "generations": int(self.generations),
            "mutation_count": int(self.mutation_count),
            "seed": int(self.seed),
            "max_architecture_workers": int(self.max_architecture_workers),
            "max_snapshot_workers": int(self.max_snapshot_workers),
            "retain_beyond_pareto": int(self.retain_beyond_pareto),
            "seed_parent_complete_layers": bool(self.seed_parent_complete_layers),
            "seed_parent_labels": [str(x) for x in self.seed_parent_labels],
            "seed_parent_complete_repeats": [
                int(x) for x in self.seed_parent_complete_repeats
            ],
        }


@dataclass(frozen=True)
class FixedVQEConditioningConfig:
    """Complete typed construction/search configuration."""

    model: FixedVQEModelConfig = field(default_factory=FixedVQEModelConfig)
    drive: FixedVQEDriveConfig = field(default_factory=FixedVQEDriveConfig)
    snapshots: SnapshotScheduleConfig = field(default_factory=SnapshotScheduleConfig)
    ground_state: GroundStateQualificationConfig = field(
        default_factory=GroundStateQualificationConfig
    )
    snapshot_fit: SnapshotFitConfig = field(default_factory=SnapshotFitConfig)
    gram: GramSpectrumConfig = field(default_factory=GramSpectrumConfig)
    pool: GeneratorPoolConfig = field(default_factory=GeneratorPoolConfig)
    search: ArchitectureSearchConfig = field(default_factory=ArchitectureSearchConfig)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema": FIXED_VQE_CONDITIONING_SCHEMA_V1,
            "model": self.model.to_json_dict(),
            "drive": self.drive.to_json_dict(),
            "snapshots": self.snapshots.to_json_dict(),
            "ground_state": self.ground_state.to_json_dict(),
            "snapshot_fit": self.snapshot_fit.to_json_dict(),
            "gram": self.gram.to_json_dict(),
            "pool": self.pool.to_json_dict(),
            "search": self.search.to_json_dict(),
        }

    def config_digest(self) -> str:
        return _digest_json(self.to_json_dict())


# ---------------------------------------------------------------------------
# Generator pool: symmetry-legal full_meta Pauli / polyterm children
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FixedVQEGeneratorAtom:
    """One reusable, symmetry-legal ``full_meta`` child generator."""

    atom_id: str
    atom_label: str
    parent_label: str
    child_kind: str
    serialized_terms: tuple[Mapping[str, Any], ...]
    pauli_words: tuple[str, ...]
    symmetry_gate: Mapping[str, Any]
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if str(self.child_kind) not in set(CHILD_KINDS):
            raise ValueError(f"child_kind must be one of {CHILD_KINDS}.")
        object.__setattr__(
            self, "serialized_terms", tuple(dict(term) for term in self.serialized_terms)
        )
        object.__setattr__(self, "pauli_words", tuple(str(x) for x in self.pauli_words))
        object.__setattr__(self, "symmetry_gate", dict(self.symmetry_gate))
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def runtime_coordinate_count(self) -> int:
        return int(len(self.pauli_words))

    def build_polynomial(self, *, tol: float = 1.0e-12) -> Any:
        return rebuild_polynomial_from_serialized_terms(
            [dict(term) for term in self.serialized_terms],
            drop_abs_tol=float(tol),
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "atom_id": str(self.atom_id),
            "atom_label": str(self.atom_label),
            "parent_label": str(self.parent_label),
            "child_kind": str(self.child_kind),
            "pauli_words": [str(x) for x in self.pauli_words],
            "runtime_coordinate_count": int(self.runtime_coordinate_count),
            "symmetry_gate": _json_safe(dict(self.symmetry_gate)),
            "provenance": _json_safe(dict(self.provenance)),
        }


@dataclass(frozen=True)
class FixedVQEGeneratorPool:
    """Ordered, deterministic set of admissible fixed-architecture generators."""

    atoms: tuple[FixedVQEGeneratorAtom, ...]
    pool_key: str
    parent_term_count: int
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        atoms = tuple(self.atoms)
        ids = [atom.atom_id for atom in atoms]
        if len(set(ids)) != len(ids):
            raise ValueError("generator pool atom ids must be unique.")
        object.__setattr__(self, "atoms", atoms)
        object.__setattr__(self, "meta", dict(self.meta))

    @property
    def atom_ids(self) -> tuple[str, ...]:
        return tuple(atom.atom_id for atom in self.atoms)

    def by_id(self, atom_id: str) -> FixedVQEGeneratorAtom:
        for atom in self.atoms:
            if atom.atom_id == str(atom_id):
                return atom
        raise KeyError(f"unknown generator atom id: {atom_id!r}")

    @property
    def ordered_atom_contract_sha256(self) -> str:
        return _digest_json([atom.to_json_dict() for atom in self.atoms])

    def to_json_dict(self, *, include_atoms: bool = False) -> dict[str, Any]:
        payload = {
            "pool_key": str(self.pool_key),
            "parent_term_count": int(self.parent_term_count),
            "atom_count": int(len(self.atoms)),
            "pauli_child_count": int(
                sum(1 for atom in self.atoms if atom.child_kind == CHILD_KIND_PAULI)
            ),
            "polyterm_child_count": int(
                sum(1 for atom in self.atoms if atom.child_kind == CHILD_KIND_POLYTERM)
            ),
            "ordered_atom_contract_sha256": str(self.ordered_atom_contract_sha256),
            "meta": _json_safe(dict(self.meta)),
        }
        if include_atoms:
            payload["atoms"] = [atom.to_json_dict() for atom in self.atoms]
        return payload


def _hard_symmetry_spec() -> dict[str, Any]:
    return {
        "hard_guard": True,
        "particle_number_mode": "preserving",
        "spin_sector_mode": "preserving",
        "source": "fixed_vqe_conditioning_generator_pool",
    }


def _gate_checked_and_passed(gate: Any) -> bool:
    return bool(
        isinstance(gate, Mapping)
        and bool(gate.get("checked", False))
        and bool(gate.get("passed", False))
    )


def build_fixed_vqe_generator_pool(
    *,
    cfg: ReplayRunConfig,
    h_poly: Any,
    model: FixedVQEModelConfig,
    pool_config: GeneratorPoolConfig,
) -> FixedVQEGeneratorPool:
    """Build the symmetry-legal ``full_meta`` Pauli/polyterm child vocabulary.

    The parent pool comes from the shared Hubbard-Holstein ``full_meta`` builder,
    which already applies the binary-boson legal-subspace filter.  Children are
    produced by the shared runtime-split helpers under a mandatory hard
    fixed-count sector guard, so only symmetry-legal children survive.
    """

    parents, parent_meta = _build_pool_for_family(
        cfg,
        family=str(pool_config.pool_key),
        h_poly=h_poly,
    )
    symmetry_spec = _hard_symmetry_spec()
    fixed_num_particles = model.num_particles
    qpb = int(max(1, model.boson_qubits_per_site))
    # Atoms stay grouped by parent, in the parent's own term order, so a
    # parent-complete run of Pauli children reproduces that parent's termwise
    # product exactly.  Global reordering would silently change the ansatz.
    grouped_atoms: list[tuple[str, list[FixedVQEGeneratorAtom]]] = []
    seen_signatures: set[tuple[tuple[str, float], ...]] = set()
    rejected_children = 0
    rejected_polyterm = 0

    for parent in parents:
        parent_meta_payload = _parent_generator_metadata(
            parent,
            model=model,
            qpb=qpb,
            symmetry_spec=symmetry_spec,
        )
        children = build_runtime_split_children(
            parent_label=str(parent.label),
            polynomial=parent.polynomial,
            family_id="hh",
            num_sites=int(model.num_sites),
            ordering=str(model.ordering),
            qpb=int(qpb),
            split_mode=CONSTRUCTION_MODE_CONVENTIONAL,
            parent_generator_metadata=parent_meta_payload,
            symmetry_spec=symmetry_spec,
            fixed_num_particles=fixed_num_particles,
            hard_guard_required=True,
            include_unsplit_singleton=True,
        )
        legal_children = []
        for child in children:
            if not _gate_checked_and_passed(child.get("symmetry_gate")):
                rejected_children += 1
                continue
            legal_children.append(child)
        # A pure-identity child carries no runtime coordinate under
        # ignore_identity, so it would be a zero-width architecture block.
        legal_children = [
            child
            for child in legal_children
            if not _is_identity_only_child(child)
        ]
        pauli_atoms: list[FixedVQEGeneratorAtom] = []
        parent_atoms: list[FixedVQEGeneratorAtom] = []
        if bool(pool_config.include_pauli_children):
            for child in legal_children:
                atom = _atom_from_child(child, parent_label=str(parent.label))
                signature = _atom_signature(atom)
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                pauli_atoms.append(atom)
            # Match the repository's sorted runtime term order so a complete run
            # of a parent's Pauli children applies exactly the same ordered
            # rotations as that parent's own termwise product.
            pauli_atoms.sort(key=lambda item: tuple(item.pauli_words))
            parent_atoms.extend(pauli_atoms)
        if bool(pool_config.include_polyterm_children) and len(legal_children) > 1:
            child_sets = build_runtime_split_child_sets(
                parent_label=str(parent.label),
                family_id="hh",
                num_sites=int(model.num_sites),
                ordering=str(model.ordering),
                qpb=int(qpb),
                split_mode=CONSTRUCTION_MODE_CONVENTIONAL,
                children=legal_children,
                parent_generator_metadata=parent_meta_payload,
                symmetry_spec=symmetry_spec,
                fixed_num_particles=fixed_num_particles,
                hard_guard_required=True,
                subset_sizes=pool_config.polyterm_subset_sizes,
            )
            for child_set in child_sets:
                if not _gate_checked_and_passed(child_set.get("symmetry_gate")):
                    rejected_polyterm += 1
                    continue
                # per_pauli_term runtime coordinates cannot execute grouped_exact
                # blocks, and a grouped recommendation means at least one Pauli
                # component is not independently sector-legal.
                if str(child_set.get("recommended_execution_mode")) != "termwise_product":
                    rejected_polyterm += 1
                    continue
                atom = _atom_from_child_set(child_set, parent_label=str(parent.label))
                signature = _atom_signature(atom)
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                parent_atoms.append(atom)
        if pool_config.max_atoms_per_parent is not None:
            parent_atoms = parent_atoms[: int(pool_config.max_atoms_per_parent)]
        if parent_atoms:
            grouped_atoms.append((str(parent.label), parent_atoms))

    # Emit the priority parents first so an atom-count cap never drops the
    # parents that conventional layered architectures are built from.
    priority = tuple(str(x) for x in pool_config.priority_parent_labels)
    grouped_atoms.sort(
        key=lambda item: (
            priority.index(item[0]) if item[0] in priority else len(priority),
        )
    )
    # Truncate at parent granularity: cutting mid-parent would leave a partial
    # child run that no longer reproduces its parent's termwise product.
    atoms: list[FixedVQEGeneratorAtom] = []
    truncated = False
    dropped_parents = 0
    for _label, parent_atoms in grouped_atoms:
        if (
            pool_config.max_pool_atoms is not None
            and atoms
            and len(atoms) + len(parent_atoms) > int(pool_config.max_pool_atoms)
        ):
            truncated = True
            dropped_parents += 1
            continue
        atoms.extend(parent_atoms)
    if not atoms:
        raise ValueError(
            "fixed-VQE conditioning found no symmetry-legal full_meta children for the "
            "requested Hubbard-Holstein problem."
        )
    return FixedVQEGeneratorPool(
        atoms=tuple(atoms),
        pool_key=str(pool_config.pool_key),
        parent_term_count=int(len(parents)),
        meta={
            "parent_pool_meta": _json_safe(dict(parent_meta)),
            "symmetry_spec": dict(symmetry_spec),
            "fixed_num_particles": [int(x) for x in fixed_num_particles],
            "rejected_pauli_child_count": int(rejected_children),
            "rejected_polyterm_child_count": int(rejected_polyterm),
            "pool_truncated_by_max_pool_atoms": bool(truncated),
            "pool_truncation_granularity": "whole_parent_child_runs_only",
            "dropped_parent_count": int(dropped_parents),
            "atom_order": "parent_pool_order_then_parent_native_child_order",
            "symmetry_guard": "mandatory_fixed_count_sector_and_binary_padding",
            "adapt_selection_used": False,
        },
    )


def _parent_generator_metadata(
    parent: Any,
    *,
    model: FixedVQEModelConfig,
    qpb: int,
    symmetry_spec: Mapping[str, Any],
) -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(
        build_generator_metadata(
            label=str(parent.label),
            polynomial=parent.polynomial,
            family_id="hh",
            num_sites=int(model.num_sites),
            ordering=str(model.ordering),
            qpb=int(qpb),
            split_policy="preserve",
            symmetry_spec=dict(symmetry_spec),
        )
    )


def _serialized_terms_from_child_metadata(meta: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    compile_meta = meta.get("compile_metadata", {}) if isinstance(meta, Mapping) else {}
    raw = compile_meta.get("serialized_terms_exyz", []) if isinstance(compile_meta, Mapping) else []
    return tuple(dict(term) for term in raw if isinstance(term, Mapping))


def _atom_from_child(child: Mapping[str, Any], *, parent_label: str) -> FixedVQEGeneratorAtom:
    meta = dict(child.get("child_generator_metadata") or {})
    serialized = _serialized_terms_from_child_metadata(meta)
    if not serialized:
        serialized = tuple(
            dict(term) for term in serialize_polynomial_terms_exyz(child["child_polynomial"])
        )
    label = str(child.get("child_label"))
    return FixedVQEGeneratorAtom(
        atom_id=_atom_id_for_label(label),
        atom_label=label,
        parent_label=str(parent_label),
        child_kind=CHILD_KIND_PAULI,
        serialized_terms=serialized,
        pauli_words=tuple(str(term.get("pauli_exyz", "")) for term in serialized),
        symmetry_gate=dict(child.get("symmetry_gate") or {}),
        provenance={
            "generator_id": meta.get("generator_id"),
            "parent_generator_id": meta.get("parent_generator_id"),
            "child_index": child.get("child_index"),
            "child_count": child.get("child_count"),
            "split_policy": meta.get("split_policy"),
        },
    )


def _atom_from_child_set(child_set: Mapping[str, Any], *, parent_label: str) -> FixedVQEGeneratorAtom:
    meta = dict(child_set.get("candidate_generator_metadata") or {})
    serialized = _serialized_terms_from_child_metadata(meta)
    if not serialized:
        serialized = tuple(
            dict(term)
            for term in serialize_polynomial_terms_exyz(child_set["candidate_polynomial"])
        )
    label = str(child_set.get("candidate_label"))
    return FixedVQEGeneratorAtom(
        atom_id=_atom_id_for_label(label),
        atom_label=label,
        parent_label=str(parent_label),
        child_kind=CHILD_KIND_POLYTERM,
        serialized_terms=serialized,
        pauli_words=tuple(str(term.get("pauli_exyz", "")) for term in serialized),
        symmetry_gate=dict(child_set.get("symmetry_gate") or {}),
        provenance={
            "generator_id": meta.get("generator_id"),
            "parent_generator_id": meta.get("parent_generator_id"),
            "child_indices": [int(x) for x in child_set.get("child_indices", [])],
            "child_labels": [str(x) for x in child_set.get("child_labels", [])],
            "subset_cardinality": child_set.get("subset_cardinality"),
            "split_policy": meta.get("split_policy"),
            "recommended_execution_mode": child_set.get("recommended_execution_mode"),
        },
    )


def _is_identity_word(label: str) -> bool:
    text = str(label).strip().lower()
    return bool(text) and set(text) == {"e"}


def _is_identity_only_child(child: Mapping[str, Any]) -> bool:
    meta = dict(child.get("child_generator_metadata") or {})
    serialized = _serialized_terms_from_child_metadata(meta)
    if not serialized:
        return False
    return all(_is_identity_word(str(term.get("pauli_exyz", ""))) for term in serialized)


def _atom_signature(atom: FixedVQEGeneratorAtom) -> tuple[tuple[str, float], ...]:
    coeffs: dict[str, float] = {}
    for term in atom.serialized_terms:
        label = str(term.get("pauli_exyz", ""))
        coeffs[label] = float(coeffs.get(label, 0.0)) + float(term.get("coeff_re", 0.0))
    return tuple(sorted((label, round(value, 12)) for label, value in coeffs.items()))


def _atom_id_for_label(label: str) -> str:
    return hashlib.sha256(str(label).encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Fixed layered architecture
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeneratorOccurrence:
    """One ordered occurrence of a pool atom inside a fixed architecture."""

    atom_id: str
    layer_index: int
    position_in_layer: int
    occurrence_index: int

    def runtime_block_label(self, atom: FixedVQEGeneratorAtom) -> str:
        return (
            f"{atom.atom_label}"
            f"::layer[{int(self.layer_index)}]"
            f"::slot[{int(self.position_in_layer)}]"
            f"::occ[{int(self.occurrence_index)}]"
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "atom_id": str(self.atom_id),
            "layer_index": int(self.layer_index),
            "position_in_layer": int(self.position_in_layer),
            "occurrence_index": int(self.occurrence_index),
        }


@dataclass(frozen=True)
class FixedArchitecture:
    """A complete fixed layered architecture, determined before any inner VQE."""

    occurrences: tuple[GeneratorOccurrence, ...]
    layer_count: int
    construction_mode: str = CONSTRUCTION_MODE_CONVENTIONAL
    construction_notes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        occurrences = tuple(self.occurrences)
        if not occurrences:
            raise ValueError("a fixed architecture requires at least one occurrence.")
        if str(self.construction_mode) not in set(CONSTRUCTION_MODES):
            raise ValueError(f"construction_mode must be one of {CONSTRUCTION_MODES}.")
        object.__setattr__(self, "occurrences", occurrences)
        object.__setattr__(self, "construction_notes", dict(self.construction_notes))

    @property
    def occurrence_count(self) -> int:
        return int(len(self.occurrences))

    @property
    def distinct_atom_count(self) -> int:
        return int(len({occ.atom_id for occ in self.occurrences}))

    @property
    def architecture_id(self) -> str:
        return _digest_json(
            {
                "construction_mode": str(self.construction_mode),
                "layer_count": int(self.layer_count),
                "occurrences": [occ.to_json_dict() for occ in self.occurrences],
            }
        )

    def layers(self) -> tuple[tuple[GeneratorOccurrence, ...], ...]:
        out: list[list[GeneratorOccurrence]] = [[] for _ in range(int(self.layer_count))]
        for occ in self.occurrences:
            out[int(occ.layer_index)].append(occ)
        return tuple(tuple(layer) for layer in out)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "architecture_id": str(self.architecture_id),
            "construction_mode": str(self.construction_mode),
            "layer_count": int(self.layer_count),
            "occurrence_count": int(self.occurrence_count),
            "distinct_atom_count": int(self.distinct_atom_count),
            "occurrences": [occ.to_json_dict() for occ in self.occurrences],
            "construction_notes": _json_safe(dict(self.construction_notes)),
            "fixed_before_inner_vqe": True,
        }


def architecture_from_layers(
    layers: Sequence[Sequence[str]],
    *,
    construction_mode: str = CONSTRUCTION_MODE_CONVENTIONAL,
    construction_notes: Mapping[str, Any] | None = None,
) -> FixedArchitecture:
    """Build a fixed architecture from ordered per-layer atom-id sequences."""

    occurrences: list[GeneratorOccurrence] = []
    counts: dict[str, int] = {}
    for layer_index, layer in enumerate(layers):
        for position, atom_id in enumerate(layer):
            key = str(atom_id)
            counts[key] = int(counts.get(key, 0)) + 1
            occurrences.append(
                GeneratorOccurrence(
                    atom_id=key,
                    layer_index=int(layer_index),
                    position_in_layer=int(position),
                    occurrence_index=int(counts[key]),
                )
            )
    return FixedArchitecture(
        occurrences=tuple(occurrences),
        layer_count=int(len(tuple(layers))),
        construction_mode=str(construction_mode),
        construction_notes=dict(construction_notes or {}),
    )


def architecture_ansatz_terms(
    architecture: FixedArchitecture,
    *,
    pool: FixedVQEGeneratorPool,
) -> tuple[AnsatzTerm, ...]:
    """Return ordered ``AnsatzTerm`` blocks with distinct runtime identities.

    Every repeated occurrence gets its own block label, so the runtime layout,
    coordinate labels, and serialized artifact never collapse two occurrences of
    the same pool atom into one coordinate.
    """

    terms: list[AnsatzTerm] = []
    seen_labels: set[str] = set()
    for occurrence in architecture.occurrences:
        atom = pool.by_id(occurrence.atom_id)
        label = occurrence.runtime_block_label(atom)
        if label in seen_labels:
            raise ValueError(f"duplicate runtime block label produced: {label!r}")
        seen_labels.add(label)
        terms.append(
            AnsatzTerm(
                label=label,
                polynomial=atom.build_polynomial(),
                execution_mode="termwise_product",
            )
        )
    return tuple(terms)


@dataclass(frozen=True)
class ArchitectureRuntime:
    """Compiled runtime view of one fixed architecture."""

    architecture: FixedArchitecture
    terms: tuple[AnsatzTerm, ...]
    layout: AnsatzParameterLayout
    executor: CompiledAnsatzExecutor

    @property
    def runtime_parameter_count(self) -> int:
        return int(self.layout.runtime_parameter_count)

    @property
    def runtime_coordinate_labels(self) -> tuple[str, ...]:
        labels: list[str] = []
        for block in self.layout.blocks:
            for local_index, spec in enumerate(block.terms):
                labels.append(
                    f"{block.candidate_label}::r{int(local_index)}::{spec.pauli_exyz}"
                )
        return tuple(labels)


def build_architecture_runtime(
    architecture: FixedArchitecture,
    *,
    pool: FixedVQEGeneratorPool,
    coefficient_tolerance: float = 1.0e-12,
) -> ArchitectureRuntime:
    terms = architecture_ansatz_terms(architecture, pool=pool)
    layout = build_parameter_layout(
        terms,
        ignore_identity=True,
        coefficient_tolerance=float(coefficient_tolerance),
        sort_terms=True,
    )
    executor = CompiledAnsatzExecutor(
        terms,
        coefficient_tolerance=float(coefficient_tolerance),
        ignore_identity=True,
        sort_terms=True,
        parameterization_mode=AP_PARAMETERIZATION_PER_PAULI_TERM,
        parameterization_layout=layout,
    )
    if int(executor.num_parameters) != int(layout.runtime_parameter_count):
        raise ValueError(
            "architecture executor coordinate count does not match the runtime layout."
        )
    return ArchitectureRuntime(
        architecture=architecture,
        terms=terms,
        layout=layout,
        executor=executor,
    )


# ---------------------------------------------------------------------------
# Construction/test modes
# ---------------------------------------------------------------------------


def exact_null_test_architecture(
    pool: FixedVQEGeneratorPool,
    *,
    atom_id: str | None = None,
    duplicate_count: int = 2,
    companion_atom_ids: Sequence[str] = (),
) -> FixedArchitecture:
    """Construct an architecture with an exactly degenerate tangent direction.

    Adjacent duplicate occurrences of one generator satisfy
    ``exp(-i a G) exp(-i b G) = exp(-i (a+b) G)``, so their horizontal tangents
    are exactly parallel at every angle and ``G`` carries an exact null space of
    dimension ``duplicate_count - 1`` per duplicated Pauli coordinate.  This is a
    construction/test mode for pseudoinverse and null-space validation; it is not
    the conventional fixed-VQE construction.
    """

    if int(duplicate_count) < 2:
        raise ValueError("exact-null test mode requires duplicate_count >= 2.")
    target = str(atom_id) if atom_id is not None else str(pool.atoms[0].atom_id)
    pool.by_id(target)
    layer: list[str] = []
    for companion in companion_atom_ids:
        pool.by_id(str(companion))
        layer.append(str(companion))
    layer.extend([target] * int(duplicate_count))
    return architecture_from_layers(
        [layer],
        construction_mode=CONSTRUCTION_MODE_EXACT_NULL_TEST,
        construction_notes={
            "duplicated_atom_id": str(target),
            "duplicate_count": int(duplicate_count),
            "expected_exact_nullity_per_pauli_coordinate": int(duplicate_count) - 1,
            "mechanism": "adjacent_identical_generator_occurrences",
            "is_conventional_fixed_vqe_construction": False,
        },
    )


def pauli_words_commute(left: str, right: str) -> bool:
    """Return whether two same-length ``e/x/y/z`` Pauli words commute."""

    a = str(left).strip().lower()
    b = str(right).strip().lower()
    if len(a) != len(b):
        raise ValueError("Pauli words must have the same length to compare commutation.")
    anticommuting_sites = sum(
        1
        for ca, cb in zip(a, b)
        if ca != "e" and cb != "e" and ca != cb
    )
    return bool(anticommuting_sites % 2 == 0)


DEFAULT_NEAR_NULL_COMPANION_COUNT = 4
DEFAULT_NEAR_NULL_SEPARATION_ANGLE = 1.0e-2


def near_null_test_architecture(
    pool: FixedVQEGeneratorPool,
    *,
    atom_id: str | None = None,
    separator_atom_id: str | None = None,
    companion_atom_ids: Sequence[str] | None = None,
    companion_count: int = DEFAULT_NEAR_NULL_COMPANION_COUNT,
) -> FixedArchitecture:
    """Construct a near-degenerate architecture with a small nonzero retained mode.

    Two occurrences of the same generator separated by one *non-commuting*
    generator at a small angle ``eps`` give tangents that are parallel only to
    first order in ``eps``, so the smallest retained Gram mode scales like
    ``eps^2`` while the rank stays nominally full -- in contrast with the
    exact-null mode's true null space.

    Companion generators are included because a bare three-coordinate
    architecture can saturate the reachable manifold from the reference state,
    which would turn the intended near-degeneracy into an exact one.
    """

    target = str(atom_id) if atom_id is not None else str(pool.atoms[0].atom_id)
    target_atom = pool.by_id(target)
    separator = separator_atom_id
    if separator is None:
        separator = _first_noncommuting_atom_id(pool, target_atom)
    if separator is None:
        raise ValueError(
            "near-null test mode requires a pool atom that does not commute with the "
            "repeated generator."
        )
    pool.by_id(str(separator))
    if companion_atom_ids is None:
        companions = [
            str(atom.atom_id)
            for atom in pool.atoms
            if atom.atom_id not in {target, str(separator)}
        ][: max(0, int(companion_count))]
    else:
        companions = [str(x) for x in companion_atom_ids]
        for companion in companions:
            pool.by_id(companion)
    return architecture_from_layers(
        [companions + [target, str(separator), target]],
        construction_mode=CONSTRUCTION_MODE_NEAR_NULL_TEST,
        construction_notes={
            "repeated_atom_id": str(target),
            "separator_atom_id": str(separator),
            "companion_atom_ids": list(companions),
            "mechanism": "repeated_generator_separated_by_one_noncommuting_generator",
            "separating_angle_scaling": "smallest_retained_gram_mode_scales_as_eps_squared",
            "suggested_separation_angle": float(DEFAULT_NEAR_NULL_SEPARATION_ANGLE),
            "is_conventional_fixed_vqe_construction": False,
        },
    )


def near_null_test_theta(
    architecture: FixedArchitecture,
    *,
    runtime: ArchitectureRuntime,
    separation_angle: float = DEFAULT_NEAR_NULL_SEPARATION_ANGLE,
    baseline_angle: float = 0.3,
) -> np.ndarray:
    """Return angles that realize the intended near-degeneracy for a test fixture."""

    if str(architecture.construction_mode) != CONSTRUCTION_MODE_NEAR_NULL_TEST:
        raise ValueError("near_null_test_theta requires a near-null test architecture.")
    separator_id = str(architecture.construction_notes.get("separator_atom_id", ""))
    theta = np.full(int(runtime.runtime_parameter_count), float(baseline_angle))
    for occurrence, block in zip(architecture.occurrences, runtime.layout.blocks):
        if str(occurrence.atom_id) != separator_id:
            continue
        theta[int(block.runtime_start) : int(block.runtime_stop)] = float(separation_angle)
    return theta


def select_near_null_test_architecture(
    ctx: FixedVQEConditioningContext,
    *,
    separation_angle: float = DEFAULT_NEAR_NULL_SEPARATION_ANGLE,
    baseline_angle: float = 0.3,
    companion_count: int = DEFAULT_NEAR_NULL_COMPANION_COUNT,
    max_candidates: int = 12,
) -> tuple[FixedArchitecture, np.ndarray, GramSpectrumRecord]:
    """Pick a near-null fixture whose Gram really does carry a small retained mode.

    Whether a repeated-generator pair lands in a *near*-degenerate or exactly
    degenerate configuration depends on the reachable manifold at the reference
    state, so this offline test-mode helper checks candidates against the actual
    Gram spectrum instead of assuming the outcome.
    """

    fallback: tuple[FixedArchitecture, np.ndarray, GramSpectrumRecord] | None = None
    for atom in ctx.pool.atoms[: max(1, int(max_candidates))]:
        try:
            architecture = near_null_test_architecture(
                ctx.pool,
                atom_id=str(atom.atom_id),
                companion_count=int(companion_count),
            )
        except ValueError:
            continue
        runtime = build_architecture_runtime(architecture, pool=ctx.pool)

        def evaluate(eps: float) -> tuple[np.ndarray, GramSpectrumRecord]:
            angles = near_null_test_theta(
                architecture,
                runtime=runtime,
                separation_angle=float(eps),
                baseline_angle=float(baseline_angle),
            )
            return angles, gram_spectrum_at(
                architecture,
                ctx=ctx,
                theta=angles,
                time=float(ctx.config.snapshots.times[0]),
                site="near_null_test",
                runtime=runtime,
            )

        theta, record = evaluate(float(separation_angle))
        if fallback is None:
            fallback = (architecture, theta, record)
        if (
            record.s_min_kept is None
            or record.s_min_kept <= record.retained_threshold
            or record.kappa_eff is None
            or float(record.kappa_eff) <= 1.0e2
        ):
            continue
        # Confirm the small retained mode is the intended near-degeneracy rather
        # than an unrelated soft direction: shrinking the separating angle by 10x
        # must shrink it by roughly 100x.
        _shrunk_theta, shrunk = evaluate(float(separation_angle) / 10.0)
        if shrunk.s_min_kept is None or shrunk.s_min_kept <= shrunk.retained_threshold:
            continue
        if float(shrunk.s_min_kept) > float(record.s_min_kept) / 10.0:
            continue
        return architecture, theta, record
    if fallback is None:
        raise ValueError("near-null test mode could not build any candidate fixture.")
    return fallback


def _first_noncommuting_atom_id(
    pool: FixedVQEGeneratorPool,
    target: FixedVQEGeneratorAtom,
) -> str | None:
    for atom in pool.atoms:
        if atom.atom_id == target.atom_id:
            continue
        if any(
            not pauli_words_commute(left, right)
            for left in target.pauli_words
            for right in atom.pauli_words
        ):
            return str(atom.atom_id)
    return None


# ---------------------------------------------------------------------------
# Construction context: problem, drive, exact ground state, exact snapshots
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExactSnapshot:
    """One cached exact driven state used only as an offline fit target."""

    index: int
    time: float
    state: np.ndarray
    energy: float

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            np.asarray(self.state, dtype="<c16").tobytes()
        ).hexdigest()

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "index": int(self.index),
            "time": float(self.time),
            "energy": float(self.energy),
            "state_sha256": str(self.digest),
        }


@dataclass(frozen=True)
class FixedVQEConditioningContext:
    """Immutable, shared, read-only construction inputs."""

    config: FixedVQEConditioningConfig
    cfg: ReplayRunConfig
    resolved_problem: Any
    static_poly: Any
    hamiltonian: TimeDependentHamiltonian
    psi_ref: np.ndarray
    static_matrix: np.ndarray
    exact_ground_energy: float
    exact_ground_state: np.ndarray
    snapshots: tuple[ExactSnapshot, ...]
    pool: FixedVQEGeneratorPool
    snapshot_trajectory_digest: str
    exact_reference_meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def dimension(self) -> int:
        return int(np.asarray(self.psi_ref, dtype=complex).reshape(-1).size)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema": FIXED_VQE_CONDITIONING_SCHEMA_V1,
            "config": self.config.to_json_dict(),
            "config_digest": str(self.config.config_digest()),
            "dimension": int(self.dimension),
            "exact_ground_energy": float(self.exact_ground_energy),
            "exact_reference": _json_safe(dict(self.exact_reference_meta)),
            "exact_snapshot_times": [float(s.time) for s in self.snapshots],
            "exact_snapshots": [s.to_json_dict() for s in self.snapshots],
            "exact_trajectory_digest": str(self.snapshot_trajectory_digest),
            "generator_pool": self.pool.to_json_dict(),
            "hamiltonian": self.hamiltonian.to_json_dict(),
            "exact_reference_scope": "offline_construction_only_never_online_controller",
        }


def build_conditioning_context(
    config: FixedVQEConditioningConfig,
    *,
    build_pool: bool = True,
) -> FixedVQEConditioningContext:
    """Resolve the problem, drive, exact ground state, snapshots, and pool once."""

    settings = config.model.to_settings_payload()
    cfg = _make_replay_run_cfg(
        {"settings": settings, "adapt_vqe": {}},
        artifact_json=Path("fixed_vqe_conditioning_construction.json"),
        tag="fixed_vqe_conditioning",
        generator_family=str(config.pool.pool_key),
        fallback_family="full_meta",
    )
    # Build the static Hamiltonian through the same helper the scaffold runtime
    # loader uses, so the constructed problem and the serialized artifact resolve
    # bit-identical operators.
    static_poly = _build_hh_hamiltonian(cfg)
    request = _problem_request_from_model(config.model)
    resolved_problem = resolve_problem_context(request, hamiltonian=static_poly)

    drive_model = None
    if bool(config.drive.enabled):
        drive_model = resolve_realtime_drive_model(
            resolved_problem=resolved_problem,
            drive_config=_drive_namespace(config),
        )
    hamiltonian = TimeDependentHamiltonian(
        static_poly=static_poly,
        drive_model=drive_model,
        metadata={
            "source": FIXED_VQE_CONDITIONING_SCHEMA_V1,
            "drive_source": "none" if drive_model is None else "resolve_realtime_drive_model",
        },
    )
    zero_drive_parity = assert_zero_drive_static_parity(hamiltonian)
    if zero_drive_parity is not None:
        hamiltonian = TimeDependentHamiltonian(
            static_poly=hamiltonian.static_poly,
            drive_model=hamiltonian.drive_model,
            metadata={
                **dict(hamiltonian.metadata or {}),
                "zero_drive_static_parity": zero_drive_parity,
            },
        )

    psi_ref = _normalize_state(
        resolved_problem.reference_state.build_state(), name="psi_ref"
    )
    static_matrix = np.asarray(hamiltonian_matrix(static_poly), dtype=complex)
    if static_matrix.shape != (int(psi_ref.size), int(psi_ref.size)):
        raise ValueError(
            "static Hamiltonian matrix does not match the reference-state dimension."
        )
    exact_ground_energy, exact_ground_state, exact_reference_meta = _exact_ground_reference(
        static_poly=static_poly,
        static_matrix=static_matrix,
        resolved_problem=resolved_problem,
        model=config.model,
    )

    snapshots = generate_exact_driven_snapshots(
        hamiltonian=hamiltonian,
        psi0=exact_ground_state,
        schedule=config.snapshots,
    )
    trajectory_digest = _digest_json(
        {
            "config_digest": str(config.config_digest()),
            "times": [float(s.time) for s in snapshots],
            "snapshot_state_sha256": [str(s.digest) for s in snapshots],
        }
    )
    pool = (
        build_fixed_vqe_generator_pool(
            cfg=cfg,
            h_poly=static_poly,
            model=config.model,
            pool_config=config.pool,
        )
        if bool(build_pool)
        else FixedVQEGeneratorPool(atoms=(), pool_key=str(config.pool.pool_key), parent_term_count=0)
    )
    return FixedVQEConditioningContext(
        config=config,
        cfg=cfg,
        resolved_problem=resolved_problem,
        static_poly=static_poly,
        hamiltonian=hamiltonian,
        psi_ref=psi_ref,
        static_matrix=static_matrix,
        exact_ground_energy=float(exact_ground_energy),
        exact_ground_state=exact_ground_state,
        snapshots=snapshots,
        pool=pool,
        snapshot_trajectory_digest=str(trajectory_digest),
        exact_reference_meta=dict(exact_reference_meta),
    )


def _exact_ground_reference(
    *,
    static_poly: Any,
    static_matrix: np.ndarray,
    resolved_problem: Any,
    model: FixedVQEModelConfig,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    """Resolve the same-cutoff, sector-filtered exact ground energy and state.

    The variational ansatz preserves the fixed particle/spin sector, so the
    qualification target must be the sector-filtered exact ground state at the
    same phonon cutoff, not the unrestricted spectrum minimum.
    """

    n_up, n_dn = model.num_particles
    exact_energy = float(
        _exact_gs_energy_for_problem(
            static_poly,
            problem="hh",
            num_sites=int(model.num_sites),
            num_particles=(int(n_up), int(n_dn)),
            indexing=str(model.ordering),
            n_ph_max=int(model.n_ph_max),
            boson_encoding=str(model.boson_encoding),
            t=float(model.t),
            u=float(model.u),
            dv=float(model.dv),
            omega0=float(model.omega0),
            g_ep=float(model.g_ep),
            boundary=str(model.boundary),
            include_zero_point=True,
        )
    )
    resolution = resolve_exact_reference_state_for_problem(
        static_poly,
        resolved_problem=resolved_problem,
    )
    if bool(getattr(resolution, "available", False)) and getattr(resolution, "state", None) is not None:
        state = _normalize_state(resolution.state, name="exact_ground_state")
        state_source = str(getattr(resolution, "source", "unknown"))
        state_energy = float(np.real(np.vdot(state, static_matrix @ state)))
    else:
        evals, evecs = np.linalg.eigh(np.asarray(static_matrix, dtype=complex))
        index = int(np.argmin(np.abs(np.asarray(evals, dtype=float) - exact_energy)))
        state = _normalize_state(evecs[:, index], name="exact_ground_state")
        state_source = "dense_eigenstate_nearest_sector_exact_energy"
        state_energy = float(np.real(evals[index]))
    meta = {
        "exact_energy_source": "sector_filtered_exact_ground_energy_same_cutoff",
        "exact_state_source": str(state_source),
        "exact_state_energy": float(state_energy),
        "exact_state_energy_delta": float(abs(state_energy - exact_energy)),
        "n_ph_work": int(model.n_ph_max),
        "num_particles": [int(n_up), int(n_dn)],
        "state_resolution_skip_reason": (
            None
            if bool(getattr(resolution, "available", False))
            else str(getattr(resolution, "skip_reason", "unavailable"))
        ),
    }
    return float(exact_energy), state, meta


def generate_exact_driven_snapshots(
    *,
    hamiltonian: TimeDependentHamiltonian,
    psi0: np.ndarray,
    schedule: SnapshotScheduleConfig,
) -> tuple[ExactSnapshot, ...]:
    """Propagate one exact driven trajectory and cache the requested snapshots.

    The propagation convention is the repository's post-run reference generator,
    so the snapshot targets match the exact trajectories used elsewhere for
    reporting.  This runs once per construction, not once per architecture.
    """

    provider = _dense_hamiltonian_provider(hamiltonian)
    times = np.asarray(schedule.times, dtype=float).reshape(-1)
    generation_config = schedule.to_reference_generation_config()
    if bool(hamiltonian.drive_enabled):
        states = _driven_reference_states(
            psi0=np.asarray(psi0, dtype=complex).reshape(-1),
            hamiltonian=provider,
            times=times,
            config=generation_config,
        )
    else:
        states = _static_reference_states(
            psi0=np.asarray(psi0, dtype=complex).reshape(-1),
            hmat=np.asarray(provider.matrix_at(float(times[0])), dtype=complex),
            times=times,
        )
    norm_drift_max = float(max(abs(float(np.linalg.norm(psi)) - 1.0) for psi in states))
    if norm_drift_max > float(schedule.norm_drift_tolerance):
        raise ValueError(
            "exact snapshot propagation norm drift exceeded tolerance: "
            f"{norm_drift_max:.3e} > {float(schedule.norm_drift_tolerance):.3e}."
        )
    out: list[ExactSnapshot] = []
    for index, (time_value, psi) in enumerate(zip(times, states)):
        state = _normalize_state(psi, name=f"exact_snapshot[{index}]")
        hmat = np.asarray(provider.matrix_at(float(time_value)), dtype=complex)
        out.append(
            ExactSnapshot(
                index=int(index),
                time=float(time_value),
                state=state,
                energy=float(np.real(np.vdot(state, hmat @ state))),
            )
        )
    return tuple(out)


def _problem_request_from_model(model: FixedVQEModelConfig) -> ProblemRequest:
    return ProblemRequest(
        problem_key="hh",
        num_sites=int(model.num_sites),
        t=float(model.t),
        u=float(model.u),
        dv=float(model.dv),
        omega0=float(model.omega0),
        g_ep=float(model.g_ep),
        n_ph_max=int(model.n_ph_max),
        boson_encoding=str(model.boson_encoding),
        ordering=str(model.ordering),
        boundary=str(model.boundary),
        include_zero_point=True,
    )


def _drive_namespace(config: FixedVQEConditioningConfig) -> Any:
    from types import SimpleNamespace

    drive = config.drive
    return SimpleNamespace(
        enabled=bool(drive.enabled),
        n_sites=int(config.model.num_sites),
        ordering=str(config.model.ordering),
        drive_A=float(drive.drive_A),
        drive_omega=float(drive.drive_omega),
        drive_tbar=float(drive.drive_tbar),
        drive_phi=float(drive.drive_phi),
        drive_pattern=str(drive.drive_pattern),
        drive_custom_weights=drive.drive_custom_weights,
        drive_include_identity=bool(drive.drive_include_identity),
        drive_time_sampling=str(drive.drive_time_sampling),
        drive_t0=float(drive.drive_t0),
    )


# ---------------------------------------------------------------------------
# Ground-state qualification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroundStateQualification:
    """Inner fixed-structure VQE result plus the hard energy gate outcome."""

    architecture_id: str
    theta: np.ndarray
    energy: float
    exact_energy: float
    delta_e: float
    qualified: bool
    delta_e_max: float
    optimizer_receipt: Mapping[str, Any]
    runtime_parameter_count: int
    prepared_state: np.ndarray

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "architecture_id": str(self.architecture_id),
            "energy": float(self.energy),
            "exact_energy": float(self.exact_energy),
            "delta_e": float(self.delta_e),
            "delta_e_max": float(self.delta_e_max),
            "qualified": bool(self.qualified),
            "runtime_parameter_count": int(self.runtime_parameter_count),
            "theta": [float(x) for x in np.asarray(self.theta, dtype=float).reshape(-1)],
            "optimizer_receipt": _json_safe(dict(self.optimizer_receipt)),
            "same_cutoff_exact_reference": True,
        }


def qualify_architecture_ground_state(
    architecture: FixedArchitecture,
    *,
    ctx: FixedVQEConditioningContext,
    runtime: ArchitectureRuntime | None = None,
) -> GroundStateQualification:
    """Run an ordinary fixed-structure VQE from the canonical HH reference state."""

    runtime = runtime or build_architecture_runtime(architecture, pool=ctx.pool)
    settings = ctx.config.ground_state
    result = vqe_minimize(
        ctx.static_poly,
        runtime.executor,
        np.asarray(ctx.psi_ref, dtype=complex).reshape(-1),
        restarts=int(settings.restarts),
        seed=int(settings.seed),
        initial_point=(
            np.zeros(int(runtime.runtime_parameter_count), dtype=float)
            if bool(settings.zero_first_restart)
            else None
        ),
        use_initial_point_first_restart=bool(settings.zero_first_restart),
        initial_point_stddev=float(settings.initial_point_stddev),
        method=str(settings.method),
        maxiter=int(settings.maxiter),
        bounds=settings.bounds,
        energy_backend=str(settings.energy_backend),
    )
    theta = np.asarray(result.theta, dtype=float).reshape(-1)
    prepared = _normalize_state(
        runtime.executor.prepare_state(theta, np.asarray(ctx.psi_ref, dtype=complex).reshape(-1)),
        name="ground_state_prepared",
    )
    energy = float(result.energy)
    delta_e = float(energy - float(ctx.exact_ground_energy))
    return GroundStateQualification(
        architecture_id=str(architecture.architecture_id),
        theta=theta,
        energy=energy,
        exact_energy=float(ctx.exact_ground_energy),
        delta_e=float(delta_e),
        qualified=bool(delta_e <= float(settings.delta_e_max)),
        delta_e_max=float(settings.delta_e_max),
        optimizer_receipt={
            "method": str(settings.method),
            "maxiter": int(settings.maxiter),
            "restarts": int(settings.restarts),
            "seed": int(settings.seed),
            "energy_backend": str(settings.energy_backend),
            "success": bool(result.success),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "nit": int(result.nit),
            "best_restart": int(result.best_restart),
        },
        runtime_parameter_count=int(runtime.runtime_parameter_count),
        prepared_state=prepared,
    )


# ---------------------------------------------------------------------------
# Independent exact-snapshot fits
# ---------------------------------------------------------------------------


def ray_distance(psi: np.ndarray, target: np.ndarray) -> float:
    """Return the Fubini--Study ray distance ``|| psi - <target|psi> target ||``.

    The orthogonal-projection form is numerically stable near unit overlap, where
    ``sqrt(1 - |<psi|target>|^2)`` amplifies roundoff.
    """

    a = np.asarray(psi, dtype=complex).reshape(-1)
    b = np.asarray(target, dtype=complex).reshape(-1)
    if int(a.size) != int(b.size):
        raise ValueError("ray distance requires equal-dimension states.")
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if not np.isfinite(a_norm) or a_norm <= 0.0 or not np.isfinite(b_norm) or b_norm <= 0.0:
        raise ValueError("ray distance requires positive finite state norms.")
    a_unit = a / a_norm
    b_unit = b / b_norm
    overlap = complex(np.vdot(b_unit, a_unit))
    return float(np.linalg.norm(a_unit - overlap * b_unit))


@dataclass(frozen=True)
class SnapshotFit:
    """One independent fit of the fixed architecture to one exact snapshot."""

    architecture_id: str
    snapshot_index: int
    time: float
    theta: np.ndarray
    ray_distance: float
    eligible: bool
    ray_distance_max: float
    optimizer_receipt: Mapping[str, Any]
    snapshot_state_sha256: str

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "architecture_id": str(self.architecture_id),
            "snapshot_index": int(self.snapshot_index),
            "time": float(self.time),
            "ray_distance": float(self.ray_distance),
            "ray_distance_max": float(self.ray_distance_max),
            "eligible": bool(self.eligible),
            "theta": [float(x) for x in np.asarray(self.theta, dtype=float).reshape(-1)],
            "optimizer_receipt": _json_safe(dict(self.optimizer_receipt)),
            "snapshot_state_sha256": str(self.snapshot_state_sha256),
            "independently_checked_against_target": True,
        }


def fit_architecture_to_snapshot(
    architecture: FixedArchitecture,
    *,
    ctx: FixedVQEConditioningContext,
    snapshot: ExactSnapshot,
    runtime: ArchitectureRuntime | None = None,
    warm_start: np.ndarray | None = None,
) -> SnapshotFit:
    """Independently minimize the ray distance to one exact snapshot."""

    runtime = runtime or build_architecture_runtime(architecture, pool=ctx.pool)
    settings = ctx.config.snapshot_fit
    psi_ref = np.asarray(ctx.psi_ref, dtype=complex).reshape(-1)
    target = np.asarray(snapshot.state, dtype=complex).reshape(-1)
    npar = int(runtime.runtime_parameter_count)

    def objective(x: np.ndarray) -> float:
        psi = runtime.executor.prepare_state(np.asarray(x, dtype=float).reshape(-1), psi_ref)
        return float(ray_distance(psi, target))

    # A per-snapshot seed keeps restarts deterministic and independent of the
    # order in which snapshots happen to be scheduled.
    rng = np.random.default_rng(int(settings.seed) + 1009 * int(snapshot.index))
    starts: list[np.ndarray] = []
    if warm_start is not None:
        starts.append(np.asarray(warm_start, dtype=float).reshape(-1))
    for _ in range(int(settings.restarts)):
        starts.append(float(settings.initial_point_stddev) * rng.normal(size=npar))

    best_value = float("inf")
    best_theta = np.zeros(npar, dtype=float)
    best_receipt: dict[str, Any] = {"backend": "none", "success": False, "message": "no run"}
    total_nfev = 0
    minimize = _try_import_scipy_minimize()
    for start_index, x0 in enumerate(starts):
        if int(np.asarray(x0, dtype=float).reshape(-1).size) != npar:
            raise ValueError("snapshot-fit start point has the wrong dimension.")
        if minimize is None:
            value = float(objective(x0))
            theta = np.asarray(x0, dtype=float).reshape(-1)
            receipt = {
                "backend": "objective_only_no_scipy",
                "success": False,
                "message": "scipy.optimize.minimize unavailable",
                "nfev": 1,
                "nit": 0,
                "start_index": int(start_index),
            }
            total_nfev += 1
        else:
            result = minimize(
                objective,
                np.asarray(x0, dtype=float).reshape(-1),
                method=str(settings.method),
                options={"maxiter": int(settings.maxiter)},
            )
            value = float(result.fun)
            theta = np.asarray(result.x, dtype=float).reshape(-1)
            receipt = {
                "backend": f"scipy_minimize_{str(settings.method).lower()}",
                "success": bool(getattr(result, "success", False)),
                "message": str(getattr(result, "message", "")),
                "nfev": int(getattr(result, "nfev", 0)),
                "nit": int(getattr(result, "nit", 0)),
                "start_index": int(start_index),
            }
            total_nfev += int(getattr(result, "nfev", 0))
        if value < best_value:
            best_value = float(value)
            best_theta = theta
            best_receipt = dict(receipt)

    # Recompute the reported distance from the returned angles rather than from
    # the optimizer's cached objective value, so every fit is checked
    # independently against its own target.
    checked_distance = float(
        ray_distance(runtime.executor.prepare_state(best_theta, psi_ref), target)
    )
    best_receipt.update(
        {
            "restarts": int(settings.restarts),
            "warm_started": bool(warm_start is not None),
            "seed": int(settings.seed),
            "nfev_total": int(total_nfev),
            "optimizer_reported_objective": float(best_value),
        }
    )
    return SnapshotFit(
        architecture_id=str(architecture.architecture_id),
        snapshot_index=int(snapshot.index),
        time=float(snapshot.time),
        theta=best_theta,
        ray_distance=float(checked_distance),
        eligible=bool(checked_distance <= float(settings.ray_distance_max)),
        ray_distance_max=float(settings.ray_distance_max),
        optimizer_receipt=best_receipt,
        snapshot_state_sha256=str(snapshot.digest),
    )


def fit_architecture_to_snapshots(
    architecture: FixedArchitecture,
    *,
    ctx: FixedVQEConditioningContext,
    runtime: ArchitectureRuntime | None = None,
    ground_state_theta: np.ndarray | None = None,
    max_workers: int = 1,
) -> tuple[SnapshotFit, ...]:
    """Fit every snapshot independently, returning results in snapshot order.

    Serial and parallel execution produce identical ordered results: warm starts
    are resolved up front from the ground-state angles only, so no fit depends on
    another fit's outcome, and results are reassembled by snapshot index.
    """

    runtime = runtime or build_architecture_runtime(architecture, pool=ctx.pool)
    settings = ctx.config.snapshot_fit
    warm = (
        np.asarray(ground_state_theta, dtype=float).reshape(-1)
        if (ground_state_theta is not None and bool(settings.warm_start_from_ground_state))
        else None
    )
    snapshots = tuple(ctx.snapshots)

    def run(snapshot: ExactSnapshot) -> SnapshotFit:
        return fit_architecture_to_snapshot(
            architecture,
            ctx=ctx,
            snapshot=snapshot,
            runtime=runtime,
            warm_start=warm,
        )

    workers = max(1, int(max_workers))
    if workers == 1 or len(snapshots) <= 1:
        fits = [run(snapshot) for snapshot in snapshots]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(snapshots))) as pool_executor:
            fits = list(pool_executor.map(run, snapshots))
    return tuple(sorted(fits, key=lambda fit: int(fit.snapshot_index)))


# ---------------------------------------------------------------------------
# Tangent Gram spectral diagnostics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GramSpectrumRecord:
    """Horizontally projected tangent Gram spectrum at one state."""

    architecture_id: str
    site: str
    snapshot_index: int | None
    time: float
    tangent_count: int
    rank: int
    nullity: int
    s_min_kept: float | None
    s_max: float
    kappa_eff: float | None
    retained_threshold: float
    retained_rcond: float
    ridge_lambda: float
    spectrum: tuple[float, ...]
    state_norm: float
    all_finite: bool
    retained_mask: tuple[bool, ...]

    @property
    def log10_kappa_eff(self) -> float | None:
        if self.kappa_eff is None or not np.isfinite(float(self.kappa_eff)) or float(self.kappa_eff) <= 0.0:
            return None
        return float(np.log10(float(self.kappa_eff)))

    @property
    def neg_log_smin_ratio(self) -> float | None:
        """Return ``-log(s_min_kept / s_max)``; larger means closer to singular."""

        if self.s_min_kept is None or float(self.s_max) <= 0.0 or float(self.s_min_kept) <= 0.0:
            return None
        return float(-np.log(float(self.s_min_kept) / float(self.s_max)))

    def to_json_dict(self, *, store_full_spectrum: bool = True) -> dict[str, Any]:
        payload = {
            "architecture_id": str(self.architecture_id),
            "site": str(self.site),
            "snapshot_index": (
                None if self.snapshot_index is None else int(self.snapshot_index)
            ),
            "time": float(self.time),
            "tangent_count": int(self.tangent_count),
            "rank": int(self.rank),
            "nullity": int(self.nullity),
            "s_min_kept": None if self.s_min_kept is None else float(self.s_min_kept),
            "s_max": float(self.s_max),
            "kappa_eff": None if self.kappa_eff is None else float(self.kappa_eff),
            "log10_kappa_eff": self.log10_kappa_eff,
            "neg_log_smin_over_smax": self.neg_log_smin_ratio,
            "retained_threshold": float(self.retained_threshold),
            "retained_rcond": float(self.retained_rcond),
            "ridge_lambda": float(self.ridge_lambda),
            "state_norm": float(self.state_norm),
            "all_finite": bool(self.all_finite),
        }
        if store_full_spectrum:
            payload["spectrum_sorted_desc"] = [float(x) for x in self.spectrum]
            payload["retained_mask_sorted_desc"] = [bool(x) for x in self.retained_mask]
        else:
            payload["spectrum_sha256"] = _digest_json([float(x) for x in self.spectrum])
        return payload


def gram_spectrum_at(
    architecture: FixedArchitecture,
    *,
    ctx: FixedVQEConditioningContext,
    theta: np.ndarray,
    time: float,
    site: str,
    snapshot_index: int | None = None,
    runtime: ArchitectureRuntime | None = None,
) -> GramSpectrumRecord:
    """Build ``G`` with the AP-McLachlan evaluator and record its spectrum.

    ``rank`` and ``nullity`` describe exact supported degeneracy under the
    declared retained-mode cutoff.  ``s_min_kept`` and ``kappa_eff`` describe
    near-null instability.  They are related but not interchangeable, so all of
    them are preserved.
    """

    runtime = runtime or build_architecture_runtime(architecture, pool=ctx.pool)
    state = _ap_state_for_runtime(ctx, runtime, theta)
    evaluation = evaluate_mclachlan_geometry(
        state=state,
        hamiltonian=ctx.hamiltonian,
        theta_runtime=np.asarray(theta, dtype=float).reshape(-1),
        time=float(time),
        metadata={"fixed_vqe_conditioning_site": str(site)},
    )
    gram = np.asarray(evaluation.geometry.K, dtype=float)
    policy = ctx.config.gram.inverse_policy
    inverse = supported_inverse(gram, policy=policy)
    eigenvalues = np.asarray(inverse.eigenvalues, dtype=float)
    retained = np.asarray(inverse.retained, dtype=bool)
    order = np.argsort(-np.abs(eigenvalues))
    spectrum_sorted = np.abs(eigenvalues[order])
    retained_sorted = retained[order]
    s_max = float(spectrum_sorted[0]) if spectrum_sorted.size else 0.0
    kept = spectrum_sorted[retained_sorted]
    s_min_kept = float(np.min(kept)) if kept.size else None
    kappa = (
        None
        if (s_min_kept is None or s_min_kept <= 0.0)
        else float(s_max / s_min_kept)
    )
    dimension = int(gram.shape[0])
    return GramSpectrumRecord(
        architecture_id=str(architecture.architecture_id),
        site=str(site),
        snapshot_index=None if snapshot_index is None else int(snapshot_index),
        time=float(time),
        tangent_count=int(dimension),
        rank=int(inverse.rank),
        nullity=int(dimension - int(inverse.rank)),
        s_min_kept=s_min_kept,
        s_max=float(s_max),
        kappa_eff=kappa,
        retained_threshold=float(inverse.retained_threshold),
        retained_rcond=float(policy.pinv_rcond),
        ridge_lambda=float(policy.ridge_lambda),
        spectrum=tuple(float(x) for x in spectrum_sorted.tolist()),
        state_norm=float(np.linalg.norm(np.asarray(evaluation.psi, dtype=complex))),
        all_finite=bool(np.all(np.isfinite(gram))),
        retained_mask=tuple(bool(x) for x in retained_sorted.tolist()),
    )


def _ap_state_for_runtime(
    ctx: FixedVQEConditioningContext,
    runtime: ArchitectureRuntime,
    theta: np.ndarray,
) -> APMcLachlanState:
    theta_arr = np.asarray(theta, dtype=float).reshape(-1)
    psi_ref = np.asarray(ctx.psi_ref, dtype=complex).reshape(-1)
    return APMcLachlanState(
        terms=runtime.terms,
        layout=runtime.layout,
        theta_runtime=theta_arr,
        psi_ref=psi_ref,
        psi_initial=runtime.executor.prepare_state(theta_arr, psi_ref),
        executor=runtime.executor,
        static_hamiltonian=ctx.static_poly,
        resolved_problem=ctx.resolved_problem,
        parameterization_mode=AP_PARAMETERIZATION_PER_PAULI_TERM,
        exact_energy=float(ctx.exact_ground_energy),
        provenance={"source": FIXED_VQE_CONDITIONING_SCHEMA_V1},
    )


# ---------------------------------------------------------------------------
# Conditioning aggregation and Pareto records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConditioningAggregate:
    """Worst / robust-mean / persistence aggregation over eligible snapshots."""

    eligible_snapshot_count: int
    sampled_snapshot_count: int
    worst_nullity: int | None
    worst_neg_log_smin_ratio: float | None
    worst_log10_kappa_eff: float | None
    mean_nullity: float | None
    robust_mean_neg_log_smin_ratio: float | None
    robust_mean_log10_kappa_eff: float | None
    warning_threshold_log10_kappa: float
    warning_exceeded_count: int
    longest_warning_run: int
    eligible_mask: tuple[bool, ...]
    active_mask: tuple[bool, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "eligible_snapshot_count": int(self.eligible_snapshot_count),
            "sampled_snapshot_count": int(self.sampled_snapshot_count),
            "worst_nullity": (
                None if self.worst_nullity is None else int(self.worst_nullity)
            ),
            "worst_neg_log_smin_over_smax": self.worst_neg_log_smin_ratio,
            "worst_log10_kappa_eff": self.worst_log10_kappa_eff,
            "mean_nullity": self.mean_nullity,
            "robust_mean_neg_log_smin_over_smax": self.robust_mean_neg_log_smin_ratio,
            "robust_mean_log10_kappa_eff": self.robust_mean_log10_kappa_eff,
            "conditioning_warning_log10_kappa": float(self.warning_threshold_log10_kappa),
            "warning_exceeded_count": int(self.warning_exceeded_count),
            "longest_warning_run": int(self.longest_warning_run),
            "eligible_snapshot_mask": [bool(x) for x in self.eligible_mask],
            "active_snapshot_mask": [bool(x) for x in self.active_mask],
            "robust_mean_definition": "median_over_eligible_snapshots",
        }


def aggregate_conditioning(
    *,
    fits: Sequence[SnapshotFit],
    records: Sequence[GramSpectrumRecord],
    warning_threshold_log10_kappa: float,
) -> ConditioningAggregate:
    """Aggregate time-local conditioning over the eligible snapshots only."""

    if len(tuple(fits)) != len(tuple(records)):
        raise ValueError("snapshot fits and Gram records must be index-aligned.")
    eligible_mask = tuple(bool(fit.eligible) for fit in fits)
    active_mask = tuple(
        bool(fit.eligible and record.all_finite) for fit, record in zip(fits, records)
    )
    eligible_records = [
        record for record, active in zip(records, active_mask) if active
    ]
    nullities = [int(record.nullity) for record in eligible_records]
    neg_logs = [
        float(record.neg_log_smin_ratio)
        for record in eligible_records
        if record.neg_log_smin_ratio is not None
    ]
    log_kappas = [
        float(record.log10_kappa_eff)
        for record in eligible_records
        if record.log10_kappa_eff is not None
    ]
    warning_flags = [
        bool(
            record.log10_kappa_eff is not None
            and float(record.log10_kappa_eff) > float(warning_threshold_log10_kappa)
        )
        for record in eligible_records
    ]
    longest_run = 0
    current_run = 0
    for flag in warning_flags:
        current_run = current_run + 1 if flag else 0
        longest_run = max(longest_run, current_run)
    return ConditioningAggregate(
        eligible_snapshot_count=int(len(eligible_records)),
        sampled_snapshot_count=int(len(tuple(records))),
        worst_nullity=(max(nullities) if nullities else None),
        worst_neg_log_smin_ratio=(max(neg_logs) if neg_logs else None),
        worst_log10_kappa_eff=(max(log_kappas) if log_kappas else None),
        mean_nullity=(float(np.mean(nullities)) if nullities else None),
        robust_mean_neg_log_smin_ratio=(float(np.median(neg_logs)) if neg_logs else None),
        robust_mean_log10_kappa_eff=(float(np.median(log_kappas)) if log_kappas else None),
        warning_threshold_log10_kappa=float(warning_threshold_log10_kappa),
        warning_exceeded_count=int(sum(1 for flag in warning_flags if flag)),
        longest_warning_run=int(longest_run),
        eligible_mask=eligible_mask,
        active_mask=active_mask,
    )


@dataclass(frozen=True)
class ArchitectureConditioningRecord:
    """Complete per-architecture construction record."""

    architecture: FixedArchitecture
    ground_state: GroundStateQualification
    ground_state_gram: GramSpectrumRecord | None
    snapshot_fits: tuple[SnapshotFit, ...]
    snapshot_grams: tuple[GramSpectrumRecord, ...]
    aggregate: ConditioningAggregate | None
    runtime_coordinate_labels: tuple[str, ...]
    stage: str

    @property
    def architecture_id(self) -> str:
        return str(self.architecture.architecture_id)

    @property
    def qualified(self) -> bool:
        return bool(self.ground_state.qualified)

    def pareto_objectives(self) -> tuple[float, float, float] | None:
        """Return the maximization objectives used by the conditioning frontier.

        ``(nullity, -log(s_min_kept/s_max), log10 kappa_eff)`` taken at the worst
        eligible snapshot.  ``None`` when no eligible snapshot exists, which keeps
        ineligible fits out of the frontier entirely.
        """

        aggregate = self.aggregate
        if aggregate is None or int(aggregate.eligible_snapshot_count) <= 0:
            return None
        if (
            aggregate.worst_nullity is None
            or aggregate.worst_neg_log_smin_ratio is None
            or aggregate.worst_log10_kappa_eff is None
        ):
            return None
        return (
            float(aggregate.worst_nullity),
            float(aggregate.worst_neg_log_smin_ratio),
            float(aggregate.worst_log10_kappa_eff),
        )

    def to_json_dict(self, *, store_full_spectrum: bool = True) -> dict[str, Any]:
        return {
            "schema": FIXED_VQE_CONDITIONING_SCHEMA_V1,
            "stage": str(self.stage),
            "architecture": self.architecture.to_json_dict(),
            "runtime_coordinate_labels": [str(x) for x in self.runtime_coordinate_labels],
            "ground_state": self.ground_state.to_json_dict(),
            "ground_state_gram": (
                None
                if self.ground_state_gram is None
                else self.ground_state_gram.to_json_dict(
                    store_full_spectrum=bool(store_full_spectrum)
                )
            ),
            "snapshot_fits": [fit.to_json_dict() for fit in self.snapshot_fits],
            "snapshot_grams": [
                record.to_json_dict(store_full_spectrum=bool(store_full_spectrum))
                for record in self.snapshot_grams
            ],
            "conditioning_aggregate": (
                None if self.aggregate is None else self.aggregate.to_json_dict()
            ),
            "pareto_objectives": (
                None
                if self.pareto_objectives() is None
                else {
                    "worst_nullity": float(self.pareto_objectives()[0]),
                    "worst_neg_log_smin_over_smax": float(self.pareto_objectives()[1]),
                    "worst_log10_kappa_eff": float(self.pareto_objectives()[2]),
                    "sense": "maximize",
                }
            ),
        }


def evaluate_architecture_stage_one(
    architecture: FixedArchitecture,
    *,
    ctx: FixedVQEConditioningContext,
) -> ArchitectureConditioningRecord:
    """Stage 1: VQE energy plus ground-state Gram geometry only."""

    runtime = build_architecture_runtime(architecture, pool=ctx.pool)
    qualification = qualify_architecture_ground_state(architecture, ctx=ctx, runtime=runtime)
    ground_gram = gram_spectrum_at(
        architecture,
        ctx=ctx,
        theta=qualification.theta,
        time=float(ctx.config.snapshots.times[0]),
        site="optimized_ground_state",
        runtime=runtime,
    )
    return ArchitectureConditioningRecord(
        architecture=architecture,
        ground_state=qualification,
        ground_state_gram=ground_gram,
        snapshot_fits=(),
        snapshot_grams=(),
        aggregate=None,
        runtime_coordinate_labels=runtime.runtime_coordinate_labels,
        stage="stage_1_ground_state",
    )


def evaluate_architecture_stage_two(
    record: ArchitectureConditioningRecord,
    *,
    ctx: FixedVQEConditioningContext,
    max_snapshot_workers: int = 1,
) -> ArchitectureConditioningRecord:
    """Stage 2: exact-snapshot fitting and time-distributed Gram geometry."""

    if not record.qualified:
        raise ValueError(
            "stage 2 requires a ground-state-qualified architecture; the delta-E gate "
            "is not tradeable against conditioning."
        )
    architecture = record.architecture
    runtime = build_architecture_runtime(architecture, pool=ctx.pool)
    fits = fit_architecture_to_snapshots(
        architecture,
        ctx=ctx,
        runtime=runtime,
        ground_state_theta=record.ground_state.theta,
        max_workers=int(max_snapshot_workers),
    )
    grams = tuple(
        gram_spectrum_at(
            architecture,
            ctx=ctx,
            theta=fit.theta,
            time=float(fit.time),
            site="driven_snapshot",
            snapshot_index=int(fit.snapshot_index),
            runtime=runtime,
        )
        for fit in fits
    )
    aggregate = aggregate_conditioning(
        fits=fits,
        records=grams,
        warning_threshold_log10_kappa=float(
            ctx.config.gram.conditioning_warning_log10_kappa
        ),
    )
    return replace(
        record,
        snapshot_fits=fits,
        snapshot_grams=grams,
        aggregate=aggregate,
        stage="stage_2_driven_snapshots",
    )


def pareto_front(
    records: Sequence[ArchitectureConditioningRecord],
) -> tuple[ArchitectureConditioningRecord, ...]:
    """Return the nondominated conditioning records, deterministically ordered.

    Only ground-state-qualified architectures with at least one eligible snapshot
    can enter.  No upper conditioning bound is imposed and no single winner is
    declared; selecting among nondominated instances is the user's decision.
    """

    scored = [
        (record, record.pareto_objectives())
        for record in records
        if record.qualified and record.pareto_objectives() is not None
    ]
    front: list[ArchitectureConditioningRecord] = []
    for record, objectives in scored:
        dominated = False
        for other, other_objectives in scored:
            if other is record:
                continue
            if _dominates(other_objectives, objectives):
                dominated = True
                break
        if not dominated:
            front.append(record)
    return tuple(
        sorted(
            front,
            key=lambda item: (
                -float(item.pareto_objectives()[0]),
                -float(item.pareto_objectives()[1]),
                -float(item.pareto_objectives()[2]),
                str(item.architecture_id),
            ),
        )
    )


def _dominates(lhs: tuple[float, ...] | None, rhs: tuple[float, ...] | None) -> bool:
    if lhs is None or rhs is None:
        return False
    return bool(
        all(float(a) >= float(b) for a, b in zip(lhs, rhs))
        and any(float(a) > float(b) for a, b in zip(lhs, rhs))
    )


# ---------------------------------------------------------------------------
# Deterministic, restartable architecture search
# ---------------------------------------------------------------------------


def atom_ids_by_parent(pool: FixedVQEGeneratorPool) -> dict[str, tuple[str, ...]]:
    """Group pool atom ids under their originating ``full_meta`` parent label."""

    grouped: dict[str, list[str]] = {}
    for atom in pool.atoms:
        grouped.setdefault(str(atom.parent_label), []).append(str(atom.atom_id))
    return {label: tuple(ids) for label, ids in grouped.items()}


def parent_complete_layer(
    pool: FixedVQEGeneratorPool,
    parent_labels: Sequence[str],
) -> tuple[str, ...]:
    """Return the ordered children of the requested parents as one layer.

    A parent's complete ordered Pauli-child sequence reproduces that parent's
    termwise product, so parent-complete layers are the conventional termwise
    HVA/Hamiltonian layer constructions expressed in child vocabulary.
    """

    grouped = atom_ids_by_parent(pool)
    out: list[str] = []
    for label in parent_labels:
        out.extend(grouped.get(str(label), ()))
    return tuple(out)


def default_seed_parent_labels(pool: FixedVQEGeneratorPool) -> tuple[str, ...]:
    """Pick the conventional Hubbard-Holstein layer parents when present."""

    grouped = atom_ids_by_parent(pool)
    preferred = [
        label
        for label in ("hop_layer", "onsite_layer", "eph_layer")
        if label in grouped
    ]
    if preferred:
        return tuple(preferred)
    return tuple(sorted(grouped)[: min(3, len(grouped))])


def enumerate_seed_architectures(
    *,
    pool: FixedVQEGeneratorPool,
    search: ArchitectureSearchConfig,
) -> tuple[FixedArchitecture, ...]:
    """Deterministically seed the outer search population."""

    rng = np.random.default_rng(int(search.seed))
    atom_ids = list(pool.atom_ids)
    out: list[FixedArchitecture] = []
    seen: set[str] = set()

    if bool(search.seed_parent_complete_layers):
        parent_labels = (
            tuple(search.seed_parent_labels)
            if search.seed_parent_labels
            else default_seed_parent_labels(pool)
        )
        base_layer = parent_complete_layer(pool, parent_labels)
        if base_layer:
            for repeats in search.seed_parent_complete_repeats:
                architecture = architecture_from_layers(
                    [list(base_layer)] * int(repeats),
                    construction_mode=str(search.construction_mode),
                    construction_notes={
                        "origin": "parent_complete_layers",
                        "parent_labels": [str(x) for x in parent_labels],
                        "layer_repeats": int(repeats),
                    },
                )
                if architecture.architecture_id in seen:
                    continue
                seen.add(architecture.architecture_id)
                out.append(architecture)

    attempts = 0
    max_attempts = int(search.population_size) * 40
    while len(out) < int(search.population_size) and attempts < max_attempts:
        attempts += 1
        layer_count = int(search.layer_counts[int(rng.integers(len(search.layer_counts)))])
        width = int(search.atoms_per_layer[int(rng.integers(len(search.atoms_per_layer)))])
        layers = [
            [str(atom_ids[int(rng.integers(len(atom_ids)))]) for _ in range(width)]
            for _ in range(layer_count)
        ]
        if not bool(search.allow_repeated_occurrences):
            flat = [atom for layer in layers for atom in layer]
            if len(set(flat)) != len(flat):
                continue
        architecture = architecture_from_layers(
            layers,
            construction_mode=str(search.construction_mode),
            construction_notes={"origin": "seed_population", "attempt": int(attempts)},
        )
        if architecture.architecture_id in seen:
            continue
        seen.add(architecture.architecture_id)
        out.append(architecture)
    if not out:
        raise ValueError("architecture search could not seed any distinct architecture.")
    return tuple(out)


def mutate_architecture(
    architecture: FixedArchitecture,
    *,
    pool: FixedVQEGeneratorPool,
    search: ArchitectureSearchConfig,
    rng: np.random.Generator,
) -> FixedArchitecture:
    """Return a deterministic single-parent mutation of a fixed architecture."""

    atom_ids = list(pool.atom_ids)
    layers = [list(layer) for layer in _layer_atom_ids(architecture)]
    for _ in range(int(max(1, search.mutation_count))):
        choice = int(rng.integers(3))
        layer_index = int(rng.integers(len(layers)))
        if choice == 0 and layers[layer_index]:
            position = int(rng.integers(len(layers[layer_index])))
            layers[layer_index][position] = str(atom_ids[int(rng.integers(len(atom_ids)))])
        elif choice == 1:
            width_cap = int(max(search.atoms_per_layer))
            if len(layers[layer_index]) < width_cap:
                layers[layer_index].append(str(atom_ids[int(rng.integers(len(atom_ids)))]))
        elif len(layers[layer_index]) > 1:
            position = int(rng.integers(len(layers[layer_index])))
            layers[layer_index].pop(position)
    layers = [layer for layer in layers if layer]
    if not layers:
        layers = [[str(atom_ids[0])]]
    return architecture_from_layers(
        layers,
        construction_mode=str(search.construction_mode),
        construction_notes={"origin": "mutation", "parent_id": str(architecture.architecture_id)},
    )


def _layer_atom_ids(architecture: FixedArchitecture) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(str(occ.atom_id) for occ in layer) for layer in architecture.layers()
    )


@dataclass(frozen=True)
class SearchBatchStore:
    """Incremental JSONL persistence so interruption never loses finished work."""

    path: Path

    def completed_architecture_ids(self) -> set[str]:
        if not self.path.exists():
            return set()
        done: set[str] = set()
        for line in self.path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                # A partially written trailing record from an interrupted run is
                # dropped rather than treated as completed work.
                continue
            architecture = payload.get("architecture", {})
            if isinstance(architecture, Mapping):
                identifier = architecture.get("architecture_id")
                if identifier:
                    done.add(str(identifier))
        return done

    def load_records(self) -> tuple[dict[str, Any], ...]:
        if not self.path.exists():
            return tuple()
        out: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text:
                continue
            try:
                out.append(dict(json.loads(text)))
            except json.JSONDecodeError:
                continue
        return tuple(out)

    def append(self, payload: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_safe(dict(payload)), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


@dataclass(frozen=True)
class FixedVQEConditioningSearchResult:
    """In-memory search result plus the JSON manifest that was written."""

    context: FixedVQEConditioningContext
    records: tuple[ArchitectureConditioningRecord, ...]
    pareto_front: tuple[ArchitectureConditioningRecord, ...]
    retained: tuple[ArchitectureConditioningRecord, ...]
    manifest: Mapping[str, Any]
    manifest_path: Path
    batches_path: Path


def record_from_payload(
    payload: Mapping[str, Any],
    *,
    ctx: FixedVQEConditioningContext,
) -> ArchitectureConditioningRecord:
    """Rebuild a persisted search record so resumed runs keep completed work.

    Everything needed to re-report and re-serialize an architecture is persisted,
    so a resumed run reconstructs its records instead of dropping them from the
    frontier or re-running their optimizations.
    """

    architecture_payload = dict(payload.get("architecture") or {})
    layers: list[list[str]] = [
        [] for _ in range(int(architecture_payload.get("layer_count", 1)))
    ]
    for occurrence in architecture_payload.get("occurrences", []):
        layers[int(occurrence["layer_index"])].append(str(occurrence["atom_id"]))
    architecture = architecture_from_layers(
        layers,
        construction_mode=str(
            architecture_payload.get("construction_mode", CONSTRUCTION_MODE_CONVENTIONAL)
        ),
        construction_notes=dict(architecture_payload.get("construction_notes") or {}),
    )
    if str(architecture.architecture_id) != str(architecture_payload.get("architecture_id")):
        raise ValueError(
            "persisted architecture id does not match the reconstructed architecture."
        )
    runtime = build_architecture_runtime(architecture, pool=ctx.pool)
    ground_payload = dict(payload.get("ground_state") or {})
    theta = np.asarray(ground_payload.get("theta", []), dtype=float).reshape(-1)
    prepared = _normalize_state(
        runtime.executor.prepare_state(
            theta, np.asarray(ctx.psi_ref, dtype=complex).reshape(-1)
        ),
        name="resumed_ground_state_prepared",
    )
    ground_state = GroundStateQualification(
        architecture_id=str(architecture.architecture_id),
        theta=theta,
        energy=float(ground_payload.get("energy", float("nan"))),
        exact_energy=float(ground_payload.get("exact_energy", ctx.exact_ground_energy)),
        delta_e=float(ground_payload.get("delta_e", float("nan"))),
        qualified=bool(ground_payload.get("qualified", False)),
        delta_e_max=float(
            ground_payload.get("delta_e_max", ctx.config.ground_state.delta_e_max)
        ),
        optimizer_receipt=dict(ground_payload.get("optimizer_receipt") or {}),
        runtime_parameter_count=int(
            ground_payload.get("runtime_parameter_count", runtime.runtime_parameter_count)
        ),
        prepared_state=prepared,
    )
    aggregate_payload = payload.get("conditioning_aggregate")
    return ArchitectureConditioningRecord(
        architecture=architecture,
        ground_state=ground_state,
        ground_state_gram=_gram_from_payload(payload.get("ground_state_gram")),
        snapshot_fits=tuple(
            _snapshot_fit_from_payload(item) for item in payload.get("snapshot_fits", [])
        ),
        snapshot_grams=tuple(
            _gram_from_payload(item) for item in payload.get("snapshot_grams", [])
        ),
        aggregate=(
            None
            if not isinstance(aggregate_payload, Mapping)
            else _aggregate_from_payload(aggregate_payload)
        ),
        runtime_coordinate_labels=tuple(
            str(x) for x in payload.get("runtime_coordinate_labels", [])
        ),
        stage=str(payload.get("stage", "stage_1_ground_state")),
    )


def _gram_from_payload(payload: Any) -> GramSpectrumRecord | None:
    if not isinstance(payload, Mapping):
        return None
    return GramSpectrumRecord(
        architecture_id=str(payload.get("architecture_id", "")),
        site=str(payload.get("site", "")),
        snapshot_index=(
            None if payload.get("snapshot_index") is None else int(payload["snapshot_index"])
        ),
        time=float(payload.get("time", 0.0)),
        tangent_count=int(payload.get("tangent_count", 0)),
        rank=int(payload.get("rank", 0)),
        nullity=int(payload.get("nullity", 0)),
        s_min_kept=(
            None if payload.get("s_min_kept") is None else float(payload["s_min_kept"])
        ),
        s_max=float(payload.get("s_max", 0.0)),
        kappa_eff=(None if payload.get("kappa_eff") is None else float(payload["kappa_eff"])),
        retained_threshold=float(payload.get("retained_threshold", 0.0)),
        retained_rcond=float(payload.get("retained_rcond", DEFAULT_GRAM_RETAINED_RCOND)),
        ridge_lambda=float(payload.get("ridge_lambda", GRAM_DIAGNOSTIC_RIDGE_LAMBDA)),
        spectrum=tuple(float(x) for x in payload.get("spectrum_sorted_desc", [])),
        state_norm=float(payload.get("state_norm", 1.0)),
        all_finite=bool(payload.get("all_finite", True)),
        retained_mask=tuple(bool(x) for x in payload.get("retained_mask_sorted_desc", [])),
    )


def _snapshot_fit_from_payload(payload: Mapping[str, Any]) -> SnapshotFit:
    return SnapshotFit(
        architecture_id=str(payload.get("architecture_id", "")),
        snapshot_index=int(payload.get("snapshot_index", 0)),
        time=float(payload.get("time", 0.0)),
        theta=np.asarray(payload.get("theta", []), dtype=float).reshape(-1),
        ray_distance=float(payload.get("ray_distance", float("inf"))),
        eligible=bool(payload.get("eligible", False)),
        ray_distance_max=float(
            payload.get("ray_distance_max", DEFAULT_SNAPSHOT_RAY_DISTANCE_MAX)
        ),
        optimizer_receipt=dict(payload.get("optimizer_receipt") or {}),
        snapshot_state_sha256=str(payload.get("snapshot_state_sha256", "")),
    )


def _aggregate_from_payload(payload: Mapping[str, Any]) -> ConditioningAggregate:
    return ConditioningAggregate(
        eligible_snapshot_count=int(payload.get("eligible_snapshot_count", 0)),
        sampled_snapshot_count=int(payload.get("sampled_snapshot_count", 0)),
        worst_nullity=(
            None if payload.get("worst_nullity") is None else int(payload["worst_nullity"])
        ),
        worst_neg_log_smin_ratio=(
            None
            if payload.get("worst_neg_log_smin_over_smax") is None
            else float(payload["worst_neg_log_smin_over_smax"])
        ),
        worst_log10_kappa_eff=(
            None
            if payload.get("worst_log10_kappa_eff") is None
            else float(payload["worst_log10_kappa_eff"])
        ),
        mean_nullity=(
            None if payload.get("mean_nullity") is None else float(payload["mean_nullity"])
        ),
        robust_mean_neg_log_smin_ratio=(
            None
            if payload.get("robust_mean_neg_log_smin_over_smax") is None
            else float(payload["robust_mean_neg_log_smin_over_smax"])
        ),
        robust_mean_log10_kappa_eff=(
            None
            if payload.get("robust_mean_log10_kappa_eff") is None
            else float(payload["robust_mean_log10_kappa_eff"])
        ),
        warning_threshold_log10_kappa=float(
            payload.get("conditioning_warning_log10_kappa", 8.0)
        ),
        warning_exceeded_count=int(payload.get("warning_exceeded_count", 0)),
        longest_warning_run=int(payload.get("longest_warning_run", 0)),
        eligible_mask=tuple(bool(x) for x in payload.get("eligible_snapshot_mask", [])),
        active_mask=tuple(bool(x) for x in payload.get("active_snapshot_mask", [])),
    )


def run_fixed_vqe_conditioning_search(
    config: FixedVQEConditioningConfig,
    *,
    output_dir: str | Path,
    ctx: FixedVQEConditioningContext | None = None,
    resume: bool = True,
    extra_architectures: Sequence[FixedArchitecture] = (),
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> FixedVQEConditioningSearchResult:
    """Run the deterministic two-stage architecture search with incremental saves."""

    ctx = ctx or build_conditioning_context(config)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    store = SearchBatchStore(path=out_dir / "search_batches.jsonl")
    if not bool(resume) and store.path.exists():
        store.path.unlink()

    search = config.search
    rng = np.random.default_rng(int(search.seed))
    population = list(enumerate_seed_architectures(pool=ctx.pool, search=search))
    population.extend(extra_architectures)

    # Rebuild completed work so a resumed run reports the same frontier as an
    # uninterrupted one instead of silently dropping finished architectures.
    records: list[ArchitectureConditioningRecord] = []
    completed: set[str] = set()
    if bool(resume):
        for persisted in store.load_records():
            try:
                records.append(record_from_payload(persisted, ctx=ctx))
            except (KeyError, ValueError) as exc:
                if progress is not None:
                    progress(
                        {
                            "event": "resume_record_skipped",
                            "reason": str(exc),
                        }
                    )
                continue
            completed.add(str(records[-1].architecture_id))
    evaluated_ids: set[str] = set()
    newly_evaluated = 0
    for generation in range(int(search.generations)):
        batch = [
            architecture
            for architecture in population
            if architecture.architecture_id not in evaluated_ids
        ]
        for architecture in batch:
            evaluated_ids.add(architecture.architecture_id)
            if architecture.architecture_id in completed:
                if progress is not None:
                    progress(
                        {
                            "event": "architecture_skipped_resume",
                            "generation": int(generation),
                            "architecture_id": str(architecture.architecture_id),
                        }
                    )
                continue
            record = evaluate_architecture_stage_one(architecture, ctx=ctx)
            if record.qualified:
                record = evaluate_architecture_stage_two(
                    record,
                    ctx=ctx,
                    max_snapshot_workers=int(search.max_snapshot_workers),
                )
            records.append(record)
            newly_evaluated += 1
            payload = record.to_json_dict(
                store_full_spectrum=bool(config.gram.store_full_spectrum)
            )
            payload["generation"] = int(generation)
            store.append(payload)
            if progress is not None:
                progress(
                    {
                        "event": "architecture_evaluated",
                        "generation": int(generation),
                        "architecture_id": str(record.architecture_id),
                        "qualified": bool(record.qualified),
                        "delta_e": float(record.ground_state.delta_e),
                        "stage": str(record.stage),
                    }
                )
        front = pareto_front(records)
        parents = list(front) or [
            record
            for record in sorted(records, key=lambda item: float(item.ground_state.delta_e))[:2]
        ]
        population = []
        for parent in parents:
            population.append(parent.architecture)
        while len(population) < int(search.population_size) and parents:
            parent = parents[int(rng.integers(len(parents)))]
            population.append(
                mutate_architecture(
                    parent.architecture,
                    pool=ctx.pool,
                    search=search,
                    rng=rng,
                )
            )

    front = pareto_front(records)
    retained = _explicitly_retained_records(
        records,
        front=front,
        limit=int(search.retain_beyond_pareto),
    )
    manifest = {
        "schema": FIXED_VQE_CONDITIONING_SCHEMA_V1,
        "construction": ctx.to_json_dict(),
        "search_batches_jsonl": str(store.path),
        "architecture_record_count": int(len(records)),
        "evaluated_architecture_count": int(newly_evaluated),
        "resumed_architecture_count": int(len(completed)),
        "ground_state_qualified_count": int(sum(1 for r in records if r.qualified)),
        "pareto_front_architecture_ids": [str(r.architecture_id) for r in front],
        "pareto_front": [
            r.to_json_dict(store_full_spectrum=bool(config.gram.store_full_spectrum))
            for r in front
        ],
        "explicitly_retained_architecture_ids": [
            str(r.architecture_id) for r in retained
        ],
        "selection_policy": (
            "nondominated_set_reported_without_declaring_a_single_winner"
        ),
        "decision_data_flow": {
            "uses_exact_reference_offline_for_construction": True,
            "uses_exact_reference_for_online_control": False,
            "uses_adapt_selection": False,
            "grows_ansatz_during_inner_vqe": False,
        },
    }
    manifest_path = out_dir / "fixed_vqe_conditioning_manifest.json"
    manifest_path.write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest["manifest_json"] = str(manifest_path)
    return FixedVQEConditioningSearchResult(
        context=ctx,
        records=tuple(records),
        pareto_front=tuple(front),
        retained=tuple(retained),
        manifest=manifest,
        manifest_path=manifest_path,
        batches_path=store.path,
    )


def _explicitly_retained_records(
    records: Sequence[ArchitectureConditioningRecord],
    *,
    front: Sequence[ArchitectureConditioningRecord],
    limit: int,
) -> tuple[ArchitectureConditioningRecord, ...]:
    """Return qualified non-frontier records kept for the user's own inspection."""

    if int(limit) <= 0:
        return tuple()
    front_ids = {str(record.architecture_id) for record in front}
    candidates = [
        record
        for record in records
        if record.qualified
        and record.pareto_objectives() is not None
        and str(record.architecture_id) not in front_ids
    ]
    candidates.sort(
        key=lambda item: (
            -float(item.pareto_objectives()[2]),
            -float(item.pareto_objectives()[1]),
            str(item.architecture_id),
        )
    )
    return tuple(candidates[: int(limit)])


# ---------------------------------------------------------------------------
# Runtime-loadable artifact serialization
# ---------------------------------------------------------------------------


def build_fixed_vqe_stress_artifact_payload(
    record: ArchitectureConditioningRecord,
    *,
    ctx: FixedVQEConditioningContext,
    subject_kind: str = FIXED_VQE_STRESS_SUBJECT_KIND_V1,
) -> dict[str, Any]:
    """Serialize one architecture as an ordinary locked fixed-scaffold artifact.

    The result loads through the normal Paper-II scaffold runtime loader with
    ``loader_mode='fixed_scaffold'``; it adds no second interpretation of
    ``U(theta)|psi_ref>``.  Exact-reference data appears only as digests and
    scalar diagnostics under the diagnostic sidecar.
    """

    runtime = build_architecture_runtime(record.architecture, pool=ctx.pool)
    theta = np.asarray(record.ground_state.theta, dtype=float).reshape(-1)
    if int(theta.size) != int(runtime.runtime_parameter_count):
        raise ValueError("serialized theta length does not match the runtime layout.")
    layout_payload = serialize_layout(runtime.layout)
    theta_logical = project_runtime_theta_block_mean(theta, runtime.layout)
    psi_ref = np.asarray(ctx.psi_ref, dtype=complex).reshape(-1)
    psi_initial = _normalize_state(
        runtime.executor.prepare_state(theta, psi_ref), name="artifact_psi_initial"
    )
    runtime_labels = [
        str(term.get("pauli_exyz", ""))
        for block in layout_payload["blocks"]
        for term in block["runtime_terms_exyz"]
    ]
    settings = ctx.config.model.to_settings_payload()
    settings["adapt_pool"] = "fixed_scaffold_locked"

    payload: dict[str, Any] = {
        "schema_version": FIXED_VQE_CONDITIONING_ARTIFACT_SCHEMA_V1,
        "pipeline": "fixed_vqe_conditioning_stress_builder_v1",
        "settings": settings,
        "ansatz_input_state": _statevector_payload(
            psi_ref,
            source="hh_canonical_reference_state",
            handoff_state_kind="reference_state",
        ),
        "initial_state": _statevector_payload(
            psi_initial,
            source="fixed_vqe_conditioning_ground_state",
            handoff_state_kind="prepared_state",
        ),
        "adapt_vqe": {
            "pool_type": "fixed_scaffold_locked",
            "structure_locked": True,
            "fixed_scaffold_kind": str(subject_kind),
            "parameterization": layout_payload,
            "parameterization_mode": AP_PARAMETERIZATION_PER_PAULI_TERM,
            "parameterization_execution_mode": AP_PARAMETERIZATION_PER_PAULI_TERM,
            "num_parameters": int(runtime.runtime_parameter_count),
            "logical_num_parameters": int(runtime.layout.logical_parameter_count),
            "ansatz_depth": int(len(runtime.layout.blocks)),
            "optimal_point": [float(x) for x in theta.tolist()],
            "logical_optimal_point": [float(x) for x in np.asarray(theta_logical).tolist()],
            "energy": float(record.ground_state.energy),
            "num_particles": {
                "n_up": int(ctx.config.model.num_particles[0]),
                "n_dn": int(ctx.config.model.num_particles[1]),
            },
            "fixed_scaffold_metadata": {
                "schema_version": 1,
                "route_family": FIXED_VQE_STRESS_ROUTE_FAMILY,
                "subject_kind": str(subject_kind),
                "structure_locked": True,
                "operator_count": int(len(runtime.layout.blocks)),
                "runtime_term_count": int(runtime.runtime_parameter_count),
                "term_order_id": "fixed_vqe_conditioning_layer_order",
                "term_order_basis": "fixed_layered_full_meta_child_occurrences",
                "source_order_runtime_indices": list(range(len(runtime_labels))),
                "source_order_runtime_term_labels_exyz": list(runtime_labels),
                "runtime_term_labels_exyz": list(runtime_labels),
                "source_artifact_json": None,
                "source_pool_type": str(ctx.pool.pool_key),
            },
        },
        "exact_energy": float(ctx.exact_ground_energy),
        "fixed_vqe_conditioning_stress": _artifact_diagnostic_sidecar(record, ctx=ctx),
    }
    payload["source_hashes"] = {
        "config_digest": str(ctx.config.config_digest()),
        "generator_pool_ordered_atom_contract_sha256": str(
            ctx.pool.ordered_atom_contract_sha256
        ),
        "exact_trajectory_digest": str(ctx.snapshot_trajectory_digest),
        "architecture_id": str(record.architecture_id),
        "parameterization_sha256": _digest_json(layout_payload),
        "theta_sha256": hashlib.sha256(
            np.asarray(theta, dtype="<f8").tobytes()
        ).hexdigest(),
    }
    return payload


def _artifact_diagnostic_sidecar(
    record: ArchitectureConditioningRecord,
    *,
    ctx: FixedVQEConditioningContext,
) -> dict[str, Any]:
    store_full = bool(ctx.config.gram.store_full_spectrum)
    return {
        "schema": FIXED_VQE_CONDITIONING_SCHEMA_V1,
        "construction_mode": str(record.architecture.construction_mode),
        "config": ctx.config.to_json_dict(),
        "generator_pool": ctx.pool.to_json_dict(),
        "architecture": record.architecture.to_json_dict(),
        "generator_provenance": [
            ctx.pool.by_id(occ.atom_id).to_json_dict()
            for occ in record.architecture.occurrences
        ],
        "runtime_coordinate_labels": [str(x) for x in record.runtime_coordinate_labels],
        "ground_state": record.ground_state.to_json_dict(),
        "ground_state_gram": (
            None
            if record.ground_state_gram is None
            else record.ground_state_gram.to_json_dict(store_full_spectrum=store_full)
        ),
        "exact_snapshot_times": [float(s.time) for s in ctx.snapshots],
        "exact_trajectory_digest": str(ctx.snapshot_trajectory_digest),
        "snapshot_fits": [fit.to_json_dict() for fit in record.snapshot_fits],
        "snapshot_grams": [
            gram.to_json_dict(store_full_spectrum=store_full)
            for gram in record.snapshot_grams
        ],
        "conditioning_aggregate": (
            None if record.aggregate is None else record.aggregate.to_json_dict()
        ),
        "exact_reference_scope": "offline_construction_only",
        "online_controller_receives_exact_reference": False,
    }


def write_fixed_vqe_stress_artifact(
    record: ArchitectureConditioningRecord,
    *,
    ctx: FixedVQEConditioningContext,
    output_json: str | Path,
    subject_kind: str = FIXED_VQE_STRESS_SUBJECT_KIND_V1,
) -> Path:
    payload = build_fixed_vqe_stress_artifact_payload(
        record, ctx=ctx, subject_kind=str(subject_kind)
    )
    path = Path(output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def fixed_vqe_stress_provenance_from_runtime_input(runtime_input: Any) -> dict[str, Any]:
    """Report whether a loaded seed is a fixed-VQE conditioning-stress artifact.

    Shared benchmark and dynamics routes read the stress identity from the
    serialized artifact they already loaded.  Nothing is injected at run time and
    no exact-reference content is surfaced here.
    """

    extensions = dict(getattr(runtime_input, "extensions", {}) or {})
    provenance = dict(getattr(runtime_input, "provenance", {}) or {})
    pool_meta = dict(extensions.get("legacy_pool_meta", {}) or {})
    loader_summary = dict(extensions.get("legacy_loader_summary", {}) or {})
    subject_kind = pool_meta.get("subject_kind") or loader_summary.get(
        "fixed_scaffold_kind"
    )
    route_family = pool_meta.get("route_family") or loader_summary.get("route_family")
    present = bool(
        str(subject_kind or "").startswith("fixed_vqe_conditioning")
        or str(subject_kind or "") in set(CONSTRUCTION_MODES)
    )
    return {
        "schema": FIXED_VQE_CONDITIONING_SCHEMA_V1,
        "source": "serialized_seed_artifact",
        "present": bool(present),
        "subject_kind": None if subject_kind is None else str(subject_kind),
        "route_family": None if route_family is None else str(route_family),
        "structure_locked": bool(pool_meta.get("structure_locked", False)),
        "artifact_json": provenance.get("artifact_json"),
        "online_injection_used": False,
    }


def _statevector_payload(
    vec: np.ndarray,
    *,
    source: str,
    handoff_state_kind: str,
    cutoff: float = 1.0e-14,
) -> dict[str, Any]:
    arr = np.asarray(vec, dtype=complex).reshape(-1)
    if arr.size <= 0 or arr.size & (arr.size - 1):
        raise ValueError("statevector length must be a positive power of two.")
    nq = int(round(math.log2(int(arr.size))))
    amplitudes: dict[str, dict[str, float]] = {}
    for index, amp in enumerate(arr):
        if abs(complex(amp)) <= float(cutoff):
            continue
        amplitudes[format(index, f"0{nq}b")] = {
            "re": float(np.real(amp)),
            "im": float(np.imag(amp)),
        }
    if not amplitudes:
        raise ValueError("statevector payload would be empty.")
    return {
        "source": str(source),
        "handoff_state_kind": str(handoff_state_kind),
        "nq_total": int(nq),
        "amplitudes_qn_to_q0": amplitudes,
    }


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------


def _try_import_scipy_minimize():
    try:
        from scipy.optimize import minimize  # type: ignore
    except Exception:  # pragma: no cover - environment dependent
        return None
    return minimize


def _normalize_state(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=complex).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"{name} must have positive finite norm.")
    return np.asarray(arr / norm, dtype=complex)


def _digest_json(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(_json_safe(payload), sort_keys=True).encode("utf-8")
    ).hexdigest()


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
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(v) for v in value]
    try:
        out = float(value)
    except (TypeError, ValueError):
        return str(value)
    return out if np.isfinite(out) else None


__all__ = [
    "CHILD_KINDS",
    "CHILD_KIND_PAULI",
    "CHILD_KIND_POLYTERM",
    "CONSTRUCTION_MODES",
    "CONSTRUCTION_MODE_CONVENTIONAL",
    "CONSTRUCTION_MODE_EXACT_NULL_TEST",
    "CONSTRUCTION_MODE_NEAR_NULL_TEST",
    "CONVENTIONAL_HH_LAYER_PARENTS",
    "DEFAULT_DELTA_E_MAX",
    "DEFAULT_GRAM_RETAINED_RCOND",
    "DEFAULT_NEAR_NULL_COMPANION_COUNT",
    "DEFAULT_NEAR_NULL_SEPARATION_ANGLE",
    "DEFAULT_SNAPSHOT_RAY_DISTANCE_MAX",
    "DEFAULT_SNAPSHOT_TIMES",
    "FIXED_VQE_CONDITIONING_ARTIFACT_SCHEMA_V1",
    "FIXED_VQE_CONDITIONING_SCHEMA_V1",
    "FIXED_VQE_STRESS_ROUTE_FAMILY",
    "FIXED_VQE_STRESS_SUBJECT_KIND_V1",
    "ArchitectureConditioningRecord",
    "ArchitectureRuntime",
    "ArchitectureSearchConfig",
    "ConditioningAggregate",
    "ExactSnapshot",
    "FixedArchitecture",
    "FixedVQEConditioningConfig",
    "FixedVQEConditioningContext",
    "FixedVQEConditioningSearchResult",
    "FixedVQEDriveConfig",
    "FixedVQEGeneratorAtom",
    "FixedVQEGeneratorPool",
    "FixedVQEModelConfig",
    "GeneratorOccurrence",
    "GeneratorPoolConfig",
    "GramSpectrumConfig",
    "GramSpectrumRecord",
    "GroundStateQualification",
    "GroundStateQualificationConfig",
    "SearchBatchStore",
    "SnapshotFit",
    "SnapshotFitConfig",
    "SnapshotScheduleConfig",
    "aggregate_conditioning",
    "architecture_ansatz_terms",
    "architecture_from_layers",
    "atom_ids_by_parent",
    "build_architecture_runtime",
    "build_conditioning_context",
    "build_fixed_vqe_generator_pool",
    "build_fixed_vqe_stress_artifact_payload",
    "default_seed_parent_labels",
    "enumerate_seed_architectures",
    "evaluate_architecture_stage_one",
    "evaluate_architecture_stage_two",
    "exact_null_test_architecture",
    "fit_architecture_to_snapshot",
    "fit_architecture_to_snapshots",
    "fixed_vqe_stress_provenance_from_runtime_input",
    "generate_exact_driven_snapshots",
    "gram_spectrum_at",
    "mutate_architecture",
    "near_null_test_architecture",
    "near_null_test_theta",
    "parent_complete_layer",
    "pauli_words_commute",
    "pareto_front",
    "qualify_architecture_ground_state",
    "ray_distance",
    "record_from_payload",
    "run_fixed_vqe_conditioning_search",
    "select_near_null_test_architecture",
    "write_fixed_vqe_stress_artifact",
]
